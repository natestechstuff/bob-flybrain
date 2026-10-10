
"""bob_server.py - let Orbit apps (Android / Xbox) talk to Bob over the LAN. Python standard library only.

  ./venv/bin/python bob_server.py                      # 0.0.0.0:8095, reduced-circuit brain, keyword matcher
  ./venv/bin/python bob_server.py --port 8095 --key mysecret
  ./venv/bin/python bob_server.py --brain full         # whole brain (slower)
  ./venv/bin/python bob_server.py --llm-url http://PHONE-OR-XBOX:PORT --llm-api openai   # optional LLM to read requests

Endpoints (JSON):  GET /health   GET /actions   POST /smell {"text"|"action"|"odors"}   POST /mind {"lines":50} (or GET /mind?lines=50)
Safety: only MENU actions from demo.py can run, only inside the labelled no-network sandbox container; request text is
never executed; optional X-Bob-Key shared key; 2 requests/second limit; one smell at a time.
"""
import json, os, sys, time, threading, argparse, secrets
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import MENU, BY_KEY, TrainedBob, MIND_PATH, mind
from talk_to_bob import handle, LLM

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='0.0.0.0')
ap.add_argument('--port', type=int, default=8095)
ap.add_argument('--brain', default='sub', choices=['sub', 'full', 'plus'])
ap.add_argument('--weights', default=None)
ap.add_argument('--key', default=os.environ.get('BOB_KEY'))
ap.add_argument('--gen-key', action='store_true', help='make up a random key')
ap.add_argument('--rate', type=float, default=2.0, help='max requests per second')
ap.add_argument('--llm-url', default=None)
ap.add_argument('--llm-api', default='llama', choices=['llama', 'openai'])
ap.add_argument('--llm-model', default=None)
a = ap.parse_args()
if a.gen_key and not a.key: a.key = secrets.token_urlsafe(9)

TB = TrainedBob(a.brain, a.weights)
LLMC = LLM(a.llm_url, a.llm_api, a.llm_model, TB.actions) if a.llm_url else None
BRAIN_LOCK = threading.Lock()
BURST = max(1.0, 2 * a.rate)
_rl = {'t': time.time(), 'tokens': BURST}; _rl_lock = threading.Lock()

def allow():
    with _rl_lock:
        now = time.time(); _rl['tokens'] = min(BURST, _rl['tokens'] + (now - _rl['t']) * a.rate); _rl['t'] = now
        if _rl['tokens'] >= 1: _rl['tokens'] -= 1; return True
        return False

def mind_tail(n):
    n = max(1, min(500, int(n)))
    try:
        with open(MIND_PATH, 'rb') as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 200_000))
            lines = f.read().decode('utf-8', 'replace').splitlines()
        return lines[-n:]
    except FileNotFoundError:
        return []

class H(BaseHTTPRequestHandler):
    server_version = 'Bob/2'
    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.address_string(), fmt % args))
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Bob-Key')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_OPTIONS(self):
        self._send(204 if False else 200, {'ok': True})
    def _gate(self, path):
        if path == '/health': return True
        if a.key and self.headers.get('X-Bob-Key') != a.key:
            self._send(401, {'ok': False, 'error': 'bad or missing X-Bob-Key'}); return False
        if not allow():
            self._send(429, {'ok': False, 'error': 'slow down'}); return False
        return True
    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 10_000: raise ValueError('body too big')
        raw = self.rfile.read(n) if n else b'{}'
        d = json.loads(raw or b'{}')
        if not isinstance(d, dict): raise ValueError('body must be a JSON object')
        return d
    def do_GET(self):
        u = urlparse(self.path); p = u.path.rstrip('/') or '/'
        if not self._gate(p): return
        if p == '/health':
            return self._send(200, {'ok': True, 'name': 'Bob', 'brain': TB.cfg['brain'], 'neurons': int(TB.bob.n), 'actions': TB.N,
                                    'held_out_accuracy': round(TB.held_out, 3), 'key_required': bool(a.key)})
        if p == '/actions':
            return self._send(200, {'ok': True, 'actions': [{'id': k, 'description': BY_KEY[k][1], 'odors': TB.odor_for(k).split('+')} for k in TB.actions]})
        if p == '/mind':
            n = parse_qs(u.query).get('lines', ['50'])[0]
            return self._send(200, {'ok': True, 'lines': mind_tail(n)})
        self._send(404, {'ok': False, 'error': 'unknown endpoint; try /health /actions /smell /mind'})
    def do_POST(self):
        p = urlparse(self.path).path.rstrip('/')
        if not self._gate(p): return
        try:
            d = self._body()
        except Exception as e:
            return self._send(400, {'ok': False, 'error': f'bad JSON: {e}'})
        if p == '/smell':
            if sum(k in d for k in ('text', 'action', 'odors')) != 1:
                return self._send(400, {'ok': False, 'error': 'send exactly one of text, action, odors'})
            with BRAIN_LOCK:
                try:
                    res = handle(TB, d, LLMC)
                except SystemExit as e:
                    res = {'ok': False, 'error': str(e)}
            return self._send(200 if res.get('ok') else 400, res)
        if p == '/mind':
            return self._send(200, {'ok': True, 'lines': mind_tail(d.get('lines', 50))})
        self._send(404, {'ok': False, 'error': 'unknown endpoint'})

if __name__ == '__main__':
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"Bob server on http://{a.host}:{a.port}  ({TB.cfg['brain']} brain, {TB.bob.n:,} neurons, {TB.N} actions, "
          f"requests read by {'LLM at ' + a.llm_url if LLMC else 'keyword matcher'})", flush=True)
    print(f"shared key: {a.key}  (send it as header X-Bob-Key)" if a.key else 'no shared key set (anyone on your network can use Bob; add --key or --gen-key)', flush=True)
    mind(f"Bob's server started on port {a.port}.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
