
"""talk_to_bob.py - the Orbit bridge: plain English in, Bob the fly acts, JSON out.

  echo "make a file called hi" | ./venv/bin/python talk_to_bob.py            # keyword matcher (default)
  ./venv/bin/python talk_to_bob.py --text "what time is it" --dry            # pick only, run nothing
  ./venv/bin/python talk_to_bob.py --llm-url http://127.0.0.1:8091           # use an LLM to understand the request:
        a llama.cpp `llama-server` (any GGUF, e.g. started with  llama-server -m YOUR_MODEL.gguf --port 8091)
        or any OpenAI-compatible /v1/chat/completions endpoint (add --llm-api openai --llm-model NAME)

Input: one request per line on stdin, plain text or JSON {"text": "..."}. Output: one JSON object per line.
Flow: request -> (LLM or keywords) -> the action that was asked for -> that action's trained smell (odor combo)
      -> Bob smells it -> Bob's MBONs vote -> Bob's OWN choice runs in the sandbox (even if he got it wrong).
The words never become a command: only MENU commands in demo.py can run, inside the no-network sandbox.
"""
import sys, os, json, time, re, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import MENU, BY_KEY, TrainedBob, run_in_sandbox, mind, describe

STOP = set('a an the to me my please can you could would bob hey yo it is are what whats for of and on in i do some how much many your you have has'.split())
FILES = {'hello', 'hi', 'bob', 'notes', 'test', 'todo', 'box', 'snacks', 'backup', 'counter', 'diary', 'crumb'}

def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())

def keyword_match(text, keys):
    """score every action Bob knows by keyword overlap; file names count extra; delete/read/write words steer"""
    w = [x for x in words(text) if x not in STOP]
    if not w: return None, 0.0
    best, bs = None, 0.0
    for k in keys:
        kw = BY_KEY[k][3].split(); desc = words(BY_KEY[k][1])
        s = 0.0
        for x in w:
            if x in kw: s += 2.0 if x in FILES else 1.0
            elif x in desc: s += 0.5
            elif len(x) > 3 and any(y.startswith(x[:4]) for y in kw): s += 0.4
        # a named file in the request that this action does NOT touch is a strong "no"
        named = [x for x in w if x in FILES]
        if named and not any(n in kw for n in named): s -= 1.5
        if s > bs: best, bs = k, s
    return (best, bs) if bs >= 1.0 else (None, bs)

class LLM:
    def __init__(self, url, api='llama', model=None, keys=()):
        self.url = url.rstrip('/'); self.api = api; self.model = model or 'local'; self.keys = list(keys)
        lines = '\n'.join(f'{k}: {BY_KEY[k][1]}' for k in self.keys)
        self.system = ('You turn a request into exactly one action id for a tiny robot fly. Reply with only the id.\n'
                       'If nothing fits, reply none.\nActions:\n' + lines)
        alts = ' | '.join(f'"{k}"' for k in self.keys + ['none'])
        self.grammar = f'root ::= {alts}'
    def pick(self, text):
        t = time.time()
        if self.api == 'llama':
            prompt = f'<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n'
            body = dict(prompt=prompt, n_predict=16, temperature=0, grammar=self.grammar, cache_prompt=True)
            r = self._post('/completion', body); out = r.get('content', '')
        else:
            body = dict(model=self.model, temperature=0, max_tokens=16,
                        messages=[{'role': 'system', 'content': self.system}, {'role': 'user', 'content': text}])
            r = self._post('/v1/chat/completions', body); out = r['choices'][0]['message']['content']
        out = out.strip().split()[0].strip('`."\'') if out.strip() else 'none'
        return (out if out in self.keys else None), round(time.time() - t, 2), out
    def _post(self, path, body):
        rq = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(rq, timeout=300).read())

def handle(tb, req, llm=None, dry=False):
    """one request -> Bob acts. req has exactly one of text / action / odors. Returns the API dict (see bob_server.py)."""
    from bobbrain import parse_combo, combo_name
    t0 = time.time()
    dry = bool(req.get('dry', dry))
    out = {'ok': True}
    asked, how, odor = None, None, None
    if req.get('action') is not None:
        asked = str(req['action']); how = 'action'
        if asked not in tb.actions:
            return {'ok': False, 'error': f'unknown action {asked!r}; GET /actions lists them'}
    elif req.get('odors') is not None:
        od = req['odors']; od = od if isinstance(od, list) else [od]
        try:
            c = parse_combo('+'.join(str(x) for x in od))
        except Exception:
            return {'ok': False, 'error': 'odors must be letters A-L, e.g. ["A","C"]'}
        if not (1 <= len(c) <= 3) or max(c) >= tb.K or len(set(c)) != len(c):
            return {'ok': False, 'error': f'give 1-3 different odors from A-{"ABCDEFGHIJKL"[tb.K-1]}'}
        odor = combo_name(c); how = 'odors'
        asked = tb.actions[tb.combos.index(c)] if c in tb.combos else None
    elif req.get('text') is not None:
        text = str(req['text'])[:500]; out['text'] = text
        if llm:
            try:
                asked, out['llm_s'], _ = llm.pick(text); how = 'llm'
            except Exception as e:
                out['llm_error'] = repr(e)[:200]
        if how is None:
            asked, _ = keyword_match(text, tb.actions); how = 'keywords'
    else:
        return {'ok': False, 'error': 'send one of {"text": ...}, {"action": ...} or {"odors": [...]}'}
    out.update(asked=asked, asked_description=BY_KEY[asked][1] if asked else None, understood_by=how)
    if odor is None:
        if asked is None:
            out.update(odors=[], bob_chose=None, correct=None, note="nothing on Bob's menu matches that, so no smell was sent",
                       ms=int((time.time() - t0) * 1000))
            mind(f'Orbit asked "{out.get("text", "")}" but nothing on Bob\'s menu matched; no smell sent.')
            return out
        odor = tb.odor_for(asked)
    res = tb.smell(odor)
    ch = res['choice']
    out.update(odors=odor.split('+'), bob_chose=ch, bob_description=BY_KEY[ch][1], confidence=res['confidence'],
               margin=res['margin'], votes={k: v for k, v in res['votes']}, kc_fired=res['kc_fired'], command=BY_KEY[ch][2],
               correct=(ch == asked) if asked else None)
    out['output'] = None if dry else run_in_sandbox(ch)
    out['ms'] = int((time.time() - t0) * 1000)
    lead = f'Orbit heard "{out["text"]}" -> sent smell {odor}. ' if 'text' in out else f'Got {how} request -> smell {odor}. '
    mind(lead + describe(res, None if dry else ch, out['output'], asked))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default=None)
    ap.add_argument('--brain', default='sub', choices=['sub', 'full', 'plus'])
    ap.add_argument('--weights', default=None)
    ap.add_argument('--llm-url', default=None)
    ap.add_argument('--llm-api', default='llama', choices=['llama', 'openai'])
    ap.add_argument('--llm-model', default=None)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    tb = TrainedBob(a.brain, a.weights)
    llm = LLM(a.llm_url, a.llm_api, a.llm_model, tb.actions) if a.llm_url else None
    sys.stderr.write(f"Bob is awake: {tb.cfg['brain']} brain, {tb.bob.n:,} neurons, knows {tb.N} smells. "
                     f"Understanding requests with: {'LLM at ' + a.llm_url if llm else 'keyword matcher'}\n")
    src = [a.text] if a.text else sys.stdin
    for line in src:
        line = line.strip()
        if not line: continue
        try: req = json.loads(line) if line.startswith('{') else {'text': line}
        except Exception: req = {'text': line}
        print(json.dumps(handle(tb, req, llm, a.dry), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
