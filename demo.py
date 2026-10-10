
"""demo.py - Bob's action menu (THE hard-coded allowlist), the sandbox runner, Bob's mind log, and a smell demo.

  ./venv/bin/python demo.py A A+C B+D+F            # Bob smells each combo, picks an action, runs it in his container
  ./venv/bin/python demo.py --list                 # print the menu: smell -> action
  ./venv/bin/python demo.py A --dry                # pick only, run nothing
  ./venv/bin/python demo.py A --brain full         # use the whole 138,639-neuron brain (slower, ~4 s per smell)

SAFETY (enforced in run_in_sandbox below):
  * only commands written in MENU below can ever run - the command text never comes from user input,
    from the fly, from Orbit, or from container output;
  * they run only via `docker exec` in the container named flybrain-sandbox, and only if it carries the label
    made-by=bob-flybrain and has networking disabled (NetworkMode=none); nothing runs on the host;
  * every file the menu writes or removes is under /fly inside that container; rm/rmdir only hit fixed names Bob makes;
  * container output is only printed/logged (control characters stripped), never executed;
  * the host side writes exactly one file: bob_mind.log (rotated to bob_mind.log.1/.2 at 2 MB).
"""
import os, sys, re, time, json, subprocess, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONTAINER = 'flybrain-sandbox'
LABEL = 'bob-flybrain'

# key, what it does (plain English), exact command (runs in /fly inside the sandbox), keywords for the matcher
MENU = [
 ('make_hello',  'make a file called hello.txt',   'touch /fly/hello.txt && echo "created /fly/hello.txt"', 'make create new file hello'),
 ('list_files',  'list Bob\'s files',             'ls -la /fly', 'list show files ls what files directory folder see'),
 ('what_time',   'tell the date and time',        'date', 'time date clock what now'),
 ('buzz',        'write a buzz in his diary',     'echo "bzzz $(date +%T)" >> /fly/buzz.log && tail -1 /fly/buzz.log', 'buzz diary log write bzzz'),
 ('processes',   'list running processes',        'ps', 'processes ps running tasks programs'),
 ('whoami',      'say which user he is',          'whoami', 'who whoami user am i'),
 ('count_up',    'add 1 to his counter',          'n=$(cat /fly/counter.txt 2>/dev/null || echo 0); echo $((n+1)) > /fly/counter.txt; echo "counter is now $((n+1))"', 'count up add increment counter plus one'),
 ('uptime',      'say how long the box has been up', 'uptime', 'uptime how long up running awake'),
 ('make_hi',     'make a file called hi.txt',     'touch /fly/hi.txt && echo "created /fly/hi.txt"', 'make create file hi'),
 ('make_bob',    'make a file called bob.txt',    'touch /fly/bob.txt && echo "created /fly/bob.txt"', 'make create file bob'),
 ('make_notes',  'make a file called notes.txt',  'touch /fly/notes.txt && echo "created /fly/notes.txt"', 'make create file notes'),
 ('make_test',   'make a file called test.txt',   'touch /fly/test.txt && echo "created /fly/test.txt"', 'make create file test'),
 ('make_todo',   'make a file called todo.txt',   'touch /fly/todo.txt && echo "created /fly/todo.txt"', 'make create file todo'),
 ('make_box',    'make a folder called box',      'mkdir -p /fly/box && echo "made folder /fly/box"', 'make create folder directory mkdir box'),
 ('make_snacks', 'make a folder called snacks',   'mkdir -p /fly/snacks && echo "made folder /fly/snacks"', 'make create folder directory mkdir snacks'),
 ('read_hello',  'read hello.txt',                'cat /fly/hello.txt 2>/dev/null || echo "(no hello.txt)"; echo "[end of hello.txt]"', 'read cat open show hello'),
 ('read_buzz',   'read his buzz diary',           'cat /fly/buzz.log 2>/dev/null || echo "(diary empty)"', 'read cat open show buzz diary log'),
 ('read_counter','read the counter',              'cat /fly/counter.txt 2>/dev/null || echo 0', 'read show what counter number value'),
 ('count_buzz',  'count lines in the diary',      'wc -l /fly/buzz.log 2>/dev/null || echo "0 lines"', 'count lines wc diary buzz how many'),
 ('what_day',    'say what day it is',            'date +%A', 'day today weekday what'),
 ('what_date',   'say today\'s date',             'date +%F', 'date today which'),
 ('clock',       'say just the time',             'date +%T', 'clock time hours minutes'),
 ('uname',       'describe the system',           'uname -a', 'system uname kernel linux os info'),
 ('arch',        'say the CPU architecture',      'uname -m', 'architecture arch cpu type x86'),
 ('hostname',    'say the box\'s name',           'hostname', 'hostname name box computer called'),
 ('id',          'show user and group ids',       'id', 'id uid gid groups user'),
 ('disk',        'show disk space',               'df -h /', 'disk space storage df free full'),
 ('memory',      'show memory',                   'free -m', 'memory ram free'),
 ('fly_size',    'show how big his folder is',    'du -sh /fly', 'size big du folder how much space'),
 ('find_all',    'find everything in his folder', 'find /fly', 'find everything all search'),
 ('tree',        'list folders inside folders',   'ls -R /fly', 'tree recursive inside subfolders'),
 ('os_name',     'say what OS the box runs',      'head -2 /etc/os-release', 'os operating system alpine distro'),
 ('cpus',        'count CPUs',                    'nproc', 'cpus cores nproc processors'),
 ('mem_cap',     'show his memory limit',         'cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown', 'limit cap memory max cgroup'),
 ('greet',       'say hi',                        'echo "hi from Bob the fly 🪰"', 'hi hello hey greet say sup yo'),
 ('roll_dice',   'roll a dice',                   'echo "Bob rolled a $((RANDOM % 6 + 1))"', 'roll dice random number die'),
 ('count_ten',   'count to 10',                   'seq 1 10 | tr "\\n" " "; echo', 'count ten 10 numbers seq'),
 ('calendar',    'show a calendar',               'cal', 'calendar cal month'),
 ('rm_hello',    'delete hello.txt',              'rm -f /fly/hello.txt && echo "deleted /fly/hello.txt"', 'delete remove rm hello'),
 ('rm_hi',       'delete hi.txt',                 'rm -f /fly/hi.txt && echo "deleted /fly/hi.txt"', 'delete remove rm hi'),
 ('rm_bob',      'delete bob.txt',                'rm -f /fly/bob.txt && echo "deleted /fly/bob.txt"', 'delete remove rm bob'),
 ('rm_notes',    'delete notes.txt',              'rm -f /fly/notes.txt && echo "deleted /fly/notes.txt"', 'delete remove rm notes'),
 ('rm_test',     'delete test.txt',               'rm -f /fly/test.txt && echo "deleted /fly/test.txt"', 'delete remove rm test'),
 ('rm_todo',     'delete todo.txt',               'rm -f /fly/todo.txt && echo "deleted /fly/todo.txt"', 'delete remove rm todo'),
 ('rm_box',      'delete the box folder',         'rm -rf /fly/box && echo "deleted /fly/box"', 'delete remove rm rmdir box folder'),
 ('rm_snacks',   'delete the snacks folder',      'rm -rf /fly/snacks && echo "deleted /fly/snacks"', 'delete remove rm rmdir snacks folder'),
 ('reset_counter','reset the counter to 0',       'echo 0 > /fly/counter.txt && echo "counter reset to 0"', 'reset zero clear counter restart'),
 ('count_down',  'take 1 off his counter',        'n=$(cat /fly/counter.txt 2>/dev/null || echo 0); echo $((n-1)) > /fly/counter.txt; echo "counter is now $((n-1))"', 'count down subtract minus decrement counter'),
 ('backup_hello','back up hello.txt',             'cp /fly/hello.txt /fly/hello.bak 2>/dev/null && echo "copied to /fly/hello.bak" || echo "no hello.txt to copy"', 'backup copy cp duplicate hello'),
 ('rm_backup',   'delete the backup',             'rm -f /fly/hello.bak && echo "deleted /fly/hello.bak"', 'delete remove backup bak'),
 ('write_hi',    'write "hi" into hi.txt',        'echo "hi" > /fly/hi.txt && echo "wrote hi into /fly/hi.txt"', 'write put text into hi'),
 ('write_note',  'write a note',                  'echo "note at $(date +%T)" >> /fly/notes.txt && tail -1 /fly/notes.txt', 'write add note jot remember'),
 ('read_notes',  'read his notes',                'cat /fly/notes.txt 2>/dev/null || echo "(no notes)"', 'read cat show notes'),
 ('write_todo',  'add "feed Bob" to the todo list', 'echo "- feed Bob" >> /fly/todo.txt && cat /fly/todo.txt', 'add todo task list feed'),
 ('read_todo',   'read the todo list',            'cat /fly/todo.txt 2>/dev/null || echo "(no todo list)"', 'read show todo list tasks'),
 ('clear_buzz',  'clear his diary',               ': > /fly/buzz.log && echo "diary cleared"', 'clear empty wipe diary buzz'),
 ('head_buzz',   'show the first 3 diary lines',  'head -3 /fly/buzz.log 2>/dev/null', 'first top head oldest diary'),
 ('tail_buzz',   'show the last 3 diary lines',   'tail -3 /fly/buzz.log 2>/dev/null', 'last latest tail newest recent diary'),
 ('sort_buzz',   'sort the diary lines',          'sort /fly/buzz.log 2>/dev/null', 'sort order diary'),
 ('ls_box',      'look inside the box folder',    'ls -la /fly/box 2>/dev/null || echo "(no box folder)"', 'inside box look in folder'),
 ('crumb',       'put a crumb in the box folder', 'mkdir -p /fly/box && touch /fly/box/crumb.txt && echo "crumb in /fly/box"', 'crumb put box drop'),
 ('count_files', 'count his files',               'ls /fly | wc -l', 'count how many files number'),
 ('biggest',     'list files biggest first',      'ls -lS /fly', 'biggest largest size sort files'),
 ('newest',      'show the newest files',         'ls -lt /fly | head -4', 'newest latest recent modified files'),
 ('checksum',    'fingerprint his text files',    'md5sum /fly/*.txt 2>/dev/null || echo "(no .txt files)"', 'checksum md5 hash fingerprint'),
 ('sha',         'sha256 his diary',              'sha256sum /fly/buzz.log 2>/dev/null || echo "(no diary)"', 'sha sha256 hash diary'),
 ('where_am_i',  'say which folder he\'s in',     'cd /fly && pwd', 'where pwd folder am i location'),
 ('unix_time',   'say the unix time',             'date +%s', 'unix epoch seconds timestamp'),
 ('loadavg',     'show how busy the box is',      'cat /proc/loadavg', 'load busy loadavg cpu usage'),
 ('cpu_name',    'say what CPU it is',            'grep -m1 "model name" /proc/cpuinfo', 'cpu model processor name chip'),
 ('count_procs', 'count running processes',       'ps | tail -n +2 | wc -l', 'count how many processes'),
 ('nap',         'take a 1-second nap',           'sleep 1 && echo "Bob napped for 1 second"', 'nap sleep rest wait'),
 ('snore',       'snore',                         'echo "zzz... zzz..."', 'snore zzz sleepy'),
 ('haiku',       'say a fly haiku',               'printf "tiny wings hum on\\nsugar water, bright window\\nI smell, therefore I am\\n"', 'haiku poem poetry'),
 ('ascii_fly',   'draw himself',                  'printf "  \\\\  /\\n -(oo)-  bob\\n  /  \\\\\\n"', 'draw picture art ascii yourself'),
 ('month',       'say the month',                 'date +%B', 'month which'),
 ('year',        'say the year',                  'date +%Y', 'year which'),
 ('etc_hostname','read the hostname file',        'cat /etc/hostname', 'etc hostname file'),
 ('ls_root',     'list the top folder of the box','ls /', 'root top slash folders'),
 ('count_bins',  'count programs in /bin',        'ls /bin | wc -l', 'programs commands bin count'),
 ('which_sh',    'say where the shell is',        'which sh', 'which shell sh where'),
 ('busybox',     'say the toolbox version',       'busybox | head -1', 'busybox version toolbox'),
 ('alpine_ver',  'say the Alpine version',        'cat /etc/alpine-release', 'alpine version release'),
 ('network',     'show his network (none!)',      'ip addr 2>/dev/null | grep -E "^[0-9]+:" || echo "no network"', 'network ip internet wifi online'),
 ('meminfo',     'show memory details',           'head -3 /proc/meminfo', 'meminfo memory details total'),
 ('proc_uptime', 'show raw uptime seconds',       'cat /proc/uptime', 'raw uptime seconds'),
 ('env_home',    'say his home folder',           'echo "$HOME"', 'home folder env'),
 ('shout',       'shout',                         'echo "BZZZZZZZZZ!!!"', 'shout yell loud scream'),
 ('whisper',     'whisper',                       'echo "bzz..."', 'whisper quiet soft'),
 ('fact',        'say a fly fact',                'echo "fact: a fruit fly brain has about 140,000 neurons"', 'fact fun trivia'),
 ('joke',        'tell a joke',                   'echo "why did the fly fly? because the spider spied her"', 'joke funny laugh'),
 ('word_count',  'count words in his notes',      'wc -w /fly/notes.txt 2>/dev/null || echo "0 words"', 'count words notes'),
 ('touch_all',   'refresh all his files',         'touch /fly/*.txt 2>/dev/null; ls -l /fly', 'refresh update touch all'),
 ('count_dirs',  'count his folders',             'find /fly -type d | wc -l', 'count folders directories'),
 ('ls_snacks',   'look inside the snacks folder', 'ls -la /fly/snacks 2>/dev/null || echo "(no snacks folder)"', 'inside snacks look'),
 ('say_name',    'say his name',                  'echo "I am Bob"', 'name your who are you'),
]
# Bob's trained smells map, in order, onto the first N entries: singles first, then pairs, then triples.
# This puts the most useful actions first (it only changes which command each trained smell means, not how well he learns).
FIRST = ['make_hello', 'list_files', 'what_time', 'buzz', 'processes', 'whoami', 'count_up',      # 7 single smells
         'memory', 'read_hello', 'rm_hello', 'uname', 'greet', 'joke',                              # 6 pairs
         'make_hi', 'disk', 'roll_dice', 'read_counter', 'make_box', 'uptime', 'rm_hi']             # 7 triples
# (the final 20-smell brain never gets A+G or D+F right in re-checks, so those two carry the least-used actions:
#  'memory' and 'uname')
_by = {m[0]: m for m in MENU}
MENU = [_by[k] for k in FIRST] + [m for m in MENU if m[0] not in FIRST]
assert len({m[0] for m in MENU}) == len(MENU) == 96
KEYS = [m[0] for m in MENU]
BY_KEY = {m[0]: m for m in MENU}

# ---------------- sandbox runner ----------------
def _docker():
    try:
        if subprocess.run(['docker', 'ps'], capture_output=True, timeout=20).returncode == 0:
            return ['docker']
    except Exception:
        pass
    return ['sudo', '-n', 'docker']
DOCKER = None

def check_sandbox():
    global DOCKER
    DOCKER = DOCKER or _docker()
    p = subprocess.run(DOCKER + ['inspect', '-f', '{{index .Config.Labels "made-by"}} {{.State.Running}} {{.HostConfig.NetworkMode}}', CONTAINER],
                       capture_output=True, text=True, timeout=30)
    f = p.stdout.split()
    if len(f) != 3 or f[0] != LABEL:
        raise SystemExit(f'refusing: {CONTAINER} is not Bob\'s labelled sandbox')
    if f[2] != 'none':
        raise SystemExit(f'refusing: {CONTAINER} has networking ({f[2]}); Bob\'s sandbox must have none')
    if f[1] != 'true':
        subprocess.run(DOCKER + ['start', CONTAINER], check=True, capture_output=True, timeout=60)

_CTRL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')
def run_in_sandbox(key):
    """run the MENU command for `key` inside the sandbox; returns its (sanitised) output text. Never executes output."""
    if key not in BY_KEY:
        raise ValueError('not on the menu: %r' % key)
    check_sandbox()
    cmd = BY_KEY[key][2]
    p = subprocess.run(DOCKER + ['exec', CONTAINER, 'sh', '-c', cmd], capture_output=True, text=True, timeout=30)
    out = (p.stdout + p.stderr)[:4000]
    return _CTRL.sub('', out).rstrip()

# ---------------- Bob's mind log (host side, the only file the host scripts write) ----------------
MIND_PATH = os.path.join(HERE, 'bob_mind.log')
def mind(line):
    try:
        if os.path.exists(MIND_PATH) and os.path.getsize(MIND_PATH) > 2_000_000:
            for i in (2, 1):
                src = MIND_PATH + ('' if i == 1 else '.1')
                if os.path.exists(src): os.replace(src, MIND_PATH + '.%d' % i)
        with open(MIND_PATH, 'a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + line.replace('\n', ' | ') + '\n')
    except OSError:
        pass

def confidence_words(score):
    s = np.sort(score)[::-1]
    if s[0] <= 0: return 'just guessing (nothing fired)', 0.0
    m = (s[0] - s[1]) / s[0]
    return ('very sure' if m >= 0.5 else 'pretty sure' if m >= 0.25 else 'leaning' if m >= 0.1 else 'basically a coin flip'), float(m)

# ---------------- trained Bob ----------------
def find_weights(brain):
    for f in [os.path.join(HERE, 'weights', f'bob2_{brain}.npz'), os.path.join(HERE, 'weights', 'bob2_sub.npz')]:
        if os.path.exists(f): return f
    raise SystemExit('no trained Bob v2 weights in weights/ (bob2_sub.npz / bob2_full.npz)')

class TrainedBob:
    def __init__(self, brain='sub', weights=None, naive=False):
        from bobbrain import Bob
        self.wfile = weights or find_weights(brain)
        W = np.load(self.wfile, allow_pickle=False)
        cfg = json.loads(str(W['cfg'])); cfg['brain'] = brain if brain != 'plus' else cfg['brain']
        if cfg['brain'] != 'plus' and brain in ('sub', 'full'): cfg['brain'] = brain
        self.bob = Bob(**{k: cfg[k] for k in ('brain', 'factor', 'K', 'glom_per_odor', 'odor_seed', 'g_pn_kc', 'g_kc_mbon', 'rate', 'rate_jitter', 't_ms', 'plus_seed')}, log=lambda *x: None)
        assert np.array_equal(self.bob.mbon_root, W['mbon_root']) and np.array_equal(self.bob.kc_root, W['kc_root']), 'weights do not match this connectome'
        if not naive:
            assert W['PW'].shape == self.bob.PW.shape
            self.bob.B.PW[:] = W['PW']
        self.combos = [tuple(c) for c in json.loads(str(W['combos']))]
        self.groups = [np.array(g) for g in json.loads(str(W['groups']))]
        self.N = len(self.combos); self.K = int(W['K']); self.cfg = cfg
        self.actions = KEYS[:self.N]            # mapping i: combos[i] -> MENU[i]
        self.held_out = float(W['test'])
        self.rng = np.random.default_rng()

    def odor_for(self, key):
        from bobbrain import combo_name
        i = self.actions.index(key)
        return combo_name(self.combos[i])

    def smell(self, combo_str):
        from bobbrain import parse_combo, combo_name
        c = parse_combo(combo_str)
        r = self.bob.smell(c, self.rng)
        ch, score, how = self.bob.decide(r['mbon'], self.groups, self.rng)
        conf, margin = confidence_words(score)
        order = np.argsort(-score)
        trained = self.combos.index(c) if c in self.combos else None
        return dict(odor=combo_name(c), kc_fired=int(r['kc_act'].sum()), spikes=r['spikes'], sim_s=round(r['secs'], 2),
                    choice=self.actions[ch], choice_idx=ch, how=how, confidence=conf, margin=round(margin, 3),
                    votes=[(self.actions[j], round(float(score[j]), 1)) for j in order[:3]],
                    trained_answer=self.actions[trained] if trained is not None else None)

def describe(res, ran=None, out=None, asked=None):
    v = res['votes']
    s = (f"Bob smelled {res['odor']}. {res['kc_fired']} memory cells (Kenyon cells) lit up. "
         f"'{BY_KEY[v[0][0]][1]}' won ({v[0][1]}) over '{BY_KEY[v[1][0]][1]}' ({v[1][1]}), {res['confidence']}. ")
    if asked: s += f"Asked for '{BY_KEY[asked][1]}' -> {'he got it right' if asked == res['choice'] else 'he got it WRONG'}. "
    if ran is not None: s += f"Ran: {BY_KEY[ran][2]} -> {(out or '(no output)')[:300]}"
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('odors', nargs='*')
    ap.add_argument('--brain', default='sub', choices=['sub', 'full', 'plus'])
    ap.add_argument('--weights', default=None)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--naive', action='store_true', help='Bob before training (connectome weights)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    tb = TrainedBob(a.brain, a.weights, a.naive)
    if a.list:
        print(f'Bob knows {tb.N} smells (held-out accuracy {tb.held_out*100:.1f}%):')
        for i, k in enumerate(tb.actions):
            print(f'  {tb.odor_for(k):<7} -> {k:<14} {BY_KEY[k][1]:<36} $ {BY_KEY[k][2]}')
        return
    print(('NAIVE ' if a.naive else '') + f"Bob is awake ({tb.cfg['brain']} brain, {tb.bob.n:,} neurons, knows {tb.N} smells, {time.time()-t0:.1f}s)")
    for od in a.odors:
        res = tb.smell(od)
        print(f"\nBob smelled {res['odor']}  ({res['kc_fired']} Kenyon cells fired, {res['spikes']:,} spikes, {res['sim_s']}s to simulate)")
        print('  top MBON votes: ' + '  '.join(f'{k}={s}' for k, s in res['votes']) + f"   ({res['confidence']})")
        tr = res['trained_answer']
        print(f"  Bob chose: {res['choice']}  ->  $ {BY_KEY[res['choice']][2]}" + ('' if tr in (None, res['choice']) else f'   (trained answer was {tr})'))
        out = None
        if not a.dry:
            out = run_in_sandbox(res['choice'])
            print('  container says:\n' + '\n'.join('    | ' + l for l in out.splitlines()[:12]))
        mind(describe(res, None if a.dry else res['choice'], out))

if __name__ == '__main__':
    main()
