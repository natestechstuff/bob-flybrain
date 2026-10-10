
"""bench2.py - speed and RAM of each brain size on this machine (one process per size).
  ./venv/bin/python bench2.py sub plus2 plus5 plus10 full"""
import sys, subprocess, json, os
code = r"""
import sys, time, json, resource, numpy as np
from bobbrain import Bob
kind = sys.argv[1]
brain, factor = ('plus', float(kind[4:])) if kind.startswith('plus') else (kind, 1)
t = time.time(); b = Bob(brain=brain, factor=factor, glom_per_odor=7, g_kc_mbon=6); load = time.time() - t
rng = np.random.default_rng(0); secs = []
for c in [(0,), (0, 1), (0, 1, 2), (3,)]:
    r = b.smell(c, rng); secs.append(r['secs'])
print(json.dumps(dict(brain=kind, neurons=int(b.n), kcs=int(len(b.kc)), synthetic=int(b.synthetic), load_s=round(load, 1),
     s_per_smell=round(float(np.median(secs)), 2), ms_per_step=round(float(np.median(secs)) * 1000 / 1000, 3),
     peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))))
"""
for kind in sys.argv[1:] or ['sub', 'plus2', 'plus5', 'plus10']:
    p = subprocess.run([sys.executable, '-c', code, kind], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    print(p.stdout.strip() or p.stderr[-500:], flush=True)
