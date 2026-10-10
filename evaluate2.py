
"""evaluate2.py - re-check trained Bob v2 weights on this machine: fresh held-out trials, no learning.
  ./venv/bin/python evaluate2.py weights/bob2_sub.npz --reps 3 [--brain full] [--naive]"""
import sys, time, argparse, resource, json
import numpy as np
from demo import TrainedBob, mind
from bobbrain import combo_name
ap = argparse.ArgumentParser(); ap.add_argument('weights'); ap.add_argument('--brain', default='sub'); ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--naive', action='store_true')
a = ap.parse_args()
t0 = time.time(); tb = TrainedBob(a.brain, a.weights, naive=a.naive); load = time.time() - t0
ok = {1: [], 2: [], 3: []}; secs = []; misses = []
for r in range(a.reps):
    for i, c in enumerate(tb.combos):
        res = tb.smell(combo_name(c)); secs.append(res['sim_s'])
        hit = res['choice'] == tb.actions[i]; ok[len(c)].append(hit)
        if not hit: misses.append(f"{res['odor']}->{res['choice']} (want {tb.actions[i]})")
allok = sum(ok.values(), [])
out = dict(weights=a.weights, brain=tb.cfg['brain'], naive=a.naive, N=tb.N, trials=len(allok), accuracy=round(float(np.mean(allok)), 3),
           by_order={k: (round(float(np.mean(v)), 3) if v else None) for k, v in ok.items()}, s_per_smell=round(float(np.median(secs)), 2),
           load_s=round(load, 1), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), misses=misses[:12])
print(json.dumps(out))
mind(f"Re-check on this machine ({'untrained ' if a.naive else ''}{tb.cfg['brain']} brain): {sum(allok)}/{len(allok)} right "
     f"({out['accuracy']*100:.0f}%), singles {out['by_order'][1]}, pairs {out['by_order'][2]}, triples {out['by_order'][3]}.")
