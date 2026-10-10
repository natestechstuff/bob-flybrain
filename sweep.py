
"""sweep.py - run 'lanes' of training jobs in parallel (one process per lane). Each lane trains Bob on a growing
number of smell->action mappings and stops after the first size whose held-out accuracy drops below --bar."""
import json, subprocess, sys, os, time, argparse
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument('lanes')            # json file: [{"name":..., "args": "--brain sub --seed 0", "Ns": [4, 8, ...]}]
ap.add_argument('--root', default='runs')
ap.add_argument('--bar', type=float, default=0.85)
ap.add_argument('--extra', default='')
ap.add_argument('--par', type=int, default=0)
a = ap.parse_args()
lanes = json.load(open(a.lanes))
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
T0 = time.time()
def lane(L):
    res = []
    for N in L['Ns']:
        out = f"{a.root}/{L['name']}_N{N}"
        cmd = f"python train_combo.py {L['args']} {a.extra} --N {N} --out {out} --quiet"
        t = time.time(); p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
        try:
            r = json.load(open(f'{out}/result.json'))
            row = dict(N=N, test=r['test'], by=r['test_by_order'], pre=r['curve']['pre_test'], ms_step=r['ms_per_step_median'],
                       rss=r['peak_rss_mb'], wall=r['wall_s'], neurons=r['n_neurons'], kcs=r['n_kc'], readout=r['readout'], K=r['K'])
        except Exception as e:
            row = dict(N=N, error=repr(e), stderr=p.stderr[-1500:])
        res.append(row)
        print(f"[{time.time()-T0:6.0f}s] {L['name']} N={N}: {row.get('test', row.get('error'))}", flush=True)
        json.dump(res, open(f"{a.root}/{L['name']}_lane.json", 'w'), indent=1)
        if 'test' not in row or row['test'] < a.bar:
            break
    return L['name'], res
with ThreadPoolExecutor(a.par or len(lanes)) as ex:
    allres = dict(ex.map(lane, lanes))
json.dump(allres, open(f'{a.root}/sweep_summary.json', 'w'), indent=1)
print('SWEEP DONE', round(time.time() - T0), 's')
for k, v in allres.items():
    print(k, [(r['N'], round(r['test'], 3) if 'test' in r else 'ERR') for r in v])
