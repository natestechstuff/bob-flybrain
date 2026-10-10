
"""export_bob.py - pack trained Bob v2 (reduced circuit from the real connectome + trained KC->MBON weights + his
smell->action map + the action menu) into bob_fly/bob_fly.npz, plus bob_fly/run_bob.py which needs only numpy.
That folder is the portable fly (phone / Xbox later). It only PICKS actions; it never runs commands itself.

  ./venv/bin/python export_bob.py weights/bob2_sub.npz
"""
import numpy as np, sys, os, json
from demo import TrainedBob, MENU
HERE = os.path.dirname(os.path.abspath(__file__))
wfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'weights', 'bob2_sub.npz')
tb = TrainedBob('sub', wfile)
b = tb.bob; B = b.B
assert b.n < 32767
os.makedirs(f'{HERE}/bob_fly', exist_ok=True)
data_syn = (B.data / B.p['w_syn']).round().astype(np.int16)   # back to signed synapse counts (incl. the PN->KC gain)
np.savez_compressed(f'{HERE}/bob_fly/bob_fly.npz',
    indptr=B.indptr.astype(np.int32), indices=B.indices.astype(np.int16), data=data_syn,
    silent=b.silent0, kc=b.kc.astype(np.int16), mbon=b.mb.astype(np.int16), PW=B.PW.astype(np.float16),
    odor_pns=np.array(json.dumps([list(map(int, p)) for p in b.odor_pns])), combos=np.array(json.dumps([list(c) for c in tb.combos])),
    groups=np.array(json.dumps([list(map(int, g)) for g in tb.groups])), cfg=np.array(json.dumps(tb.cfg)),
    keys=np.array([m[0] for m in MENU[:tb.N]]), means=np.array([m[1] for m in MENU[:tb.N]]), commands=np.array([m[2] for m in MENU[:tb.N]]),
    held_out=tb.held_out)
src = open(f'{HERE}/flyengine.py').read()
runner = src + r"""

# ---- Bob v2, portable: numpy only. Picks an action for a smell; never runs anything. ----
LETTERS = 'ABCDEFGHIJKL'
if __name__ == '__main__':
    import sys, os, json, time, resource
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bob_fly.npz'))
    cfg = json.loads(str(z['cfg'])); odor_pns = [np.array(p) for p in json.loads(str(z['odor_pns']))]
    combos = [tuple(c) for c in json.loads(str(z['combos']))]; groups = [np.array(g) for g in json.loads(str(z['groups']))]
    keys = list(z['keys']); n = len(z['silent'])
    B = Brain(z['indptr'].astype(np.int64), z['indices'].astype(np.int32), z['data'].astype(np.float32), n,
              z['kc'].astype(np.int64), z['mbon'].astype(np.int64), z['PW'].astype(np.float32))
    rng = np.random.default_rng()
    names = ['+'.join(LETTERS[i] for i in c) for c in combos]
    if not sys.argv[1:] or sys.argv[1] == '--list':
        for nm, k, d in zip(names, keys, z['means']): print(f'{nm:<7} -> {k:<14} {d}')
        sys.exit()
    for arg in sys.argv[1:]:
        c = tuple(sorted(LETTERS.index(x) for x in arg.upper().split('+')))
        drive = np.concatenate([odor_pns[i] for i in c]); sil = z['silent'].copy(); sil[drive] = False
        rate = cfg['rate'] * (1 + cfg['rate_jitter'] * (2 * rng.random() - 1)) / len(c) ** cfg.get('mix_norm', 0.0)
        t = time.time(); cnt = B.run(drive, rate, cfg['t_ms'], rng, silent=sil)
        m = cnt[z['mbon'].astype(np.int64)]; score = np.array([m[g].mean() for g in groups])
        ch = int(np.argmax(score + rng.random(len(score)) * 1e-3))
        want = keys[combos.index(c)] if c in combos else '?'
        print(f"Bob smelled {arg.upper():<7} -> {keys[ch]:<14} ($ {z['commands'][ch]})   trained answer {want}   [{time.time()-t:.2f}s]")
    print(f"neurons {n}, knows {len(combos)} smells, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
"""
open(f'{HERE}/bob_fly/run_bob.py', 'w').write(runner)
print('exported', wfile, '->', f'{HERE}/bob_fly', os.path.getsize(f'{HERE}/bob_fly/bob_fly.npz') // 1024, 'KB')
