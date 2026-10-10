
"""train_combo.py - teach Bob N smell -> action mappings (single smells, pairs, triples) and measure held-out accuracy.

  python train_combo.py --brain sub --N 32 --seed 0 --out runs/sub_N32_s0
Writes result.json (learning curve + held-out test), bob_weights.npz, and mind_training.log (plain English).
Held-out test = fresh trials Bob never trained on (new random spike noise + new odor strength), no learning.
"""
import numpy as np, json, os, sys, time, argparse, resource
from bobbrain import Bob, mapping_set, make_groups, combo_name, k_for, spread_set, balanced_set

ap = argparse.ArgumentParser()
ap.add_argument('--brain', default='sub', choices=['full', 'sub', 'plus'])
ap.add_argument('--factor', type=float, default=1)
ap.add_argument('--N', type=int, default=16)
ap.add_argument('--K', type=int, default=0, help='base odors (0 = smallest K that has N combos, min 8)')
ap.add_argument('--glom', type=int, default=0, help='glomeruli per base odor (0 = 7, or fewer when more base odors are needed)')
ap.add_argument('--readout', default='auto', choices=['auto', 'groups', 'code'])
ap.add_argument('--code_size', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--g_kc_mbon', type=float, default=6.0)
ap.add_argument('--t_ms', type=float, default=100.0)
ap.add_argument('--mix_norm', type=float, default=0.0)
ap.add_argument('--blocks', type=int, default=12)
ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--test_reps', type=int, default=3)
ap.add_argument('--pre_reps', type=int, default=1)
ap.add_argument('--lr_pot', type=float, default=0.3)
ap.add_argument('--lr_dep', type=float, default=0.2)
ap.add_argument('--margin', type=float, default=0.15)
ap.add_argument('--rule', default='delta', choices=['delta', 'perceptron', 'hinge'])
ap.add_argument('--lr', type=float, default=0.2)
ap.add_argument('--target', type=float, default=15.0)
ap.add_argument('--cap', type=float, default=10.0)
ap.add_argument('--anneal', type=float, default=0.0, help='learning rate falls linearly to (1-anneal) x lr by the last round')
ap.add_argument('--stop_at', type=float, default=0.99)
ap.add_argument('--out', default='runs/test')
ap.add_argument('--quiet', action='store_true')
ap.add_argument('--set', default='nested', choices=['nested', 'balanced', 'spread'], help='nested = singles, then pairs, then triples of K odors; balanced = ~1/3 each; spread = no combo inside another')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.K == 0: a.K = max(8, k_for(a.N)) if a.set != 'spread' else 12
if a.glom == 0: a.glom = min(7, 69 // a.K)
T0 = time.time()
MIND = open(f'{a.out}/mind_training.log', 'w')
def mind(s):
    MIND.write(time.strftime('%H:%M:%S ') + s + '\n'); MIND.flush()
def log(*x):
    s = ' '.join(str(i) for i in x)
    if not a.quiet: print(s, flush=True)
    mind('[tech] ' + s)

bob = Bob(brain=a.brain, factor=a.factor, K=a.K, glom_per_odor=a.glom, odor_seed=11 + a.seed, plus_seed=a.seed, g_kc_mbon=a.g_kc_mbon, t_ms=a.t_ms, mix_norm=a.mix_norm)
combos = mapping_set(a.N, a.K) if a.set == 'nested' else balanced_set(a.N, a.K, a.seed) if a.set == 'balanced' else spread_set(a.N, a.seed)
if combos is None: raise SystemExit(f'no spread set of {a.N} combos fits in 12 odors')
ro = a.readout if a.readout != 'auto' else ('groups' if a.N <= len(bob.mb) else 'code')
groups = make_groups(len(bob.mb), a.N, bob.kcin, ro, a.code_size, a.seed)
name = {'full': 'full brain', 'sub': 'reduced circuit', 'plus': f'Bob+ x{a.factor:g} (PARTLY SYNTHETIC: {bob.synthetic} made-up Kenyon cells)'}[a.brain]
log(f'Bob is awake: {name}, {bob.n} neurons, {bob.n_syn} connections, {len(bob.kc)} Kenyon cells, load {bob.load_s:.1f}s')
log(f'{a.N} smell->action mappings ({sum(len(c)==1 for c in combos)} single, {sum(len(c)==2 for c in combos)} pairs, {sum(len(c)==3 for c in combos)} triples), K={a.K} base odors, readout={ro} ({[len(g) for g in groups][:3]}... MBONs per action)')
for i in range(a.K):
    log(f'base odor {"ABCDEFGHIJKL"[i]} = glomeruli {bob.odor_gloms[i]}')
mind(f'Bob is starting school: he has to learn {a.N} smells, each meaning a different action.')
rng = np.random.default_rng(1000 + a.seed)
kc_frac = {}

LR = [a.lr]
def block(tag, learn, reps):
    order = np.repeat(np.arange(a.N), reps); rng.shuffle(order)
    ok_all = []; by_order = {1: [], 2: [], 3: []}
    for j in order:
        r = bob.smell(combos[j], rng)
        ch, score, how = bob.decide(r['mbon'], groups, rng)
        ok = ch == j
        kc_frac.setdefault(len(combos[j]), []).append(r['kc_act'].mean())
        verdict = ''
        if learn:
            if a.rule == 'delta':
                verdict = bob.learn_delta(r['kc_act'], r['mbon'], groups, j, LR[0], a.target, a.cap)
                verdict = ('right -> ' if ok else 'wrong -> ') + verdict
            elif a.rule == 'hinge':
                verdict = bob.learn_hinge(r['kc_act'], groups, j, score, a.lr_pot, a.lr_dep, a.margin, a.cap)
            else:
                verdict = bob.learn(r['kc_act'], groups, ch, j, score, a.lr_pot, a.lr_dep, a.margin, a.cap)
        ok_all.append(ok); by_order[len(combos[j])].append(ok)
        if learn and len(ok_all) % max(1, a.N // 4) == 0:
            s = np.sort(score)[::-1]
            mind(f'{tag}: smelled {combo_name(combos[j])} ({int(r["kc_act"].sum())} memory cells lit), '
                 f'picked action #{ch} {"RIGHT" if ok else "wrong (should be #%d)" % j}, top vote {s[0]:.1f} vs {s[1]:.1f} -> {verdict}')
    acc = float(np.mean(ok_all)); bo = {k: (float(np.mean(v)) if v else None) for k, v in by_order.items()}
    log(f'{tag}: accuracy {acc:.3f} ({sum(ok_all)}/{len(ok_all)})  singles {bo[1]} pairs {bo[2]} triples {bo[3]}  [{time.time()-T0:.0f}s]')
    return acc, bo

curve = {}
curve['pre_test'], _ = block('before training (no learning)', False, a.pre_reps)
mind(f'Before any training Bob got {curve["pre_test"]*100:.0f}% right (pure chance is {100/a.N:.1f}%).')
streak = 0
for b in range(a.blocks):
    LR[0] = a.lr * (1 - a.anneal * b / max(1, a.blocks - 1))
    acc, _ = block(f'training round {b+1}', True, a.reps)
    curve[f'train_{b+1}'] = acc
    mind(f'Training round {b+1} done: Bob got {acc*100:.0f}% right while learning.')
    streak = streak + 1 if acc >= a.stop_at else 0
    if streak >= 2:
        log('stopping early: two rounds in a row at/above', a.stop_at); break
test, test_by = block('HELD-OUT TEST (fresh trials, no learning)', False, a.test_reps)
curve['post_test'] = test
mind(f'Final exam on {a.N * a.test_reps} fresh trials he never trained on: {test*100:.1f}% right '
     f'(singles {test_by[1]}, pairs {test_by[2]}, triples {test_by[3]}).')
st = np.array(bob.step_times)
out = dict(brain=a.brain, factor=a.factor, synthetic_kcs=bob.synthetic, N=a.N, K=a.K, seed=a.seed, readout=ro, n_neurons=int(bob.n),
           n_kc=int(len(bob.kc)), curve=curve, test=test, test_by_order=test_by,
           kc_active_frac={k: float(np.mean(v)) for k, v in kc_frac.items()},
           ms_per_step_median=float(np.median(st) * 1000), trials=len(st), wall_s=time.time() - T0,
           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, params=vars(a))
json.dump(out, open(f'{a.out}/result.json', 'w'), indent=1)
np.savez_compressed(f'{a.out}/bob_weights.npz', PW=bob.PW.astype(np.float32), cfg=json.dumps(bob.cfg), N=a.N, K=a.K,
                    combos=json.dumps([list(map(int, c)) for c in combos]), groups=json.dumps([list(map(int, g)) for g in groups]),
                    readout=ro, mbon_root=bob.mbon_root, kc_root=bob.kc_root, odor_gloms=json.dumps([list(map(str, g)) for g in bob.odor_gloms]),
                    test=test)
log('RESULT', json.dumps(dict(N=a.N, brain=a.brain, factor=a.factor, test=test, by=test_by, curve=curve)))
log(f'median {out["ms_per_step_median"]:.3f} ms per 0.1 ms step, peak RSS {out["peak_rss_mb"]:.0f} MB, wall {out["wall_s"]:.0f}s')
