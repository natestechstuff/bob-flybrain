
"""bobbrain.py - Bob's brain v2: odor bank + smell combos, all-96-MBON readout, teacher-dopamine learning.

Pipeline (same neuron model as before, flyengine.py):
  odor combo -> union of its base odors' projection neurons (PNs) -> real PN->KC wiring (x g_pn_kc)
  -> Kenyon cells (KCs) -> plastic KC->MBON block -> 96 MBONs -> action = MBON group with most spikes
Brains:
  full : the whole FlyWire brain (138,639 neurons)
  sub  : reduced circuit cut from the connectome (ALPNs, KCs, MBONs, DANs/MBINs, APL, DPM)
  plus : 'Bob+' = sub + SYNTHETIC extra Kenyon cells (factor x the real ~5k). PARTLY SYNTHETIC - see make_plus().
"""
import numpy as np, os, itertools, time
from flycircuit import load_cells, load_csr, strip_block, scale_block, subnetwork
from flyengine import Brain

LETTERS = 'ABCDEFGHIJKL'

def combo_name(c):
    return '+'.join(LETTERS[i] for i in c)

def parse_combo(s):
    s = s.upper().replace(' ', '').replace(',', '+')
    parts = s.split('+') if '+' in s else list(s)
    return tuple(sorted(LETTERS.index(p) for p in parts))

def combo_list(K, max_order=3):
    out = []
    for r in range(1, max_order + 1):
        out += list(itertools.combinations(range(K), r))
    return out

def k_for(N):
    """smallest number of base odors whose singles+pairs+triples give >= N combos (8 -> 92, 9 -> 129, 10 -> 175 ...)"""
    K = 1
    while len(combo_list(K)) < N: K += 1
    return max(K, 4)

def mapping_set(N, K):
    """first N combos in curriculum order: singles, then pairs, then triples (K base odors)"""
    cl = combo_list(K)
    assert N <= len(cl), f'only {len(cl)} combos with K={K}'
    return cl[:N]

def balanced_set(N, K=8, seed=0):
    """'balanced' mapping set: about a third singles, a third pairs, a third triples of K base odors
    (pairs and triples picked at random, so some are nested: A, A+B, A+B+C can all be in it)."""
    rng = np.random.default_rng(seed + 55)
    n1 = min(K, round(N / 3)); n2 = round((N - n1) / 2); n3 = N - n1 - n2
    pairs = list(itertools.combinations(range(K), 2)); trip = list(itertools.combinations(range(K), 3))
    assert n2 <= len(pairs) and n3 <= len(trip)
    p = [pairs[i] for i in sorted(rng.choice(len(pairs), n2, replace=False))]
    t = [trip[i] for i in sorted(rng.choice(len(trip), n3, replace=False))]
    return [(i,) for i in range(n1)] + p + t

def spread_set(N, seed=0, bank=12):
    """'spread' mapping set: singles, pairs and triples from a 12-odor bank chosen so that no combo is inside
    another (no A next to A+B, no A+B next to A+B+C). Mix: up to 4 singles, ~35% pairs, the rest triples.
    Returns None if N doesn't fit."""
    rng = np.random.default_rng(seed + 77)
    want = {1: min(4, max(1, round(N * 0.08)))}
    want[2] = round(N * 0.35); want[3] = N - want[1] - want[2]
    chosen = []
    def ok(c):
        s = set(c)
        return all(not (s <= set(d) or set(d) <= s) for d in chosen)
    for r in (1, 2, 3):
        pool = [tuple(sorted(c)) for c in itertools.combinations(range(bank), r)]
        rng.shuffle(pool)
        got = 0
        for c in pool:
            if got >= want[r]: break
            if ok(c): chosen.append(c); got += 1
    if len(chosen) < N:   # top up with any size that still fits
        for r in (3, 2):
            for c in itertools.combinations(range(bank), r):
                if len(chosen) >= N: break
                if ok(c): chosen.append(tuple(c))
    return sorted(chosen, key=lambda c: (len(c), c)) if len(chosen) >= N else None

def make_groups(n_mbon, N, kcin, mode='groups', code_size=8, seed=0):
    """readout: N disjoint MBON groups balanced by KC input (needs N <= n_mbon), or a population code
    (each action = a random set of code_size MBONs, sets overlap) for any N."""
    if mode == 'groups':
        assert N <= n_mbon, f'{N} actions need a population code: only {n_mbon} MBONs'
        order = np.argsort(-kcin)
        g = [[] for _ in range(N)]
        for r, m in enumerate(order):
            row = r // N; j = r % N
            g[j if row % 2 == 0 else N - 1 - j].append(int(m))
        return [np.array(x) for x in g]
    rng = np.random.default_rng(seed + 991)
    return [np.sort(rng.choice(n_mbon, code_size, replace=False)) for _ in range(N)]

def make_plus(ip, ix, dd, n, kc, pns, factor, seed, kc_w_scale=True):
    """Bob+: add (factor-1)*len(kc) SYNTHETIC Kenyon cells to a (sub) circuit given as CSR.
    Each synthetic KC copies a random real KC ('template'): it gets the template's number and strengths of
    PN inputs but from RANDOMLY chosen projection neurons (fly-like random PN->KC sampling),
    the template's other inputs (APL, DPM, DANs, other KCs) and the template's outputs. All KC outputs are
    scaled by 1/factor so total KC drive onto APL/MBONs stays at the real brain's level.
    Returns new CSR, new n, synthetic indices, template index per synthetic KC."""
    rng = np.random.default_rng(seed + 4242)
    n_new = int(round((factor - 1) * len(kc)))
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip)); cols = ix.astype(np.int64); vals = dd.astype(np.float32)
    syn = np.arange(n, n + n_new, dtype=np.int64)
    tmpl = rng.choice(kc, n_new, replace=True)
    is_pn = np.zeros(n, bool); is_pn[pns] = True
    # inputs to each real KC: gather column entries (transpose once)
    order = np.argsort(cols, kind='stable'); rc, cc, vc = rows[order], cols[order], vals[order]
    cptr = np.searchsorted(cc, np.arange(n + 1))
    nr, nc, nv = [], [], []
    for s, t in zip(syn, tmpl):
        a, b = cptr[t], cptr[t + 1]
        pre, w = rc[a:b], vc[a:b]
        p = is_pn[pre]
        if p.any():  # PN inputs: same strengths, random PN identities
            newpre = rng.choice(pns, int(p.sum()), replace=False)
            nr.append(newpre); nc.append(np.full(len(newpre), s)); nv.append(w[p])
        if (~p).any():
            nr.append(pre[~p]); nc.append(np.full(int((~p).sum()), s)); nv.append(w[~p])
        a, b = ip[t], ip[t + 1]  # outputs: template's row
        nr.append(np.full(b - a, s)); nc.append(ix[a:b].astype(np.int64)); nv.append(dd[a:b].astype(np.float32))
    R = np.concatenate([rows] + nr); Cc = np.concatenate([cols] + nc); V = np.concatenate([vals] + nv)
    N2 = n + n_new
    if kc_w_scale:
        allkc = np.zeros(N2, bool); allkc[kc] = True; allkc[syn] = True
        V = np.where(allkc[R], V / factor, V).astype(np.float32)
    o = np.argsort(R, kind='stable'); R, Cc, V = R[o], Cc[o], V[o]
    ip2 = np.zeros(N2 + 1, np.int64); np.add.at(ip2, R + 1, 1); ip2 = np.cumsum(ip2)
    return ip2, Cc.astype(np.int32), V, N2, syn, tmpl

class Bob:
    def __init__(self, brain='sub', factor=1, K=8, glom_per_odor=5, odor_seed=11, g_pn_kc=4.0, g_kc_mbon=2.0,
                 rate=100.0, rate_jitter=0.2, t_ms=100.0, plus_seed=0, mix_norm=0.0, log=print):
        t0 = time.time()
        self.cfg = dict(brain=brain, factor=factor, K=K, glom_per_odor=glom_per_odor, odor_seed=odor_seed, g_pn_kc=g_pn_kc,
                        g_kc_mbon=g_kc_mbon, rate=rate, rate_jitter=rate_jitter, t_ms=t_ms, plus_seed=plus_seed, mix_norm=mix_norm)
        cells = load_cells(); ip, ix, dd, ids = load_csr()
        cc = cells.cell_class.fillna('').values
        kc = cells.index[cells.cell_class == 'Kenyon_Cell'].values
        mb = cells.index[cells.cell_class == 'MBON'].values
        pn = cells[(cells.cell_class == 'ALPN') & cells.cell_sub_class.fillna('').str.contains('uniglomerular')]
        alpn = cells.index[cells.cell_class == 'ALPN'].values
        # base odors = disjoint random glomerulus sets (both hemispheres)
        rng = np.random.default_rng(odor_seed)
        gl = pn.groupby('cell_type').size(); gl = np.sort(gl[gl >= 2].index.values)
        bank = min(12, len(gl) // glom_per_odor)   # a fixed bank of 12 base odors; Bob uses the first K
        assert K <= bank
        pick = rng.choice(gl, glom_per_odor * bank, replace=False)
        self.odor_gloms = [sorted(pick[i * glom_per_odor:(i + 1) * glom_per_odor]) for i in range(K)]
        odor_pns = [pn.index[pn.cell_type.isin(t)].values for t in self.odor_gloms]
        # plastic block KC -> all 96 MBONs, from the connectome
        mpos = -np.ones(len(cells), np.int64); mpos[mb] = np.arange(len(mb))
        PW0 = np.zeros((len(kc), len(mb)), np.float32)
        for i, k in enumerate(kc):
            a, b = ip[k], ip[k + 1]; c = ix[a:b]; s = mpos[c] >= 0
            np.add.at(PW0[i], mpos[c[s]], dd[a:b][s])
        self.mbon_root = ids[mb]; self.mbon_type = cells.cell_type.fillna('?').values[mb]
        self.kc_root = ids[kc]
        silent = np.isin(cc, ['olfactory', 'ALLN', 'ALPN'])
        ip2, ix2, dd2 = strip_block(ip, ix, dd, kc, mb)
        dd2 = scale_block(ip2, ix2, dd2, alpn, kc, g_pn_kc)
        self.synthetic = 0
        if brain in ('sub', 'plus'):
            keep, ip2, ix2, dd2, newpos = subnetwork(cells, ip2, ix2, dd2)
            kc2 = newpos[kc]; mb2 = newpos[mb]; odor_pns = [newpos[p] for p in odor_pns]; silent = silent[keep]; n = len(keep)
            if brain == 'plus' and factor > 1:
                pn_sub = newpos[pn.index.values]
                ip2, ix2, dd2, n, syn, tmpl = make_plus(ip2, ix2, dd2, n, kc2, pn_sub, factor, plus_seed)
                tpos = -np.ones(n, np.int64); tpos[kc2] = np.arange(len(kc2))
                PW0 = np.vstack([PW0, PW0[tpos[tmpl]]]) / factor
                kc2 = np.concatenate([kc2, syn]); silent = np.concatenate([silent, np.zeros(len(syn), bool)])
                self.synthetic = len(syn)
        else:
            kc2, mb2, n = kc, mb, len(cells)
        self.n, self.kc, self.mb, self.odor_pns, self.silent0 = n, kc2, mb2, odor_pns, silent
        self.PW0 = PW0 * g_kc_mbon
        self.mask = self.PW0 > 0
        self.B = Brain(ip2, ix2, dd2, n, kc2, mb2, self.PW0.copy())
        self.n_syn = int(len(ix2) + self.mask.sum())
        self.kcin = self.PW0.sum(0)
        self.load_s = time.time() - t0
        self.step_times = []

    @property
    def PW(self):
        return self.B.PW

    def smell(self, combo, rng):
        drive = np.concatenate([self.odor_pns[i] for i in combo])
        sil = self.silent0.copy(); sil[drive] = False
        rate = self.cfg['rate'] * (1 + self.cfg['rate_jitter'] * (2 * rng.random() - 1))
        # antennal-lobe gain control (stands in for the lateral inhibition the clamped nose removed):
        # each PN fires at rate / (number of odors in the mix) ** mix_norm
        rate = rate / len(combo) ** self.cfg.get('mix_norm', 0.0)
        t = time.time()
        c = self.B.run(drive, rate, self.cfg['t_ms'], rng, silent=sil)
        el = time.time() - t; self.step_times.append(el / (self.cfg['t_ms'] / 0.1))
        return dict(mbon=c[self.mb].astype(np.float32), kc_act=c[self.kc] > 0, spikes=int(c.sum()), secs=el, rate=rate)

    def decide(self, mbon, groups, rng):
        score = np.array([mbon[g].mean() for g in groups])
        if score.max() <= 0:
            return int(rng.integers(len(groups))), score, 'guess'
        return int(np.argmax(score + rng.random(len(score)) * 1e-3)), score, 'brain'

    def learn_delta(self, act, mbon, groups, correct, lr=0.15, target=25.0, cap=4.0):
        """prediction-error dopamine rule (the default): every MBON compartment gets a dopamine signal equal to
        (what it SHOULD have fired - what it DID fire). The right action's MBONs should fire `target` spikes, all others 0.
        Active KC synapses change by lr x error x their start weight, kept between 0 and `cap` x the start weight.
        Too loud -> depression, too quiet -> potentiation, already right -> no change (so learning settles)."""
        PW, W0 = self.B.PW, self.PW0
        tgt = np.zeros(len(mbon), np.float32)
        tgt[groups[correct]] = target
        err = np.clip((tgt - mbon) / target, -1, 1).astype(np.float32)
        rows = np.flatnonzero(act)
        sub = PW[rows] + lr * W0[rows] * err[None, :]
        PW[rows] = np.clip(sub, 0, cap * W0[rows])
        return 'rewarded' if err[groups[correct]].mean() >= 0 else 'calmed'

    def learn_hinge(self, act, groups, correct, score, lr_pot=0.3, lr_dep=0.3, margin=0.2, cap=10.0):
        """perceptron-style dopamine rule: nothing changes when the right group already wins by `margin`.
        Otherwise: reward strengthens active KC synapses onto the right group, punishment weakens them onto every group
        that beat it or came within the margin."""
        PW, W0, M = self.B.PW, self.PW0, self.mask
        gc = groups[correct]; sc = score[correct]
        rivals = [j for j in range(len(groups)) if j != correct and score[j] >= (1 - margin) * sc - 1e-9]
        if not rivals:
            return 'already sure, no change'
        rows = np.flatnonzero(act)
        s = PW[np.ix_(rows, gc)]; w0 = W0[np.ix_(rows, gc)]
        PW[np.ix_(rows, gc)] = np.where(M[np.ix_(rows, gc)], s + lr_pot * (cap * w0 - s), s)
        for j in rivals:
            cols = np.setdiff1d(groups[j], gc)
            if len(cols): PW[np.ix_(rows, cols)] *= (1 - lr_dep)
        return 'rewarded right / punished %d rival%s' % (len(rivals), '' if len(rivals) == 1 else 's')

    def learn(self, act, groups, chosen, correct, score, lr_pot=0.3, lr_dep=0.2, margin=0.15, cap=4.0):
        """dopamine-style update at KC->MBON synapses of the KCs that fired this trial (teacher-guided):
        reward  -> strengthen active KC synapses onto the RIGHT group (bounded at 4x its start weight); this happens
                   on every training trial, so when Bob is wrong the trainer still shows him the right answer
        punish  -> weaken active KC synapses onto every group that out-voted the right one (x(1-lr_dep)),
                   and, more gently, onto groups that came within `margin` of it"""
        PW, W0, M = self.B.PW, self.PW0, self.mask
        gc = groups[correct]
        def pot(cols):
            if len(cols) == 0: return
            s = PW[np.ix_(act, cols)]; w0 = W0[np.ix_(act, cols)]; m = M[np.ix_(act, cols)]
            PW[np.ix_(act, cols)] = np.where(m, s + lr_pot * (cap * w0 - s), s)
        def dep(cols, f):
            cols = np.setdiff1d(cols, gc)
            if len(cols) == 0: return
            PW[np.ix_(act, cols)] *= (1 - f)
        pot(gc)
        # every group that out-voted (or nearly matched) the right one gets the punishment signal
        rivals = [j for j in range(len(groups)) if j != correct and score[j] >= (1 - margin) * score[correct]]
        for j in rivals:
            dep(groups[j], lr_dep if score[j] >= score[correct] else lr_dep * 0.5)
        return 'rewarded' if chosen == correct else 'punished'
