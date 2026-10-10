
"""flycircuit.py - pick odors (projection-neuron patterns), readout MBONs, and the plastic KC->MBON block
from the real FlyWire connectome; build either the full brain or the reduced olfactory->MB->MBON subnetwork."""
import numpy as np, pandas as pd, os
from flyengine import Brain
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

ACTIONS = [
    ('touch',  'touch /fly/hello.txt && echo "created /fly/hello.txt"'),
    ('echo',   'echo "bzzz $(date +%T)" >> /fly/buzz.log && tail -1 /fly/buzz.log'),
    ('ls',     'ls -la /fly'),
    ('ps',     'ps'),
]
ODORS = ['A', 'B', 'C', 'D']

def load_cells():
    return pd.read_csv(f'{D}/cells.csv', dtype={'cell_type': str, 'cell_class': str, 'super_class': str})

def load_csr():
    z = np.load(f'{D}/brain_csr.npz')
    return z['indptr'], z['indices'], z['data'], z['ids']

def circuit(cells, indptr, indices, data, n_actions=4, glom_per_odor=6, seed=7, n_candidates=10, readout_types=None):
    rng = np.random.default_rng(seed)
    kc = cells.index[cells.cell_class == 'Kenyon_Cell'].values
    mbon = cells[cells.cell_class == 'MBON']
    pn = cells[(cells.cell_class == 'ALPN') & cells.cell_sub_class.fillna('').str.contains('uniglomerular')]
    if len(pn) < 100:
        pn = cells[cells.cell_class == 'ALPN']
    mb_idx = mbon.index.values
    mpos = -np.ones(len(cells), np.int64); mpos[mb_idx] = np.arange(len(mb_idx))
    PW0 = np.zeros((len(kc), len(mb_idx)), np.float32)
    for i, k in enumerate(kc):
        a, b = indptr[k], indptr[k + 1]
        cols = indices[a:b]; w = data[a:b]
        sel = mpos[cols] >= 0
        PW0[i, mpos[cols[sel]]] = w[sel]
    mbon = mbon.assign(kcin=PW0.sum(0))
    by_type = mbon.groupby('cell_type').agg(kcin=('kcin', 'mean'), n=('kcin', 'size'))
    by_type = by_type[by_type.n >= 2].sort_values('kcin', ascending=False)
    cand_types = list(readout_types) if readout_types else list(by_type.index[:n_candidates])
    cand = np.concatenate([mbon.index[mbon.cell_type == t].values for t in cand_types])
    ctype = np.concatenate([[t] * int((mbon.cell_type == t).sum()) for t in cand_types])
    rpos = {m: j for j, m in enumerate(mb_idx)}
    PW = PW0[:, [rpos[m] for m in cand]].copy()
    # odors: disjoint random glomerulus (PN cell type) sets, both hemispheres
    gl = pn.groupby('cell_type').size()
    gl = gl[gl >= 2].index.values
    pick = rng.choice(gl, glom_per_odor * n_actions, replace=False)
    odor_types = [list(pick[i * glom_per_odor:(i + 1) * glom_per_odor]) for i in range(n_actions)]
    odor_pns = [pn.index[pn.cell_type.isin(t)].values for t in odor_types]
    return dict(kc=kc, cand=cand, cand_type=ctype, cand_types=cand_types, PW0=PW,
                odor_types=odor_types, odor_pns=odor_pns, pn_all=pn.index.values,
                alpn_all=cells.index[cells.cell_class == 'ALPN'].values)

def strip_block(indptr, indices, data, pre, post):
    """remove pre->post entries (they move into the plastic block)"""
    keep = np.ones(len(indices), bool)
    ispost = np.zeros(indptr.size - 1, bool); ispost[post] = True
    for k in pre:
        a, b = indptr[k], indptr[k + 1]
        keep[a:b] &= ~ispost[indices[a:b]]
    lens = np.add.reduceat(keep.astype(np.int64), indptr[:-1]) if False else None
    rows = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    rows = rows[keep]
    nip = np.zeros(indptr.size, np.int64); np.add.at(nip, rows + 1, 1); nip = np.cumsum(nip)
    return nip, indices[keep], data[keep]

def subnetwork(cells, indptr, indices, data, extra=()):
    """reduced circuit: all ALPNs, KCs, MBONs, DANs/MBINs, APL, DPM (real connectome edges among them)"""
    ct = cells.cell_type.fillna(''); cc = cells.cell_class.fillna('')
    sel = cc.isin(['ALPN', 'Kenyon_Cell', 'MBON', 'DAN', 'MBIN']) | ct.isin(['APL', 'DPM'])
    keep_idx = np.sort(np.unique(np.concatenate([cells.index[sel].values, np.asarray(extra, np.int64)])))
    newpos = -np.ones(indptr.size - 1, np.int64); newpos[keep_idx] = np.arange(len(keep_idx))
    ip, ix, dd = [0], [], []
    for k in keep_idx:
        a, b = indptr[k], indptr[k + 1]
        c = newpos[indices[a:b]]; m = c >= 0
        ix.append(c[m]); dd.append(data[a:b][m]); ip.append(ip[-1] + int(m.sum()))
    return keep_idx, np.array(ip, np.int64), np.concatenate(ix).astype(np.int32), np.concatenate(dd).astype(np.float32), newpos



def scale_block(indptr, indices, data, pre, post, gain):
    data = data.copy()
    ispost = np.zeros(indptr.size - 1, bool); ispost[post] = True
    for k in pre:
        a, b = indptr[k], indptr[k + 1]
        m = ispost[indices[a:b]]
        data[a:b][m] *= gain
    return data
