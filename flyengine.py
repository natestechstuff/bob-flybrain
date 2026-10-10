
"""flyengine.py - numpy-only leaky integrate-and-fire engine for the FlyWire connectome.

Same neuron model and parameters as Shiu et al. 2024 (philshiu/Drosophila_brain_model, model.py):
  dv/dt = (v0 - v + g)/t_mbr ;  dg/dt = -g/tau ; spike when v > v_th ; reset v=v_rst, g=0
  refractory 2.2 ms, synaptic delay 1.8 ms, 0.275 mV per synapse (sign from neurotransmitter),
  Poisson drive 0.275*250 mV per input spike (no refractory period for driven neurons).
Only numpy is needed here, so it can be ported (phone / Xbox) as-is.
"""
import numpy as np

P = dict(dt=0.1, v0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0,
         t_rfc=2.2, t_dly=1.8, w_syn=0.275, f_poi=250.0)

def rows_gather(indptr, s):
    """flat positions of all entries in CSR rows s (numpy only)"""
    starts = indptr[s]; lens = indptr[s + 1] - starts
    tot = int(lens.sum())
    if tot == 0:
        return np.empty(0, dtype=np.int64)
    offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return np.arange(tot, dtype=np.int64) + offs

class Brain:
    """W given as CSR with rows = presynaptic, columns = postsynaptic, data = signed synapse count.
    Optional plastic block: rows = plastic_pre (e.g. Kenyon cells), cols = plastic_post (readout MBONs),
    weights PW (len(pre) x len(post)) in synapse-count units; these entries must NOT also be in W."""
    def __init__(self, indptr, indices, data, n, plastic_pre=None, plastic_post=None, PW=None, params=None):
        self.p = dict(P); self.p.update(params or {})
        self.indptr = indptr.astype(np.int64); self.indices = indices.astype(np.int32)
        self.data = (data * self.p['w_syn']).astype(np.float32)
        self.n = n
        self.pre = None
        if plastic_pre is not None:
            self.pre = np.asarray(plastic_pre); self.post = np.asarray(plastic_post)
            self.pre_pos = -np.ones(n, dtype=np.int64); self.pre_pos[self.pre] = np.arange(len(self.pre))
            self.PW = np.asarray(PW, dtype=np.float32)
        self.dly = int(round(self.p['t_dly'] / self.p['dt']))
        self.rfc_steps = int(round(self.p['t_rfc'] / self.p['dt']))

    def run(self, drive_idx, rate_hz, t_ms, rng, record=None, silent=None):
        """Poisson-drive neurons drive_idx at rate_hz for t_ms. Returns spike counts per neuron (int32)."""
        p = self.p; n = self.n; dt = p['dt']
        v = np.full(n, p['v0'], np.float32); g = np.zeros(n, np.float32)
        rfc = np.zeros(n, np.int16)
        counts = np.zeros(n, np.int32)
        ring = [np.empty(0, np.int64) for _ in range(self.dly)]
        a_m = np.float32(dt / p['t_mbr']); decay = np.float32(np.exp(-dt / p['tau']))
        kick = np.float32(p['w_syn'] * p['f_poi']); prob = rate_hz * dt / 1000.0
        drive_idx = np.asarray(drive_idx, dtype=np.int64)
        steps = int(round(t_ms / dt))
        v0 = np.float32(p['v0']); vth = np.float32(p['v_th'])
        for t in range(steps):
            # deliver spikes emitted dly steps ago
            s = ring[t % self.dly]
            if s.size:
                pos = rows_gather(self.indptr, s)
                if pos.size:
                    g += np.bincount(self.indices[pos], weights=self.data[pos], minlength=n).astype(np.float32)
                if self.pre is not None:
                    k = self.pre_pos[s]; k = k[k >= 0]
                    if k.size:
                        g[self.post] += self.PW[k].sum(0) * p['w_syn']
            # Poisson input
            if drive_idx.size:
                hit = drive_idx[rng.random(drive_idx.size) < prob]
                if hit.size:
                    np.add.at(g, hit, kick)
            # integrate (frozen while refractory, as in Brian2 'unless refractory')
            free = rfc <= 0
            v = np.where(free, v + a_m * (v0 - v + g), v)
            g = np.where(free, g * decay, g)
            rfc -= 1
            sp = np.flatnonzero(v > vth)
            if silent is not None and sp.size:
                sp = sp[~silent[sp]]
            if sp.size:
                counts[sp] += 1
                v[sp] = p['v_rst']; g[sp] = 0.0
                r = np.full(sp.size, self.rfc_steps, np.int16)
                if drive_idx.size:
                    r[np.isin(sp, drive_idx)] = 0
                rfc[sp] = r
            ring[t % self.dly] = sp
            if record is not None:
                record(t, sp)
        return counts

