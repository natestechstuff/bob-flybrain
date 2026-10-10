
"""build_connectome.py - turn Connectivity_783.parquet into a cached CSR matrix + cell-type table."""
import numpy as np, pandas as pd, time, resource, os
t0 = time.time()
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
comp = pd.read_csv(f'{D}/Completeness_783.csv', index_col=0)
ids = comp.index.values.astype(np.int64); n = len(ids)
con = pd.read_parquet(f'{D}/Connectivity_783.parquet',
                      columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'])
print('columns ok, rows', len(con), 'neurons', n)
pre = con['Presynaptic_Index'].values.astype(np.int64)
post = con['Postsynaptic_Index'].values.astype(np.int32)
w = con['Excitatory x Connectivity'].values.astype(np.float32)
del con
order = np.argsort(pre, kind='stable')
pre, post, w = pre[order], post[order], w[order]; del order
indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, pre + 1, 1); indptr = np.cumsum(indptr)
np.savez(f'{D}/brain_csr.npz', indptr=indptr, indices=post, data=w, ids=ids)
ann = pd.read_csv(f'{D}/neuron_annotations.tsv', sep='\t', usecols=['root_id', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'side', 'top_nt'], dtype=str)
ann['root_id'] = ann['root_id'].astype(np.int64)
ann = ann.drop_duplicates('root_id').set_index('root_id')
table = ann.reindex(ids)
table.index.name = 'root_id'
table['idx'] = np.arange(n)
table.to_csv(f'{D}/cells.csv')
print('annotated:', table['cell_class'].notna().sum() + table['super_class'].notna().sum() - (table['cell_class'].notna() & table['super_class'].notna()).sum(), 'of', n)
print('synapse entries', len(w), 'excitatory', int((w > 0).sum()), 'inhibitory', int((w < 0).sum()))
print('KCs', int((table.cell_class == 'Kenyon_Cell').sum()), 'MBONs', int((table.cell_class == 'MBON').sum()), 'ALPNs', int((table.cell_class == 'ALPN').sum()),
      'APL', int((table.cell_type == 'APL').sum()), 'DAN', int((table.cell_class == 'DAN').sum()))
print(f'build time {time.time()-t0:.1f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB')

