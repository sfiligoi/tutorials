import numpy
import pandas
import skbio
import h5py
import time

md = pandas.read_csv('emp_qiime_mapping_release1.tsv',sep='\t').set_index('#SampleID')
with h5py.File('uw_emp.h5','r') as f:
    mat = f['matrix'][:,:]
    ids = [c.decode('ascii') for c in f['order'][:]]

mat_small=mat[:2794,:2794].copy()
ids_small=ids[:2794].copy()

mat_med=mat[:8382,:8382].copy()
ids_med=ids[:8382].copy()

dm_small=skbio.DistanceMatrix(mat_small,ids_small)
dm_med=skbio.DistanceMatrix(mat_med,ids_med)
dm=skbio.DistanceMatrix(mat,ids)

# wamup the backend
p1 = skbio.stats.distance.permanova(dm_small.copy(),md['empo_3'].copy(), permutations=99)

print("Small matrix shape: ", mat_small.shape)

t1 = time.time()
p1 = skbio.stats.distance.permanova(dm_small,md['empo_3'], permutations=999)
t2 = time.time()
print("Small permutations=1k  : ", t2-t1)

t1 = time.time()
p1 = skbio.stats.distance.permanova(dm_small,md['empo_3'], permutations=99999)
t2 = time.time()
print("Small permutations=100k: ", t2-t1)

print("Medium matrix shape: ", mat_med.shape)

t1 = time.time()
p1 = skbio.stats.distance.permanova(dm_med,md['empo_3'], permutations=999)
t2 = time.time()
print("Medium permutations=1k  : ", t2-t1)

t1 = time.time()
p1 = skbio.stats.distance.permanova(dm_med,md['empo_3'], permutations=99999)
t2 = time.time()
print("Medium permutations=100k: ", t2-t1)

print("Large matrix shape: ", mat.shape)

t1 = time.time()
p1 = skbio.stats.distance.permanova(dm,md['empo_3'], permutations=999)
t2 = time.time()
print("Large permutations=1k  : ", t2-t1)

t1 = time.time()
p1 = skbio.stats.distance.permanova(dm,md['empo_3'], permutations=99999)
t2 = time.time()
print("Large permutations=100k: ", t2-t1)


