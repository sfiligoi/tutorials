import numpy
import cupy
import h5py
import time

# A variant of 
#https://github.com/sfiligoi/scikit-bio/blob/b0ed7ed183ed9d128fcae8b51288e3e57ddd65a1/skbio/stats/ordination/_utils.py#L201
def e_matrix(distance_matrix):
    return distance_matrix * distance_matrix / -2

def f_matrix(E_matrix):
    row_means = E_matrix.mean(axis=1, keepdims=True)
    col_means = E_matrix.mean(axis=0, keepdims=True)
    matrix_mean = E_matrix.mean()
    return E_matrix - row_means - col_means + matrix_mean

def center_distance_matrix(distance_matrix):
    return f_matrix(e_matrix(distance_matrix))

# read input


with h5py.File('uw_emp.h5','r') as f:
    mat = f['matrix'][:,:].copy()

# get smaller variants
# and create cupy equivalents

cmat = cupy.asarray(mat)

mat_small=mat[:2794,:2794].copy()
cmat_small = cupy.asarray(mat_small)

mat_med=mat[:8382,:8382].copy()
cmat_med = cupy.asarray(mat_med)

# initialize the GPU compute, to make benchmarking results fair
r2 = center_distance_matrix(cupy.asarray(mat[:100,:100].copy()).copy())
cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute

t1 = time.time()
r1 = center_distance_matrix(mat_small)
t2 = time.time()
print("Small on CPU: ", t2-t1)

t1 = time.time()
r2 = center_distance_matrix(cmat_small)
cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
t2 = time.time()
print("Small on GPU: ", t2-t1)


print("Medium matrix shape: ", mat_med.shape)

t1 = time.time()
r1 = center_distance_matrix(mat_med)
t2 = time.time()
print("Medium on CPU: ", t2-t1)

t1 = time.time()
r2 = center_distance_matrix(cmat_med)
cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
t2 = time.time()
print("Medium on GPU: ", t2-t1)

print("Large matrix shape: ", mat.shape)

t1 = time.time()
r1 = center_distance_matrix(mat)
t2 = time.time()
print("Large on CPU: ", t2-t1)

t1 = time.time()
r2 = center_distance_matrix(cmat)
cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
t2 = time.time()
print("Large on GPU: ", t2-t1)

print("... Retrying")
t1 = time.time()
r2 = center_distance_matrix(mat)
t2 = time.time()
print("Large on CPU: ", t2-t1)

t1 = time.time()
r1 = center_distance_matrix(cmat)
cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
t2 = time.time()
print("Large on GPU: ", t2-t1)

#
# Modern scikit-bio contains a CPU-optimized version of center_distance_matrix
# You can check how it performs as an exercise
#
#from skbio.stats.ordination._utils import center_distance_matrix as center_distance_matrix_skbio
#t1 = time.time()
#r2 = center_distance_matrix_skbio(mat)
#t2 = time.time()
#print("Large on CPU using skbio: ", t2-t1)


