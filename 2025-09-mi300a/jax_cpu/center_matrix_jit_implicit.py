import numpy
import jax
import h5py
import time

# A variant of 
#https://github.com/sfiligoi/scikit-bio/blob/b0ed7ed183ed9d128fcae8b51288e3e57ddd65a1/skbio/stats/ordination/_utils.py#L201

@jax.jit
def e_matrix(distance_matrix):
    return distance_matrix * distance_matrix / -2

@jax.jit
def f_matrix(E_matrix):
    row_means = E_matrix.mean(axis=1, keepdims=True)
    col_means = E_matrix.mean(axis=0, keepdims=True)
    matrix_mean = E_matrix.mean()
    return E_matrix - row_means - col_means + matrix_mean

# Automatically JIT compile the function using JAX
@jax.jit
def center_distance_matrix(distance_matrix):
    return f_matrix(e_matrix(distance_matrix))

# read input


with h5py.File('uw_emp.h5','r') as f:
    mat = f['matrix'][:,:].copy()

# get smaller variants
# and create jax equivalents

jmat = jax.numpy.asarray(mat)

mat_small=mat[:2794,:2794].copy()
jmat_small = jax.numpy.asarray(mat_small)

mat_med=mat[:8382,:8382].copy()
jmat_med = jax.numpy.asarray(mat_med)

#
# Note:
#  JAX calls are asynchronous, so we will use
#  explict waits in the code to make benchmarking results valid.
#  You do not need that in production code, 
#  as JAX will wait for results as needed.
#

# we will also compare against a slightly more modern, tuned CPU version
from skbio.stats.ordination._utils import center_distance_matrix as center_distance_matrix_skbio

for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    r1 = center_distance_matrix(mat_small)
    # r1 type is jax.array
    r1.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Small on CPU numpy: ", t2-t1)

    t1 = time.time()
    r2 = center_distance_matrix(jmat_small)
    r2.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Small on CPU jax  :: ", t2-t1)

    t1 = time.time()
    r3 = center_distance_matrix_skbio(mat_small)
    t2 = time.time()
    print("Small on CPU tuned: ", t2-t1)


    print("Medium matrix shape: ", mat_med.shape)

    t1 = time.time()
    r1 = center_distance_matrix(mat_med)
    # r1 type is jax.array
    r1.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Medium on CPU numpy: ", t2-t1)

    t1 = time.time()
    r2 = center_distance_matrix(jmat_med)
    r2.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Medium on CPU jax  : ", t2-t1)

    t1 = time.time()
    r3 = center_distance_matrix_skbio(mat_med)
    t2 = time.time()
    print("Medium on CPU tuned: ", t2-t1)

    print("Large matrix shape: ", mat.shape)

    t1 = time.time()
    r1 = center_distance_matrix(mat)
    # r1 type is jax.array
    r1.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Large on CPU numpy: ", t2-t1)

    t1 = time.time()
    r2 = center_distance_matrix(jmat)
    r2.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Large on CPU jax  : ", t2-t1)

    t1 = time.time()
    r3 = center_distance_matrix_skbio(mat)
    t2 = time.time()
    print("Large on CPU tuned: ", t2-t1)

    # touch values to get a different result at next try
    mat[2,3] -= 0.001
    mat[3,2] -= 0.001
    mat_small[2,3] -= 0.001
    mat_small[3,2] -= 0.001
    mat_med[2,3] -= 0.001
    mat_med[3,2] -= 0.001
    # JAX arrays are immutable, must do it the hard way
    nval = mat_med[2,3] 
    jmat.at[2,3].set(nval)
    jmat.at[3,2].set(nval)
    jmat_small.at[2,3].set(nval)
    jmat_small.at[3,2].set(nval)
    jmat_med.at[2,3].set(nval)
    jmat_med.at[3,2].set(nval)



