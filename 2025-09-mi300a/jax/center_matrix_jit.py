import numpy
import jax
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

# a function can be implicitly JAX compiled
@jax.jit
def center_distance_matrix_v2(distance_matrix):
    return f_matrix(e_matrix(distance_matrix))

# read input


with h5py.File('uw_emp.h5','r') as f:
    mat = f['matrix'][:,:].copy()

# get smaller variants
# and create jax equivalents

mat2 = mat.copy()
mat3 = mat.copy()
jmat = jax.numpy.asarray(mat)
jmat2 = jax.numpy.asarray(mat2)

mat_small=mat[:2794,:2794].copy()
mat2_small = mat_small.copy()
mat3_small = mat_small.copy()
jmat_small = jax.numpy.asarray(mat_small)
jmat2_small = jax.numpy.asarray(mat2_small)

mat_med=mat[:8382,:8382].copy()
mat2_med = mat_med.copy()
mat3_med = mat_med.copy()
jmat_med = jax.numpy.asarray(mat_med)
jmat2_med = jax.numpy.asarray(mat2_med)

#
# Note:
#  JAX calls are asynchronous, so we will use
#  explict waits in the code to make benchmarking results valid.
#  You do not need that in production code, 
#  as JAX will wait for results as needed.
#

# explicitly JIT compile the function
t1 = time.time()
jcenter_distance_matrix = jax.jit(center_distance_matrix)
t2 = time.time()
print("JIT compilation: ", t2-t1)

# we will also compare against a slightly more modern, tuned CPU version
from skbio.stats.ordination._utils import center_distance_matrix as center_distance_matrix_skbio

for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    r1 = center_distance_matrix(mat_small)
    t2 = time.time()
    print("Small on CPU naive    : ", t2-t1)

    t1 = time.time()
    r3 = center_distance_matrix_skbio(mat2_small)
    t2 = time.time()
    print("Small on CPU tuned    : ", t2-t1)

    t1 = time.time()
    r2 = jcenter_distance_matrix(jmat_small)
    r2.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Small on GPU expl JIT : ", t2-t1)

    t1 = time.time()
    r4 = center_distance_matrix_v2(jmat2_small)
    r4.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Small on GPU impl JIT : ", t2-t1)

    t1 = time.time()
    # Note: This will run using JAX, on the GPU
    #       The numpy array is automatically converted to jax array
    #       and the output is a jax array, too
    r4 = center_distance_matrix_v2(mat3_small)
    r4.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Small on GPU JIT+numpy: ", t2-t1)


    print("Medium matrix shape: ", mat_med.shape)

    t1 = time.time()
    r1 = center_distance_matrix(mat_med)
    t2 = time.time()
    print("Medium on CPU naive    : ", t2-t1)

    t1 = time.time()
    r3 = center_distance_matrix_skbio(mat2_med)
    t2 = time.time()
    print("Medium on CPU tuned    : ", t2-t1)

    t1 = time.time()
    r2 = jcenter_distance_matrix(jmat_med)
    r2.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Medium on GPU expl JIT : ", t2-t1)

    t1 = time.time()
    r4 = center_distance_matrix_v2(jmat2_med)
    r4.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Medium on GPU impl JIT : ", t2-t1)

    t1 = time.time()
    # Note: This will run using JAX, on the GPU
    #       The numpy array is automatically converted to jax array
    #       and the output is a jax array, too
    r4 = center_distance_matrix_v2(mat3_med)
    r4.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Medium on GPU JIT+numpy: ", t2-t1)



    print("Large matrix shape: ", mat.shape)

    t1 = time.time()
    r1 = center_distance_matrix(mat)
    t2 = time.time()
    print("Large on CPU naive    : ", t2-t1)

    t1 = time.time()
    r3 = center_distance_matrix_skbio(mat2)
    t2 = time.time()
    print("Large on CPU tuned    : ", t2-t1)

    t1 = time.time()
    r2 = jcenter_distance_matrix(jmat)
    r2.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Large on GPU expl JIT : ", t2-t1)

    t1 = time.time()
    r4 = center_distance_matrix_v2(jmat2)
    r4.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Large on GPU impl JIT : ", t2-t1)

    t1 = time.time()
    # Note: This will run using JAX, on the GPU
    #       The numpy array is automatically converted to jax array
    #       and the output is a jax array, too
    r4 = center_distance_matrix_v2(mat3)
    r4.block_until_ready()  # JAX is async... wait for actual compute
    t2 = time.time()
    print("Large on GPU JIT+numpy: ", t2-t1)

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



