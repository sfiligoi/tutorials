import numpy
import jax
import h5py
import time

def e_matrix(distance_matrix):
    return distance_matrix * distance_matrix / -2

def f_matrix(E_matrix):
    row_means = E_matrix.mean(axis=1, keepdims=True)
    col_means = E_matrix.mean(axis=0, keepdims=True)
    matrix_mean = E_matrix.mean()
    return E_matrix - row_means - col_means + matrix_mean

# implicitly JAX compiled function
@jax.jit
def center_distance_matrix_jax(distance_matrix):
    return f_matrix(e_matrix(distance_matrix))

# we will also compare against the tuned CPU version
from skbio.stats.ordination._utils import center_distance_matrix as center_distance_matrix_skbio

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

for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    c1 = center_distance_matrix_skbio(mat_small)
    q1,r1 = numpy.linalg.qr(c1)
    t2 = time.time()
    print("Small on CPU: ", t2-t1)

    t1 = time.time()
    c2 = center_distance_matrix_jax(jmat_small)
    # no need to explictly wait on c2
    q2,r2 = jax.numpy.linalg.qr(c2)
    # JAX is async... we wait before the timing function
    q2.block_until_ready()
    r2.block_until_ready()
    t2 = time.time()
    print("Small on GPU: ", t2-t1)


    print("Medium matrix shape: ", mat_med.shape)

    t1 = time.time()
    r1 = center_distance_matrix_skbio(mat_med)
    q1 = numpy.linalg.qr(r1)
    t2 = time.time()
    print("Medium on CPU: ", t2-t1)

    t1 = time.time()
    c2 = center_distance_matrix_jax(jmat_med)
    # no need to explictly wait on c2
    q2,r2 = jax.numpy.linalg.qr(c2)
    # JAX is async... we wait before the timing function
    q2.block_until_ready()
    r2.block_until_ready()
    t2 = time.time()
    print("Medium on GPU: ", t2-t1)


    print("Large matrix shape: ", mat.shape)

    t1 = time.time()
    r1 = center_distance_matrix_skbio(mat)
    q1 = numpy.linalg.qr(r1)
    t2 = time.time()
    print("Large on CPU: ", t2-t1)

    t1 = time.time()
    c2 = center_distance_matrix_jax(jmat)
    # no need to explictly wait on c2
    q2,r2 = jax.numpy.linalg.qr(c2)
    # JAX is async... we wait before the timing function
    q2.block_until_ready()
    r2.block_until_ready()
    t2 = time.time()
    print("Large on GPU: ", t2-t1)
