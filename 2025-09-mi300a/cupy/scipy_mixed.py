import numpy
import cupy
import scipy
import h5py
import time

# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

with h5py.File('uw_emp.h5','r') as f:
    mat_large = f['matrix'][:,:]

# get a divisible size
mat1_large = mat_large[:25024,:25024].copy()
mat2_large = mat1_large.copy()
cmat1_large = cupy.asarray(mat1_large).copy()
cmat2_large = cupy.asarray(mat2_large).copy()

for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    mat1_large=numpy.sqrt(mat1_large*1.01)**2
    mat2_large=numpy.sqrt(mat2_large*0.99)**2
    # spearman is slow, so only test with small matrix
    mat1 = rebin(mat1_large, (1564,1564))
    mat2 = rebin(mat2_large, (1564,1564))
    t2 = time.time()
    print("Prepare on CPU : ", t2-t1)
    n1c = scipy.stats.spearmanr(mat1,mat2)
    t3 = time.time()
    print("Spearman on CPU: ", t3-t2)

    t1 = time.time()
    cmat1_large=cupy.sqrt(cmat1_large*1.01)**2
    cmat2_large=cupy.sqrt(cmat2_large*0.99)**2
    # spearman is slow, so only test with small matrix
    cmat1 = rebin(cmat1_large, (1564,1564))
    cmat2 = rebin(cmat2_large, (1564,1564))
    t2 = time.time()
    print("Prepare on GPU : ", t2-t1)
    # spearmanr is not cupy aware, so we need to explicitly convert
    n1c = scipy.stats.spearmanr(cupy.asnumpy(cmat1),cupy.asnumpy(cmat2))
    t3 = time.time()
    print("Spearman on CPU: ", t3-t2)
