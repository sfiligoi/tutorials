import numpy
import cupy
import h5py
import time

with h5py.File('uw_emp.h5','r') as f:
    mat = f['matrix'][:,:]

cmat = cupy.asarray(mat)

mat_small=mat[:2794,:2794].copy()
cmat_small = cupy.asarray(mat_small)

mat_med=mat[:8382,:8382].copy()
cmat_med = cupy.asarray(mat_med)

# initialize the GPU compute, to make benchmarking results fair
n = cupy.linalg.qr(cmat_small)

print("Small matrix shape: ", mat_small.shape)

t1 = time.time()
n = numpy.linalg.qr(mat_small)
t2 = time.time()
print("Small on CPU: ", t2-t1)

t1 = time.time()
n = cupy.linalg.qr(cmat_small)
t2 = time.time()
print("Small on GPU: ", t2-t1)

t1 = time.time()
n = numpy.linalg.qr(cmat_small)
t2 = time.time()
print("Small on GPU using numpy: ", t2-t1)

t1 = time.time()
try:
  n = cupy.linalg.qr(mat_small)
  t2 = time.time()
  print("Small using numpy array with cupy: ", t2-t1)
except:
  print("[INFO] Cannot pass numpy array to cupy")

print("Medium matrix shape: ", mat_med.shape)

t1 = time.time()
n = numpy.linalg.qr(mat_med)
t2 = time.time()
print("Medium on CPU: ", t2-t1)

t1 = time.time()
n = cupy.linalg.qr(cmat_med)
t2 = time.time()
print("Medium on GPU: ", t2-t1)

t1 = time.time()
n = numpy.linalg.qr(cmat_med)
t2 = time.time()
print("Medium on GPU using numpy: ", t2-t1)

print("Large matrix shape: ", mat.shape)

t1 = time.time()
n = numpy.linalg.qr(cmat)
t2 = time.time()
print("Large on GPU using numpy: ", t2-t1)

t1 = time.time()
n = cupy.linalg.qr(cmat)
t2 = time.time()
print("Large on GPU: ", t2-t1)

# we expect this to tak a long time
print("About to compute Large on CPU")
t1 = time.time()
n = numpy.linalg.qr(mat)
t2 = time.time()
print("Large on CPU: ", t2-t1)


