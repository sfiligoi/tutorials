import numpy
import cupy
import scipy
import cupyx
import h5py
import time
import math

with h5py.File('uw_emp.h5','r') as f:
    mat = f['matrix'][:,:]

# avoid zero elements
mat += 0.001
mat2 = mat.copy()

cmat1 = cupy.asarray(mat).copy()
cmat2 = cupy.asarray(mat).copy()
cmat3 = cupy.asarray(mat).copy()
cmat4 = cupy.asarray(mat).copy()

mat_small=mat[:2794,:2794].copy()
mat2_small=mat[:2794,:2794].copy()
cmat1_small = cupy.asarray(mat_small).copy()
cmat2_small = cupy.asarray(mat_small).copy()
cmat3_small = cupy.asarray(mat_small).copy()
cmat4_small = cupy.asarray(mat_small).copy()

mat_med=mat[:8382,:8382].copy()
cmat1_med = cupy.asarray(mat_med).copy()
cmat2_med = cupy.asarray(mat_med).copy()

pi = math.pi

print("=====  Small matrix shape: ", mat_small.shape)
for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    n1c = scipy.special.cotdg(mat2_small).max()
    t2 = time.time()
    print("Small base on CPU   : ", t2-t1)
    n2c = scipy.special.entr(scipy.special.cotdg(mat_small*pi).flatten()).max()
    t3 = time.time()
    print("Small compose on CPU: ", t3-t2)

    t1 = time.time()
    n1 = cupyx.scipy.special.cotdg(cmat3_small).max()
    if abs(n1-n1c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t2 = time.time()
    print("Small base on GPU   : ", t2-t1)
    n2 = cupyx.scipy.special.entr(cupyx.scipy.special.cotdg(cmat1_small*pi).flatten()).max()
    if abs(n2-n2c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t3 = time.time()
    print("Small compose on GPU: ", t3-t2)

    t1 = time.time()
    n1 = scipy.special.cotdg(cmat4_small).max()
    if abs(n1-n1c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t2 = time.time()
    print("Small base on GPU using scipy   : ", t2-t1)
    n2 = scipy.special.entr(scipy.special.cotdg(cmat2_small*pi).flatten()).max()
    if abs(n2-n2c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t3 = time.time()
    print("Small compose on GPU using scipy: ", t3-t2)

t1 = time.time()
try:
  n = cupyx.scipy.special.entr(cupyx.scipy.special.cotdg(mat_small*pi).flatten()).max()
  t2 = time.time()
  print("Small compose using numpy array with cupy: ", t2-t1)
except:
  print("[INFO] Cannot pass numpy array to cupy")

print("====  Medium matrix shape: ", mat_med.shape)

for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    n2c = scipy.special.entr(scipy.special.cotdg(mat_med*pi).flatten()).max()
    t2 = time.time()
    print("Medium compose on CPU: ", t2-t1)

    t1 = time.time()
    n2 = cupyx.scipy.special.entr(cupyx.scipy.special.cotdg(cmat1_med*pi).flatten()).max()
    if abs(n2-n2c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t2 = time.time()
    print("Medium compose on GPU: ", t2-t1)

    t1 = time.time()
    n2 = scipy.special.entr(scipy.special.cotdg(cmat2_med*pi).flatten()).max()
    if abs(n2-n2c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t2 = time.time()
    print("Medium compose on GPU using scipy: ", t2-t1)

print("====  Large matrix shape: ", mat.shape)

for i in range(2):
    print("---- Try ", i+1)

    t1 = time.time()
    n1c = scipy.special.cotdg(mat).max()
    t2 = time.time()
    print("Large base on CPU   : ", t2-t1)
    n2c = scipy.special.entr(scipy.special.cotdg(mat*pi).flatten()).max()
    t3 = time.time()

    print("Large compose on CPU: ", t3-t2)
    t1 = time.time()
    n1 = cupyx.scipy.special.cotdg(cmat3).max()
    if abs(n1-n1c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t2 = time.time()
    print("Large base on GPU   : ", t2-t1)
    n2 = cupyx.scipy.special.entr(cupyx.scipy.special.cotdg(cmat1*pi).flatten()).max()
    if abs(n2-n2c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t3 = time.time()
    print("Large compose on GPU: ", t3-t2)

    t1 = time.time()
    n1 = scipy.special.cotdg(cmat4).max()
    if abs(n1-n1c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t2 = time.time()
    print("Large base on GPU using scipy   : ", t2-t1)
    n2 = scipy.special.entr(scipy.special.cotdg(cmat2*pi).flatten()).max()
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    if abs(n2-n2c) > 0.1:
       printf("[ERROR] Precision test failed") 
    cupy.cuda.Device(0).synchronize() # cuPy is async... wait for actual compute
    t3 = time.time()
    print("Large compose on GPU using scipy: ", t3-t2)


