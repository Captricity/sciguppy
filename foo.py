import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.linalg as linalg
import scikits.cuda.cublas as cublas
import scikits.cuda.misc as misc

import time

a = np.random.rand(10000, 50).astype(np.float32)
x = np.random.rand(50, 1).astype(np.float32)

a_gpu = gpuarray.to_gpu(a)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.empty((10000, 1), np.float32)
alpha = np.float32(1.0)
beta = np.float32(0.0)

h = cublas.cublasCreate()

start = time.time()
asdf = linalg.dot(a_gpu, x_gpu, handle=h, out=y_gpu)
print time.time() - start

start = time.time()
a_gpuT = linalg.transpose(a_gpu, handle=h)
cublas.cublasSgemv(h, 'n', 10000, 50, alpha, 
                           a_gpuT.gpudata, 10000, x_gpu.gpudata,
                                              1, beta, y_gpu.gpudata, 1)
print time.time() - start

start = time.time()
z = np.dot(a, x)
print time.time() - start

cublas.cublasDestroy(h)

assert np.allclose(y_gpu.get(), z)
assert np.allclose(asdf.get(), z)

