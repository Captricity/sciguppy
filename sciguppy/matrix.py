import numpy
import atexit
import time
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import scikits.cuda.cublas as cublas
import scikits.cuda.linalg as linalg

from .utils import gpu_func

handle = cublas.cublasCreate()
def destroy_cublas():
    cublas.cublasDestroy(handle)
atexit.register(destroy_cublas)

@gpu_func
def dot(d_a, d_b, out=None):
    if out is None:
        out = gpuarray.empty((d_a.shape[0], d_b.shape[1]), numpy.float32)
    return linalg.dot(d_a, d_b, handle=handle, out=out)

@gpu_func
def vector_addition(d_a, d_b):
    d_b = d_b.copy()
    assert len(d_a.shape) == 1
    assert len(d_b.shape) == 1
    assert d_a.size == d_b.size
    cublas.cublasSaxpy(handle, d_a.size, 1.0, d_a.gpudata, 1, d_b.gpudata, 1)
    return d_b
