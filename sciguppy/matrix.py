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
def dot(d_a, d_b, transa='N', transb='N', out=None):
    if out is None:
        if transa == 'T':
            out_x = d_a.shape[1]
        else:
            out_x = d_a.shape[0]
        if transb == 'T':
            out_y = d_b.shape[0]
        else:
            out_y = d_b.shape[1]
        out = gpuarray.empty((out_x, out_y), numpy.float32)
    return linalg.dot(d_a, d_b, transa=transa, transb=transb, handle=handle, out=out)

@gpu_func
def matrix_addition(d_a, d_b):
    # Overwrites d_a
    assert d_a.shape == d_b.shape
    if len(d_a.shape) == 1:
        # Vector addition
        cublas.cublasSaxpy(handle, d_a.size, 1.0, d_b.gpudata, 1, d_a.gpudata, 1)
    else:
        # Matrix addition
        m, n = d_a.shape
        cublas.cublasSgeam(handle,
                'N', 'N',
                m, n,
                1.0,
                d_a.gpudata, m,
                1.0,
                d_b.gpudata, m,
                d_a.gpudata, m)
    return d_a
