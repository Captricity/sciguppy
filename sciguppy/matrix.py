__all__ = ['dot', 'subset_assignment', 'subset_slice_assignment', 'vector_subset_slice_assignment', 'matrix_addition']

import os
import numpy
import atexit
import math
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import scikits.cuda.cublas as cublas
import scikits.cuda.linalg as linalg
from pycuda.compiler import SourceModule

from .utils import gpu_func
from .enums import MAX_BLOCK_SIZE, CUR_DIR, CACHE_DIR

handle = cublas.cublasCreate()
def destroy_cublas():
    cublas.cublasDestroy(handle)
atexit.register(destroy_cublas)

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/matrix.cu')).read(), cache_dir=CACHE_DIR)
subset_assignment_kernel = mod.get_function('subset_assignment_kernel')
subset_slice_assignment_kernel = mod.get_function('subset_slice_assignment_kernel')
vector_subset_slice_assignment_kernel = mod.get_function('vector_subset_slice_assignment_kernel')

# TODO: generalize
def subset_assignment(d_a, d_b, a_x):
    thread_size = min(d_b.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_b.size / float(thread_size))), 1)
    subset_assignment_kernel(
            d_a, d_b, numpy.int32(a_x), numpy.int32(d_b.size),
            block=(thread_size,1,1), grid=(block_size,1,1))
def subset_slice_assignment(d_a, d_b, a_x_slice):
    thread_size = min(d_b.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_b.size / float(thread_size))), 1)
    assert (a_x_slice[1] - a_x_slice[0]) == d_b.shape[0]
    subset_slice_assignment_kernel(
            d_a, d_b, numpy.int32(a_x_slice[0]), numpy.int32(d_b.size), numpy.int32(numpy.prod(d_b.shape[1:])),
            block=(thread_size,1,1), grid=(block_size,1,1))
def vector_subset_slice_assignment(d_a, d_b, a_slice):
    thread_size = min(d_b.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_b.size / float(thread_size))), 1)
    assert len(d_a.shape) == 1
    assert len(d_b.shape) == 1
    assert (a_slice[1] - a_slice[0]) == d_b.shape[0]
    vector_subset_slice_assignment_kernel(
            d_a, d_b, numpy.int32(a_slice[0]), numpy.int32(d_b.size),
            block=(thread_size,1,1), grid=(block_size,1,1))



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
    elif len(d_a.shape) == 2:
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
    else:
        tmp = (d_a.ravel() + d_b.ravel()).reshape(d_a.shape)
        cuda.memcpy_dtod(d_a.gpudata, tmp.gpudata, d_a.nbytes)
    return d_a
