__all__ = ['maxpool', 'maxpool_back']

import os
import math
import numpy
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .utils import gpu_func
from .enums import MAX_BLOCK_SIZE, CUR_DIR, CACHE_DIR

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/maxpool.cu')).read(), cache_dir=CACHE_DIR)
maxpool_kernel = mod.get_function('maxpool_kernel')
maxpool_back_kernel = mod.get_function('maxpool_back_kernel')
d_a_size = mod.get_global('d_a_size')[0]
d_out_size = mod.get_global('d_out_size')[0]

@gpu_func
def maxpool(d_a, window_shape):
    h, w = window_shape
    in_z, in_y, in_x = d_a.shape
    out_z = in_z
    out_y = in_y / h
    out_x = in_x / w

    assert h*w < MAX_BLOCK_SIZE

    cuda.memcpy_htod(d_a_size, numpy.array(d_a.shape, dtype=numpy.int32))
    cuda.memcpy_htod(d_out_size, numpy.array((out_z, out_y, out_x), dtype=numpy.int32))

    d_out = gpuarray.zeros((out_z, out_y, out_x), dtype=numpy.float32)
    d_out_idxs = gpuarray.zeros((out_z, out_y, out_x, 2), dtype=numpy.int32)
    block = (h, w, 1)
    grid = (out_z, out_y, out_x)
    maxpool_kernel(d_a, d_out, d_out_idxs, numpy.int32(h), numpy.int32(w),
            block=block, grid=grid)
    return d_out, d_out_idxs

@gpu_func
def maxpool_back(d_error, d_max_idxs, out_shape):
    cuda.memcpy_htod(d_a_size, numpy.array(d_error.shape, dtype=numpy.int32))
    cuda.memcpy_htod(d_out_size, numpy.array(out_shape, dtype=numpy.int32))

    d_out = gpuarray.zeros(out_shape, dtype=numpy.float32)
    thread_size = min(d_error.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_error.size / float(thread_size))), 1)
    maxpool_back_kernel(d_error, d_max_idxs, d_out,
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out
