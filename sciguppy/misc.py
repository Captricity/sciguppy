__all__ = ['ewsum', 'ewsum_back']

import os
import math
import numpy
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .matrix import matrix_addition
from .utils import gpu_func
from .enums import MAX_BLOCK_SIZE

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(CUR_DIR, 'cache')

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/ewsum.cu')).read(), cache_dir=CACHE_DIR)
ewsum_kernel = mod.get_function('ewsum_kernel')
ewsum_sum_kernel = mod.get_function('ewsum_sum_kernel')
ewsum_back_kernel = mod.get_function('ewsum_back_kernel')

@gpu_func
def ewsum(d_a, d_w):
    """
    YORI NOTES

    This method is faster than CPU if num_w is large, and non_width is small:
        When num_w is large, the for loop is small
        When non_width is large, there are more threads necessary
    """
    width = d_a.shape[0]
    total_dim = d_a.size
    num_w = d_w.shape[0]
    d_tmp_out = gpuarray.zeros_like(d_a)
    
    thread_size = min(d_a.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_a.size / float(thread_size))), 1)
    ewsum_kernel(d_a, d_w, d_tmp_out,
            numpy.int32(num_w), numpy.int32(width), numpy.int32(total_dim),
            block=(thread_size,1,1), grid=(block_size,1,1))

    # TODO: There HAS to be a better way to do this
    x = width / num_w
    d_out = gpuarray.zeros((x,) + d_a.shape[1:], numpy.float32)
    thread_size = min(d_out.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_out.size / float(thread_size))), 1)
    ewsum_sum_kernel(d_tmp_out, d_out,
            numpy.int32(num_w), numpy.int32(width), numpy.int32(total_dim),
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out

@gpu_func
def ewsum_back(d_error, d_w):
    d_out = gpuarray.zeros((d_w.shape[0]*d_error.shape[0],) + d_error.shape[1:], dtype=d_error.dtype)
    err_width = d_error.shape[0]
    width = d_out.shape[0]
    total_dim = d_out.size
    num_w = d_w.shape[0]

    thread_size = min(d_out.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_out.size / float(thread_size))), 1)
    ewsum_back_kernel(d_error, d_w, d_out,
            numpy.int32(num_w), numpy.int32(err_width), numpy.int32(width), numpy.int32(total_dim),
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out
