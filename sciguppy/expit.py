__all__ = ['expit', 'expit_back', 'exp']

import os
import math
import numpy
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .utils import gpu_func
from .enums import MathModes, MAX_BLOCK_SIZE, CUR_DIR, CACHE_DIR

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/expit.cu')).read(), cache_dir=CACHE_DIR)
expit_kernel = mod.get_function('expit_kernel')
expit_fast_kernel = mod.get_function('expit_fast_kernel')
expit_back_kernel = mod.get_function('expit_back_kernel')
exp_fast_kernel = mod.get_function('exp_fast_kernel')

@gpu_func
def exp(d_a, mode=MathModes.ACC):
    if mode == MathModes.ACC:
        return cumath.exp(d_a)

    d_out = gpuarray.zeros_like(d_a)
    thread_size = min(d_a.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_a.size / float(thread_size))), 1)
    exp_fast_kernel(d_a, d_out, numpy.int32(d_a.size),
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out

@gpu_func
def expit(d_a, mode=MathModes.ACC):
    """Implements the expit function (aka sigmoid)

    expit(x) = 1 / (1 + exp(-x))
    """
    d_out = gpuarray.zeros_like(d_a)
    thread_size = min(d_a.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_a.size / float(thread_size))), 1)
    kernel = expit_fast_kernel if mode == MathModes.FAST else expit_kernel
    kernel(d_a, d_out, numpy.int32(d_a.size),
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out

@gpu_func
def expit_back(d_a, d_error):
    """Implments the following function

    out = in * (1 - in) * error
    """
    d_out = gpuarray.zeros_like(d_a)
    thread_size = min(d_a.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_a.size / float(thread_size))), 1)
    expit_back_kernel(d_a, d_error, d_out, numpy.int32(d_a.size),
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out
