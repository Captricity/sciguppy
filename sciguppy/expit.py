__all__ = ['expit']

import os
import math
import numpy
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .utils import gpu_func
from .enums import MathModes, MAX_BLOCK_SIZE

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(CUR_DIR, 'cache')

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/expit.cu')).read(), cache_dir=CACHE_DIR)
expit_kernel = mod.get_function('expit_kernel')
expit_fast_kernel = mod.get_function('expit_fast_kernel')

@gpu_func
def expit(d_a, mode=MathModes.ACC):
    """Implements the expit function (aka sigmoid)

    expit(x) = 1 / (1 + exp(-x))
    """
    d_out = pycuda.gpuarray.zeros(d_a.shape, numpy.float32)
    thread_size = min(d_a.size, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(d_a.size / float(thread_size))), 1)
    kernel = expit_fast_kernel if mode == MathModes.FAST else expit_kernel
    kernel(d_a, d_out, numpy.int32(d_a.size),
            block=(thread_size,1,1), grid=(block_size,1,1))
    return d_out
