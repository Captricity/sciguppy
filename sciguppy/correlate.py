__all__ = ['correlate']

import os
import numpy
import math
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray
from pycuda.compiler import SourceModule
from .utils import as_gpu, as_cpu
from .enums import CorrelationModes, ArrayReturnTypes, MAX_BLOCK_SIZE

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(CUR_DIR, 'cache')

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/correlate.cu')).read(), cache_dir=CACHE_DIR)
correlate_kernel = mod.get_function('correlate_kernel')
d_ax_size = mod.get_global('d_ax_size')[0]
d_ay_size = mod.get_global('d_ay_size')[0]
d_aout_size = mod.get_global('d_aout_size')[0]
d_padding = mod.get_global('d_padding')[0]

def correlate(a1, a2, mode=CorrelationModes.FULL, return_type=ArrayReturnTypes.CPU):
    x1, y1, z1 = a1.shape
    x2, y2, z2 = a2.shape
    if mode == CorrelationModes.FULL:
        # In FULL mode, cycle through minimum overlap, including those where
        # the array is out of bounds. Out of bound values are treated as 0s
        outx, outy, outz = x1+x2-1, y1+y2-1, z1+z2-1
        xpad = x2 - 1
        ypad = y2 - 1
        zpad = z2 - 1
    else:
        outx, outy, outz = x1-x2+1, y1-y2+1, z1-z2+1
        xpad = 0
        ypad = 0
        zpad = 0

    d_a1 = as_gpu(a1)
    cuda.memcpy_htod(d_ax_size, numpy.array(a1.shape, dtype=numpy.int32))
    d_a2 = as_gpu(a2)
    cuda.memcpy_htod(d_ay_size, numpy.array(a2.shape, dtype=numpy.int32))
    d_aout = pycuda.gpuarray.zeros((outx, outy, outz), numpy.float32)
    cuda.memcpy_htod(d_aout_size, numpy.array(d_aout.shape, dtype=numpy.int32))
    cuda.memcpy_htod(d_padding, numpy.array((xpad, ypad, zpad), dtype=numpy.int32))

    thread_size = min(outx*outy*outz, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(outx*outy*outz / float(thread_size))), 1)
    correlate_kernel(d_a1, d_a2, d_aout, 
            block=(thread_size,1,1), grid=(block_size,1,1))

    if return_type == ArrayReturnTypes.CPU:
        return as_cpu(d_aout)
    else:
        return as_gpu(d_aout)
