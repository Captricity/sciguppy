__all__ = ['FULL', 'VALID', 'correlate']

import os
import numpy
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(CUR_DIR, 'cache')

FULL = 'full'
VALID = 'valid'
MAX_BLOCK_SIZE = 512

mod = SourceModule(open(os.path.join(CUR_DIR, 'kernel/correlate.cu')).read(), cache_dir=CACHE_DIR)
correlate_kernel = mod.get_function('correlate_kernel')
d_ax_size = mod.get_global('d_ax_size')[0]
d_ay_size = mod.get_global('d_ay_size')[0]
d_aout_size = mod.get_global('d_aout_size')[0]

def correlate(a1, a2, mode=FULL):
    x2, y2, z2 = a2.shape
    if mode == FULL:
        xpad = x2 - 1
        ypad = y2 - 1
        zpad = z2 - 1
        a1 = numpy.pad(a1, ((xpad, xpad), (ypad, ypad), (zpad, zpad)), 'constant', constant_values=0)

    x1, y1, z1 = a1.shape
    outx, outy, outz = x1-x2+1, y1-y2+1, z1-z2+1
    aout = numpy.zeros((outx, outy, outz))

    a1 = a1.astype(numpy.float32)
    d_a1 = cuda.mem_alloc(a1.nbytes)
    cuda.memcpy_htod(d_a1, a1)
    cuda.memcpy_htod(d_ax_size, numpy.array(a1.shape, dtype=numpy.int32))
    a2 = a2.astype(numpy.float32)
    d_a2 = cuda.mem_alloc(a2.nbytes)
    cuda.memcpy_htod(d_a2, a2)
    cuda.memcpy_htod(d_ay_size, numpy.array(a2.shape, dtype=numpy.int32))
    aout = aout.astype(numpy.float32)
    d_aout = cuda.mem_alloc(aout.nbytes)
    cuda.memcpy_htod(d_aout_size, numpy.array(aout.shape, dtype=numpy.int32))

    thread_size = min(outx*outy*outz, MAX_BLOCK_SIZE)
    block_size = max(int(math.ceil(outx*outy*outz / float(thread_size))), 1)
    correlate_kernel(d_a1, d_a2, d_aout, 
            block=(thread_size,1,1), grid=(block_size,1,1))
    cuda.memcpy_dtoh(aout, d_aout)
    return aout
