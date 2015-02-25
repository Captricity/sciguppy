import time
import numpy as np
import pycuda.gpuarray as gp
from sciguppy import collapse, ArrayReturnTypes

inn = np.random.rand(10, 100*90).astype(np.float32)
d_inn = gp.to_gpu(inn)

collapse(d_inn)

start = time.time()
out_cpu = np.sum(inn, axis=(1,))
print 'cpu', time.time() - start

start = time.time()
out_gpu = collapse(d_inn, return_type=ArrayReturnTypes.GPU)
print 'gpu', time.time() - start

print out_cpu.shape, out_gpu.shape
print out_cpu
print out_gpu
print np.allclose(out_cpu, out_gpu.get())
