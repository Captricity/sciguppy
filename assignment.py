import numpy as np
import pycuda.gpuarray as gp
from sciguppy import subset_assignment, subset_slice_assignment

a = np.random.rand(10, 2, 3).astype(np.float32)
b = np.random.rand(2, 3).astype(np.float32)
c = np.random.rand(2, 2, 3).astype(np.float32)
d_a = gp.to_gpu(a)
d_b = gp.to_gpu(b)
d_c = gp.to_gpu(c)

subset_assignment(d_a, d_b, 2)
subset_slice_assignment(d_a, d_c, (2, 4))

print d_a[2]
print d_b
print
print d_a[2:4]
print d_c
