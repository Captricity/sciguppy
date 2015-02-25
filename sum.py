import numpy as np
import pycuda.gpuarray as gp
from sciguppy import matrix_addition

a = np.random.rand(2, 2, 3).astype(np.float32)
b = np.random.rand(2, 2, 3).astype(np.float32)
d_a = gp.to_gpu(a)
d_b = gp.to_gpu(b)

x = a + b

matrix_addition(d_a, d_b)
y = d_a.get()

print x
print y
