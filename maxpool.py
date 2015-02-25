import time
import numpy as np
import pycuda.gpuarray as gp
from sciguppy import maxpool, ArrayReturnTypes

a = np.random.rand(100, 51, 50).astype(np.float32)
d_a = gp.to_gpu(a)

def maxpool_cpu(inputt, h, w):
    in_z = inputt.shape[0]
    in_y = inputt.shape[1]
    in_x = inputt.shape[2]
    out_y = in_y // h
    out_x = in_x // w

    out = np.empty((in_z, out_y, out_x), dtype=np.float32)
    max_indices = np.empty((in_z, out_y, out_x, 2), dtype=np.float32)

    for z in range(in_z):
        start_y = 0
        for bucket_y in range(out_y):
            start_x = 0
            for bucket_x in range(out_x):
                max_val = -999999

                for dy in range(h):
                    for dx in range(w):
                        y = start_y + dy
                        x = start_x + dx
                        val = inputt[z, y, x]
                        if val > max_val:
                            max_val = val
                            out[z, bucket_y, bucket_x] = max_val
                            max_indices[z, bucket_y, bucket_x, 0] = y
                            max_indices[z, bucket_y, bucket_x, 1] = x
                start_x += w
            start_y += h

    return out, max_indices

gpu = maxpool(a, (2,2))
gpu = maxpool(a, (2,2))

start = time.time()
out, max_idxs = maxpool_cpu(a, 2, 2)
print 'cpu', time.time() - start

start = time.time()
d_out, d_max_idxs = maxpool(d_a, (2, 2), return_type=ArrayReturnTypes.GPU)
print 'gpu', time.time() - start

print np.allclose(out, d_out.get())
print np.allclose(max_idxs, d_max_idxs.get())
