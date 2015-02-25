import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gp

def ewsum_back_cpu(inputt, error, w):
    s = list(error.shape)
    step = s[0]
    s[0] *= len(w)
    in_error = np.empty(s, dtype=np.float32)
    start = 0
    for weight in w:
        in_error[start:start+step] = weight * error
        start += step
    return in_error

inn = np.random.rand(2, 10, 30).astype(np.float32)
w = np.random.rand(2).astype(np.float32)
err = np.random.rand(1, 10, 20).astype(np.float32)
d_inn = gp.to_gpu(inn)
d_w = gp.to_gpu(w)
d_err = gp.to_gpu(err)

print ewsum_back_cpu(inn, err, w).shape
