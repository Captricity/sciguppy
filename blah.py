import time
import numpy as np
import pycuda.gpuarray as gp
from sciguppy import matrix_addition, ArrayReturnTypes, ewsum, ewsum_back

def ewsum_cpu(a, w):
    assert a.shape[0] % len(w) == 0
    step = a.shape[0] / len(w)
    output = w[0] * a[:step]
    start = step
    for weight in w[1:]:
        output += weight * a[start:start+step]
        start += step
    return output

def ewsum_back_cpu(error, w):
    s = list(error.shape)
    step = s[0]
    s[0] *= len(w)
    in_error = np.empty(s, dtype=np.float32)
    start = 0
    for weight in w:
        in_error[start:start+step] = weight * error
        start += step
    return in_error

# REALLY CRAPPY
#def ewsum_super_crappy(d_a, w):
#    st = time.time()
#    step = d_a.shape[0] / len(w)
#    d_output = w[0] * d_a[:step]
#    start = step
#    #print time.time() - st
#    for weight in w[1:]:
#        tic = time.time()
#        matrix_addition(d_output, weight*d_a[start:start+step], return_type=ArrayReturnTypes.GPU)
#        start += step
#        #print '\t', time.time() - tic
#    #print time.time() - st
#    return d_output

# CRAPPY
#@gpu_func
#def ewsum_crappy(d_a, d_w):
#    width = d_a.shape[0]
#    total_dim = d_a.size
#    num_w = d_w.shape[0]
#    d_tmp_out = gpuarray.zeros_like(d_a)
#
#    thread_size = min(d_a.size, MAX_BLOCK_SIZE)
#    block_size = max(int(math.ceil(d_a.size / float(thread_size))), 1)
#    ewsum_kernel(d_a, d_w, d_tmp_out,
#            numpy.int32(num_w), numpy.int32(width), numpy.int32(total_dim),
#            block=(thread_size,1,1), grid=(block_size,1,1))
#
#    # TODO: There HAS to be a better way to do this
#    step = width / num_w
#    d_out = d_tmp_out[:step]
#    start = step
#    while start < width:
#        matrix_addition(d_out, d_tmp_out[start:start+step])
#        start += step
#    return d_out


#inn = np.random.rand(10, 4).astype(np.float32)
inn = np.random.rand(10, 1000, 1000).astype(np.float32)
w = np.random.rand(5).astype(np.float32)
d_inn = gp.to_gpu(inn)
d_w = gp.to_gpu(w)

ewsum(d_inn, d_w)
ewsum_back(d_inn, d_w)

start = time.time()
out_cpu = ewsum_cpu(inn, w)
print 'cpu', time.time() - start

start = time.time()
out_gpu = ewsum(d_inn, d_w, return_type=ArrayReturnTypes.GPU)
print 'gpu', time.time() - start

print out_cpu.shape, out_gpu.shape
print np.allclose(out_cpu, out_gpu.get())

start = time.time()
out_cpu = ewsum_back_cpu(inn, w)
print 'back cpu', time.time() - start

start = time.time()
out_gpu = ewsum_back(d_inn, d_w, return_type=ArrayReturnTypes.GPU)
print 'back gpu', time.time() - start

print out_cpu.shape, out_gpu.shape
print np.allclose(out_cpu, out_gpu.get())

