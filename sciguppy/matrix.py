import numpy
import atexit
import time
import scikits.cuda.cublas as cublas
import scikits.cuda.linalg as linalg

from .utils import gpu_func

handle = cublas.cublasCreate()
def destroy_cublas():
    cublas.cublasDestroy(handle)
atexit.register(destroy_cublas)

@gpu_func
def dot(d_a, d_b):
    return linalg.dot(d_a, d_b, handle=handle)
