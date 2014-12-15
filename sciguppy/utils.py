__all__ = ['as_gpu', 'as_cpu']

import numpy
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

def as_gpu(a):
    """ Returns a as a GPUArray

    If a is already a GPUArray, simply returns it. Otherwise, copies the array
    to the GPU and returns the reference.
    """
    if type(a) == gpuarray.GPUArray:
        return a
    else:
        return gpuarray.to_gpu(a)

def as_cpu(a):
    """ Returns a as a numpy array

    If a is already a numpy array, simply returns it. Otherwise, copies the
    array to the CPU and returns the reference.
    """
    if type(a) == gpuarray.GPUArray:
        return a.get()
    else:
        return a
