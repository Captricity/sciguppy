from __future__ import absolute_import

__all__ = ['as_gpu', 'as_cpu', 'gpu_func']

import numpy
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from functools import wraps

import sciguppy.config as config
from .enums import ArrayReturnTypes

def as_gpu(a):
    """ Returns a as a GPUArray

    If a is already a GPUArray, simply returns it. Otherwise, copies the array
    to the GPU and returns the reference.
    """
    if isinstance(a, gpuarray.GPUArray):
        return a
    else:
        return gpuarray.to_gpu(a)

def as_cpu(a):
    """ Returns a as a numpy array

    If a is already a numpy array, simply returns it. Otherwise, copies the
    array to the CPU and returns the reference.
    """
    if isinstance(a, gpuarray.GPUArray):
        return a.get()
    else:
        return a

def gpu_func(f):
    """Helper decorator for converting input arrays into GPUArrays. Also
    implements the optional return type.

    Assumes that ALL input arrays are for the gpu. To avoid GPUArray
    conversion, use tuples or lists.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], (numpy.ndarray, numpy.generic)):
                args[i] = as_gpu(args[i])
        if 'return_type' in kwargs:
            return_type = kwargs.pop('return_type')
        else:
            return_type = config.DEFAULT_OUTPUT_TYPE

        out = f(*args, **kwargs)

        if isinstance(out, tuple) and return_type == ArrayReturnTypes.CPU:
            outs = []
            for item in out:
                if isinstance(item, gpuarray.GPUArray):
                    outs.append(as_cpu(item))
            out = tuple(outs)
        elif isinstance(out, gpuarray.GPUArray) and return_type == ArrayReturnTypes.CPU:
            out = as_cpu(out)
        return out
    return wrapper
