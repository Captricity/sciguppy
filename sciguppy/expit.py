__all__ = ['expit']

import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit

from .utils import as_gpu, as_cpu
from .enums import ArrayReturnTypes

# TODO: Handle x < 0 case better, like scipy does
def expit(a, return_type=ArrayReturnTypes.CPU):
    """Implements the expit function (aka sigmoid)

    expit(x) = 1 / (1 + exp(-x))
    """
    d_a = as_gpu(a)
    d_out = 1 / (1 + cumath.exp(-d_a));
    if return_type == ArrayReturnTypes.CPU:
        return as_cpu(d_out)
    return d_out
