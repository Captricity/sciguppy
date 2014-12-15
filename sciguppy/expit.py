__all__ = ['expit']

import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit

from .utils import gpu_func
from .enums import ArrayReturnTypes

# TODO: Handle x < 0 case better, like scipy does
@gpu_func
def expit(d_a):
    """Implements the expit function (aka sigmoid)

    expit(x) = 1 / (1 + exp(-x))
    """
    d_out = 1 / (1 + cumath.exp(-d_a));
    return d_out
