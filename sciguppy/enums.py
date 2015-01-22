__all__ = ['CorrelationModes', 'MathModes', 'ArrayReturnTypes', 'MAX_BLOCK_SIZE', 'CUR_DIR', 'CACHE_DIR']

import os
from enum import Enum

class CorrelationModes(Enum):
    FULL = 'full'
    VALID = 'valid'

class MathModes(Enum):
    FAST = 'fast'
    ACC = 'accurate'

class ArrayReturnTypes(Enum):
    CPU = 'cpu'
    GPU = 'gpu'

MAX_BLOCK_SIZE = 512
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(CUR_DIR, 'cache')
