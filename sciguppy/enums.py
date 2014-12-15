__all__ = ['CorrelationModes', 'MathModes', 'ArrayReturnTypes', 'MAX_BLOCK_SIZE']

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
