__all__ = ['CorrelationModes', 'ArrayReturnTypes', 'MAX_BLOCK_SIZE']

from enum import Enum

class CorrelationModes(Enum):
    FULL = 'full'
    VALID = 'valid'

class ArrayReturnTypes(Enum):
    CPU = 'cpu'
    GPU = 'gpu'

MAX_BLOCK_SIZE = 512
