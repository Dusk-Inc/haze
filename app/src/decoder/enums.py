from enum import StrEnum

class DecoderType(StrEnum):
    ARGMAX = 'argmax'
    REGRESSOR = 'regressor'
    SOFTMAX = 'softmax'
    BINARY ='binary'
    VECTOR = 'vector'
    BITMASK = 'bitmask'
    TOP_K = 'top_k'
    SEQUENTIAL = 'sequential'
    SERIES = 'series'
