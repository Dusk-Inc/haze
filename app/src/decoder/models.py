from dataclasses import dataclass
from .core import ArgMax, SoftMax, Series, Sequential, Binary, Bitmask, Vector

@dataclass
class Decoder:
    VECTOR: Vector
    ARGMAX: ArgMax
    SOFTMAX: SoftMax
    SERIES: Series
    SEQUENTIAL: Sequential
    BINARY: Binary
    BITMASK: Bitmask