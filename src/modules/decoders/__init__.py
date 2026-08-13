"""Decoders turning motor activation into an answer and a confidence."""

from .core import ArgMax, Binary, Bitmask, CodeBook, Decoder, Regressor, SoftMax, TopK, Vector

__all__ = [
    "ArgMax",
    "Binary",
    "Bitmask",
    "CodeBook",
    "Decoder",
    "Regressor",
    "SoftMax",
    "TopK",
    "Vector",
]
