"""Tokens naming the decoder strategies a decoder may be configured with."""
from enum import StrEnum


class DecoderType(StrEnum):
    """The decoding strategy a decoder applies to neuron output."""

    ARGMAX = 'argmax'
    REGRESSOR = 'regressor'
    SOFTMAX = 'softmax'
    BINARY = 'binary'
    VECTOR = 'vector'
    BITMASK = 'bitmask'
    TOP_K = 'top_k'
    SEQUENTIAL = 'sequential'
    SERIES = 'series'
