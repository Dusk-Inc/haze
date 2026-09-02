"""Tokens naming the encoder strategies an encoder may be configured with."""
from enum import StrEnum


class EncoderType(StrEnum):
    """The kind of input data an encoder maps onto sensor neurons."""

    NUMERIC = "numeric"
    TEXT = "text"
    VECTOR = "vector"
    IMAGE = "image"
    SERIES = "series"
