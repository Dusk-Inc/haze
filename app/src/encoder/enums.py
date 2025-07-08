from enum import StrEnum

class EncoderType(StrEnum):
    NUMERIC = "numeric"
    TEXT = "text"
    VECTOR = "vector"
    IMAGE = "image"
    SERIES = "series"