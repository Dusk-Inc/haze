"""Tokens naming the transformer halves that move data in and out of the mesh."""
from enum import StrEnum


class TransformerTypes(StrEnum):
    """Which side of neuron IO a transformer sits on."""

    ENCODER = "encoder"
    DECODER = "decoder"
