"""Tokens naming the kinds of mesh a neuron may belong to."""
from enum import StrEnum


class MeshType(StrEnum):
    """The kind of mesh a neuron is placed in."""

    NEXUS = "nexus"
    TERMINUS = "terminus"
