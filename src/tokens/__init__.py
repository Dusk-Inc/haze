"""Constants and enumerations used across Haze, in place of string literals."""

from . import defaults
from .neurons import (
    BLOCK_ORDER,
    INTERNEURON_KINDS,
    MESH_REGION_KINDS,
    UNOWNED,
    LabelType,
    MeshRegion,
    NeuronKind,
    PortRole,
)

__all__ = [
    "BLOCK_ORDER",
    "INTERNEURON_KINDS",
    "MESH_REGION_KINDS",
    "UNOWNED",
    "LabelType",
    "MeshRegion",
    "NeuronKind",
    "PortRole",
    "defaults",
]
