"""Haze: a continuously learning, self-organizing neural mesh with forward-only learning."""

from .models import HazeConfig, HazeHyper, HazeOutput, MeshCapacity, Stage
from .modules import Haze, MeshState, PortRegistry, makeHaze

__version__ = "0.2.0"

__all__ = [
    "Haze",
    "HazeConfig",
    "HazeHyper",
    "HazeOutput",
    "MeshCapacity",
    "MeshState",
    "PortRegistry",
    "Stage",
    "__version__",
    "makeHaze",
]
