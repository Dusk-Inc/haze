"""Haze: a continuously learning, self-organizing neural mesh with forward-only learning."""

from .models import HazeConfig, HazeHyper, HazeOutput, MeshCapacity, Stage
from .modules import Auditor, Haze, MeshState, PortRegistry, Trainer, makeHaze

__version__ = "0.2.0"

__all__ = [
    "Auditor",
    "Haze",
    "HazeConfig",
    "HazeHyper",
    "HazeOutput",
    "MeshCapacity",
    "MeshState",
    "PortRegistry",
    "Stage",
    "Trainer",
    "__version__",
    "makeHaze",
]
