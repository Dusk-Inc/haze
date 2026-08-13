"""Domain errors raised by Haze."""

from .core import (
    CapacityExceededError,
    CheckpointCorruptError,
    HazeError,
    IncompatibleChainError,
    InvalidRewardError,
    LabelSpaceError,
    LearningDisabledError,
    PortAlreadyRegisteredError,
    PortNotFoundError,
    SignalDidNotReachMotorsError,
    SignalRangeError,
    UntrustedImplementationError,
)

__all__ = [
    "CapacityExceededError",
    "CheckpointCorruptError",
    "HazeError",
    "IncompatibleChainError",
    "InvalidRewardError",
    "LabelSpaceError",
    "LearningDisabledError",
    "PortAlreadyRegisteredError",
    "PortNotFoundError",
    "SignalDidNotReachMotorsError",
    "SignalRangeError",
    "UntrustedImplementationError",
]
