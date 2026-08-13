"""Protocols constraining the encoders, decoders, policies, and stores Haze accepts."""

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from torch import Tensor

from ..models import HazeOutput, LabelEntry, Stage


@runtime_checkable
class IEncoder(Protocol):
    """Turns one observation of a single modality into sensor signal.

    The width is fixed and the output band is a hard contract; see specs/encoding.md.
    """

    key: str
    width: int

    def encodeFeatures(self, value: Any) -> Tensor:
        """Maps one observation to a float vector of length `width` within the signal band."""
        ...

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this encoder from a checkpoint."""
        ...

    def applyReward(self, reward: float) -> None:
        """Consumes a reward. A no-op seam for future forward-only sensor-layer learning."""
        ...


@runtime_checkable
class IDecoder(Protocol):
    """Turns the accumulated activation of its motors into an answer and a confidence."""

    key: str

    def decodeMotors(self, states: Tensor, labels: LabelEntry) -> Any:
        """Maps active motor activation to one answer."""
        ...

    def calcMotorConfidence(self, states: Tensor, labels: LabelEntry) -> float:
        """Returns how peaked the active motor activation is, in [0, 1]."""
        ...

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this decoder from a checkpoint."""
        ...


@runtime_checkable
class IChainPolicy(Protocol):
    """Decides which stage runs next, or that the chain is finished.

    The seam behind which a future policy could choose chain length and modality flow from
    context, without the surrounding code changing. See specs/ports.md.
    """

    def routeChainNext(self, output: HazeOutput, step: int) -> Stage | None:
        """Returns the next stage to run, or None to end the chain."""
        ...


@runtime_checkable
class ICheckpointer(Protocol):
    """Writes and locates model checkpoints during a run."""

    def saveHazeCheckpointFull(self, model: Any, path: Path) -> None:
        """Writes the complete model to path, atomically."""
        ...

    def saveHazeCheckpointDelta(self, model: Any, since: int, path: Path) -> None:
        """Writes only what changed since a step. Unimplemented; see specs/persistence.md."""
        ...

    def findHazeCheckpointLatest(self, root: Path) -> Path | None:
        """Returns the most recent complete checkpoint under root, or None."""
        ...


@runtime_checkable
class IGrowthPolicy(Protocol):
    """Decides whether and how much the mesh should grow."""

    def calcGrowthPlan(self, rewards: Sequence[float], confidences: Sequence[float]) -> Any:
        """Returns a growth plan from the recent reward and confidence windows."""
        ...
