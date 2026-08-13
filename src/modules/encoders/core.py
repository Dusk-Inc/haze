"""Encoders turning an observation of one modality into sensor signal."""

from typing import Any

import torch
from torch import Tensor

from ...errors import SignalRangeError
from ...tokens import defaults


def ensureSignalRange(values: Tensor, key: str) -> Tensor:
    """Raises if any encoded value falls outside the band the mesh's thresholds assume.

    Clipping instead would hide a broken encoder behind plausible-looking output; see
    specs/encoding.md.
    """
    if values.numel() == 0:
        return values
    if not bool(torch.isfinite(values).all()):
        raise SignalRangeError(f"encoder {key!r} produced a non-finite value")
    low, high = float(values.min()), float(values.max())
    if low < defaults.SIGNAL_LOWER - 1e-6 or high > defaults.SIGNAL_UPPER + 1e-6:
        raise SignalRangeError(
            f"encoder {key!r} produced values in [{low:.4f}, {high:.4f}], outside the signal "
            f"band [{defaults.SIGNAL_LOWER}, {defaults.SIGNAL_UPPER}] the mesh's thresholds "
            "assume"
        )
    return values


SIGNAL_MID = (defaults.SIGNAL_LOWER + defaults.SIGNAL_UPPER) / 2.0
"""Where an input carrying no variation is placed.

The band's floor would be the obvious choice and is wrong: a value at the floor cannot pass
the edge gate under any strength, so a constant observation would propagate nothing at all and
be indistinguishable from not having been observed. The midpoint says "no information" without
also saying "no signal".
"""


def toSignalBand(values: Tensor, low: float | None = None, high: float | None = None) -> Tensor:
    """Rescales a tensor into the signal band, against given bounds or the tensor's own."""
    lo = values.min() if low is None else torch.as_tensor(low, dtype=values.dtype)
    hi = values.max() if high is None else torch.as_tensor(high, dtype=values.dtype)
    span = hi - lo
    if float(span) <= 0:
        return torch.full_like(values, SIGNAL_MID)
    normalized = ((values - lo) / span).clamp(0.0, 1.0)
    return defaults.SIGNAL_LOWER + normalized * (defaults.SIGNAL_UPPER - defaults.SIGNAL_LOWER)


class Encoder:
    """Base encoder: declares how many sensors it claims and produces signal for them."""

    def __init__(self, width: int) -> None:
        """Records how many sensors this encoder claims."""
        self.width = width
        self.key = ""

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this encoder."""
        return {"width": self.width}

    def encodeFeatures(self, value: Any) -> Tensor:
        """Maps one observation to a float vector of length `width` within the signal band."""
        raise NotImplementedError

    def applyReward(self, reward: float) -> None:
        """Consumes a reward. A no-op seam for future forward-only sensor-layer learning."""
        return None


class NumericEncoder(Encoder):
    """Encodes a row of numbers by rescaling it into the signal band."""

    def __init__(self, width: int | None = None, norm: str = "running") -> None:
        """Records the claimed width and whether scaling uses running statistics or each row.

        Running is the default because per-row scaling is scale-destroying: a row of all ones and
        a row of all zeros are both constant, so they encode identically and the mesh cannot tell
        them apart no matter how well it learns. Per-row is the prior engine's behavior and stays
        available as a deliberate choice; see specs/encoding.md.
        """
        super().__init__(width or 0)
        self.norm = norm
        self._low: float | None = None
        self._high: float | None = None

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this encoder."""
        return {"width": self.width, "norm": self.norm}

    def encodeFeatures(self, value: Any) -> Tensor:
        """Returns the row rescaled into the signal band, widening the claim if the row grew."""
        values = torch.as_tensor(value, dtype=torch.float32).flatten()
        if values.numel() == 0:
            return values
        self.width = max(self.width, int(values.numel()))

        if self.norm == "running":
            self._low = float(values.min()) if self._low is None else min(self._low, float(values.min()))
            self._high = (
                float(values.max()) if self._high is None else max(self._high, float(values.max()))
            )
            scaled = toSignalBand(values, self._low, self._high)
        else:
            scaled = toSignalBand(values)

        return ensureSignalRange(scaled, self.key)
