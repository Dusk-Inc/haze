"""Decoders turning motor activation into an answer and a confidence."""

from typing import Any

import torch
from torch import Tensor

from ...errors import SignalDidNotReachMotorsError
from ...functions.learning import calcConfidenceEntropy
from ...models import LabelEntry


class Decoder:
    """Base decoder: holds a label set and reads only the motors its active labels own."""

    def __init__(self, labels: list[Any] | None = None) -> None:
        """Records the labels this decoder claims motors for."""
        self.labels = list(labels) if labels else []
        self.key = ""

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this decoder."""
        return {"labels": list(self.labels)}

    def findActiveValues(self, entry: LabelEntry) -> list[Any]:
        """Returns the labels currently active, in table order."""
        return [v for v, on in zip(entry.values, entry.active) if on]

    def ensureSignalReached(self, states: Tensor) -> None:
        """Raises when no active motor received signal, so there is no answer to decode.

        Distinct from answering wrongly: it means the mesh has not connected its sensors to its
        motors, which is what drives reverse learning.
        """
        if states.numel() == 0 or float(states.sum()) <= 0:
            raise SignalDidNotReachMotorsError(
                f"no active motor of decoder {self.key!r} received signal; the mesh has not "
                "connected its sensors to its motors"
            )

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> Any:
        """Maps active motor activation to one answer."""
        raise NotImplementedError

    def calcMotorConfidence(self, states: Tensor, entry: LabelEntry) -> float:
        """Returns how peaked the active motor activation is.

        Computed over the same motors used to decode. The prior engine took entropy over every
        motor while decoding from the active ones, deflating confidence whenever the label space
        narrowed — and confidence feeds both the learning delta and the growth trigger.
        """
        return calcConfidenceEntropy(states)


class ArgMax(Decoder):
    """Answers with the label whose motor is most activated."""

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> Any:
        """Returns the active label holding the largest activation."""
        self.ensureSignalReached(states)
        return self.findActiveValues(entry)[int(torch.argmax(states))]


class SoftMax(Decoder):
    """Answers by sampling the label distribution implied by motor activation."""

    def __init__(self, labels: list[Any] | None = None, temperature: float = 1.0) -> None:
        """Records the labels and how sharply activation is turned into a distribution."""
        super().__init__(labels)
        self.temperature = temperature

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this decoder."""
        return {"labels": list(self.labels), "temperature": self.temperature}

    def calcProbabilities(self, states: Tensor) -> Tensor:
        """Returns the label distribution, computed so that any activation magnitude is safe.

        Motor activation is an unbounded sum, and a naive exponential overflows on it; the prior
        engine raised OverflowError here once a run went on long enough.
        """
        return torch.softmax(states.to(torch.float64) / max(self.temperature, 1e-6), dim=-1)

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> Any:
        """Returns the most probable active label under the softened distribution."""
        self.ensureSignalReached(states)
        probabilities = self.calcProbabilities(states)
        return self.findActiveValues(entry)[int(torch.argmax(probabilities))]

    def calcMotorConfidence(self, states: Tensor, entry: LabelEntry) -> float:
        """Returns the peakedness of the softened distribution rather than of raw activation."""
        if states.numel() == 0 or float(states.sum()) <= 0:
            return 0.0
        return calcConfidenceEntropy(self.calcProbabilities(states))


class Regressor(Decoder):
    """Answers with a number: the activation-weighted centre of its numeric labels."""

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> float:
        """Returns the centre of mass of the active labels under their activation."""
        self.ensureSignalReached(states)
        values = torch.tensor(
            [float(v) for v in self.findActiveValues(entry)], dtype=torch.float64
        )
        weights = states.to(torch.float64)
        return float((values * weights).sum() / weights.sum())


class TopK(Decoder):
    """Answers with the k most activated labels, most activated first."""

    def __init__(self, labels: list[Any] | None = None, k: int = 3) -> None:
        """Records the labels and how many of them an answer holds."""
        super().__init__(labels)
        self.k = k

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this decoder."""
        return {"labels": list(self.labels), "k": self.k}

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> list[Any]:
        """Returns the k most activated active labels."""
        self.ensureSignalReached(states)
        values = self.findActiveValues(entry)
        order = torch.argsort(states, descending=True)[: min(self.k, len(values))]
        return [values[int(i)] for i in order]


class Bitmask(Decoder):
    """Answers with every label whose motor crossed a share of the strongest one."""

    def __init__(self, labels: list[Any] | None = None, ratio: float = 0.5) -> None:
        """Records the labels and the share of the peak a label must reach to be included."""
        super().__init__(labels)
        self.ratio = ratio

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this decoder."""
        return {"labels": list(self.labels), "ratio": self.ratio}

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> list[Any]:
        """Returns every active label activated to at least `ratio` of the peak."""
        self.ensureSignalReached(states)
        values = self.findActiveValues(entry)
        cutoff = float(states.max()) * self.ratio
        return [v for v, s in zip(values, states.tolist()) if s >= cutoff]


class Vector(Decoder):
    """Answers with the activation itself, normalized into a distribution."""

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> list[float]:
        """Returns the active motors' activation as a distribution over the active labels."""
        self.ensureSignalReached(states)
        weights = states.to(torch.float64)
        return (weights / weights.sum()).tolist()


class Binary(Decoder):
    """Answers with one of exactly two labels, by which motor leads."""

    def decodeMotors(self, states: Tensor, entry: LabelEntry) -> Any:
        """Returns whichever of the two active labels is more activated."""
        self.ensureSignalReached(states)
        values = self.findActiveValues(entry)
        return values[int(torch.argmax(states))]
