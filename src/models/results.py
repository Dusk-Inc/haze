"""Result models returned by prediction, learning, growth, and pruning."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HazeOutput(BaseModel):
    """What one prediction produced, keyed by decoder port key."""

    model_config = ConfigDict(extra="forbid")

    predictions: dict[str, list[Any]] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    hops: int = 0
    reached: bool = False
    stages: int = 0

    def __getitem__(self, key: str) -> list[Any]:
        """Returns one decoder's predictions, so an output reads like the mapping it replaces."""
        return self.predictions[key]


class LearnResult(BaseModel):
    """What one learning step changed."""

    model_config = ConfigDict(extra="forbid")

    edges_updated: int = 0
    reward: float = 0.0
    confidence: float = 0.0
    reverse: bool = False
    mean_delta: float = 0.0
    healed: int = 0
    """Neurons whose outgoing edges were scaled back to conducting after the update.

    Zero on every step unless `conductance_healing` is on. A run where this stays high is a mesh
    being held open against learning that keeps pushing it shut, which is worth seeing rather than
    silently absorbing. See specs/propagation.md.
    """


class ScoreProfile(BaseModel):
    """How well a mesh answered a task, separated into how much it answered and how well.

    Raw accuracy alone is not interpretable, because a mesh that declines to answer and one that
    answers wrongly score identically — and a trained mesh does both. Conditional accuracy alone is
    not interpretable either, because the answered set is chosen by the mesh: one that keeps only
    majority-label inputs scores near 1.0 by always guessing that label. Only `lift`, which measures
    conditional accuracy against the majority share *of the answered set*, says whether the mesh
    discriminated at all. Both confusions were live on this branch and each inflated a reported
    result before being caught. See specs/learning.md.
    """

    model_config = ConfigDict(extra="forbid")

    coverage: float = 0.0
    """Share of inputs the mesh answered rather than going silent on."""

    accuracy: float = 0.0
    """Share of all inputs answered correctly, counting a silence as wrong."""

    conditional: float = 0.0
    """Share of the *answered* inputs answered correctly."""

    baseline: float = 0.0
    """Majority label's share of the answered inputs — what guessing that label would score."""

    @property
    def lift(self) -> float:
        """Returns how far conditional accuracy beat guessing the answered set's majority label."""
        return self.conditional - self.baseline


class GrowthPlan(BaseModel):
    """How many interneurons to add to each mesh, and why."""

    model_config = ConfigDict(extra="forbid")

    nexus: int = Field(default=0, ge=0)
    terminus: int = Field(default=0, ge=0)
    error_rate: float = 0.0
    confidence_rate: float = 0.0
    triggered: bool = False

    @property
    def isEmpty(self) -> bool:
        """Returns whether the plan would add nothing."""
        return self.nexus == 0 and self.terminus == 0


class PruneReport(BaseModel):
    """What one pruning pass removed."""

    model_config = ConfigDict(extra="forbid")

    edges_removed: int = 0
    edges_remaining: int = 0
    neurons_rewired: int = 0


class AuditResult(BaseModel):
    """The auditor's read of recent performance."""

    model_config = ConfigDict(extra="forbid")

    error_rate: float = 0.0
    confidence_rate: float = 0.0
    window_full: bool = False


class TrainReport(BaseModel):
    """Summary of a training run."""

    model_config = ConfigDict(extra="forbid")

    steps: int = 0
    rewards: list[float] = Field(default_factory=list)
    grew: int = 0
    pruned: int = 0
    lost: int = 0
    """Observations the mesh could not answer at all, so they carry no reward.

    Reported rather than folded into the reward as a zero, which would read as "answered
    wrongly" and quietly understate accuracy by however often the mesh went silent.
    """

    def calcRewardMeanLast(self, n: int) -> float:
        """Returns the mean reward over the last n steps, or over all steps if fewer ran."""
        if not self.rewards:
            return 0.0
        window = self.rewards[-n:] if n > 0 else self.rewards
        return sum(window) / len(window)
