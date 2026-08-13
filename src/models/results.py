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

    def calcRewardMeanLast(self, n: int) -> float:
        """Returns the mean reward over the last n steps, or over all steps if fewer ran."""
        if not self.rewards:
            return 0.0
        window = self.rewards[-n:] if n > 0 else self.rewards
        return sum(window) / len(window)
