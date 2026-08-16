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


class FiringProfile(BaseModel):
    """How much of the mesh one observation used, and how much of that was specific to it.

    The measurement behind the claim that every input uses the whole mesh. Share alone cannot
    settle it: a mesh could fire few edges and still fire the *same* few for every input, which is
    sparsity with no capacity allocated and is what both fan-out experiments measured without being
    able to say so. `conditionality` is the quantity that separates the two, and it is what a
    change aiming at per-label capacity has to move. See specs/propagation.md.
    """

    model_config = ConfigDict(extra="forbid")

    share: float = 0.0
    """Share of live edges that carried signal on an average observation."""

    hops: float = 0.0
    """Mean hops a propagation ran before no edge passed its gate."""

    neurons: float = 0.0
    """Mean count of distinct interneurons that emitted signal."""

    within_label_overlap: float = 0.0
    """Mean Jaccard of the fired-edge sets of two observations sharing a label."""

    between_label_overlap: float = 0.0
    """Mean Jaccard of the fired-edge sets of two observations with different labels."""

    @property
    def conditionality(self) -> float:
        """Returns how much more of the mesh two same-label inputs share than two different ones.

        Zero means the edges an input fires say nothing about its label, so no capacity is
        allocated per label and every label's learning writes over every other's. Positive means
        the mesh routes different inputs through different structure, which is the property
        input-conditional firing exists to create.

        Measured at 0.022-0.028 on a fresh mesh over 3 seeds, against a share of 0.87-0.89 — so
        nearly nine edges in ten fire for every input and almost none of that is specific to the
        input. This is the quantity the two fan-out sparsity experiments lacked, which is why they
        could report that sparsity did not raise the bound without being able to say why.
        """
        return self.within_label_overlap - self.between_label_overlap


class ArrivalProfile(BaseModel):
    """What arrives at a neuron, and how much of that is explained by its wiring rather than input.

    The instrument for the signal economy. A mesh whose arriving value tracks fan-in is one where
    a neuron's influence is set by how well connected it happens to be, so ranking arrivals selects
    topology rather than relevance — measured at +0.637 on a fresh mesh, which is why top-k over
    the shipped economy would pick the same clique for every input. See specs/propagation.md.
    """

    model_config = ConfigDict(extra="forbid")

    fanin_correlation: float = 0.0
    """Correlation between an interneuron's in-degree and its mean arriving magnitude.

    Measured at +0.69 to +0.78 on a fresh mesh over 3 seeds. A top-k over arrivals at that
    correlation ranks how well connected a neuron happens to be, which is fixed for the mesh, so
    the same neurons would win for every input.
    """

    arrival_cv: float = 0.0
    """Coefficient of variation of mean arriving magnitude across interneurons.

    Measured at 0.95-1.03 over 3 seeds: the spread across neurons is as large as the mean, so
    arrivals are not comparable between neurons and nothing can be ranked meaningfully.
    """

    depth_gain: float = 0.0
    """Peak median frontier magnitude over the first hop's, across a propagation.

    Peak over first rather than a mean of per-hop ratios, because a pass both climbs and then dies
    out as the fire-once guard is exhausted, and averaging the two phases together reports 1.27 for
    a mesh whose frontier tripled. Above one means the mesh amplifies with depth rather than
    attenuating: measured at 3.1-5.2 over 3 seeds against a signal band topping out at 0.9.

    This is the reading that corrects specs/propagation.md, which states a path's value "only ever
    multiplies by strengths below one". True per edge, false per neuron: with a mean fan-in of 10
    and a mean strength of 0.7, a neuron's summed arrival gains where each of its edges lost.
    """

    median_arrival: float = 0.0
    """Median arriving magnitude across interneurons.

    Measured at 4.6-6.5 over 3 seeds against a `neuron_firing_threshold` of 0.5, so the neuron gate
    refuses nothing. Propagation ends by exhausting the fire-once guard rather than by any gate,
    which is why nine edges in ten fire.
    """


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
