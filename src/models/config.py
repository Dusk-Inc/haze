"""Configuration models. The config drives allocation, so it is read before any tensor."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..tokens import LabelType, PortRole, defaults


class MeshCapacity(BaseModel):
    """Allocated size of each capacity block, which fixes every buffer's shape.

    Capacity is configuration rather than something inferred, because strict tensor loading
    requires shapes to agree and a variable-shape model can only satisfy that by allocating at
    the recorded capacity before loading. See specs/packaging.md.
    """

    model_config = ConfigDict(extra="forbid")

    sensors: int = Field(default=defaults.MIN_SENSOR_CAPACITY, ge=1)
    motors: int = Field(default=defaults.MIN_MOTOR_CAPACITY, ge=1)
    nexus: int = Field(default=defaults.DEFAULT_NEXUS_SIZE, ge=1)
    terminus: int = Field(default=defaults.DEFAULT_TERMINUS_SIZE, ge=1)
    edges: int = Field(default=defaults.MIN_EDGE_CAPACITY, ge=1)

    @property
    def neurons(self) -> int:
        """Returns the total neuron slab size across all four blocks."""
        return self.sensors + self.motors + self.nexus + self.terminus


class MeshCounts(BaseModel):
    """Live occupancy of each capacity block, always at or below the matching capacity."""

    model_config = ConfigDict(extra="forbid")

    sensors: int = Field(default=0, ge=0)
    motors: int = Field(default=0, ge=0)
    nexus: int = Field(default=0, ge=0)
    terminus: int = Field(default=0, ge=0)
    edges: int = Field(default=0, ge=0)


class HazeHyper(BaseModel):
    """Thresholds and rates governing propagation, learning, growth, and pruning.

    Field bounds replace the prior engine's hand-rolled validation ladder, which raised a bare
    `Exception("Value must be between 0.1 and 0.9")` naming neither the field nor its value.
    """

    model_config = ConfigDict(extra="forbid")

    signal_lower: float = Field(default=defaults.SIGNAL_LOWER, gt=0.0, lt=1.0)
    signal_upper: float = Field(default=defaults.SIGNAL_UPPER, gt=0.0, le=1.0)
    signal_threshold: float = Field(default=defaults.SIGNAL_THRESHOLD, gt=0.0, lt=1.0)
    neuron_firing_threshold: float = Field(
        default=defaults.NEURON_FIRING_THRESHOLD, gt=0.0
    )
    strength_lower: float = Field(default=defaults.STRENGTH_LOWER, gt=0.0, lt=1.0)
    strength_upper: float = Field(default=defaults.STRENGTH_UPPER, gt=0.0, le=1.0)
    strength_init_lower: float = Field(default=defaults.STRENGTH_INIT_LOWER, gt=0.0, lt=1.0)
    strength_init_upper: float = Field(default=defaults.STRENGTH_INIT_UPPER, gt=0.0, le=1.0)
    epsilon_start: float = Field(default=defaults.EPSILON_START, gt=0.0, le=1.0)
    epsilon_decay: float = Field(default=defaults.EPSILON_DECAY, gt=0.0, le=1.0)
    prune_threshold: float = Field(default=defaults.PRUNE_THRESHOLD, gt=0.0, lt=1.0)
    growth_rate: float = Field(default=defaults.GROWTH_RATE, gt=0.0, lt=1.0)
    growth_threshold: float = Field(default=defaults.GROWTH_THRESHOLD, gt=0.0, lt=1.0)
    audit_window: int = Field(default=defaults.AUDIT_WINDOW, ge=1)
    max_steps: int = Field(default=defaults.MAX_STEPS, ge=1)
    relearn_limit: int = Field(default=defaults.RELEARN_LIMIT, ge=0)
    fanout_min: int = Field(default=defaults.FANOUT_MIN, ge=1)
    fanout_max: int = Field(default=defaults.FANOUT_MAX, ge=1)
    sensor_fanout: int = Field(default=defaults.SENSOR_FANOUT, ge=1)
    motor_fanin: int = Field(default=defaults.MOTOR_FANIN, ge=1)
    inhibitory_ratio: float = Field(default=defaults.INHIBITORY_RATIO, ge=0.0, lt=1.0)
    """Share of new edges drawn inhibitory, so the mesh can express what an answer is not.

    With only positive attenuating strengths a mesh can excite but never veto, so no input can
    mean "not that answer". Zero restores the purely excitatory model. See specs/learning.md.
    """
    credit_assignment: bool = True
    """Whether an edge's update is weighted by its own contribution to the chosen answer.

    False restores the inherited uniform rule, which cannot make one motor beat another and is
    kept only so the difference can be measured. See specs/learning.md.
    """
    reward_baseline: bool = True
    """Whether the learning signal is measured against expected reward rather than confidence.

    False restores the inherited `reward - confidence` signal, which is not centred on anything
    the mesh is trying to beat and therefore pushes the incumbent answer regardless of whether it
    was right. Kept only so the difference can be measured. See specs/learning.md.
    """
    reward_baseline_rate: float = Field(
        default=defaults.REWARD_BASELINE_RATE, gt=0.0, le=1.0
    )
    """How fast the running reward baseline tracks recent reward. See specs/learning.md."""
    explore_rate: float = Field(default=defaults.EXPLORE_RATE, ge=0.0, lt=1.0)
    """Share of learning observations answered with a random label rather than the best one.

    Zero is a valid setting and restores a purely greedy readout, but it is not a safe one while
    `reward_baseline` is on: the two are one mechanism, and the baseline has no way out of a
    uniformly-wrong answer without variance to learn from. See specs/learning.md.
    """

    def calcStrengthFloor(self) -> float:
        """Returns the lowest strength an edge may hold, which inhibition puts below zero."""
        return -self.strength_upper if self.inhibitory_ratio > 0.0 else self.strength_lower

    def calcConductionFloor(self, hops: int = 1) -> float:
        """Returns the weakest uniform strength that still carries the band's best signal `hops` far.

        A path's arriving value is its running product of strengths scaled by their geometric mean,
        so a uniform strength `s` over `n` hops arrives at `signal * s**(n+1)` — one hop costs
        roughly the square of a strength, not the strength. Inverting that against
        `signal_threshold` gives the weakest edge that conducts at all, and it rises with depth.
        """
        return (self.signal_threshold / self.signal_upper) ** (1.0 / (max(hops, 1) + 1))

    def calcDeadBand(self, hops: int = 1) -> tuple[float, float] | None:
        """Returns the strength range in which an edge is alive, unprunable, and mute, if any.

        An edge under `prune_threshold` is removed and rewired, so it recovers. An edge over the
        conduction floor carries signal. Between the two it does neither: `ensureNoOrphans` cannot
        see it because it is structurally connected, and pruning cannot reach it. Learning that
        drives an edge into this range removes it from the mesh's behaviour permanently, because a
        mute edge fires no trace and a trace is what every update is gated on.

        Non-empty under the shipped defaults, where it spans (0.2, 0.527). See specs/propagation.md.
        """
        ceiling = min(self.calcConductionFloor(hops), self.strength_upper)
        return (self.prune_threshold, ceiling) if ceiling > self.prune_threshold else None

    @model_validator(mode="after")
    def ensureHyperCoherent(self) -> "HazeHyper":
        """Rejects hyperparameter combinations that make a whole mechanism unreachable."""
        if self.strength_lower >= self.strength_upper:
            raise ValueError("strength_lower must be below strength_upper")
        if self.strength_init_lower > self.strength_init_upper:
            raise ValueError("strength_init_lower must not exceed strength_init_upper")
        if not (
            self.strength_lower <= self.strength_init_lower
            and self.strength_init_upper <= self.strength_upper
        ):
            raise ValueError("the initial strength range must lie within the strength rails")
        if self.prune_threshold <= self.strength_lower:
            raise ValueError(
                "prune_threshold must exceed strength_lower, or learning can never drive an "
                "edge low enough to be pruned and pruning becomes dead code"
            )
        if self.prune_threshold >= self.strength_upper:
            raise ValueError(
                "prune_threshold must stay below strength_upper, or every edge prunes at once"
            )
        if self.fanout_min > self.fanout_max:
            raise ValueError("fanout_min must not exceed fanout_max")

        reachable = self.signal_lower * self.strength_init_upper * self.strength_init_upper
        if reachable <= self.signal_threshold:
            raise ValueError(
                f"a feature at the signal band's floor ({self.signal_lower}) attenuates to at "
                f"most {reachable:.4f} across one edge, which never clears signal_threshold "
                f"({self.signal_threshold}). The bottom of the band would be mute: a feature at "
                "its minimum would fire no edge at all and be indistinguishable from not having "
                "been observed. Raise signal_lower or lower signal_threshold."
            )
        if self.signal_lower >= self.signal_upper:
            raise ValueError("signal_lower must be below signal_upper")
        return self


class PortSpec(BaseModel):
    """How to reconstruct one encoder or decoder, without restating which neurons it owns.

    Neuron ownership lives in the owner tensor: an index range in a config is true only until
    the first free slot is reused. See specs/ports.md.
    """

    model_config = ConfigDict(extra="forbid")

    slot: int = Field(ge=0)
    role: PortRole
    key: str = Field(min_length=1)
    impl: str = Field(min_length=1)
    width: int | None = Field(default=None, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    bundle_weights: bool = False


class Stage(BaseModel):
    """One step of a chain: the encoders it feeds and the decoders it reads."""

    model_config = ConfigDict(extra="forbid")

    encoders: list[str] = Field(min_length=1)
    decoders: list[str] = Field(min_length=1)


class ChainSpec(BaseModel):
    """The ordered stages a prediction runs through, and how it decides to stop."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["static"] = "static"
    stages: list[Stage] = Field(default_factory=list)
    sequential: bool = False
    end_token: str = defaults.END_TOKEN
    max_stages: int = Field(default=16, ge=1)


class LabelEntry(BaseModel):
    """One decoder's labels and the motors that answer for them.

    Holds one of two bindings. In **one-hot** mode a label owns a motor and `motor_ids` runs
    parallel to `values`. In **code** mode a label owns a codeword, `codes` runs parallel to
    `values`, and `bit_motors` holds one `[on, off]` pair per bit — so the motor count is
    logarithmic in the label count rather than equal to it. See specs/decoding.md.

    `value_type` is recorded because decoders distinguish an integer label from its string
    form and JSON does not.
    """

    model_config = ConfigDict(extra="forbid")

    value_type: LabelType
    values: list[Any]
    motor_ids: list[int]
    active: list[bool]
    codes: list[list[int]] = Field(default_factory=list)
    """One codeword per label. Empty means one-hot, which is retained so the two are comparable."""
    bit_motors: list[int] = Field(default_factory=list)
    """The `[on, off]` motor pairs, flattened. Length is twice the code width."""

    @property
    def isCoded(self) -> bool:
        """Returns whether this decoder answers by codeword rather than by one motor per label."""
        return bool(self.codes)

    @property
    def width(self) -> int:
        """Returns how many bits a codeword holds, or zero in one-hot mode."""
        return len(self.codes[0]) if self.codes else 0

    @model_validator(mode="after")
    def ensureLabelsAligned(self) -> "LabelEntry":
        """Rejects a label table whose parallel lists disagree or whose motors collide."""
        if len(self.values) != len(self.active):
            raise ValueError("values and active must be the same length")

        if self.isCoded:
            if self.motor_ids:
                raise ValueError(
                    "a coded decoder binds no motor to a label; its motors are bit pairs"
                )
            if len(self.codes) != len(self.values):
                raise ValueError("codes and values must be the same length")
            widths = {len(code) for code in self.codes}
            if len(widths) != 1:
                raise ValueError("every codeword must be the same width")
            if len(self.bit_motors) != 2 * self.width:
                raise ValueError(
                    f"a {self.width}-bit codebook needs {2 * self.width} motors, "
                    f"got {len(self.bit_motors)}"
                )
            if len(set(self.bit_motors)) != len(self.bit_motors):
                raise ValueError("a motor appears in more than one bit pair")
            if len({tuple(code) for code in self.codes}) != len(self.codes):
                raise ValueError(
                    "two labels share a codeword, so they could never be told apart"
                )
            return self

        if len(self.values) != len(self.motor_ids):
            raise ValueError("values, motor_ids, and active must be the same length")
        if len(set(self.motor_ids)) != len(self.motor_ids):
            raise ValueError("a motor is bound to more than one label")
        return self


class HazeConfig(BaseModel):
    """The complete description of a Haze model, sufficient to allocate it before loading."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["haze"] = defaults.ARCHITECTURE
    haze_version: str = "0.2.0"
    format_version: int = defaults.FORMAT_VERSION
    dtype: Literal["float32", "float64"] = "float32"
    seed: int = defaults.DEFAULT_SEED

    capacity: MeshCapacity = Field(default_factory=MeshCapacity)
    counts: MeshCounts = Field(default_factory=MeshCounts)
    hyper: HazeHyper = Field(default_factory=HazeHyper)
    ports: list[PortSpec] = Field(default_factory=list)
    chain: ChainSpec = Field(default_factory=ChainSpec)

    nexus_size: int = Field(default=defaults.DEFAULT_NEXUS_SIZE, ge=0)
    terminus_size: int = Field(default=defaults.DEFAULT_TERMINUS_SIZE, ge=0)

    @model_validator(mode="after")
    def ensureConfigCoherent(self) -> "HazeConfig":
        """Rejects a config whose occupancy exceeds its capacity or whose port keys collide."""
        for block in ("sensors", "motors", "nexus", "terminus", "edges"):
            live = getattr(self.counts, block)
            room = getattr(self.capacity, block)
            if live > room:
                raise ValueError(
                    f"{block} count {live} exceeds capacity {room}; the checkpoint's config "
                    "cannot allocate a model large enough to hold its own tensors"
                )
        keys = [port.key for port in self.ports]
        if len(set(keys)) != len(keys):
            raise ValueError("two ports share a key")
        slots = [port.slot for port in self.ports]
        if len(set(slots)) != len(slots):
            raise ValueError("two ports share a slot")
        return self
