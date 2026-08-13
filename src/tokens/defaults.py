"""Default hyperparameters, capacities, and format identifiers."""

SIGNAL_LOWER = 0.4
"""Lower bound of the band every encoder must emit into.

Not a free choice: a feature at the floor attenuates by roughly the square of an edge strength
over one hop, so a floor below `SIGNAL_THRESHOLD / strength^2` cannot traverse a single edge.
At the inherited floor of 0.1 the bottom third of the band was mute — a feature at its minimum
fired nothing and was indistinguishable from not having been observed. HazeHyper refuses a
combination where that is true. See specs/encoding.md.
"""

SIGNAL_UPPER = 0.9
"""Upper bound of the band every encoder must emit into. See specs/encoding.md."""

SIGNAL_THRESHOLD = 0.25
"""Edge gate: a signal below this after attenuation does not traverse the edge.

Kept below what the band's floor can reach, so the gate sparsifies on attenuated downstream
signal rather than muting weak inputs at the sensors.
"""

NEURON_FIRING_THRESHOLD = 0.5
"""Neuron gate: accumulated arrivals below this are discarded rather than emitted.

Carried over from the prior engine and known to need recalibration upward, because
interneurons now genuinely accumulate fan-in where previously the buffer was cleared on
every call. Set by the Phase 3 sweep recorded in specs/propagation.md.
"""

STRENGTH_LOWER = 0.1
"""Lower rail an edge strength is clamped to. Must stay below PRUNE_THRESHOLD."""

STRENGTH_UPPER = 0.9
"""Upper rail an edge strength is clamped to."""

STRENGTH_INIT_LOWER = 0.4
"""Lower bound of the uniform range a new edge's strength is drawn from."""

STRENGTH_INIT_UPPER = 0.9
"""Upper bound of the uniform range a new edge's strength is drawn from."""

EPSILON_START = 0.7
"""Learning rate a new edge begins with."""

EPSILON_DECAY = 0.9999
"""Multiplier applied to an edge's learning rate each time it is updated."""

PRUNE_THRESHOLD = 0.2
"""An edge at or below this strength is removed. See specs/pruning.md."""

GROWTH_RATE = 0.01
"""Weighting between error rate and uncertainty when sizing a growth plan."""

GROWTH_THRESHOLD = 0.5
"""Error rate above which growth is planned, once the audit windows are full."""

AUDIT_WINDOW = 10
"""Number of recent observations the auditor considers before it will plan growth."""

MAX_STEPS = 32
"""Safety cap on propagation hops. Termination normally comes from an empty wavefront."""

RELEARN_LIMIT = 100
"""Attempts the recovery loop makes to open a path when no signal reaches the motors."""

DEFAULT_NEXUS_SIZE = 64
"""Interneurons the nexus mesh starts with."""

DEFAULT_TERMINUS_SIZE = 32
"""Interneurons the terminus mesh starts with."""

MIN_SENSOR_CAPACITY = 64
"""Floor on the sensor capacity block, so ordinary feature widths never force a relayout."""

MIN_MOTOR_CAPACITY = 32
"""Floor on the motor capacity block, so ordinary label counts never force a relayout."""

MIN_EDGE_CAPACITY = 1024
"""Floor on the edge arrays, so early growth writes into slack rather than reallocating."""

FANOUT_MIN = 2
"""Fewest outgoing edges a newly wired interneuron is given."""

FANOUT_MAX = 6
"""Most outgoing edges a newly wired interneuron is given."""

DEFAULT_SEED = 42
"""Seed for the model's own generator. Haze never touches the global RNG."""

END_TOKEN = "<!END!>"
"""Sentinel a sequential decoder emits to end its own output."""

CONFIG_FILE = "config.json"
LABELS_FILE = "labels.json"
WEIGHTS_FILE = "model.safetensors"
EVENTS_FILE = "events.jsonl"
MANIFEST_FILE = "manifest.json"

FORMAT_VERSION = 1
"""Checkpoint layout version, bumped when the payload's meaning changes."""

ARCHITECTURE = "haze"
"""Architecture identifier written into every checkpoint config."""

TRUSTED_IMPL_PREFIXES = ("haze.modules.encoders.", "haze.modules.decoders.")
"""Import paths a checkpoint may name without the caller opting in to remote code.

A config field naming an import path is arbitrary code execution on load; see
specs/packaging.md.
"""
