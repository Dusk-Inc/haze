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

EPSILON_START = 0.1
"""Learning rate a new edge begins with.

The inherited value was 0.7, which is rail-to-rail on a [0.1, 0.9] range in a single observation:
within a handful of steps every fired edge sat at the ceiling, both motors received numerically
identical activation, confidence fell to zero, and so did the update. The mesh stopped changing
at all. Chosen by measurement rather than inherited — 0.15 scores higher on copy and majority but
drives parity, a task at its own chance bound, to 0.23, which is a mesh confidently learning a
wrong rule rather than declining to learn. See specs/learning.md.
"""

EPSILON_DECAY = 0.9999
"""Multiplier applied to an edge's learning rate each time it is updated."""

REWARD_BASELINE_RATE = 0.02
"""How fast the running reward baseline tracks recent reward.

The baseline is what makes the learning signal an advantage rather than a raw reward, and its
rate is the only tuning it has. Too fast and it absorbs the very improvement it should be
measuring; too slow and it lags a mesh whose behaviour has changed. See specs/learning.md.
"""

EXPLORE_RATE = 0.05
"""Share of learning observations answered with a random label rather than the best one.

Inseparable from the reward baseline: a centred advantage learns from the difference between
outcome and expectation, so a greedy readout on a uniformly-wrong mesh produces no variance and
therefore no signal. Measured, zero exploration left one seed in three stuck at 0.00 on the
constant task forever. Applies only while learning; `eval()` always answers greedily.
"""

PRUNE_THRESHOLD = 0.2
"""An edge at or below this strength is removed. See specs/pruning.md."""

GROWTH_RATE = 0.1
"""Share of a population added when it grows, before error and uncertainty scale it.

Proportional rather than a fixed count, so growth stays meaningful as the mesh gets larger
instead of becoming a rounding error. The inherited value was 0.01 and its docstring described
a weighting no code implemented; this is the first version that both means something and is
read. See specs/growth.md.
"""

GROWTH_THRESHOLD = 0.6
"""Error rate above which growth is planned, once the audit windows are full.

Above chance rather than at it. The inherited 0.5 is exactly the error rate of an untrained mesh
on a two-label task, so growth fired on the first full window of every run — before the mesh had
any chance to learn — and restructuring a mesh mid-learning cost one seed its whole result, 0.95
down to 0.46. A threshold at chance cannot distinguish "too small to solve this" from "has not
learned it yet", which are the two things the trigger exists to tell apart.
"""

AUDIT_WINDOW = 100
"""Number of recent observations the auditor considers before it will plan growth.

Also the cadence: the trainer audits once per window, so this is how often the mesh may
restructure. The inherited value of 10 is too few observations to estimate an error rate from
and, worse, let growth fire every ten steps — measured, a 400-step run added 253 neurons and
pruned 5089 edges while ending at chance, because the mesh was rebuilt faster than it could
learn anything on it. It also gives an edge crossing zero between excitation and inhibition
time to get there before a prune pass reads its magnitude. See specs/growth.md.
"""

MAX_STEPS = 32
"""Safety cap on propagation hops. Termination normally comes from an empty wavefront."""

RELEARN_LIMIT = 0
"""Attempts the recovery loop makes to open a path when no signal reaches the motors.

Zero, because recovery was measured and is worse than doing nothing. Reverse learning was
designed against a symptom that was mostly a defect elsewhere: 70 of 91 apparent failures were
motors that had received signal and been net-inhibited, which the readout misread as silence.
Once that was fixed, genuine failures fell to 0.4% of observations, and every form of
intervention tried cost more than it saved — held-out copy accuracy 0.95 skipping the
observation, 0.74 strengthening every unfired edge, 0.66 strengthening only the wavefront.

The mechanism is kept and tested rather than deleted, because a mesh that truly cannot reach its
motors has no other way back, and this default is a measurement on one task family rather than a
proof. See specs/learning.md.
"""

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

SENSOR_FANOUT = 48
"""Nexus interneurons each sensor projects to.

Sampling rather than full bipartite wiring is what could give a feature a pathway of its own, but
it was measured and does not on its own raise what a readout can achieve: at a fan-out of 8 the
probe bound moved from 0.57 to 0.58 on copy and from 0.74 to 0.69 on majority, while edge
activation went bimodal — either a few percent, where signal never reaches the motors, or
near-total, where it reaches all of them equally. The default is therefore set high enough to
keep propagation stable, and left as a knob rather than a claim. See specs/growth.md.
"""

MOTOR_FANIN = 24
"""Terminus interneurons each motor draws from, sampled separately per motor.

Sampled per motor so different answers can read different evidence, which fully bipartite wiring
makes impossible. Kept wide for the same stability reason as SENSOR_FANOUT.
"""

INHIBITORY_RATIO = 0.2
"""Share of new edges drawn inhibitory.

Roughly the excitatory-to-inhibitory balance of cortex. Without inhibition a mesh can only ever
excite, so it cannot express that one input rules an answer out, and a readout built from
non-negative weights cannot subtract.
"""

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
