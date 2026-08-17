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

Carried over from the prior engine, and it does not merely need recalibrating upward — **it never
binds at all.** Median arrival across interneurons measures 4.6-6.5 over 3 seeds, nine to thirteen
times this value, so no neuron is ever refused and propagation ends by exhausting the fire-once
guard instead. That is the mechanism behind nine edges in ten firing on every observation, and it
is why the gate could be called "far too permissive" for a version of this file and still understate
the problem.

The value turns out to be right and what it judged was wrong. Under `SIGNAL_ECONOMY` with
`GAIN_CONTROL`, median arrival is 0.511 over 8 seeds — inside the signal band and sitting on this
threshold — so it begins to select without being retuned. See specs/propagation.md.
"""

STRENGTH_LOWER = 0.1
"""Lower rail an edge strength is clamped to. Must stay below PRUNE_THRESHOLD."""

STRENGTH_UPPER = 0.9
"""Upper rail an edge strength is clamped to."""

STRENGTH_INIT_LOWER = 0.55
"""Lower bound of the uniform range a new edge's strength is drawn from.

Held above `HazeHyper.calcConductionFloor()`, which is 0.527 under these defaults, so no edge is
born unable to pass signal. The inherited 0.4 put the bottom quarter of the initial range inside
the dead band: roughly a quarter of every fresh mesh's edges were alive, unprunable, and mute
before a single observation, and since a mute edge fires no trace and every update is gated on a
trace, nothing could ever recover them. Measured with `conductance_healing`, this raised the share
of seeds that learn anything on a three-label task from 13/40 to 23/40 and coverage from 0.59 to
0.73. See specs/propagation.md.
"""

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
"""Multiplier applied to an edge's learning rate each time it is updated.

Outcome-blind and one-directional: an edge that has been consistently right and one that has been
consistently wrong cool at the same rate, and neither ever warms again. It is also inert at the
scales the model trains at — a half-life of 6,931 updates leaves an edge holding 86% of its
starting rate after a 1,500-step run, and it only updates on steps where it fires. Used when
`crystallize` is off. See EPSILON_COOL for the mechanism that replaces it.
"""

EPSILON_COOL = 0.99
"""Multiplier on an edge's learning rate when the outcome beat expectation.

An edge that keeps being part of good answers becomes rigid, so a single bad result cannot tear
down a record it took hundreds of steps to build. That hysteresis is the point: the conduction
collapse in specs/propagation.md is driven by sustained negative advantage pushing established
edges under the signal gate, and an edge that has earned its place should barely move under it.
"""

EPSILON_WARM = 1.01
"""Multiplier on an edge's learning rate when the outcome fell short of expectation.

The half the inherited decay never had. Cooling with no path back is a second absorbing state —
the same shape of bug as an edge that has fallen mute — and it is what leaves a mesh unable to
relearn when its task changes. Warming is per-event and gradual, so a crystallized section takes
about as many bad outcomes to soften as it took good ones to set.

Paired with EPSILON_COOL this puts the drift's zero at `log(warm) / (log(warm) - log(cool))`,
which for 0.99 and 1.01 is 0.4975 — an edge cools while it is right slightly more often than half
the time, and warms when it is not. See specs/learning.md.
"""

EPSILON_FLOOR = 0.005
"""Lower bound on an edge's learning rate, below which crystallization would be permanent.

An edge at zero learning rate can never change again whatever happens to it, which is precisely
the absorbing state this whole mechanism exists to avoid. The floor keeps a fully crystallized
edge slow rather than frozen.
"""

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

Sampling rather than full bipartite wiring is what could give a feature a pathway of its own. It
was measured twice and appeared not to raise what a readout can achieve: at a fan-out of 8 the
probe bound moved from 0.57 to 0.58 on copy and from 0.74 to 0.69 on majority, while edge
activation went bimodal — either a few percent, where signal never reaches the motors, or
near-total, where it reaches all of them equally.

**Both measurements were taken on tasks that could not show it.** `copy` and `majority` are
saturated at initialization, which is why growth could never be demonstrated on them either. On
`first-set` capped at 3 labels — the one task whose ceiling moves with mesh size — a fan-out of 8
beats this default on **49 of 72 paired seeds, 3.1σ**, median bound 0.736 against 0.667.

**The trained sweep was run and it reversed the reading.** 24 paired seeds at each of k=2,3,4,6,9,
fan-out 8 loses at every one by 2.4σ to 4.5σ and never wins more than 6 of 24 seeds. It costs
0.30-0.55 of conduction reach, which is roughly 4× the bound it gains, so this default stays at 48.
The probe cannot see that cost because it never trains, so it never drives an edge under the gate.

What the sweep establishes is why 48 is right: it is redundancy insurance against silence being
absorbing, not a tuning of what the mesh can represent. Fewer edges per sensor leaves fewer
alternative routes when learning mutes some. Under `signal_economy`, where an edge cannot go mute,
fan-out 8 holds reach at 1.00 and lift stays negative anyway. See specs/propagation.md.
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

HEAL_MARGIN = 1.15
"""How far above the conduction floor `applyConductanceHealing` lifts a mute neuron.

Exactly at the floor an edge carries only the top of the signal band and only across a single hop,
so a healed neuron would fall mute again on the next unlucky observation and healing would become a
treadmill. The margin is small because lifting further overwrites more of what learning decided:
the scaling preserves an edge's rank among its siblings whatever the factor, but a large factor
pushes the whole fan toward the upper rail where differences compress.
"""


SIGNAL_ECONOMY = False
"""Whether a strength is read as a share of its neuron's output rather than as an absolute.

Off until measured, which is the house pattern. It changes three things at once: an edge carries a
share of what its source emits, a neuron's arrival is divided by its static incoming budget, and
the geometric-mean path correction is dropped because it corrects a decay the mesh no longer
applies.

The reading it answers: interneuron fan-in is mean 10.06 (range 4-66) against a mean |strength| of
0.715, so a neuron gains where each of its edges lost, and arrival correlates +0.69 to +0.78 with
in-degree over 3 seeds. A neuron's influence is therefore set by how well connected it happens to
be rather than by what its inputs said, and nothing downstream can rank arrivals meaningfully.

**It does what it was built for and does not earn a default.** Conduction reach holds at 1.00
through 1,500 steps at both three and nine labels, against 0.64 and 0.78 shipped — the absorbing
state of ROADMAP.md's gap 1 removed rather than mitigated. Accuracy falls: lift +0.118 to -0.077 at
three labels and +0.010 to -0.356 at nine, seeds learning 3/8 to 1/8 and 2/8 to 0/8. The erosion it
removes was also differentiating the mesh, badly but not randomly, and nothing replaces that.
Kept reachable and comparable so it can be re-tested against a better learning rule rather than
re-derived. See specs/propagation.md.
"""

OUT_BUDGET = 1.0
"""Total magnitude a neuron's outgoing edges sum to under the economy.

Governs competition among siblings, **not** the level of the signal — the outgoing and incoming
scalings cancel in level, which is set by the incoming budget alone. What the budget buys is that
raising one edge lowers every sibling's share, so learning becomes locally zero-sum, and that an
edge can never be mute: a share is a ratio, so driving every out-edge of a neuron to
`STRENGTH_LOWER` leaves the shares unchanged. That is the absorbing state of ROADMAP.md's gap 1
closed by construction rather than mitigated.
"""

EDGE_SIGNAL_FLOOR = 0.01
"""Magnitude below which an edge carries nothing under the economy.

A numerical floor and explicitly not a selection device. Under the shipped economy the edge gate
doubles as the sparsifier; under this one selection belongs to the neuron, where it can be ranked
rather than thresholded, so this only stops denormal values propagating.
"""

FIRING_FRACTION = 0.0
"""Share of each interneuron population allowed to emit per hop. Zero disables the ranking.

Ranked rather than thresholded because a hard threshold on arrival was measured to go bimodal —
raising strengths admits more edges, which accumulates more, which admits more, with no stable
middle. Scaling every strength cannot change how many neurons a rank admits.

Requires `SIGNAL_ECONOMY`: ranking arrivals that correlate +0.7 with in-degree selects the
best-connected neurons and selects the same ones for every input, which is sparsity with no
capacity allocated and is what both fan-out experiments measured.

**Measured, and it does not earn a default.** Over the economy it takes the firing share from 0.894
to 0.333 while conduction reach holds at 0.88-0.97 — sparsity stops costing conduction, which no
previous attempt achieved. But conditionality only doubles, +0.021 to +0.050, and does not rise as
the mesh fires less (0.039, 0.030, 0.037, 0.050, 0.042 across fractions 0.05 to 0.60). The winner
set is largely the same one whatever the input. A rank can only select among distinctions the
representation already carries, and it carries almost none. See specs/propagation.md.
"""

GAIN_CONTROL = False
"""Whether a hop's emitted wavefront is rescaled toward the band after ranking.

Silencing part of a wavefront lowers what the next hop receives, because the incoming budget it
divides by is static and does not shrink with the survivors. Ordering and level are separate jobs:
the rank decides which neurons emit and one scalar per hop decides how loud. A single scalar
cannot reorder anything, so it cannot interfere with the selection it follows.
"""

GAIN_TARGET = 0.65
"""Mean magnitude a rescaled wavefront is aimed at, at the middle of the signal band."""

GAIN_CEILING = 4.0
"""Most the gain may multiply a wavefront by.

Bounds the failure mode where a nearly-dead pass is amplified into a confident answer. Its
signature is coverage high and lift negative, which is worse than going silent because a silence is
visible in `calcConductionReach` and a confident wrong answer is not.
"""

PRUNE_SHARE = 0.05
"""Share of its neuron's output below which an edge is pruned under the economy.

Relative because the absolute `PRUNE_THRESHOLD` names the wrong thing once strengths are shares: an
edge at 0.1 whose siblings are all 0.1 carries a full share of its neuron's output and is doing its
job, yet an absolute threshold would delete it.
"""
