# Learning

How edge strengths change in response to a reward.

## Description

Haze does not use backpropagation. It uses a forward-only rule: propagation records which edges
carried signal, and after an answer is scored, exactly those edges are moved toward or away from
the reward. There is no gradient, no loss surface, and no optimizer. An edge's own learning rate
(`epsilon`) decays each time it is updated, so early experience moves an edge more than late
experience.

## Policies

### The learning signal is the gap between reward and confidence

**Given** a prediction scored with reward `r`, made with confidence `c`
**When** learning is applied
**Then** every edge in the trace moves by `epsilon · (r − c)`, and its `epsilon` is multiplied by
`epsilon_decay`.

The sign follows directly: a confident wrong answer is punished hardest, a hesitant right answer is
reinforced hardest, and a confident right answer barely moves — the network stops adjusting what it
already reliably knows.

### Strength is bounded

**Given** an edge whose update would carry it outside `[strength_lower, strength_upper]`
**When** the update is applied
**Then** it is clamped to that interval.

Bounding is load-bearing in both directions: an upper bound prevents any single path dominating,
and the lower bound must stay below the pruning trigger or pruning becomes unreachable dead code.

### Only edges that fired are learned

**Given** an observation in which some edges carried signal and others did not
**When** learning is applied in the forward direction
**Then** only edges present in the fired trace are updated.

The trace is the `fired` tensor from propagation, reduced across lanes. Edges belonging to dead
slack slots are excluded by the alive mask — the prior engine's failure to do this meant orphaned
slots from pruned edges were reverse-learned on every reverse pass, forever.

### Reverse learning inverts the mask

**Given** a prediction in which no signal reached the motors
**When** the engine attempts recovery
**Then** it applies the update to the complement of the fired trace, strengthening the edges that
did *not* carry signal, so that a mesh which has failed to connect its sensors to its motors can
open a path.

### Learning is disabled in eval mode

**Given** a model in `eval()` mode
**When** `learn()` is called
**Then** it raises `LearningDisabledError` rather than silently doing nothing, and no fired trace
is recorded during propagation at all.

### A non-finite reward is refused

**Given** a reward that is NaN or infinite
**When** learning is applied
**Then** it raises `InvalidRewardError`.

Without this the clamp propagates NaN into every strength in the active mask, silently destroying
the network with no error and no observable symptom until predictions become nonsense.

### Batched learning is mini-batch, and is a deliberate change

**Given** a batch of B > 1 observations learned together
**When** the update is applied
**Then** weights are frozen across the batch and the per-edge delta is averaged over the rows that
fired that edge, so the update magnitude matches a single online update.

This is exactly the SGD → mini-batch SGD change and is a semantic difference from the prior
engine, which updated after every row. `B = 1` recovers exact online semantics and is the default.
`epsilon_decay ** count` reproduces B sequential decays exactly, since decay is multiplicative and
mask-independent. One inexactness remains at large B: sequential updates clamp after each step
while the batched form clamps once, so edges whose trajectory would have hit a rail mid-batch
differ. Keep B modest.

### The update is weighted by each edge's own responsibility

**Given** an observation that produced an answer
**When** learning is applied
**Then** each fired edge moves by three factors multiplied together — how much signal it carried,
how much credit reaches the neuron it fed, and how far the reward diverged from the confidence —
rather than by one shared scalar.

Credit is seeded at the motor the decoder actually chose, because learning is told a reward and
never the correct label. A rewarded choice reinforces the path that produced it and weakens its
rivals; an unrewarded one does the reverse, which is what lets a rival overtake it. Rival seeds
share the opposite sign so they sum to zero and total mesh strength is not driven one way.

The seeds are then spread backward along the fired edges, attenuated by the same strengths the
forward pass used. This is not a gradient and there is no chain rule — it is one extra scatter per
hop over the same edge list — but it is what makes an edge feeding the winning motor move opposite
to an edge feeding a losing one. Eligibility is kept per lane and weighted by that lane's share of
the chosen motor's activation before being pooled, since a lane is one input feature and pooling
first would discard what distinguishes one feature from another.

Setting `credit_assignment` false restores the uniform rule, which is retained only so the
difference stays measurable.

## What credit assignment fixed, and what it did not

The inherited rule moved **every edge in the trace by the same scalar**. When two motors compete
for one answer, both of their inbound edges are in the trace and both receive the same delta with
the same sign, so nothing distinguishes the motor that should have won from the one that did.
Whichever motor led at initialization kept leading, and the answer was effectively fixed by the
random wiring.

Measured, on the constant task — always answer the same label, chance 0.5, ceiling 1.0, the floor
any working learner must reach:

| rule | `epsilon_start` | reward over the last 100 rows, seeds 1/2/3 |
|---|---|---|
| uniform | 0.7 (inherited default) | 0.00, 0.00, 0.00 — saturated at the ceiling |
| uniform | 0.05 | 0.00, 0.00, 0.00 |
| uniform | 0.005 | 0.00, 0.06, 0.00 |
| **credited** | **0.05** | **1.00, 1.00, 0.99** |

Two distinct effects were separated here:

1. **Saturation.** At the inherited `epsilon_start = 0.7`, one observation can move an edge ±0.7 on
   a [0.1, 0.9] range — rail to rail in a single step. Within a handful of observations every
   fired edge sits at the ceiling, both motors receive **numerically identical** activation,
   confidence falls to exactly zero, and therefore the delta falls to exactly zero. The mesh stops
   changing at all. This is a calibration fault and a smaller rate avoids it.
2. **No credit assignment.** With the rate reduced far enough that strengths stay spread, the
   reward *still* does not move off chance. This is not calibration. A uniform delta over the
   fired set cannot create the asymmetry that choosing between labels requires.

Credited learning resolves the second effect for a task that needs a **bias**: the constant task
goes from 0.00 to 0.99–1.00 across seeds, because the chosen motor's path can now be reinforced
while its rivals' are weakened.

It does **not** yet resolve tasks that need the answer to depend on the **input**:

| task | chance | ceiling | uniform | credited |
|---|---|---|---|---|
| constant | 0.50 | 1.00 | 0.00 | **1.00** |
| copy `row[0]` | 0.50 | 1.00 | 0.45 | 0.44 |
| majority | 0.50 | 1.00 | 0.61 | 0.39 |
| parity | 0.50 | 1.00 | 0.53 | 0.35 |

Two structural facts, measured, bound this and are not learning-rule questions:

- **A third of the signal band is mute.** An input at the band floor of 0.1 attenuates to at most
  0.09 across one edge, which never clears the 0.3 edge gate, so a feature at its minimum fires no
  edge at all and an all-minimum observation produces no answer. The band and the gate were never
  reconciled with each other.
- **The mesh has no input-specific representation.** Between 55% and 83% of all edges fire on any
  given observation, and the fired sets of different inputs overlap by a Jaccard of 0.67 to 0.83.
  A sweep over `signal_threshold` and `neuron_firing_threshold` traded density against depth
  without producing above-chance discrimination anywhere: sparser settings starved the motors
  entirely rather than making the representation selective.

The likely cause is the wiring, not the rule: every sensor connects to **every** nexus interneuron,
so each feature excites the same population in the same way and there is no feature-specific
pathway for learning to strengthen differentially. Resolving that means changing the topology —
sparse or locally-structured sensor projection — which is a change to the architecture, and is
recorded here rather than made unilaterally.

### Updates use full-tensor selection, never boolean indexing

**Given** a learning step
**When** the masked update is applied
**Then** it uses `torch.where` over the full edge tensors.

Boolean indexing allocates and, on CUDA, forces a device-to-host synchronization through
`nonzero()`. This is a performance contract, and it is stated here because the natural way to write
the rule is the slow way.
