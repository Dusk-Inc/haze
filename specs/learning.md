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

## Known limitation: the rule performs no credit assignment

The rule above moves **every edge in the trace by the same scalar**. When two motors compete for
one answer, both of their inbound edges are in the trace, and both receive the same delta with the
same sign. Nothing in the rule distinguishes the motor that should have won from the motor that
did. Whichever motor happens to lead at initialization keeps leading, and the mesh's answer is
effectively fixed by its random wiring.

Measured, on the constant task — always answer the same label, chance 0.5, ceiling 1.0, the floor
any working learner must reach:

| `epsilon_start` | reward over the last 50 of 200 rows, seeds 1/2/3 | strength spread |
|---|---|---|
| 0.7 (inherited default) | 0.00, 0.00, 0.00 | saturated at the ceiling |
| 0.05 | 0.00, 0.00, 0.00 | unsaturated |
| 0.005 | 0.00, 0.06, 0.00 | unsaturated |

Two distinct effects were separated here:

1. **Saturation.** At the inherited `epsilon_start = 0.7`, one observation can move an edge ±0.7 on
   a [0.1, 0.9] range — rail to rail in a single step. Within a handful of observations every
   fired edge sits at the ceiling, both motors receive **numerically identical** activation,
   confidence falls to exactly zero, and therefore the delta falls to exactly zero. The mesh stops
   changing at all. This is a calibration fault and a smaller rate avoids it.
2. **No credit assignment.** With the rate reduced far enough that strengths stay spread, the
   reward *still* does not move off chance. This is not calibration. A uniform delta over the
   fired set cannot create the asymmetry that choosing between labels requires.

The second effect is architectural, and it is inherited rather than introduced: the prior engine
applied the same rule to the same mask, and its measured baseline is likewise flat at chance (see
LOG.md). Relieving it means changing the rule — weighting each edge's update by its own
contribution to the chosen answer versus the correct one — which is a change to what Haze *is*,
not to how fast it runs, and is therefore held for an explicit decision rather than folded into
this rewrite.

### Updates use full-tensor selection, never boolean indexing

**Given** a learning step
**When** the masked update is applied
**Then** it uses `torch.where` over the full edge tensors.

Boolean indexing allocates and, on CUDA, forces a device-to-host synchronization through
`nonzero()`. This is a performance contract, and it is stated here because the natural way to write
the rule is the slow way.
