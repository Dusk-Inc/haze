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

### Updates use full-tensor selection, never boolean indexing

**Given** a learning step
**When** the masked update is applied
**Then** it uses `torch.where` over the full edge tensors.

Boolean indexing allocates and, on CUDA, forces a device-to-host synchronization through
`nonzero()`. This is a performance contract, and it is stated here because the natural way to write
the rule is the slow way.
