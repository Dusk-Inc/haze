# Learning

How edge strengths change in response to a reward.

## Description

Haze does not use backpropagation. It uses a forward-only rule: propagation records which edges
carried signal, and after an answer is scored, exactly those edges are moved toward or away from
the reward. There is no gradient, no loss surface, and no optimizer. An edge's own learning rate
(`epsilon`) decays each time it is updated, so early experience moves an edge more than late
experience.

## Policies

### The learning signal is how much the reward beat expectation

**Given** a prediction scored with reward `r`, against a running mean reward `r̄`
**When** learning is applied
**Then** the update is driven by the advantage `r − r̄`, and `r̄` moves toward `r` at
`reward_baseline_rate`.

What the reward is measured against decides what the mesh learns, and the inherited signal was
measured against **confidence**, which is not something the mesh is trying to beat. A correct
answer contributed `1 − c` and a wrong one only `−c`, so at even odds and typical confidence the
chosen answer's path was reinforced by a net positive amount **whether or not it was right**. That
is a positive feedback loop on whichever motor happened to lead, and it has a fixed point.

Measured on the copy task, it reached that fixed point every time: one motor won 96–100% of
observations, the mesh answered the same label forever, and reward sat at exactly chance. The
discriminative part of the signal was there but buried — the label-covarying component of an
edge's eligibility measured 2.5–3% of its mean, so the uniform push outweighed it roughly forty to
one.

Centring on the mesh's own recent reward removes exactly that term. The expectation of the
advantage is zero by construction, so the uniform push cancels and only the part of an edge's
activity that covaries with the outcome accumulates. Nothing else has to change: `eligibility ×
credit` already forms a covariance once the factor multiplying it is centred.

Centring the **eligibility** as well was tried and is worse — copy 0.83 → 0.51 — because the
product is then centred twice and the second centring only adds variance.

The baseline is seeded from the first reward rather than from zero or from an assumed 0.5. One
sample is its own expectation, so the first advantage is exactly zero; seeding at zero would spend
the whole warm-up applying the very push this removes, and seeding at 0.5 would assume a reward
scale the caller never agreed to. Setting `reward_baseline` false restores the inherited signal,
which is retained only so the difference stays measurable.

### A centred advantage requires a stochastic readout

**Given** a mesh whose answer is wrong on every observation
**When** learning is applied
**Then** it would receive no signal at all, so `explore_rate` of observations are answered with a
random active label instead of the best one.

These two are one mechanism and cannot be shipped apart. An advantage learns from the difference
between outcome and expectation, so it learns nothing when nothing varies — and a greedy readout
on a uniformly wrong mesh produces exactly that: reward is constant, the baseline meets it, the
advantage is zero, and the mesh stays wrong forever. Measured on the constant task with no
exploration, one seed in three settled at 0.00 and never moved. Reward variance is not a nuisance
here; it is the only thing there is to learn from, and a deterministic readout produces none.

Exploration is a **swap** into the lead rather than a boost, so the multiset of activations is
preserved: confidence, entropy, and every magnitude-sensitive decoder behave exactly as they would
have, and the one thing that changes is which label holds the peak. A boost would instead make the
mesh look more certain precisely when it is guessing, and confidence feeds both the learning
signal and the growth trigger. It applies only while learning — `eval()` always answers greedily,
so exploration never reaches a deployed answer.

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
how much credit reaches the neuron it fed, and how far the reward beat expectation — rather than
by one shared scalar.

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

### Edges may inhibit as well as excite

**Given** a mesh whose strengths may be negative
**When** signal traverses an inhibitory edge
**Then** the signal it delivers is inverted, so an input can rule an answer out rather than only
vote for one.

Three things move together and each is a place the obvious code is wrong:

- The path correction accumulates `log` of a strength's **magnitude**, since `log` is undefined for
  a negative. The sign belongs to the signal rather than to the attenuation — it is the value that
  inverts, while the amount of attenuation is a magnitude — so the sign travels in `value`.
- The edge gate tests the **absolute** value, so a strongly inhibitory signal propagates as readily
  as a strongly excitatory one. The neuron gate keeps testing the signed sum, since a net-inhibited
  neuron should stay quiet.
- Pruning tests magnitude. A useless edge is one near zero; testing the signed value would delete
  every inhibitory edge the moment it was created.

Measured over seeds 1-3, inhibition at a fifth of new edges raises the probe bound on copy from
0.72 to 0.75 and on parity from 0.48 to 0.51, raises achieved majority from 0.27 to 0.34, and
moderates edge activation from a 37-88% spread to 31-72%. Real but modest. `inhibitory_ratio = 0`
restores the purely excitatory model.

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

It did **not** on its own resolve tasks whose answer must depend on the **input**. That took two
further changes to the representation (one observation per lane, and the band reconciled with the
gate — see specs/propagation.md) and then the centred advantage above.

## Where the tasks stand

Held-out greedy accuracy on 200 rows never trained on, after 800 training observations, averaged
over seeds 1–6, at the shipped defaults. Scored in `eval()` so exploration never flatters the
number, and on fresh rows so memorisation cannot.

| task | chance | probe bound | inherited `r − c` | centred advantage |
|---|---|---|---|---|
| constant | 0.50 | 1.00 | 0.83 (one seed at 0.00) | **1.00** |
| copy `row[0]` | 0.50 | 0.80 | 0.49 | **0.86** |
| majority | 0.50 | 0.81 | 0.58 | **0.60** |
| parity | 0.50 | 0.50 | 0.49 | 0.50 |

Copy is the change: 0.49 to 0.86, from chance to most of the way to the ceiling. Constant is now
solved on every seed rather than most of them. Majority moves much less, and remains the clearest
open gap — 0.60 achieved against a bound of 0.81. Parity's bound *is* chance, so it is
representation-limited and no rule change touches it; it is kept as a diagnostic of whether the
mesh forms a genuine conjunction, and it does not.

### The probe bound is not a ceiling on a trained mesh

It is a bound on what a matched readout could extract from the terminus of an **untrained** one.
Learning moves the sensor→nexus→terminus edges too, so the representation being read is not the
representation that was measured, and a trained mesh may legitimately exceed it — as copy does.

This matters because the earlier stages of this work read "achieved vs bound" as if the bound were
a ceiling on the finished system. It is not, and a gap below it is the diagnostic it was built to
be — the information is present at initialization and the rule is failing to extract it — while a
result above it is not a contradiction and not an error.

Two related traps in the same instrument, both hit and both now closed: the probe scored on its own
training data with signed unbounded weights and a bias term the mesh has no equivalent of, which
inflated copy to 0.91 and majority to 0.98; and it took `strength_lower` as the readout's floor
after inhibition had moved the real floor to `−strength_upper`, reporting a bound the mesh had
already beaten. `probeRepresentation` now takes the hyperparameters rather than a weight range so
the second cannot recur.

### Rate is a real trade-off, not a free parameter

`epsilon_start` was inherited at 0.7, which is rail-to-rail in one observation and saturates the
mesh within a handful of steps. It is now 0.1, chosen by measurement over seeds 1–6:

| `epsilon_start` | `explore_rate` | constant | copy | majority | parity |
|---|---|---|---|---|---|
| 0.05 | 0.00 | 0.83 | 0.79 | 0.61 | 0.51 |
| 0.05 | 0.05 | 1.00 | 0.75 | 0.56 | 0.46 |
| 0.10 | 0.00 | 1.00 | 0.90 | 0.47 | 0.47 |
| **0.10** | **0.05** | **1.00** | **0.86** | **0.60** | **0.50** |
| 0.10 | 0.10 | 1.00 | 0.85 | 0.51 | 0.39 |
| 0.15 | 0.05 | 1.00 | 0.91 | 0.65 | 0.23 |

0.15 scores highest on copy and majority and is not the default, because it drives parity to 0.23
— well *below* the 0.50 chance its own bound sits at. A mesh scoring below chance on a task it
cannot represent has not failed to learn, it has confidently learned a wrong rule, and the same
instability shows up as `majority` at 0.04 on a single seed at 0.10/0.00. Declining to learn is
the safe failure and being confidently wrong is not, so the default is the setting where nothing
is driven below chance.

### Updates use full-tensor selection, never boolean indexing

**Given** a learning step
**When** the masked update is applied
**Then** it uses `torch.where` over the full edge tensors.

Boolean indexing allocates and, on CUDA, forces a device-to-host synchronization through
`nonzero()`. This is a performance contract, and it is stated here because the natural way to write
the rule is the slow way.
