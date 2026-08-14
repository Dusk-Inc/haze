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

**What centring cost, measured later.** The inherited signal decomposes exactly as
`r − c ≡ (r − r̄) + (r̄ − c)`: today's advantage plus a bias measuring about +0.08. That bias was
not only a defect. Run head to head on `first-set` capped at `k`, 6 seeds:

| rule | reach 0 → 1500 | accuracy 0 → 1500 | share on one label |
|---|---|---|---|
| centred, k=3 | 1.00 → 0.69 | 0.27 → **0.66** | 0.84 |
| inherited, k=3 | **1.00 → 1.00** | 0.27 → 0.27 | 1.00 |
| centred, k=9 | 1.00 → 0.62 | 0.11 → **0.45** | 0.92 |
| inherited, k=9 | **1.00 → 1.00** | 0.11 → 0.21 | 0.96 |

The inherited rule never loses an input and never learns — by step 250 it answers one label for
everything and never moves again, scoring 0.27 at three labels, *below* a majority guess, because
it locks onto a minority label. The centred rule learns and pays for it in conduction. **The bias
was the entire conduction-preservation mechanism**, and no constant serves both roles: one large
enough to hold conduction open is large enough to freeze the policy. The two rules fail in opposite
directions and neither is closer to correct. Selection belongs to the reward signal; staying
conductive is a structural property and belongs in the strength rails — see specs/propagation.md.

**Confidence is not a usable value estimate, for a different reason than recorded above.** The
objection here was its positive mean. Measured since, its correlation with reward *changes sign
with label count*: +0.217 at three labels with a cleanly monotonic calibration curve (0.555 rising
to 0.971), −0.08 at two, and inverted at nine, where the top confidence bucket scores 0.28 against
0.54 at the bottom — the mesh is most certain when it is most wrong. A uniform bias is fixable by
calibration; a sign flip is not. But a **state-dependent** baseline is still worth having: bucketing
confidence and tracking mean reward per bucket took two labels from 5/8 to 7/8 seeds learning
(0.75 → 0.84). A critic predicting reward from terminus activation rather than from a one-number
summary is the version worth building, and is tracked in ROADMAP.md.

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

### An edge's learning rate hardens with reinforcement and softens with correction

**Given** a fired edge and the move it just made
**When** `crystallize` is on
**Then** its learning rate is multiplied by `epsilon_cool` if it was strengthened, `epsilon_warm`
if it was weakened, and held if it did not move, bounded to `[epsilon_floor, epsilon_start]`.

A section that has regularly been part of good answers should not be torn down by one bad result.
That hysteresis is what the inherited `epsilon_decay` promised and never delivered: it cooled every
fired edge identically whatever the outcome, never warmed one again, and at 0.9999 had a half-life
of 6,931 updates — leaving an edge holding 97-98% of its starting rate after a 3,000-step run.
Measured, not inferred: the settled mean learning rate under the shipped decay is 0.0963-0.0979
against a start of 0.1.

**Read per edge, from its own move, not from the observation's advantage.** The advantage version
was built first and does not work: a mesh performing steadily — well or badly — has an advantage
centred on zero once its baseline catches up, so nothing crystallizes at exactly the point a
section has earned it. An edge's own move has no such degeneracy, and being per-edge lets two
clusters in one mesh harden independently, which no global scalar could express.

`epsilon_floor` is load-bearing rather than defensive. An edge at a zero learning rate can never be
moved again whatever happens to it, which is the same absorbing state as an edge that has fallen
mute (see [propagation.md](propagation.md)). A fully crystallized edge is slow, never frozen.

**Measured on a mid-run task switch** — `copy(bit 0)` for 1,500 steps, then `copy(bit 3)`, 8 seeds:

| arm | pre-switch | +300 steps | recovered |
|---|---|---|---|
| off (shipped) | 0.96 / 0.84 | 0.54 / 0.47 | **0.39 / 0.37** |
| **on** | 0.97 / 0.79 | **0.95** / 0.56 | **0.73 / 0.54** |
| on, `epsilon_cool` 0.97 | 1.00 / 0.69 | 1.00 / 0.52 | 1.00 / 0.51 |

(reach / accuracy; chance 0.51.) **Three of eight seeds die outright with it off** — ending at
0.00-0.01, raising `SignalDidNotReachMotorsError` — and **none die with it on**. Conduction 300
steps after the switch is 0.95 against 0.54: the established mesh absorbs the shock instead of
collapsing. Recovered accuracy crosses from below chance to above it, with two seeds fully
relearning the new task (0.96, 1.00), though the mean of 0.54 says the mesh mostly *survives* the
change rather than relearning it well.

**On the stationary label sweep it is mixed**, 8 seeds, accuracy and seeds-that-learn:

| labels | off | on |
|---|---|---|
| 2 | 0.75, 5/8 | 0.75, 5/8 |
| 3 | 0.63, 3/8 | **0.68, 6/8** |
| 9 | **0.48, 1/8** | 0.33, 0/8 |

Three labels is the first movement on the bimodality recorded below — seeds that learn double, and
reach rises 0.66 to 0.83. Two labels is unchanged. **Nine labels regresses and is unexplained**;
the rate does fall to 0.071 there, so it is crystallizing, and why that costs accuracy is open.

`epsilon_cool` is a genuine trade-off rather than a knob to turn up. At 0.97 conduction goes to
0.93-1.00 everywhere and accuracy falls to chance — over-crystallization freezes the policy, which
is the same failure shape as the inherited `reward - confidence` rule. Defaulted off until the
nine-label regression is understood.

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

### Signal is judged to have arrived by magnitude, not by sum

**Given** motor activation that is net negative because inhibition outweighed excitation
**When** the decoder checks whether signal reached it
**Then** it answers from that activation, because evidence arrived.

Testing the signed sum conflates *silent* with *net inhibited*, and once edges may inhibit, the
second is an ordinary outcome rather than a failure. Measured across six runs of 800 observations,
**70 of 91** reported "no signal reached the motors" failures were motors that had received signal
and been voted down. Each was an answer thrown away and an observation not learned from; fixing
the test alone took held-out accuracy on copy from 0.86 to **0.95**, and its worst seed from 0.47
to 0.84.

Confidence is computed over magnitudes for the same reason. On signed activation the previous form
reported total certainty for any motor set summing positive and total uncertainty for any set
summing negative, which measures nothing and feeds the growth trigger.

### Reverse learning is available, and off by default

**Given** a prediction in which no signal reached the motors at all
**When** `relearn_limit` is greater than zero
**Then** the engine strengthens unfired edges to reopen a path; at the default of zero it does
nothing and the caller skips the observation.

This mechanism was designed against a symptom that was mostly the defect above. Once that was
fixed, genuine silences fell to **0.4%** of observations, and at that frequency every form of
intervention measured worse than leaving them alone:

| handling of an unanswerable observation | copy | worst seed |
|---|---|---|
| **skip it** | **0.95** | **0.84** |
| strengthen every unfired edge | 0.74 | 0.45 |
| strengthen only the wavefront's own edges | 0.66 | 0.47 |

The reason is blunt force: a reverse pass moves on the order of a hundred and seventy edges at
once, so firing it a handful of times across a run is enough to flatten what the mesh had learned.
Restricting it to the wavefront — the unfired edges leaving neurons the signal actually reached,
which are the only ones that can extend reachability — is better targeted and still worse, because
the surviving damage outweighs the rare recovery.

It is kept, tested, and reachable rather than deleted, because a mesh that genuinely cannot reach
its motors has no other way back, and a default set from one task family is a measurement rather
than a proof.

Worth recording that the mechanism did nothing at all before this: the update was
`epsilon · (reward − confidence)`, and on a reverse pass no signal reached the motors, so
confidence was 0 and the caller had no outcome to report but 0. The delta was exactly zero. The
recovery path had never once moved an edge.

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

Probing the same mesh before and after 800 observations, over seeds 1–4, says how much of each
task's outcome is representation and how much is readout:

| task | probe on a fresh mesh | probe on the trained mesh | shift |
|---|---|---|---|
| constant | 1.00 | 1.00 | +0.00 |
| copy | 0.81 | **0.99** | **+0.18** |
| majority | 0.82 | 0.79 | −0.03 |
| parity | 0.52 | 0.51 | −0.01 |

Copy and majority fail — and succeed — for different reasons, which no single number was going to
show. On copy the mesh **reshapes its own representation** toward the task, and that is why its
achieved 0.86 exceeds the untrained bound of 0.80 rather than contradicting it. On majority the
representation does not move at all, and 0.60 is extracted from information that stays worth 0.79.
Majority is therefore a readout failure and not a representational one, and it is the one place a
better rule still has something to collect. Parity does not move because there is nothing there to
move: its bound is chance before and after.

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

### The rule decides between two labels, and does not yet decide between more

**Given** a decoder with three or more active labels
**When** learning is applied
**Then** it extracts almost nothing, and past about nine labels it scores below chance.

Measured on one task held fixed and coarsened only in its label count, with the majority class
held at half the rows so chance stays 0.51 throughout. **Twelve seeds, which matters — the outcome
is bimodal, so a four-seed mean is a draw from a coin flip rather than an estimate:**

| labels | mean | seeds that learned (>0.60) |
|---|---|---|
| 2 | 0.64 | 6/12 |
| 3 | 0.59 | 3/12 |
| 4 | 0.44 | 1/12 |
| 6 | 0.49 | 2/12 |
| 9 | 0.50 | 3/12 |

A run either learns or sits at the majority rate; almost nothing lands between. What falls with
label count is the *probability of landing in the learning mode*, not the accuracy achieved. An
earlier version of this section reported 0.86 at two labels from four seeds; on twelve it is 0.64.

**Two explanations were recorded here and both are refuted.**

The first was that a scalar reward cannot carry which of `k − 1` rivals should have won, so the
information is absent. Measured: replacing the reward with the correct answer, at matched step
size, changes nothing at two labels and *hurts* above — seeds that learn go 2→0 at six labels and
3→0 at nine. At two labels the two signals are provably identical, since with two motors
`calcMotorSeeds` puts the positive push on the correct motor whether it was chosen (gain positive)
or not (gain negative); **that identity, not the information content, is why two labels work.**
Supervision that ignores the choice also destroys the choice-outcome correlation the rule estimates
from, which is why it actively hurts.

The second was that a code over motors would put every decision back in the binary regime. Built as
`CodeBook`, measured, and it loses to one-hot at every label count above two — a codeword is a
conjunction, so the barrier appears in both the decode (joint accuracy is the product of per-bit
accuracies) and the learning signal (the per-pair push is `4 * q**m * (1 - q)`, which peaks at
`q = m/(m+1)` and is 16× weaker than one-hot's at initialisation). See specs/decoding.md.

**What is actually happening is conduction collapse.** Training drives the edges serving harder
inputs under the gate, and a mute edge fires no trace, so no update of any kind can reach it again.
Reach falls 1.00 → 0.52 by step 1000 at nine labels, and the surviving inputs share one label. The
0.51 plateau is not the mesh predicting the majority class — it is the mesh only still conducting
for it. Supervision cannot help an input that generates no trace, which is why the strictly better
signal did nothing. See specs/propagation.md for the conduction floor and the dead band, and
ROADMAP.md for the remedies measured so far.

Not the cause, each checked separately: the eligibility trace discriminates at every label count
(same-vs-different-label cosine gap +0.377 at two, +0.364 at three, +0.249 at nine, against +0.03
fresh); the training budget is ample (nothing moves between step 1,500 and step 15,000); and the
representation holds the information (matched probe 0.81 on `majority` against 0.65 achieved).

This is the binding constraint on multi-modal work, where label spaces are large by nature.

### Updates use full-tensor selection, never boolean indexing

**Given** a learning step
**When** the masked update is applied
**Then** it uses `torch.where` over the full edge tensors.

Boolean indexing allocates and, on CUDA, forces a device-to-host synchronization through
`nonzero()`. This is a performance contract, and it is stated here because the natural way to write
the rule is the slow way.
