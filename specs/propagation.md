# Propagation

How a signal moves from sensor neurons to motor neurons.

## Description

Haze's mesh is a directed graph of neurons in four populations — sensor, nexus interneuron,
terminus interneuron, and motor. A signal enters at the sensors, traverses edges that attenuate it
by their strength, accumulates at interneurons that fire once they cross a threshold, and lands at
the motors, whose accumulated activation is what a decoder reads.

Propagation is expressed as tensor operations over an **edge list**, not a matrix product. The
state of every neuron is a vector; the graph is a pair of index vectors `src` and `dst` plus a
parallel `strength` vector; one hop is a gather, an elementwise product, and a scatter-accumulate.

## Policies

### Representation is an edge list, never a dense or sparse matrix

**Given** a mesh of N neurons and E edges
**When** the engine advances one hop
**Then** it gathers `val[src]`, multiplies by `strength`, and scatters into `dst` with
`index_add_`, and it never materializes an N×N matrix or a `torch.sparse` tensor.

The obvious formulation is `s_{t+1} = f(W · s_t + x_t)` with a sparse `W`. It does not work here,
for a reason that is functional before it is about performance: the edge gate is
`signal_after_this_edge > signal_threshold`, which requires the per-edge product as a materialized
intermediate. Both `torch.mm` and `torch.sparse.mm` fuse multiply-and-reduce and never expose it.
The per-edge fired mask is likewise per-edge state. Once an E-length vector is materialized for
both, the edge list is already present and the matrix product adds nothing.

The measured shape of the problem at a realistically grown mesh (N≈1900, E≈26k, density 0.65%):

| form | strength storage | fired mask at L=256 lanes | element-ops per step |
|---|---|---|---|
| dense N×N | 14.6 MB | 956 MB — does not fit | 934 M |
| sparse COO | 0.4 MB | not expressible | requires rebuild + coalesce on every `learn()` |
| edge list | 0.10 MB | 6.6 MB | 6.7 M |

Dense wins only above roughly 20% density with no per-edge state required. Haze is at 0.65% and
requires per-edge state. There is one code path.

### The edge gate is applied after attenuation, in edge space

**Given** an edge with strength `s` carrying a signal whose actual value after traversal is `a`
**When** the gate is evaluated
**Then** the edge passes only if `a > signal_threshold`, where `a` is computed after multiplying by
`s`, so the gate is an E-length operation and not an N-length one.

### The neuron gate is a sum over arrivals, in node space

**Given** an interneuron receiving signal from one or more edges in a hop
**When** the arrivals are accumulated
**Then** the neuron fires if the accumulated raw value is at least `neuron_firing_threshold`,
emitting the accumulated value onward; otherwise it emits nothing and its accumulated value is
discarded rather than retained for a later hop.

The two gates are deliberately asymmetric: the edge gate tests the actual (geometric-mean
corrected) value, the neuron gate tests the raw accumulated value. This asymmetry is inherited from
the prior engine and preserved intentionally.

### Sub-threshold accumulation is discarded, not integrated over time

**Given** an interneuron whose arrivals in a hop sum below `neuron_firing_threshold`
**When** the next hop begins
**Then** its accumulated value is zero; Haze is one-shot integrate-and-fire, not a leaky
integrator. Changing this makes it a materially different model and is not a tuning knob.

### An edge fires at most once per observation

**Given** an edge that has already passed its gate during the current observation
**When** a later hop would route signal through it again
**Then** the edge is refused.

This is the loop guard, and it is what makes propagation terminate: the nexus mesh **is not
acyclic** — the wiring routine samples targets randomly within the same mesh, so nexus→nexus cycles
exist. The guard is a `[lanes, E]` boolean `fired` tensor, zeroed once per observation. Because it
is monotone and bounded, each hop either sets at least one new bit or sets none; when none are set
the wavefront is empty and propagation ends. The loop therefore terminates in at most E hops
regardless of cycles, and in practice converges in graph-depth (3–12) hops.

The same tensor serves three purposes — the loop guard, the fired trace, and the learning mask —
which is why the prior engine's separate `Connector._history` and `Registry._status` both
disappear, and with them the unbounded-growth leak that made cost scale with training time.

### Inactive motors are gated before accumulation, not at readout

**Given** a motor that a decoder has marked inactive
**When** an edge would deliver signal to it
**Then** the edge is refused before the scatter, so the motor accumulates nothing **and** the edge
is absent from the learning trace.

Masking at readout instead would let inactive motors accumulate and would place their inbound edges
into the trace. That is a different model, not an implementation detail.

### Path statistics are carried, and their merge is an approximation

**Given** a signal that has traversed a path of edges with strengths `s₁…s_L`
**When** its actual value is computed
**Then** it equals `value · exp(sum_log / path_length)`, where `value` is the running product,
`sum_log` the running sum of `log(sᵢ)`, and `path_length` the hop count — the geometric mean of the
traversed strengths.

`value`, `sum_log`, and `path_length` are carried as three parallel node vectors. Per-edge this is
exact. At merge points a single node must summarize several arriving paths, and it does so with a
**value-weighted mean** of `sum_log` and `path_length`:

- Exact wherever fan-in is 1.
- Where fan-in exceeds 1, Jensen's inequality makes it an under-estimate of approximately
  `Var_w(g)/2` where `g = sum_log/path_length`. Bounded at ≤8% for strengths in [0.4, 0.9] and
  ~45% in the pathological two-path extreme spanning [0.1, 0.9].
- The error is one-signed. It tightens the edge gate, never loosens it, which is why it is
  absorbable by threshold recalibration rather than being a correctness defect.

Note that the correction factor is the geometric mean of the traversed strengths — a value in
[0.1, 0.9] that is essentially **independent of depth**. It does not stabilize long paths as the
README's original wording suggested; it applies a roughly constant haircut, with a weak preference
for paths built from uniformly strong edges over paths of the same product built from mixed ones.

### One observation is one lane

**Given** an observation of several features
**When** it is propagated
**Then** every feature enters the same lane, so an interneuron accumulates across features and
conjunctive structure can form.

This is load-bearing, and it was the single most damaging thing carried over from the prior
engine. That engine gave each feature its own propagation context because it ran one thread per
feature, and reproducing that isolation faithfully confined the mesh to a **sum over independent
features**: no interneuron ever saw two features at once, so parity was not merely hard but
unrepresentable, and the terminus representation collapsed into a small subspace.

Measured on the terminus activation, which is what the motors read:

| lanes | rank of the terminus representation |
|---|---|
| one per feature | 8-11 of 33 |
| one per observation | **32 of 32** |

A rank far below the neuron count means different inputs are not being represented differently,
and no readout can recover what was never encoded.

### Propagation is a synchronous wavefront

**Given** several edges leaving the same neuron
**When** a hop is advanced
**Then** all of them fire in the same hop, and contention for a shared downstream edge is resolved
by summation rather than by arrival order.

The prior engine drained one out-edge's entire cascade before enqueueing the next, so edge claiming
depended on traversal order and, under its thread pool, on scheduling. The wavefront makes results
order-independent and reproducible.

### Termination

**Given** a propagation in progress
**When** no edge passes its gate in a hop, or `max_steps` hops have elapsed
**Then** propagation ends.

`max_steps` is a safety cap (default 32), not the binding constraint.

## Calibration

`neuron_firing_threshold` and `signal_threshold` are **not** inherited from the prior engine's
values. Interneurons now genuinely accumulate fan-in, where previously the buffer was cleared on
every call and the effective gate was per-arriving-edge; the prior `0.5` is therefore far too
permissive. `signal_threshold` additionally absorbs the merge approximation above. Both are set by
the sweep recorded below.

The signal band and the edge gate are now reconciled with each other, and `HazeHyper` refuses a
combination where they are not. A feature at the band's floor attenuates by roughly the square of
an edge strength over one hop, so a floor below `signal_threshold / strength^2` cannot cross a
single edge. At the inherited floor of 0.1 against a gate of 0.3 the bottom third of the band was
**mute** — a feature at its minimum fired nothing and was indistinguishable from not having been
observed, and an all-minimum observation produced no answer at all. The shipped band is
`[0.4, 0.9]` against a gate of 0.25.

`neuron_firing_threshold` remains at its inherited value pending the topology work below.

## Measured ceiling

The instrument is a probe **matched to the readout it bounds**: one weight per neuron and class,
clamped to the same range the mesh clamps strengths to, no bias term, argmax across motors, and
accuracy scored on data held out of the fit. An unconstrained least-squares probe answers a
different and far more flattering question — it fits signed unbounded weights, carries a bias the
mesh has no equivalent of, and scores on its own training data. It reported copy at 0.91 and
majority at 0.98 where the matched probe reports 0.57 and 0.74. A ceiling that cannot be reached
is worse than no ceiling, because work gets spent chasing it.

The instrument had one further defect, found and fixed after its first use: it read `val`, which
holds only the current frontier and is empty once a pass has finished, so it was measuring
whatever happened to be in flight on the last hop. Terminus activity is now accumulated across the
whole propagation, the way motor activity already was. **The numbers below supersede an earlier
reading taken through the broken probe, and they reverse its conclusion.**

Over seeds 1-3, chance 0.50, with per-observation lanes, the reconciled band, and orphan rescue:

| task | matched bound | achieved | gap |
|---|---|---|---|
| constant | 1.00 | **1.00** | 0.00 |
| copy | 0.72 | 0.47 | -0.25 |
| majority | 0.79 | 0.27 | -0.52 |
| parity | 0.48 | 0.43 | -0.05 |

The terminus representation is full rank on every seed. The reading:

- **constant is solved.** It sat at one seed in three before the band and lane changes.
- **copy and majority are readout-limited, not representation-limited.** The representation
  supports 0.72 and 0.79; learning extracts 0.47 and 0.27. That is a large gap and it is where the
  remaining work is.
- **parity is representation-limited.** Achieved sits within 0.05 of a bound that is barely above
  chance, so no amount of better learning would help; it needs the mesh to form a conjunction it
  currently does not.

## Sparsity, measured twice

Wiring each sensor to a sample of the nexus rather than to all of it, and giving each motor its own
sample of the terminus, is the obvious way to give a feature somewhere of its own to act on. It was
measured before and after the representation fix and **did not raise the bound either time**: at a
sensor fan-out of 8 the bound moved from 0.57 to 0.58 on copy and from 0.74 to 0.69 on majority,
while edge activation went bimodal — either a few percent, where signal never reaches the motors
and the model has no answer at all, or near-total, where it reaches all of them equally.

The bimodality is the signature of a hard threshold with positive feedback: learning raises
strengths, more edges pass, more accumulates, more pass. There is no stable middle.

Fan-out is therefore a configured knob with a wide default rather than a claim, and the orphan
rescue built alongside it is kept on its own merits — pruning is allowed to disconnect a neuron,
and something has to reconnect it.

### Every input uses the whole mesh

The wiring is one way to be sparse; what actually *fires* is another, and the second one is
measured flat. Share of live edges carrying signal on an average observation, on the copy task
after 20 warm-up steps:

| nexus | neurons | live edges | ms/observation | share of edges fired |
|---|---|---|---|---|
| 64 | 96 | 998 | 2.11 | 0.89 |
| 128 | 192 | 1,612 | 2.44 | 0.90 |
| 256 | 384 | 3,589 | 3.60 | 0.90 |
| 512 | 768 | 7,341 | 5.55 | 0.91 |
| 1,024 | 1,536 | 14,227 | 8.45 | 0.91 |

**Nine edges in ten fire for every input, and enlarging the mesh sixteenfold does not change that.**
Three consequences follow, and they are the same fact seen from three sides:

- **Cost is the mesh, not the input.** Observation time tracks live edges, so a mesh large enough
  to hold a large problem is a mesh that pays for all of it on every observation.
- **Capacity is not allocated.** Two inputs that should be represented by different structure are
  represented by the same 90% of the edges, so every label's learning writes over every other
  label's. This is a mechanism for the label cliff that does not depend on the learning rule at
  all, and it predicts the rule-level fixes tried on this branch would each buy little — which is
  what they did.
- **Growth cannot localize.** New interneurons join the same undifferentiated pool, so adding
  capacity adds substrate every input immediately consumes rather than a region a failing input
  can move into.

The tension with the sparsity result above is real and is the open problem: firing less is what
capacity needs, and firing less is what stops signal arriving, because a path's value only ever
multiplies by strengths below one and convergence is what keeps it alive. Fan-out sparsity was
tried against that economy and lost. Changing the economy — so a neuron's arriving value does not
depend on how many edges fed it — is what would make sparsity affordable, and it is untried.

## An edge below the conduction floor is alive and mute

**Given** an edge whose strength has fallen below what the gate admits
**When** signal arrives at it
**Then** nothing traverses it, and because nothing traverses it the edge fires no trace, so no
subsequent update can reach it.

A path's arriving value is its running product of strengths scaled by their geometric mean, so a
uniform strength `s` over `n` hops arrives at `signal * s**(n+1)`. **One hop costs roughly the
square of a strength, not the strength.** Inverting that against `signal_threshold` gives
`HazeHyper.calcConductionFloor`, the weakest edge that carries the band's best signal:

| hops | at signal 0.9 | at signal 0.4 |
|---|---|---|
| 1 | 0.527 | 0.791 |
| 2 | 0.652 | 0.855 |
| 3 | 0.726 | 0.889 |

**The floor is exact for a sensor-adjacent edge and an upper bound elsewhere.** It inverts
attenuation against a signal drawn from the band, which is what a sensor emits. An interneuron's
arriving value is the *sum* over its incoming edges, so it can exceed the band's ceiling, and edges
downstream of a well-fed neuron conduct at lower strengths than the table gives. That is why a
fresh mesh reaches 1.00 while a quarter of its edges sit under the single-hop floor: fan-in
accumulation carries what no individual weak edge could.

Under the shipped rails — `strength_lower` 0.1, `prune_threshold` 0.2, `strength_upper` 0.9 —
this leaves a **dead band** at `(0.2, 0.527)`, reported by `HazeHyper.calcDeadBand`. An edge there
is alive, above the prune threshold, and carries nothing of its own. `ensureNoOrphans` cannot rescue it
because it is structurally connected; pruning cannot remove it because it is not weak enough. It is
an absorbing state, because every update is gated on a fired trace and a mute edge fires none.

This is what `calcConductionReach` measures and what accuracy hides. Trained on `first-set` capped
at nine labels with pruning off, the share of held-out inputs that reach the motors at all falls
1.00 → 0.99 → 0.74 → 0.51 over steps 0/250/500/1000, and `calcReachByLabel` shows the survivors
share one label: three of four seeds end at `1.0,0.0,0.0,0.0,0.0,0.0,0.0`. **An accuracy of 0.51
there is not a mesh predicting the majority class — it is a mesh that only still conducts for it.**

The existing coherence check validates `signal_lower * strength_init_upper**2`, i.e. that the
*strongest initial* edge conducts at the *weakest* signal. Nothing checks that a *surviving* edge
conducts, and the dead band is a representable state under the defaults. Whether to close it by
raising `strength_lower` to the conduction floor is a measured trade rather than a cleanup: on eight
seeds it takes three labels from 0.63 to 0.68 — the same 3/8 seeds learning, with better wins rather
than more of them — and does nothing at nine. The gain is not merely the smaller effective step a
narrowed strength range implies, since the shipped rails at the matched `epsilon` reach only 0.52.
Tracked in ROADMAP.md.

## Performance crossover

Tensor dispatch overhead dominates at small N. The measured crossover — the network size below
which the prior pure-Python engine is faster — is recorded here rather than omitted.

> Benchmark table recorded here in Phase 4.
