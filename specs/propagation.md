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

> Sweep results and the chosen values are recorded here in Phase 3.

## Performance crossover

Tensor dispatch overhead dominates at small N. The measured crossover — the network size below
which the prior pure-Python engine is faster — is recorded here rather than omitted.

> Benchmark table recorded here in Phase 4.
