# Haze Log

Context that must be preserved but does not belong in code comments. Newest first.

---

## feat-169 — Vectorized tensor engine

Fibery: [Feature 169](https://dusk-inc.fibery.io/Development/Features/169) (Product: Haze, Type: Update).

Replaces the threaded/queue object-graph simulation with a PyTorch tensor engine, and packages
the result as a loadable Hugging Face model. Branch `feat-169`, cut from `dev`.

### Why the rewrite

The prior engine represented every neuron as a Python object holding `list[Connector]`, every
signal as a `Signal` object mutated in place, and propagated by a `ThreadPoolExecutor` fanning one
thread per input feature, each draining a `queue.Queue` of `(callable, args)` closures. Three
problems, in order of severity:

1. **Cost grows with training time, not just network size.** The per-edge loop guard
   `Connector._history` was a `list[UUID]` that was never cleared — `clear_history()` had zero
   callers — and was scanned with a linear `in` on every edge fire. `Motor._signals` grew by
   `np.append` for the same reason. The engine was asymptotically wrong, not constant-factor slow.
2. **The threads bought nothing.** The work is GIL-bound Python, and the per-object `Queue` in
   `Connector`/`Inter`/`Motor`/`Registry` was always enqueue-then-drain-one — no buffering, no
   concurrency, pure lock and allocation overhead on the hot path.
3. **It was racy.** N feature threads mutated shared `Registry` numpy arrays, `Inter._signal_buffer`,
   `Connector._history`, and `Motor._signals` under per-object `RLock`s that covered only the
   trivial dequeue, never the read-modify-write. `Inter.transmit` was not synchronized at all.

### Baseline of the prior engine, measured before replacement

Captured on the 8-bit binary-sequence echo demo (`app/main.py`), 200 rows per seed, on this
devcontainer. The prior engine remains recoverable at commit `20ef84f`.

| seed | mean reward | 1st half → 2nd half | median ms/step | first 10 steps → last 10 steps |
|---|---|---|---|---|
| 1  | 0.492 | 0.495 → 0.489 | 167 | 54 ms → 458 ms |
| 4  | 0.498 | 0.514 → 0.483 | 190 | 68 ms → 481 ms |
| 42 | 0.488 | 0.499 → 0.478 | 171 | 63 ms → 501 ms |

Three facts to carry forward:

- **No learning is demonstrated.** Reward sits at chance and if anything declines across the run.
  This is not conclusive evidence the engine failed to learn, because the task cannot show
  learning either way: `generate_binary_sequence` emits uniform random bits, so chance = ceiling =
  0.5 and a flat curve is indistinguishable from a broken engine. It does mean the demo is
  worthless as a regression oracle, which is why `src/functions/score.py` adds tasks with a real
  learnable ceiling (constant, copy, parity) as the actual validation method.
- **Step time grew ~8× over 200 rows** and shows no sign of levelling. Extrapolated to the demo's
  own 10,000 rows, a single step would take tens of seconds. This is the `_history` and
  `_signals` growth described above, and it is the primary thing the rewrite fixes.
- **Half the seeds do not run at all.** Seeds 2, 3, and 5 die with `Network did not connect signal
  to motors within the iteration limit` — the random wiring fails to reach the motors and the
  recovery loop in `Haze.predict` exhausts its retries. Any claim about the prior engine's
  behavior is conditioned on a surviving seed.

### Semantics: deliberate divergences from the prior engine

The rewrite adopts clean semantics rather than reproducing the prior behavior bug-for-bug. Each
item below is a behavior change, not a refactor, and each is landed as its own commit so a
regression is attributable.

- **Motors reset per observation.** `Motor.reset_state()` set `self._state = 0`, an attribute
  nothing read; the real accumulator `self._signals` was never cleared. Motor activations therefore
  accumulated for the entire process lifetime, so every prediction after the first was contaminated
  by all prior ones. `Decoder._reset_motors()` called it before and after every `predict()`
  believing state was cleared.
- **Interneurons genuinely accumulate fan-in.** `Inter.transmit` buffered arriving signals, summed
  them, and then cleared `_signal_buffer` unconditionally at the end of every call — and
  `Context.run` delivered one ingress per call. So `total_value` was almost always a single
  signal, and the effective gate was per-arriving-edge rather than a spatial sum. Multi-signal
  merges happened only when two feature threads raced. The tensor engine's scatter-accumulate
  implements the accumulation the prior code described but did not perform, which means
  `neuron_firing_threshold` must be recalibrated upward (see `specs/propagation.md`).
- **Terminus→motor edges are learnable.** `Decoder.check_terminus` constructed `Connector`s and
  posted them without calling `registry.add_connector`, leaving `_index is None`, so
  `get_strength()` returned a hardcoded `0.9` and `activate_connector` was never reached. Every
  such edge was permanently pinned at near-maximum strength and invisible to learning.
- **Pruned edges stop participating in learning.** Pruning removed connectors from
  `Registry._connectors` but never removed their `_strength`/`_epsilon`/`_status` slots. Because
  `learn(reverse=True)` selects on `~_status`, every orphaned slot was reverse-learned on every
  reverse pass, forever, and the arrays grew monotonically.
- **Confidence is computed over active motors only.** `Decoder.confidence()` took entropy over
  `get_motors()` while `_predict_impl` used `get_active_motors()`, so confidence was systematically
  deflated by an inactive label space. It feeds both the learning delta and the growth trigger.
- **Propagation order changes from depth-first drain to synchronous wavefront.** `Sensor.transmit`
  called `context.run()` *inside* its loop over connections, so one out-edge's entire cascade
  drained before the next was enqueued, and edge claiming was arrival-order dependent. The tensor
  engine fires all edges of a wavefront together and resolves contention by summation. This makes
  results order-independent and reproducible; the prior engine was not reproducible even with a
  fixed seed.

### Approximation introduced

`Signal.get_actual() = value · exp(sum_log / path_length)` is a per-path quantity, and a state
vector merges paths. The engine carries `value`, `sum_log`, and `path_length` as parallel node
vectors and, at merge points, takes a value-weighted mean of `sum_log` and `path_length`. This is
algebraically exact wherever fan-in is 1. Where fan-in exceeds 1 it under-estimates by
approximately `Var_w(g)/2` by Jensen — bounded at ≤8% for realistic strengths in [0.4, 0.9] and
~45% in the pathological two-path extreme. The error is one-signed (never an over-estimate), so it
tightens the edge gate rather than loosening it, and is absorbed by recalibrating
`signal_threshold`.

### What the rewrite revealed: the learning rule does not do credit assignment

The tensor engine is verified correct — it agrees with a plain-Python reference implementation to
float32 precision across every seed tested, propagation terminates on the cyclic nexus mesh, and
step time is now **flat** where the prior engine's grew without bound. It is also demonstrably
faster: ~2.6 ms per observe/predict/learn step at 300 rows against the prior engine's ~470 ms at
the same point, and the gap widens with run length rather than closing.

But validating on task rather than on bug-parity surfaced something the speedup does not fix.
`Δ = ε · (reward − confidence)` is applied uniformly to every edge in the fired trace. When two
motors compete, **both** of their inbound edges are in that trace and both move by the same delta
with the same sign. The rule therefore cannot make the correct motor win; whichever motor leads at
initialization keeps leading. Full measurements are in `specs/learning.md` — the short version is
that the constant task, which any working learner must score ~1.0 on, scores 0.00 across seeds and
across three orders of magnitude of learning rate.

Two effects were separated. Saturation is a calibration fault: at the inherited `epsilon_start =
0.7`, a single observation moves an edge rail to rail, every fired edge pins to the ceiling within
a few steps, both motors receive numerically identical activation, and confidence and therefore the
delta both collapse to exactly zero. A smaller rate avoids that. But with the rate small enough
that strengths stay spread, reward still does not leave chance — so the second effect is not
calibration.

This is inherited, not introduced. The prior engine applied the same rule to the same mask, and its
own measured baseline is flat at chance with three of six seeds failing to reach the motors at all.
It also explains that baseline, which the random-bit demo could not: the demo's chance and ceiling
are both 0.5, so it could never have distinguished a learning engine from a broken one.

### Credited learning, and where it gets to

On that finding the learning rule was changed deliberately rather than silently, keeping it
forward-only: an edge now moves by three factors multiplied — how much signal it carried, how much
credit reaches the neuron it fed, and how far the reward diverged from the confidence. Credit is
seeded at the motor the decoder actually chose, since `learn` is told a reward and never the
correct label, and spread backward along the fired edges by one extra scatter per hop. Full
description and measurements in `specs/learning.md`.

It resolves what it was aimed at. The constant task — which needs only a bias — goes from 0.00 to
0.99–1.00 across seeds where the uniform rule scored 0.00 at every learning rate tried.

It does not yet resolve tasks whose answer must depend on the input: copy, majority, and parity all
remain at chance. Two measured structural facts bound that, and neither is a learning-rule
question:

- **A third of the signal band cannot propagate.** A feature at the band floor of 0.1 attenuates to
  at most 0.09 over one edge and never clears the 0.3 edge gate, so a feature at its minimum fires
  nothing and an all-minimum observation yields no answer at all. The band and the gate were never
  reconciled with each other.
- **The mesh forms no input-specific representation.** 55–83% of all edges fire on any observation,
  and different inputs' fired sets overlap by a Jaccard of 0.67–0.83. A sweep over both thresholds
  traded density against depth without finding above-chance discrimination anywhere; sparser
  settings starved the motors rather than making the representation selective.

The likely cause is the wiring rather than the rule: every sensor connects to **every** nexus
interneuron, so each feature excites the same population identically and there is no
feature-specific pathway for learning to strengthen differentially. Changing that means changing
the topology, which is a further architectural decision and is left open rather than taken
unilaterally.

### Conventions

The rewrite conforms to the workspace `src/{tokens,models,interfaces,modules,functions,errors,tests}`
layout, Pydantic models, the `[Verb][Domain][Qualifier]` function taxonomy, and the
docstring-per-declaration gate. Framework-mandated names (`forward`, `save_pretrained`,
`from_pretrained`, `state_dict`, `train`, `eval`) are exempt from the taxonomy; that exemption is
recorded in `specs/packaging.md`.

`pyproject.toml` maps `"haze" = "src"` rather than following `dusk-core-py`'s
`where = ["src"]`, which would install `functions`, `models`, and `tokens` as top-level import
names — acceptable for an internal lib, disqualifying for one published to PyPI and loaded from
the Hub as `from haze import Haze`.

## Centring the learning signal

The gap the previous entry left open — copy, majority, and parity stuck at chance — is closed for
copy and partly for majority. It was not the wiring, and it was not the credit rule. It was what
the reward was being measured against.

The inherited signal was `reward − confidence`. Confidence is not something the mesh is trying to
beat, and the arithmetic follows: a correct answer contributed `1 − c` and a wrong one only `−c`,
so at even odds the chosen answer's path was reinforced by a net positive amount whether or not it
was right. Instrumenting a run made the consequence unambiguous — one motor was chosen on 98–100%
of observations, the mesh answered the same label forever, and reward sat at exactly chance. The
discriminative information was present the whole time and simply outweighed: the label-covarying
part of an edge's eligibility measured 2.5–3% of its mean, against a uniform push roughly forty
times larger.

Measuring the reward against the mesh's own running mean removes that term exactly, because the
expectation of the advantage is zero by construction. Nothing else needed to change — `eligibility
× credit` already forms a covariance once the factor multiplying it is centred. Centring the
eligibility as well was tried and is worse (copy 0.83 → 0.51): the product is then centred twice
and the second centring only adds variance. That is the opposite of what I predicted, and the
prediction was the reason the experiment was run.

Centring alone creates a new failure that had to be fixed in the same change, so the two ship
together. An advantage learns from the difference between outcome and expectation, so it learns
nothing when nothing varies — and a greedy readout on a uniformly wrong mesh produces exactly
that: reward constant, baseline meets it, advantage zero, stuck forever. One seed in three settled
at 0.00 on the constant task and never moved. Exploration supplies the variance, as a swap into
the lead rather than a boost so that confidence is not inflated precisely when the mesh is
guessing, and only while learning.

Held-out greedy accuracy on fresh rows, seeds 1–6, like for like at `epsilon_start = 0.1`:

| task | inherited `r − c` | centred | centred + exploring |
|---|---|---|---|
| constant | 0.83 (one seed 0.00) | 1.00 | **1.00** |
| copy | 0.49 | 0.90 | **0.86** |
| majority | 0.58 | 0.47 (one seed 0.04) | **0.60** |
| parity | 0.49 | 0.47 | **0.50** |

### Two corrections to earlier entries

**The sparse-wiring hypothesis recorded above is withdrawn.** It was tested twice, before and after
the representation fix, and did not raise what a readout could achieve either time; the entry
kept it as "the likely cause" longer than the evidence supported.

**The probe bound is not a ceiling on a trained mesh.** It bounds what a matched readout could
extract from an *untrained* terminus, and learning moves the sensor→nexus→terminus edges too, so a
trained mesh may legitimately exceed it — as copy now does at 0.86 against 0.80. Earlier stages
read "achieved vs bound" as if it bounded the finished system. A gap below it remains the
diagnostic it was built to be; a result above it is neither a contradiction nor an error.

The instrument also had two defects of its own, both now closed: it scored on its own training
data with signed unbounded weights and a bias term the mesh has no equivalent of, and after
inhibition it was still taking `strength_lower` as the readout's floor when the real floor had
become `−strength_upper`. The second reported a bound the mesh had already beaten, which is worse
than no bound, since it retires work that is not finished. `probeRepresentation` now takes the
hyperparameters rather than a weight range so it cannot recur.

### What is still open

`epsilon_start` is now 0.1, measured rather than inherited — the inherited 0.7 is rail-to-rail in
one observation. 0.15 scores higher on copy and majority and was rejected because it drives parity
to 0.23, below the 0.50 chance its own bound sits at; a mesh scoring below chance on a task it
cannot represent has confidently learned a wrong rule rather than declined to learn, and the same
instability appears as majority at 0.04 on one seed. Declining to learn is the safe failure.

Majority remains the clearest gap: 0.60 achieved against a bound of 0.81. Parity's bound is chance,
so it is representation-limited and no rule change reaches it.
