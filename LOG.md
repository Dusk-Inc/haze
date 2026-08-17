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

Probing the same mesh before and after training separates copy from majority for the first time.
On copy the mesh reshapes its own representation toward the task — the probe rises 0.81 → 0.99 —
which is why achieved 0.86 exceeds the untrained bound of 0.80 rather than contradicting it. On
majority the representation does not move (0.82 → 0.79) and the mesh extracts 0.60 from information
still worth 0.79. So majority is a readout failure and not a representational one, and it is the
one place a better rule still has something to collect. Parity does not move because there is
nothing there to move.

## Wiring growth, pruning, and the trainer — and what that turned up

The auditor, growth, pruning, and the trainer loop are now built and wired: `Auditor` holds the
rolling windows, `calcGrowthPlan`/`applyGrowth` size and apply a plan, `applyPrune` removes dead
edges and rescues what that strands, and `Trainer` drives observe → answer → score → learn →
audit. With restructuring disabled the trainer reproduces hand-driven training exactly, which is
the check that it adds nothing of its own.

Building it surfaced three defects that mattered more than the feature.

**The readout was misreading inhibition as silence.** `ensureSignalReached` tested whether motor
activation summed above zero. That was a correct test for "nothing arrived" only while activation
was non-negative; once Stage 4 made strengths signed, a net-inhibited motor set — an ordinary
outcome — read as a dead mesh. Measured, **70 of 91** apparent failures were answers being thrown
away. Fixing that one comparison took held-out copy accuracy from 0.86 to **0.95** and its worst
seed from **0.47 to 0.84**.

That is the answer to the seed-variance question. It was never a fragile initialization: it was one
comparison that had stopped being valid two stages earlier, and the seeds it hit hardest were
simply the ones whose meshes leaned most inhibitory.

**Reverse learning had never once moved an edge.** Its delta was `epsilon · (reward − confidence)`,
and on a reverse pass no signal reached the motors, so confidence was 0 and the caller had nothing
to report but reward 0. The product was exactly zero, every time, since the rewrite began.

Giving it a real magnitude then showed it does not earn its place. With the readout fixed, genuine
silences are 0.4% of observations, and at that rate skipping them beats every intervention: 0.95
skipping, 0.74 strengthening all unfired edges, 0.66 strengthening only the wavefront. A reverse
pass moves ~170 edges at once, so a handful of firings flattens what was learned. `relearn_limit`
now defaults to 0. The mechanism stays tested and reachable, because a mesh that truly cannot reach
its motors has no other way back and one task family is not a proof.

**Both restructuring triggers were set at values that fire on a healthy mesh.** `audit_window = 10`
is too few observations to estimate an error rate and let growth fire every ten steps — a 400-step
run added 253 neurons and pruned 5089 edges while ending at chance. And `growth_threshold = 0.5` is
exactly the error rate of an untrained mesh on a two-label task, so growth fired on the first full
window of every run; restructuring mid-learning cost one seed 0.95 → 0.46. Pruning also had to be
gated on the audit's verdict rather than its cadence: the removal is not what hurts, the repair is,
since rescuing stranded neurons with fresh random edges injects noise into a converged mesh.

### Growth does nothing on these tasks, and that is the expected result

With the triggers corrected, restructuring is exactly neutral — identical results live and
disabled. That is correct rather than disappointing. The matched probe says the terminus already
carries 0.80–0.82 of the available information at initialization, so this task family is not
capacity-limited, and growth answers a capacity limit. A task that cannot pose the problem cannot
demonstrate the solution; it can only show the cost of applying it anyway, which is what the two
regressions above measured. Demonstrating growth needs a task whose ceiling moves with mesh size,
and that is outstanding.

### Where copy now stands

Held-out greedy accuracy on 200 fresh rows, seeds 1-6: copy **0.95** (worst seed 0.84), majority
0.65, against probe bounds of 0.80 and 0.81. Copy is now above its untrained bound on every seed,
which the trained-mesh probe explains — the mesh reshapes its representation toward that task.
Majority still does not, and remains the open readout gap.

## Looking for a task that could demonstrate growth, and finding the label cliff

Growth was wired but could not be shown to do anything, because every task in the suite is already
saturated at initialization — the probe reports the terminus carrying 0.80–0.82 of the available
information before a single observation. Growth answers a capacity limit, so a task with no
capacity limit can only measure its cost.

`first-set` — the index of the row's first set bit — is that task. Its matched-probe bound rises
with terminus width, 0.50 on a terminus of 4 (exactly chance) against 0.65 on 32, and a test now
pins that so it cannot quietly stop being capacity-limited.

It still did not demonstrate growth, for a reason worth more than the demonstration would have
been. Growth from a narrow start made things worse (0.32 to 0.15 from a terminus of 4) and grew the
mesh enormously, 4→35 terminus and 64→372 nexus. But the absolute numbers were the tell: every
variant was *below* chance. The mesh was not learning the task at any size, so there was nothing
for growth to improve.

### It is the label count, and the cliff is between two and three

Same task, coarsened only in how many labels it has, with the majority class held at half the rows
so chance stays 0.51 throughout:

| labels | achieved | lift over chance |
|---|---|---|
| 2 | 0.86 | **+0.35** |
| 3 | 0.59 | +0.08 |
| 4 | 0.58 | +0.07 |
| 6 | 0.57 | +0.06 |
| 9 | 0.46 | **−0.05** |

Not capacity: the same task at two labels reaches 0.94–0.98 on the same mesh, and a *larger* mesh
makes nine labels worse (0.46 at terminus 32, 0.27 at 64). Not a gradual decay either — it falls
off almost entirely at the third label.

The cause is what a scalar reward can carry. Learning is told whether its answer was right, never
what the right answer was. At two labels that is complete information, because "not the one I
chose" names the other exactly, and the rival seed is the correct corrective signal. At three or
more it is ambiguous, and the seed spread across rivals is right about at most one of them.

I expected the rival seeding to be the culprit and tested it: seeding only the chosen motor,
halving the rival share, and quadrupling exploration all land within noise of the current rule and
of each other. The information is absent, not misallocated. Exploration cannot rescue it either —
a random alternative is correct one time in `k−1` while the signal promoting it is already divided
by `k−1`.

The promising direction is structural rather than another rule: express a `k`-label choice as
`⌈log₂ k⌉` binary ones, a code over motors instead of one motor per label, which puts every
decision back in the regime that demonstrably works and makes label count logarithmic. That is a
change to the decoder contract, so it is recorded rather than taken.

This is the binding constraint on the multi-modal goal, where label spaces are large by nature, and
it outranks both the majority readout gap and growth as the thing to resolve next.

## The codebook decoder: built, measured, and does not work

The label cliff — two labels 0.94-0.98, three or more barely above chance, nine below it — was the
binding constraint on everything multi-modal, so the proposed fix was to stop giving each label a
motor and give it a **codeword** instead: motors in pairs, one pair per bit, each pair a two-way
race in the regime the rule handles exactly.

It is built and tested (`CodeBook`, `src/functions/codebook.py`, 23 tests) and it loses to one-hot
at every label count above two. Full numbers in `specs/decoding.md`.

**The reason is multiplicative and I did not see it coming.** A codeword is a conjunction: the
answer is right only when every bit is right, so joint accuracy is the *product* of per-bit
accuracies. Nine labels over eight bits needs each bit right 0.92 of the time before the whole
answer beats simply always naming the majority label. One-hot faces no such threshold, because one
motor winning is a common event and it collects reward from the first observation. The reduction
does not divide the problem, it conjoins it.

The `0.00` cells are that effect at its sharpest: the majority label's thermometer codeword is all
zeros, so producing it requires `m` independent pair-races to all land the same way — about `2^-m`
by chance. The most frequent answer became the least reachable one.

**What was ruled out.** Three exploration strategies (whole codeword, one random bit, per-bit
independent) at two rates; two code families; and per-bit supervision, which replaces the shared
scalar with each pair's own target bit and is the most information a teacher could supply. That
last one is the informative negative: at 0.30 for nine labels it lifts the floor without clearing
the bar, which says the barrier is **not** credit assignment and **not** the teaching signal. It is
that per-bit accuracy is too low for the product to survive.

**A process note.** I changed the exploration strategy from single-bit to whole-codeword on the
strength of one cell of one row — the k=2 random-code result — before ever measuring k≥3 with the
original. The change turned out to be right on its own merits (with a redundant code a single
flipped bit still decodes to the same label, so bit-level exploration cannot change the answer at
all, 0.27 against 0.91) but I did not know that when I made it. Measuring both properly afterwards
is what turned a guess into a result.

**What is left untried:** a code with real Hamming distance whose bits are also individually easy,
so residual errors are corrected rather than multiplied. Thermometer gives easy bits and distance 1;
random codes give distance and harder bits; nothing here gives both. That, or raise per-bit
accuracy — which is the same work as the majority readout gap.

`CodeBook` ships available and non-default, and one-hot stays the default, so the two remain
comparable and this result can be re-tested against a better rule rather than re-derived.

## Chasing the label cliff to its root: conduction collapse

Nine measurements in one session, prompted by two questions: does the network actually grow new
capacity when established behaviour goes stale, and was `reward − confidence` retired for a good
reason. Both answers turned out to depend on a failure neither question was about.

### Three root causes proposed, two refuted by the next measurement

Recorded in order, because the pattern matters more than the conclusion: mechanisms reasoned about
confidently kept being overturned by cheap experiments, and only the third survived contact.

1. **"We stop training too early."** The advantage contrast falls as `1/k` while per-step noise
   stays flat, so resolving it should cost `k²` more samples — 1,500 steps at two labels being
   worth ~30,000 at nine. **Refuted:** out to 15,000 steps everything is settled by ~1,500 and then
   nothing moves at all — accuracy, mean `|strength|` (0.51, mid-band), rail occupancy (0.17–0.30)
   all static for the remaining 13,500. The drift is not small, it is zero, and no amount of
   sampling resolves a contrast that is not there.
2. **"The update cannot be made input-conditional."** Supported by supervision failing and by the
   probe reading 0.81 where the mesh achieves 0.65. **Refuted:** the eligibility trace discriminates
   at every label count. Same-label vs different-label cosine gap after training is +0.377 (k=2),
   +0.364 (k=3), +0.249 (k=9), against +0.03 on a fresh mesh. Selectivity is learned and it works.
3. **"Silence is absorbing."** Survives all nine.

### The mechanism

Training destroys the mesh's ability to conduct, one input at a time, and conduction does not come
back. An input whose answer is usually wrong accumulates negative advantage on the edges it fires;
those edges fall under the gate; a mute edge fires **no trace**, and every update is gated on a
trace, so nothing can ever strengthen it again. The caller scores the silence as reward 0, which
deepens the collapse for everything else.

Share of 200 held-out inputs still reaching the motors, `first-set` capped at `k`, pruning off:

| step | k=3 | k=9 |
|---|---|---|
| 0 | 1.00 | 1.00 |
| 500 | 0.95 | 0.74 |
| 1000 | 0.63 | 0.51 |
| 1500 | 0.65 | 0.52 |

Per-label at 1500, the majority class survives in every seed: `1.0,1.0,0.3 | 1.0,0.5,0.5 |
1.0,0.0,0.0 | 1.0,0.0,0.0` at three labels, `1.0,0.0,0.0,0.0,0.0,0.0,0.0` in three of four at nine.

**This re-reads the whole task suite.** The 0.51 plateau is not a mesh predicting the majority class
— it is a mesh that only still conducts for it, scoring zero on the silence elsewhere. "Converges by
1,500 steps" is the erosion completing. Supervision cannot help because a silent input has no trace
to apply it to. The task-switch deaths are the same mechanism with every pathway turning negative at
once.

### It has a parameter-level cause

Arriving value is `signal × prod(strengths) × geomean(strengths)`, so one hop costs roughly the
*square* of a strength. Against `signal_threshold` 0.25 and a band topping at 0.9, an edge below
**0.527** cannot conduct at any signal. `strength_lower` is 0.1 and `prune_threshold` is 0.2, so
`(0.2, 0.527)` is a **dead band**: alive, above the prune threshold, and mute. `ensureNoOrphans`
cannot see it (structurally connected), pruning cannot reach it (not weak enough). A settled mesh
measures mean `|strength|` 0.51 — sitting on the floor. Now computable as
`HazeHyper.calcConductionFloor` / `calcDeadBand`, and covered by `src/tests/test_conduction.py`.

The existing coherence check validates that the *strongest initial* edge conducts at the *weakest*
signal. Nothing checks that a *surviving* edge conducts.

### The two learning rules fail in opposite directions

`r − c ≡ (r − r̄) + (r̄ − c)` — the inherited signal is exactly today's advantage plus a bias of
about +0.08. Head to head, 6 seeds: the inherited rule holds reach at **1.00 throughout** and never
learns (by step 250 it answers one label for everything; 0.27 at three labels, below a majority
guess, because it locks onto a minority label). The centred rule learns (0.27 → 0.66) and loses a
third of its input space. **The bias was the entire conduction-preservation mechanism**, and no
constant serves both roles: one big enough to hold conduction open is big enough to freeze the
policy. Selection belongs to the reward; staying conductive is structural.

Confidence remains unusable as the value estimate, but for a better reason than the one recorded
when it was dropped: its correlation with reward **changes sign with label count** (+0.217 at three
with a monotonic calibration curve, −0.08 at two, inverted at nine where the top confidence bucket
scores 0.28 against 0.54 at the bottom). A uniform bias is fixable by calibration; a sign flip is
not. A *state-dependent* baseline is still worth having — bucketing confidence and tracking mean
reward per bucket took two labels from 5/8 to 7/8 seeds learning.

### Remedies measured

| arm | k=3 (4 seeds) | k=3 (8 seeds) | k=9 (8 seeds) |
|---|---|---|---|
| shipped, no pruning | 0.59 | 0.63 | 0.48 |
| shipped, pruning on | 0.66 | — | **0.58** |
| prune at the conduction floor (recycle mute edges) | 0.44 | — | 0.52 |
| **`strength_lower` at the conduction floor 0.55** | 0.77 | **0.68** | 0.46 |

**The 0.77 was a four-seed draw and does not hold.** On eight seeds the floor gives 0.68 against
0.63, with **3/8 seeds learning in both arms** — it does not win more often, its wins are better
(1.00, 1.00, 0.86 against 0.86, 0.79, 0.77). Recording the correction rather than the corrected
number alone, because the error is the one this same entry criticises two sections above: a
four-seed mean of a bimodal outcome is a coin flip, and I published one within the hour.

It does not convert at nine labels at all: conduction is necessary and not sufficient. Recycling
mute edges, the fix argued for most confidently, measures *worse* than doing nothing at three
labels — it destroys learned structure faster than it restores conduction.

**The step-size control passes.** Narrowing strengths to `[0.55, 0.9]` also shrinks the dynamic
range, so the shipped rails were re-run at the matched `epsilon` of 0.044 and below: they reach
0.52 against the floor's 0.68. The gain is not a smaller step.

**Conduction and learning trade off under three separate mechanisms**, which outlasts any of the
individual arms. Lowering `epsilon` on the shipped rails takes reach 0.66 → 0.93 → 0.98 at
0.100 / 0.025 / 0.010 while accuracy falls 0.63 → 0.53 → 0.41. The inherited rule holds reach at
1.00 and never learns. Both buy conduction and pay everything for it. The strength floor is the
only mechanism measured that improves both at once, which is what keeps it interesting at an effect
size this small.

A floor and a prune threshold cannot both be used: `ensureHyperCoherent` requires
`prune_threshold > strength_lower`, so an edge clamped to a conduction floor sits permanently below
the prune threshold and is stripped on the next pass (0.40 accuracy, 0.42 reach).

### Corrections to earlier entries

- **Severity was overstated by my own harness.** Every reach measurement above ran with pruning
  disabled, because `applyPrune` is called from the trainer's `flowRestructure` and a harness driving
  `learn()` directly never invokes it. Switching it on recovers reach 0.52→0.70 and accuracy
  0.46→0.58 at nine labels. The erosion is real; "conduction never comes back" is only true with
  pruning off.
- **The label-cliff table was four seeds and the outcome is bimodal.** On twelve seeds two labels is
  0.64 with 6/12 learning, not the 0.86 previously recorded. A run either learns or sits at the
  majority rate; what falls with label count is the probability of the former.
- **`calcCodeSeeds`'s docstring claimed the shared gain "is noise rather than bias."** The direction
  is right, the magnitude is not: the per-pair contrast is `4·q^m·(1−q)`, which peaks at
  `q = m/(m+1)` (0.89 for eight bits) and is 16× weaker than one-hot's at initialisation. That claim
  was asserted without measurement; corrected in place.
- **Reverse learning is worse than neutral.** With `relearn_limit=3`, reach at step 1500 falls from
  0.65 to 0.39 (k=3) and 0.52 to 0.38 (k=9). It is driven from the observation's own wavefront —
  the edges that still work — so it rescues the living. Defaulting it off was right for the wrong
  reason.

### Instruments added

`calcConductionReach` and `calcReachByLabel` in `src/functions/score.py`. Accuracy hides this
failure completely; none of it was visible until conduction was watched directly, and the per-label
split is what makes it legible (the aggregate falls smoothly while whole labels drop to zero).

## Crystallization: the mechanism the per-edge learning rate was always for

`epsilon` has been a per-edge buffer since the prior engine, decayed by `epsilon_decay` on every
update of a fired edge. Three properties of the mechanism it was standing in for were missing, and
the prior engine at `app/src/registry/core.py:66` had exactly the same gap, so this was never built
rather than lost in the rewrite:

- **Outcome-blind.** An edge that had been consistently right and one consistently wrong cooled at
  the same rate.
- **One-directional.** Cooling never reversed, so a hardened section that went bad could not become
  plastic again — a second absorbing state, the same shape as an edge fallen mute.
- **Inert.** A half-life of 6,931 updates leaves an edge at 0.096-0.098 of a 0.1 start after 3,000
  steps. Measured, not estimated.

Now `calcPlasticityFactors`: cool on reinforcement, warm on correction, hold on no move, bounded to
`[epsilon_floor, epsilon_start]`. Off by default behind `crystallize`.

### Read per edge, because the obvious version is degenerate

Built first as the direct reading of the design — good outcome cools, bad outcome warms, driven by
the observation's advantage. Three tests failed and showed why it cannot work: **a mesh performing
steadily has an advantage centred on zero once its baseline catches up**, so a perfectly-performing
cluster crystallizes not at all. That is the same zero-drift trap this branch spent the day
diagnosing, reappearing inside the mechanism built to escape it.

Reading each edge's own move has no such degeneracy, and is better than the design rather than
equivalent to it: two clusters in one mesh harden independently, which a single scalar could not
express. The hysteresis falls out of the multiplicative form — undoing a 300-round record takes
comparably many reversals, and one bad result moves a settled mesh by under 2%.

`epsilon_floor` is load-bearing. An edge at zero learning rate can never change again whatever
happens to it, which is precisely the absorbing state the mechanism exists to avoid.

### It fixes the adaptation case

Task switch, `copy(bit 0)` then `copy(bit 3)`, 8 seeds, reach / accuracy:

| arm | pre-switch | +300 | recovered |
|---|---|---|---|
| off | 0.96 / 0.84 | 0.54 / 0.47 | 0.39 / **0.37** |
| on | 0.97 / 0.79 | **0.95** / 0.56 | 0.73 / **0.54** |

**Three of eight seeds die with it off; none die with it on.** That failure — a mesh ending at 0.00,
raising `SignalDidNotReachMotorsError` on every input — has recurred in every part of this branch's
work, and this is the first mechanism that removes rather than mitigates it. Conduction 300 steps
after the switch is 0.95 against 0.54. Two seeds fully relearn the new task (0.96, 1.00); the mean
of 0.54 against chance 0.51 says the mesh mostly *survives* the change rather than relearning it.

### On stationary tasks it is mixed, and nine labels regresses

| labels | off | on |
|---|---|---|
| 2 | 0.75, 5/8 learned | 0.75, 5/8 |
| 3 | 0.63, 3/8 | **0.68, 6/8** |
| 9 | **0.48, 1/8** | 0.33, 0/8 |

Three labels doubles the seeds that learn and lifts reach 0.66 to 0.83 — the first movement on the
bimodality all branch, and the property that matters most for stability, since the failure there is
a seed lottery rather than a low ceiling. Two labels unchanged.

**Nine labels regresses and is not explained.** The rate does fall to 0.071 there, so crystallization
is happening; why it costs accuracy is open. Recorded unexplained rather than narrated, given how
many mechanism guesses this branch has had overturned by the next measurement.

`epsilon_cool` at 0.97 over-crystallizes: conduction 0.93-1.00 and accuracy at chance, which is the
frozen-policy failure of the inherited `reward - confidence` rule reached by another road. 0.99 sits
near that edge, so the parameter is a trade-off and not a knob to turn up.

Default off until the nine-label regression is understood.

## There is no signature of a bad seed at initialization

A seed's outcome looks patterned — seed 3 reads 0.51 at every label count, seed 10 never clears
0.47 — so sixteen fresh meshes were measured on eight readings taken before any training and
correlated against their trained accuracy on `first-set` capped at three labels.

The 16-seed result was striking: path depth correlated **+0.651**, share of edges above the
conduction floor **−0.563**, mean strength **−0.484**, edge count **+0.433**. Screening at
`hops >= 5.5` kept every good seed and doubled the yield from 25% to 50%. A coherent story came
with it — deeper paths mean more edges to differentiate over and more redundancy against erosion.

**It does not survive out of sample.** Forty unseen seeds (17-56):

| reading | 16 seeds (fitted) | 40 unseen |
|---|---|---|
| hops | +0.651 | **+0.109** |
| conductive | −0.563 | **+0.232** |
| strength | −0.484 | **+0.293** |
| edges | +0.433 | −0.032 |
| bound | +0.009 | +0.083 |
| reach, rank, motor_in | 0.000 | 0.000 |

Every correlation collapsed and two flipped sign. Good seeds average 5.52 hops against 5.45 —
indistinguishable. The pattern was fitted noise, found on the same sample it was derived from.

**Three things this establishes.**

- **Every fresh mesh is structurally identical** on everything that could matter: conduction reach
  1.00, representation rank 32/32, edges into motors 72, across all 56 seeds. No mesh is born
  deprived, and no reading available at construction predicts what it will become. Screening for a
  good mesh before training is not possible, so it is not worth building.
- **The probe bound has no predictive power** — ~0 in both samples. It answers whether the
  information is present, never whether the mesh will extract it. Seed 5 holds the lowest bound of
  its sample (0.47) and is among its best performers; seed 2 holds the highest (0.78) and fails.
- **Bad seeds are a dynamics problem, not a wiring problem.** What separates meshes is which
  pathways erode first during training, which is the same conclusion the conduction work reached
  from the other side, and why crystallization — acting only on training dynamics and doing nothing
  at initialization — is the only mechanism that has moved the seed lottery.

**A calibration for every other measurement on this branch.** The learned-seed base rate was 4/16
in the first sample and 17/40 in the second (25% against 43%). Sixteen seeds cannot estimate the
rate, let alone correlate against it, and the four- and eight-seed tables quoted throughout this
log are far noisier than they read.

## Raw accuracy was the wrong metric, and it inflated two results before it was caught

Chasing whether conduction healing worked turned up something that reframes the branch's whole
measurement basis, and produced two corrections inside an hour — one of them to a claim made in
this log.

**Silence and error score identically.** A trained mesh both declines to answer and answers wrongly.
Split apart on `first-set` at three labels, the shipped mesh has coverage 0.59 and **conditional
accuracy 0.96** — it answers 59% of its inputs and is right on nearly all of them. Read as one
number that is 0.57 and looks like a mesh converged near chance.

**That reading was too generous, and the correction is the more important half.** The answered set
is chosen by the mesh, and its silences fall on minority labels, so what it keeps is **83% a single
label** at three labels and 81% at nine. Guessing that label scores 0.96 with no skill at all.
Against the honest baseline — the majority share *of the answered set* — the lift is:

| arm | k=3 lift | seeds beating own baseline | k=9 lift | seeds |
|---|---|---|---|---|
| shipped | +0.13 | 18/39 | +0.05 | 9/38 |
| crystallize | +0.11 | 21/40 | **−0.07** | 7/39 |
| init above floor | +0.15 | 19/40 | +0.04 | 8/40 |
| **init + heal** | **+0.22** | **28/39** | +0.02 | 9/40 |

So the mesh does not "learn nearly perfectly and abstain" — it narrows to a majority-dominated
subset and largely guesses within it, which is the original conduction-collapse reading rather than
the reprieve it briefly looked like. `crystallize` at nine labels scores *below* what guessing the
majority of its own answered set would.

**Init-above-floor plus conductance healing survives the corrected metric**, which is the one
mechanism this session that does. Coverage 0.59 → 0.73, and the answered set falls from 83% to
**70%** one label — a *harder* baseline — while lift rises +0.13 → +0.22 and seeds beating their own
baseline go 18/39 → 28/39. Answering a less skewed subset better is the opposite of a coverage
artifact. Nothing moves nine labels: +0.05 → +0.02.

`calcScoreProfile` and `ScoreProfile` land as the form results are quoted in from here.

### The ceiling is mostly representational, which reframes the session's nine mechanisms

The matched probe bound on a *fresh* mesh for the three-label task averages **0.65** across 56
seeds. The mesh reaches 0.57 raw. **It is already extracting 88% of what a matched linear readout
could get from the representation**, against a 0.51 majority baseline.

Nine mechanisms were tried this session — credit seeding, supervision, codes, baselines,
exploration, crystallization, healing, initialization, pruning — and every one targeted the
learning rule or the mesh's dynamics. If the terminus activation supports only 0.65, none of them
could have worked, and their small inconsistent gains are what tuning an extractor against a low
ceiling looks like. The one large gain this branch ever recorded was per-row lanes, which took the
probe from rank 8/33 to full and moved copy 0.37 → 0.87: a **representation** change.

Full rank with a low bound is the specific thing to explain — the dimensions are present and the
structure the task needs is not being formed in them.

---

## feat-169 — Both remedies become defaults, and the vocabulary is sized

### The defaults

`strength_init_lower` 0.4 → **0.55** and `conductance_healing` False → **True**. The first keeps
the initial range above the single-hop conduction floor of 0.527, so no edge is born mute; the
shipped 0.4 put roughly a quarter of every fresh mesh's edges in the dead band before a single
observation. The second scales a neuron's out-edges proportionally back over the floor after each
update, so learning can weaken an edge but cannot delete it from the mesh's behaviour.

Gain at three labels, cost nowhere. Three labels: 13/40 seeds learning → 23/40, coverage
0.59 → 0.73, lift +0.13 → +0.22. Two labels through the full trainer loop over 24 seeds: mean
reward 0.71 → 0.70, conditional 0.77 → 0.77, lift +0.24 → +0.24, coverage 0.97 → 0.99. Nine labels
unmoved. Neither mechanism touches the representational ceiling, and neither is claimed to.

### Two gate tests moved, and one of them was measuring its seed

`test_flowRecoverSignal_doesRaiseWhenRecoveryIsRefused` zeroes every strength and expects silence.
With healing on the mesh recovers: reverse learning lifts the strengths a little and healing scales
that back over the floor. That is the mechanism working, so the test now turns healing off rather
than being weakened.

`test_flowTraining_doesLearnWhileRestructuring` ran one seed against a 0.6 threshold and went
0.92 → 0.49. It was not a regression. Across twelve seeds at that exact configuration the arms are
indistinguishable — mean 0.70 → 0.68, and **6/12 seeds clear 0.6 under each** — so a single-seed
threshold there passes or fails on which seed it was written against. It now averages six seeds,
which is what the claim ("the full loop beats chance") actually says.

### Sizing a million-label vocabulary

Four obstacles, and only three are about the label count.

**The label count is the cheap axis.** With a code that corrects 15 errors, a million labels needs
141 bits — **282 motors**. Nothing in the architecture strains at that.

**`calcCodeWidth`'s redundancy never becomes distance.** The books `makeRandomCode` returns have
minimum distance **1 at every size measured** — 16, 64, 256, 1,024, and 4,096 labels — so the spare
bits correct nothing. That is what the width asks for, not a bad draw: `k` random `m`-bit words
expect `C(k,2)·(m+1)/2^m` pairs at distance ≤ 1, which exceeds one at `m = 2·log2(k)` above a
handful of labels. The docstring's claim that spare bits "let a codeword be decoded to its nearest
neighbour instead" has never been true in a shipped configuration.

**Construction is quadratic.** `calcCodeDistance` materializes a `k × k × m` comparison per try:
7.7 s and 4·10⁸ bytes at 4,096 labels, ~2·10¹⁴ bytes at a million. The book cannot be built past a
few thousand labels whatever its width.

**The label set is frozen once bound.** Adding one label regenerates the whole book — 0/8 original
codewords survive at 8 → 9, and the width changes with it — so `setCodedLabels` correctly refuses
and a coded decoder can never take a label it has not already seen. Only one-hot grows. **The mode
that scales cannot grow, and the mode that grows cannot scale.**

### The obstacle that is not about labels

Seventeen labels, 10-bit random code, 8 seeds, defaults on: five seeds answer nothing, and the
three that answer emit **one or two distinct bit patterns across 400 held-out inputs**. Wrong-bit
variance is 0.00–0.10 where independent errors would give 1.7–2.4. The bits are constant, not
noisy. Their conditional accuracy of 1.00 is the answered-set confound at its most extreme — a
fixed pattern is right about exactly the inputs whose label it decodes to.

So there is no per-bit accuracy to size a code against, and every width calculation above is
premature arithmetic.

The mechanism, measured: **nine live edges in ten fire on every observation**, flat at 0.89–0.91 as
the mesh grows 96 → 1,536 neurons. Every input uses the whole mesh. Cost is therefore the mesh
rather than the input (2.11 → 8.45 ms as edges go 998 → 14,227); no capacity is allocated per
label, so every label's learning overwrites every other's; and growth adds substrate every input
immediately consumes rather than a region a failing input can move into.

**This is a mechanism for the label cliff that does not involve the learning rule**, and it
retrodicts the session: nine rule-level mechanisms each bought little, which is what tuning an
extractor against a shared undifferentiated substrate looks like.

The tension is that firing less is what capacity needs and firing less is what stops signal
arriving — a path's value only ever multiplies by strengths below one, and convergence is what
keeps it alive. Fan-out sparsity was measured against that economy twice and lost both times.
Changing the economy, so a neuron's arriving value does not depend on how many edges fed it, is
what would make sparsity affordable. It is untried, and it is the same per-neuron normalisation
already listed as a remaining candidate for the conduction collapse.

---

### The mesh amplifies, and the neuron gate has never once refused anything

Written while building the instruments that ROADMAP.md's next step is judged by, and it changed
what that step is for. Measured on a fresh mesh, `nexus_size=64`, `terminus_size=32`:

| quantity | measured |
|---|---|
| interneuron fan-in | mean 10.06, sd 6.53, range 4-66 |
| mean \|strength\| | 0.715 |
| frontier median by hop, seed 5 | 1.34 → 2.08 → 4.35 → 3.14 → 0.89 |
| peak frontier over first hop | 3.1-5.2 over 3 seeds |
| median arrival across interneurons | 4.6-6.5 over 3 seeds |
| `neuron_firing_threshold` | 0.5 |
| signal band ceiling | 0.9 |

This file and specs/propagation.md have both stated the open problem as "a path's value only ever
multiplies by strengths below one and convergence is what keeps it alive". **That is true per edge
and false per neuron.** A neuron sums roughly ten arrivals of mean strength 0.7, so it gains about
sevenfold where each of its edges lost, and the frontier climbs to five times the band it started
in.

Two consequences, and both re-read earlier entries rather than adding to them.

**The neuron gate is dead code.** Arrivals sit nine to thirteen times above the threshold meant to
refuse them. Nine edges in ten fire not because the gate is permissive but because nothing is ever
refused and propagation ends by exhausting the fire-once guard. The defaults file called 0.5 "far
too permissive" and needing "recalibration upward"; that understated it — a threshold that never
binds is not mis-set, it is absent.

**Fan-in is the mesh's only amplifier, which is why sparsity killed conduction both times.** The
fan-out experiments did not remove a compensation for a lossy path. They removed the gain. That is
also why edge activation went bimodal with no stable middle: an amplifier with a hard threshold on
its output has two fixed points and no third.

So the remedy is not to make signal cheaper to carry, which is what "changing the economy" was
taken to mean. It is to stop taking gain from topology and set the level explicitly.

### Conditionality, and what share alone could never say

`calcFiringShare`, `calcFiringProfile` and `calcArrivalProfile` ship as instruments. The share was
quoted here at 0.89-0.91 and no function computed it, so it was produced by hand and could not be
regression-tested.

The new quantity is **conditionality**: same-label minus different-label Jaccard of the fired-edge
sets. On a fresh mesh it reads **+0.022 to +0.028 over 3 seeds against a share of 0.87-0.89**.
Nearly nine edges in ten fire for every input, and almost none of that is specific to the input.

That is the direct statement of "every input uses the whole mesh", and it is the measurement the
two fan-out experiments lacked. They could report that sparsity did not raise the bound without
being able to say why, because a share cannot distinguish a mesh that fires few edges *per input*
from one that fires the same few edges *for every* input. Only the second is capacity allocation,
and it is what a rank has to produce.

`depth_gain` is peak-over-first rather than a mean of per-hop ratios. The mean conflates the climb
with the die-out and reports 1.27 for a mesh whose frontier tripled; that version was built first
and measured before it was kept.

### The signal economy, and the three things built wrong on the way

`signal_economy`, default off. A strength becomes a share of what its neuron emits, an arrival is
divided by a **static** incoming budget, and the geometric-mean correction is dropped because it
corrects a decay the mesh no longer applies. Computed per pass and stored nowhere, so
`mesh.strength` is never rewritten, no checkpoint changes, and the learning, plasticity and
structure suites are untouched by construction.

Static is the whole distinction. A denominator counting the edges that *actually fired* is the
fan-in mean, and two features arriving would then produce what one does — the collapse the
per-observation lane was introduced to prevent. Static leaves that intact and removes only the
topological advantage of a well-connected neuron. **Capacity is normalised away; coincidence is
kept.** The roadmap's "arriving value does not depend on fan-in" and this file's "interneurons
genuinely accumulate fan-in" are about different senses of the word and are not in conflict.

An edge can no longer be driven mute, because a share is a ratio: scaling every out-edge of a
neuron down by any factor leaves what it carries unchanged. Gap 1's absorbing state becomes
unrepresentable rather than mitigated, so `calcDeadBand` returns `None` and `conductance_healing`
is refused alongside the economy instead of silently becoming a no-op.

Three defects, each of which measured as working:

1. **The gain control ran after the neuron gate.** It measured identical to no gain at all, for a
   reason that is obvious afterwards — rescaling what already passed cannot restore the level that
   is the reason nothing passed.
2. **The rank ranked a different population than the reference did**, and the path statistics
   divided by the normalised arrival where the reference divided by the raw sum. Both put the two
   engines out of step by a factor of the incoming budget, and **neither would have failed the
   lockstep test**, because `slog` and `plen` are unused once the correction is dropped. Found by
   reading, not by testing.
3. **The incoming budget was clamped up to a minimum of one.** Shares average roughly the
   reciprocal of a neuron's fan-out, so a real budget is usually *below* one: 64.6% of interneurons
   sat under the clamp, the median true budget is 0.624, and the least-connected were divided by up
   to fourteen times too much. The clamp did the opposite of the economy's purpose, penalising
   exactly the neurons a fan-in normalisation exists to protect.

The third inverted the published result, so a table was committed and corrected one commit later.
8 seeds, fresh meshes, 30 held-out rows, shipped hyperparameters otherwise:

| arm | fan-in corr | cv | depth gain | median arrival | share | conditionality | reach |
|---|---|---|---|---|---|---|---|
| shipped | +0.672 | 0.94 | 4.52 | 6.085 | 0.889 | +0.0235 | 1.00 |
| economy, no gain | -0.288 | 1.13 | 0.50 | 0.046 | 0.243 | +0.0225 | **0.00** |
| economy + gain | **-0.046** | 0.48 | 1.38 | **0.511** | 0.897 | +0.0105 | 1.00 |
| clamped budget (wrong) | -0.191 | 1.01 | — | 0.024 | 0.243 | +0.0223 | 0.00 |

**`neuron_firing_threshold` needs no change**, which reverses what the first version of this entry
concluded. This file has recorded since the tensor rewrite that the value "remains at its inherited
value pending the topology work below". This is that work, and 0.5 turns out to be right: the
threshold was never wrong, the arrivals it judged were. It refused nothing at a median of 6.085 and
begins to select at 0.511.

The economy is not a sparsifier and firing share is unchanged at 0.897, which is the predicted
result rather than a disappointing one — removing the amplifier also removes the reason the
wavefront died out. Conditionality *falls*, because nothing yet selects per input; that is
`firing_fraction`'s job and it is measured separately. One reading is still short: `arrival_cv` at
0.48 is half what it was but wider than a rank wants.

Lockstep is parametrized over shipped, economy, ranked and gained at 6 seeds each — 24 equivalence
assertions rather than 6 — because the economy changes what an edge carries, what a neuron divides
by, which neurons emit, and at what level, and those are four places the two engines could drift
apart independently.

### The conduction collapse is fixed, and the label cliff did not move

The result the economy was built to produce, and the one that matters is the second half. 8 seeds,
`first-set` capped at `k`, 1,500 steps, 200 held-out rows, no growth or pruning:

| k | arm | reach | coverage | conditional | baseline | lift | seeds learning |
|---|---|---|---|---|---|---|---|
| 3 | shipped | 0.64 | 0.64 | 0.97 | 0.85 | **+0.118** | 3/8 |
| 3 | economy | **1.00** | 1.00 | 0.44 | 0.51 | -0.077 | 1/8 |
| 9 | shipped | 0.78 | 0.78 | 0.68 | 0.67 | +0.010 | 2/8 |
| 9 | economy | **1.00** | 1.00 | 0.16 | 0.51 | **-0.356** | 0/8 |

Reach 1.00 at nine labels, flat at every checkpoint, against a shipped mesh that erodes to 0.78. No
mechanism on this branch has come close. And every accuracy reading moves the wrong way; at nine
labels the mesh answers every input at 0.16 against a majority share of 0.51 — by the standard this
file already set for `epsilon_start`, that is confidently learning a wrong rule rather than
declining to learn.

**So ROADMAP.md's ordering is wrong.** It calls gap 1 "the root failure — most of what follows is
downstream of it". Gap 1 is now solved, structurally and completely, and closing it moved accuracy
down at both label counts. **Conduction was never what bounded accuracy.**

**Why, and this is the part worth keeping.** The erosion was doing two jobs. It destroyed reach, and
it *differentiated*: a shipped mesh answers 64% of inputs at three labels and 85% of what it keeps
is one label, so the surviving set is selected — badly, but not randomly. The economy answers
everything and its answered set is balanced at 0.51. Removing the collapse removed the accidental
differentiation with it, and nothing replaced it. The mesh that conducts perfectly is the mesh with
nothing to say.

**Deliberate differentiation does not replace it either.** `firing_fraction` over the economy takes
the firing share from 0.894 to 0.333 while reach holds at 0.88-0.97 — **sparsity no longer costs
conduction**, which is genuinely new, since both fan-out experiments lost the mesh's answers the
moment they fired less. But conditionality only doubles, +0.021 to +0.050 at best, and does not rise
as the mesh fires less: 0.039, 0.030, 0.037, 0.050, 0.042 across fractions 0.05 to 0.60 is noise
rather than a trend. The winner set is largely the same one whatever the input.

A rank can only select among distinctions the representation already carries, and the measurement
says it carries almost none. That is the same conclusion the probe reached from the other side —
the ceiling is representational — arrived at now from firing rather than from a readout.

**Both mechanisms ship default-off with their numbers**, in the disposition `crystallize` and
`CodeBook` already have: reachable, tested, comparable, and re-testable against a better learning
rule rather than re-derived. Neither is a candidate for a default under the current rule.

**Left untried and named so it is not re-derived:** per-neuron adaptive thresholds with a target
firing rate, so a neuron that keeps winning raises its own bar and usage is forced to spread. It
needs persistent per-neuron state and a checkpoint bump, and it is the one remaining idea aimed
squarely at an input-independent winner set.

**A fifth defect, found by the result rather than by a test.** `calcNeuronCredit` spreads the
teaching signal backward "attenuated by the same strengths the forward pass used" — its own
docstring, and a real invariant. Under the economy the forward pass carries shares and this still
read `mesh.strength`, so credit travelled along weights no signal took. Fixed before the table
above was taken. The pre-fix run reported reach 1.00 with lift -0.318 at nine labels; correcting it
improved conduction further and left the learning verdict unchanged, so the trade-off is real and
not an artefact of the defect.

**A correction to this file and to ROADMAP.md.** Gap 1's headline table — reach falling to 0.51 by
step 1000 at nine labels — predates `strength_init_lower` 0.55 and `conductance_healing`. Shipped
today it is 0.88 at step 1000 and 0.78 at 1500 over 8 seeds; reverting both defaults reproduces
0.62, so the mitigations account for the gap and the residual is the four-seed sample. That also
refutes the claim that the pair leaves nine labels "unmoved" — true for accuracy, which is what it
was checked on, false for conduction, where it is worth +0.26 reach. The section making that claim
is the one arguing at length that accuracy cannot see this failure.

### Sparse wiring does raise the bound, and both earlier measurements were taken on tasks that could not show it

`sensor_fanout` was measured twice, before and after the representation fix, and recorded both
times as not raising the probe bound. Both were measured on `copy` and `majority` — the two tasks
specs/growth.md separately records as **saturated at initialization**, carrying 0.80-0.82 of the
available information before any training. That is the same reason growth could never be
demonstrated on them: a task with no capacity limit cannot show a capacity gain. The experiment
was run twice on the instruments that structurally could not reveal the effect.

On `first-set` capped at 3 labels, the one task built because its ceiling moves with mesh size,
matched probe bound on fresh meshes, **72 paired seeds**:

| sensor fan-out | median | mean | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| 8 | **0.736** | 0.717 | 0.556 | 0.847 | 0.444 | 0.944 |
| 48 (shipped) | 0.667 | 0.664 | 0.556 | 0.806 | 0.472 | 0.861 |

Fan-out 8 beats the shipped 48 on **49 of 72 paired seeds, 3.1σ**, median delta +0.056. Found by a
6-seed grid over fan-outs 4/8/16/32/48 in which 48 was the worst cell in both the shipped and the
economy arm, 8 of 8 sparser cells better.

**Recorded with the shrinkage, because it is the fifth instance on this branch.** At 24 seeds the
same contrast read +0.097 median on 16/24 seeds — a one-sided p of 0.08, which is not significant,
and an effect nearly double the truth. The direction survived and the magnitude did not. The
threshold for calling it real was fixed at 2.5σ before the 72-seed run, precisely because the
contrast had already been looked at twice and liked twice.

**The seed lottery dominates it.** Fan-out 8 spans 0.444 to 0.944 across seeds against a median
gap of 0.056, so this is a shift of a wide distribution and not a per-seed gain. Any future claim
about the probe bound on this mesh needs samples of this size; the 6-seed grid that started this
had cells differing by less than their spread.

**The default is unchanged.** The bound is measured on an untrained mesh, so it says the
representation supports more, not that learning extracts more — and this session has just finished
demonstrating that those come apart, since `signal_economy` raises conduction to 1.00 and drives
lift negative. A trained sweep on the `k` ladder is owed before `SENSOR_FANOUT` moves.
