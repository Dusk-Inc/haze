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
