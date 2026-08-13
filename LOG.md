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
