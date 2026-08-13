# Haze Roadmap

What is not built yet, why it matters, and what it depends on. Built work is recorded in
[LOG.md](LOG.md); this file holds only what remains.

The goal this serves: Haze takes in sound, text, and images together — whatever context is
available — distils them for relevance, accumulates that context into an opinion, and produces
sound, text, or images back. Arrival *timing* should matter: an image reaching the sensors while a
sound is still being processed should not be equivalent to both arriving at once.

Three architectural gaps stand between the current model and that goal. They are independent of
each other and are listed in the order they should be taken.

---

## 1. Large output spaces — the label cliff

**Status:** attempted and unresolved. `CodeBook` is built, tested, and does not work; the barrier
is now measured rather than guessed. See [specs/decoding.md](specs/decoding.md).

Haze decides between two labels well (0.94–0.98 held-out) and between three or more barely at all
— the cliff is at the third label, and by nine it is below chance. A larger mesh makes it worse.
The cause is that a scalar reward cannot say *which* of `k−1` rivals should have won, so the
corrective signal is diluted by `1/k`. Measurements in [specs/learning.md](specs/learning.md).

Every output modality in the goal above is a large label space, so nothing else in this roadmap is
reachable while this holds.

**The change:** a label stops owning a motor. Motors are allocated in **pairs, one pair per bit of
a code**, each pair a two-way race — the regime that demonstrably works. A label becomes a
codeword across those bits. `k` labels cost `O(log k)` motors instead of `k`, and each bit receives
the full binary correction instead of a `1/k` share.

**What happened:** the risk was real but was not the binding one. A codeword is a conjunction, so
joint accuracy is the *product* of the per-bit accuracies — nine labels over eight bits needs every
bit right 0.92 of the time before the whole answer beats a constant guess. One-hot has no such
threshold, because one motor winning is a common event. Exploration by whole codeword, by single
bit, and per-bit independently all fail above three labels, and so does replacing the scalar reward
with per-bit supervision, which is the most a teacher could give.

**What is still untried:** a code with real Hamming distance *and* individually easy bits, so
residual per-bit errors are corrected rather than multiplied. Thermometer gives easy bits and no
distance; random codes give distance and harder bits. Nothing yet gives both. Raising per-bit
accuracy is the other half of the same problem and is the same work as the majority readout gap
below.

The alternative reading is that the reduction is sound and the mesh is simply not yet accurate
enough per decision for any of it to pay — in which case the readout gap is the thing to fix first
and this becomes viable on its own.

---

## 2. Residual mesh state — making time matter

**Status:** not started. Depends on nothing, but is far easier to evaluate after (1).

`flowSignalPass` allocates fresh state on every call and runs to quiescence, so **every
observation starts from silence and arrival order is currently unobservable**. An image arriving
mid-sound is, today, exactly the same as the two arriving together.

**The change:** node activation persists between observations and decays, so a mesh still ringing
from one input responds differently to the next. That makes the propagation contract stateful,
which touches reproducibility (a checkpoint must carry the residue), termination (the wavefront
guard assumes a fresh `fired` mask per pass), and learning (the fired trace would span
observations).

**Open question:** what decays, how fast, and whether the decay is a hyperparameter or a per-neuron
learned quantity. A learned time constant is the more interesting answer and the harder one.

---

## 3. Context accumulation — the opinion

**Status:** not started. Partially mistaken for built, so stated precisely.

`observe()` begins with `self._observations = {}`. It **replaces** rather than accumulates: two
successive calls do not build on each other, the second discards the first. Context does accumulate
across *modalities within a single call* — `observe({"sound": …, "image": …})` — and that multi-key
path is the seed of the mechanism, but accumulation across calls does not exist.

**The change:** an observation joins a running context rather than replacing it, with an explicit
way to close or reset it. This is what turns a sequence of inputs into a position rather than a
series of independent reactions.

**Interacts with (2):** residual mesh state and accumulated observations are two different memories
— one in the mesh, one in the model — and the design should say which holds what rather than
letting them overlap by accident.

---

## Supporting work

### Multi-modal encoders
Real `HashTextEncoder`, `PatchImageEncoder`, and `SeriesEncoder`, plus the decoders that emit each
modality back. Blocked behind (1): every one of them implies a large label space.

### The majority readout gap
0.65 achieved against a probe bound of 0.81, and unlike copy the representation does not move
during training — so the information is present and the rule is not extracting it. The one clean
learning-rule gap left on the binary suite. Independent of everything above.

### Demonstrating growth
Growth, pruning, the auditor, and the trainer are built and tested but have never been shown to
help, because every task in the suite is saturated at initialization. `first-set` is genuinely
capacity-limited — probe bound 0.50 at a terminus of 4, 0.65 at 32 — but needs nine labels, so it
is blocked behind (1). Growth remains an unvalidated mechanism until then.

`growth_threshold` is also an absolute error rate, which makes it task-dependent: 0.6 is above
chance for a two-label task and below it for a uniform nine-label one. A trigger relative to what
the mesh has recently achieved would not need retuning per task.

### Honest benchmark
The speedup figure must be re-measured with the prior engine's per-edge JSON writes and its
`print` of the strength array disabled, so the headline is not "we stopped writing 40,000 files per
row". Owed before any number reaches the README.

### Packaging and publication
Delete `app/`, rewrite the README (it still documents a deleted layer), write the model card, and
push to the Hub. The packaging itself is built and round-trips bit-identically; this is the
presentation around it.
