# Decoding

How motor activation becomes an answer, and how sure the model is of it.

## Description

A decoder reads the accumulated activation of the motor neurons it owns and turns that vector into
an answer — a chosen label, a number, a set, a sequence. It also reports a confidence, which is not
decoration: confidence is an input to learning and to the decision to grow.

## Policies

### A decoder reads only its own active motors

**Given** a decoder owning a set of motors, some inactive
**When** it decodes
**Then** it considers only the active ones.

### Confidence is the peakedness of the motor distribution

**Given** the activation of a decoder's active motors
**When** confidence is computed
**Then** it is one minus the normalized entropy of that activation treated as a distribution — near
one when a single motor dominates, near zero when activation is spread evenly.

### Confidence is computed over the same motors used to decode

**Given** a decoder whose label space is wider than its currently active set
**When** confidence is computed
**Then** the inactive labels are excluded.

The prior engine took entropy over all motors while decoding from active ones, deflating confidence
by an arbitrary factor whenever the label space narrowed. Confidence feeds the learning delta and
the growth trigger, so this was not cosmetic.

### Confidence is defined at the degenerate cases

**Given** a decoder with exactly one active label, or with no motor activation at all
**When** confidence is computed
**Then** it is defined and finite rather than a division by zero.

Maximum entropy over one label is zero, and total activation of zero has no distribution. Both
occur in ordinary use — a single-label decoder, and a mesh that has not yet connected to its motors.

### Normalization is numerically safe at any activation magnitude

**Given** motor activations of arbitrary magnitude
**When** a decoder normalizes them into a distribution
**Then** the computation does not overflow.

Motor activation is a sum with no upper bound, and a naive exponential overflows. The prior engine
raised `OverflowError` from the softmax decoder once accumulated state grew large enough — which,
given that motor state was never reset, was a matter of running long enough.

### An answer with no signal behind it is an error, not a guess

**Given** a prediction in which no active motor received any signal
**When** the decoder is asked for an answer
**Then** it raises rather than returning an arbitrary label.

This is the signal that drives reverse learning: the mesh has failed to connect its sensors to its
motors and needs a path opened, which is a different situation from having answered wrongly.

### A label's type survives a round trip

**Given** a decoder whose labels are integers, strings, or vectors
**When** the model is checkpointed and reloaded
**Then** the labels are restored with their original types.

Decoders distinguish an integer label from its string form, and JSON does not.

## Codebook decoding, and why it does not yet work

The one-motor-per-label contract asks the reward a question it cannot answer. Told only that its
answer was wrong, learning cannot know which of `k − 1` rivals should have won, so the correction
is divided among them and the share reaching the right one falls as `1/k`. Measured, that puts a
two-label decision at 0.94–0.98 and a nine-label one below chance, and a *larger* mesh makes it
worse. See specs/learning.md.

`CodeBook` is the attempt to remove that. A label stops owning a motor and owns a **codeword**;
motors are allocated in pairs, one pair per bit, each pair a two-way race — the regime the rule
handles exactly. `k` labels cost `O(log k)` motors instead of `k`.

It is built, tested, and **does not beat one-hot at any label count above two.** Held-out accuracy
on one task coarsened only in its label count, chance held at 0.51:

| labels | one-hot | codebook (thermometer) | codebook (random) |
|---|---|---|---|
| 2 | 0.86 | **0.96** | 0.52 |
| 3 | 0.59 | 0.52 | 0.38 |
| 4 | 0.58 | 0.27 | 0.29 |
| 6 | 0.57 | 0.00 | 0.26 |
| 9 | 0.46 | 0.12 | 0.00 |

### Why: a codeword is a conjunction, and accuracy multiplies

An answer is right only when **every** bit is right, so joint accuracy is the product of the
per-bit accuracies. At `m` bits and per-bit accuracy `p`, the label is correct `p^m` of the time,
and the bar to clear is the majority label's 0.51:

| labels | bits | per-bit accuracy needed |
|---|---|---|
| 3 | 2 | 0.71 |
| 6 | 5 | 0.87 |
| 9 | 8 | **0.92** |

Every bit must be near-perfect before the whole answer is better than a constant guess. One-hot has
no such threshold: one motor winning is a common event, so it collects reward from the first
observation.

The `0.00` entries are the same effect at its sharpest. The majority label's thermometer codeword
is all zeros, so producing it needs `m` independent pair-races to *all* land the same way, which
happens about `2^-m` of the time by chance. The most frequent answer became the least reachable
one.

### What was tried

- **Exploration by whole codeword, by one random bit, and per-bit independently**, at two rates.
  None reaches chance above `k = 3`. Whole-codeword exploration is nonetheless the right default of
  the three: with a redundant code, flipping a single bit still decodes to the *same label*, so
  bit-level exploration cannot change the answer at all and reinstates the dead zone exploration
  exists to prevent — 0.27 against 0.91 on a two-label task.
- **Thermometer and random codes.** Thermometer wins where it applies, because every bit is a
  threshold; random codes buy Hamming distance but spend it on bits that are harder to learn.
- **Per-bit supervision**, replacing the shared scalar with each pair's own target bit. This is the
  strongest form — it is the most information a teacher could give — and it still lands at 0.30 for
  nine labels. It lifts the floor (seeds spread rather than pinning at 0.00) without clearing the
  multiplicative bar.

So the barrier is not credit assignment, and not the teaching signal. It is that per-bit accuracy
is not high enough for the product to survive, and thermometer's distance-1 code offers no error
correction to offset it. A code with real distance would correct the residual errors, but its bits
are harder, which lowers `p` — and that trade was not resolved.

`CodeBook` ships available and non-default. One-hot remains what a decoder uses unless asked
otherwise, and the two stay comparable so this result can be re-tested against a better rule.
