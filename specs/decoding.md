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
