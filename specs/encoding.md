# Encoding

How data of any modality becomes signal a mesh can carry.

## Description

An encoder turns one observation — a row of numbers, a paragraph, an image, a window of a time
series — into a fixed-width vector of floats that seeds the sensor neurons. It is the only place
modality is understood; downstream of it the mesh sees floats and nothing else. This is what makes
Haze multi-modal without the mesh knowing anything about modality.

## Policies

### Every encoder emits a fixed-width vector in the signal band

**Given** an observation of any modality
**When** an encoder encodes it
**Then** the result is a float vector of the encoder's declared width, every element within
`[0.1, 0.9]`.

The band is a hard contract, not a convention. `signal_threshold`, `neuron_firing_threshold`, and
the geometric-mean correction all assume it. A value outside it either fails to propagate at all or
saturates the mesh, and both failures are silent.

### An out-of-band value is refused, not clipped

**Given** an encoder returning a value outside the signal band
**When** the value is injected
**Then** it raises `SignalRangeError`.

Clipping would hide a broken encoder behind plausible-looking output.

### Encoders are frozen

**Given** an encoder wrapping model weights
**When** the mesh learns
**Then** the encoder's weights do not change.

Haze has no backpropagation, so there is no gradient with which to train an encoder. A trainable
encoder with no loss is a false promise. Encoders are nonetheless torch modules, so that device
placement and weight round-tripping come for free, and a reward hook exists as a no-op seam should
forward-only sensor-layer learning ever be pursued.

### A default encoder for each modality carries no external dependency

**Given** a caller with no optional extras installed
**When** they encode text, an image, or a series
**Then** a working encoder is available.

Hashing-trick text, patch-mean image, and flattened-window series encoders are deterministic and
depend on nothing beyond the base install, so the claim that Haze handles a modality is true
without a model download. Pretrained backbones are optional extras that raise fidelity, not the
price of admission.

### Pretrained encoder weights are not bundled into a checkpoint by default

**Given** an encoder wrapping a pretrained backbone
**When** the model is checkpointed
**Then** only the reference to that backbone is written, unless bundling is explicitly requested.

A mesh checkpoint is a few megabytes and is written frequently during continuous learning. Folding
a backbone an order of magnitude larger into every one destroys that. Bundling exists for
air-gapped deployment, where the tradeoff reverses.

### Feature width may grow; it may not be reinterpreted

**Given** an encoder that begins presenting more features than before
**When** it observes
**Then** the additional features claim new sensors, and the existing features keep the sensors they
had.

### Normalization scope is declared, not assumed

**Given** a numeric encoder
**When** it normalizes
**Then** its scope is explicit — per-row, or against running statistics.

Per-row min-max is scale-destroying: a row of all ones and a row of all zeros encode identically,
because both are constant. It is the prior engine's behavior and remains available, but it is a
choice a caller makes rather than a default that goes unnoticed.
