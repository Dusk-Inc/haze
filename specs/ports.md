# Ports

How encoders and decoders bind to the neurons they own.

## Description

A **port** is one encoder or decoder registered with a model under a caller-chosen key, holding a
claim on a set of sensor or motor neurons. Ports are what make encoder and decoder assignment
dynamic: one is registered at any point in a model's life, allocates neurons when it first needs
them, and releases nothing.

## Policies

### A port is identified by a caller-supplied key, never by its class or type

**Given** two encoders of the same class registered under different keys
**When** each observes
**Then** each drives its own disjoint set of sensor neurons.

This is the central constraint. The prior engine keyed its encoder-to-neuron registry by an
`EncoderType` enum, so two `NumericEncoder` instances shared one sensor pool — and the registry
raised `More than one instance of the encoder or decoder was found` in the case it half-anticipated.
Genuine multi-modal use was impossible: two numeric feeds, or a text encoder alongside a second
text encoder, could not coexist. A string key is also serializable, which the object identity it
replaces was not.

### Ownership is recorded on the neuron, not as a range

**Given** a port that has allocated neurons
**When** its neurons are looked up
**Then** the lookup is a mask over an owner vector.

Ranges are true only until the first free-slot reuse; after that a port's neurons are a scattered
set. Storing ownership per neuron keeps one source of truth and survives arbitrary growth.

### A port may be registered at any point in a model's life

**Given** a model that has already trained
**When** a new encoder is registered
**Then** it allocates a slot immediately and its sensor neurons on its first observation, wired
into the existing nexus, and no existing neuron or edge is disturbed.

### A binding, once made, is immutable

**Given** a sensor index assigned to a port's feature, or a motor index assigned to a decoder's
label
**When** the model continues to grow, prune, or checkpoint
**Then** that index continues to mean that feature or that label for the model's life.

Everything the mesh has learned about a feature or a label is held in the strengths of the edges
touching its neuron. Reassigning the index reassigns the learning.

### A decoder's active label set is a mask, not a reallocation

**Given** a decoder whose label set narrows between calls
**When** it predicts
**Then** the absent labels' motors are marked inactive rather than deallocated, and become active
again if the label returns.

### An unknown port key is an error

**Given** an observation naming a key no port has registered
**When** it is dispatched
**Then** it raises `PortNotFoundError`.

Silently creating a port for an unrecognized key would turn a typo into a permanently allocated
region of the mesh.

### Port metadata round-trips; port indices are not restated in config

**Given** a checkpoint
**When** it is written
**Then** the config records each port's key, role, implementation, and construction parameters,
while which neurons it owns lives in the owner tensor.

On load the two are cross-checked, and a disagreement raises `CheckpointCorruptError` naming the
port. Silent index drift is the failure mode this design is most exposed to, so it is checked at
every load boundary.

## Chaining

Ports are composed into a **chain**: an ordered set of stages, each naming the encoders it feeds
and the decoders it reads, where one stage's output may become the next stage's input. The chain is
reached through a single `ChainPolicy` seam that answers one question — given the last output and
the step index, what stage runs next, or is the chain finished.

The only implementation is a static policy that walks a fixed list, reproducing the prior engine's
lexical chain. The seam exists so that a future policy can choose chain length and modality flow
from context — the README's "Automated Chaining" — without the surrounding code changing. No
learned policy is built.
