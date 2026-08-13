# Packaging

How Haze is distributed, loaded, and made safe to load.

## Description

Haze is a torch model. It is constructed from a config, its state lives in registered buffers, and
it saves and loads through the standard pretrained-model interface. What makes it unusual is that
its shape is not fixed: the number of neurons and edges changes as it learns, and a loader must
allocate the right shape before it can load anything into it.

## Policies

### Capacity is configuration, and configuration is read first

**Given** a checkpoint of a mesh that has grown
**When** it is loaded
**Then** the config is read first and the model is constructed at the recorded capacity, and only
then is the tensor payload loaded into it.

Strict loading requires shapes to agree. Making capacity a config field means they agree by
construction. The alternative — resizing buffers during the load — is invisible behavior that
breaks assignment and meta-device loading, and is refused.

### All state is a buffer, never a parameter

**Given** the model's tensors
**When** they are registered
**Then** they are registered as buffers.

There are no gradients. Parameters would attract optimizers and gradient tracking, and would appear
in the model's parameter list where they would be meaningless to anyone reading it.

### Configuration holds metadata; tensors hold indices

**Given** a port's neurons
**When** the model is saved
**Then** the config records what the port is and how to reconstruct it, and the tensors record
which neurons it owns.

Index arithmetic in a config file is a claim that goes stale the moment a free slot is reused. See
[ports.md](ports.md).

### Labels are stored beside the config, not inside it

**Given** a decoder with a large or string-valued label set
**When** the model is saved
**Then** its labels, their motor indices, their value type, and their active flags are written to a
separate file.

An unbounded, string-valued map does not belong in a config of scalars, which is why vocabularies
are conventionally separate. The value type is recorded because a decoder distinguishes an integer
label from its string form and JSON does not.

### Every load boundary cross-checks labels against ownership

**Given** a checkpoint whose label file and owner tensor disagree
**When** it is loaded
**Then** it raises `CheckpointCorruptError` naming the port.

Index drift is this design's most dangerous failure because it produces a model that loads without
error and answers wrongly. It is checked wherever it could be introduced.

### A shape or format mismatch raises a named error

**Given** a truncated, bit-flipped, or mismatched checkpoint
**When** it is loaded
**Then** it raises a Haze error naming what disagreed, not a raw framework error.

### Loading an untrusted model does not execute untrusted code

**Given** a published model whose config names an encoder implementation
**When** it is loaded
**Then** only first-party implementations are constructed unless the caller explicitly opts in to
remote code.

A config field naming an import path is arbitrary code execution on load. The allow-list is in
place before anything is published, not after.

### Saved tensors do not share storage

**Given** buffers reallocated during growth
**When** they are saved
**Then** each is written as a contiguous copy.

Views created during reallocation share storage, and the tensor serialization format refuses that.

## Naming exemption

The workspace function taxonomy is `[Verb][Domain][Qualifier]`. Names mandated by torch and the
model hub interface — `forward`, `save_pretrained`, `from_pretrained`, `_save_pretrained`,
`_from_pretrained`, `state_dict`, `load_state_dict`, `train`, `eval`, `to` — are exempt, because
they are how the surrounding ecosystem calls into the model and renaming them would make Haze stop
being a torch model. This exemption is recorded so it is not later mistaken for an oversight.

## Package layout

The repository follows the workspace `src/{tokens,models,interfaces,modules,functions,errors,tests}`
convention on disk while installing under a single import root, so that `from haze import Haze`
works. The mapping is declared in `pyproject.toml`. The alternative used by other Python libraries
in this workspace — treating `src` as the package root — would install `functions`, `models`, and
`tokens` as top-level import names, which is a latent collision for an internal library and
disqualifying for a published one.

## Train and eval semantics

Haze has no backpropagation, so the standard mode switch is given an explicit meaning rather than
being inherited without one:

- **Eval** disables learning, records no fired trace at all, and freezes growth and pruning.
  Calling `learn` in eval raises rather than silently doing nothing.
- **Train** enables learning, records the trace, and permits growth and pruning.
