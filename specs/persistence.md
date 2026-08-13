# Persistence

How a mesh is checkpointed, restored, and audited across a long run.

## Description

Haze learns continuously, so persistence is not a thing that happens once at the end. A run writes
whole-model checkpoints on a policy and an append-only journal of structural events as it goes. The
checkpoint format is the same format the model is published in — a local checkpoint directory and a
published model repository are byte-identical.

## Policies

### The checkpoint format and the published format are the same

**Given** a model saved locally and the same model published
**When** the two directories are compared
**Then** they are identical.

One writer and one format removes the entire class of defect where what is tested locally differs
from what is published. See [packaging.md](packaging.md) for the format itself.

### A checkpoint is whole, not incremental

**Given** a checkpoint policy firing during a run
**When** it writes
**Then** it writes the complete model.

A mesh of half a million edges is a few megabytes; a full write is milliseconds. Delta encoding
earns its keep several orders of magnitude beyond that, and the seam for it exists unimplemented so
the decision can be revisited with measurements rather than guesses.

### A checkpoint write is atomic

**Given** a process that dies mid-write
**When** the run is resumed
**Then** it finds either the previous complete checkpoint or the new complete one, never a partial.

Writes go to a temporary directory and are renamed into place; the manifest pointing at the latest
is rewritten last.

### Publishing is not checkpointing

**Given** a run checkpointing frequently
**When** the model is published
**Then** publication happens at run boundaries and milestones only.

Every publish is a version-controlled commit whose binary payload is retained indefinitely.
Publishing on a checkpoint cadence produces a repository orders of magnitude larger than the model
within a single run. The two cadences are deliberately separate settings.

### Structural events are journalled as they happen

**Given** growth, pruning, label allocation, or a scored observation
**When** it occurs
**Then** a line is appended to an event journal recording it.

This is what makes a continuous run auditable: the growth curve and the reward stream are readable
without loading a single checkpoint. It is append-only and costs a line of text per event.

### Persistence belongs to the training loop, not to the model

**Given** a model in memory
**When** it learns
**Then** it writes nothing.

The prior engine wrote one JSON file per edge on every learning call, and printed the full strength
array to standard output alongside. Neither survives contact with a real dataset — a mesh of forty
thousand edges over ten thousand rows implies hundreds of millions of file writes. Persistence is a
policy the caller sets on the trainer.

### Restoring a checkpoint restores behavior, not just weights

**Given** a checkpoint restored into a fresh process
**When** the model predicts on the same input
**Then** it produces the same answer as the model that wrote it.

Which means the random generator state, the auditor's rolling windows, and the step count are part
of the checkpoint, not just the strengths.
