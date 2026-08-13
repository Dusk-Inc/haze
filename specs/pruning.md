# Pruning

How the mesh removes edges that have stopped carrying useful signal.

## Description

Learning drives an unhelpful edge's strength down. Once it falls far enough that it could never
pass the edge gate again, the edge is dead weight — it costs memory and a slot in every edge-space
operation while contributing nothing. Pruning removes such edges and compacts the edge arrays.

## Policies

### An edge is pruned when it can no longer pass the gate

**Given** an edge whose strength has fallen to or below `prune_threshold`
**When** pruning runs
**Then** the edge is removed from the mesh.

### Pruning compacts every edge-parallel array with one shared permutation

**Given** a set of surviving edges
**When** pruning compacts
**Then** `src`, `dst`, `strength`, `epsilon`, `log_str`, and `edge_id` are all permuted by the same
index tensor.

This makes index consistency a structural invariant rather than something to maintain. The prior
engine removed connectors from its list while leaving their strength, epsilon, and status slots in
place; those orphans were then reverse-learned on every reverse pass and the arrays grew
monotonically. Sharing one permutation makes that class of defect unrepresentable.

Compaction re-sorts edges by source, which keeps the gather in the propagation step sequential. The
sort is free because compaction already rewrites the arrays.

### An edge's identity survives compaction

**Given** an edge that is moved by compaction
**When** it is referred to across a checkpoint boundary
**Then** its `edge_id` is unchanged.

Position is not identity. `edge_id` is what persistence and the event journal record.

### Neurons are never compacted

**Given** a neuron that has become isolated
**When** pruning runs
**Then** the neuron is retained; freed neuron slots go on a free list and are reused, but live
neurons are never moved.

Neurons are referenced by integer index from three independent places — the label-to-motor table,
port ownership, and every edge endpoint. Compacting them invalidates all three simultaneously, and
the failure mode is the worst available: the model still runs, raises nothing, and predicts
nonsense. Reusing free slots costs a little fragmentation and removes the failure mode entirely.

### A neuron left with too few connections is rewired, not deleted

**Given** an interneuron whose edges have nearly all been pruned
**When** pruning completes
**Then** its remaining edges are cleared and it is rewired into its mesh.

Growth is expensive and a neuron that has lost its edges has lost its learned role, not its
capacity to take a new one.

### Pruning happens at a batch boundary

**Given** a batch in flight
**When** pruning would apply
**Then** it is deferred until the batch completes.

Compaction changes the meaning of every edge index, including those in a live fired trace.
