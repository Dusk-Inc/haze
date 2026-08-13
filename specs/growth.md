# Growth

How the mesh adds neurons and edges in response to poor performance.

## Description

Haze is self-organizing: when it performs badly and is unsure, it grows. An auditor watches rolling
windows of recent reward and recent confidence and decides both whether to grow and by how much.
Growth adds interneurons to the nexus and terminus meshes and wires them in; sensors and motors are
added separately and on demand, driven by input width and label count rather than by the auditor.

## Policies

### Growth is triggered by sustained error, not by a single bad answer

**Given** rolling windows of the last `audit_window` rewards and confidences
**When** both windows are full and the observed error rate exceeds `growth_threshold`
**Then** growth is planned; otherwise the mesh is left alone.

Requiring a full window is what keeps a single unlucky observation from restructuring the network.

### Growth amount scales with error and uncertainty together

**Given** an error rate and a confidence rate
**When** the growth amount is computed
**Then** it increases with both error and uncertainty, so a mesh that is wrong *and* unsure grows
more than one that is wrong but confident.

### Growth writes into pre-allocated slack

**Given** a growth plan that fits within current capacity
**When** it is applied
**Then** neurons and edges are written into unused slots with no reallocation, and existing indices
are never renumbered.

Capacity doubles when exhausted, giving amortized O(1) growth. Never renumbering is the invariant
that lets learned edges survive growth: every label, port binding, and edge endpoint refers to a
neuron by integer index.

### Neuron populations occupy separate capacity blocks

**Given** the four neuron populations
**When** capacity is laid out
**Then** each holds its own contiguous block with its own slack, ordered so that populations which
grow on demand but stay small (sensors, motors) precede those that grow steadily (nexus, terminus).

This keeps motors a zero-copy slice for readout and means nexus growth does not shift the motor
block. When a block does overflow, the layout is rebuilt and every edge endpoint is remapped by a
single gather.

### Sensors are allocated per input feature, on demand

**Given** an encoder presenting more features than it currently has sensors for
**When** an observation is made
**Then** the shortfall is allocated and wired into the nexus.

This is the property the original design note called out as the point of the architecture: a
network that handles a change in feature size without being rebuilt.

### Motors are allocated per answer label, on demand

**Given** a decoder whose label set includes a label with no motor
**When** the label set is set
**Then** a motor is allocated for it and wired from every terminus interneuron.

Existing label-to-motor bindings are immutable for the model's life. A label's identity is its
motor index, and reassigning one silently invalidates everything the mesh learned about it.

### An edge is never duplicated

**Given** a growth plan proposing an edge between a pair already connected
**When** the plan is applied
**Then** the duplicate is dropped.

### Neurons are added but never removed

**Given** a mesh that has grown
**When** pruning runs
**Then** edges may be removed but neurons are not.

See [pruning.md](pruning.md) for why neuron removal is refused rather than merely unimplemented.
