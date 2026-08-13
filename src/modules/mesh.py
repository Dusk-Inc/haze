"""The tensor slab: neuron and edge state, and the allocation that keeps their indices stable."""

import torch
from torch import Tensor, nn

from ..errors import CapacityExceededError, LabelSpaceError
from ..functions.allocate import (
    calcBlockCapacity,
    calcBlockOffsets,
    calcGrownCapacity,
    dedupeEdges,
    makeFanout,
    makeRemap,
    sampleTargets,
)
from ..models import HazeConfig, MeshCapacity, MeshCounts
from ..tokens import BLOCK_ORDER, UNOWNED, MeshRegion, NeuronKind

NEURON_BUFFERS = ("kind", "owner", "active", "alive_n")
EDGE_BUFFERS = ("src", "dst", "strength", "log_str", "epsilon", "edge_id", "alive_e")


class MeshState(nn.Module):
    """Holds every tensor describing a mesh, and owns the invariants over their indices.

    All state is registered as a buffer rather than a parameter: there are no gradients, and
    parameters would attract optimizers and appear in the model's parameter list where they
    would be meaningless. See specs/packaging.md.
    """

    def __init__(self, config: HazeConfig) -> None:
        """Allocates every buffer at the config's recorded capacity."""
        super().__init__()
        self.config = config
        self.dtype = torch.float32 if config.dtype == "float32" else torch.float64

        capacity = config.capacity
        n_cap = capacity.neurons
        e_cap = capacity.edges

        self.register_buffer("kind", torch.full((n_cap,), int(NeuronKind.FREE), dtype=torch.uint8))
        self.register_buffer("owner", torch.full((n_cap,), UNOWNED, dtype=torch.int32))
        self.register_buffer("active", torch.zeros(n_cap, dtype=torch.bool))
        self.register_buffer("alive_n", torch.zeros(n_cap, dtype=torch.bool))

        self.register_buffer("src", torch.zeros(e_cap, dtype=torch.int64))
        self.register_buffer("dst", torch.zeros(e_cap, dtype=torch.int64))
        self.register_buffer("strength", torch.ones(e_cap, dtype=self.dtype))
        self.register_buffer("log_str", torch.zeros(e_cap, dtype=self.dtype))
        self.register_buffer("epsilon", torch.zeros(e_cap, dtype=self.dtype))
        self.register_buffer("edge_id", torch.full((e_cap,), -1, dtype=torch.int64))
        self.register_buffer("alive_e", torch.zeros(e_cap, dtype=torch.bool))

        self.register_buffer("step_count", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("next_edge_id", torch.zeros(1, dtype=torch.int64))

        self.counts = MeshCounts(**config.counts.model_dump())
        self.free_slots: dict[NeuronKind, list[int]] = {kind: [] for kind in BLOCK_ORDER}
        self.generator = torch.Generator().manual_seed(config.seed)


    @property
    def capacity(self) -> MeshCapacity:
        """Returns the capacity the buffers are currently allocated at."""
        return self.config.capacity

    def findBlockOffset(self, kind: NeuronKind) -> int:
        """Returns the first slab index belonging to a neuron population."""
        return calcBlockOffsets(self.capacity)[kind]

    def findBlockCount(self, kind: NeuronKind) -> int:
        """Returns how many neurons of a population are live."""
        return {
            NeuronKind.SENSOR: self.counts.sensors,
            NeuronKind.MOTOR: self.counts.motors,
            NeuronKind.NEXUS: self.counts.nexus,
            NeuronKind.TERMINUS: self.counts.terminus,
        }[kind]

    def findMotorSlice(self) -> slice:
        """Returns the motor block as a slice, so readout is a view rather than a gather."""
        lo = self.findBlockOffset(NeuronKind.MOTOR)
        return slice(lo, lo + self.capacity.motors)

    def findNeuronIds(self, kind: NeuronKind) -> Tensor:
        """Returns the slab indices of every live neuron of a population."""
        return ((self.kind == int(kind)) & self.alive_n).nonzero(as_tuple=True)[0]

    def findPortNeuronIds(self, slot: int) -> Tensor:
        """Returns the slab indices a port owns.

        A mask rather than a stored range: ranges stop being true the moment a freed slot is
        reused, which is why ownership lives on the neuron. See specs/ports.md.
        """
        return ((self.owner == slot) & self.alive_n).nonzero(as_tuple=True)[0]

    @property
    def is_motor(self) -> Tensor:
        """Returns a mask over the slab selecting motor neurons."""
        return self.kind == int(NeuronKind.MOTOR)

    @property
    def is_inter(self) -> Tensor:
        """Returns a mask over the slab selecting interneurons of either mesh."""
        return (self.kind == int(NeuronKind.NEXUS)) | (self.kind == int(NeuronKind.TERMINUS))


    def allocNeuronIds(self, count: int, kind: NeuronKind, owner: int = UNOWNED) -> Tensor:
        """Reserves slab slots for a population, reusing freed slots before extending.

        Never renumbers a live neuron: every label binding, port claim, and edge endpoint refers
        to a neuron by index, so moving one silently reassigns what the mesh learned about it.
        """
        if count <= 0:
            return torch.empty(0, dtype=torch.int64)

        reused = [self.free_slots[kind].pop() for _ in range(min(count, len(self.free_slots[kind])))]
        fresh_needed = count - len(reused)

        if fresh_needed > 0:
            live = self.findBlockCount(kind)
            room = calcBlockCapacity(self.capacity, kind)
            if live + fresh_needed > room:
                self.growCapacity(kind, live + fresh_needed)
            offset = self.findBlockOffset(kind)
            fresh = list(range(offset + live, offset + live + fresh_needed))
            self._addBlockCount(kind, fresh_needed)
        else:
            fresh = []

        ids = torch.tensor(reused + fresh, dtype=torch.int64)
        self.kind[ids] = int(kind)
        self.owner[ids] = owner
        self.alive_n[ids] = True
        self.active[ids] = kind != NeuronKind.MOTOR
        return ids

    def _addBlockCount(self, kind: NeuronKind, delta: int) -> None:
        """Adjusts a population's live count."""
        field = {
            NeuronKind.SENSOR: "sensors",
            NeuronKind.MOTOR: "motors",
            NeuronKind.NEXUS: "nexus",
            NeuronKind.TERMINUS: "terminus",
        }[kind]
        setattr(self.counts, field, getattr(self.counts, field) + delta)

    def growCapacity(self, kind: NeuronKind, needed: int) -> None:
        """Rebuilds the slab with a larger block, remapping every live neuron and edge endpoint."""
        old_capacity = self.capacity.model_copy(deep=True)
        field = {
            NeuronKind.SENSOR: "sensors",
            NeuronKind.MOTOR: "motors",
            NeuronKind.NEXUS: "nexus",
            NeuronKind.TERMINUS: "terminus",
        }[kind]
        grown = calcGrownCapacity(getattr(old_capacity, field), needed)
        new_capacity = old_capacity.model_copy(update={field: grown})

        live = {k: self.findBlockCount(k) for k in BLOCK_ORDER}
        remap = makeRemap(old_capacity, new_capacity, live).to(self.kind.device)

        old_offsets = calcBlockOffsets(old_capacity)
        new_offsets = calcBlockOffsets(new_capacity)

        for name in NEURON_BUFFERS:
            old = getattr(self, name)
            new = torch.zeros(new_capacity.neurons, dtype=old.dtype, device=old.device)
            if name == "kind":
                new.fill_(int(NeuronKind.FREE))
            elif name == "owner":
                new.fill_(UNOWNED)
            for block in BLOCK_ORDER:
                n = live[block]
                if n:
                    o_lo, n_lo = old_offsets[block], new_offsets[block]
                    new[n_lo : n_lo + n] = old[o_lo : o_lo + n]
            setattr(self, name, new)

        self.free_slots = {
            block: [int(remap[i].item()) for i in slots if remap[i] >= 0]
            for block, slots in self.free_slots.items()
        }

        if self.counts.edges:
            live_e = slice(0, self.counts.edges)
            self.src[live_e] = remap[self.src[live_e]]
            self.dst[live_e] = remap[self.dst[live_e]]

        self.config.capacity = new_capacity


    def addMeshEdges(self, src: Tensor, dst: Tensor, strength: Tensor | None = None) -> Tensor:
        """Appends edges into slack, growing the edge arrays only when they overflow."""
        count = int(src.numel())
        if count == 0:
            return torch.empty(0, dtype=torch.int64)

        if self.counts.edges + count > self.capacity.edges:
            self.growEdgeCapacity(self.counts.edges + count)

        lo = self.counts.edges
        span = slice(lo, lo + count)
        hyper = self.config.hyper

        self.src[span] = src.to(torch.int64)
        self.dst[span] = dst.to(torch.int64)
        if strength is None:
            drawn = torch.empty(count, dtype=self.dtype).uniform_(
                hyper.strength_init_lower, hyper.strength_init_upper, generator=self.generator
            )
        else:
            drawn = strength.to(self.dtype)
        self.strength[span] = drawn
        self.log_str[span] = drawn.clamp_min(1e-6).log()
        self.epsilon[span] = hyper.epsilon_start

        first_id = int(self.next_edge_id.item())
        self.edge_id[span] = torch.arange(first_id, first_id + count, dtype=torch.int64)
        self.next_edge_id += count
        self.alive_e[span] = True
        self.counts.edges += count

        return torch.arange(lo, lo + count, dtype=torch.int64)

    def growEdgeCapacity(self, needed: int) -> None:
        """Reallocates the edge arrays to hold `needed` edges, preserving every live index."""
        grown = calcGrownCapacity(self.capacity.edges, needed)
        live = self.counts.edges
        for name in EDGE_BUFFERS:
            old = getattr(self, name)
            new = torch.zeros(grown, dtype=old.dtype, device=old.device)
            if name == "strength":
                new.fill_(1.0)
            elif name == "edge_id":
                new.fill_(-1)
            new[:live] = old[:live]
            setattr(self, name, new)
        self.config.capacity = self.capacity.model_copy(update={"edges": grown})

    def findEdgeKeys(self) -> Tensor:
        """Returns a packed key per live edge, so duplicate pairs can be tested in one op."""
        live = slice(0, self.counts.edges)
        return self.src[live] * self.capacity.neurons + self.dst[live]

    def connectNeurons(
        self, sources: Tensor, candidates: Tensor, fanout: Tensor | None = None
    ) -> Tensor:
        """Wires each source to a sample of candidates, dropping self-loops and duplicates."""
        if sources.numel() == 0 or candidates.numel() == 0:
            return torch.empty(0, dtype=torch.int64)

        hyper = self.config.hyper
        if fanout is None:
            fanout = makeFanout(
                int(sources.numel()), hyper.fanout_min, hyper.fanout_max, self.generator
            )
        fanout = fanout.clamp_max(int(candidates.numel()))

        src, dst = sampleTargets(sources, candidates, fanout, self.generator)
        src, dst = dedupeEdges(src, dst, self.findEdgeKeys(), self.capacity.neurons)
        return self.addMeshEdges(src, dst)

    def connectAll(self, sources: Tensor, targets: Tensor) -> Tensor:
        """Wires every source to every target, dropping self-loops and duplicates."""
        if sources.numel() == 0 or targets.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        src = sources.repeat_interleave(targets.numel())
        dst = targets.repeat(sources.numel())
        src, dst = dedupeEdges(src, dst, self.findEdgeKeys(), self.capacity.neurons)
        return self.addMeshEdges(src, dst)


    def buildMesh(self, nexus_size: int, terminus_size: int) -> None:
        """Creates the two interneuron meshes and wires nexus into terminus."""
        nexus = self.allocNeuronIds(nexus_size, NeuronKind.NEXUS)
        terminus = self.allocNeuronIds(terminus_size, NeuronKind.TERMINUS)
        self.connectNeurons(nexus, nexus)
        self.connectNeurons(terminus, terminus)
        self.connectNeurons(nexus, terminus)

    def connectSensors(self, sensors: Tensor) -> Tensor:
        """Wires new sensors into every nexus interneuron."""
        return self.connectAll(sensors, self.findNeuronIds(NeuronKind.NEXUS))

    def connectMotors(self, motors: Tensor) -> Tensor:
        """Wires every terminus interneuron into new motors.

        These edges are registered like any other. The prior engine created them outside its
        registry, leaving them pinned at a hardcoded strength and invisible to learning.
        """
        return self.connectAll(self.findNeuronIds(NeuronKind.TERMINUS), motors)


    def compactEdges(self, keep: Tensor) -> int:
        """Drops the edges `keep` excludes, permuting every edge array by one shared index.

        Sharing the permutation is what makes index consistency structural rather than
        maintained: the prior engine dropped connectors from its list while leaving their
        strength, epsilon, and status slots behind, and those orphans were then reverse-learned
        forever. Re-sorting by source keeps the propagation gather sequential, and costs nothing
        because the arrays are being rewritten anyway. See specs/pruning.md.
        """
        live = self.counts.edges
        if live == 0:
            return 0

        keep = keep[:live]
        idx = keep.nonzero(as_tuple=True)[0]
        removed = live - int(idx.numel())
        if removed == 0:
            return 0

        idx = idx[self.src[idx].argsort(stable=True)]
        count = int(idx.numel())

        for name in EDGE_BUFFERS:
            buffer = getattr(self, name)
            buffer[:count] = buffer.index_select(0, idx)
            buffer[count:live] = 0 if name != "strength" else 1.0

        self.edge_id[count:live] = -1
        self.alive_e[count:live] = False
        self.counts.edges = count
        return removed


    def ensureMeshConsistent(self) -> None:
        """Raises if any structural invariant the index scheme depends on has been broken."""
        live = slice(0, self.counts.edges)
        if self.counts.edges > self.capacity.edges:
            raise CapacityExceededError(
                f"edge count {self.counts.edges} exceeds capacity {self.capacity.edges}"
            )
        if not bool(self.alive_n[self.src[live]].all()):
            raise LabelSpaceError("an edge leaves a neuron that is not live")
        if not bool(self.alive_n[self.dst[live]].all()):
            raise LabelSpaceError("an edge arrives at a neuron that is not live")
        keys = self.findEdgeKeys()
        if int(torch.unique(keys).numel()) != int(keys.numel()):
            raise LabelSpaceError("the mesh holds a duplicate edge")
        for kind in BLOCK_ORDER:
            live_ids = self.findNeuronIds(kind)
            free = set(self.free_slots[kind])
            if free & set(live_ids.tolist()):
                raise LabelSpaceError(f"a {kind.name.lower()} slot is both free and live")

    def toCounts(self) -> MeshCounts:
        """Returns a snapshot of live occupancy, for writing into a checkpoint config."""
        return MeshCounts(**self.counts.model_dump())


def findMeshRegionKind(region: MeshRegion) -> NeuronKind:
    """Returns the neuron population a mesh region names."""
    return NeuronKind.NEXUS if region == MeshRegion.NEXUS else NeuronKind.TERMINUS
