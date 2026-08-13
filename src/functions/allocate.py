"""Capacity arithmetic for the neuron slab and the edge arrays."""

import torch
from torch import Tensor

from ..models import MeshCapacity
from ..tokens import BLOCK_ORDER, NeuronKind


def calcBlockOffsets(capacity: MeshCapacity) -> dict[NeuronKind, int]:
    """Returns the first slab index of each capacity block, in BLOCK_ORDER."""
    offsets: dict[NeuronKind, int] = {}
    cursor = 0
    for kind in BLOCK_ORDER:
        offsets[kind] = cursor
        cursor += calcBlockCapacity(capacity, kind)
    return offsets


def calcBlockCapacity(capacity: MeshCapacity, kind: NeuronKind) -> int:
    """Returns how many slots a capacity block holds."""
    return {
        NeuronKind.SENSOR: capacity.sensors,
        NeuronKind.MOTOR: capacity.motors,
        NeuronKind.NEXUS: capacity.nexus,
        NeuronKind.TERMINUS: capacity.terminus,
    }[kind]


def calcGrownCapacity(current: int, needed: int) -> int:
    """Returns a capacity that holds `needed`, doubling so growth is amortized constant time."""
    if needed <= current:
        return current
    grown = max(current, 1)
    while grown < needed:
        grown *= 2
    return grown


def makeRemap(
    old_capacity: MeshCapacity, new_capacity: MeshCapacity, live: dict[NeuronKind, int]
) -> Tensor:
    """Returns a slab-index remap carrying every live neuron from an old layout to a new one.

    Only a block overflow forces this. Edge endpoints are then remapped by a single gather, which
    is why relayout is affordable and why blocks are ordered so the two that grow steadily sit
    last.
    """
    old_offsets = calcBlockOffsets(old_capacity)
    new_offsets = calcBlockOffsets(new_capacity)
    old_total = old_capacity.neurons

    remap = torch.full((old_total,), -1, dtype=torch.int64)
    for kind in BLOCK_ORDER:
        count = live.get(kind, 0)
        if count == 0:
            continue
        old_lo = old_offsets[kind]
        new_lo = new_offsets[kind]
        remap[old_lo : old_lo + count] = torch.arange(new_lo, new_lo + count, dtype=torch.int64)
    return remap


def makeFanout(
    count: int, low: int, high: int, generator: torch.Generator | None = None
) -> Tensor:
    """Returns an out-degree for each of `count` neurons, drawn uniformly from [low, high]."""
    if count <= 0:
        return torch.empty(0, dtype=torch.int64)
    return torch.randint(low, high + 1, (count,), generator=generator, dtype=torch.int64)


def sampleTargets(
    sources: Tensor,
    candidates: Tensor,
    fanout: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Returns candidate edges wiring each source to `fanout` distinct targets, excluding itself.

    Oversamples and de-duplicates rather than permuting per source, which keeps the cost tied to
    the fan-out rather than to the candidate-set size.
    """
    if sources.numel() == 0 or candidates.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty

    width = int(fanout.max().item())
    if width <= 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty

    draw = torch.randint(
        0, candidates.numel(), (sources.numel(), width), generator=generator, dtype=torch.int64
    )
    picked = candidates[draw]
    src = sources.unsqueeze(1).expand_as(picked)

    within_fanout = torch.arange(width).unsqueeze(0) < fanout.unsqueeze(1)
    keep = within_fanout & (picked != src)

    return src[keep], picked[keep]


def dedupeEdges(
    src: Tensor, dst: Tensor, existing_key: Tensor | None, stride: int
) -> tuple[Tensor, Tensor]:
    """Returns the proposed edges with self-loops, internal repeats, and existing pairs removed."""
    if src.numel() == 0:
        return src, dst

    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return src, dst

    key = src * stride + dst
    key, first = torch.unique(key, return_inverse=False, sorted=True), None
    del first

    src = torch.div(key, stride, rounding_mode="floor")
    dst = key - src * stride

    if existing_key is not None and existing_key.numel() > 0:
        fresh = ~torch.isin(key, existing_key)
        src, dst = src[fresh], dst[fresh]

    return src, dst
