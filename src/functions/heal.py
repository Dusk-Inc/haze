"""Healing: restoring a neuron's ability to pass signal without discarding what it learned."""

import torch
from torch import Tensor

from ..models import HazeHyper


def calcOutgoingPeak(mesh) -> Tensor:
    """Returns each neuron's strongest live outgoing edge, by magnitude.

    The peak rather than the sum, because signal leaves a neuron edge by edge and each edge is
    gated on its own arriving magnitude. A neuron with one strong out-edge conducts; one with
    twenty weak ones summing to the same total conducts nothing.
    """
    live = mesh.counts.edges
    nodes = int(mesh.kind.numel())
    peak = torch.zeros(nodes, dtype=mesh.dtype)
    if live == 0:
        return peak
    span = slice(0, live)
    magnitude = torch.where(
        mesh.alive_e[span], mesh.strength[span].abs(), torch.zeros(live, dtype=mesh.dtype)
    )
    peak.scatter_reduce_(0, mesh.src[span].to(torch.int64), magnitude, reduce="amax")
    return peak


def calcHealingFactors(peak: Tensor, hyper: HazeHyper) -> Tensor:
    """Returns the multiplier each neuron's outgoing edges need to carry signal again.

    One for a neuron already conducting, so healing is silent where nothing is wrong. A neuron
    whose every out-edge has fallen to zero is left alone rather than scaled by an unbounded
    factor: proportional scaling of nothing is still nothing, and that neuron needs rewiring
    rather than lifting.
    """
    target = hyper.calcConductionFloor() * hyper.heal_margin
    mute = (peak > 0) & (peak < target)
    return torch.where(mute, target / peak.clamp_min(1e-12), torch.ones_like(peak))


def applyConductanceHealing(mesh, hyper: HazeHyper) -> int:
    """Lifts every mute neuron's outgoing edges back to conducting, and returns how many were lifted.

    **Proportional, which is the whole point.** Learning decides which of a neuron's out-edges it
    prefers; healing decides only whether any of them can be expressed. Scaling them all by one
    factor leaves every relative preference exactly where learning put it — the ordering is
    untouched and no edge overtakes another — while restoring the absolute level that makes the
    ordering reachable at all. Rewiring a mute neuron instead discards what it learned, and measured
    that way it scores worse than leaving the mesh alone.

    Reached by neuron rather than by trace, which is what lets it work where reverse learning could
    not. Every learning update in Haze is gated on a fired trace, and a mute edge fires none, so a
    trace-driven remedy can only ever reach the edges that are already fine. See
    specs/propagation.md for the conduction floor this restores against.
    """
    live = mesh.counts.edges
    if live == 0:
        return 0

    factors = calcHealingFactors(calcOutgoingPeak(mesh), hyper)
    lifted = int((factors > 1.0).sum())
    if lifted == 0:
        return 0

    span = slice(0, live)
    scaled = mesh.strength[span] * factors[mesh.src[span].to(torch.int64)]
    mesh.strength[span] = torch.where(
        mesh.alive_e[span],
        scaled.clamp(hyper.calcStrengthFloor(), hyper.strength_upper),
        mesh.strength[span],
    )
    mesh.log_str[span] = mesh.strength[span].abs().clamp_min(1e-6).log()
    return lifted
