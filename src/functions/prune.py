"""Pruning: removing edges that have stopped carrying useful signal, and repairing what that cuts off."""

import torch
from torch import Tensor

from ..models import HazeHyper, PruneReport


def calcPruneMask(mesh, hyper: HazeHyper) -> Tensor:
    """Returns which live edges have fallen to or below the pruning threshold in magnitude.

    Magnitude, not value: a useless edge is one near zero, while a strongly negative edge is a
    strongly inhibitory one and carries as much information as a strongly positive one. Testing
    the signed value would delete every inhibitory edge the moment it was created.
    """
    live = mesh.counts.edges
    if live == 0:
        return torch.zeros(0, dtype=torch.bool)
    return mesh.alive_e[:live] & (mesh.strength[:live].abs() <= hyper.prune_threshold)


def applyPrune(mesh, hyper: HazeHyper) -> PruneReport:
    """Removes every edge too weak to carry signal, then rewires whatever that stranded.

    The rescue is not optional cleanup. Pruning is allowed to disconnect a neuron and is not
    allowed to leave it disconnected: a sensor with no way out or a motor with no way in is a
    break in the model's interface, so the feature or label it stands for silently stops
    existing. Rescue runs after compaction rather than before, since compaction is what makes
    the stranding visible. See specs/pruning.md.
    """
    live = mesh.counts.edges
    if live == 0:
        return PruneReport()

    doomed = calcPruneMask(mesh, hyper)
    removed = int(doomed.sum())
    if removed == 0:
        return PruneReport(edges_remaining=live)

    mesh.compactEdges(~doomed)
    rewired = mesh.ensureNoOrphans()
    return PruneReport(
        edges_removed=removed,
        edges_remaining=mesh.counts.edges,
        neurons_rewired=rewired,
    )
