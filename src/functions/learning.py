"""Forward-only learning: moving the edges that carried signal toward the reward."""

import math

import torch
from torch import Tensor

from ..errors import InvalidRewardError
from ..models import HazeHyper, LearnResult


def ensureRewardFinite(reward: float, confidence: float) -> None:
    """Raises if a reward or confidence is not finite.

    Without this the clamp propagates NaN into every strength in the active mask, destroying the
    network with no error and no symptom until its answers become nonsense.
    """
    for name, value in (("reward", reward), ("confidence", confidence)):
        if not math.isfinite(float(value)):
            raise InvalidRewardError(
                f"{name} was {value!r}; a non-finite value would poison every strength in the "
                "learning mask and leave no trace of where it came from"
            )


def calcConfidenceEntropy(states: Tensor, epsilon: float = 1e-9) -> float:
    """Returns how peaked an activation is, as one minus its normalized entropy.

    Defined at both degenerate cases: a single label has zero maximum entropy, and an activation
    summing to zero has no distribution at all. Both arise in ordinary use.
    """
    signals = states.to(torch.float64).flatten()
    if signals.numel() == 0:
        return 0.0
    if signals.numel() == 1:
        return 1.0 if float(signals[0]) > 0 else 0.0

    total = float(signals.sum())
    if total <= 0:
        return 0.0

    probs = signals / total
    entropy = float(-(probs * (probs + epsilon).log()).sum())
    max_entropy = math.log(signals.numel())
    if max_entropy <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


def applyLearning(
    mesh,
    trace: Tensor,
    reward: float,
    confidence: float,
    hyper: HazeHyper,
    reverse: bool = False,
) -> LearnResult:
    """Moves every edge in the trace by epsilon times the gap between reward and confidence.

    A confident wrong answer is punished hardest and a confident right answer barely moves, so
    the mesh stops adjusting what it already reliably knows. Selection is by `torch.where` over
    the full edge tensors rather than boolean indexing, which allocates and forces a device
    synchronization; see specs/learning.md.
    """
    ensureRewardFinite(reward, confidence)

    live = mesh.counts.edges
    if live == 0:
        return LearnResult(reward=reward, confidence=confidence, reverse=reverse)

    span = slice(0, live)
    fired = trace[:live] if trace.numel() >= live else torch.zeros(live, dtype=torch.bool)
    mask = (~fired if reverse else fired) & mesh.alive_e[span]
    updated = int(mask.sum())
    if updated == 0:
        return LearnResult(reward=reward, confidence=confidence, reverse=reverse)

    signed = float(reward) - float(confidence)
    delta = mesh.epsilon[span] * signed
    zero = torch.zeros_like(delta)

    moved = torch.where(mask, delta, zero)
    mesh.strength[span] = torch.where(
        mask,
        (mesh.strength[span] + moved).clamp(hyper.strength_lower, hyper.strength_upper),
        mesh.strength[span],
    )
    mesh.epsilon[span] = torch.where(
        mask, mesh.epsilon[span] * hyper.epsilon_decay, mesh.epsilon[span]
    )
    mesh.log_str[span] = mesh.strength[span].clamp_min(1e-6).log()

    return LearnResult(
        edges_updated=updated,
        reward=float(reward),
        confidence=float(confidence),
        reverse=reverse,
        mean_delta=float(moved.sum() / max(updated, 1)),
    )


def calcPruneMask(mesh, hyper: HazeHyper) -> Tensor:
    """Returns which live edges have fallen to or below the pruning threshold."""
    live = mesh.counts.edges
    if live == 0:
        return torch.zeros(0, dtype=torch.bool)
    return mesh.alive_e[:live] & (mesh.strength[:live] <= hyper.prune_threshold)
