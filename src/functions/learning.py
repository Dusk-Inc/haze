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
        (mesh.strength[span] + moved).clamp(hyper.calcStrengthFloor(), hyper.strength_upper),
        mesh.strength[span],
    )
    mesh.epsilon[span] = torch.where(
        mask, mesh.epsilon[span] * hyper.epsilon_decay, mesh.epsilon[span]
    )
    mesh.log_str[span] = mesh.strength[span].abs().clamp_min(1e-6).log()

    return LearnResult(
        edges_updated=updated,
        reward=float(reward),
        confidence=float(confidence),
        reverse=reverse,
        mean_delta=float(moved.sum() / max(updated, 1)),
    )


def calcNeuronCredit(
    mesh, trace: Tensor, seeds: dict[int, float], hops: int
) -> Tensor:
    """Spreads a per-motor teaching signal backward along the edges that carried signal.

    This is not a gradient and there is no chain rule: a scalar per motor is pushed back over the
    fired edges, attenuated by the same strengths the forward pass used, so a neuron's credit is
    how much it fed the motors that were rewarded. It costs one extra scatter per hop over the
    same edge list the forward pass walks.
    """
    neurons = int(mesh.kind.numel())
    credit = torch.zeros(neurons, dtype=mesh.dtype)
    if not seeds:
        return credit
    for motor, value in seeds.items():
        credit[motor] = value

    live = mesh.counts.edges
    if live == 0:
        return credit

    src, dst = mesh.src[:live], mesh.dst[:live]
    strength = mesh.strength[:live]
    fired = trace[:live]
    zero = torch.zeros(live, dtype=mesh.dtype)

    total = credit.clone()
    frontier = credit
    for _ in range(max(hops, 1)):
        carried = torch.where(fired, frontier[dst] * strength, zero)
        if not bool((carried != 0).any()):
            break
        frontier = torch.zeros(neurons, dtype=mesh.dtype).index_add_(0, src, carried)
        total = total + frontier
    return total


def calcMotorSeeds(motors: Tensor, chosen: int, gain: float) -> dict[int, float]:
    """Returns the teaching signal for each competing motor, given which one the decoder chose.

    Learning is told a reward, never the correct label, so credit is seeded at the motor that was
    actually chosen and signed by how the reward turned out. A rewarded choice reinforces the path
    that produced it and weakens its rivals; an unrewarded one does the reverse, which is what lets
    a rival overtake it. Rivals share the opposite signal so the seeds sum to zero and the mesh's
    total strength is not driven in one direction.
    """
    ids = motors.tolist()
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: gain}
    share = gain / (len(ids) - 1)
    return {motor: (gain if motor == chosen else -share) for motor in ids}


def applyCreditedLearning(
    mesh,
    trace: Tensor,
    eligibility: Tensor,
    credit: Tensor,
    reward: float,
    confidence: float,
    hyper: HazeHyper,
) -> LearnResult:
    """Moves each fired edge by its own eligibility times the credit reaching its destination.

    Three factors, none of them a gradient: how much the edge carried, how much its destination
    was worth crediting, and how surprising the outcome was. The uniform rule this replaces moved
    every fired edge by one scalar, which cannot make one motor beat another because both of their
    inbound edges are in the trace with the same sign. See specs/learning.md.
    """
    ensureRewardFinite(reward, confidence)

    live = mesh.counts.edges
    if live == 0:
        return LearnResult(reward=reward, confidence=confidence)

    span = slice(0, live)
    fired = trace[:live]
    mask = fired & mesh.alive_e[span]
    updated = int(mask.sum())
    if updated == 0:
        return LearnResult(reward=reward, confidence=confidence)

    weight = eligibility[:live]
    scale = weight.abs().max().clamp_min(1e-12)
    signal = (weight / scale) * credit[mesh.dst[span]]

    zero = torch.zeros(live, dtype=mesh.dtype)
    moved = torch.where(mask, mesh.epsilon[span] * signal, zero)

    mesh.strength[span] = torch.where(
        mask,
        (mesh.strength[span] + moved).clamp(hyper.calcStrengthFloor(), hyper.strength_upper),
        mesh.strength[span],
    )
    mesh.epsilon[span] = torch.where(
        mask, mesh.epsilon[span] * hyper.epsilon_decay, mesh.epsilon[span]
    )
    mesh.log_str[span] = mesh.strength[span].abs().clamp_min(1e-6).log()

    return LearnResult(
        edges_updated=updated,
        reward=float(reward),
        confidence=float(confidence),
        mean_delta=float(moved.sum() / max(updated, 1)),
    )


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
