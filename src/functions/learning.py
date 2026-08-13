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


def calcRewardAdvantage(mesh, reward: float, hyper: HazeHyper) -> float:
    """Returns how much a reward beat what the mesh had come to expect, and records the reward.

    This is the factor that decides the sign of every update, so what it is measured against
    decides what the mesh learns. Measured against `confidence` — the inherited signal — it is
    not centred on anything: a correct answer contributes `1 - c` and a wrong one only `-c`, so
    at even odds and typical confidence the chosen answer's path is reinforced by a net positive
    amount *whether or not it was right*. Measured, that ran to a fixed point where one motor won
    on 96-100% of observations and the mesh answered the same label forever.

    Centring on the mesh's own recent reward removes exactly that term. The expectation of the
    advantage is zero by construction, so a uniform push cancels and only the part of an edge's
    activity that covaries with the outcome accumulates. See specs/learning.md.

    The baseline is seeded from the first reward rather than from zero or from an assumed 0.5.
    A single sample is its own expectation, so the first advantage is exactly zero and nothing is
    learned from an outcome there was no expectation to compare against. Seeding at zero would
    instead spend the whole warm-up applying the uniform push this function exists to remove, and
    seeding at 0.5 would assume a reward scale the caller never agreed to.

    The reward is checked here as well as at the update, because the baseline is persistent state:
    a non-finite reward that reached it would survive into the checkpoint and poison every later
    advantage, long after the observation that caused it.
    """
    ensureRewardFinite(reward, 0.0)
    if not hyper.reward_baseline:
        return float(reward)

    seen = int(mesh.learn_count[0])
    baseline = float(reward) if seen == 0 else float(mesh.reward_bar[0])
    mesh.reward_bar[0] = baseline + hyper.reward_baseline_rate * (float(reward) - baseline)
    mesh.learn_count[0] = seen + 1
    return float(reward) - baseline


def switchMotorChoice(states: Tensor, rate: float, generator: torch.Generator) -> Tensor:
    """Returns motor activation with a random motor swapped into the lead, at the explore rate.

    A centred advantage learns from the difference between what happened and what was expected,
    which means it learns nothing at all when nothing varies. A greedy readout on a mesh that is
    uniformly wrong produces exactly that: reward is constant, the baseline meets it, the
    advantage is zero, and the mesh stays wrong forever. Measured on the constant task, one seed
    in three settled at 0.00 and never moved. Reward variance is not a nuisance here, it is the
    only thing there is to learn from, and a deterministic readout produces none.

    A swap rather than a boost, because the multiset of activations is preserved: confidence,
    entropy, and every magnitude-sensitive decoder behave exactly as they would have, and the one
    thing that changes is which label holds the peak. Promoting by an added margin would instead
    make the mesh look more certain precisely when it is guessing.
    """
    if rate <= 0.0 or states.numel() < 2:
        return states
    if float(torch.rand(1, generator=generator)) >= rate:
        return states

    pick = int(torch.randint(0, states.numel(), (1,), generator=generator))
    lead = int(torch.argmax(states))
    if pick == lead:
        return states

    switched = states.clone()
    switched[lead], switched[pick] = states[pick], states[lead]
    return switched


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
