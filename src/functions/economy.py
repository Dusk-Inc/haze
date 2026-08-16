"""The signal economy: what a strength means, and what sets the level that arrives at a neuron."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..models import HazeHyper


@dataclass(frozen=True)
class SignalEconomy:
    """One pass's edge shares and per-neuron incoming budget.

    Computed per propagation and never stored. Projecting these onto `mesh.strength` instead was
    considered and is worse in three ways: it collides with the strength rails, since a neuron with
    twelve out-edges each clamped to `strength_lower` already sums past any budget below 1.2; it
    changes what `prune_threshold` names, because a pruned strength would then be a share; and it
    makes every existing checkpoint incomparable. Computing it costs two scatters over the edge
    list once per observation, against roughly six edge-length operations per hop.
    """

    share: Tensor
    """Each edge's strength as a signed share of its source neuron's total output."""

    log_share: Tensor
    """Log of each share's magnitude, for the path statistics propagation carries."""

    in_scale: Tensor
    """Each neuron's incoming budget: the sum of its live in-edges' share magnitudes."""


def calcOutgoingScale(mesh, budget: float) -> Tensor:
    """Returns the factor dividing each neuron's out-edges so their magnitudes sum to `budget`.

    A neuron with no live out-edges gets one rather than zero, so dividing by this is always safe
    and an unwired neuron's (absent) edges are left alone.
    """
    neurons = int(mesh.kind.numel())
    live = mesh.counts.edges
    total = torch.zeros(neurons, dtype=mesh.dtype)
    if live == 0:
        return total.clamp_min(1.0)

    span = slice(0, live)
    magnitude = torch.where(
        mesh.alive_e[span], mesh.strength[span].abs(), torch.zeros(live, dtype=mesh.dtype)
    )
    total.index_add_(0, mesh.src[span].to(torch.int64), magnitude)
    return torch.where(total > 0, total / budget, torch.ones_like(total))


def calcIncomingScale(mesh, share: Tensor) -> Tensor:
    """Returns each neuron's incoming budget, summed over every live in-edge whether it fired or not.

    **Static by design, and this is the whole distinction.** Dividing instead by the edges that
    actually fired is the fan-in *mean*, and it destroys conjunction: two features arriving would
    produce the same value as one, which is exactly the collapse specs/propagation.md was fixed to
    prevent. A static denominator leaves conjunction intact — two arrivals still produce twice one
    — while removing the purely topological advantage of a well-connected neuron. The roadmap's
    "arriving value does not depend on fan-in" and the spec's "interneurons genuinely accumulate
    fan-in" are about different senses of the word: capacity is normalised away, coincidence is
    kept.

    A neuron with no live in-edges gets one, so a sensor divides by unity rather than by zero.
    **Only that case is substituted.** Clamping the budget up to one instead was built first and is
    wrong: shares average roughly the reciprocal of a neuron's fan-out, so a real budget is usually
    below one — measured, 64.6% of interneurons sit under it and the median is 0.624. Clamping
    replaced the true budget for two neurons in three and divided the least-connected of them by
    fourteen times too much, which inverted the very correlation the economy exists to remove.
    """
    neurons = int(mesh.kind.numel())
    live = mesh.counts.edges
    total = torch.zeros(neurons, dtype=mesh.dtype)
    if live == 0:
        return torch.ones_like(total)

    span = slice(0, live)
    magnitude = torch.where(
        mesh.alive_e[span], share.abs(), torch.zeros(live, dtype=mesh.dtype)
    )
    total.index_add_(0, mesh.dst[span].to(torch.int64), magnitude)
    return torch.where(total > 0, total, torch.ones_like(total))


def makeSignalEconomy(mesh, hyper: HazeHyper) -> SignalEconomy | None:
    """Returns the shares and budgets for one pass, or None when the economy is switched off.

    Returning None rather than an identity economy keeps the shipped path free of any extra tensor
    work, so the flag costs nothing while it is off.
    """
    if not hyper.signal_economy:
        return None

    live = mesh.counts.edges
    if live == 0:
        empty = torch.zeros(0, dtype=mesh.dtype)
        return SignalEconomy(share=empty, log_share=empty, in_scale=calcIncomingScale(mesh, empty))

    span = slice(0, live)
    out_scale = calcOutgoingScale(mesh, hyper.out_budget)
    share = mesh.strength[span] / out_scale[mesh.src[span].to(torch.int64)]
    return SignalEconomy(
        share=share,
        log_share=share.abs().clamp_min(1e-6).log(),
        in_scale=calcIncomingScale(mesh, share),
    )


def calcArrivalGate(arrived: Tensor, gate: Tensor, is_nexus: Tensor, hyper: HazeHyper) -> Tensor:
    """Returns which of the already-gated interneurons may emit, keeping the top `firing_fraction`.

    **Ranked rather than thresholded, and that is the point.** specs/propagation.md records that a
    hard threshold on arrival went bimodal — "learning raises strengths, more edges pass, more
    accumulates, more pass. There is no stable middle." A rank is immune to that loop by
    construction: scaling every strength cannot change how many neurons fire.

    Ranks within the set that already passed the neuron gate rather than over every neuron that
    received anything, so the two selections compose rather than fight: the threshold says what is
    worth emitting, the rank says how much of that the mesh can afford.

    Nexus and terminus rank separately, because one population sits deeper than the other and
    arrives at a different level; pooling them would let the shallower one take every slot.

    Selection uses the k-th value with `>=` rather than `topk` indices, so ties all pass and the
    result does not depend on any ordering. That is what lets the plain-Python reference
    implementation, which has no stable order, agree exactly.
    """
    if hyper.firing_fraction <= 0.0:
        return gate

    level = arrived.abs()
    allowed = torch.zeros_like(gate)
    for population in (is_nexus, ~is_nexus):
        member = gate & population.unsqueeze(0)
        count = int(member.sum())
        if count == 0:
            continue
        keep = max(1, int(count * hyper.firing_fraction))
        cut = float(level[member].sort(descending=True).values[keep - 1])
        allowed |= member & (level >= cut)
    return allowed


def applyGainControl(arrived: Tensor, received: Tensor, hyper: HazeHyper) -> Tensor:
    """Returns the arriving wavefront rescaled toward the band, without reordering it.

    **Applied before the neuron gate, not after.** A neuron's arrival is divided by its *static*
    incoming budget, but only a fraction of its in-edges fire on any one hop, so the quotient sits
    far below the band — measured at a median of 0.037 against a gate of 0.5, which stops
    propagation at the first hop. Rescaling only what already passed the gate cannot fix that,
    because nothing passes; the gain has to reach the value the gate is about to judge.

    Ordering and level are separate jobs and get separate mechanisms: the rank decides which
    neurons emit, and one scalar per hop decides how loud. A single scalar cannot change any
    ordering, so it cannot interfere with the selection that follows it.

    A wavefront already below the edge floor is left alone rather than amplified, so a pass that is
    genuinely dead stays dead and `calcConductionReach` keeps meaning what it says.
    """
    if not hyper.gain_control:
        return arrived

    live = arrived.abs()[received]
    if live.numel() == 0:
        return arrived
    level = float(live.mean())
    if level <= hyper.edge_signal_floor:
        return arrived
    return arrived * min(max(hyper.gain_target / level, 1.0), hyper.gain_ceiling)
