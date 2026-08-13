"""Signal propagation: the tensor engine, and a reference implementation that pins its meaning."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..models import HazeHyper
from ..tokens import NeuronKind


@dataclass
class SignalState:
    """The transient state of one propagation, across `lanes` independent signal streams.

    A lane is one input feature of one observation. The prior engine gave each feature its own
    propagation context, and the loop guard was keyed to it, so lanes reproduce that isolation
    exactly rather than approximating it.
    """

    val: Tensor
    slog: Tensor
    plen: Tensor
    fired: Tensor
    motor_acc: Tensor
    eligibility: Tensor
    hops: int = 0
    reached: bool = False
    per_lane_hops: list[int] = field(default_factory=list)

    @property
    def lanes(self) -> int:
        """Returns how many independent signal streams this state carries."""
        return int(self.val.shape[0])

    def toEdgeEligibility(self, lane_weights: Tensor | None = None) -> Tensor:
        """Returns how much signal each edge carried, weighted by how much its lane mattered.

        This is the eligibility half of reward-modulated learning: an edge that carried more of
        the signal that produced an answer is more responsible for that answer. Pooling lanes
        before weighting them would lose exactly what distinguishes one input feature from
        another, since a lane is a feature. See specs/learning.md.
        """
        if lane_weights is None:
            return self.eligibility.sum(0)
        return (self.eligibility * lane_weights.unsqueeze(1)).sum(0)

    def calcLaneShare(self, motor: int) -> Tensor:
        """Returns each lane's share of the activation arriving at one motor."""
        arrived = self.motor_acc[:, motor]
        total = arrived.sum()
        if float(total) <= 0:
            return torch.full((self.lanes,), 1.0 / max(self.lanes, 1), dtype=arrived.dtype)
        return arrived / total

    def toEdgeTrace(self) -> Tensor:
        """Returns which edges carried signal, reduced across lanes.

        This one tensor is the loop guard, the fired trace, and the learning mask. The prior
        engine kept the first two separately, and the guard grew without bound.
        """
        return self.fired.any(0)


def makeSignalState(lanes: int, neurons: int, edges: int, dtype: torch.dtype) -> SignalState:
    """Allocates the transient state for a propagation of `lanes` streams."""
    return SignalState(
        val=torch.zeros(lanes, neurons, dtype=dtype),
        slog=torch.zeros(lanes, neurons, dtype=dtype),
        plen=torch.zeros(lanes, neurons, dtype=dtype),
        fired=torch.zeros(lanes, edges, dtype=torch.bool),
        motor_acc=torch.zeros(lanes, neurons, dtype=dtype),
        eligibility=torch.zeros(lanes, edges, dtype=dtype),
    )


def calcEdgeSignal(
    state: SignalState, src: Tensor, strength: Tensor, log_str: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns each edge's carried value, path statistics, and geometric-mean-corrected actual.

    Per edge this reproduces the prior engine's signal arithmetic exactly: the value is the
    running product of strengths, and the actual is that value scaled by the geometric mean of
    the strengths traversed. The approximation enters only where paths merge; see
    specs/propagation.md.
    """
    e_val = state.val[:, src] * strength
    e_slog = state.slog[:, src] + log_str
    e_plen = state.plen[:, src] + 1.0
    e_act = e_val * torch.exp(e_slog / e_plen.clamp_min(1.0))
    return e_val, e_slog, e_plen, e_act


def flowSignalStep(
    state: SignalState,
    src: Tensor,
    dst: Tensor,
    strength: Tensor,
    log_str: Tensor,
    alive_e: Tensor,
    active: Tensor,
    is_motor: Tensor,
    is_inter: Tensor,
    hyper: HazeHyper,
) -> Tensor:
    """Advances every lane one hop and returns the mask of edges that fired.

    Ordering matters in one place: an edge into an inactive motor is refused *before* the fired
    mask is set, so it accumulates nothing and stays out of the learning trace. Masking the
    readout instead would be a different model.
    """
    e_val, e_slog, e_plen, e_act = calcEdgeSignal(state, src, strength, log_str)

    edge_pass = (
        (e_act > hyper.signal_threshold) & alive_e & active[dst] & (~state.fired)
    )
    if not bool(edge_pass.any()):
        return edge_pass

    state.fired |= edge_pass

    carried = torch.where(is_motor[dst], e_act, e_val)
    contrib = torch.where(edge_pass, carried, torch.zeros_like(carried))

    arrived = torch.zeros_like(state.val).index_add_(1, dst, contrib)
    arrived_slog = torch.zeros_like(state.val).index_add_(1, dst, contrib * e_slog)
    arrived_plen = torch.zeros_like(state.val).index_add_(1, dst, contrib * e_plen)
    weight = arrived.clamp_min(1e-12)

    state.eligibility[:, : contrib.shape[1]] += contrib

    state.motor_acc = state.motor_acc + torch.where(
        is_motor.unsqueeze(0), arrived, torch.zeros_like(arrived)
    )

    gate = (arrived >= hyper.neuron_firing_threshold) & is_inter.unsqueeze(0)
    zero = torch.zeros_like(state.val)
    state.val = torch.where(gate, arrived, zero)
    state.slog = torch.where(gate, arrived_slog / weight, zero)
    state.plen = torch.where(gate, arrived_plen / weight, zero)
    return edge_pass


def flowSignalPass(
    mesh,
    inputs: Tensor,
    sensor_ids: Tensor,
    hyper: HazeHyper,
) -> SignalState:
    """Runs a full propagation and returns its final state.

    Every feature of an observation enters the same lane, so an interneuron sees the whole
    observation at once and conjunctive structure can form. Propagating each feature in its own
    lane instead confines the mesh to a sum over features: measured, that collapses the terminus
    representation to a rank of 8-11 out of 33 and puts a linear probe at chance, while one lane
    per observation reaches full rank. See specs/propagation.md.

    Terminates when no edge passes its gate, which is guaranteed: the fired mask is monotone and
    bounded, so each hop either claims a new edge or ends the pass. That is what makes a mesh
    with cycles terminate at all.
    """
    features = int(sensor_ids.numel())
    rows = int(inputs.shape[0]) if inputs.dim() > 1 else 1
    flat = inputs.reshape(rows, features)

    state = makeSignalState(rows, int(mesh.kind.numel()), int(mesh.capacity.edges), mesh.dtype)
    state.val[:, sensor_ids] = flat.to(mesh.dtype)

    live = slice(0, mesh.counts.edges)
    src, dst = mesh.src[live], mesh.dst[live]
    strength, log_str = mesh.strength[live], mesh.log_str[live]
    alive_e = mesh.alive_e[live]
    is_motor, is_inter = mesh.is_motor, mesh.is_inter

    if src.numel() == 0:
        return state

    state.fired = state.fired[:, live]
    state.eligibility = state.eligibility[:, live]
    for _ in range(hyper.max_steps):
        passed = flowSignalStep(
            state, src, dst, strength, log_str, alive_e, mesh.active, is_motor, is_inter, hyper
        )
        if not bool(passed.any()):
            break
        state.hops += 1

    state.reached = bool((state.motor_acc > 0).any())
    return state


def flowSignalPassReference(
    mesh, inputs: Tensor, sensor_ids: Tensor, hyper: HazeHyper
) -> dict[int, float]:
    """Returns motor activation computed edge by edge in plain Python, at any cost.

    This exists to pin what the tensor engine means. The three subtle behaviors — carrying path
    statistics, gating an edge once per observation rather than once per hop, and discarding
    sub-threshold accumulation — all fail silently if implemented wrongly, producing a model
    that runs and never learns. A slow implementation nobody would mistake for clever is the
    thing to check the fast one against.
    """
    import math

    live = mesh.counts.edges
    edges = [
        (int(mesh.src[i]), int(mesh.dst[i]), float(mesh.strength[i]))
        for i in range(live)
        if bool(mesh.alive_e[i])
    ]
    kinds = mesh.kind.tolist()
    active = mesh.active.tolist()

    features = int(sensor_ids.numel())
    flat = inputs.reshape(-1)[:features].tolist()
    motor_acc: dict[int, float] = {}

    node = {
        sensor: (float(flat[feature]), 0.0, 0.0)
        for feature, sensor in enumerate(sensor_ids.tolist())
    }
    claimed: set[int] = set()

    for _ in range(hyper.max_steps):
        arrivals: dict[int, list[tuple[float, float, float]]] = {}
        any_fired = False

        for index, (source, target, weight) in enumerate(edges):
            if index in claimed or source not in node:
                continue
            if kinds[target] == int(NeuronKind.MOTOR) and not active[target]:
                continue
            value, slog, plen = node[source]
            value *= weight
            slog += math.log(weight)
            plen += 1.0
            actual = value * math.exp(slog / max(plen, 1.0))
            if actual <= hyper.signal_threshold:
                continue
            claimed.add(index)
            any_fired = True
            carried = actual if kinds[target] == int(NeuronKind.MOTOR) else value
            arrivals.setdefault(target, []).append((carried, slog, plen))

        if not any_fired:
            break

        node = {}
        for target, incoming in arrivals.items():
            total = sum(c for c, _, _ in incoming)
            if kinds[target] == int(NeuronKind.MOTOR):
                motor_acc[target] = motor_acc.get(target, 0.0) + total
                continue
            if total < hyper.neuron_firing_threshold:
                continue
            weight = max(total, 1e-12)
            node[target] = (
                total,
                sum(c * s for c, s, _ in incoming) / weight,
                sum(c * p for c, _, p in incoming) / weight,
            )

    return motor_acc
