"""Tasks with a real learnable ceiling, and the probe that bounds what a readout could reach."""

import random
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from ..errors import SignalDidNotReachMotorsError
from ..models import ArrivalProfile, FiringProfile, ScoreProfile
from .propagate import SignalState

Task = Callable[[Sequence[int]], Any]
Observer = Callable[[Sequence[int]], SignalState]


def makeBinaryRows(count: int, width: int, seed: int) -> list[list[int]]:
    """Returns `count` rows of `width` random bits, drawn from a generator this call owns."""
    rng = random.Random(seed)
    return [[rng.randint(0, 1) for _ in range(width)] for _ in range(count)]


def scoreConstant(row: Sequence[int]) -> int:
    """Returns the same answer whatever the row says.

    The sanity floor: it needs only a bias, so any learner that cannot reach it is broken and no
    result on a harder task means anything.
    """
    return 1


def scoreCopy(row: Sequence[int]) -> int:
    """Returns the row's first bit, which requires reading one specific feature."""
    return int(row[0])


def scoreMajority(row: Sequence[int]) -> int:
    """Returns whether most of the row's bits are set, which requires reading all of them."""
    return 1 if sum(row) * 2 > len(row) else 0


def scoreParity(row: Sequence[int]) -> int:
    """Returns the row's parity, which no sum over independent features can express.

    Retained as a diagnostic of representational depth rather than as a target: it needs a
    genuine conjunction, so it reports whether the mesh forms one at all.
    """
    return sum(row) % 2


def scoreFirstSet(row: Sequence[int]) -> int:
    """Returns the index of the row's first set bit, or its width if none is set.

    The capacity task: unlike the binary four, what a readout can extract from this one rises
    with the width of the terminus — measured at a matched-probe bound of 0.50 on a terminus of
    4, which is exactly chance, against 0.65 on a terminus of 32. That makes it the only task
    here that can demonstrate growth at all, since growth answers a capacity limit and the others
    are already saturated at initialization. See specs/growth.md.
    """
    for index, bit in enumerate(row):
        if bit:
            return index
    return len(row)


TASKS: dict[str, Task] = {
    "constant": scoreConstant,
    "copy": scoreCopy,
    "majority": scoreMajority,
    "parity": scoreParity,
    "first-set": scoreFirstSet,
}

CHANCE: dict[str, float] = {
    "constant": 0.5,
    "copy": 0.5,
    "majority": 0.5,
    "parity": 0.5,
    "first-set": 0.51,
}
"""Accuracy the majority label alone reaches, which is the floor a result must clear.

Not 1/classes: `first-set` has nine labels but its first is set on half of all rows, so always
answering it scores 0.51 and a nine-class result near 0.5 has learned nothing.
"""


def calcRewardMean(rewards: Sequence[float], window: int = 200) -> float:
    """Returns the mean reward over the last `window` steps, or over all of them if fewer ran."""
    if not rewards:
        return 0.0
    tail = rewards[-window:] if window > 0 else list(rewards)
    return sum(tail) / len(tail)


def calcConductionReach(
    answer: Callable[[Sequence[int]], Any], rows: Sequence[Sequence[int]]
) -> float:
    """Returns the share of `rows` the mesh still answers at all, rather than answers correctly.

    The measurement accuracy hides. A trained mesh does not merely answer its harder inputs wrongly
    — it stops answering them, because learning drives their edges under the gate and a mute edge
    fires no trace to recover on. Measured on `first-set`, this falls from 1.00 on a fresh mesh to
    0.52 by step 1000, and the inputs that survive share one label: an accuracy of 0.51 there is
    not a mesh predicting the majority class but a mesh that only still conducts for it.

    Kept beside the probe because the two bound different things. The probe says what a readout
    could extract; this says how much of the input space is still connected to a readout at all.
    See specs/propagation.md.
    """
    if not rows:
        return 0.0
    reached = 0
    for row in rows:
        try:
            answer(row)
            reached += 1
        except SignalDidNotReachMotorsError:
            continue
    return reached / len(rows)


def calcScoreProfile(
    answer: Callable[[Sequence[int]], Any], rows: Sequence[Sequence[int]], task: Task
) -> ScoreProfile:
    """Returns how much of the task the mesh answered and how well it answered that much.

    The measurement raw accuracy cannot make. A trained mesh both declines to answer and answers
    wrongly, and one score cannot separate them: measured on `first-set` at three labels, a mesh
    scoring 0.57 is answering 59% of its inputs at 96% accuracy, not answering all of them badly.

    The baseline is the majority share **of the answered set**, not of the task, because the mesh
    chooses what it answers. Its silences fall preferentially on minority labels, so the set it
    keeps is skewed — 83% one label at three labels — and always guessing that label would score
    0.83 with no skill at all. Scoring conditional accuracy against the task's own majority share
    reported a lift of +0.46 where the honest figure is +0.13.
    """
    answered, correct, seen = 0, 0, []
    for row in rows:
        want = task(row)
        try:
            got = answer(row)
        except SignalDidNotReachMotorsError:
            continue
        answered += 1
        seen.append(want)
        correct += 1 if got == want else 0

    if not rows:
        return ScoreProfile()
    if answered == 0:
        return ScoreProfile(coverage=0.0)
    return ScoreProfile(
        coverage=answered / len(rows),
        accuracy=correct / len(rows),
        conditional=correct / answered,
        baseline=max(seen.count(label) for label in set(seen)) / answered,
    )


def calcReachByLabel(
    answer: Callable[[Sequence[int]], Any], rows: Sequence[Sequence[int]], task: Task
) -> dict[Any, float]:
    """Returns conduction reach split by each row's correct label, worst-served label first.

    The split is what makes the failure legible: the aggregate falls smoothly while the truth is
    that whole labels drop to zero and the majority label stays at one.
    """
    total: dict[Any, int] = {}
    reached: dict[Any, int] = {}
    for row in rows:
        want = task(row)
        total[want] = total.get(want, 0) + 1
        try:
            answer(row)
            reached[want] = reached.get(want, 0) + 1
        except SignalDidNotReachMotorsError:
            continue
    shares = {want: reached.get(want, 0) / count for want, count in total.items()}
    return dict(sorted(shares.items(), key=lambda pair: pair[1]))


def calcFiringShare(state: SignalState, live_edges: int) -> float:
    """Returns the share of live edges that carried signal on one observation.

    The figure specs/propagation.md reports at 0.89-0.91 and which no function computed until now,
    so it was produced by hand and could not be regression-tested. It is the primary readout for
    any change aiming to make firing input-conditional.
    """
    if live_edges <= 0:
        return 0.0
    return float(state.toEdgeTrace().sum()) / live_edges


def calcFiringProfile(observe: Observer, rows: Sequence[Sequence[int]], task: Task) -> FiringProfile:
    """Returns how much of the mesh each observation used, and how much of it was label-specific.

    Takes an observer rather than a model for the same reason `calcScoreProfile` takes an answerer:
    score.py stays decoupled from how a caller drives propagation.

    The overlap halves are O(pairs), so this is written for tens of rows rather than thousands. A
    larger sample buys precision on a quantity whose fresh-mesh value is near zero by construction,
    which is not where the measurement is hard.
    """
    if not rows:
        return FiringProfile()

    traces: list[tuple[Any, torch.Tensor]] = []
    shares, hops, neurons = [], [], []
    for row in rows:
        state = observe(row)
        live = int(state.fired.shape[1])
        traces.append((task(row), state.toEdgeTrace()))
        shares.append(calcFiringShare(state, live))
        hops.append(float(state.hops))
        neurons.append(float((state.node_acc.sum(0) != 0).sum()))

    within, between = [], []
    for left in range(len(traces)):
        for right in range(left + 1, len(traces)):
            label_l, trace_l = traces[left]
            label_r, trace_r = traces[right]
            union = float((trace_l | trace_r).sum())
            overlap = float((trace_l & trace_r).sum()) / union if union > 0 else 0.0
            (within if label_l == label_r else between).append(overlap)

    return FiringProfile(
        share=sum(shares) / len(shares),
        hops=sum(hops) / len(hops),
        neurons=sum(neurons) / len(neurons),
        within_label_overlap=sum(within) / len(within) if within else 0.0,
        between_label_overlap=sum(between) / len(between) if between else 0.0,
    )


def calcArrivalProfile(mesh, states: Sequence[SignalState]) -> ArrivalProfile:
    """Returns how much of what arrives at an interneuron is explained by its wiring.

    Correlating in-degree against arriving magnitude is what says whether a top-k over arrivals
    would select relevance or merely topology. On the shipped economy it reads +0.637, so it would
    select topology, and the same neurons would win for every input.
    """
    if not states:
        return ArrivalProfile()

    live = mesh.counts.edges
    neurons = int(mesh.kind.numel())
    fanin = torch.zeros(neurons, dtype=mesh.dtype)
    if live > 0:
        fanin.index_add_(0, mesh.dst[:live], torch.ones(live, dtype=mesh.dtype))

    arrival = torch.zeros(neurons, dtype=mesh.dtype)
    for state in states:
        arrival += state.toNodeActivation().abs()
    arrival /= len(states)

    inter = mesh.is_inter
    degree, magnitude = fanin[inter], arrival[inter]
    gains = [
        max(state.frontier) / state.frontier[0]
        for state in states
        if state.frontier and state.frontier[0] > 0
    ]

    return ArrivalProfile(
        fanin_correlation=calcCorrelation(degree, magnitude),
        arrival_cv=float(magnitude.std() / magnitude.mean()) if float(magnitude.mean()) > 0 else 0.0,
        depth_gain=sum(gains) / len(gains) if gains else 0.0,
        median_arrival=float(magnitude.median()) if magnitude.numel() else 0.0,
    )


def calcCorrelation(left: Tensor, right: Tensor) -> float:
    """Returns the Pearson correlation of two equal-length vectors, or zero if either is constant."""
    if left.numel() < 2:
        return 0.0
    a = left.to(torch.float64) - left.to(torch.float64).mean()
    b = right.to(torch.float64) - right.to(torch.float64).mean()
    scale = float(a.norm() * b.norm())
    return float((a * b).sum() / scale) if scale > 0 else 0.0


def fitReadoutProbe(
    features: Tensor,
    targets: Tensor,
    classes: int,
    lower: float,
    upper: float,
    holdout: float = 0.3,
    steps: int = 1500,
    rate: float = 0.05,
    seed: int = 0,
) -> float:
    """Returns held-out accuracy for the best readout Haze's own weights could express.

    Deliberately matched to the readout it bounds rather than to whatever is easiest to fit. An
    unconstrained least-squares probe answers a different and far more flattering question: it
    fits signed unbounded weights, carries a bias term the mesh has no equivalent of, and scores
    on the data it was fitted to. Each of those inflates the ceiling, and a ceiling that is not
    reachable is worse than no ceiling at all, because work gets spent chasing it.

    So: one weight per (neuron, class) clamped to the same range the mesh clamps strengths to,
    no bias, argmax across classes, and accuracy reported on data held out of the fit.
    """
    rows = int(features.shape[0])
    if rows < 4:
        return 0.0
    if classes < 2:
        return 1.0

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(rows, generator=generator)
    cut = max(1, int(rows * (1.0 - holdout)))
    train, test = order[:cut], order[cut:]
    if test.numel() == 0:
        return 0.0

    x = features.to(torch.float64)
    scale = x.abs().max().clamp_min(1e-9)
    x = x / scale
    y = targets.to(torch.int64)

    weights = (
        torch.rand(x.shape[1], classes, dtype=torch.float64, generator=generator)
        * (upper - lower)
        + lower
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([weights], lr=rate)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x[train] @ weights, y[train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights.clamp_(lower, upper)

    with torch.no_grad():
        predicted = (x[test] @ weights).argmax(dim=1)
        return float((predicted == y[test]).to(torch.float64).mean())


def calcRepresentationRank(features: Tensor) -> tuple[int, int]:
    """Returns how many independent directions a representation spans, and how many it could.

    A rank far below the neuron count means the layer's activations lie in a small subspace, so
    different inputs are not actually being represented differently and no readout can recover
    what was never encoded.
    """
    matrix = features.to(torch.float64)
    if matrix.numel() == 0:
        return 0, 0
    return int(torch.linalg.matrix_rank(matrix)), int(matrix.shape[1])


def probeRepresentation(
    features: Tensor,
    rows: Sequence[Sequence[int]],
    hyper: Any,
    tasks: dict[str, Task] | None = None,
) -> dict[str, Any]:
    """Returns what the recorded activations can support, per task, under a matched readout.

    This is the instrument every structural change is judged by. A change that raises these
    numbers improved the representation; a change that raises achieved reward without raising
    them improved the learning rule. Distinguishing the two is the whole point.

    Takes the hyperparameters rather than a weight range because the range a caller would reach
    for is the wrong one. Once inhibition exists the readout's floor is `calcStrengthFloor()`,
    not `strength_lower`, and passing the latter reports a bound the mesh has already beaten —
    which is worse than no bound, since it retires work that is not finished.
    """
    tasks = tasks or TASKS
    lower, upper = hyper.calcStrengthFloor(), hyper.strength_upper
    rank, dimension = calcRepresentationRank(features)
    bounds: dict[str, float] = {}
    for name, task in tasks.items():
        answers = [task(row) for row in rows]
        labels = sorted({int(a) for a in answers})
        index = {label: position for position, label in enumerate(labels)}
        targets = torch.tensor([index[int(a)] for a in answers], dtype=torch.int64)
        bounds[name] = fitReadoutProbe(features, targets, len(labels), lower, upper)
    return {"rank": rank, "dimension": dimension, "bounds": bounds}
