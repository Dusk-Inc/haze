"""Tasks with a real learnable ceiling, and the probe that bounds what a readout could reach."""

import random
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

Task = Callable[[Sequence[int]], Any]


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


TASKS: dict[str, Task] = {
    "constant": scoreConstant,
    "copy": scoreCopy,
    "majority": scoreMajority,
    "parity": scoreParity,
}

CHANCE: dict[str, float] = {"constant": 0.5, "copy": 0.5, "majority": 0.5, "parity": 0.5}


def calcRewardMean(rewards: Sequence[float], window: int = 200) -> float:
    """Returns the mean reward over the last `window` steps, or over all of them if fewer ran."""
    if not rewards:
        return 0.0
    tail = rewards[-window:] if window > 0 else list(rewards)
    return sum(tail) / len(tail)


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
    lower: float,
    upper: float,
    tasks: dict[str, Task] | None = None,
) -> dict[str, Any]:
    """Returns what the recorded activations can support, per task, under a matched readout.

    This is the instrument every structural change is judged by. A change that raises these
    numbers improved the representation; a change that raises achieved reward without raising
    them improved the learning rule. Distinguishing the two is the whole point.
    """
    tasks = tasks or TASKS
    rank, dimension = calcRepresentationRank(features)
    bounds: dict[str, float] = {}
    for name, task in tasks.items():
        answers = [task(row) for row in rows]
        labels = sorted({int(a) for a in answers})
        index = {label: position for position, label in enumerate(labels)}
        targets = torch.tensor([index[int(a)] for a in answers], dtype=torch.int64)
        bounds[name] = fitReadoutProbe(features, targets, len(labels), lower, upper)
    return {"rank": rank, "dimension": dimension, "bounds": bounds}
