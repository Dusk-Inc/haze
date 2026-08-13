"""Tests for lane semantics, the band/gate contract, and the probe that bounds a readout."""

import pytest
import torch

from haze import makeHaze
from haze.functions.propagate import flowSignalPass
from haze.functions.score import (
    TASKS,
    calcRepresentationRank,
    calcRewardMean,
    fitReadoutProbe,
    makeBinaryRows,
    probeRepresentation,
    scoreConstant,
    scoreCopy,
    scoreMajority,
    scoreParity,
)
from haze.models import HazeHyper
from haze.modules.encoders import NumericEncoder
from haze.modules.decoders import ArgMax
from haze.tokens import NeuronKind, defaults


def makeProbeModel(seed: int, features: int = 8):
    """Returns a wired model plus its encoder, sensors, and terminus ids, ready to probe."""
    model = makeHaze(nexus_size=48, terminus_size=24, seed=seed)
    encoder = NumericEncoder()
    model.registerEncoder("bits", encoder)
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    encoder.key = "bits"
    encoder.encodeFeatures([0, 1] * (features // 2))
    sensors = model.ports.ensureSensorCapacity("bits", features)
    return model, encoder, sensors, model.mesh.findNeuronIds(NeuronKind.TERMINUS)


def collectTerminus(seed: int, rows: list[list[int]]) -> torch.Tensor:
    """Returns the terminus activation each row produces, which is what the motors read."""
    model, encoder, sensors, terminus = makeProbeModel(seed, len(rows[0]))
    collected = []
    for row in rows:
        state = flowSignalPass(
            model.mesh, encoder.encodeFeatures(row).unsqueeze(0), sensors, model.config.hyper
        )
        collected.append(state.toNodeActivation()[terminus])
    return torch.stack(collected)


# -- Domain --------------------------------------------------------------------------------


def test_flowSignalPass_doesGiveOneLanePerObservation():
    """Asserts every feature enters the same lane, so an interneuron sees the whole observation.

    One lane per feature confines the mesh to a sum over independent features, which is what
    collapsed the terminus representation. See specs/propagation.md.
    """
    model, encoder, sensors, _ = makeProbeModel(seed=3)

    state = flowSignalPass(
        model.mesh, encoder.encodeFeatures([1, 0, 1, 0, 1, 0, 1, 0]).unsqueeze(0),
        sensors, model.config.hyper,
    )

    assert state.lanes == 1


def test_flowSignalPass_doesMakeInterneuronsIntegrateAcrossFeatures():
    """Asserts a neuron's activation is not the sum of what each feature would produce alone.

    If it were, the mesh would be additive in features by construction and no conjunction could
    ever form.
    """
    model, encoder, sensors, terminus = makeProbeModel(seed=3, features=8)
    hyper = model.config.hyper

    def terminusFor(row):
        """Returns terminus activation for one row."""
        state = flowSignalPass(
            model.mesh, encoder.encodeFeatures(row).unsqueeze(0), sensors, hyper
        )
        return state.toNodeActivation()[terminus]

    encoder.encodeFeatures([0] * 8)
    encoder.encodeFeatures([1] * 8)
    both = terminusFor([1, 1, 1, 1, 0, 0, 0, 0])
    first = terminusFor([1, 1, 0, 0, 0, 0, 0, 0])
    second = terminusFor([0, 0, 1, 1, 0, 0, 0, 0])

    assert float(both.abs().sum()) > 0, "no signal reached the terminus at all"
    assert not torch.allclose(both, first + second, atol=1e-3)


def test_calcRepresentationRank_doesReportFullRankUnderPerRowLanes():
    """Asserts the terminus representation spans its full dimensionality.

    Measured at 8-11 of 33 under one lane per feature, which put every linear probe at chance
    because different inputs were not being represented differently.
    """
    rows = makeBinaryRows(200, 8, seed=5)
    features = collectTerminus(5, rows)

    rank, dimension = calcRepresentationRank(features)

    assert rank >= dimension * 0.75, (
        f"terminus representation spans only {rank} of {dimension} directions; one lane per "
        "feature measured 8-11 of 33"
    )


def test_probeRepresentation_doesReportABoundPerTask():
    """Asserts the probe returns a rank and a per-task bound under the mesh's own weight range."""
    rows = makeBinaryRows(200, 8, seed=5)
    features = collectTerminus(5, rows)
    hyper = HazeHyper()

    report = probeRepresentation(features, rows, hyper.strength_lower, hyper.strength_upper)

    assert report["rank"] > 0
    assert set(report["bounds"]) == set(TASKS)
    assert all(0.0 <= v <= 1.0 for v in report["bounds"].values())


def test_fitReadoutProbe_doesRecoverASeparableTarget():
    """Asserts the probe finds a readout that exists, so a low bound means the task is hard."""
    signal = torch.randn(300, 6, generator=torch.Generator().manual_seed(1))
    targets = (signal[:, 0] > 0).long()
    features = torch.cat([signal, targets.unsqueeze(1).float() * 4.0], dim=1)

    accuracy = fitReadoutProbe(features, targets, classes=2, lower=0.1, upper=0.9)

    assert accuracy > 0.85


def test_scoreTasks_doesDefineTheirOwnCeilings():
    """Asserts each task computes what its name says, so a reported bound means something."""
    assert scoreConstant([0, 1, 0, 1]) == 1
    assert scoreCopy([1, 0, 0, 0]) == 1 and scoreCopy([0, 1, 1, 1]) == 0
    assert scoreMajority([1, 1, 1, 0]) == 1 and scoreMajority([1, 0, 0, 0]) == 0
    assert scoreParity([1, 1, 0, 0]) == 0 and scoreParity([1, 0, 0, 0]) == 1


def test_makeBinaryRows_doesProduceIndependentRows():
    """Asserts rows are drawn from one generator rather than one seeded per row.

    A generator re-seeded per row yields near-identical rows, which silently collapses the
    dataset and reports a rank of 2 on a representation that is actually full rank.
    """
    rows = makeBinaryRows(200, 8, seed=1)

    assert len({tuple(r) for r in rows}) > 50
    assert 0.3 < sum(sum(r) for r in rows) / (200 * 8) < 0.7


# -- Boundary ------------------------------------------------------------------------------


def test_fitReadoutProbe_doesReturnCertaintyForASingleClass():
    """Asserts a target with one class reads as fully predictable rather than as failure."""
    features = torch.randn(50, 4)
    targets = torch.zeros(50, dtype=torch.long)

    assert fitReadoutProbe(features, targets, classes=1, lower=0.1, upper=0.9) == 1.0


def test_fitReadoutProbe_doesDeclineTooFewRows():
    """Asserts a probe with nothing to hold out reports nothing rather than a fabricated score."""
    assert fitReadoutProbe(torch.randn(2, 3), torch.tensor([0, 1]), 2, 0.1, 0.9) == 0.0


def test_calcRewardMean_doesFallBackToEveryStep():
    """Asserts a window longer than the run averages what there is."""
    assert calcRewardMean([1.0, 0.0], window=100) == 0.5
    assert calcRewardMean([]) == 0.0


def test_calcRepresentationRank_doesHandleAnEmptyRecording():
    """Asserts probing before anything was recorded returns zero rather than raising."""
    assert calcRepresentationRank(torch.empty(0, 0)) == (0, 0)


# -- Error ---------------------------------------------------------------------------------


def test_ensureHyperCoherent_doesRefuseAMuteSignalBand():
    """Asserts a band floor that cannot traverse one edge is refused at construction.

    At the inherited floor of 0.1 against a gate of 0.3, a feature at its minimum fired no edge
    at all and was indistinguishable from not having been observed.
    """
    with pytest.raises(ValueError, match="mute"):
        HazeHyper(signal_lower=0.1, signal_threshold=0.3)


def test_ensureHyperCoherent_doesAcceptTheShippedDefaults():
    """Asserts the defaults satisfy their own coherence rule."""
    hyper = HazeHyper()

    reachable = hyper.signal_lower * hyper.strength_init_upper**2
    assert reachable > hyper.signal_threshold
    assert defaults.SIGNAL_LOWER < defaults.SIGNAL_UPPER


def test_ensureHyperCoherent_doesRefuseAnInvertedBand():
    """Asserts a band whose floor is above its ceiling is refused."""
    with pytest.raises(ValueError):
        HazeHyper(signal_lower=0.9, signal_upper=0.4)


# -- Chaos ---------------------------------------------------------------------------------


def test_probeRepresentation_doesSurviveADegenerateRecording():
    """Asserts an all-identical representation reports rank 1 and chance-level bounds."""
    rows = makeBinaryRows(80, 8, seed=2)
    features = torch.ones(len(rows), 6)

    report = probeRepresentation(features, rows, 0.1, 0.9)

    assert report["rank"] <= 1
    assert report["bounds"]["parity"] < 0.75


def test_encodeFeatures_doesEmitInsideTheShippedBand():
    """Asserts the encoder respects the band the thresholds were reconciled against."""
    encoder = NumericEncoder()
    encoder.key = "bits"
    encoder.encodeFeatures([0, 1])

    for row in ([0, 0, 0], [1, 1, 1], [0, 1, 0], [5, -5, 0]):
        encoded = encoder.encodeFeatures(row)
        assert float(encoded.min()) >= defaults.SIGNAL_LOWER - 1e-6
        assert float(encoded.max()) <= defaults.SIGNAL_UPPER + 1e-6
