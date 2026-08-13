"""Tests for signal propagation and forward-only learning."""

import math

import pytest
import torch

from haze import makeHaze
from haze.errors import InvalidRewardError, SignalRangeError
from haze.functions.learning import applyLearning, calcConfidenceEntropy
from haze.functions.prune import calcPruneMask
from haze.functions.propagate import flowSignalPass, flowSignalPassReference
from haze.modules.decoders import ArgMax, Regressor, SoftMax
from haze.modules.encoders import NumericEncoder, toSignalBand
from haze.tokens import NeuronKind, defaults


def makeWiredModel(seed: int, features: int = 4, labels: int = 3):
    """Returns a model with sensors and motors already wired, and their ids."""
    model = makeHaze(nexus_size=10, terminus_size=5, seed=seed)
    sensors = model.mesh.allocNeuronIds(features, NeuronKind.SENSOR, owner=0)
    model.mesh.connectSensors(sensors)
    motors = model.mesh.allocNeuronIds(labels, NeuronKind.MOTOR, owner=1)
    model.mesh.connectMotors(motors)
    model.mesh.active[motors] = True
    return model, sensors, motors


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13])
def test_flowSignalPass_doesMatchTheReferenceImplementation(seed: int):
    """Asserts the tensor engine computes what the plain-Python reference computes.

    The three subtle behaviors — carrying path statistics, gating an edge once per observation
    rather than once per hop, and discarding sub-threshold accumulation — all fail silently if
    implemented wrongly, producing a model that runs and never learns.
    """
    model, sensors, motors = makeWiredModel(seed)
    inputs = torch.tensor([[0.9, 0.5, 0.7, 0.3]])

    state = flowSignalPass(model.mesh, inputs, sensors, model.config.hyper)
    reference = flowSignalPassReference(model.mesh, inputs, sensors, model.config.hyper)

    engine = state.motor_acc.sum(0)
    for motor in motors.tolist():
        assert float(engine[motor]) == pytest.approx(reference.get(motor, 0.0), abs=1e-4)


def test_flowSignalPass_doesTerminateOnACyclicMesh():
    """Asserts propagation ends even though the nexus mesh contains cycles.

    The fired mask is monotone and bounded, so each hop either claims a new edge or ends the
    pass. Without it a cyclic mesh would never stop.
    """
    model, sensors, _ = makeWiredModel(seed=4)
    inputs = torch.full((1, 4), 0.9)

    state = flowSignalPass(model.mesh, inputs, sensors, model.config.hyper)

    assert state.hops < model.config.hyper.max_steps
    assert state.reached


def test_flowSignalPass_doesFireEachEdgeAtMostOncePerLane():
    """Asserts an edge that has carried signal is not re-fired within the same observation."""
    model, sensors, _ = makeWiredModel(seed=6)
    inputs = torch.full((1, 4), 0.9)

    state = flowSignalPass(model.mesh, inputs, sensors, model.config.hyper)

    assert state.fired.dtype == torch.bool
    assert int(state.toEdgeTrace().sum()) <= model.mesh.counts.edges


def test_flowSignalPass_doesNotDeliverToAnInactiveMotor():
    """Asserts an inactive motor accumulates nothing and stays out of the learning trace.

    Masking at readout instead would let it accumulate and would place its inbound edges into
    the trace, which is a different model.
    """
    model, sensors, motors = makeWiredModel(seed=7)
    silenced = motors[0]
    model.mesh.active[silenced] = False
    inputs = torch.full((1, 4), 0.9)

    state = flowSignalPass(model.mesh, inputs, sensors, model.config.hyper)

    assert float(state.motor_acc.sum(0)[silenced]) == 0.0
    trace = state.toEdgeTrace()
    live = slice(0, model.mesh.counts.edges)
    into_silenced = model.mesh.dst[live] == silenced
    assert not bool((trace & into_silenced).any())


def test_calcEdgeSignal_doesApplyTheGeometricMeanCorrection():
    """Asserts a single hop's actual value equals value times the traversed geometric mean."""
    model = makeHaze(nexus_size=2, terminus_size=1, seed=1)
    mesh = model.mesh
    sensor = mesh.allocNeuronIds(1, NeuronKind.SENSOR, owner=0)
    target = mesh.findNeuronIds(NeuronKind.NEXUS)[:1]
    mesh.compactEdges(torch.zeros(mesh.capacity.edges, dtype=torch.bool))
    mesh.addMeshEdges(sensor, target, strength=torch.tensor([0.8]))

    state = flowSignalPass(model.mesh, torch.tensor([[0.9]]), sensor, model.config.hyper)

    expected = 0.9 * 0.8
    assert float(state.val[0, target[0]]) == pytest.approx(expected, abs=1e-5)


def test_applyLearning_doesMoveOnlyTheEdgesThatFired():
    """Asserts the update reaches the traced edges and leaves the rest untouched."""
    model, sensors, _ = makeWiredModel(seed=9)
    mesh = model.mesh
    live = mesh.counts.edges
    trace = torch.zeros(live, dtype=torch.bool)
    trace[:5] = True
    before = mesh.strength[:live].clone()

    result = applyLearning(mesh, trace, reward=1.0, confidence=0.0, hyper=model.config.hyper)

    assert result.edges_updated == 5
    assert bool((mesh.strength[:5] > before[:5]).all())
    assert torch.equal(mesh.strength[5:live], before[5:live])


def test_applyLearning_doesInvertTheMaskWhenReversed():
    """Asserts reverse learning strengthens the edges that did not carry signal."""
    model, _, _ = makeWiredModel(seed=9)
    mesh = model.mesh
    live = mesh.counts.edges
    trace = torch.zeros(live, dtype=torch.bool)
    trace[:5] = True

    result = applyLearning(
        mesh, trace, reward=1.0, confidence=0.0, hyper=model.config.hyper, reverse=True
    )

    assert result.edges_updated == live - 5


def test_applyLearning_doesDecayEpsilonOnEveryUpdatedEdge():
    """Asserts an edge's learning rate shrinks each time it is moved."""
    model, _, _ = makeWiredModel(seed=9)
    mesh = model.mesh
    trace = torch.ones(mesh.counts.edges, dtype=torch.bool)
    before = mesh.epsilon[0].clone()

    applyLearning(mesh, trace, reward=1.0, confidence=0.5, hyper=model.config.hyper)

    assert float(mesh.epsilon[0]) == pytest.approx(
        float(before) * model.config.hyper.epsilon_decay, rel=1e-6
    )


def test_applyLearning_doesRefreshTheCachedLogStrength():
    """Asserts the cached log of each strength is rebuilt, since propagation reads it."""
    model, _, _ = makeWiredModel(seed=9)
    mesh = model.mesh
    trace = torch.ones(mesh.counts.edges, dtype=torch.bool)

    applyLearning(mesh, trace, reward=1.0, confidence=0.0, hyper=model.config.hyper)

    live = slice(0, mesh.counts.edges)
    assert torch.allclose(mesh.log_str[live], mesh.strength[live].abs().log(), atol=1e-5)


def test_calcConfidenceEntropy_doesPeakOnAConcentratedActivation():
    """Asserts a single dominant motor reads as confident and a uniform one as not."""
    assert calcConfidenceEntropy(torch.tensor([10.0, 0.0001])) > 0.9
    assert calcConfidenceEntropy(torch.tensor([1.0, 1.0])) < 0.01


def test_toSignalBand_doesSpanTheBand():
    """Asserts a varying input is mapped across the whole band."""
    scaled = toSignalBand(torch.tensor([0.0, 5.0, 10.0]))

    assert float(scaled.min()) == pytest.approx(defaults.SIGNAL_LOWER)
    assert float(scaled.max()) == pytest.approx(defaults.SIGNAL_UPPER)


def test_encodeFeatures_doesDistinguishConstantRowsUnderRunningNormalization():
    """Asserts an all-zero row and an all-one row encode differently.

    Per-row min-max maps both to the same constant, which makes them indistinguishable to the
    mesh no matter how well it learns, and places them at the band floor where they cannot pass
    the edge gate at all.
    """
    encoder = NumericEncoder(norm="running")
    encoder.key = "bits"

    encoder.encodeFeatures([0, 1, 0, 1])
    zeros = encoder.encodeFeatures([0, 0, 0, 0])
    ones = encoder.encodeFeatures([1, 1, 1, 1])

    assert not torch.allclose(zeros, ones)
    assert float(ones.min()) > defaults.SIGNAL_THRESHOLD


def test_toSignalBand_doesPlaceAConstantInputMidBand():
    """Asserts an input with no variation is placed where it can still propagate.

    The band floor cannot pass the edge gate under any strength, so a constant observation would
    be indistinguishable from not having observed at all.
    """
    scaled = toSignalBand(torch.tensor([3.0, 3.0, 3.0]))

    assert float(scaled.min()) > defaults.SIGNAL_THRESHOLD
    assert float(scaled.max()) <= defaults.SIGNAL_UPPER


def test_flowSignalPass_doesReturnQuietlyWhenTheMeshHasNoEdges():
    """Asserts an unwired mesh yields no activation rather than raising."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    sensors = model.mesh.allocNeuronIds(2, NeuronKind.SENSOR, owner=0)
    model.mesh.compactEdges(torch.zeros(model.mesh.capacity.edges, dtype=torch.bool))

    state = flowSignalPass(model.mesh, torch.tensor([[0.9, 0.9]]), sensors, model.config.hyper)

    assert not state.reached
    assert float(state.motor_acc.sum()) == 0.0


def test_applyLearning_doesClampAtBothRails():
    """Asserts a large update cannot carry a strength outside its bounds."""
    model, _, _ = makeWiredModel(seed=9)
    mesh = model.mesh
    hyper = model.config.hyper
    trace = torch.ones(mesh.counts.edges, dtype=torch.bool)

    applyLearning(mesh, trace, reward=1.0, confidence=0.0, hyper=hyper)
    applyLearning(mesh, trace, reward=1.0, confidence=0.0, hyper=hyper)
    live = slice(0, mesh.counts.edges)
    assert float(mesh.strength[live].max()) <= hyper.strength_upper + 1e-6

    for _ in range(6):
        applyLearning(mesh, trace, reward=0.0, confidence=1.0, hyper=hyper)
    assert float(mesh.strength[live].min()) >= hyper.calcStrengthFloor() - 1e-6


def test_calcConfidenceEntropy_doesHandleTheDegenerateCases():
    """Asserts a single label and a zero activation both return a defined confidence."""
    assert calcConfidenceEntropy(torch.tensor([5.0])) == 1.0
    assert calcConfidenceEntropy(torch.tensor([0.0, 0.0])) == 0.0
    assert calcConfidenceEntropy(torch.tensor([])) == 0.0


def test_applyLearning_doesRefuseANonFiniteReward():
    """Asserts NaN is refused rather than poisoning every strength in the mask."""
    model, _, _ = makeWiredModel(seed=9)
    trace = torch.ones(model.mesh.counts.edges, dtype=torch.bool)

    with pytest.raises(InvalidRewardError, match="reward"):
        applyLearning(model.mesh, trace, reward=math.nan, confidence=0.0, hyper=model.config.hyper)

    with pytest.raises(InvalidRewardError, match="confidence"):
        applyLearning(model.mesh, trace, reward=1.0, confidence=math.inf, hyper=model.config.hyper)

    assert bool(torch.isfinite(model.mesh.strength).all())


def test_encodeFeatures_doesRefuseANonFiniteObservation():
    """Asserts an infinite input is refused at the encoder rather than saturating the mesh."""
    encoder = NumericEncoder()
    encoder.key = "bits"

    with pytest.raises(SignalRangeError):
        encoder.encodeFeatures([0.0, float("inf")])


def test_decodeMotors_doesRefuseWhenNoMotorReceivedSignal():
    """Asserts a decoder with no signal raises rather than returning an arbitrary label."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    entry = model.ports.findLabels("bit")

    with pytest.raises(Exception, match="did not|no active motor"):
        ArgMax(labels=[0, 1]).decodeMotors(torch.tensor([0.0, 0.0]), entry)


def test_calcProbabilities_doesNotOverflowOnLargeActivation():
    """Asserts a softmax decoder survives the unbounded sums motor accumulation produces.

    The prior engine raised OverflowError here once a run went on long enough, because motor
    state was never reset and the exponential was taken on the raw sum.
    """
    decoder = SoftMax(labels=[0, 1])
    probabilities = decoder.calcProbabilities(torch.tensor([10_000.0, 9_999.0]))

    assert bool(torch.isfinite(probabilities).all())
    assert float(probabilities.sum()) == pytest.approx(1.0)


def test_calcPruneMask_doesSelectOnlyWeakLiveEdges():
    """Asserts pruning selects edges at or below the threshold and never a dead slot."""
    model, _, _ = makeWiredModel(seed=9)
    mesh = model.mesh
    mesh.strength[: mesh.counts.edges] = 0.9
    mesh.strength[0] = model.config.hyper.prune_threshold

    mask = calcPruneMask(mesh, model.config.hyper)

    assert int(mask.sum()) == 1
    assert bool(mask[0])


def test_decodeMotors_doesReturnANumberFromARegressor():
    """Asserts a regressor answers with the activation-weighted centre of its labels."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    model.registerDecoder("value", Regressor(labels=[1.0, 10.0]))
    entry = model.ports.findLabels("value")

    answer = model.ports.impls["value"].decodeMotors(torch.tensor([0.3, 0.7]), entry)

    assert answer == pytest.approx((0.3 * 1.0 + 0.7 * 10.0) / 1.0)


def test_addMeshEdges_doesDrawAMixOfExcitatoryAndInhibitoryEdges():
    """Asserts new edges include negative strengths, so the mesh can express a veto.

    With only positive attenuating strengths a mesh can excite but never rule an answer out.
    """
    model, _, _ = makeWiredModel(seed=11)
    live = slice(0, model.mesh.counts.edges)
    strengths = model.mesh.strength[live]

    assert bool((strengths < 0).any()), "no inhibitory edge was drawn"
    assert bool((strengths > 0).any()), "no excitatory edge was drawn"


def test_addMeshEdges_doesLogMagnitudeSoInhibitionIsDefined():
    """Asserts the cached log holds the magnitude, since log of a negative strength is undefined.

    The sign belongs to the signal rather than to the attenuation: it is the value that inverts,
    while the amount of attenuation is a magnitude.
    """
    model, _, _ = makeWiredModel(seed=11)
    live = slice(0, model.mesh.counts.edges)

    assert bool(torch.isfinite(model.mesh.log_str[live]).all())
    assert torch.allclose(
        model.mesh.log_str[live], model.mesh.strength[live].abs().clamp_min(1e-6).log(), atol=1e-5
    )


def test_calcPruneMask_doesKeepStronglyInhibitoryEdges():
    """Asserts pruning drops edges near zero rather than every negative one.

    A strongly negative edge is a strongly inhibitory one and carries as much information as a
    strongly positive one; testing the signed value would delete inhibition on creation.
    """
    model, _, _ = makeWiredModel(seed=11)
    mesh = model.mesh
    mesh.strength[: mesh.counts.edges] = -0.9
    mesh.strength[0] = 0.0

    mask = calcPruneMask(mesh, model.config.hyper)

    assert int(mask.sum()) == 1
    assert bool(mask[0])


def test_flowSignalPass_doesCarrySignThroughAnInhibitoryEdge():
    """Asserts an inhibitory edge inverts the signal it carries rather than dropping it."""
    model = makeHaze(nexus_size=2, terminus_size=1, seed=1)
    mesh = model.mesh
    sensor = mesh.allocNeuronIds(1, NeuronKind.SENSOR, owner=0)
    target = mesh.findNeuronIds(NeuronKind.NEXUS)[:1]
    mesh.compactEdges(torch.zeros(mesh.capacity.edges, dtype=torch.bool))
    mesh.addMeshEdges(sensor, target, strength=torch.tensor([-0.8]))

    state = flowSignalPass(mesh, torch.tensor([[0.9]]), sensor, model.config.hyper)

    assert float(state.node_acc.sum(0)[target[0]]) < 0
