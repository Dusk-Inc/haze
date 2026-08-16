"""The signal economy's instruments: what fires, what arrives, and how much of it tracks wiring."""

import pytest
import torch

from haze import makeHaze
from haze.functions.propagate import flowSignalPass
from haze.functions.score import (
    TASKS,
    calcArrivalProfile,
    calcCorrelation,
    calcFiringProfile,
    calcFiringShare,
    makeBinaryRows,
)
from haze.models import ArrivalProfile, FiringProfile
from haze.modules.decoders import ArgMax
from haze.modules.encoders import NumericEncoder


def makeObservedModel(seed: int, features: int = 8, **hyper):
    """Returns a model wired for the binary suite, and an observer returning its raw signal state."""
    model = makeHaze(nexus_size=64, terminus_size=32, seed=seed, **hyper)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    encoder = model.ports.impls["bits"]
    encoder.encodeFeatures([1] * features)
    encoder.encodeFeatures([0] * features)
    sensors = model.ports.ensureSensorCapacity("bits", features)

    def observe(row):
        """Propagates one row and returns the state it produced."""
        encoded = encoder.encodeFeatures(row).unsqueeze(0)
        return flowSignalPass(model.mesh, encoded, sensors, model.config.hyper)

    return model, observe


class TestFiringShareDomain:
    """Domain: the share reports how much of the mesh one observation used."""

    def testFreshMeshFiresNearlyEveryEdge(self):
        """The 0.89-0.91 figure specs/propagation.md reports, now computed rather than asserted."""
        model, observe = makeObservedModel(seed=5)
        state = observe([1, 0, 1, 1, 0, 0, 1, 0])
        assert calcFiringShare(state, model.mesh.counts.edges) > 0.8

    def testShareIsTheTraceOverLiveEdges(self):
        """The share is exactly the fired trace's count divided by the live edge count."""
        model, observe = makeObservedModel(seed=2)
        state = observe([1, 1, 0, 0, 1, 0, 1, 1])
        live = model.mesh.counts.edges
        assert calcFiringShare(state, live) == pytest.approx(float(state.toEdgeTrace().sum()) / live)


class TestFiringShareBoundary:
    """Boundary: a share is defined only against a mesh that has edges."""

    def testNoLiveEdgesScoresZero(self):
        """A mesh with no edges fired none of them, rather than dividing by zero."""
        _, observe = makeObservedModel(seed=1)
        assert calcFiringShare(observe([1, 0, 1, 0, 1, 0, 1, 0]), 0) == 0.0


class TestFiringProfileDomain:
    """Domain: the profile separates how much fired from how much of it was label-specific."""

    def testFreshMeshAllocatesAlmostNothingPerLabel(self):
        """Nine edges in ten fire for every input, and almost none of that is specific to it.

        The measurement the two fan-out sparsity experiments lacked. A share near 0.9 with a
        conditionality near 0 is the direct statement of ROADMAP.md's finding that every input uses
        the whole mesh, and it is what a change aiming at per-label capacity has to move.
        """
        model, observe = makeObservedModel(seed=5)
        profile = calcFiringProfile(observe, makeBinaryRows(30, 8, seed=501), TASKS["copy"])
        assert profile.share > 0.8
        assert profile.conditionality < 0.1
        assert profile.within_label_overlap > 0.5

    def testConditionalityIsWithinMinusBetween(self):
        """The headline is exactly the gap between same-label and different-label overlap."""
        profile = FiringProfile(within_label_overlap=0.9, between_label_overlap=0.6)
        assert profile.conditionality == pytest.approx(0.3)


class TestFiringProfileBoundary:
    """Boundary: a profile over too few rows has no pairs to compare."""

    def testNoRowsProducesAnEmptyProfile(self):
        """Nothing observed means nothing measured, rather than a divide by zero."""
        _, observe = makeObservedModel(seed=1)
        assert calcFiringProfile(observe, [], TASKS["copy"]) == FiringProfile()

    def testOneRowHasNoPairsSoOverlapsStayZero(self):
        """Overlap needs two observations, so a single row reports share without conditionality."""
        _, observe = makeObservedModel(seed=1)
        profile = calcFiringProfile(observe, [[1, 0, 1, 0, 1, 0, 1, 0]], TASKS["copy"])
        assert profile.share > 0.0
        assert profile.within_label_overlap == 0.0
        assert profile.between_label_overlap == 0.0


class TestArrivalProfileDomain:
    """Domain: arrivals track wiring, which is why ranking them would select topology."""

    def testArrivalTracksFanInOnAFreshMesh(self):
        """A well-connected neuron receives more, whatever the input said.

        The precondition for input-conditional firing, stated as the number that refutes it: at
        this correlation a top-k over arrivals picks the best-connected neurons and picks the same
        ones for every input, which is what both fan-out experiments measured without seeing.
        """
        model, observe = makeObservedModel(seed=5)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        assert calcArrivalProfile(model.mesh, states).fanin_correlation > 0.5

    def testArrivalsAreNotComparableAcrossNeurons(self):
        """The spread across neurons is as large as the mean, so nothing can be ranked."""
        model, observe = makeObservedModel(seed=5)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        assert calcArrivalProfile(model.mesh, states).arrival_cv > 0.5

    def testTheMeshAmplifiesWithDepth(self):
        """The frontier climbs above the band it started in, rather than attenuating.

        specs/propagation.md states a path's value "only ever multiplies by strengths below one".
        That is true per edge and false per neuron: a neuron sums its arrivals, so with a mean
        fan-in of 10 and a mean strength of 0.7 it gains where each of its edges lost.
        """
        model, observe = makeObservedModel(seed=5)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        assert calcArrivalProfile(model.mesh, states).depth_gain > 1.5

    def testTheNeuronGateNeverBinds(self):
        """Median arrival sits far above the threshold meant to refuse it.

        This is why nine edges in ten fire: nothing is ever refused, and propagation ends by
        exhausting the fire-once guard rather than by any gate.
        """
        model, observe = makeObservedModel(seed=5)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        profile = calcArrivalProfile(model.mesh, states)
        assert profile.median_arrival > model.config.hyper.neuron_firing_threshold * 4


class TestArrivalProfileBoundary:
    """Boundary: a profile with nothing to read reports zeros rather than failing."""

    def testNoStatesProducesAnEmptyProfile(self):
        """Nothing propagated means nothing measured."""
        model, _ = makeObservedModel(seed=1)
        assert calcArrivalProfile(model.mesh, []) == ArrivalProfile()


class TestCorrelationDomain:
    """Domain: the correlation helper matches the definition it stands for."""

    def testPerfectAgreementScoresOne(self):
        """Two vectors moving together correlate at one."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert calcCorrelation(values, values * 2.0 + 1.0) == pytest.approx(1.0)

    def testPerfectDisagreementScoresMinusOne(self):
        """Two vectors moving oppositely correlate at minus one."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert calcCorrelation(values, -values) == pytest.approx(-1.0)


class TestCorrelationError:
    """Error: a correlation is undefined where one side does not vary."""

    def testAConstantVectorCorrelatesWithNothing(self):
        """A constant has no variance to share, so the correlation is zero rather than a NaN."""
        assert calcCorrelation(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 2.0, 3.0])) == 0.0

    def testASingleObservationCorrelatesWithNothing(self):
        """One point defines no relationship."""
        assert calcCorrelation(torch.tensor([1.0]), torch.tensor([2.0])) == 0.0


class TestFrontierChaos:
    """Chaos: the frontier record survives a pass that carries nothing."""

    def testAnObservationAtTheBandFloorStillRecordsItsFrontier(self):
        """A uniform minimum observation propagates, so its frontier is recorded rather than empty."""
        _, observe = makeObservedModel(seed=3)
        state = observe([0] * 8)
        assert len(state.frontier) == state.hops
        assert all(value >= 0.0 for value in state.frontier)
