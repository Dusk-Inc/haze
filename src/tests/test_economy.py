"""The signal economy's instruments: what fires, what arrives, and how much of it tracks wiring."""

import pytest
import torch

from haze import makeHaze
from haze.functions.economy import calcArrivalGate, makeSignalEconomy
from haze.functions.propagate import flowSignalPass
from haze.functions.score import (
    TASKS,
    calcArrivalProfile,
    calcCorrelation,
    calcFiringProfile,
    calcFiringShare,
    makeBinaryRows,
)
from haze.models import ArrivalProfile, FiringProfile, HazeHyper
from haze.modules.decoders import ArgMax
from haze.modules.encoders import NumericEncoder
from haze.tokens import NeuronKind

ECONOMY = {"signal_economy": True, "conductance_healing": False}
"""The smallest configuration that turns the economy on.

Healing must go off with it, and that pairing is enforced rather than conventional: a uniform
scale of a neuron's out-edges cancels exactly in a normalised share, so healing under the economy
would run on every update and change nothing.
"""


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


class TestEconomyDomain:
    """Domain: a strength becomes a share, so what a neuron carries stops being absolute."""

    def testSiblingsShareOneBudget(self):
        """A neuron's outgoing shares sum to `out_budget`, whatever its strengths were."""
        model = makeHaze(nexus_size=16, terminus_size=8, seed=4, **ECONOMY)
        economy = makeSignalEconomy(model.mesh, model.config.hyper)
        live = model.mesh.counts.edges
        total = torch.zeros(int(model.mesh.kind.numel()), dtype=model.mesh.dtype)
        total.index_add_(0, model.mesh.src[:live].to(torch.int64), economy.share.abs())
        emitting = total[total > 0]
        assert torch.allclose(emitting, torch.full_like(emitting, model.config.hyper.out_budget))

    def testLearningIsZeroSumAmongSiblings(self):
        """Raising one edge lowers every sibling's share, rather than draining the neuron.

        The property ROADMAP.md names as the candidate remedy for gap 1, here as a consequence of
        the representation rather than as a mechanism of its own.
        """
        model = makeHaze(nexus_size=16, terminus_size=8, seed=4, **ECONOMY)
        before = makeSignalEconomy(model.mesh, model.config.hyper).share.clone()
        edge = int((model.mesh.src[: model.mesh.counts.edges] == int(model.mesh.src[0])).nonzero()[0])
        siblings = model.mesh.src[: model.mesh.counts.edges] == model.mesh.src[edge]

        model.mesh.strength[edge] = model.config.hyper.strength_upper
        after = makeSignalEconomy(model.mesh, model.config.hyper).share
        moved = after[siblings].abs() - before[siblings].abs()
        assert float(moved.sum()) == pytest.approx(0.0, abs=1e-5)
        assert float(moved.max()) > 0.0
        assert float(moved.min()) < 0.0


class TestEconomyIntegration:
    """Domain: with the economy on, arrival stops tracking wiring and lands inside the band."""

    def testArrivalStopsTrackingFanIn(self):
        """The correlation the economy exists to remove is removed.

        Measured at +0.672 shipped and -0.046 under the economy over 8 seeds. This is the one
        structural claim the whole change rests on, so it is asserted rather than recorded.
        """
        model, observe = makeObservedModel(seed=5, **ECONOMY, gain_control=True)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        assert abs(calcArrivalProfile(model.mesh, states).fanin_correlation) < 0.15

    def testTheMeshStopsAmplifyingAndTheGateBeginsToBind(self):
        """Arrival falls from seven times the band's ceiling to inside it, onto the neuron gate."""
        model, observe = makeObservedModel(seed=5, **ECONOMY, gain_control=True)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        profile = calcArrivalProfile(model.mesh, states)
        assert profile.median_arrival < model.config.hyper.signal_upper
        assert profile.depth_gain < 2.0

    def testTheMeshStillAnswersEveryInput(self):
        """Conduction is not the price of the economy: a fresh mesh reaches its motors as before."""
        model, observe = makeObservedModel(seed=5, **ECONOMY, gain_control=True)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        assert all(state.reached for state in states)

    def testWithoutGainControlTheMeshGoesSilent(self):
        """The economy alone cannot conduct, which is why the gain is not optional.

        Only a fraction of a neuron's in-edges fire per hop while the budget counts all of them, so
        arrival lands an order of magnitude under the gate and nothing emits at the first hop.
        """
        model, observe = makeObservedModel(seed=5, **ECONOMY)
        states = [observe(row) for row in makeBinaryRows(20, 8, seed=501)]
        assert not any(state.reached for state in states)


class TestEconomyBoundary:
    """Boundary: the states that were absorbing under the shipped economy are unreachable."""

    def testAnEdgeCannotBeDrivenMute(self):
        """Scaling every out-edge of a neuron to the lower rail leaves what it carries unchanged.

        This is ROADMAP.md gap 1 closed by construction. Under the shipped economy learning can
        push an edge below the conduction floor, and since a mute edge fires no trace and every
        update is trace-gated, nothing can ever recover it. A share is a ratio, so the same
        collapse is not expressible.
        """
        model = makeHaze(nexus_size=16, terminus_size=8, seed=4, **ECONOMY)
        live = model.mesh.counts.edges
        siblings = model.mesh.src[:live] == int(model.mesh.src[0])
        before = makeSignalEconomy(model.mesh, model.config.hyper).share[siblings].clone()

        weakened = model.mesh.strength[:live].clone()
        weakened[siblings] *= 0.01
        model.mesh.strength[:live] = weakened

        after = makeSignalEconomy(model.mesh, model.config.hyper).share[siblings]
        assert torch.allclose(after, before, atol=1e-6)
        assert float(after.abs().sum()) == pytest.approx(model.config.hyper.out_budget, abs=1e-5)

    def testABudgetBelowOneIsNotClampedUpToOne(self):
        """A neuron's real incoming budget is usually under one, and must be used as it is.

        Shares average roughly the reciprocal of a neuron's fan-out, so most budgets fall below
        one. Clamping up to one — built that way first — replaced the true budget for two
        interneurons in three, divided the least-connected by up to fourteen times too much, and
        inverted the correlation the economy exists to remove.
        """
        model = makeHaze(nexus_size=64, terminus_size=32, seed=5, **ECONOMY)
        economy = makeSignalEconomy(model.mesh, model.config.hyper)
        budgets = economy.in_scale[model.mesh.is_inter]
        assert float(budgets.min()) < 1.0
        assert float(budgets.median()) < 1.0

    def testANeuronWithNoIncomingEdgesDividesByOne(self):
        """The only case a substituted budget covers is the one that would divide by zero."""
        model = makeHaze(nexus_size=16, terminus_size=8, seed=4, **ECONOMY)
        sensors = model.mesh.allocNeuronIds(3, NeuronKind.SENSOR, owner=0)
        economy = makeSignalEconomy(model.mesh, model.config.hyper)
        assert torch.allclose(
            economy.in_scale[sensors], torch.ones_like(economy.in_scale[sensors])
        )

    def testTheDeadBandIsUnrepresentable(self):
        """No strength range is alive, unprunable, and mute once strengths are shares."""
        assert HazeHyper(**ECONOMY).calcDeadBand() is None
        assert HazeHyper().calcDeadBand() is not None


class TestEconomyError:
    """Error: a configuration that would make a mechanism dead code is refused."""

    def testRankingNeedsTheEconomy(self):
        """Ranking arrivals that track in-degree selects topology, so it is refused without it."""
        with pytest.raises(ValueError, match="firing_fraction needs signal_economy"):
            HazeHyper(firing_fraction=0.2)

    def testGainControlNeedsTheEconomy(self):
        """The gain restores a level only the economy removes."""
        with pytest.raises(ValueError, match="gain_control needs signal_economy"):
            HazeHyper(gain_control=True)

    def testHealingIsRefusedAlongsideTheEconomy(self):
        """A uniform scale of a neuron's out-edges cancels in a share, so healing would do nothing.

        A deliberate trip-wire rather than a silent no-op: the flag cannot be turned on while a
        mechanism it makes inert is still claiming to run.
        """
        with pytest.raises(ValueError, match="incompatible"):
            HazeHyper(signal_economy=True, conductance_healing=True)


class TestEconomyChaos:
    """Chaos: the rank stays order-free where every arrival is identical."""

    def testTiedStrengthsDoNotDependOnOrdering(self):
        """A mesh whose strengths are all equal ranks by value, so ties all pass together.

        The k-th-value formulation is what makes this true. Taking `topk` indices instead would
        break the tie arbitrarily and put the tensor engine and the plain-Python reference, which
        has no stable ordering, silently out of step.
        """
        model = makeHaze(nexus_size=16, terminus_size=8, seed=4, **ECONOMY, firing_fraction=0.25)
        model.mesh.strength[: model.mesh.counts.edges] = 0.5
        model.mesh.log_str[: model.mesh.counts.edges] = torch.tensor(0.5).abs().log()

        arrived = torch.full((1, int(model.mesh.kind.numel())), 0.3, dtype=model.mesh.dtype)
        gate = model.mesh.is_inter.unsqueeze(0).clone()
        allowed = calcArrivalGate(arrived, gate, model.mesh.is_nexus, model.config.hyper)
        assert bool((allowed == gate).all())


class TestFrontierChaos:
    """Chaos: the frontier record survives a pass that carries nothing."""

    def testAnObservationAtTheBandFloorStillRecordsItsFrontier(self):
        """A uniform minimum observation propagates, so its frontier is recorded rather than empty."""
        _, observe = makeObservedModel(seed=3)
        state = observe([0] * 8)
        assert len(state.frontier) == state.hops
        assert all(value >= 0.0 for value in state.frontier)
