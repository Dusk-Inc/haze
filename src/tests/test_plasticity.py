"""Crystallization: an edge's learning rate hardening with good outcomes and softening with bad."""

import pytest
import torch

from haze import makeHaze
from haze.errors import SignalDidNotReachMotorsError
from haze.functions.learning import applyCreditedLearning, calcPlasticityFactors
from haze.functions.score import TASKS, makeBinaryRows
from haze.models import HazeHyper
from haze.modules.decoders import ArgMax
from haze.modules.encoders import NumericEncoder


def makeMesh(**hyper):
    """Returns a small wired mesh with no training applied."""
    model = makeHaze(nexus_size=32, terminus_size=16, seed=1, growth_threshold=0.99, **hyper)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    return model


def applyMoves(model, rounds, direction):
    """Drives every live edge the same way `rounds` times, so its plasticity can be read off.

    Driven directly rather than by training, because a mesh rewarded identically every step has an
    advantage of zero once its baseline catches up, so no edge moves and there is nothing to
    measure. The mechanism under test is what an edge does when it *is* moved.
    """
    live = model.mesh.counts.edges
    trace = torch.ones(live, dtype=torch.bool)
    eligibility = torch.ones(live, dtype=model.mesh.dtype)
    credit = torch.full(
        (int(model.mesh.kind.numel()),), float(direction), dtype=model.mesh.dtype
    )
    for _ in range(rounds):
        applyCreditedLearning(
            model.mesh, trace, eligibility, credit, 1.0, 0.0, model.config.hyper
        )
    return model


def makeTrained(steps, task, **hyper):
    """Returns a mesh trained on a real task, where reward and therefore the advantage vary.

    An observation whose signal never reaches the motors is counted rather than dropped, and a mesh
    that is mute throughout fails the test outright — it would otherwise return a model that had
    learned nothing at all and read as a result about plasticity.
    """
    model = makeMesh(**hyper)
    mute = 0
    for row in makeBinaryRows(steps, 8, 500):
        try:
            out = model({"bits": row})
        except SignalDidNotReachMotorsError:
            mute += 1
            continue
        model.learn(1.0 if out["bit"][0] == task(row) else 0.0)
    if mute == steps:
        pytest.fail(f"the mesh reached no motors in any of {steps} observations")
    return model


def calcLiveEpsilon(model):
    """Returns the mean learning rate across the mesh's live edges."""
    live = model.mesh.counts.edges
    alive = model.mesh.alive_e[:live]
    return float(model.mesh.epsilon[:live][alive].mean())


class TestPlasticityFactorDomain:
    """Domain: each edge's factor reads the sign of its own move and nothing else."""

    def testReinforcedEdgeCools(self):
        """An edge that was just strengthened hardens."""
        hyper = HazeHyper(crystallize=True)
        got = calcPlasticityFactors(torch.tensor([0.4, 0.01]), hyper)
        assert got.tolist() == pytest.approx([hyper.epsilon_cool] * 2)
        assert float(got.max()) < 1.0

    def testWeakenedEdgeWarms(self):
        """An edge that was just corrected softens."""
        hyper = HazeHyper(crystallize=True)
        got = calcPlasticityFactors(torch.tensor([-0.4, -0.01]), hyper)
        assert got.tolist() == pytest.approx([hyper.epsilon_warm] * 2)
        assert float(got.min()) > 1.0

    def testMagnitudeDoesNotMatter(self):
        """A small move and a large one harden identically; only the sign is read."""
        hyper = HazeHyper(crystallize=True)
        got = calcPlasticityFactors(torch.tensor([0.001, 9.9]), hyper)
        assert float(got[0]) == float(got[1])

    def testEdgesInOneMeshHardenIndependently(self):
        """Two clusters moving opposite ways take opposite factors on the same step."""
        hyper = HazeHyper(crystallize=True)
        got = calcPlasticityFactors(torch.tensor([0.5, -0.5]), hyper)
        assert float(got[0]) < 1.0 < float(got[1])

    def testDisabledKeepsTheInheritedDecay(self):
        """With crystallize off every edge takes the unconditional decay, whatever it did."""
        hyper = HazeHyper()
        got = calcPlasticityFactors(torch.tensor([0.5, -0.5, 0.0]), hyper)
        assert got.tolist() == pytest.approx([hyper.epsilon_decay] * 3)


class TestPlasticityFactorBoundary:
    """Boundary: an edge that did not move is evidence for neither direction."""

    def testAnUnmovedEdgeHolds(self):
        """A move of exactly zero leaves the learning rate where it is."""
        assert float(calcPlasticityFactors(torch.zeros(1), HazeHyper(crystallize=True))) == 1.0

    def testCoolingAndWarmingNearlyCancel(self):
        """One reinforcement and one correction leave an edge close to where it started."""
        hyper = HazeHyper(crystallize=True)
        got = calcPlasticityFactors(torch.tensor([1.0, -1.0]), hyper)
        assert float(got[0] * got[1]) == pytest.approx(1.0, abs=0.001)

    def testShapeIsPreserved(self):
        """The factors line up with the edges they came from."""
        got = calcPlasticityFactors(torch.zeros(17), HazeHyper(crystallize=True))
        assert got.shape == (17,)


class TestPlasticityFactorChaos:
    """Chaos: a non-finite move must not produce a non-finite learning rate."""

    def testNaNMoveIsTreatedAsNoEvidence(self):
        """A NaN compares false both ways, so it holds rather than poisoning the rate."""
        got = calcPlasticityFactors(torch.tensor([float("nan")]), HazeHyper(crystallize=True))
        assert float(got) == 1.0

    def testInfiniteMoveStillPicksASide(self):
        """An unbounded move hardens or softens by its sign, never by its size."""
        hyper = HazeHyper(crystallize=True)
        got = calcPlasticityFactors(torch.tensor([float("inf"), float("-inf")]), hyper)
        assert float(got[0]) == pytest.approx(hyper.epsilon_cool)
        assert float(got[1]) == pytest.approx(hyper.epsilon_warm)


class TestCrystallizationDomain:
    """Domain: an edge repeatedly reinforced hardens, one repeatedly corrected softens."""

    def testReinforcementHardensTheMesh(self):
        """Driving every edge upward drives the mean learning rate down."""
        model = applyMoves(makeMesh(crystallize=True), 50, 1.0)
        assert calcLiveEpsilon(model) < model.config.hyper.epsilon_start

    def testCorrectionSoftensRelativeToReinforcement(self):
        """A mesh that keeps being corrected stays more plastic than one that keeps being kept."""
        kept = applyMoves(makeMesh(crystallize=True), 50, 1.0)
        corrected = applyMoves(makeMesh(crystallize=True), 50, -1.0)
        assert calcLiveEpsilon(corrected) > calcLiveEpsilon(kept)

    def testOneCorrectionDoesNotUndoALongRecord(self):
        """The hysteresis: one reversal barely moves a mesh that spent 300 rounds hardening."""
        model = applyMoves(makeMesh(crystallize=True), 300, 1.0)
        before = calcLiveEpsilon(model)
        applyMoves(model, 1, -1.0)
        assert calcLiveEpsilon(model) == pytest.approx(before, rel=0.02)

    def testALongRecordTakesALongRecordToUndo(self):
        """Hysteresis is symmetric: undoing the hardening takes comparably many reversals."""
        model = applyMoves(makeMesh(crystallize=True), 200, 1.0)
        hardened = calcLiveEpsilon(model)
        applyMoves(model, 200, -1.0)
        assert calcLiveEpsilon(model) > hardened * 2

    def testTrainingOnARealTaskMovesTheRate(self):
        """The mechanism is reachable through the ordinary learn path, not only when driven."""
        model = makeTrained(400, TASKS["copy"], crystallize=True)
        assert calcLiveEpsilon(model) != pytest.approx(model.config.hyper.epsilon_start)


class TestCrystallizationBoundary:
    """Boundary: the rate is bounded at both ends, so neither state is absorbing."""

    def testHardeningStopsAtTheFloor(self):
        """A permanently reinforced mesh becomes slow, never frozen."""
        model = applyMoves(makeMesh(crystallize=True), 3000, 1.0)
        live = model.mesh.counts.edges
        alive = model.mesh.alive_e[:live]
        floor = model.config.hyper.epsilon_floor
        assert float(model.mesh.epsilon[:live][alive].min()) == pytest.approx(floor, rel=1e-5)

    def testSofteningStopsAtTheStartingRate(self):
        """A permanently corrected mesh returns to fresh plasticity and goes no further."""
        model = applyMoves(makeMesh(crystallize=True), 3000, -1.0)
        live = model.mesh.counts.edges
        alive = model.mesh.alive_e[:live]
        start = model.config.hyper.epsilon_start
        assert float(model.mesh.epsilon[:live][alive].max()) == pytest.approx(start, rel=1e-5)

    def testDisabledLeavesTheRateAlmostUntouched(self):
        """The inherited decay is inert at this scale, which is why it needed replacing."""
        model = applyMoves(makeMesh(), 400, 1.0)
        assert calcLiveEpsilon(model) > 0.9 * model.config.hyper.epsilon_start


class TestCrystallizationError:
    """Error: configurations that would make the mechanism unreachable are refused."""

    def testAFloorAtOrAboveTheStartIsRefused(self):
        """An edge starting fully crystallized could never be moved by any outcome."""
        with pytest.raises(ValueError, match="epsilon_floor"):
            HazeHyper(epsilon_floor=0.2, epsilon_start=0.1)

    def testAnInertCrystallizationIsRefused(self):
        """Turning the mechanism on without rates that move is dead code, not a configuration."""
        with pytest.raises(ValueError, match="crystallize"):
            HazeHyper(crystallize=True, epsilon_cool=1.0, epsilon_warm=1.0)


class TestCrystallizationPersistence:
    """Domain: a hardened mesh stays hardened across a save and load."""

    def testLearningRatesSurviveARoundTrip(self, tmp_path):
        """Crystallization is learned state, so a checkpoint that loses it loses the record."""
        model = makeTrained(300, TASKS["copy"], crystallize=True)
        live = model.mesh.counts.edges
        before = model.mesh.epsilon[:live].clone()
        model.save_pretrained(tmp_path / "hardened")
        loaded = type(model).from_pretrained(tmp_path / "hardened")
        assert torch.equal(loaded.mesh.epsilon[:live], before)
