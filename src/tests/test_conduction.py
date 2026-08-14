"""Conduction: the strength range that carries signal, and how much input space still reaches."""

import pytest

from haze.errors import SignalDidNotReachMotorsError
from haze.functions.score import (
    TASKS,
    calcConductionReach,
    calcReachByLabel,
    calcScoreProfile,
    makeBinaryRows,
)
from haze.models import HazeHyper


def makeAnswerer(mute):
    """Returns an answer callable that goes silent for rows whose first bit is in `mute`."""
    def answer(row):
        """Answers a row, or raises as a mute mesh would."""
        if row[0] in mute:
            raise SignalDidNotReachMotorsError("mute")
        return row[0]
    return answer


class TestConductionFloorDomain:
    """Domain: the floor inverts the attenuation the propagation path actually applies."""

    def testFloorMatchesOneHopAttenuation(self):
        """A strength at the floor arrives exactly at the gate over one hop."""
        hyper = HazeHyper()
        floor = hyper.calcConductionFloor(hops=1)
        assert hyper.signal_upper * floor**2 == pytest.approx(hyper.signal_threshold)

    def testFloorRisesWithDepth(self):
        """A deeper path needs stronger edges, since every hop attenuates again."""
        hyper = HazeHyper()
        floors = [hyper.calcConductionFloor(hops=n) for n in (1, 2, 3, 4)]
        assert floors == sorted(floors)
        assert floors[0] < floors[-1]

    def testShippedDefaultsCarryADeadBand(self):
        """The shipped configuration has a range that is unprunable and mute."""
        band = HazeHyper().calcDeadBand()
        assert band is not None
        lower, upper = band
        assert lower == pytest.approx(0.2)
        assert upper == pytest.approx(0.527, abs=0.001)


class TestConductionFloorBoundary:
    """Boundary: the band closes exactly when pruning reaches the floor."""

    def testDeadBandClosesWhenPruningReachesTheFloor(self):
        """Raising prune_threshold to the conduction floor leaves nothing stranded."""
        hyper = HazeHyper()
        raised = hyper.model_copy(update={"prune_threshold": hyper.calcConductionFloor()})
        assert raised.calcDeadBand() is None

    def testZeroHopsIsTreatedAsOne(self):
        """A path of no hops is meaningless, so the floor clamps to the single-hop value."""
        hyper = HazeHyper()
        assert hyper.calcConductionFloor(hops=0) == hyper.calcConductionFloor(hops=1)

    def testBandNeverExtendsAboveTheUpperRail(self):
        """A reported band always names strengths an edge can actually hold."""
        for hyper in (
            HazeHyper(),
            HazeHyper(signal_lower=0.85, signal_upper=0.9, signal_threshold=0.68),
            HazeHyper(signal_threshold=0.1),
        ):
            band = hyper.calcDeadBand()
            if band is not None:
                assert band[1] <= hyper.strength_upper


class TestConductionReachDomain:
    """Domain: reach counts what the mesh answers, not what it answers correctly."""

    def testFullReachWhenNothingIsMute(self):
        """A mesh answering every row reaches all of them."""
        rows = makeBinaryRows(40, 8, 1)
        assert calcConductionReach(makeAnswerer(set()), rows) == 1.0

    def testReachIgnoresWhetherTheAnswerIsRight(self):
        """A mesh answering every row wrongly still has full reach."""
        rows = makeBinaryRows(40, 8, 2)

        def wrong(row):
            """Answers every row with a constant that is often incorrect."""
            return 1 - row[0]

        assert calcConductionReach(wrong, rows) == 1.0

    def testReachFallsAsInputsGoMute(self):
        """Silencing one half of the input space halves reach."""
        rows = makeBinaryRows(200, 8, 3)
        share = calcConductionReach(makeAnswerer({0}), rows)
        assert 0.4 < share < 0.6


class TestReachByLabelDomain:
    """Domain: the split is what shows a whole label dropping out."""

    def testSplitNamesTheSilencedLabel(self):
        """A label whose rows all go mute reports zero while the other reports one."""
        rows = makeBinaryRows(200, 8, 4)
        shares = calcReachByLabel(makeAnswerer({0}), rows, TASKS["copy"])
        assert shares[0] == 0.0
        assert shares[1] == 1.0

    def testSplitIsOrderedWorstFirst(self):
        """The worst-served label comes first, so the failure is legible at a glance."""
        rows = makeBinaryRows(200, 8, 5)
        shares = calcReachByLabel(makeAnswerer({0}), rows, TASKS["copy"])
        assert list(shares.values()) == sorted(shares.values())


class TestConductionReachBoundary:
    """Boundary: the degenerate input sets."""

    def testNoRowsIsZeroRatherThanADivideByZero(self):
        """An empty row set reaches nothing and does not raise."""
        assert calcConductionReach(makeAnswerer(set()), []) == 0.0

    def testEveryRowMuteIsZero(self):
        """A fully silent mesh reaches none of its inputs."""
        rows = makeBinaryRows(20, 8, 6)
        assert calcConductionReach(makeAnswerer({0, 1}), rows) == 0.0

    def testEmptyRowsGiveAnEmptySplit(self):
        """A split over no rows is empty rather than raising."""
        assert calcReachByLabel(makeAnswerer(set()), [], TASKS["copy"]) == {}


class TestConductionReachError:
    """Error: only the silence error is absorbed; a real fault still surfaces."""

    def testAnUnrelatedErrorIsNotSwallowed(self):
        """Reach absorbs a mute mesh, never a broken one."""
        def broken(row):
            """Raises a fault that is not about conduction."""
            raise ValueError("decoder is misconfigured")

        with pytest.raises(ValueError):
            calcConductionReach(broken, makeBinaryRows(4, 8, 7))


class TestScoreProfileDomain:
    """Domain: coverage and correctness are separated, and judged against the answered set."""

    def testAPerfectMeshProfilesCleanly(self):
        """Answering every row correctly gives full coverage and full conditional accuracy."""
        rows = makeBinaryRows(60, 8, 11)
        got = calcScoreProfile(TASKS["copy"], rows, TASKS["copy"])
        assert got.coverage == 1.0
        assert got.accuracy == 1.0
        assert got.conditional == 1.0

    def testAbstentionRaisesConditionalAboveRaw(self):
        """Declining to answer costs coverage and raw accuracy, never conditional accuracy."""
        rows = makeBinaryRows(200, 8, 12)
        got = calcScoreProfile(makeAnswerer({0}), rows, TASKS["copy"])
        assert got.conditional == 1.0
        assert got.accuracy < 0.6
        assert got.coverage < 0.6

    def testBaselineIsTakenFromTheAnsweredSet(self):
        """A mesh that keeps only one label's rows is scored against that label, not the task.

        The confusion this exists to prevent: silencing every minority input leaves a set whose
        majority share is 1.0, so answering it perfectly is worth no lift at all.
        """
        rows = makeBinaryRows(200, 8, 13)
        got = calcScoreProfile(makeAnswerer({0}), rows, TASKS["copy"])
        assert got.baseline == 1.0
        assert got.lift == 0.0

    def testLiftIsPositiveOnlyWhenTheMeshDiscriminates(self):
        """Answering a mixed set correctly beats guessing its majority label."""
        rows = makeBinaryRows(200, 8, 14)
        got = calcScoreProfile(TASKS["copy"], rows, TASKS["copy"])
        assert got.baseline < 0.7
        assert got.lift > 0.25


class TestScoreProfileBoundary:
    """Boundary: the degenerate mesh and the degenerate row set."""

    def testAFullyMuteMeshProfilesAtZero(self):
        """A mesh answering nothing reports no coverage rather than dividing by zero."""
        rows = makeBinaryRows(40, 8, 15)
        got = calcScoreProfile(makeAnswerer({0, 1}), rows, TASKS["copy"])
        assert got.coverage == 0.0
        assert got.conditional == 0.0
        assert got.lift == 0.0

    def testNoRowsProfilesAtZero(self):
        """An empty row set is not an error."""
        assert calcScoreProfile(makeAnswerer(set()), [], TASKS["copy"]).coverage == 0.0

    def testAConstantAnswerNeverEarnsLift(self):
        """Answering everything with one label can only ever match its share, never beat it.

        Zero when it picks the majority label and negative when it picks a minority one, which is
        the property that makes lift readable: only discrimination puts it above zero.
        """
        rows = makeBinaryRows(200, 8, 16)
        for label in (0, 1):
            got = calcScoreProfile(lambda row: label, rows, TASKS["copy"])
            assert got.coverage == 1.0
            assert got.lift <= 0.0
