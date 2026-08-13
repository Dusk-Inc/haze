"""Tests for the learning signal: what a reward is measured against, and what that decides."""

import pytest
import torch

from haze import makeHaze
from haze.errors import InvalidRewardError, SignalDidNotReachMotorsError
from haze.functions.learning import calcRewardAdvantage, switchMotorChoice
from haze.functions.score import TASKS, calcRewardMean, makeBinaryRows
from haze.models import HazeHyper
from haze.modules.decoders import ArgMax
from haze.modules.encoders import NumericEncoder


def makeTaskModel(seed: int, **hyper):
    """Returns a model wired for the binary task suite."""
    model = makeHaze(nexus_size=64, terminus_size=32, seed=seed, **hyper)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    return model


def runTask(model, task, rows):
    """Runs a task and returns the rewards and the motor chosen at each step.

    Recovers through reverse learning when an observation reaches no motor, which is what that
    mechanism is for: learning can drive a path below the edge gate, and a trainer that treated
    the resulting error as a wrong answer would leave the mesh with no way back.
    """
    rewards, chosen = [], []
    for row in rows:
        try:
            answer = model({"bits": row})
        except SignalDidNotReachMotorsError:
            model.learn(0.0, reverse=True)
            rewards.append(0.0)
            continue
        reward = 1.0 if answer["bit"][0] == task(row) else 0.0
        rewards.append(reward)
        chosen.append(model._chosen["bit"])
        model.learn(reward)
    return rewards, chosen


def scoreHeldOut(model, task, seed: int, count: int = 150):
    """Returns greedy accuracy and the minority answer's share on rows never trained on.

    Scored in eval mode because a trained model's own answers are contaminated by exploration:
    a collapsed mesh at a 5% explore rate still gives its minority answer 2-3% of the time, which
    is indistinguishable from a mesh that learned very little.
    """
    model.eval()
    answers, correct = [], []
    for row in makeBinaryRows(count, 8, seed):
        try:
            answer = model({"bits": row})["bit"][0]
        except SignalDidNotReachMotorsError:
            correct.append(0.0)
            continue
        answers.append(answer)
        correct.append(1.0 if answer == task(row) else 0.0)
    share = min(answers.count(0), answers.count(1)) / len(correct)
    return sum(correct) / len(correct), share


# -- Domain --------------------------------------------------------------------------------


def test_calcRewardAdvantage_doesReturnZeroForTheFirstReward():
    """Asserts a first reward beats no expectation, since one sample is its own expectation."""
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh

    assert calcRewardAdvantage(mesh, 1.0, HazeHyper()) == 0.0
    assert float(mesh.reward_bar[0]) == pytest.approx(1.0)


def test_calcRewardAdvantage_doesFallTowardZeroAsARewardBecomesExpected():
    """Asserts a reward that keeps arriving stops being news.

    This is what stops the mesh adjusting what it already reliably gets right, and it is the
    property the inherited `reward - confidence` signal lacked.
    """
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh
    hyper = HazeHyper(reward_baseline_rate=0.5)

    calcRewardAdvantage(mesh, 0.0, hyper)
    first = calcRewardAdvantage(mesh, 1.0, hyper)
    for _ in range(20):
        last = calcRewardAdvantage(mesh, 1.0, hyper)

    assert first == pytest.approx(1.0)
    assert abs(last) < 0.01


def test_calcRewardAdvantage_doesGoNegativeWhenARewardFallsShort():
    """Asserts falling below what the mesh had come to expect reverses the update's sign."""
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh
    hyper = HazeHyper(reward_baseline_rate=0.5)

    for _ in range(10):
        calcRewardAdvantage(mesh, 1.0, hyper)

    assert calcRewardAdvantage(mesh, 0.0, hyper) < -0.9


@pytest.mark.slow
def test_learn_doesNotCollapseToASingleAnswer():
    """Asserts the mesh keeps giving both answers rather than entrenching whichever led first.

    The inherited signal contributed `1 - c` for a correct answer and only `-c` for a wrong one,
    so at even odds the chosen path was reinforced by a net positive amount whether or not it was
    right. Measured on this task it ran to a fixed point: one motor won 96-100% of observations
    and the mesh answered the same label forever, at exactly chance. See specs/learning.md.
    """
    model = makeTaskModel(seed=2)
    runTask(model, TASKS["copy"], makeBinaryRows(600, 8, seed=502))

    _, minority = scoreHeldOut(model, TASKS["copy"], seed=9002)

    assert minority > 0.2, (
        f"the mesh gave its minority answer on only {minority:.0%} of held-out rows; the "
        "uncentred signal measured 0-2% here, which is collapse rather than learning"
    )


@pytest.mark.slow
def test_learn_doesLearnAnInputDependentAnswer():
    """Asserts reward on a task whose answer depends on the input rises above chance.

    The constant task only needs a bias and was already solved; this one needs the answer to
    track a specific feature, which is what a collapsed mesh cannot do at all.
    """
    model = makeTaskModel(seed=2)
    runTask(model, TASKS["copy"], makeBinaryRows(600, 8, seed=502))

    accuracy, _ = scoreHeldOut(model, TASKS["copy"], seed=9002)

    assert accuracy > 0.7


def test_switchMotorChoice_doesPreserveTheActivationMultiset():
    """Asserts exploring reorders which label leads without inventing certainty.

    A boost would make the mesh look more confident precisely when it is guessing, and confidence
    feeds both the learning signal and the growth trigger.
    """
    states = torch.tensor([0.2, 0.9, 0.4, 0.1])
    generator = torch.Generator().manual_seed(3)

    switched = [switchMotorChoice(states, 1.0, generator) for _ in range(20)]

    assert any(int(torch.argmax(s)) != 1 for s in switched), "no exploration ever happened"
    for s in switched:
        assert torch.equal(torch.sort(s).values, torch.sort(states).values)


@pytest.mark.slow
def test_switchMotorChoice_doesEscapeAUniformlyWrongAnswer():
    """Asserts a mesh answering the same wrong label every time can still find the right one.

    With a greedy readout and a centred advantage this task has no exit: reward never varies, the
    baseline meets it, and the advantage is zero forever. One seed in three measured exactly 0.00
    and stayed there. See specs/learning.md.
    """
    model = makeTaskModel(seed=1, explore_rate=0.0)
    runTask(model, TASKS["constant"], makeBinaryRows(400, 8, seed=501))
    stuck, _ = scoreHeldOut(model, TASKS["constant"], seed=9001)

    explorer = makeTaskModel(seed=1)
    runTask(explorer, TASKS["constant"], makeBinaryRows(400, 8, seed=501))
    escaped, _ = scoreHeldOut(explorer, TASKS["constant"], seed=9001)

    assert stuck < 0.1, "this seed no longer demonstrates the dead zone the test is about"
    assert escaped > 0.9


def test_predict_doesAnswerGreedilyInEvalMode():
    """Asserts exploration is a training behaviour and never reaches a deployed answer."""
    model = makeTaskModel(seed=4, explore_rate=0.9)
    row = [1, 0, 1, 0, 1, 0, 1, 0]
    model({"bits": row})
    greedy = model.predict()

    model.eval()
    model.observe({"bits": row})
    answers = {model.predict()["bit"][0] for _ in range(15)}

    assert len(answers) == 1, "eval mode gave more than one answer for one observation"
    assert greedy is not None


# -- Boundary ------------------------------------------------------------------------------


def test_calcRewardAdvantage_doesRespectTheDisableFlag():
    """Asserts the inherited uncentred signal is still reachable, so the difference stays measurable."""
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh
    hyper = HazeHyper(reward_baseline=False)

    assert calcRewardAdvantage(mesh, 1.0, hyper) == 1.0
    assert calcRewardAdvantage(mesh, 1.0, hyper) == 1.0
    assert float(mesh.reward_bar[0]) == 0.0


def test_calcRewardAdvantage_doesTrackTheLastRewardExactlyAtRateOne():
    """Asserts the fastest baseline compares each reward against only the one before it."""
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh
    hyper = HazeHyper(reward_baseline_rate=1.0)

    calcRewardAdvantage(mesh, 1.0, hyper)

    assert calcRewardAdvantage(mesh, 0.0, hyper) == pytest.approx(-1.0)
    assert calcRewardAdvantage(mesh, 0.0, hyper) == pytest.approx(0.0)


def test_calcRewardAdvantage_doesAcceptARewardOutsideTheUnitInterval():
    """Asserts the baseline assumes no reward scale, since it is seeded from what it is given."""
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh
    hyper = HazeHyper()

    assert calcRewardAdvantage(mesh, -40.0, hyper) == 0.0
    assert calcRewardAdvantage(mesh, -40.0, hyper) == pytest.approx(0.0)


# -- Error ---------------------------------------------------------------------------------


def test_calcRewardAdvantage_doesRefuseANonFiniteReward():
    """Asserts a non-finite reward never reaches the baseline.

    The baseline is persistent state, so unlike a poisoned strength this would survive into the
    checkpoint and corrupt every later advantage, long after the observation that caused it.
    """
    mesh = makeHaze(nexus_size=4, terminus_size=2, seed=1).mesh
    calcRewardAdvantage(mesh, 1.0, HazeHyper())

    with pytest.raises(InvalidRewardError):
        calcRewardAdvantage(mesh, float("nan"), HazeHyper())

    assert float(mesh.reward_bar[0]) == pytest.approx(1.0)
    assert int(mesh.learn_count[0]) == 1


def test_ensureHyperCoherent_doesRefuseABaselineRateOutsideItsRange():
    """Asserts a rate of zero is refused, since a baseline that never moves is not a baseline."""
    with pytest.raises(ValueError):
        HazeHyper(reward_baseline_rate=0.0)
    with pytest.raises(ValueError):
        HazeHyper(reward_baseline_rate=1.5)


# -- Chaos ---------------------------------------------------------------------------------


def test_rewardBaseline_doesSurviveACheckpointRoundTrip(tmp_path):
    """Asserts what the mesh had come to expect is part of what a checkpoint carries.

    A model reloaded with its baseline reset would measure its next reward against nothing and
    apply the uncentred push all over again, so this is learned state and not a scratch variable.
    """
    from haze import Haze

    model = makeTaskModel(seed=1, epsilon_start=0.05)
    runTask(model, TASKS["copy"], makeBinaryRows(40, 8, seed=501))

    model.save_pretrained(tmp_path / "ckpt")
    loaded = Haze.from_pretrained(str(tmp_path / "ckpt"))

    assert torch.equal(loaded.mesh.reward_bar, model.mesh.reward_bar)
    assert int(loaded.mesh.learn_count[0]) == 40
    assert float(model.mesh.reward_bar[0]) != 0.0


def test_learn_doesStayFiniteUnderAConstantReward():
    """Asserts a reward that never varies drives the advantage to zero rather than to a rail.

    With nothing to distinguish one observation from another there is no information to learn
    from, and the correct behaviour is to stop moving rather than to keep pushing in whichever
    direction the first observation happened to favour.
    """
    model = makeTaskModel(seed=2, epsilon_start=0.05)
    rows = makeBinaryRows(120, 8, seed=77)

    for row in rows:
        model({"bits": row})
        model.learn(1.0)

    live = model.mesh.counts.edges
    assert torch.isfinite(model.mesh.strength[:live]).all()
    assert abs(calcRewardAdvantage(model.mesh, 1.0, model.config.hyper)) < 0.01
