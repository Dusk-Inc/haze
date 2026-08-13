"""Tests for the auditor, growth, pruning, and the trainer loop that drives them."""

import pytest
import torch

from haze import Auditor, Haze, HazeConfig, MeshCapacity, Trainer, makeHaze
from haze.errors import SignalDidNotReachMotorsError
from haze.functions.grow import (
    applyGrowth,
    calcErrorRate,
    calcGrowthAmount,
    calcGrowthPlan,
    calcUncertaintyRate,
)
from haze.functions.prune import applyPrune, calcPruneMask
from haze.functions.score import TASKS, makeBinaryRows
from haze.models import GrowthPlan, HazeHyper, MeshCounts
from haze.modules.decoders import ArgMax
from haze.modules.encoders import NumericEncoder
from haze.tokens import NeuronKind


def makeTaskModel(seed: int, **hyper):
    """Returns a model wired for the binary task suite."""
    model = makeHaze(nexus_size=32, terminus_size=16, seed=seed, **hyper)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    return model


def scoreCopy(answer, inputs):
    """Scores an answer against the copy task."""
    return 1.0 if answer["bit"][0] == TASKS["copy"](inputs["bits"]) else 0.0


# -- Domain --------------------------------------------------------------------------------


def test_calcGrowthPlan_doesWaitForAFullWindow():
    """Asserts one unlucky observation cannot restructure the network."""
    hyper = HazeHyper(audit_window=10)
    counts = MeshCounts(nexus=64, terminus=32)

    plan = calcGrowthPlan([0.0] * 3, [0.1] * 3, counts, hyper)

    assert not plan.triggered and plan.isEmpty


def test_calcGrowthPlan_doesTriggerOnSustainedError():
    """Asserts a mesh that is wrong across a whole window is grown."""
    hyper = HazeHyper(audit_window=10, growth_threshold=0.5)
    counts = MeshCounts(nexus=64, terminus=32)

    plan = calcGrowthPlan([0.0] * 10, [0.1] * 10, counts, hyper)

    assert plan.triggered and plan.nexus > 0 and plan.terminus > 0
    assert plan.error_rate == pytest.approx(1.0)


def test_calcGrowthAmount_doesGrowMoreWhenWrongAndUnsure():
    """Asserts uncertainty adds to the amount, so a confident wrong mesh grows less.

    A confident wrong mesh has a working rule and the wrong one; more neurons do not fix that.
    """
    hyper = HazeHyper()

    unsure = calcGrowthAmount(100, error=1.0, uncertainty=1.0, hyper=hyper)
    certain = calcGrowthAmount(100, error=1.0, uncertainty=0.0, hyper=hyper)

    assert unsure > certain


def test_applyGrowth_doesWriteIntoSlackWithoutRenumbering():
    """Asserts growth that fits in slack moves no existing neuron, so learned edges survive it.

    Index stability is the whole invariant: every label binding, port claim, and edge endpoint
    names a neuron by integer, so renumbering one silently reassigns what the mesh learned about
    it. Built with headroom deliberately, because `makeHaze` allocates each block exactly its
    starting size and so the first growth always reallocates instead.
    """
    config = HazeConfig(
        seed=1,
        nexus_size=32,
        terminus_size=16,
        capacity=MeshCapacity(nexus=64, terminus=32),
    )
    model = Haze(config)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    mesh = model.mesh
    live = mesh.counts.edges
    before_src = mesh.src[:live].clone()
    before_strength = mesh.strength[:live].clone()

    added = applyGrowth(mesh, GrowthPlan(nexus=8, terminus=4, triggered=True))

    assert added == 12
    assert torch.equal(mesh.src[:live], before_src)
    assert torch.equal(mesh.strength[:live], before_strength)
    assert mesh.counts.nexus == 40 and mesh.counts.terminus == 20
    assert mesh.capacity.nexus == 64, "growth reallocated when it should have used slack"


def test_applyPrune_doesRemoveWeakEdgesAndRewireWhatItStrands():
    """Asserts pruning removes dead weight and never leaves a neuron cut off."""
    model = makeTaskModel(seed=2)
    model({"bits": [1, 1, 1, 1, 0, 0, 0, 0]})
    mesh = model.mesh
    hyper = model.config.hyper
    mesh.strength[: mesh.counts.edges : 2] = 0.05

    report = applyPrune(mesh, hyper)

    assert report.edges_removed > 0
    assert report.edges_remaining == mesh.counts.edges
    assert not calcPruneMask(mesh, hyper).any()
    assert all(ids.numel() == 0 for ids in mesh.findOrphans().values())


def test_flowRecoverSignal_doesReopenAPathWhenRecoveryIsEnabled():
    """Asserts the recovery mechanism works when asked for, though it is off by default.

    Off by default because it was measured and costs more than it saves — held-out copy accuracy
    0.95 skipping a lost observation against 0.74 recovering from it. Kept and tested because a
    mesh that truly cannot reach its motors has no other way back, and that default is one
    measurement rather than a proof. See specs/learning.md.
    """
    model = makeTaskModel(seed=3, relearn_limit=20)
    trainer = Trainer(model)
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    motors = model.ports.findActiveMotorIds("bit")
    into_motor = torch.isin(model.mesh.dst[: model.mesh.counts.edges], motors)
    model.mesh.strength[: model.mesh.counts.edges][into_motor] = 0.01

    answer = trainer.flowRecoverSignal({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})

    assert answer["bit"][0] in (0, 1)


def test_flowTraining_doesLearnWhileRestructuring():
    """Asserts the trainer's full loop reaches above chance with growth and pruning live."""
    model = makeTaskModel(seed=2)
    trainer = Trainer(model)
    samples = [{"bits": row} for row in makeBinaryRows(400, 8, seed=502)]

    report = trainer.flowTraining(samples, scoreCopy)

    assert report.steps == 400
    assert report.calcRewardMeanLast(100) > 0.6


def test_Auditor_doesKeepABoundedWindow():
    """Asserts the windows do not grow without bound over a long run."""
    auditor = Auditor(HazeHyper(audit_window=10))

    for step in range(500):
        auditor.onOutcome(float(step % 2), 0.5)

    assert len(auditor.rewards) == 10 and len(auditor.confidences) == 10
    assert auditor.isWindowFull()


# -- Boundary ------------------------------------------------------------------------------


def test_calcGrowthPlan_doesNotGrowAMeshThatIsPerforming():
    """Asserts a mesh below the error threshold is left alone."""
    hyper = HazeHyper(audit_window=5, growth_threshold=0.5)

    plan = calcGrowthPlan([1.0] * 5, [0.9] * 5, MeshCounts(nexus=64, terminus=32), hyper)

    assert not plan.triggered and plan.isEmpty
    assert plan.error_rate == pytest.approx(0.0)


def test_calcGrowthAmount_doesAlwaysAddAtLeastOne():
    """Asserts a triggered plan never resolves to adding nothing through rounding."""
    assert calcGrowthAmount(4, error=0.51, uncertainty=0.0, hyper=HazeHyper()) >= 1


def test_calcGrowthAmount_doesReturnNothingForAnEmptyPopulation():
    """Asserts a population that does not exist is not grown proportionally to itself."""
    assert calcGrowthAmount(0, error=1.0, uncertainty=1.0, hyper=HazeHyper()) == 0


def test_applyPrune_doesReturnQuietlyWhenNothingIsWeak():
    """Asserts a healthy mesh is not compacted, so edge indices stay put."""
    model = makeTaskModel(seed=4)
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    before = model.mesh.counts.edges

    report = applyPrune(model.mesh, model.config.hyper)

    assert report.edges_removed == 0
    assert model.mesh.counts.edges == before


def test_applyGrowth_doesIgnoreAnEmptyPlan():
    """Asserts an untriggered plan changes nothing at all."""
    model = makeTaskModel(seed=5)
    model({"bits": [0, 1, 0, 1, 0, 1, 0, 1]})
    before = model.calcMeshSize()

    assert applyGrowth(model.mesh, GrowthPlan()) == 0
    assert model.calcMeshSize() == before


def test_calcErrorRate_doesHandleAnEmptyWindow():
    """Asserts an unstarted run reports no error rather than dividing by zero."""
    assert calcErrorRate([]) == 0.0
    assert calcUncertaintyRate([]) == 0.0


# -- Error ---------------------------------------------------------------------------------


def test_flowRecoverSignal_doesRaiseWhenRecoveryIsRefused():
    """Asserts an unrecoverable mesh fails loudly rather than looping to the cap silently."""
    model = makeTaskModel(seed=6, relearn_limit=3)
    trainer = Trainer(model)
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    live = model.mesh.counts.edges
    model.mesh.strength[:live] = 0.0
    model.mesh.log_str[:live] = torch.full((live,), -13.8)

    with pytest.raises(SignalDidNotReachMotorsError):
        trainer.flowRecoverSignal({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})


def test_flowRecoverSignal_doesNotRecoverInEvalMode():
    """Asserts a deployed model reports that it cannot answer rather than silently rewiring."""
    model = makeTaskModel(seed=6)
    trainer = Trainer(model)
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    live = model.mesh.counts.edges
    model.mesh.strength[:live] = 0.0
    model.mesh.log_str[:live] = torch.full((live,), -13.8)
    model.eval()

    with pytest.raises(SignalDidNotReachMotorsError):
        trainer.flowRecoverSignal({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})


# -- Chaos ---------------------------------------------------------------------------------


def test_flowTraining_doesSurviveAGrowthAndPruneStorm():
    """Asserts a run that always fails grows and prunes repeatedly without corrupting itself.

    Every audit triggers growth here, which is the worst case for index stability: growth
    reallocates the slab and pruning permutes every edge array, both while the mesh is being
    read on every step.
    """
    model = makeTaskModel(seed=7, audit_window=5, growth_threshold=0.1)
    trainer = Trainer(model)
    samples = [{"bits": row} for row in makeBinaryRows(200, 8, seed=99)]

    report = trainer.flowTraining(samples, lambda answer, inputs: 0.0)

    mesh = model.mesh
    live = mesh.counts.edges
    assert report.grew > 0
    assert torch.isfinite(mesh.strength[:live]).all()
    assert int(mesh.src[:live].max()) < mesh.capacity.neurons
    assert int(mesh.dst[:live].max()) < mesh.capacity.neurons
    assert all(ids.numel() == 0 for ids in mesh.findOrphans().values())
    mesh.ensureMeshConsistent()


def test_flowTraining_doesSurviveAMeshPrunedToAlmostNothing():
    """Asserts a mesh whose edges are nearly all dead still trains rather than raising."""
    model = makeTaskModel(seed=8)
    trainer = Trainer(model)
    model({"bits": [1, 1, 1, 1, 1, 1, 1, 1]})
    live = model.mesh.counts.edges
    model.mesh.strength[:live] = 0.05
    model.mesh.log_str[:live] = model.mesh.strength[:live].abs().clamp_min(1e-6).log()
    applyPrune(model.mesh, model.config.hyper)

    report = trainer.flowTraining(
        [{"bits": row} for row in makeBinaryRows(60, 8, seed=11)], scoreCopy
    )

    assert report.steps == 60
    assert model.mesh.counts.edges > 0
    assert all(ids.numel() == 0 for ids in model.mesh.findOrphans().values())


def test_applyGrowth_doesGrowPastTheAllocatedCapacity():
    """Asserts growth beyond the slab reallocates and remaps rather than overflowing."""
    model = makeTaskModel(seed=9)
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    mesh = model.mesh
    room = mesh.capacity.nexus
    terminus_before = mesh.findNeuronIds(NeuronKind.TERMINUS).tolist()

    applyGrowth(mesh, GrowthPlan(nexus=room * 2, triggered=True))

    assert mesh.capacity.nexus > room
    assert mesh.counts.nexus == 32 + room * 2
    assert mesh.findNeuronIds(NeuronKind.TERMINUS).numel() == len(terminus_before)
    mesh.ensureMeshConsistent()
