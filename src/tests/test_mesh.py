"""Tests for the tensor slab: allocation, wiring, relayout, and compaction."""

import pytest
import torch

from haze import makeHaze
from haze.errors import CapacityExceededError, LabelSpaceError
from haze.models import HazeConfig, MeshCapacity
from haze.modules import MeshState
from haze.tokens import NeuronKind

from .conftest import StubEncoder


# -- Domain --------------------------------------------------------------------------------


def test_buildMesh_doesWireBothMeshesAndOnlyInterneurons():
    """Asserts a fresh mesh holds interneurons wired to each other and nothing else."""
    model = makeHaze(nexus_size=10, terminus_size=5, seed=3)
    counts = model.calcMeshSize()

    assert counts["nexus"] == 10
    assert counts["terminus"] == 5
    assert counts["sensors"] == 0 and counts["motors"] == 0
    assert counts["edges"] > 0
    model.mesh.ensureMeshConsistent()


def test_connectSensors_doesSampleTheNexusRatherThanTakeAllOfIt():
    """Asserts a sensor projects to a sample of the nexus, not to every interneuron.

    Wiring every sensor to every interneuron makes each feature excite the same population in
    the same way, leaving no feature-specific pathway for learning to strengthen.
    """
    model = makeHaze(nexus_size=48, terminus_size=8, seed=3)
    fanout = model.config.hyper.sensor_fanout
    before = model.mesh.counts.edges

    sensors = model.mesh.allocNeuronIds(3, NeuronKind.SENSOR, owner=0)
    model.mesh.connectSensors(sensors)
    added = model.mesh.counts.edges - before

    assert added <= 3 * fanout
    assert added < 3 * 48, "sensors are still fully bipartite with the nexus"


def test_connectMotors_doesGiveEachMotorItsOwnSampleOfTheTerminus():
    """Asserts two motors read different terminus populations rather than the identical one.

    Under fully bipartite wiring both motors see the same evidence and can differ only by edge
    strength, which is most of why competing motors receive near-identical totals.
    """
    model = makeHaze(nexus_size=16, terminus_size=40, seed=3)
    motors = model.mesh.allocNeuronIds(2, NeuronKind.MOTOR, owner=0)
    model.mesh.connectMotors(motors)

    live = slice(0, model.mesh.counts.edges)
    feeding = {
        int(m): {
            int(s)
            for s, d in zip(model.mesh.src[live].tolist(), model.mesh.dst[live].tolist())
            if d == int(m)
        }
        for m in motors.tolist()
    }
    first, second = feeding[motors[0].item()], feeding[motors[1].item()]

    assert first != second, "both motors draw the identical terminus population"
    assert len(first) <= model.config.hyper.motor_fanin


def test_connectMotors_doesRegisterEdgesAsLearnable():
    """Asserts terminus-to-motor edges carry a strength and epsilon like any other edge.

    The prior engine created these outside its registry, leaving them pinned at a hardcoded
    strength and invisible to learning.
    """
    model = makeHaze(nexus_size=6, terminus_size=4, seed=3)
    motors = model.mesh.allocNeuronIds(2, NeuronKind.MOTOR, owner=0)
    added = model.mesh.connectMotors(motors)

    assert added.numel() == 4 * 2
    assert bool((model.mesh.epsilon[added] > 0).all())
    assert bool((model.mesh.strength[added] < 1.0).all())


def test_allocNeuronIds_doesNotRenumberLiveNeuronsOnRelayout():
    """Asserts a block overflow preserves every edge's identity and its endpoints' roles."""
    config = HazeConfig(
        capacity=MeshCapacity(sensors=2, motors=2, nexus=8, terminus=4, edges=64),
        nexus_size=8,
        terminus_size=4,
        seed=5,
    )
    mesh = MeshState(config)
    mesh.buildMesh(8, 4)
    mesh.connectSensors(mesh.allocNeuronIds(2, NeuronKind.SENSOR, owner=0))
    mesh.connectMotors(mesh.allocNeuronIds(2, NeuronKind.MOTOR, owner=1))

    def fingerprint() -> list[tuple]:
        """Returns each edge keyed by its stable id and the roles of its endpoints."""
        n = mesh.counts.edges
        return sorted(
            zip(
                mesh.edge_id[:n].tolist(),
                mesh.kind[mesh.src[:n]].tolist(),
                mesh.owner[mesh.src[:n]].tolist(),
                mesh.kind[mesh.dst[:n]].tolist(),
                mesh.owner[mesh.dst[:n]].tolist(),
            )
        )

    before = fingerprint()
    motors_before = mesh.findNeuronIds(NeuronKind.MOTOR).tolist()

    mesh.allocNeuronIds(5, NeuronKind.SENSOR, owner=0)

    assert mesh.capacity.sensors >= 7
    assert mesh.findNeuronIds(NeuronKind.MOTOR).tolist() != motors_before, "motors should move"
    assert fingerprint() == before, "relayout changed what an edge means"
    mesh.ensureNoOrphans()
    mesh.ensureMeshConsistent()


def test_compactEdges_doesPermuteEveryEdgeArrayTogether():
    """Asserts compaction keeps each surviving edge's strength, epsilon, and id aligned."""
    model = makeHaze(nexus_size=8, terminus_size=4, seed=9)
    mesh = model.mesh
    live = mesh.counts.edges

    survivors = torch.zeros(mesh.capacity.edges, dtype=torch.bool)
    survivors[: mesh.capacity.edges] = False
    keep_idx = torch.arange(0, live, 2)
    survivors[keep_idx] = True

    expected = {
        int(mesh.edge_id[i]): (
            int(mesh.src[i]),
            int(mesh.dst[i]),
            float(mesh.strength[i]),
            float(mesh.epsilon[i]),
        )
        for i in keep_idx.tolist()
    }

    removed = mesh.compactEdges(survivors)

    assert removed == live - len(keep_idx)
    assert mesh.counts.edges == len(keep_idx)
    actual = {
        int(mesh.edge_id[i]): (
            int(mesh.src[i]),
            int(mesh.dst[i]),
            float(mesh.strength[i]),
            float(mesh.epsilon[i]),
        )
        for i in range(mesh.counts.edges)
    }
    assert actual == expected
    assert bool((mesh.src[: mesh.counts.edges].diff() >= 0).all()), "not sorted by source"

    mesh.ensureNoOrphans()
    mesh.ensureMeshConsistent()


def test_compactEdges_doesLeaveNoOrphanSlots():
    """Asserts dropped edges leave no dead slot that a later mask could select.

    The prior engine's orphaned strength and epsilon slots were reverse-learned on every
    reverse pass, forever.
    """
    model = makeHaze(nexus_size=6, terminus_size=3, seed=9)
    mesh = model.mesh
    live = mesh.counts.edges

    keep = torch.zeros(mesh.capacity.edges, dtype=torch.bool)
    keep[:3] = True
    mesh.compactEdges(keep)

    assert mesh.counts.edges == 3
    assert not bool(mesh.alive_e[3:live].any())
    assert bool((mesh.edge_id[3:live] == -1).all())


def test_findPortNeuronIds_doesGiveEachPortDisjointSensors():
    """Asserts two encoders of the same class hold disjoint sensors.

    The prior engine keyed its registry by encoder type, so two instances shared one pool and
    genuine multi-modal use was impossible.
    """
    model = makeHaze(nexus_size=6, terminus_size=3, seed=13)
    model.registerEncoder("left", StubEncoder(width=3))
    model.registerEncoder("right", StubEncoder(width=3))

    left = model.ports.ensureSensorCapacity("left", 3)
    right = model.ports.ensureSensorCapacity("right", 3)

    assert set(left.tolist()).isdisjoint(right.tolist())
    assert len(left) == len(right) == 3

    left_slot = model.ports.findPortSpec("left").slot
    right_slot = model.ports.findPortSpec("right").slot
    assert model.mesh.findPortNeuronIds(left_slot).tolist() == left.tolist()
    assert model.mesh.findPortNeuronIds(right_slot).tolist() == right.tolist()


def test_ensureSensorCapacity_doesKeepExistingSensorsWhenWidthGrows():
    """Asserts a widening feature set allocates only the shortfall, preserving learned indices."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=13)
    model.registerEncoder("bits", StubEncoder(width=2))

    first = model.ports.ensureSensorCapacity("bits", 2).tolist()
    second = model.ports.ensureSensorCapacity("bits", 5).tolist()

    assert second[:2] == first
    assert len(second) == 5


# -- Boundary ------------------------------------------------------------------------------


def test_growEdgeCapacity_doesDoubleRatherThanFitExactly():
    """Asserts an edge array that overflows by one doubles, so growth stays amortized."""
    config = HazeConfig(capacity=MeshCapacity(edges=16), nexus_size=0, terminus_size=0, seed=1)
    mesh = MeshState(config)
    src = torch.zeros(16, dtype=torch.int64)
    dst = torch.arange(16, dtype=torch.int64)
    mesh.alive_n[:] = True
    mesh.addMeshEdges(src, dst)

    assert mesh.capacity.edges == 16
    mesh.addMeshEdges(torch.tensor([0]), torch.tensor([17]))
    assert mesh.capacity.edges == 32


def test_allocNeuronIds_doesReturnNothingForAnEmptyRequest():
    """Asserts a zero-count allocation is a no-op rather than an error or a stray neuron."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    before = model.calcMeshSize()

    assert model.mesh.allocNeuronIds(0, NeuronKind.SENSOR).numel() == 0
    assert model.calcMeshSize() == before


def test_connectNeurons_doesClampFanoutToTheCandidateCount():
    """Asserts wiring into a candidate set smaller than the fan-out does not loop forever."""
    model = makeHaze(nexus_size=3, terminus_size=2, seed=1)

    assert model.mesh.counts.edges > 0
    model.mesh.ensureMeshConsistent()


def test_buildMesh_doesAcceptAnEmptyMesh():
    """Asserts a mesh with no interneurons builds without error, holding no edges."""
    config = HazeConfig(
        capacity=MeshCapacity(nexus=1, terminus=1, edges=8), nexus_size=0, terminus_size=0
    )
    mesh = MeshState(config)
    mesh.buildMesh(0, 0)

    assert mesh.counts.edges == 0
    mesh.ensureMeshConsistent()


# -- Error ---------------------------------------------------------------------------------


def test_ensureMeshConsistent_doesRejectAnEdgeIntoADeadNeuron():
    """Asserts a dangling edge endpoint is caught rather than silently propagated through."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=1)
    model.mesh.alive_n[model.mesh.dst[0]] = False

    with pytest.raises(LabelSpaceError, match="not live"):
        model.mesh.ensureMeshConsistent()


def test_ensureMeshConsistent_doesRejectADuplicateEdge():
    """Asserts the same source-destination pair appearing twice is caught."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=1)
    mesh = model.mesh
    mesh.addMeshEdges(mesh.src[:1].clone(), mesh.dst[:1].clone())

    with pytest.raises(LabelSpaceError, match="duplicate edge"):
        mesh.ensureMeshConsistent()


def test_ensureMeshConsistent_doesRejectAnOverfullEdgeCount():
    """Asserts a count exceeding capacity is reported rather than silently indexing past."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    model.mesh.counts.edges = model.mesh.capacity.edges + 1

    with pytest.raises(CapacityExceededError):
        model.mesh.ensureMeshConsistent()


# -- Chaos ---------------------------------------------------------------------------------


def test_addMeshEdges_doesSurviveARepeatedGrowthStorm():
    """Asserts many small growths in a row keep every invariant, exercising repeated doubling."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=17)
    mesh = model.mesh

    for round_index in range(25):
        sensors = mesh.allocNeuronIds(2, NeuronKind.SENSOR, owner=round_index)
        mesh.connectSensors(sensors)
        motors = mesh.allocNeuronIds(1, NeuronKind.MOTOR, owner=round_index)
        mesh.connectMotors(motors)
        mesh.ensureMeshConsistent()

    assert mesh.counts.sensors == 50
    assert mesh.counts.motors == 25
    assert mesh.counts.edges <= mesh.capacity.edges


def test_compactEdges_doesSurviveDroppingEverything():
    """Asserts compacting away every edge leaves a coherent, empty mesh."""
    model = makeHaze(nexus_size=5, terminus_size=3, seed=17)
    mesh = model.mesh

    removed = mesh.compactEdges(torch.zeros(mesh.capacity.edges, dtype=torch.bool))

    assert removed > 0
    assert mesh.counts.edges == 0
    assert not bool(mesh.alive_e.any())

    assert mesh.ensureNoOrphans() > 0
    mesh.ensureMeshConsistent()
