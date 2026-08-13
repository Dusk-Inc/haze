"""Tests for checkpoint round-tripping and the checks that keep its indices honest."""

import json
from pathlib import Path

import pytest
import torch

from haze import Haze, makeHaze
from haze.errors import CheckpointCorruptError, UntrustedImplementationError
from haze.tokens import NeuronKind, defaults

from .conftest import StubDecoder, StubEncoder, assertMeshEqual


def saveTrusted(model: Haze, path: Path) -> Path:
    """Saves a model and returns the checkpoint directory, for tests that reload it."""
    model.save_pretrained(path)
    return path


# -- Domain --------------------------------------------------------------------------------


def test_savePretrained_doesWriteTheDocumentedFileSet(mesh_model: Haze, tmp_path: Path):
    """Asserts a checkpoint holds a config, a label table, and a tensor payload."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")

    names = {entry.name for entry in path.iterdir()}
    assert {defaults.CONFIG_FILE, defaults.LABELS_FILE, defaults.WEIGHTS_FILE} <= names

    config = json.loads((path / defaults.CONFIG_FILE).read_text())
    assert config["architecture"] == defaults.ARCHITECTURE
    assert config["counts"]["edges"] == mesh_model.mesh.counts.edges


def test_fromPretrained_doesRoundTripAFreshMesh(tmp_path: Path):
    """Asserts a model with no ports round-trips buffer for buffer."""
    model = makeHaze(nexus_size=8, terminus_size=4, seed=21)
    saveTrusted(model, tmp_path / "ckpt")

    assertMeshEqual(model, Haze.from_pretrained(str(tmp_path / "ckpt")))


def test_fromPretrained_doesRoundTripAfterGrowth(mesh_model: Haze, tmp_path: Path):
    """Asserts a model whose sensors and motors were allocated on demand round-trips exactly."""
    saveTrusted(mesh_model, tmp_path / "ckpt")

    back = Haze.from_pretrained(str(tmp_path / "ckpt"), trust_remote_code=True)

    assertMeshEqual(mesh_model, back)
    assert back.ports.findLabels("bit").values == [0, 1]
    assert back.ports.sensor_ids["bits"].tolist() == mesh_model.ports.sensor_ids["bits"].tolist()


def test_fromPretrained_doesRoundTripAfterRelayout(tmp_path: Path):
    """Asserts a model that outgrew a capacity block round-trips at its new capacity."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=23)
    model.registerEncoder("wide", StubEncoder(width=200))
    model.ports.ensureSensorCapacity("wide", 200)
    assert model.mesh.capacity.sensors >= 200

    saveTrusted(model, tmp_path / "ckpt")
    back = Haze.from_pretrained(str(tmp_path / "ckpt"), trust_remote_code=True)

    assertMeshEqual(model, back)


def test_fromPretrained_doesRoundTripAfterPruning(mesh_model: Haze, tmp_path: Path):
    """Asserts a compacted mesh round-trips, since compaction rewrites every edge array."""
    keep = torch.zeros(mesh_model.mesh.capacity.edges, dtype=torch.bool)
    keep[: mesh_model.mesh.counts.edges : 3] = True
    mesh_model.mesh.compactEdges(keep)
    mesh_model.mesh.ensureMeshConsistent()

    saveTrusted(mesh_model, tmp_path / "ckpt")
    back = Haze.from_pretrained(str(tmp_path / "ckpt"), trust_remote_code=True)

    assertMeshEqual(mesh_model, back)


def test_fromPretrained_doesRestoreGeneratorState(mesh_model: Haze, tmp_path: Path):
    """Asserts the model's own generator round-trips, so behavior resumes rather than restarts."""
    saveTrusted(mesh_model, tmp_path / "ckpt")
    back = Haze.from_pretrained(str(tmp_path / "ckpt"), trust_remote_code=True)

    original = torch.empty(8).uniform_(0, 1, generator=mesh_model.mesh.generator)
    restored = torch.empty(8).uniform_(0, 1, generator=back.mesh.generator)

    assert torch.equal(original, restored)


def test_savePretrained_doesReplaceAnEarlierCheckpointInPlace(mesh_model: Haze, tmp_path: Path):
    """Asserts saving twice to one path leaves only the newer checkpoint."""
    path = tmp_path / "ckpt"
    saveTrusted(mesh_model, path)
    first = mesh_model.mesh.counts.edges

    mesh_model.mesh.connectSensors(mesh_model.mesh.allocNeuronIds(2, NeuronKind.SENSOR, owner=0))
    saveTrusted(mesh_model, path)

    config = json.loads((path / defaults.CONFIG_FILE).read_text())
    assert config["counts"]["edges"] > first
    assert not (tmp_path / "ckpt.tmp").exists()


# -- Boundary ------------------------------------------------------------------------------


def test_fromPretrained_doesLoadAModelWithNoPorts(tmp_path: Path):
    """Asserts a checkpoint carrying an empty label table loads without special-casing."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    saveTrusted(model, tmp_path / "ckpt")

    back = Haze.from_pretrained(str(tmp_path / "ckpt"))

    assert back.ports.toLabelTable() == {}
    assertMeshEqual(model, back)


def test_fromPretrained_doesLoadASingleLabelDecoder(tmp_path: Path):
    """Asserts a decoder with exactly one label round-trips, the degenerate entropy case."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    model.registerDecoder("only", StubDecoder(labels=["yes"]))
    saveTrusted(model, tmp_path / "ckpt")

    back = Haze.from_pretrained(str(tmp_path / "ckpt"), trust_remote_code=True)

    assert back.ports.findLabels("only").values == ["yes"]


def test_fromPretrained_doesPreserveLabelValueTypes(tmp_path: Path):
    """Asserts labels come back as the Python types they went in as, which JSON alone erases."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=1)
    model.registerDecoder("words", StubDecoder(labels=["a", "b"]))
    model.registerDecoder("flags", StubDecoder(labels=[True, False]))
    saveTrusted(model, tmp_path / "ckpt")

    back = Haze.from_pretrained(str(tmp_path / "ckpt"), trust_remote_code=True)

    assert back.ports.findLabels("words").value_type == "str"
    assert back.ports.findLabels("flags").value_type == "bool"


# -- Error ---------------------------------------------------------------------------------


def test_fromPretrained_doesRefuseAnUntrustedImplementation(mesh_model: Haze, tmp_path: Path):
    """Asserts a checkpoint naming a non-first-party import path does not execute it."""
    saveTrusted(mesh_model, tmp_path / "ckpt")

    with pytest.raises(UntrustedImplementationError, match="allow-list"):
        Haze.from_pretrained(str(tmp_path / "ckpt"))


def test_fromPretrained_doesRefuseAMissingConfig(tmp_path: Path):
    """Asserts a checkpoint with no config raises a named error, since it cannot be sized."""
    (tmp_path / "ckpt").mkdir()

    with pytest.raises(CheckpointCorruptError, match="missing"):
        Haze.from_pretrained(str(tmp_path / "ckpt"))


def test_fromPretrained_doesRefuseAForeignArchitecture(mesh_model: Haze, tmp_path: Path):
    """Asserts a config for some other model is refused by name rather than half-loaded."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    config = json.loads((path / defaults.CONFIG_FILE).read_text())
    config["architecture"] = "transformer"
    (path / defaults.CONFIG_FILE).write_text(json.dumps(config))

    with pytest.raises(CheckpointCorruptError, match="architecture"):
        Haze.from_pretrained(str(path))


def test_fromPretrained_doesRefuseAFutureFormatVersion(mesh_model: Haze, tmp_path: Path):
    """Asserts a checkpoint written by a newer build is refused rather than misread."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    config = json.loads((path / defaults.CONFIG_FILE).read_text())
    config["format_version"] = defaults.FORMAT_VERSION + 1
    (path / defaults.CONFIG_FILE).write_text(json.dumps(config))

    with pytest.raises(CheckpointCorruptError, match="format version"):
        Haze.from_pretrained(str(path))


def test_fromPretrained_doesRefuseCountsExceedingCapacity(mesh_model: Haze, tmp_path: Path):
    """Asserts a config that could not allocate a model large enough for its tensors is refused."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    config = json.loads((path / defaults.CONFIG_FILE).read_text())
    config["counts"]["edges"] = config["capacity"]["edges"] + 1
    (path / defaults.CONFIG_FILE).write_text(json.dumps(config))

    with pytest.raises(CheckpointCorruptError, match="config"):
        Haze.from_pretrained(str(path))


def test_fromPretrained_doesRefuseAShapeMismatch(mesh_model: Haze, tmp_path: Path):
    """Asserts a tensor whose shape disagrees with the config names itself in the error."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    config = json.loads((path / defaults.CONFIG_FILE).read_text())
    config["capacity"]["edges"] = config["capacity"]["edges"] * 2
    (path / defaults.CONFIG_FILE).write_text(json.dumps(config))

    with pytest.raises(CheckpointCorruptError, match="strength|src|dst|shape"):
        Haze.from_pretrained(str(path), trust_remote_code=True)


# -- Chaos ---------------------------------------------------------------------------------


def test_fromPretrained_doesRefuseALabelBoundOutsideTheSlab(mesh_model: Haze, tmp_path: Path):
    """Asserts an out-of-range motor id is caught rather than indexing into whatever is there."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    labels = json.loads((path / defaults.LABELS_FILE).read_text())
    labels["bit"]["motor_ids"][0] = 10**6
    (path / defaults.LABELS_FILE).write_text(json.dumps(labels))

    with pytest.raises(CheckpointCorruptError, match="outside the slab"):
        Haze.from_pretrained(str(path), trust_remote_code=True)


def test_fromPretrained_doesRefuseALabelBoundToANonMotor(mesh_model: Haze, tmp_path: Path):
    """Asserts a label pointing at an interneuron is caught.

    This is the index-drift failure the design is most exposed to: without the check the model
    loads without complaint and answers from the wrong neuron.
    """
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    labels = json.loads((path / defaults.LABELS_FILE).read_text())
    nexus = mesh_model.mesh.findNeuronIds(NeuronKind.NEXUS)[0].item()
    labels["bit"]["motor_ids"][0] = int(nexus)
    (path / defaults.LABELS_FILE).write_text(json.dumps(labels))

    with pytest.raises(CheckpointCorruptError, match="not a motor"):
        Haze.from_pretrained(str(path), trust_remote_code=True)


def test_fromPretrained_doesRefuseALabelTableNamingAnUnknownPort(mesh_model: Haze, tmp_path: Path):
    """Asserts a label table referring to a port the config never declared is caught."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    labels = json.loads((path / defaults.LABELS_FILE).read_text())
    labels["ghost"] = labels["bit"]
    (path / defaults.LABELS_FILE).write_text(json.dumps(labels))

    with pytest.raises(CheckpointCorruptError, match="does not declare"):
        Haze.from_pretrained(str(path), trust_remote_code=True)


def test_fromPretrained_doesRefuseMisalignedLabelColumns(mesh_model: Haze, tmp_path: Path):
    """Asserts a label table whose parallel lists disagree in length is rejected on read."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    labels = json.loads((path / defaults.LABELS_FILE).read_text())
    labels["bit"]["values"].append(99)
    (path / defaults.LABELS_FILE).write_text(json.dumps(labels))

    with pytest.raises(CheckpointCorruptError):
        Haze.from_pretrained(str(path), trust_remote_code=True)


def test_fromPretrained_doesRefuseATruncatedTensorPayload(mesh_model: Haze, tmp_path: Path):
    """Asserts a half-written tensor file raises a Haze error, not a raw framework error."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    blob = (path / defaults.WEIGHTS_FILE).read_bytes()
    (path / defaults.WEIGHTS_FILE).write_bytes(blob[: len(blob) // 2])

    with pytest.raises(CheckpointCorruptError):
        Haze.from_pretrained(str(path), trust_remote_code=True)


def test_fromPretrained_doesRefuseAMalformedConfig(mesh_model: Haze, tmp_path: Path):
    """Asserts unparseable config text raises a Haze error naming the file."""
    path = saveTrusted(mesh_model, tmp_path / "ckpt")
    (path / defaults.CONFIG_FILE).write_text("{not json")

    with pytest.raises(CheckpointCorruptError, match="could not be read|readable Haze config"):
        Haze.from_pretrained(str(path))
