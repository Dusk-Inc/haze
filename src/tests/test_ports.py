"""Tests for port registration, sensor allocation, and label-to-motor binding."""

import pytest

from haze import makeHaze
from haze.errors import (
    LabelSpaceError,
    LearningDisabledError,
    PortAlreadyRegisteredError,
    PortNotFoundError,
)
from haze.tokens import LabelType, NeuronKind

from .conftest import StubDecoder, StubEncoder


# -- Domain --------------------------------------------------------------------------------


def test_registerDecoder_doesAllocateOneMotorPerLabel():
    """Asserts each declared label claims its own motor, wired from every terminus."""
    model = makeHaze(nexus_size=6, terminus_size=4, seed=31)
    model.registerDecoder("bit", StubDecoder(labels=["a", "b", "c"]))

    entry = model.ports.findLabels("bit")
    assert len(entry.motor_ids) == 3
    assert len(set(entry.motor_ids)) == 3
    assert model.mesh.counts.motors == 3
    for motor in entry.motor_ids:
        assert int(model.mesh.kind[motor]) == int(NeuronKind.MOTOR)


def test_setLabelsActive_doesKeepBindingsWhenTheLabelSetGrows():
    """Asserts a label added later leaves every earlier label on the motor it already had."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=31)
    model.registerDecoder("bit", StubDecoder(labels=[0, 1]))
    before = dict(zip(model.ports.findLabels("bit").values, model.ports.findLabels("bit").motor_ids))

    model.ports.setLabelsActive("bit", [0, 1, 2])
    entry = model.ports.findLabels("bit")
    after = dict(zip(entry.values, entry.motor_ids))

    assert after[0] == before[0] and after[1] == before[1]
    assert 2 in after


def test_setLabelsActive_doesMaskRatherThanDeallocateWhenTheLabelSetNarrows():
    """Asserts a label that leaves the active set keeps its motor and can come back."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=31)
    model.registerDecoder("bit", StubDecoder(labels=[0, 1, 2]))
    original = dict(zip(model.ports.findLabels("bit").values, model.ports.findLabels("bit").motor_ids))

    model.ports.setLabelsActive("bit", [0, 2])
    entry = model.ports.findLabels("bit")

    assert entry.active == [True, False, True]
    assert not bool(model.mesh.active[original[1]])
    assert model.ports.findActiveMotorIds("bit").tolist() == [original[0], original[2]]

    model.ports.setLabelsActive("bit", [0, 1, 2])
    assert bool(model.mesh.active[original[1]])
    assert dict(zip(entry.values, entry.motor_ids)) == original


def test_registerPort_doesGiveEachPortItsOwnSlot():
    """Asserts slots are distinct, so ownership masks never collide."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    slots = [
        model.registerEncoder("a", StubEncoder(2)),
        model.registerEncoder("b", StubEncoder(2)),
        model.registerDecoder("c", StubDecoder(labels=[1])),
    ]

    assert len(set(slots)) == 3


def test_calcLabelType_doesDistinguishBoolFromInt():
    """Asserts a boolean label is not recorded as an integer.

    bool subclasses int, so the obvious check reports the wrong type and a decoder gets 1 back
    where it stored True.
    """
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerDecoder("flags", StubDecoder(labels=[True, False]))
    model.registerDecoder("counts", StubDecoder(labels=[0, 1]))

    assert model.ports.findLabels("flags").value_type == LabelType.BOOL
    assert model.ports.findLabels("counts").value_type == LabelType.INT


def test_findMotorId_doesResolveVectorLabels():
    """Asserts list-valued labels are usable as table keys despite being unhashable."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerDecoder("vec", StubDecoder(labels=[[0.0, 1.0], [1.0, 0.0]]))

    assert model.ports.findLabels("vec").value_type == LabelType.VECTOR
    assert model.ports.findMotorId("vec", [0.0, 1.0]) != model.ports.findMotorId("vec", [1.0, 0.0])


# -- Boundary ------------------------------------------------------------------------------


def test_ensureSensorCapacity_doesAllocateNothingWhenWidthIsUnchanged():
    """Asserts asking for the width already held allocates no further sensors."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerEncoder("bits", StubEncoder(3))
    first = model.ports.ensureSensorCapacity("bits", 3)
    edges = model.mesh.counts.edges

    second = model.ports.ensureSensorCapacity("bits", 3)

    assert first.tolist() == second.tolist()
    assert model.mesh.counts.edges == edges


def test_setLabelsActive_doesAcceptASingleLabel():
    """Asserts a one-label decoder binds without special-casing."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerDecoder("only", StubDecoder(labels=["yes"]))

    assert model.ports.findActiveMotorIds("only").numel() == 1


def test_ensureSensorCapacity_doesReturnAPrefixWhenAskedForFewer():
    """Asserts a narrower request returns the first sensors rather than reallocating."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerEncoder("bits", StubEncoder(5))
    wide = model.ports.ensureSensorCapacity("bits", 5)

    narrow = model.ports.ensureSensorCapacity("bits", 2)

    assert narrow.tolist() == wide.tolist()[:2]


# -- Error ---------------------------------------------------------------------------------


def test_findPortSpec_doesNameTheRegisteredPortsWhenAKeyIsUnknown():
    """Asserts an unknown key raises rather than silently creating a port for a typo."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerEncoder("bits", StubEncoder(2))

    with pytest.raises(PortNotFoundError, match="bits"):
        model.ports.findPortSpec("bts")


def test_registerPort_doesRefuseADuplicateKey():
    """Asserts re-registering a key raises rather than rebinding its neurons."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerEncoder("bits", StubEncoder(2))

    with pytest.raises(PortAlreadyRegisteredError, match="bits"):
        model.registerEncoder("bits", StubEncoder(2))


def test_setLabelsActive_doesRefuseAnEmptyLabelSet():
    """Asserts a decoder with no labels raises rather than producing a mesh with no output."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerDecoder("bit", StubDecoder(labels=[0]))

    with pytest.raises(LabelSpaceError, match="empty"):
        model.ports.setLabelsActive("bit", [])


def test_setLabelsActive_doesRefuseDuplicateLabels():
    """Asserts a repeated label raises rather than binding one label to two motors."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerDecoder("bit", StubDecoder(labels=[0]))

    with pytest.raises(LabelSpaceError, match="duplicated"):
        model.ports.setLabelsActive("bit", [1, 1])


def test_findLabels_doesRefuseADecoderWithNoLabelsYet():
    """Asserts reading labels before any are set raises rather than returning an empty table."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.ports.registerPort("bare", StubDecoder(labels=[0]), role=_decoderRole())

    with pytest.raises(LabelSpaceError, match="no labels"):
        model.ports.findLabels("bare")


def test_learn_doesRefuseInEvalMode():
    """Asserts learning in eval raises rather than silently doing nothing."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31).eval()

    with pytest.raises(LearningDisabledError, match="eval mode"):
        model.ensureLearningEnabled()


# -- Chaos ---------------------------------------------------------------------------------


def test_setLabelsActive_doesRefuseAnUnpersistableLabelType():
    """Asserts a label that cannot round-trip through JSON is refused at binding time."""
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.ports.registerPort("odd", StubDecoder(labels=[0]), role=_decoderRole())

    with pytest.raises(LabelSpaceError, match="cannot be persisted"):
        model.ports.setLabelsActive("odd", [object()])


def test_setLabelsActive_doesRefuseMixingLabelTypes():
    """Asserts a differently-typed label added later is refused rather than stored unfaithfully.

    The value type is recorded once per decoder, so a string joining an integer table would
    come back from a checkpoint as the wrong type.
    """
    model = makeHaze(nexus_size=4, terminus_size=2, seed=31)
    model.registerDecoder("bit", StubDecoder(labels=[0, 1]))

    with pytest.raises(LabelSpaceError, match="cannot round-trip"):
        model.ports.setLabelsActive("bit", [0, 1, "two"])


def test_setLabelsActive_doesSurviveARepeatedlyChangingLabelSet():
    """Asserts churning the active set never reassigns a binding or leaks motors."""
    model = makeHaze(nexus_size=6, terminus_size=3, seed=37)
    model.registerDecoder("bit", StubDecoder(labels=[0]))
    bindings: dict[int, int] = {}

    for round_index in range(20):
        labels = list(range(round_index % 5 + 1))
        model.ports.setLabelsActive("bit", labels)
        entry = model.ports.findLabels("bit")
        for value, motor in zip(entry.values, entry.motor_ids):
            if value in bindings:
                assert bindings[value] == motor, f"label {value} was rebound"
            bindings[value] = motor
        model.mesh.ensureMeshConsistent()

    assert model.mesh.counts.motors == len(bindings)


def _decoderRole():
    """Returns the decoder port role, imported lazily to keep the test module's imports flat."""
    from haze.tokens import PortRole

    return PortRole.DECODER
