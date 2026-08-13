"""Tests for codebook decoding: the reduction of one many-way choice to several binary ones."""

import pytest
import torch

from haze import Haze, makeHaze
from haze.errors import LabelSpaceError
from haze.functions.codebook import (
    calcBitMargins,
    calcCodeDistance,
    calcCodeWidth,
    findNearestCode,
    makeCodebook,
    makeRandomCode,
    makeThermometerCode,
    toCodeBits,
)
from haze.functions.learning import calcCodeSeeds, switchCodeChoice
from haze.functions.score import TASKS, makeBinaryRows
from haze.models import LabelEntry
from haze.modules.decoders import CodeBook
from haze.modules.encoders import NumericEncoder


def makeCodedModel(seed: int, count: int, **kwargs):
    """Returns a model whose single decoder answers by codeword."""
    model = makeHaze(nexus_size=64, terminus_size=32, seed=seed, growth_threshold=0.99)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", CodeBook(labels=list(range(count)), **kwargs))
    return model


# -- Domain --------------------------------------------------------------------------------


def test_makeThermometerCode_doesAskOnlyThresholdQuestions():
    """Asserts every bit asks whether the label exceeds a position, which is learnable.

    Writing the label out in binary instead would make its lowest bit ask whether the label is
    odd — parity, the one function Haze provably cannot represent. See specs/decoding.md.
    """
    codes = makeThermometerCode(5)

    assert codes[0] == [0, 0, 0, 0]
    assert codes[2] == [1, 1, 0, 0]
    assert codes[4] == [1, 1, 1, 1]
    for position in range(4):
        column = [code[position] for code in codes]
        assert column == sorted(column), "a thermometer bit must be monotone in the label"


def test_makeRandomCode_doesSeparateItsCodewords():
    """Asserts an unstructured code still spreads its labels apart, which is what decodes."""
    codes = makeRandomCode(9, calcCodeWidth(9), torch.Generator().manual_seed(0))

    assert len({tuple(c) for c in codes}) == 9
    assert calcCodeDistance(codes) >= 2, "a distance below 2 survives no wrong bit at all"


def test_makeRandomCode_doesRejectAConstantColumn():
    """Asserts no bit asks a question whose answer never varies.

    Such a bit consumes two motors and carries nothing, and its motor pair would be trained on
    noise forever.
    """
    codes = makeRandomCode(8, 10, torch.Generator().manual_seed(3))

    for position in range(len(codes[0])):
        column = {code[position] for code in codes}
        assert column == {0, 1}


def test_toCodeBits_doesReadPairsAsATwoWayRace():
    """Asserts each bit is decided by which of its two motors won, not by a fixed cutoff."""
    states = torch.tensor([0.9, 0.2, 0.1, 0.7, 0.5, 0.4])

    assert toCodeBits(states).tolist() == [1, 0, 1]


def test_findNearestCode_doesResolveAnUnreadableCodeword():
    """Asserts a read matching no codeword resolves to the likeliest one rather than failing.

    This is what the spare bits buy, and it matters beyond ordinary noise: some bits will be
    functions the mesh cannot represent, and redundancy turns those from fatal into wasteful.
    """
    codes = [[0, 0, 0, 0], [1, 1, 1, 1]]

    assert findNearestCode(torch.tensor([1, 1, 0, 1]), codes) == 1
    assert findNearestCode(torch.tensor([0, 1, 0, 0]), codes) == 0


def test_calcCodeSeeds_doesGiveEachPairTheUndilutedTwoLabelSignal():
    """Asserts the winner of every pair takes the full gain and its partner the full negative.

    Nothing is divided among rivals, because within a pair there is exactly one rival. That is
    the whole reason a codebook restores the signal a large label set destroys.
    """
    seeds = calcCodeSeeds([10, 11, 20, 21], torch.tensor([1, 0]), gain=0.4)

    assert seeds == {10: 0.4, 11: -0.4, 21: 0.4, 20: -0.4}
    assert sum(seeds.values()) == pytest.approx(0.0)


def test_setLabelsActive_doesAllocateOneMotorPairPerBit():
    """Asserts motor count is logarithmic in label count, not equal to it."""
    model = makeCodedModel(seed=1, count=9)
    entry = model.ports.findLabels("bit")

    assert entry.isCoded and entry.width == 8
    assert len(entry.bit_motors) == 16
    assert len(set(entry.bit_motors)) == 16


def test_CodeBook_doesDecodeToAnActiveLabel():
    """Asserts the answer is always a label the decoder currently holds active."""
    model = makeCodedModel(seed=2, count=9)

    answers = {model({"bits": row})["bit"][0] for row in makeBinaryRows(30, 8, seed=4)}

    assert answers and answers <= set(range(9))


def test_CodeBook_doesInferAnOrdinalLabelSet():
    """Asserts consecutive integers are read as a scale and given threshold bits."""
    ordinal = CodeBook(labels=[0, 1, 2, 3])
    named = CodeBook(labels=["cat", "dog", "bird", "fish"])

    assert ordinal.ensureOrdinal(4) is True
    assert named.ensureOrdinal(4) is False
    assert ordinal.calcCodebook(4) == makeThermometerCode(4)


# -- Boundary ------------------------------------------------------------------------------


def test_calcCodeWidth_doesSpendMoreThanABareEnumeration():
    """Asserts the width leaves room for error correction rather than using every pattern."""
    for count in (2, 3, 9, 256, 10_000):
        assert calcCodeWidth(count) > max(1, (count - 1).bit_length())


def test_calcCodeWidth_doesStayLogarithmic():
    """Asserts a large vocabulary costs motors in the dozens rather than the thousands."""
    assert calcCodeWidth(10_000) * 2 < 100


def test_switchCodeChoice_doesLandOnAWholeCodeword():
    """Asserts exploring reaches a label the mesh could have answered, not a bit pattern.

    Flipping one random bit reads as the natural local move and does not work: a code with spare
    distance is built so a single wrong bit still decodes to the same label, so at a distance of
    4 bit-level exploration cannot change the answer at all. See specs/decoding.md.
    """
    codes = [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]]
    states = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    generator = torch.Generator().manual_seed(1)

    explored = [switchCodeChoice(states, codes, 1.0, generator) for _ in range(25)]

    for result in explored:
        assert toCodeBits(result).tolist() in codes
        assert torch.equal(torch.sort(result).values, torch.sort(states).values)
    assert len({tuple(toCodeBits(r).tolist()) for r in explored}) > 1


def test_switchCodeChoice_doesEscapeAHighDistanceCode():
    """Asserts redundancy no longer locks the answer in, which single-bit flipping did.

    Measured, that lock cost a two-label task 0.91 against 0.27 — the mesh could not try the only
    alternative it had.
    """
    codes = [[0, 0, 0, 0], [1, 1, 1, 1]]
    states = torch.tensor([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    generator = torch.Generator().manual_seed(2)

    reached = {
        tuple(toCodeBits(switchCodeChoice(states, codes, 1.0, generator)).tolist())
        for _ in range(30)
    }

    assert (0, 0, 0, 0) in reached, "exploration never reached the other label"


def test_switchCodeChoice_doesNothingAtZeroRate():
    """Asserts a greedy readout is left exactly alone."""
    states = torch.tensor([0.9, 0.1, 0.2, 0.8])
    codes = [[1, 0], [0, 1]]

    assert torch.equal(switchCodeChoice(states, codes, 0.0, torch.Generator()), states)


def test_makeCodebook_doesFallBackToRandomForALargeLabelSet():
    """Asserts thermometer is dropped once its per-label bit stops paying for itself."""
    generator = torch.Generator().manual_seed(0)

    small = makeCodebook(8, None, generator, ordinal=True)
    large = makeCodebook(64, None, generator, ordinal=True)

    assert len(small[0]) == 7
    assert len(large[0]) < 64


def test_calcBitMargins_doesReportEachPairsDecisiveness():
    """Asserts confidence is read from the gap within a pair, not across pairs."""
    margins = calcBitMargins(torch.tensor([0.9, 0.1, 0.5, 0.45]))

    assert margins.tolist() == pytest.approx([0.8, 0.05])


# -- Error ---------------------------------------------------------------------------------


def test_setCodedLabels_doesRefuseToReassignACodeword():
    """Asserts growing the label set never rewrites the codewords already in use.

    A label's identity is its codeword exactly as it was its motor index, so reassigning one
    would silently transfer everything the mesh learned to a different answer.
    """
    model = makeCodedModel(seed=3, count=4)

    with pytest.raises(LabelSpaceError, match="codeword"):
        model.ports.setLabelsActive("bit", [0, 1, 2, 3, 4, 5, 6, 7, 8])


def test_LabelEntry_doesRefuseAMotorBoundToTwoBits():
    """Asserts a motor cannot serve two pairs, which would couple unrelated decisions."""
    with pytest.raises(ValueError, match="more than one bit pair"):
        LabelEntry(
            value_type="int",
            values=[0, 1],
            motor_ids=[],
            active=[True, True],
            codes=[[0], [1]],
            bit_motors=[5, 5],
        )


def test_LabelEntry_doesRefuseTwoLabelsSharingACodeword():
    """Asserts labels that could never be told apart are rejected at construction."""
    with pytest.raises(ValueError, match="share a codeword"):
        LabelEntry(
            value_type="int",
            values=[0, 1],
            motor_ids=[],
            active=[True, True],
            codes=[[1, 0], [1, 0]],
            bit_motors=[5, 6, 7, 8],
        )


def test_LabelEntry_doesRefuseAMismatchedMotorCount():
    """Asserts a codebook whose width disagrees with its motor pairs is refused."""
    with pytest.raises(ValueError, match="needs 4 motors"):
        LabelEntry(
            value_type="int",
            values=[0, 1],
            motor_ids=[],
            active=[True, True],
            codes=[[0, 1], [1, 0]],
            bit_motors=[5, 6],
        )


def test_calcCodeWidth_doesRefuseASingleLabel():
    """Asserts a codebook with nothing to distinguish is refused rather than degenerate."""
    with pytest.raises(LabelSpaceError):
        calcCodeWidth(1)


# -- Chaos ---------------------------------------------------------------------------------


def test_CodeBook_doesRoundTripThroughACheckpoint(tmp_path):
    """Asserts the codebook travels with the model, since it is the label's identity."""
    model = makeCodedModel(seed=5, count=9)
    for row in makeBinaryRows(20, 8, seed=7):
        model({"bits": row})
        model.learn(1.0)

    model.save_pretrained(tmp_path / "ckpt")
    loaded = Haze.from_pretrained(str(tmp_path / "ckpt"))

    before, after = model.ports.findLabels("bit"), loaded.ports.findLabels("bit")
    assert after.codes == before.codes
    assert after.bit_motors == before.bit_motors
    row = [1, 0, 1, 1, 0, 0, 1, 0]
    model.eval()
    loaded.eval()
    assert loaded({"bits": row})["bit"][0] == model({"bits": row})["bit"][0]


def test_CodeBook_doesSurviveEveryPairTied():
    """Asserts a mesh whose pairs all read identically still answers rather than raising."""
    model = makeCodedModel(seed=6, count=9)
    model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
    entry = model.ports.findLabels("bit")
    states = torch.full((len(entry.bit_motors),), 0.4)

    assert model.ports.impls["bit"].decodeMotors(states, entry) in entry.values


def test_CodeBook_doesTrainWithoutCorruptingTheMesh():
    """Asserts a full run over a nine-label task leaves every invariant intact."""
    model = makeCodedModel(seed=7, count=9)
    task = TASKS["first-set"]

    for row in makeBinaryRows(150, 8, seed=8):
        answer = model({"bits": row})
        model.learn(1.0 if answer["bit"][0] == task(row) else 0.0)

    live = model.mesh.counts.edges
    assert torch.isfinite(model.mesh.strength[:live]).all()
    assert all(ids.numel() == 0 for ids in model.mesh.findOrphans().values())
    model.mesh.ensureMeshConsistent()
