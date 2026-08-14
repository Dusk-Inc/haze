"""Codebooks: expressing one choice among many as several binary ones."""

import torch
from torch import Tensor

from ..errors import LabelSpaceError


def calcCodeWidth(count: int, redundancy: int = 2) -> int:
    """Returns how many bits to spend on `count` labels, intended to leave room for correction.

    More than the `log2(count)` a bare enumeration needs, on the reasoning that spare bits let a
    codeword be decoded to its nearest neighbour rather than silently becoming a different answer.

    **Measured, this width delivers none of that.** The books `makeRandomCode` returns at it have
    minimum distance 1 at every size tried — 16, 64, 256, 1,024, and 4,096 labels — so no error is
    survivable. That is what the width asks for rather than an unlucky draw: `count` random `m`-bit
    words expect `C(count,2)·(m+1)/2**m` pairs at distance 1 or less, and at `m = 2·log2(count)`
    that expectation exceeds one above a handful of labels. Distance `d` needs `m` large enough
    that `C(count,2)·V(m,d-1)/2**m < 1`, which for a million labels is 50 bits to correct one error
    and 141 to correct fifteen — still only 282 motors, so the width is worth spending.

    Left as it is rather than widened, because there is no per-bit error rate to size against yet:
    a trained mesh at seventeen labels emits a constant codeword. See specs/decoding.md.
    """
    if count < 2:
        raise LabelSpaceError("a codebook needs at least two labels to encode")
    minimum = max(1, (count - 1).bit_length())
    return max(minimum * redundancy, minimum + 3)


def makeThermometerCode(count: int) -> list[list[int]]:
    """Returns an ordinal code where bit `i` asks whether the label exceeds `i`.

    For a label that means a position on a scale, this is the code whose every bit is a threshold
    — "is the answer at least this large" — and thresholds are what the mesh finds easy. The
    obvious alternative, writing the label out in binary, makes its lowest bit ask whether the
    label is odd, which is parity, which is the one function Haze provably cannot represent.

    Costs `count - 1` bits, so it is for small ordinal spaces. A large one has to give up the
    per-bit guarantee and take a random code instead.
    """
    return [[1 if label > position else 0 for position in range(count - 1)] for label in range(count)]


def calcCodeDistance(codes: list[list[int]]) -> int:
    """Returns the smallest Hamming distance between any two codewords.

    How many wrong bits the codebook can absorb: a distance of `d` decodes correctly through
    `(d - 1) // 2` errors. A distance of 1 means no error is survivable and any single wrong bit
    changes the answer.

    Quadratic in the label count, and that is what bounds the label space rather than the motor
    count: the comparison materializes `count**2 * width` values, measured at 4·10**8 bytes and
    7.7 s for 4,096 labels and around 2·10**14 bytes for a million. A large codebook has to be
    generated with a known distance instead of drawn and measured. See specs/decoding.md.
    """
    if len(codes) < 2:
        return 0
    table = torch.tensor(codes, dtype=torch.int16)
    spread = (table.unsqueeze(0) != table.unsqueeze(1)).sum(-1)
    spread.fill_diagonal_(table.shape[1] + 1)
    return int(spread.min())


def makeRandomCode(count: int, width: int, generator: torch.Generator, tries: int = 64) -> list[list[int]]:
    """Returns the best-separated random codebook found in `tries` draws.

    Random rather than constructed, because for labels with no order or structure there is nothing
    for a construction to exploit — every bit is an arbitrary binary partition of the label set, and
    what matters is only that the codewords sit far apart. Drawing several and keeping the most
    separated is the cheap way to get that.

    A column that is constant across every label is redrawn: it asks a question whose answer never
    varies, so it consumes two motors and carries nothing.
    """
    if count < 2:
        raise LabelSpaceError("a codebook needs at least two labels to encode")

    best: list[list[int]] | None = None
    best_distance = -1
    for _ in range(max(tries, 1)):
        table = (torch.rand(count, width, generator=generator) < 0.5).to(torch.int16)
        column_sums = table.sum(0)
        if bool(((column_sums == 0) | (column_sums == count)).any()):
            continue
        codes = table.tolist()
        if len({tuple(row) for row in codes}) != count:
            continue
        distance = calcCodeDistance(codes)
        if distance > best_distance:
            best, best_distance = codes, distance

    if best is None:
        raise LabelSpaceError(
            f"no usable {width}-bit codebook was found for {count} labels in {tries} draws; "
            "the width is too small to give every label a distinct codeword"
        )
    return best


def makeCodebook(
    count: int, width: int | None, generator: torch.Generator, ordinal: bool
) -> list[list[int]]:
    """Returns a codebook suited to the label set: thermometer if ordinal and small, else random.

    Thermometer is preferred where it applies because every one of its bits is a threshold and so
    is individually learnable, but it costs a bit per label and stops paying for itself once the
    label set is large. See specs/decoding.md.
    """
    if ordinal and count <= 16:
        return makeThermometerCode(count)
    return makeRandomCode(count, width or calcCodeWidth(count), generator)


def toCodeBits(states: Tensor) -> Tensor:
    """Returns the bit each motor pair voted for, reading `states` as `[on, off, on, off, ...]`.

    Pairs rather than one motor against a threshold, because a threshold has nothing to push
    against. The two-motor race is what makes each bit the same two-way decision the learning rule
    already handles, and a fixed cutoff would instead reintroduce the calibration problem the
    signal band and edge gate took two rounds to settle.
    """
    paired = states.reshape(-1, 2)
    return (paired[:, 0] > paired[:, 1]).to(torch.int16)


def findNearestCode(bits: Tensor, codes: list[list[int]]) -> int:
    """Returns the index of the codeword closest to the bits that were read.

    Nearest rather than exact, which is the whole point of spending spare bits: a read that
    matches no codeword is resolved to the most plausible one instead of failing or answering
    arbitrarily.
    """
    if not codes:
        raise LabelSpaceError("cannot decode against an empty codebook")
    table = torch.tensor(codes, dtype=torch.int16)
    return int((table != bits.to(torch.int16).unsqueeze(0)).sum(-1).argmin())


def calcBitMargins(states: Tensor) -> Tensor:
    """Returns how decisively each pair voted, as the gap between its two motors."""
    paired = states.reshape(-1, 2)
    return (paired[:, 0] - paired[:, 1]).abs()
