"""Codebooks: expressing one choice among many as several binary ones."""

import torch
from torch import Tensor

from ..errors import LabelSpaceError


def calcCodeWidth(count: int, redundancy: int = 2) -> int:
    """Returns how many bits to spend on `count` labels, including room for error correction.

    More than the `log2(count)` a bare enumeration needs. With the minimum, every bit pattern is a
    valid label, so one wrong bit silently yields a different answer and nothing can detect it.
    The spare bits are what let a codeword be decoded to its nearest neighbour instead — which
    matters here beyond ordinary noise, because some bits will be functions the mesh cannot
    represent at all, and redundancy is what turns those from fatal into merely wasteful.
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
