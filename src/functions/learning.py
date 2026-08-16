"""Forward-only learning: moving the edges that carried signal toward the reward."""

import math

import torch
from torch import Tensor

from ..errors import InvalidRewardError
from ..models import HazeHyper, LearnResult


def ensureRewardFinite(reward: float, confidence: float) -> None:
    """Raises if a reward or confidence is not finite.

    Without this the clamp propagates NaN into every strength in the active mask, destroying the
    network with no error and no symptom until its answers become nonsense.
    """
    for name, value in (("reward", reward), ("confidence", confidence)):
        if not math.isfinite(float(value)):
            raise InvalidRewardError(
                f"{name} was {value!r}; a non-finite value would poison every strength in the "
                "learning mask and leave no trace of where it came from"
            )


def calcConfidenceEntropy(states: Tensor, epsilon: float = 1e-9) -> float:
    """Returns how peaked an activation is, as one minus its normalized entropy.

    Defined at both degenerate cases: a single label has zero maximum entropy, and an activation
    summing to zero has no distribution at all. Both arise in ordinary use.

    Computed over magnitudes, since an inhibited motor carries as much evidence as an excited
    one and a signed value is not a probability. On non-negative activation this is exactly the
    previous behaviour; on signed activation the previous form reported total certainty for any
    set with a positive sum and total uncertainty for any set without one, which is not a
    measurement of anything and fed straight into the growth trigger.
    """
    signals = states.to(torch.float64).flatten().abs()
    if signals.numel() == 0:
        return 0.0
    if signals.numel() == 1:
        return 1.0 if float(signals[0]) > 0 else 0.0

    total = float(signals.sum())
    if total <= 0:
        return 0.0

    probs = signals / total
    entropy = float(-(probs * (probs + epsilon).log()).sum())
    max_entropy = math.log(signals.numel())
    if max_entropy <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


def calcRecoveryMask(mesh, node_activation: Tensor, trace: Tensor) -> Tensor:
    """Returns the unfired edges leaving a neuron the signal actually reached.

    The wavefront's own boundary, rather than every edge that did not fire. A mesh that failed to
    reach its motors stopped somewhere specific, and the only edges that can extend it are the
    ones leaving neurons the signal got to; strengthening an edge whose source was never active
    cannot improve reachability and does damage learned structure.

    That distinction is not theoretical. Pushing every unfired edge up — which is what "strengthen
    the edges that did not carry signal" reads as — moves on the order of a hundred and seventy
    edges at once, and firing it a handful of times across a run took held-out accuracy on copy
    from 0.95 to 0.74. See specs/learning.md.
    """
    live = mesh.counts.edges
    if live == 0:
        return torch.zeros(0, dtype=torch.bool)

    fired = trace[:live] if trace.numel() >= live else torch.zeros(live, dtype=torch.bool)
    reached = node_activation.abs() > 0
    frontier = (~fired) & reached[mesh.src[:live]] & mesh.alive_e[:live]
    if not bool(frontier.any()):
        return (~fired) & mesh.alive_e[:live]
    return frontier


def applyLearning(
    mesh,
    trace: Tensor,
    reward: float,
    confidence: float,
    hyper: HazeHyper,
    reverse: bool = False,
    mask: Tensor | None = None,
) -> LearnResult:
    """Moves every edge in the trace by epsilon times the gap between reward and confidence.

    A confident wrong answer is punished hardest and a confident right answer barely moves, so
    the mesh stops adjusting what it already reliably knows. Selection is by `torch.where` over
    the full edge tensors rather than boolean indexing, which allocates and forces a device
    synchronization; see specs/learning.md.

    A reverse pass applies a definite positive push instead, because on a reverse pass there was
    no answer and so nothing a reward could be compared against. Deriving its magnitude from
    `reward - confidence` made it exactly zero in the only situation it exists for: no signal
    reached the motors, so confidence was 0 and the caller had no outcome to report but 0, and
    the update that was supposed to reopen a path silently did nothing at all.
    """
    ensureRewardFinite(reward, confidence)

    live = mesh.counts.edges
    if live == 0:
        return LearnResult(reward=reward, confidence=confidence, reverse=reverse)

    span = slice(0, live)
    fired = trace[:live] if trace.numel() >= live else torch.zeros(live, dtype=torch.bool)
    if mask is None:
        mask = (~fired if reverse else fired) & mesh.alive_e[span]
    else:
        mask = mask[:live] & mesh.alive_e[span]
    updated = int(mask.sum())
    if updated == 0:
        return LearnResult(reward=reward, confidence=confidence, reverse=reverse)

    signed = 1.0 if reverse else float(reward) - float(confidence)
    delta = mesh.epsilon[span] * signed
    zero = torch.zeros_like(delta)

    moved = torch.where(mask, delta, zero)
    mesh.strength[span] = torch.where(
        mask,
        (mesh.strength[span] + moved).clamp(hyper.calcStrengthFloor(), hyper.strength_upper),
        mesh.strength[span],
    )
    mesh.epsilon[span] = torch.where(
        mask,
        (mesh.epsilon[span] * calcPlasticityFactors(moved, hyper)).clamp(
            hyper.epsilon_floor, hyper.epsilon_start
        ),
        mesh.epsilon[span],
    )
    mesh.log_str[span] = mesh.strength[span].abs().clamp_min(1e-6).log()

    return LearnResult(
        edges_updated=updated,
        reward=float(reward),
        confidence=float(confidence),
        reverse=reverse,
        mean_delta=float(moved.sum() / max(updated, 1)),
    )


def calcRewardAdvantage(mesh, reward: float, hyper: HazeHyper) -> float:
    """Returns how much a reward beat what the mesh had come to expect, and records the reward.

    This is the factor that decides the sign of every update, so what it is measured against
    decides what the mesh learns. Measured against `confidence` — the inherited signal — it is
    not centred on anything: a correct answer contributes `1 - c` and a wrong one only `-c`, so
    at even odds and typical confidence the chosen answer's path is reinforced by a net positive
    amount *whether or not it was right*. Measured, that ran to a fixed point where one motor won
    on 96-100% of observations and the mesh answered the same label forever.

    Centring on the mesh's own recent reward removes exactly that term. The expectation of the
    advantage is zero by construction, so a uniform push cancels and only the part of an edge's
    activity that covaries with the outcome accumulates. See specs/learning.md.

    The baseline is seeded from the first reward rather than from zero or from an assumed 0.5.
    A single sample is its own expectation, so the first advantage is exactly zero and nothing is
    learned from an outcome there was no expectation to compare against. Seeding at zero would
    instead spend the whole warm-up applying the uniform push this function exists to remove, and
    seeding at 0.5 would assume a reward scale the caller never agreed to.

    The reward is checked here as well as at the update, because the baseline is persistent state:
    a non-finite reward that reached it would survive into the checkpoint and poison every later
    advantage, long after the observation that caused it.
    """
    ensureRewardFinite(reward, 0.0)
    if not hyper.reward_baseline:
        return float(reward)

    seen = int(mesh.learn_count[0])
    baseline = float(reward) if seen == 0 else float(mesh.reward_bar[0])
    mesh.reward_bar[0] = baseline + hyper.reward_baseline_rate * (float(reward) - baseline)
    mesh.learn_count[0] = seen + 1
    return float(reward) - baseline


def switchMotorChoice(states: Tensor, rate: float, generator: torch.Generator) -> Tensor:
    """Returns motor activation with a random motor swapped into the lead, at the explore rate.

    A centred advantage learns from the difference between what happened and what was expected,
    which means it learns nothing at all when nothing varies. A greedy readout on a mesh that is
    uniformly wrong produces exactly that: reward is constant, the baseline meets it, the
    advantage is zero, and the mesh stays wrong forever. Measured on the constant task, one seed
    in three settled at 0.00 and never moved. Reward variance is not a nuisance here, it is the
    only thing there is to learn from, and a deterministic readout produces none.

    A swap rather than a boost, because the multiset of activations is preserved: confidence,
    entropy, and every magnitude-sensitive decoder behave exactly as they would have, and the one
    thing that changes is which label holds the peak. Promoting by an added margin would instead
    make the mesh look more certain precisely when it is guessing.
    """
    if rate <= 0.0 or states.numel() < 2:
        return states
    if float(torch.rand(1, generator=generator)) >= rate:
        return states

    pick = int(torch.randint(0, states.numel(), (1,), generator=generator))
    lead = int(torch.argmax(states))
    if pick == lead:
        return states

    switched = states.clone()
    switched[lead], switched[pick] = states[pick], states[lead]
    return switched


def calcNeuronCredit(
    mesh, trace: Tensor, seeds: dict[int, float], hops: int, strength: Tensor | None = None
) -> Tensor:
    """Spreads a per-motor teaching signal backward along the edges that carried signal.

    This is not a gradient and there is no chain rule: a scalar per motor is pushed back over the
    fired edges, attenuated by the same strengths the forward pass used, so a neuron's credit is
    how much it fed the motors that were rewarded. It costs one extra scatter per hop over the
    same edge list the forward pass walks.

    "The same strengths the forward pass used" is a real invariant and not a turn of phrase. Under
    `signal_economy` the forward pass carries shares rather than raw strengths, so the caller must
    pass those shares here; leaving this reading `mesh.strength` sends credit backward along
    weights the signal never travelled, and a neuron's credit stops meaning how much it fed the
    motors at all.
    """
    neurons = int(mesh.kind.numel())
    credit = torch.zeros(neurons, dtype=mesh.dtype)
    if not seeds:
        return credit
    for motor, value in seeds.items():
        credit[motor] = value

    live = mesh.counts.edges
    if live == 0:
        return credit

    src, dst = mesh.src[:live], mesh.dst[:live]
    strength = mesh.strength[:live] if strength is None else strength[:live]
    fired = trace[:live]
    zero = torch.zeros(live, dtype=mesh.dtype)

    total = credit.clone()
    frontier = credit
    for _ in range(max(hops, 1)):
        carried = torch.where(fired, frontier[dst] * strength, zero)
        if not bool((carried != 0).any()):
            break
        frontier = torch.zeros(neurons, dtype=mesh.dtype).index_add_(0, src, carried)
        total = total + frontier
    return total


def calcMotorSeeds(motors: Tensor, chosen: int, gain: float) -> dict[int, float]:
    """Returns the teaching signal for each competing motor, given which one the decoder chose.

    Learning is told a reward, never the correct label, so credit is seeded at the motor that was
    actually chosen and signed by how the reward turned out. A rewarded choice reinforces the path
    that produced it and weakens its rivals; an unrewarded one does the reverse, which is what lets
    a rival overtake it. Rivals share the opposite signal so the seeds sum to zero and the mesh's
    total strength is not driven in one direction.
    """
    ids = motors.tolist()
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: gain}
    share = gain / (len(ids) - 1)
    return {motor: (gain if motor == chosen else -share) for motor in ids}


def calcCodeSeeds(bit_motors: list[int], bits: Tensor, gain: float) -> dict[int, float]:
    """Returns the teaching signal for each motor of a coded decoder, given the bits it voted.

    Every pair is scored on its own, and inside a pair the seeds are exactly the two-label rule:
    the motor that won gets the gain and the one that lost gets its negative. Nothing is divided
    among rivals, because within a pair there is only one rival and "not the one I chose" names
    it precisely — which is the whole reason a codebook restores the signal a large label set
    destroys. See specs/decoding.md.

    The pairs that were wrong and the pairs that were right receive the same signed gain, since a
    scalar reward cannot say which bits were at fault. The direction that leaves is correct — a bit
    whose vote tracks the outcome accumulates the right way — but its **magnitude collapses**, which
    an earlier version of this note asserted was fine without measuring it. With reward paid only
    for a whole correct codeword the baseline is `q**m`, and the push separating a pair's correct
    motor from its wrong one is `4 * q**m * (1 - q)`. That peaks at `q = m/(m+1)` — 0.89 for eight
    bits — and falls away exponentially below it: at initialisation it is 0.008 against a one-hot
    decoder's 0.125 at nine labels. The codebook only learns quickly once it has nearly solved the
    problem, which is why it loses to one-hot at every label count above two. See specs/decoding.md.
    """
    seeds: dict[int, float] = {}
    for position, bit in enumerate(bits.tolist()):
        on, off = bit_motors[2 * position], bit_motors[2 * position + 1]
        winner, loser = (on, off) if bit else (off, on)
        seeds[winner] = seeds.get(winner, 0.0) + gain
        seeds[loser] = seeds.get(loser, 0.0) - gain
    return seeds


def switchCodeChoice(
    states: Tensor, codes: list[list[int]], rate: float, generator: torch.Generator
) -> Tensor:
    """Returns the pair activations rearranged to spell a random label, at the explore rate.

    Explores in label space rather than bit space, which is not the obvious choice and is the one
    that works. Flipping a single random bit reads as the natural local move, but a code with
    spare distance is built precisely so that one wrong bit still decodes to the same label — so
    at a Hamming distance of 4, single-bit exploration cannot change the answer at all. Measured,
    that reinstated the exact dead zone exploration exists to prevent: the mesh locked onto one
    label, could never try another, and scored 0.27 against 0.91 on a task it should have found
    easy. Redundancy defeats bit-level exploration in proportion to how much of it there is.

    Landing on a whole codeword also keeps the credit honest, since every pair is then scored on a
    label the mesh could actually have answered. Each pair is swapped rather than overwritten, so
    the activation multiset survives and confidence is not invented. See specs/decoding.md.
    """
    if rate <= 0.0 or states.numel() < 2 or len(codes) < 2:
        return states
    if float(torch.rand(1, generator=generator)) >= rate:
        return states

    target = codes[int(torch.randint(0, len(codes), (1,), generator=generator))]
    switched = states.clone()
    for position, bit in enumerate(target):
        on, off = 2 * position, 2 * position + 1
        if (float(states[on]) > float(states[off])) != bool(bit):
            switched[on], switched[off] = states[off], states[on]
    return switched


def calcPlasticityFactors(moved: Tensor, hyper: HazeHyper) -> Tensor:
    """Returns the multiplier each fired edge's learning rate takes, read from its own last move.

    Cooling an edge that was just reinforced and warming one that was just weakened, rather than
    the unconditional decay this replaces, which cooled an edge that had been consistently right
    and one that had been consistently wrong at the same rate and never let either warm again.

    Read per edge rather than from the observation's advantage, which is the version that does not
    work: a mesh performing steadily — well or badly — has an advantage centred on zero once its
    baseline catches up, so a global signal crystallizes nothing at exactly the point a section has
    earned it. An edge's own move does not have that problem. An edge repeatedly part of answers
    that are kept is repeatedly reinforced and hardens; one repeatedly part of answers that are
    corrected is repeatedly weakened and softens; and two clusters in the same mesh harden
    independently, which a single scalar could never express.

    Multiplicative in both directions, which is what produces the hysteresis: undoing a long record
    takes about as many reversals as the record took to build, so one bad answer cannot flatten a
    section that spent hundreds of steps earning its place. An edge that did not move is evidence
    for neither and holds — which also makes a non-finite move safe, since it compares false both
    ways rather than poisoning the rate.
    """
    if not hyper.crystallize:
        return torch.full_like(moved, hyper.epsilon_decay)
    return torch.where(
        moved > 0,
        torch.full_like(moved, hyper.epsilon_cool),
        torch.where(
            moved < 0, torch.full_like(moved, hyper.epsilon_warm), torch.ones_like(moved)
        ),
    )


def applyCreditedLearning(
    mesh,
    trace: Tensor,
    eligibility: Tensor,
    credit: Tensor,
    reward: float,
    confidence: float,
    hyper: HazeHyper,
) -> LearnResult:
    """Moves each fired edge by its own eligibility times the credit reaching its destination.

    Three factors, none of them a gradient: how much the edge carried, how much its destination
    was worth crediting, and how surprising the outcome was. The uniform rule this replaces moved
    every fired edge by one scalar, which cannot make one motor beat another because both of their
    inbound edges are in the trace with the same sign. See specs/learning.md.
    """
    ensureRewardFinite(reward, confidence)

    live = mesh.counts.edges
    if live == 0:
        return LearnResult(reward=reward, confidence=confidence)

    span = slice(0, live)
    fired = trace[:live]
    mask = fired & mesh.alive_e[span]
    updated = int(mask.sum())
    if updated == 0:
        return LearnResult(reward=reward, confidence=confidence)

    weight = eligibility[:live]
    scale = weight.abs().max().clamp_min(1e-12)
    signal = (weight / scale) * credit[mesh.dst[span]]

    zero = torch.zeros(live, dtype=mesh.dtype)
    moved = torch.where(mask, mesh.epsilon[span] * signal, zero)

    mesh.strength[span] = torch.where(
        mask,
        (mesh.strength[span] + moved).clamp(hyper.calcStrengthFloor(), hyper.strength_upper),
        mesh.strength[span],
    )
    mesh.epsilon[span] = torch.where(
        mask,
        (mesh.epsilon[span] * calcPlasticityFactors(moved, hyper)).clamp(
            hyper.epsilon_floor, hyper.epsilon_start
        ),
        mesh.epsilon[span],
    )
    mesh.log_str[span] = mesh.strength[span].abs().clamp_min(1e-6).log()

    return LearnResult(
        edges_updated=updated,
        reward=float(reward),
        confidence=float(confidence),
        mean_delta=float(moved.sum() / max(updated, 1)),
    )
