# Haze Roadmap

What is not built yet, why it matters, and what it depends on. Built work is recorded in
[LOG.md](LOG.md); this file holds only what remains.

The goal this serves: Haze takes in sound, text, and images together — whatever context is
available — distils them for relevance, accumulates that context into an opinion, and produces
sound, text, or images back. Arrival *timing* should matter: an image reaching the sensors while a
sound is still being processed should not be equivalent to both arriving at once. A section of the
mesh that stops working should be replaced by new growth rather than dragging the whole network
down with it.

Five gaps stand between the current model and that goal. The first is the root of the second and
third and must be taken before either; the last two are independent.

---

## 1. Silence is absorbing — the root failure

**Status:** measured and understood, not fixed. This is the one that matters; most of what
follows is downstream of it.

**Training destroys the mesh's ability to conduct, one input at a time, and conduction never comes
back.** A fresh mesh answers every input. A trained one answers only the majority class. The loop:

1. An input whose answer is usually wrong accumulates negative advantage on the edges it fires.
2. Those edges fall under the edge gate, and that input stops reaching the motors.
3. A silent input produces **no fired trace** — so no eligibility, so no update — so those edges
   can never be strengthened again. The state is absorbing.
4. The caller scores the silence as reward 0, which deepens the collapse for everything else.

**Measured** — share of 200 held-out inputs still reaching the motors, `first-set` capped at `k`,
4 seeds, growth disabled:

| step | k=3 | k=9 |
|---|---|---|
| 0 | 1.00 | 1.00 |
| 250 | 1.00 | 0.99 |
| 500 | 0.95 | 0.74 |
| 1000 | 0.63 | 0.51 |
| 1500 | 0.65 | 0.52 |

Per-label at step 1500, the majority class is the survivor in every seed —
`1.0,1.0,0.3 | 1.0,0.5,0.5 | 1.0,0.0,0.0 | 1.0,0.0,0.0` at three labels, and
`1.0,0.0,0.0,0.0,0.0,0.0,0.0` in three of four seeds at nine.

**Those figures were measured with pruning off**, which overstates the severity: `applyPrune` is
called from the trainer's `flowRestructure`, and a harness driving `learn()` directly never invokes
it. Running the same sweep with pruning every 100 steps at the shipped threshold improves both
conduction and accuracy — reach 0.52→0.70 and accuracy 0.46→0.58 at nine labels, 0.65→0.75 and
0.59→0.66 at three. The erosion is still there (reach 1.00→0.70, not 1.00→1.00), so the diagnosis
holds, but pruning is a partial remedy that already exists and is easy to leave switched off.

**Pruning mute edges harder makes it worse.** Raising `prune_threshold` to the conduction floor so
a mute edge is pruned and rewired fresh — the obvious fix, since it recycles dead paths through
`ensureNoOrphans` with no new mechanism — measures worse than the shipped threshold everywhere and
worse than doing nothing at three labels (reach 0.44, accuracy 0.44). Recycling at that rate
destroys learned structure faster than it restores conduction.

**This re-reads every other measurement in this file.** The 0.51 plateau is not the mesh
*predicting* the majority class; it is the mesh only still *conducting* for the majority class and
scoring zero on the silence elsewhere — 0.51 is the majority share. "Converges by 1,500 steps" is
the erosion completing, not learning settling: nothing changes after step 1,000 because nothing is
left conducting to learn with. Supervision cannot help, however good the signal, because an input
that produces no trace has nothing to apply it to. The task-switch deaths in (2) are the same
mechanism with every pathway turning negative at once.

**Reverse learning, the built-in remedy, makes it worse.** Reach at step 1500 falls from 0.65 to
0.39 (k=3) and 0.52 to 0.38 (k=9) with `relearn_limit=3`. It is defaulted off for measuring as
harmful, which was the right call for the wrong reason — it is not a neutral extra, it is an
accelerant, and `calcRecoveryMask` does not do the job it exists to do.

**Ruled out as the cause:**

- **Not the eligibility trace.** It discriminates at every label count — same-label vs
  different-label cosine gap of +0.377 (k=2), +0.364 (k=3), +0.249 (k=9) after training, against
  +0.03 on a fresh mesh. Selectivity is learned and it works.
- **Not the teaching signal.** Replacing scalar reward with the correct answer at matched step size
  changes nothing at two labels and *hurts* above (2→0 seeds learning at six labels, 3→0 at nine).
  At two labels the two are provably the same signal: with two motors `calcMotorSeeds` puts the
  positive push on the correct motor whether it was chosen (gain positive) or not (gain negative).
  That identity is why two labels work at all.
- **Not the training budget.** Nothing moves between step 1,500 and step 15,000.
- **Not the representation.** The matched linear probe reads 0.81 off terminus activation on
  `majority` where the mesh achieves 0.65.

**The dead band.** Attenuation is `signal × prod(strengths) × geomean(strengths)`, so one hop costs
roughly the square of an edge strength: `signal × s²` for a single hop. Against the shipped
`signal_threshold` of 0.25 and a band topping out at 0.9, **an edge below 0.527 cannot conduct at
any signal**, and deeper paths need more (0.652 at two hops, 0.726 at three). `strength_lower` is
0.1 and `prune_threshold` is 0.2, so `[0.1, 0.527]` is a range in which an edge is alive,
unprunable, and mute. That is where eroded edges land: `ensureNoOrphans` cannot see them because
they are structurally connected, and pruning cannot remove them because they sit above its
threshold. The mean `|strength|` of a settled mesh measures 0.51 — on the single-hop floor.

`ensureHyperCoherent` validates `signal_lower × strength_init_upper²`, i.e. that the *strongest*
initial edge conducts at the *weakest* signal. Nothing checks that a *surviving* edge conducts.
Making the dead band unrepresentable is the durable half of any fix here, independent of which
dynamical remedy wins.

**Prevention beats recycling, modestly.** Clamping `strength_lower` to 0.55 — above the single-hop
conduction floor, so learning cannot push an edge mute — measured on `first-set` capped at `k`,
accuracy at step 1500. **Read the 8-seed column; the 4-seed one is kept only to show how far a
small sample of a bimodal outcome can mislead:**

| arm | k=3 (4 seeds) | k=3 (8 seeds) | k=9 (8 seeds) |
|---|---|---|---|
| shipped, no pruning | 0.59 | 0.63 | 0.48 |
| shipped, pruning on | 0.66 | — | 0.58 |
| prune at conduction floor 0.55 | 0.44 | — | 0.52 |
| **floor 0.55, no pruning** | 0.77 | **0.68** | 0.46 |

On eight seeds the floor gives 0.68 against 0.63, and **both arms have 3/8 seeds learning** — the
floor does not win more often, its wins are simply better (1.00, 1.00, 0.86 against 0.86, 0.79,
0.77). At nine labels it does nothing. It holds conduction longest of any arm (1.00 through step
500, 0.97 at step 1000) and that does not convert: **conduction is necessary and not sufficient.**

**A floor and a prune threshold cannot both be used.** `ensureHyperCoherent` requires
`prune_threshold > strength_lower`, so an edge clamped to a conduction floor sits permanently below
the prune threshold and is stripped on the next pass — measured, that configuration collapses to
0.40 accuracy and 0.42 reach. The two remedies are structurally exclusive as the config is shaped.

**The step-size control passes.** Narrowing strengths to `[0.55, 0.9]` also shrinks the dynamic
range, which acts as a smaller learning rate, so the shipped rails were re-run at the matched
`epsilon` of 0.044 and below. They reach 0.52, against the floor's 0.68 — the gain is not step size.

**Conduction and learning trade off under three separate mechanisms**, which is the more durable
finding. Lowering `epsilon` on the shipped rails takes reach 0.66 → 0.93 → 0.98 at 0.100 / 0.025 /
0.010 while accuracy falls 0.63 → 0.53 → 0.41. The inherited `reward - confidence` rule holds reach
at 1.00 and never learns. Both buy conduction and pay everything for it. The strength floor is the
only mechanism measured so far that improves both at once, which is why it stays interesting at an
effect size this small.

**Remaining candidates:** asymmetric rates so weakening is slower than strengthening; per-neuron
outgoing normalisation so weakening one edge strengthens its siblings rather than draining the
neuron; or treating silence as its own outcome rather than as reward 0, since scoring it as failure
is what makes the loop self-reinforcing.

**The two rules fail in opposite directions, and neither is closer to right.** The inherited signal
decomposes exactly as `r - c ≡ (r - r̄) + (r̄ - c)` — today's centred advantage plus a bias measuring
about +0.08. Run head to head on `first-set` capped at `k`, 6 seeds:

| | reach 0→1500 | accuracy 0→1500 | share on one label |
|---|---|---|---|
| centred, k=3 | 1.00 → 0.69 | 0.27 → 0.66 | 0.84 |
| inherited, k=3 | **1.00 → 1.00** | **0.27 → 0.27** | **1.00** |
| centred, k=9 | 1.00 → 0.62 | 0.11 → 0.45 | 0.92 |
| inherited, k=9 | **1.00 → 1.00** | 0.11 → 0.21 | 0.96 |

The inherited rule never loses an input and never learns: by step 250 it answers one label for
every input and never moves again (at three labels that is 0.27, *below* the 0.51 a majority guess
scores, because it locks onto a minority label). The centred rule learns and pays in conduction.
**The bias was the whole conduction-preservation mechanism**, and no constant can serve both roles —
one large enough to hold conduction open is large enough to freeze the policy. Selection belongs to
the reward signal; staying conductive is a structural property and belongs elsewhere.

**Related, and worth keeping.** A global baseline makes every deterministic policy a zero-drift
fixed point: at reward rate `p` the chosen motor gets `+(1-p)` on a win and `-p` on a loss, for an
expected push of exactly zero. The rule this replaced used `reward - confidence`, whose
four-quadrant behaviour (confidently wrong corrects hard, luckily right reinforces hard) is the
right shape for a critic, and was dropped for a persistent positive mean (+0.077 to +0.104).
Measured since: confidence's correlation with reward **changes sign with label count** — +0.217 at
three labels with a cleanly monotonic calibration curve, -0.08 at two, and inverted at nine, where
the top confidence bucket scores 0.28 against 0.54 at the bottom. A uniform bias is fixable by
calibration; a sign flip is not. But a bucketed state-dependent baseline still took two labels from
5/8 to 7/8 seeds learning (0.75 → 0.84), so the state-dependence was worth keeping and the global
scalar gave it up for nothing. A critic predicting reward from terminus activation rather than from
a one-number summary is the version worth building — after (1), since a baseline changes how hard
an update pushes and never which edges survive.

---

## 2. Adaptation — growth that answers a change

**Status:** growth is built and wired; it does not do this, and during stable learning it hurts.

A section that works and then stops working should be replaced by new growth. Measured — train
`copy(bit 0)` for 1,500 steps, switch the target to `copy(bit 3)`, 6 seeds:

| mode | pre-switch | recovered | grown |
|---|---|---|---|
| growth off | **0.99** | **0.25** | 0 |
| growth on, absolute (shipped) | 0.87 | 0.48 | 162 |
| growth on, relative (prototype) | 0.71 | 0.59 | 213 |

**Without growth the mesh does not adapt — it dies.** Three of six seeds end raising
`SignalDidNotReachMotorsError`, and 0.25 is far below the 0.51 a constant guess would score. This
is gap (1) at its sharpest: a task switch turns *every* pathway negative at once, so the erosion
that normally takes the minority classes takes the whole mesh. Growth's one measured benefit is
that no seed dies in either growth arm — it is currently working as life support against (1)
rather than as adaptation, which is why it looks useful here and harmful during stable learning.

Four structural gaps, none of them tuning:

- **No change-detection.** [auditor.py](src/modules/auditor.py) holds one window and compares its
  error rate to an absolute threshold. Nothing compares now to before, so "used to work, now
  doesn't" is not a state the code can represent. The shipped trigger fires during early training
  (where it interferes — pre-switch 0.99→0.87) and once the mesh is already dying (where it acts
  as life support: no seed dies in either growth arm). It cannot fire at the moment behaviour goes
  stale.
- **No localisation.** `applyGrowth` enlarges the whole nexus and terminus population. Nothing
  represents *which* section failed, so new capacity cannot be aimed. This is the same missing
  quantity as gap 1 — knowing which part is at fault is what credit assignment is failing to
  produce.
- **No protection of what worked.** Nothing shields the established section while the mesh
  re-learns, so correct growth would still be building beside a corpse.
- **`growth_threshold` is an absolute error rate**, so it is task-dependent: 0.6 is above chance
  for a two-label task and below it for a uniform nine-label one. A trigger relative to what the
  mesh has recently achieved would not need retuning per task — the prototype above is that, and
  recovers best (0.59, two seeds at 0.93 and 0.79) at the cost of firing on ordinary early-training
  noise.

---

## 3. Large output spaces — the label cliff

**Status:** attempted and unresolved; now understood as **downstream of gap 1** rather than as its
own problem. `CodeBook` is built, tested, and does not work. See [specs/decoding.md](specs/decoding.md).

**The change attempted:** a label stops owning a motor. Motors are allocated in pairs, one pair per
bit of a code, each pair a two-way race; a label becomes a codeword. `k` labels cost `O(log k)`
motors instead of `k`.

**What happened:** it loses to one-hot at every `k` above two, and the reason is structural rather
than incidental. A codeword is a conjunction, so the barrier appears twice:

- **In decoding** — joint accuracy is the *product* of per-bit accuracies. Nine labels over eight
  bits needs every bit right 0.92 of the time before the whole answer beats a constant guess.
- **In learning** — with reward 1 only for a whole correct codeword, the baseline is `q^m` and the
  push separating a pair's correct motor from its wrong one is `4·q^m·(1-q)`, which **peaks at
  `q = m/(m+1)`** — 0.89 for eight bits — and falls away exponentially below it. At initialisation
  (`q = 0.5`) that is 0.008 against one-hot's 0.125 at nine labels: **16× weaker**. The codebook
  only learns quickly once it has nearly solved the problem.

One-hot's decode is a single argmax — one decision, no conjunction — so it degrades only
polynomially (`~2/k`). **Any factored code degrades exponentially in both places.** One-hot is
therefore the right output structure, and the fix belongs in credit assignment, not in output
encoding.

Ruled out, so it is not re-tried blind: three exploration strategies at two rates, two code
families, and per-bit supervision (the most a teacher could give). `CodeBook` ships available and
non-default so the two stay comparable when there is a better rule to re-test against.

**Still untried:** a code with real Hamming distance *and* individually easy bits. Thermometer is
distance 1 at every size (adjacent labels differ in one bit — that is what makes it ordinal), so
nearest-codeword decoding degenerates to exact match and corrects nothing. Random codes buy
distance 2-3 by asking harder questions, which raises the per-bit error rate; correcting one error
in eight is worth +0.38 at a 0.10 error rate and +0.09 at 0.40, and we are on the wrong side of
that curve. Nothing yet gives both. Distance also assumes errors are independent and in the
minority, and ours are neither — every bit reads the same terminus activation through the same
rule, and a bit whose partition the mesh cannot represent is wrong systematically rather than
occasionally.

---

## 4. Residual mesh state — making time matter

**Status:** not started. Depends on nothing, but is far easier to evaluate after (1).

`flowSignalPass` allocates fresh state on every call and runs to quiescence, so **every
observation starts from silence and arrival order is currently unobservable**. An image arriving
mid-sound is, today, exactly the same as the two arriving together.

**The change:** node activation persists between observations and decays, so a mesh still ringing
from one input responds differently to the next. That makes the propagation contract stateful,
which touches reproducibility (a checkpoint must carry the residue), termination (the wavefront
guard assumes a fresh `fired` mask per pass), and learning (the fired trace would span
observations).

**Open question:** what decays, how fast, and whether the decay is a hyperparameter or a per-neuron
learned quantity. A learned time constant is the more interesting answer and the harder one.

---

## 5. Context accumulation — the opinion

**Status:** not started. Partially mistaken for built, so stated precisely.

`observe()` begins with `self._observations = {}`. It **replaces** rather than accumulates: two
successive calls do not build on each other, the second discards the first. Context does accumulate
across *modalities within a single call* — `observe({"sound": …, "image": …})` — and that multi-key
path is the seed of the mechanism, but accumulation across calls does not exist.

**The change:** an observation joins a running context rather than replacing it, with an explicit
way to close or reset it. This is what turns a sequence of inputs into a position rather than a
series of independent reactions.

**Interacts with (4):** residual mesh state and accumulated observations are two different memories
— one in the mesh, one in the model — and the design should say which holds what rather than
letting them overlap by accident.

---

## Supporting work

### Multi-modal encoders
Real `HashTextEncoder`, `PatchImageEncoder`, and `SeriesEncoder`, plus the decoders that emit each
modality back. Blocked behind (1) and (3): every one of them implies a large label space.

### The majority readout gap
0.65 achieved against a probe bound of 0.81, and unlike copy the representation does not move
during training. Not a separate item any more — it is gap (1) measured on one task.

### Orphan rescue does not cover this
`ensureNoOrphans` rescues a neuron with no edges in or out. It cannot see the failure in (1),
where every edge is individually alive and structurally connected, and the path is collectively
too weak to carry signal past the gate. The invariant worth adding is about *conduction*, not
about *connection*: some share of the input space must still reach the motors, and the mesh
should be able to assert it.

### Honest benchmark
The speedup figure must be re-measured with the prior engine's per-edge JSON writes and its
`print` of the strength array disabled, so the headline is not "we stopped writing 40,000 files per
row". Owed before any number reaches the README.

### Packaging and publication
Delete `app/`, rewrite the README (it still documents a deleted layer), write the model card, and
push to the Hub. The packaging itself is built and round-trips bit-identically; this is the
presentation around it.

### Corrections owed in code
The `calcCodeSeeds` docstring claims the shared gain "is noise rather than bias". The sign is
right; the magnitude is not — the useful contrast is `4·q^m·(1-q)`, which vanishes exponentially.
That claim was asserted without measurement and needs rewriting.
