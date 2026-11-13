# ST-2 Look-alike tail — pre-registration

**Status: REVISED DRAFT, not registered.** Frozen when tagged
`prereg-stress-st2-<date>`. This revision settles what building the runner left open:
the backend list now that P0a is `modeswitch_guard`, how parents and look-alikes are
drawn, the G arm's extent, the populations each prediction is scored on, and the
*Scoring* rules. Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.
Build notes and timing: `STRESS_ST2_BUILD.md`.

## Question

Real long tails are full of objects that resemble common ones: the rare subspecies next
to the common species. The §3.11 clean run had 2 novelty misses, and both were novel
objects whose answers happened to match a known object closely. How close can a novel
object sit to a known one before the catch-all stops seeing it? And when the answerer
can confuse the two names, what does that do to the backends?

## Decisions taken in this revision

**1. Backends.** P0a became `modeswitch_guard` on 2026-11-12
(`STRESS_REINSTATE_GUARD.md`, *Decision*). The grid runs `soft`, `hard`, `modeswitch`
(unguarded, K = 3, as drafted and as in the retraction study) **and**
`modeswitch_guard` (K = 3). Every prediction that names `modeswitch` is scored
separately for both. The guard changes only games that would reinstate on the final
answer, so the two should mostly agree; where they differ, the guard's verdict is the
one that speaks for P0a.

**2. Parents.** 100 distinct parents are drawn *without* replacement from the 158
real objects, one at a time. Each draw is weighted by the remaining objects' Zipf
weights in the stream the games are played on (the rank map of `make_stream` over
258 objects, seed 7; it depends only on the object count and the seed). So common
objects are likelier to have a look-alike, and no object has two. A draw with
replacement was considered and rejected before tagging. It gives only 37–42
distinct parents per set, and at d = 1, 12–26 of the 100 look-alikes would be exact
twins of a sibling (`STRESS_ST2_BUILD.md`, census). The 258 objects' ranks are the
stream's random rank map, so a look-alike's own frequency is independent of its
parent's.

**3. Nested look-alike sets.** For each look-alike seed, the 100 parents and, per
look-alike, a seeded random order of the parent's flippable cells (cells not at
exactly 0.5) are drawn once. The look-alike at distance d flips the first d cells of
that order. Each look-alike is still a uniform d-subset flip, the distribution
`lookalike(parent, d)` draws from, but the sets at different d share parents and are
nested. The d axis is then compared on common random numbers.

**4. Catch-all prior.** The PY (σ, θ) is fitted on the stream's own first-appearance
flags, as §3.11 did. The stream of object indices is the same in every job, so the fit
is too.

**5. G arm extent.** The G arm runs at d ∈ {1, 2} only. At d > 2 no look-alike is
confusable and the cell would be the exact cell replayed. The noise seed drives the
confusion draws. Confusion applies in every game whose secret is a look-alike, before
and after its admission.

**6. No `no_repeat` arm.** This is the published architecture in accumulation mode.
On this stream the PY prior (σ = 0.300, θ = 20) is above the repeat-loop threshold
of ≈ 0.15 for the first 274 of 2500 games; the §3.11 stream was above it for its
first 187 of 2000 (`STRESS_ST0.md`, Finding 2). Repeat questions are logged per game
and reported per cell, and P1's `wrong` games can be split by whether they repeated,
so a loop would be visible.

**7. Planner path.** Games use the Python planner, the path that produced §3.11. Its
float sums are in a fixed order, so given the interpreter and `PYTHONHASHSEED` a game
is reproducible across machines. (ST-1's numpy path needed a rounding tie rule
because BLAS summation order can vary; nothing here uses BLAS for ranking.) A
2500-game job takes 0.5–2 min (`STRESS_ST2_BUILD.md`), so no speed path is needed.

**8. Populations.** A parent can only be guessed by name once it has been admitted.
Early in the stream many parents are not yet known, and a look-alike game then looks
like ordinary novelty. The mechanism in P1's cost claim and in P4 needs a known
parent, so those parts are scored on look-alike games whose parent was admitted
before the game (*parent known*). The unconditional figures are reported beside them.

**9. P5's control.** `H_known` grows through the stream, so later games of any object
take more questions, and the first ~270 games are played at a high catch-all prior.
Comparing a parent's games before and after its look-alike's admission picks that
drift up. P5 therefore adjusts each parent game by the mean questions of control
games within 100 games of it. Control games are in-pool games of real objects that
are no look-alike's parent. And P5 uses only games from game 274 on, after which the
prior stays below the repeat-loop threshold. Earlier, in-pool games of the discrete
backends run to 20 questions in the loop, and a parent's few "before" games would
mostly be loop games. The raw before/after difference is reported beside it.

## Setup

Accumulation mode on the real matrix. The 158 objects are joined by 100 look-alikes,
each made by `lookalike(parent, d)` (ST-0a): flip exactly *d* of the parent's 44 cells
across 0.5. The look-alike keeps the parent's category. Parents are drawn with the
parent's Zipf weight, so common objects are likelier to have a look-alike, as with
subspecies (Decision 2: distinct parents). The 258 objects are streamed by the existing `make_stream` (Zipf 1.1, random
rank, seed 7), 2500 games. An admitted look-alike joins `H_known`, so later games of its
parent face a closer neighbour.

| axis | levels |
|---|---|
| distance d | 1, 2, 3, 4, 6, 8, 12 |
| attribute-flip rate | 0, 0.05, 0.10 |
| backend | `soft`, `hard`, `modeswitch` (K = 3) |
| guess channel (G arm) | exact; near-name false-yes g = 0.05, 0.20 |
| seeds | 3 look-alike seeds × 3 noise seeds |

The backend row is `soft`, `hard`, `modeswitch` and `modeswitch_guard` (Decision 1).
Budget 20, confidence 0.90, ε = 0.05. At flip rate 0 there is no noise seed. Grid:
588 exact-arm jobs and 144 G-arm jobs, 732 in all.

**G arm.** When the planner guesses a known by name and the secret is that known's
look-alike with d ≤ 2, the answerer says "yes" with probability g. The answerer
is treating the subspecies as the species. All other guesses stay exact. Scored with
ST-0c, so a false "yes" is a `false_confirm`, not a win. The G arm runs at flip rate 0
only, to isolate the guess channel, and at d ∈ {1, 2} (Decision 5).

**Interpreter and hash seed.** Python 3.9 (`LLM_RL`) and `PYTHONHASHSEED=0`. The
runner refuses otherwise, and records both in the summary along with the git
commit and the registration tag.

### Built before tagging

None of these change the design above. Status is in `STRESS_ST2_BUILD.md`.

* **The runner** (`stress_st2.py`): `--plan`, `--self-test` and `--smoke` (synthetic
  matrix only), and `--run` (registration-checked, resumable per job). Each job stores
  per-game columns, including the by-name guesses' relation to the secret's parent.
* **The analyzer** (`analyze_st2.py`): implements *Scoring* below. Its figure captions
  name it and the result directory.

## Metrics

* Separately for look-alike novelty and for ordinary novelty: detector AUC, TPR/FPR at
  the gate, median fire turn.
* Look-alike games: outcome split (`expert` = correct, `wrong`, `false_confirm`), and
  whether the parent was guessed by name.
* Parent games after the look-alike's admission: questions per game against the same
  parent's games before admission.

## Predictions

* **P1 — under exact guesses, look-alikes are cheap, not dangerous, for the discrete
  backends.** At flip rate 0, `hard` and `modeswitch` fire on ≥ 0.95 of look-alike
  first appearances at every d, with `wrong` = 0. The cost at d ≤ 2 is about one extra
  question: the parent survives, is guessed by name, gets "no", is struck, the live set
  empties, and the expert fires.
* **P2 — the detector score separates look-alikes worse as d shrinks.** For every
  backend, look-alike AUC is non-decreasing in d (allowing seed noise) and is lowest at
  d = 1. Median fire turn falls as d grows.
* **P3 — under noise, `modeswitch` spends its probes on the parent.** At d ≤ 2 the
  parent is usually struck by exactly one answer: the differing cell, or its own failed
  by-name guess if that cell was never asked. That is the profile of a flip victim, so
  `modeswitch` reinstates and re-tests it. On look-alike games its median
  fire turn exceeds `hard`'s by at least 1 and at most K = 3. Its `wrong` stays 0.
* **P4 — name confusion reverses the ranking (G arm, headline).** With g > 0, the
  `false_confirm` rate on look-alike games is about g × P(parent guessed by name). For
  `hard` and `modeswitch`, P(parent guessed) ≥ 0.8 at d ≤ 2, because they always test
  the sole survivor by name. So their `false_confirm` rate is ≥ `soft`'s at every
  g > 0. The exact-test step that makes P0a safe is the exposure once names can be
  confused.
* **P5 — admitted look-alikes tax their parents.** After admission, the parent's games
  take more questions: +0.5 to +1.5 per game at d ≤ 2, and no measurable change at
  d ≥ 6.

## Scoring

`analyze_st2.py` implements exactly these rules.

**Cells and populations.**

* A **cell** is (arm, g, d, flip rate, backend). It pools every game of its 3
  look-alike seeds and, under noise or in the G arm, its 3 noise seeds.
* *Look-alike games* have a look-alike secret. *Novel games* are games whose secret
  was not in `H_known` when the game started. *First appearances* are an object's first
  game in the stream. *Parent known*: the look-alike's parent was admitted before
  the game.
* A game **fires** when it ends with an expert call. The **fire turn** is the turn
  the catch-all first crossed the gate. A **median fire turn** is over the population's
  games that crossed it.
* **Look-alike AUC** is the detector AUC of `p_unknown_max`, look-alike novel games
  against all in-pool games of the cell. Ordinary-novelty AUC uses the real objects'
  novel games instead.

**Verdicts.** Each prediction is PASS or FAIL, and it passes only if every listed
part holds. A NaN, such as an empty population, counts as not holding. "Discrete
backends" are `hard`, `modeswitch` and `modeswitch_guard`. A prediction that names
`modeswitch` is scored for `modeswitch` and `modeswitch_guard` separately.

**P1** is scored per discrete backend, exact arm, flip 0.
* (a) At every d, ≥ 0.95 of look-alike first appearances fire.
* (b) At every d, no look-alike game ends `wrong`.
* (c) At d ∈ {1, 2}: (mean questions of fired look-alike first appearances with the
  parent known) − (mean questions of fired first appearances of real objects) is in
  [0.5, 1.5].

**P2** is scored per backend × flip rate, exact arm, over d in increasing order.
* (a) Look-alike AUC is non-decreasing in d, allowing an adjacent drop of up to 0.01.
* (b) Look-alike AUC at d = 1 is the minimum over d.
* (c) The look-alike median fire turn is non-increasing in d, allowing an adjacent
  rise of up to 1 turn, and it is strictly lower at d = 12 than at d = 1.

**P3** is scored per `modeswitch` variant, exact arm, over flip ∈ {0.05, 0.10} × d ∈
{1, 2}. In each of the four cells:
* the look-alike novel median fire turn minus `hard`'s is in [1, 3];
* no look-alike game ends `wrong`.

**P4** (headline) is scored over the G arm's g ∈ {0.05, 0.20} × d ∈ {1, 2}, on
look-alike novel games with the parent known.
* (a) Per backend, in each of the four cells: `false_confirm` rate / (g × P(parent
  guessed by name)) is in [2/3, 3/2]. For the discrete backends P(parent guessed) is
  also ≥ 0.8.
* (b) Per discrete backend, in each of the four cells: its `false_confirm` rate is ≥
  `soft`'s.

**P5** is scored per backend, exact arm, flip 0, on games from game 274 on (Decision
9). An *event* is a parent's earliest look-alike admission at game a ≥ 274, when the
parent was admitted before a and has in-pool games from 274 on both before and after
a. Each of the parent's games is adjusted by subtracting
the mean questions of the control games within 100 games of it (Decision 9). The
event's value is the mean adjusted questions after a minus the mean before a.
* At d ∈ {1, 2}: the mean event value is in [0.5, 1.5].
* At d ∈ {6, 8, 12}: the 95% interval of the mean event value (mean ± 1.96 SE over
  events) contains 0.

**Reported, not scored.** Re-guesses of an already refuted name per game (*Notes*),
look-alike outcome splits before and after admission, P(parent guessed) and
`false_confirm` unconditionally, raw P5 differences, repeat-question rate, and the
look-alike census (distance to the parent, to the nearest other object, and how
often another object is nearer than the parent).

## What would falsify, and what it would mean

* P1 failing (look-alikes produce `wrong` under exact guesses): the discrete backends
  have a look-alike failure that does not run through the guess channel. That would be
  more serious than P4 and would need a trajectory read before anything else.
* P4 failing (`false_confirm` for `hard` ≤ `soft` at d ≤ 2): the discrete backends'
  exposure to name confusion is not worse, and P0a's safety claim needs a weaker
  caveat than predicted.

## Notes

* Reading `ModeSwitchedRetraction` while drafting P3: a known struck by a "no" to its
  own exact by-name guess has one killer, so it can be reinstated and guessed again.
  Under exact guesses that second guess cannot succeed. It costs a question on novelty
  games, and is part of the "delayed by at most K" detector cost the retraction study
  reported. This study measures how often it happens (re-guesses of an already-refuted
  name per game) but does not change the backend. Excluding guess-killed knowns from
  reinstatement is a one-line variant, and becomes a real design choice only once
  guesses are noisy (ST-4 G arm, where re-testing a false-no is exactly what rescues
  the truth).
* The G-arm rates (0.05, 0.20) are placeholders. ST-6 Stage 0 measures real near-name
  false-yes rates. If those land outside this range, the grid is extended before
  registration, not after. *Revision:* ST-2 is registered before ST-6 Stage 0 has
  run, so these placeholders are what is registered. If Stage 0 lands outside them,
  the extra G-arm rates are a separate, separately registered follow-up.
* Flipping cells uniformly at random is one look-alike model. A variant that flips only
  the parent's least diagnostic cells (|p − 0.5| smallest) is harder to see. It is
  listed as exploratory and will not be scored against predictions.
