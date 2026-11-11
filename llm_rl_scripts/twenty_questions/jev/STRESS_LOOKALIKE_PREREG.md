# ST-2 Look-alike tail — pre-registration

**Status: DRAFT, not registered.** Frozen when tagged `prereg-stress-st2-<date>`.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.

## Question

Real long tails are full of objects that resemble common ones: the rare subspecies next
to the common species. The §3.11 clean run had 2 novelty misses, and both were novel
objects whose answers happened to match a known object closely. How close can a novel
object sit to a known one before the catch-all stops seeing it? And when the answerer
can confuse the two names, what does that do to the backends?

## Setup

Accumulation mode on the real matrix. The 158 objects are joined by 100 look-alikes,
each made by `lookalike(parent, d)` (ST-0a): flip exactly *d* of the parent's 44 cells
across 0.5. The look-alike keeps the parent's category. Parents are drawn with the
parent's Zipf weight, so look-alikes of common objects are commoner, as with
subspecies. The 258 objects are streamed by the existing `make_stream` (Zipf 1.1, random
rank, seed 7), 2500 games. An admitted look-alike joins `H_known`, so later games of its
parent face a closer neighbour.

| axis | levels |
|---|---|
| distance d | 1, 2, 3, 4, 6, 8, 12 |
| attribute-flip rate | 0, 0.05, 0.10 |
| backend | `soft`, `hard`, `modeswitch` (K = 3) |
| guess channel (G arm) | exact; near-name false-yes g = 0.05, 0.20 |
| seeds | 3 look-alike seeds × 3 noise seeds |

**G arm.** When the planner guesses a known by name and the secret is that known's
look-alike with d ≤ 2, the answerer says "yes" with probability g. The answerer
is treating the subspecies as the species. All other guesses stay exact. Scored with
ST-0c, so a false "yes" is a `false_confirm`, not a win. The G arm runs at flip rate 0
only, to isolate the guess channel.

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
  registration, not after.
* Flipping cells uniformly at random is one look-alike model. A variant that flips only
  the parent's least diagnostic cells (|p − 0.5| smallest) is harder to see. It is
  listed as exploratory and will not be scored against predictions.
