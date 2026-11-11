# ST-1 Crowding and scale — pre-registration

**Status: DRAFT, not registered.** Frozen when tagged `prereg-stress-st1-<date>`.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.

## Question

The production rule P0a (`modeswitch`: hard elimination, bounded reinstate-and-test)
was measured on 158 objects. How does it behave as the pool grows and its objects pack
closer together? The headline metric is the **split of in-pool failures into safe
escalations and `wrong` outcomes**, not the win rate. P0a's value is that its failures
are safe.

## Why crowding, not just size

The real matrix is already crowded: 140 distinct rows for 158 objects, 98 objects
within one attribute of a neighbour (`STRESS_DESIGN.md`, census). Adding objects with
the same 44 questions mainly adds collisions. So the study crosses size with packing
density and reports results against the measured census of each pool, not just N.

## Setup

Snapshot mode (ST-0d). `H_known` is seeded with the whole pool. Each cell plays 1000
in-pool games (secrets uniform over the pool) and 200 novelty games (secrets from a
held-out set of 200 generated objects that are not in `H_known`). `p_unknown_prior` is
set to the novelty share of the game mix, 200/1200. Budget 20, confidence 0.90,
ε = 0.05, `modeswitch` K = 3, all as in the retraction study.

| axis | levels |
|---|---|
| pool size N | 158 (real matrix), 500, 1000, 2000, 5000 |
| packing ρ | ρ_real/4, ρ_real/2, ρ_real, 2·ρ_real. Lower ρ means more crowded. For N = 158 only the real matrix is used. |
| attribute-flip rate | 0, 0.05, 0.10 (i.i.d., `noisy_matrix_oracle`) |
| backend | `soft`, `hard`, `modeswitch` |
| seeds | 3 generator seeds × 3 noise seeds |

Pools with N > 158 are the 158 real objects plus N − 158 generated ones. The held-out
novelty set comes from the same generator at the same ρ.

**Runtime cap:** if one `soft` cell at N = 5000 takes more than 2 h after ST-0f, `soft`
runs only at N ≤ 2000. That decision is made and recorded before the grid starts.

## Metrics

Per cell, from the shared scorecard:

* in-pool outcome split: `won` / safe escalation / `wrong`; `false_confirm` is 0 by
  construction here (exact guesses)
* questions per game, and the share of them that are by-name guesses
* novelty TPR/FPR at the gate, AUC, median fire turn
* the census of the pool

Two diagnostics specific to this study:

* **Budget-exhaustion share.** Of `wrong` games, the fraction that ended with ≥ 2 live
  knowns at turn 20.
* **Rescue trigger rate.** Among in-pool noisy games where a flip struck the truth, the
  fraction in which the live set then emptied. That is the precondition for
  `modeswitch` to reinstate at all.

## Predictions

* **P1 — crowding costs questions before it costs errors.** At flip rate 0, `hard` and
  `modeswitch` have `wrong` = 0 in every cell whose clean 95th-percentile
  questions-per-game is below 20. The extra questions that crowding adds are mostly
  by-name guesses: the guess share of questions rises with the pool's identical-row
  fraction.
* **P2 — when the discrete backends do fail, it is the budget.** In every cell with
  `wrong` > 0 for `hard` or `modeswitch` at flip rate 0, the budget-exhaustion share is
  ≥ 0.95. Discrete backends do not confirm a wrong neighbour under exact guesses; they
  run out of questions.
* **P3 — crowding disarms the rescue.** Under noise, `modeswitch`'s relative recovery
  over `hard`, (fail_hard − fail_ms) / fail_hard, falls as crowding rises: Spearman
  correlation with the pool's `nn_within_1` fraction ≤ −0.5 across cells. The rescue
  trigger rate falls the same way. The mechanism is that neighbours survive the flip
  that struck the truth, so the live set never empties.
* **P4 — noise moves the failure boundary in.** For `hard` and `modeswitch`, the
  crowding level at which `wrong` first exceeds 0.01 is lower at 10% flips than at 0%.
  Flips strike the truth while leaving its neighbours to be guessed one by one, which
  spends the budget.
* **P5 — the detector degrades with crowding, discrete backends first** (lower
  confidence). Held-out novelty AUC falls with crowding for all backends. `soft` stays
  ≥ 0.95 at a crowding level where `hard` has fallen below it, because `hard`'s novelty
  signal needs the live set to empty, and close neighbours of a novel object delay
  that.
* **P6 — the real-matrix anchor.** At N = 158 on the real matrix, snapshot mode
  reproduces the retraction study's qualitative ordering at 10% flips: `hard` and
  `modeswitch` `wrong` = 0, and `modeswitch` in-pool failure below `hard`'s.

## What would falsify, and what it would mean

* `wrong` = 0 for `modeswitch` in every cell, including N = 5000 at ρ_real/4: P0a is
  crowding-robust at this budget. That is good news and gets written up as such.
* P3 failing (recovery does not fall with crowding): the reinstate trigger is not the
  weak point, and mechanism 3 in the design note is wrong.
* P2 failing (`wrong` games with a single live known at turn 20): there is a failure
  path under exact guesses that the design note missed. Stop and read the trajectories
  before interpreting anything else.

## Out of scope

Growing the question bank alongside N, for example by eliciting more attributes, is the
obvious fix for collisions. It is a follow-on study, not part of this grid. This study
measures the problem at a fixed 44-attribute vocabulary.
