# ST-1 Crowding and scale — pre-registration

**Status: REVISED DRAFT, not registered.** Frozen when tagged
`prereg-stress-st1-<date>`. This revision settles the decisions `STRESS_ST0.md`
raised: the catch-all prior and the repeat-question loop, and the crowding levels.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`. Build notes and
the loop finding: `STRESS_ST0.md`.

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

## Decisions taken in this revision

**1. Catch-all prior and the repeat loop.** The draft set the prior to a novelty share
of 200/1200 = 0.167. That is inside the range where `hard` and `modeswitch` loop on a
repeated question: on the real pool they win 0.433 of in-pool games at a prior of 0.15
(`STRESS_ST0.md`, Finding 2). This revision makes three changes:

* **The game mix is 1000 in-pool + 100 novelty games**, so the calibrated prior is
  100/1100 = **0.091**. That is below the loop's threshold on the real pool (none of
  the 150 probe games looped at 0.10), and the prior still equals the true novelty
  rate. The probe was re-run on fork main d2a258b, after the hash-independence fix,
  and is identical to `results/stress_st0_diagnostics_20261111/repeat_probe.json`.
* **The primary arm is the published architecture** (`no_repeat` off) for all three
  backends.
* **A secondary arm runs `hard` and `modeswitch` with `no_repeat` on.** `soft` is not
  run there: it wins every in-pool game at every prior tested, with or without
  repeats.

Every cell logs its **repeat rate**: the fraction of games that ask the same non-guess
question twice. If the loop's threshold falls as crowding rises, the two discrete arms
diverge, and the analysis shows it directly instead of reading it as a crowding effect.

**2. Crowding levels.** The draft's ρ_real/4 and ρ_real/2 (about 126 and 119 of 158
near-neighbours) sit within seed noise of each other. Its 2·ρ_real (about 82) sits
on the generator's floor (74.6 at ρ = 1). The revised levels span everything the
generator can produce:

| level | ρ | nn_within_1 of an all-generated 158-object pool (20 seeds) | meaning |
|---|---:|---:|---|
| twins | 0 | 130.7 (93 distinct rows) | maximum budget exhaustion |
| crowded | ρ_real/3 = 0.078 | 120.3 | |
| real | ρ_real = 0.2345 | 98 (the target) | |
| floor | 1 | 74.6 | the least crowding this generator can produce |

Calibration: `results/stress_calibration_20261111/calibration.json`; the 0 and 0.078
rows are `st1_levels.json` in the same directory.

The calibration pools are all-generated, so at ρ = 0 only children that share a parent
are twins. ST-1's pools contain the real parents themselves, so there **every
generated object at ρ = 0 is an exact twin of a real object in the pool**. The
twins level is therefore more crowded in ST-1 than its calibration row suggests.

**What this study cannot test.** Pools much *less* crowded than the real matrix: the
generator bottoms out at about 75 of 158, and the real matrix sits at 98. Widening the
low end needs a different generator or more attributes, which is a follow-on (see
*Out of scope*).

## Setup

Snapshot mode (ST-0d). `H_known` is seeded with the whole pool. Each cell plays 1000
in-pool games (secrets uniform over the pool) and 100 novelty games (secrets from a
held-out set of 100 generated objects at the same ρ, not in `H_known`), in a seeded
shuffled order. `p_unknown_prior` = 100/1100. Budget 20, confidence 0.90, ε = 0.05,
`modeswitch` K = 3, all as in the retraction study.

| axis | levels |
|---|---|
| pool size N | 158 (real matrix only), 500, 1000, 2000, 5000 |
| packing ρ | 0, 0.078, 0.2345, 1 (only for N > 158; N = 158 is the real matrix) |
| attribute-flip rate | 0, 0.05, 0.10 (i.i.d., `noisy_matrix_oracle`) |
| arm | primary: `soft`, `hard`, `modeswitch`. Secondary (`no_repeat`): `hard`, `modeswitch` |
| seeds | 3 generator seeds × 3 noise seeds (one run at flip rate 0, where noise seeds don't apply) |

Pools with N > 158 are the 158 real objects plus N − 158 generated ones. Every run
pins `PYTHONHASHSEED=0` and runs under the `LLM_RL` env (Python 3.9). Both are
recorded in the summary.

**Runtime cap.** If one `soft` cell at N = 5000 takes more than 2 h after ST-0f,
`soft` runs only at N ≤ 2000. That decision is made and recorded before the grid
starts. The discrete arms always run the full grid.

### To build before tagging

These do not change the design above:

* **ST-0f:** a numpy EIG path for `soft`, matching the Python path to 1e-9 on a
  200-game check.
* **A repeat counter** on `SnapshotRecord`: games that asked the same non-guess
  question twice.
* **A rescue-trigger diagnostic:** whether the live set emptied after a flip struck
  the truth.
* **The ST-1 runner and analyzer**, with figures whose captions name the analyzer and
  the result directory.

## Metrics

Per cell, from the shared scorecard:

* in-pool outcome split: `won` / safe escalation / `wrong`. `false_confirm` is 0 by
  construction here (exact guesses).
* questions per game, and the share of them that are by-name guesses
* novelty TPR/FPR at the gate, AUC, median fire turn. AUC is pooled over a cell's
  seeds (300 novelty games at flip rate 0, 900 under noise), because 100 per run is
  thin.
* the census of the pool
* the repeat rate

Two diagnostics specific to this study:

* **Budget-exhaustion share.** Of `wrong` games, the fraction that ended with ≥ 2 live
  knowns at turn 20.
* **Rescue trigger rate.** Among in-pool noisy games where a flip struck the truth, the
  fraction in which the live set then emptied. That is the precondition for
  `modeswitch` to reinstate at all.

## Predictions

**Which arm each prediction is scored on.** P1–P5 are scored on **both** discrete
arms. A prediction that holds in the `no_repeat` arm but fails in the primary arm is
attributed to the loop (P7), not to crowding.

* **P1 — crowding costs questions before it costs errors.** At flip rate 0, `hard` and
  `modeswitch` have `wrong` = 0 in every cell whose clean 95th-percentile
  questions-per-game is below 20. The extra questions that crowding adds are mostly
  by-name guesses: the guess share of questions rises with the pool's identical-row
  fraction.
* **P2 — when the discrete backends do fail, it is the budget.** In every cell with
  `wrong` > 0 for `hard` or `modeswitch` at flip rate 0, the budget-exhaustion share is
  ≥ 0.95. Under exact guesses the discrete backends do not confirm a wrong neighbour;
  they run out of questions. The twins level (ρ = 0) at N ≥ 2000 is where `wrong` > 0
  should first appear.
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
  confidence). Pooled novelty AUC falls with crowding for all backends. `soft` stays
  ≥ 0.95 at a crowding level where `hard` has fallen below it, because `hard`'s novelty
  signal needs the live set to empty, and close neighbours of a novel object delay
  that. At the twins level, novel objects copy a known parent's row exactly; they can
  be told apart only by a failed by-name guess. So AUC there is the lowest in the grid
  for every backend.
* **P6 — the real-matrix anchor.** At N = 158 on the real matrix, in the primary arm,
  snapshot mode reproduces the retraction study's qualitative ordering at 10% flips:
  `hard` and `modeswitch` `wrong` = 0, and `modeswitch` in-pool failure below `hard`'s.
* **P7 — crowding pulls the loop in** (lower confidence). At N = 158 on the real
  matrix, the primary arm's repeat rate for `hard` and `modeswitch` is ≤ 0.02 at flip
  rate 0: a prior of 0.091 is below the loop's threshold there. The repeat rate rises
  with crowding: Spearman correlation with `nn_within_1` fraction ≥ +0.5 across cells.
  At the twins level with N ≥ 2000, the primary and `no_repeat` arms' in-pool win
  rates differ by ≥ 0.05. The mechanism is that in a crowded pool, every attribute
  question splits the survivors poorly and guesses split a large uniform set weakly.
  So a question that splits `UNKNOWN` from the knowns wins the EIG ranking at a lower
  prior than on the real pool.

## What would falsify, and what it would mean

* **`wrong` = 0 for `modeswitch` in every cell,** including N = 5000 at the twins
  level: P0a is crowding-robust at this budget. That is good news and gets written up
  as such.
* **P3 failing** (recovery does not fall with crowding): the reinstate trigger is not
  the weak point, and mechanism 3 in the design note is wrong.
* **P2 failing** (`wrong` games with a single live known at turn 20): there is a
  failure path under exact guesses that the design note missed. Stop and read the
  trajectories before interpreting anything else.
* **P7 failing because the repeat rate stays ≤ 0.02 everywhere:** the loop is a
  high-prior phenomenon only, and a calibrated prior below 0.10 is enough to avoid it
  at any crowding.
* **P7 failing because the repeat rate is high already at N = 158:** the loop's
  threshold under snapshot mode is below 0.091. The primary arm's crowding results are
  then confounded, and the `no_repeat` arm becomes the reading of record for P1–P5.

## Consequence for the production rule, whatever the outcome

Adding `no_repeat` to P0a is supported by the loop finding on its own, independently
of ST-1. ST-1's secondary arm adds the evidence of whether it matters at realistic
novelty rates. Making it the default would change committed results (the 12 clean
games behind `hard`'s gap to `soft`), so it remains a separate decision, not part of
this study.

## Out of scope

* Growing the question bank alongside N, for example by eliciting more attributes, is
  the obvious fix for collisions, and the way to reach pools less crowded than the
  real one. It is a follow-on study, not part of this grid. This study measures the
  problem at a fixed 44-attribute vocabulary.
* `no_repeat` for `soft`, and priors above 0.10. The loop as a stressor across priors
  was considered and set aside: it would double the grid and its predictions.
