# 20Q stress tests — design note

**Status: DRAFT, not registered.** This note and the six pre-registrations it indexes
are drafts for review. Nothing here has been run. Each pre-registration is frozen only
when its commit is tagged `prereg-stress-<id>-<date>`, before its first run, following
the `prereg-noul1-20261108` convention.

## Why stress test

The 20Q BED line ended in a production recommendation (`PRODUCTION_RECOMMENDATIONS.md`):
ship the discrete, hard-gated backend (`modeswitch`) with an append-only op-log, spend
the expensive oracle early and sparingly, add a Pitman–Yor catch-all, skip onboarding.
Every one of those findings was measured on a single world: the 158-object benchmark
matrix, a self-consistent matrix oracle, i.i.d. attribute flips, and a 20-question
budget. The stress tests ask where each finding stops holding, by attacking the
assumptions it rests on one at a time.

## What the earlier results rest on

Read from the code, not only the write-ups:

| assumption | where it lives | attacked by |
|---|---|---|
| The answerer reads the same matrix the planner believes (self-consistent oracle). | `matrix_oracle`; §3.11 caveat "the detector inherits the oracle's calibration" | ST-4, ST-6 |
| Attribute noise is i.i.d. and independent of the object. | `noisy_matrix_oracle` | ST-4, ST-6 |
| By-name guesses and category answers are exact. | `noisy_matrix_oracle` docstring; `modeswitch`'s reinstate-and-test depends on it | ST-2 (G arm), ST-4 (G arm), ST-6 |
| Novel objects are ordinary draws from the 158, not look-alikes of known ones. | `make_stream` | ST-2 |
| The support is finite (158); the tail comes from draw order. | `OPENWORLD_RESULTS.md` caveats | ST-3 |
| The PY prior is fitted on the stream it is scored on. | `OPENWORLD_RESULTS.md` caveats | ST-3 |
| The budget is loose relative to the pool. | `PRODUCTION_RECOMMENDATIONS.md` "Before you trust the numbers" | ST-5, ST-1 |
| Pool size and crowding are whatever the benchmark happened to have. | the 158-object dictionary | ST-1 |
| An admitted object arrives with a correct row. | §3.11 caveats; ENV-704-A models a *partial* row, not a *wrong* one | ST-6 |

## Three mechanisms the drafts lean on

These come from reading `openworld.py` and `openworld_backends.py`. They shape the
predictions, so they are stated once here.

1. **For the discrete backends, "wrong" only ever means "out of budget".** A by-name
   guess is exact and a "no" does not end the game: the guessed known is struck and play
   continues. A game is scored `wrong` only when the 20 questions run out with a known on
   top. So the zero-wrong-guess property of `hard` and `modeswitch` is really two
   properties: **exact guesses** and **budget slack**. Crowding (ST-1) and tight budgets
   (ST-5) attack the slack; guess noise (ST-2, ST-4, ST-6) attacks exactness. Today the
   loop scores any "yes" to a guess as `won`, so a noisy guess channel would hide a wrong
   confirmation as a win. ST-0c fixes the scoring before any guess noise is introduced.
2. **`hard` kills on the binarized side, not the graded value.** A known is struck when
   the observed answer has likelihood below 0.5 under its row. A cell at p = 0.45 is
   killed by a "yes" exactly as hard as a cell at p = 0.02. With a thresholding oracle
   the truth never contradicts its own row. With any answerer that is not a 0.5
   threshold of the planner's matrix (a sampler, another model, the same model on
   another day), every ambiguous cell becomes a way to strike the truth.
3. **`modeswitch` only reinstates when the live set is empty.** Its rescue of a
   flip-struck truth fires at the escalation boundary. If close neighbours of the truth
   survive the flip, the set never empties and the rescue never triggers. Crowding
   should therefore weaken `modeswitch` specifically.

## Baseline crowding census (real matrix)

The 44 attribute questions are already close to saturation at 158 objects. Binarized
at 0.5 (the oracle's threshold):

| quantity | value |
|---|---:|
| distinct attribute rows | 140 of 158 |
| objects sharing an identical row with another object | 32 |
| largest identical-row cluster | 4 |
| objects whose nearest neighbour is ≤ 1 attribute away | 98 of 158 |
| ambiguous cells (0.2 < p < 0.8) | 1035 of 6952 (14.9%) |
| attribute-flip rate if the answerer samples yes ~ Bernoulli(p), uniform over cells | 0.090 |

The 17 categories hold 5–14 objects each. Within a category, 31 objects have an
identical-row twin. So by-name guesses and category questions, not attributes, do the
last step of most games. Reproduce (deterministic, no API):

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_census \
    --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json
```

The sampling flip rate (0.090) is about 11× the 0.8% inconsistency Jev showed live
(`RESULTS.md` §3). So Jev does not behave like a sampler. A3 in ST-4 is a pessimistic
bound, not a model of Jev.

## The six studies

| id | pre-registration | question | API cost |
|---|---|---|---|
| ST-1 | `STRESS_CROWDING_PREREG.md` | As the pool grows and packs tighter, where do the discrete backends start returning wrong answers, and does `modeswitch`'s rescue survive? | none |
| ST-2 | `STRESS_LOOKALIKE_PREREG.md` | Can the catch-all see a novel object that differs from a known one by *d* attributes, and what happens when the answerer confuses names? | none |
| ST-3 | `STRESS_TAIL_PREREG.md` | With a truly unbounded supply of species and tails up to σ = 0.75, does expert load decay, and does a misspecified PY prior break the gate? | none |
| ST-4 | `STRESS_ANSWERER_PREREG.md` | Realistic answerer errors: graded/sampled, sticky per object, bursty, a different model, and noisy guesses. | one second-model matrix (~158 calls) |
| ST-5 | `STRESS_BUDGET_PREREG.md` | Where do "spend early and sparingly" and "skip onboarding" flip as the budget tightens against the pool? | none |
| ST-6 | `STRESS_SALIENCE_PREREG.md` | Does model knowledge degrade with salience, and what does a tail that is both rare *and* poorly known do to the architecture? | Stage 0: a few thousand calls. Stage 2: live games |
| ST-7 (draft) | `STRESS_OPENSIDE_PREREG.md` | As the novelty share grows, how far does the lumped catch-all's EIG drift from an exact treatment of the open side, and what does that cost? | none |

## Shared infrastructure (ST-0)

Built once, before any study runs. Each piece has a regression gate: with defaults, the
committed §3.11 clean run (146 distinct, 144 expert, 0 false-positive, 1854 won,
2 wrong) and the retraction grid `openworld_retraction_20261108_213525` must reproduce
bit-for-bit.

**Built: see `STRESS_ST0.md`.** As built, the gate is split in two, because the
survivor-set backends' committed values depend on Python's string-hash order and
cannot be reproduced exactly. The build also found a repeat-question loop in the
discrete backends at catch-all priors ≥ 0.15, which affects ST-1 and ST-3 as
drafted.

* **ST-0a — pool generator** (`stress_generator.py`). Makes new objects with realistic
  attribute structure from the real matrix. A new object picks a category (weighted by
  real category size) and a parent object in it. Each of its 44 cells is then
  independently, with probability ρ, replaced by a draw from that category's empirical
  values for the attribute, otherwise kept from the parent. Graded values survive, and
  so do the within-category correlations. ρ is the crowding knob: small ρ gives
  near-duplicates. **Calibration:** `ρ_real` is the ρ at which a generated 158-object
  pool matches the real census on `nn_within_1` (98/158), mean over 5 seeds. Crowding
  levels in ST-1/3/5 are stated relative to `ρ_real`. A second entry point,
  `lookalike(parent, d)`, flips exactly *d* cells across 0.5 (p → 1 − p) for ST-2.
* **ST-0b — answer channel split.** The oracle reads an `answer_matrix` (default: the
  belief matrix) through a pluggable noise model per question kind. Attribute models:
  i.i.d. (the existing `noisy_matrix_oracle`, kept untouched so its RNG stream is
  unchanged), sampled, sticky, bursty, cross-model. Guess model: false-yes and
  false-no. The existing `like_matrix` seam already separates belief from truth on the
  planner side; this adds the matching seam on the answerer side.
* **ST-0c — true-outcome scoring.** A "yes" to a by-name guess is scored `won` only if
  the guessed name is the secret; otherwise it is a new terminal outcome,
  `false_confirm`, the unsafe failure the offline studies could not produce. With exact
  guesses, `false_confirm` is identically 0, so all prior results are unchanged.
* **ST-0d — snapshot mode.** `H_known` is seeded with a whole pool, and novelty comes
  from a held-out set drawn by the same generator. This isolates crowding (ST-1, ST-5)
  from the discovery dynamics that ST-3 studies.
* **ST-0e — census per cell.** Every grid cell records `stress_census.census` of the
  pool it ran on, so results can be plotted against crowding coordinates, not only
  against nominal N and ρ.
* **ST-0f — speed.** EIG costs O(|H_known| × candidates) per turn. `soft` at
  N = 5000 is the bottleneck. A numpy path is allowed for the new modes only, and must
  match the Python path to 1e-9 on a 200-game check. If a `soft` cell still exceeds the
  runtime cap written into its pre-registration, it is dropped *before* the run and
  the drop is recorded.

## Shared scorecard

Carried over from the retraction study, plus the new outcome:

* **In-pool outcomes, split:** `won`, safe escalation (an in-pool game that called the
  expert), `wrong` (out of budget with a known on top), `false_confirm`.
* **Novelty:** detector AUC (`max_t p_unknown`), TPR and FPR at the 0.90 gate, median
  fire turn.
* **Cost:** questions per game, by-name guesses per game, expert calls.
* **Spread:** every stochastic number is a mean over ≥ 3 seeds with its per-seed range.
  Results notes quote the range alongside the mean, never a single run.

## Order and dependencies

1. ST-0 (all pieces, regression gates green).
2. In parallel: ST-1, ST-2, ST-4 offline arms (A0, A2, A3, A4, G). ST-6 Stage 0 can
   start at the same time, since it only measures and shares no code with the others.
3. ST-3 (uses the generator in streaming mode and ST-1's runtime experience).
4. ST-5 (reuses the ST-1 pools).
5. ST-4 A1 needs a second model's matrix. It shares the elicitation code with ST-6
   Stage 0, so it lands with it.
6. ST-6 Stages 1–2, only if Stage 0 passes its gate.

## What would change the production recommendation

* ST-1 or ST-5 showing `wrong > 0` for `modeswitch` inside realistic crowding or budget:
  P0a needs a budget precondition stated next to it.
* ST-2, ST-4 or ST-6 showing `false_confirm` under realistic guess noise: P0a's safety
  is conditional on guess exactness. The recommendation would then need a confirm step
  (for example, a second exact check before committing) and its cost.
* ST-3 showing the soft gate breaking under a misspecified prior while `hard` holds:
  one more argument *for* P0a, worth stating.
* ST-6 passing its gate: the "detector inherits the oracle's calibration" caveat
  becomes a measured, salience-dependent cost instead of a footnote.
