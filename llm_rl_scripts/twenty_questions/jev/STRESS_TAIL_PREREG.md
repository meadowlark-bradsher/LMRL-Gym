# ST-3 Endless tail — pre-registration

**Status: DRAFT, not registered.** Frozen when tagged `prereg-stress-st3-<date>`.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.

## Question

§3.11's heavy tail (σ = 0.200) came from the draw order of a Zipf law over a *finite*
158 objects. Given enough games, novelty would run out. And the live gate used PY
parameters fitted on the very stream it was scored on. This study gives the loop a
genuinely unbounded supply of species and tails heavier than anything measured. It
asks whether expert load still decays, and whether the gate survives a prior that is
estimated online or simply wrong.

## Setup

Accumulation mode with a streaming generator. Game n draws a new species with
probability (θ + Kσ)/(θ + n), the Pitman–Yor / Chinese-restaurant rule. Otherwise it
draws a seen species in proportion to (count − σ). A new species' row comes from the
ST-0a generator at ρ = ρ_real. The first 158 new species are the real objects in a
seeded random order; after that, they are generated. 5000 games per run.

| axis | levels |
|---|---|
| true tail σ | 0 (Dirichlet), 0.25, 0.5, 0.75 (θ = 20 throughout) |
| planner prior | `true` (the generating σ, θ); `online` (sequential PY MLE refit every 100 games on history only); `fixed` (σ = 0.2, θ = 20, the §3.11 fit) |
| attribute-flip rate | 0, 0.05 |
| backend | `soft`, `hard`, `modeswitch` (K = 3) |
| seeds | 3 stream seeds × 2 noise seeds |

σ = 0.9 is excluded. K would pass several thousand within 5000 games, which turns this
into the crowding study (ST-1). σ = 0.75 is the heaviest level whose pool stays inside
ST-1's measured range.

## Metrics

* Expert-call rate per 250-game window, split into necessary (true novelty) and
  false-positive calls. The autonomous share in games 2501–5000 minus that in games
  1–2500.
* Novelty TPR/FPR at the gate, AUC, median fire turn; per window and overall.
* **Prior calibration:** the gate's `p_unknown_prior` against the realised novelty
  frequency, per window (a reliability curve).
* The online PY estimate (σ̂, θ̂) over time.
* The census of `H_known` per window, so the effects of tail shape can be separated
  from the effects of pool size.

## Predictions

* **P1 — expert load tracks the missing mass.** The windowed necessary-call rate
  follows the true p_new(t): it falls like t^(σ−1) for σ > 0, and like 1/t for σ = 0.
  At σ = 0.75 the autonomous share rises by less than 0.05 between the two halves of
  the stream. At σ = 0 it rises by more than §3.11's 0.092 (0.882 → 0.974).
* **P2 — the discrete gate ignores the prior.** For `hard` and `modeswitch`,
  novelty TPR and FPR at the gate differ by less than 0.02 between the `true`,
  `online` and `fixed` priors, at every σ. Their expert call fires when the live set
  empties; the prior only matters when it exceeds 0.90 itself, which happens only in
  the first few games.
* **P3 — the soft gate does not.** At σ = 0.75 under the `fixed` prior (which
  under-predicts novelty), `soft`'s TPR at the gate falls by more than 0.05 against
  the `true` prior. At σ = 0 under `fixed` (which over-predicts), `soft`'s FPR rises.
  This would be another argument for P0a: the hard gate is robust to a misspecified
  novelty prior, and the soft gate is not.
* **P4 — the online estimate is usable.** For σ ≥ 0.25, the online σ̂ is within
  ±0.05 of the true σ by game 2000, mean over seeds, and the `online` arm's soft TPR is
  within 0.02 of the `true` arm's after game 2000.
* **P5 — the ranking survives, the pool size is what hurts it.** AUC falls with σ, but
  when windows are matched on |H_known| across σ levels, the AUC difference between σ
  levels is below 0.02. Any loss of separation at heavy tails comes from the larger,
  more crowded pool, not from the tail shape itself.

## What would falsify, and what it would mean

* P2 failing: the discrete gate depends on the prior after all. That would have to be
  traced in the backend code before trusting P0a's "no tuning needed" reading.
* P5 failing (AUC differs at matched pool size): the tail shape itself degrades
  detection, perhaps through how often the background marginal is refreshed. That
  would be a new effect worth its own write-up.

## Note on what stays fixed

The expert still supplies a correct row on admission. Wrong rows on admission are
ST-6's subject. Keeping them out of this study keeps the tail effect clean.
