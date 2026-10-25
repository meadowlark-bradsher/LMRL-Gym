# Pre-registration: the responder effect (confirmatory replication)

**Registered:** 2026-10-25, before any episode of this run was collected.
**Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Regime:** easy (r = 0.40)

## Origin and status

`QUALITY_RESULTS.md` §Exploratory found, **post hoc**, that the pre-registered
"seller asymmetry" (Δ G_S resolved, Δ G_B not) is better described as a
*responder* effect: learning helps whichever side moves second, because one
observed concession lets the responder counter with a mutual-improvement deal
instead of accepting the starter's offer. That analysis was not pre-registered
and is therefore a hypothesis. This document fixes the hypothesis, the
predictions, the analysis and the decision rules **before** collecting the data
that will test it.

## Design — identical to `PREREGISTRATION_QUALITY.md`

- Conditions: `--aid --aid-mode bayes` vs `--aid --aid-mode prior_only`.
  β = 8, ε = 0.05, own-utility revealed, no salience, no option shuffling,
  6-proposal budget, same instruction text.
- All **36** ordered type pairs × both starting sides × **K = 8** → 576 episodes
  per arm. Unit of inference = ordered type pair.
- Hardened runner (streamed records, per-episode retry ×3, failures recorded and
  counted, never dropped).

**On "fresh seed".** The runner's `--seed` governs only option shuffling (off)
and pair sampling (all pairs), so it does *not* make Jev reproducible:
`jev-latest` is stochastic server-side and every run is an independent draw.
This replication's independence comes from the API, not from our seed.
`--seed 1` is set for provenance only. The bootstrap seed is fixed at 1 so the
analysis is reproducible from the logs.

```bash
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
  --pairs all --regime easy --start both --repeats 8 --seed 1 --aid --aid-mode bayes \
  --out results/jj_quality_rep1_20261025/bayes
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
  --pairs all --regime easy --start both --repeats 8 --seed 1 --aid --aid-mode prior_only \
  --out results/jj_quality_rep1_20261025/prior_only
```

## Predictions (fixed in advance) and decision rules

All Δ are bayes − prior-only, with paired type-pair-clustered bootstrap 95%
percentile intervals (B = 5000, seed 1, 36 clusters). Point predictions are the
exploratory run's values, stated so they can be wrong.

| | Prediction | Decision rule | Exploratory value |
|---|---|---|---|
| **P1 (primary)** | Δ G_responder > 0 | CI for Δ G_responder excludes 0, positive | +0.208 (0.146, 0.272) |
| **P2** | Δ G_starter ≈ 0 | CI includes 0 **and** \|Δ\| ≤ 0.10 | −0.037 (−0.091, 0.018) |
| **P3** | starter-proposal-accepted rate falls | Δ rate < 0, CI excludes 0 | −0.324 (−0.408, −0.238) |
| **P4 (validity gate)** | starter-first-offer placebo ≈ 0 | all four STARTER first-offer Δ CIs include 0 | all ≈ 0 |

**P4 is a validity gate, not a finding.** The starter's first offer sees identical
information in both arms; if any of its four Δs resolves away from 0 the
pipeline is suspect and interpretation of P1–P3 is suspended pending diagnosis.

**On P2.** "≈ 0" is a weaker claim than a positive effect: a CI that includes 0
can reflect low power. The magnitude bound is what gives it content. If the CI
includes 0 but \|Δ\| > 0.10, P2 is **not** supported and we report the interval.

**Replication of the original primary.** Δ Nash Q_N > 0 (CI excludes 0) is also
re-tested, as a replication outcome of `PREREGISTRATION_QUALITY.md`
(exploratory-run value +0.158 (0.054, 0.263)).

**Mechanism corollary (reported, not a decision):** the RESPONDER's first offer
should be better for *both* sides than in prior-only (Δ own utility > 0 and Δ
recipient utility > 0). Exploratory: +0.101 and +0.138.

## Analysis (fixed in advance)

```bash
# P1–P4 and the mechanism corollary (unchanged script; hypotheses now fixed before data)
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_seller_asymmetry \
  results/jj_quality_rep1_20261025/bayes results/jj_quality_rep1_20261025/prior_only --seed 1
# Δ Nash and the other quality secondaries
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
  results/jj_quality_rep1_20261025/bayes results/jj_quality_rep1_20261025/prior_only --seed 1
```

- The replication is analysed **alone** first and reported against the rules
  above. A pooled two-run estimate may follow, clearly labelled secondary; it
  never substitutes for the replication-alone verdict.
- Analysis runs on successful episodes only; the failure count is reported.
- No rule, bound or outcome is changed after data are seen. Secondaries are
  reported in full; none is promoted.

## Sanity gates (must pass or the comparison is not interpretable)

- Agreement rate ≥ 0.95 in both arms.
- 0 ideal offers in both arms.
- bayes mean log-score improvement > 0.
- Failure count reported.

## What would falsify

- P1 CI includes 0 → the responder effect did not replicate; the exploratory
  finding is reported as a non-replication, with both intervals shown.
- P3 CI includes 0 → the proposed mechanism (countering instead of accepting)
  is not supported even if P1 holds.
- P4 fails → validity problem; nothing else is interpreted until explained.

## Limits accepted in advance

Easy regime only; one prompt; no candidate-order randomization; β and ε not
swept; `jev-latest` pinned by name, not version; two runs total after this.
