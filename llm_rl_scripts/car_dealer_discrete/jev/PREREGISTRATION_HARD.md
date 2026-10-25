# Pre-registration: bayes vs prior-only in the HARD regime (r = 0.55)

**Registered:** 2026-10-25, before any episode of this run was collected.
**Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Regime:** hard (r = 0.55)

## What changes, structurally

- The universal compromise `P1-W1-D1` (worth 0.5 to every ranking) is **not**
  individually rational. The aid's prior-averaged P(accept) is therefore more
  discriminating, and there is no "safe" deal to fall back on.
- **30 / 36** ordered type pairs admit an IR deal (`test_game.py` reproduces this);
  **6 / 36** admit none. For those six, *no deal is the correct outcome* (spec
  §4.1/4.4). They are scored on **correct-no-deal** and recognition efficiency,
  and are **never** scored as G/Nash = 0. All quality analysis runs with
  `--feasible-only` (30 clusters); the infeasible six are analysed separately.
- Agreement on feasible pairs is now an **outcome**, not a sanity gate. The easy
  regime's "agreement ≥ 0.95" gate does not carry over and is not imposed.

## Design — otherwise identical to the easy-regime runs

`--aid --aid-mode bayes` vs `--aid --aid-mode prior_only`; β = 8, ε = 0.05;
own-utility revealed; no salience; no option shuffling; 6-proposal budget; same
instruction text. All 36 pairs × both starts × **K = 8** → 576 episodes per arm.
Hardened runner. `--seed 3` is provenance only (`jev-latest` is stochastic
server-side; our seed does not control it). Bootstrap seed 3.

```bash
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
  --pairs all --regime hard --start both --repeats 8 --seed 3 --aid --aid-mode bayes \
  --out results/jj_hard_20261025/bayes
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
  --pairs all --regime hard --start both --repeats 8 --seed 3 --aid --aid-mode prior_only \
  --out results/jj_hard_20261025/prior_only
```

## Outcomes and decision rules (fixed in advance)

All Δ = bayes − prior-only, paired type-pair-clustered bootstrap, B = 5000, seed 3,
95% percentile intervals.

### A. Feasible pairs (30 clusters) — does the quality effect survive without the safe compromise?

| | Prediction | Rule | Easy-regime value (for reference) |
|---|---|---|---|
| **A1 (primary)** | Δ Nash Q_N > 0 (no-deal = 0) | CI excludes 0, positive | +0.158 / +0.154 (two runs) |
| A2 | Δ agreement rate | reported; no directional rule — expected *lower* than easy in both arms | 1.00 vs 0.98 |
| A3 | Δ min(G_B, G_S), Δ Pareto rate, Δ proposals | reported in full | all > 0 in easy |
| A4 | Responder effect: Δ G_responder > 0; Δ G_starter ≈ 0 (CI incl 0 and \|Δ\| ≤ 0.10); starter-proposal-accepted Δ < 0 | each as in `PREREGISTRATION_RESPONDER.md` | confirmed in easy |
| A5 (validity gate) | starter-first-offer placebo ≈ 0 | all four STARTER Δ CIs include 0 | passed in easy |

### B. Infeasible pairs (6 clusters) — the hard regime's distinctive question: does learning help *recognise* incompatibility?

**Power caveat, stated in advance:** six clusters give wide intervals. These
hypotheses are pre-registered, but a CI that includes 0 here is reported as
**underpowered**, not as falsified; only a CI that excludes 0 is a result.

| | Prediction | Rule |
|---|---|---|
| **B1** | bayes burns fewer proposals before no-deal: Δ proposals_used < 0 | CI excludes 0, negative |
| B2 | bayes walks (recognises) rather than times out: Δ walk rate > 0 | CI excludes 0, positive |
| B3 | invalid-acceptance attempts ≈ 0 in both arms | reported; any non-zero rate is flagged |
| **B4 (mechanism)** | the aid's own verdict falls under learning: Δ final_max_pacc_own_ir < 0 (max P(opp accepts) over deals IR for me, at the last decision) | CI excludes 0, negative |
| B5 | correct-no-deal rate | reported; expected ≈ 1.0 in both arms because the engine rejects non-IR acceptances — any deviation is an engine/logging alarm |

B4 is the mechanism behind B1/B2: prior-only cannot update, so its verdict is
constant; the Bayes filter should learn that nothing it wants is acceptable.

## Analysis (fixed in advance)

```bash
# A1–A3 (feasible, 30 clusters)
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
  results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --feasible-only --seed 3
# A4–A5 (feasible)
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_seller_asymmetry \
  results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --feasible-only --seed 3
# B1–B5 (infeasible, 6 clusters)
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_infeasible \
  results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --seed 3
```

No rule or bound changes after the data are seen; secondaries are reported in
full and none is promoted. Verdicts are applied by code and frozen before reading.

## Sanity gates (must pass or the comparison is not interpretable)

- The logs contain exactly 30 feasible and 6 infeasible ordered pairs per arm.
- 0 ideal offers in both arms.
- bayes mean log-score improvement > 0 on feasible pairs.
- Episode failure count reported.
- Correct-no-deal on infeasible pairs ≈ 1.0 in both arms (B5; an engine check).

## Predictions, stated to be falsifiable

A1: the quality effect persists (Δ Nash > 0) — the easy-regime effect did not
depend on the compromise deal, since the responder's counters were log-rolls,
not compromises. Honest alternative: with fewer IR deals the prior-only aid's
first-acceptable offers may already be near-efficient, shrinking Δ Nash toward 0.
A2: agreement falls below easy in both arms. B1/B2/B4: bayes recognises
incompatibility sooner. Honest alternative: six proposals may be too few for the
posterior to rule out every IR-for-me deal, so both arms time out and B is null.

## Limits accepted in advance

One run (no replication yet); one prompt; no candidate-order randomization;
β and ε unswept; six infeasible clusters; `jev-latest` pinned by name.
