# Pre-registration: does Bayesian learning improve deal quality? (bayes vs prior-only)

**Registered:** 2026-10-25, before any episode of this run was analyzed.
**Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Regime:** easy (r = 0.40)

## Question

The mechanism controls (`AID_RESULTS.md` §Mechanism controls) showed that the
prior-only forward model is sufficient for *agreement* (12/12, same as the full
Bayes filter). The pilot hinted that learning buys *quality* — Nash 0.808 vs
0.734, minG 0.666 vs 0.619 — at the cost of more proposals (2.25 vs 1.17). At
K = 2 on 3 pairs those gaps were explicitly labelled suggestive. This run asks
whether they are real.

## Design

- Conditions: `--aid --aid-mode bayes` vs `--aid --aid-mode prior_only`.
  Everything else identical: β = 8, ε = 0.05, own-utility revealed, no salience,
  no option shuffling, 6-proposal budget, same instruction text as the pilot.
- Type pairs: **all 36** ordered (buyer, seller) rankings. Unit of inference =
  type pair (cluster).
- Starting side: both (counterbalanced).
- Repeats: **K = 8** per (pair, start) cell → 576 episodes per condition.
- Runner: hardened (streamed records; per-episode retry ×3 with backoff; failures
  recorded as error stubs, never dropped silently). The failure count is reported.

```bash
# bayes
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
  --pairs all --regime easy --start both --repeats 8 --aid --aid-mode bayes \
  --out results/jj_quality_20261025/bayes
# prior-only
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
  --pairs all --regime easy --start both --repeats 8 --aid --aid-mode prior_only \
  --out results/jj_quality_20261025/prior_only
```

## Outcomes (fixed in advance)

**Primary:** Δ Nash ratio `Q_N` = mean(bayes) − mean(prior-only), over all
episodes (no-deal counts as 0, per spec §4.2).

**Secondary:** Δ min(G_B, G_S); Δ G_B; Δ G_S; Δ Pareto-frontier rate (over valid
agreements); Δ proposals-to-agreement (over valid agreements); Δ agreement rate.

**Epistemic (bayes only, descriptive):** log-score improvement over the uniform
prior; final entropy; argmax-correct rate.

## Analysis (fixed in advance)

Paired, type-pair-clustered bootstrap: resample the 36 pairs with replacement
(B = 5000, `random.Random(0)`), pool *both* conditions' episodes for the
resampled pairs, recompute each Δ; report the 95% percentile interval. Also
per-condition cluster-bootstrap means, the per-pair Δ `Q_N`, and the sign count
(pairs with Δ > 0 / < 0 / = 0).

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
  results/jj_quality_20261025/bayes results/jj_quality_20261025/prior_only
```

## Decision rules

- **Learning improves quality** if the primary 95% CI for Δ `Q_N` excludes 0 on
  the positive side.
- **No resolved benefit** if the CI includes 0. We will then report the interval
  as the bound on the effect, not search for a secondary that "works".
- Secondaries are reported in full regardless; none is promoted to primary after
  the fact.

## Sanity checks (must pass or the comparison is not interpretable)

- Agreement rate ≥ 0.95 in both conditions (the quality question presupposes
  agreement; a drop would be a finding in its own right and reported first).
- 0 ideal offers in both conditions (the aid is being used).
- bayes mean log-score improvement > 0 (learning actually occurred at scale).
- Episode failure count reported; analysis runs on successful episodes only and
  states how many were excluded.

## Prediction (stated to be falsifiable)

From the pilot: Δ `Q_N` ≈ +0.07, Δ minG ≈ +0.05, Δ proposals ≈ +1.1. Honest
prior: the pilot gaps were K = 2 noise and Δ `Q_N` ≈ 0. Both are live.

## Limits accepted in advance

Easy regime only; one prompt; no candidate-order randomization; β and ε not
swept; `jev-latest` pinned by name, not version. Not a claim about the hard
regime (r = 0.55), where the universal compromise is not individually rational.
