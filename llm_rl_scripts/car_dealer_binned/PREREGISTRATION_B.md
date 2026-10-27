# Pre-registration: Jev on Binned Car Dealer, Condition B

**Registered:** 2026-10-26, before any Jev decision was collected on this game.
**Game:** `binned-car-dealer-b-v0.1` · **Model:** `jev-latest` · **Regime:** easy (r = 0.40) ·
**Opponent:** frozen scripted (truthful, deterministic IR acceptance, no noise).
**Design:** [`DESIGN_B.md`](DESIGN_B.md). **Scripted baselines already frozen:** [`SCRIPTED_RESULTS.md`](SCRIPTED_RESULTS.md).

## What is already settled without Jev

The document's **H2** (greedy EIG can underperform the utility-optimal policy in Condition B) is a
property of the game, decided exactly by the scripted arms: it holds at the 2-round horizon and
fails (greedy = exact) at horizons ≥ 3 in the easy regime. It is reported in `SCRIPTED_RESULTS.md`
and is **not** re-tested here. This note is about Jev.

## Conditions

Two Jev arms, identical code path, one annotation apart:

- `jev_plain` — lettered options in random order; offers show own utility; nothing about information.
- `jev_eig`   — the same, plus each option's exact expected information gain in bits.

Neither shows P(accept), the SAFE/AMBITIOUS tags, or the posterior.

**Primary horizon: 2 rounds + final offer (3 decisions max)** — the horizon at which the
ask-vs-act trade-off binds. **Secondary horizon: 6 rounds + final (7 decisions max)** — the
document's parameterisation, where information is free and regret can come only from closing
too early or on the wrong deal.

All 72 cells (2 roles × 6 own types × 6 opponent types), K repeats per cell, seeds as provenance
(`jev-latest` is stochastic server-side; `--seed` fixes only the option shuffle and the cell order).

| Run | rounds | K | episodes / arm | max Jev calls / arm |
|---|---|---|---|---|
| primary   | 2 | 5 | 360 | 1080 |
| secondary | 6 | 3 | 216 | 1512 |

```bash
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_binned.run_condition_b \
  --arms jev_plain,jev_eig,random --rounds 2 --regime easy --repeats 5 --seed 11 \
  --jev-backend http --out results/binned_b_jev_r2_easy_<date>
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_binned.run_condition_b \
  --arms jev_plain,jev_eig,random --rounds 6 --regime easy --repeats 3 --seed 12 \
  --jev-backend http --out results/binned_b_jev_r6_easy_<date>
```

The `exact`, `greedy_eig` and `myopic` arms are deterministic and already frozen in
`results/binned_b_scripted_r2_easy_20261026` and `..._r6_easy_20261026`; the analyzer pairs Jev
cells with those by (role, own type, opponent type). `random` is re-drawn inside each Jev run so
its comparison shares the run.

## Hypotheses and decision rules (fixed in advance)

Unit of inference: the cell (role, own type, opponent type); 72 clusters. All Δ use the paired,
cluster-resampled bootstrap in `analyze_condition_b.py`, B = 5000, seed = the run's `--seed`.
"Confirmed" = the 95 % CI excludes 0 on the predicted side; "not supported" otherwise; the
opposite sign excluding 0 is reported as such. **One primary; everything else is secondary and
carries no multiplicity correction, so a single secondary CI touching 0 is not a failure of the
program** (the lesson recorded in `car_dealer_discrete/README.md`).

| | Quantity (primary horizon unless stated) | Prediction | Rule |
|---|---|---|---|
| **P1 (primary)** | Δ total decision regret, `jev_plain` − `random` | < 0 (Jev recognises when to act better than chance) | CI excludes 0, negative → confirmed |
| S1 | Δ per-decision EIG regret, `jev_eig` − `jev_plain` | < 0 (showing EIG lowers EIG regret) | CI |
| S2 | Δ total decision regret, `jev_eig` − `jev_plain` | **not** < 0 at the binding horizon (information-chasing); two-sided | CI reported; "confirmed" if CI includes 0 or is positive |
| S3 | Δ questions asked, `jev_eig` − `jev_plain` | > 0 | CI |
| S4 (H3) | Spearman ρ(choice entropy, decision regret), pooled Jev decisions, within terciles of the Q gap | ρ > 0 in the large-gap tercile; ρ ≈ 0 in the small-gap tercile | cluster-bootstrap CI on ρ per tercile; pattern "supported" if large-gap CI > 0 and small-gap CI includes 0 |
| S5 | Δ learner utility, each Jev arm − `exact` | < 0 | CI; the magnitude is the headline "optimality gap" |
| S6 (secondary horizon) | Δ total decision regret, `jev_plain` − `random`, at 6 rounds | < 0 | CI |
| descriptive | chosen-position distribution (positional bias), well-formed share, confidence, deal distribution, SAFE vs AMBITIOUS take-up | reported, no rule |

## Sanity gates (fail → fix the pipeline, do not read results)

- `well_formed_share` = 1.0 for both Jev arms (every answer maps to a lettered option).
- The letter → option map is uniform over positions across decisions (shuffle active).
- Every cell populated for both arms with exactly K episodes; no error stubs.
- The `random` arm inside the Jev run matches the frozen scripted `random` arm's mean utility
  within its own bootstrap CI (same game, same menu rule).

## Predictions, stated plainly

From the earlier Car-Dealer program, Jev has no belief state and no forward model, but here the
menu already filters offers to two sensible candidates. Prediction: P1 confirmed but small; S1
confirmed (Jev reads numbers it is given); S2 confirmed in the "not lower" sense (EIG shown →
more asking → no better decisions at the binding horizon); S4 the tercile pattern. Honest
alternative: Jev's choice is near-uniform over the lettered menu and P1 is null, in which case
the record says so and the Jev arm is reported as chance-level on this game.

## Analysis (fixed in advance)

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b \
  results/binned_b_jev_r2_easy_<date> --seed 11 --B 5000 --json results/binned_b_jev_r2_easy_<date>/analysis.json
```

Verdicts applied by code and frozen before reading. No rule changes after data.

## Limits accepted in advance

Frozen opponent (single-agent POMDP, not the bilateral game); one question bank; one menu rule;
easy regime only; zero noise; `jev-latest` pinned by name; 72 cells is the whole type space, so
cluster intervals have a precision ceiling that more K cannot lower.
