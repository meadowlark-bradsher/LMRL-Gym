# The responder effect — pre-registered replication results

**Date:** 2026-10-25 · **Pre-registration:** `PREREGISTRATION_RESPONDER.md` (written 22:54:09Z, before any episode) · **Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Regime:** easy (r = 0.40)

Every number here is **rendered from frozen JSON** — nothing typed by hand:
`results/jj_quality_rep1_20261025/{confirmatory_responder.json, confirmatory_responder_verdicts.json, analysis_quality.json}`
and, for the side-by-side, run 1's `results/jj_quality_20261025/{exploratory_seller_asymmetry.json, analysis_quality.json}`.
Verdicts were applied **by code** against the pre-registered rules and frozen before being read. Regenerate:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_seller_asymmetry \
    results/jj_quality_rep1_20261025/bayes results/jj_quality_rep1_20261025/prior_only --seed 1
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
    results/jj_quality_rep1_20261025/bayes results/jj_quality_rep1_20261025/prior_only --seed 1
```

All Δ are bayes − prior-only with paired type-pair-clustered bootstrap 95% intervals
(B = 5000, seed 1, 36 clusters). "Run 1" is the exploratory run that
generated the hypothesis; "Replication" is this independent run. Independence comes from
`jev-latest` being stochastic server-side, not from `--seed` (see the pre-registration).

## Sanity gates (pre-registered; all passed)

| Gate | bayes | prior-only |
|---|---|---|
| Episodes attempted / failed | 576 / 0 | 576 / 0 |
| Agreement rate ≥ 0.95 | 0.9983 | 0.9844 |
| Ideal offers (must be 0) | 0/1299 | 0/988 |
| bayes learning > 0 bits | +0.831 over 1152 seats | — |

## Verdicts on the pre-registered predictions

| | Prediction | Rule | Run 1 (exploratory) | **Replication** | Verdict |
|---|---|---|---|---|---|
| **P4** validity gate | starter first-offer placebo ≈ 0 | all 4 STARTER Δ CIs include 0 | all ≈ 0 | all ≈ 0 (table below) | **PASS** |
| **P1** primary | Δ G_responder > 0 | CI excludes 0, positive | +0.208 [+0.146, +0.272] | **+0.200 [+0.138, +0.266]** | **CONFIRMED** |
| **P2** | Δ G_starter ≈ 0 | CI includes 0 and \|Δ\| ≤ 0.10 | -0.037 [-0.091, +0.018] | **-0.041 [-0.097, +0.018]** | **SUPPORTED** |
| **P3** | starter-proposal-accepted rate falls | Δ < 0, CI excludes 0 | -0.324 [-0.408, -0.238] | **-0.309 [-0.396, -0.218]** | **CONFIRMED** |
| Replication of original primary | Δ Nash Q_N > 0 | CI excludes 0 | +0.158 [+0.054, +0.263] | **+0.154 [+0.053, +0.258]** | **CONFIRMED** |

### P4 — the placebo, in full (starter sees identical information in both arms)

| STARTER first offer | Run 1 Δ | Replication Δ |
|---|---|---|
| own utility  | +0.002 [-0.006, +0.011] | -0.000 [-0.005, +0.004] |
| utility to recipient | +0.003 [-0.004, +0.009] | +0.004 [-0.002, +0.011] |
| aid P(accept) shown | -0.003 [-0.010, +0.004] | -0.002 [-0.007, +0.003] |
| acceptable to recipient | +0.009 [-0.009, +0.028] | +0.000 [-0.010, +0.010] |

### Mechanism corollary (reported, not a decision) — the responder's first offer after one observation

| RESPONDER first offer | Run 1 Δ | Replication Δ |
|---|---|---|
| own utility | +0.101 [+0.045, +0.164] | +0.101 [+0.044, +0.167] |
| utility to recipient | +0.138 [+0.099, +0.179] | +0.142 [+0.098, +0.190] |
| acceptable to recipient (rate) | +0.258 [+0.138, +0.386] | +0.276 [+0.136, +0.410] |

Better for itself *and* for the recipient, in both runs: one observed concession is
enough for the posterior to surface the log-roll.

## The "seller asymmetry" is gone

| By starting side | Run 1 Δ | Replication Δ |
|---|---|---|
| G_B \| seller starts (buyer responds) | +0.198 [+0.096, +0.300] | +0.200 [+0.103, +0.300] |
| G_S \| buyer starts (seller responds) | +0.217 [+0.113, +0.324] | +0.200 [+0.102, +0.305] |
| G_B \| buyer starts (buyer is starter) | -0.062 [-0.164, +0.042] | -0.041 [-0.131, +0.053] |
| G_S \| seller starts (seller is starter) | -0.013 [-0.089, +0.068] | -0.042 [-0.124, +0.042] |

In the replication the two responder gains are **+0.1995** and **+0.2004** and the
two starter penalties **-0.0407** and **-0.0420** — symmetric to three decimals.
Pooled by role, Δ G_B = +0.0794 [+0.001, +0.156] and
Δ G_S = +0.0792 [+0.004, +0.158]: **both resolved, and equal.**
Run 1's "seller asymmetry" (Δ G_S resolved, Δ G_B not) was sampling noise on a
symmetric responder effect — exactly what the exploratory analysis claimed, now
shown by an independent draw rather than argued from the same data.

## Everything else replicates too

| Δ (bayes − prior-only) | Run 1 | Replication |
|---|---|---|
| min(G_B, G_S) | +0.109 [+0.033, +0.190] | +0.108 [+0.033, +0.187] |
| Pareto-frontier rate | +0.103 [+0.025, +0.179] | +0.107 [+0.025, +0.194] |
| Proposals to agreement | +0.580 [+0.252, +0.908] | +0.610 [+0.265, +0.957] |
| Agreement rate | +0.021 [+0.005, +0.042] | +0.014 [+0.002, +0.033] |
| Per-pair Δ Nash sign (+/−/0) | 24/10/2 | 23/12/1 |
| Learning asymmetry, seller→buyer − buyer→seller (bits) | +0.107 [-0.204, +0.457] | +0.108 [-0.187, +0.415] |
| Pair classification (joint / →seller / →buyer / loss / flat) | 12 / 11 / 10 / 1 / 2 | 10 / 11 / 12 / 2 / 1 |

Learning remains symmetric by role and imperfect: the bayes arm's worst seat again
finished at -4.94 bits (the same floor as run 1 — a structural minimum of the
likelihood under ε = 0.05, not a coincidence worth over-reading, but a reminder
that a confidently-wrong posterior is a live failure mode).

## Pooled two-run estimate (labelled secondary, per the pre-registration)

Both runs concatenated: 1152 episodes per arm, the same 36 type-pair clusters,
B = 5000, seed 2. Frozen at `results/jj_quality_pooled_20261025/{analysis_quality_pooled.json, responder_pooled.json}`.
This **never substitutes** for the replication-alone verdicts above; it is reported for precision.

| Δ (bayes − prior-only) | Pooled |
|---|---|
| Nash Q_N (pre-reg 1 primary) | +0.156 [+0.054, +0.259] |
| G_responder (pre-reg 2 P1) | +0.204 [+0.140, +0.269] |
| G_starter (P2) | -0.039 [-0.093, +0.017] |
| starter-proposal-accepted rate (P3) | -0.317 [-0.401, -0.230] |
| min(G_B, G_S) | +0.109 [+0.034, +0.188] |
| Pareto-frontier rate | +0.105 [+0.028, +0.186] |
| Proposals to agreement | +0.595 [+0.274, +0.914] |
| Per-pair Δ Nash sign (+/−/0) | 23/12/1 |

Two things the pooled numbers say that should not be smoothed over:

**Pooled *by role*, run 1's tilt resurfaces.** Δ G_S = +0.091 [+0.015, +0.164] resolves, while
Δ G_B = +0.074 [-0.005, +0.153] just touches zero — because pooling averages run 1's noisy
seller tilt with the replication's symmetry. This is not evidence for a role
effect; it is what averaging a noisy draw with a clean one looks like. The
start-split framing (responder vs starter) is the stable description in every
run and in the pool, which is exactly why it was pre-registered.

**Pooling barely tightened the intervals.** The Δ Nash CI width is 0.204 pooled versus
0.208 (run 1) and 0.205 (replication). Doubling the episodes did
almost nothing, because the bootstrap unit is the type pair and there are only
36 of them — between-pair heterogeneity sets the floor, and 36 is the
*entire* type space. For pair-clustered inference this design is near its
precision ceiling: more repeats per pair will not narrow these intervals. Sharper
estimates would need a different design (e.g. a richer type space), not more K.

## What is now established

1. **Learning improves deal quality** (Δ Nash > 0): pre-registered, and replicated
   in an independent run with a near-identical estimate.
2. **The gain goes to whoever moves second**, not to a role: pre-registered as P1–P3
   after an exploratory finding, and confirmed with P4 as a passing validity gate.
3. **Mechanism:** the responder, after one observed concession, counters with a
   deal better for both sides instead of accepting the starter's offer; the
   starter loses its proposer premium but faces a better counter and lands flat.
4. **For the BED question:** the margin that opponent inference buys is real,
   replicated, and has a specific locus — information advantage of the second
   mover. In this protocol, the first mover has no information to act on.

## Limits

Two independent runs, both easy regime, one prompt, no candidate-order
randomization, β and ε unswept, `jev-latest` pinned by name. The hard regime (r = 0.55) is reported in `HARD_RESULTS.md`, where the responder
effect holds (P1–P3) but a placebo-gate false positive and a seller-as-responder
asymmetry are recorded. The starter penalty is "≈ 0" by a magnitude bound, not
shown to be exactly zero.
