# Does learning improve deal quality? bayes vs prior-only at scale — results

**Date:** 2026-10-25 · **Pre-registered:** `PREREGISTRATION_QUALITY.md` (before analysis) · **Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Regime:** easy (r = 0.40)

Every number in this document is **rendered from** the frozen analysis
`results/jj_quality_20261025/analysis_quality.json` — nothing here was typed by
hand. Regenerate the analysis with the pre-registered command:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
    results/jj_quality_20261025/bayes results/jj_quality_20261025/prior_only
```

All quantities below are **stochastic** (draws from live `jev-latest`) and are
therefore reported with paired, type-pair-clustered bootstrap 95% intervals
(B = 5000, seed 0, unit = ordered type pair, 36 clusters),
not as point values.

## Design (as pre-registered)

All 36 ordered type pairs × both starting sides × K = 8 repeats per cell →
576 episodes (bayes) and 576 episodes (prior-only). Identical prompts, β = 8,
ε = 0.05, own-utility revealed, no salience, no option shuffling. The runner
streamed records and retried failures; **0 + 0 episodes failed** (none excluded).

## Sanity gates (pre-registered; all must pass)

| Gate | bayes | prior-only | |
|---|---|---|---|
| Agreement rate ≥ 0.95 | 1.0000 | 0.9792 | pass / pass |
| Zero ideal offers (aid in use) | 0/1279 | 0/997 | pass / pass |
| bayes learning > 0 bits | +0.851 mean over 1152 seats | — | pass |
| Failures reported | 0 | 0 | pass |

## Results — Δ = bayes − prior-only

| Metric | bayes | prior-only | Δ | 95% CI (cluster bootstrap) | |
|---|---|---|---|---|---|
| Nash ratio Q_N  **(PRIMARY)** | 0.759 | 0.601 | **+0.158** | (0.054, 0.263) | excludes 0 ↑ |
| min(G_B, G_S) | 0.557 | 0.448 | **+0.109** | (0.033, 0.190) | excludes 0 ↑ |
| G_B (buyer gain) | 0.674 | 0.606 | **+0.068** | (-0.017, 0.151) | includes 0 |
| G_S (seller gain) | 0.714 | 0.611 | **+0.102** | (0.022, 0.178) | excludes 0 ↑ |
| Pareto-frontier rate (valid agreements) | 0.929 | 0.826 | **+0.103** | (0.025, 0.179) | excludes 0 ↑ |
| Proposals to agreement (valid agreements) | 2.220 | 1.640 | **+0.580** | (0.252, 0.908) | excludes 0 ↑ |
| Agreement rate | 1.000 | 0.979 | **+0.021** | (0.005, 0.042) | excludes 0 ↑ |

**Pre-registered decision rule → learning improves quality (primary CI excludes 0, positive).**

Per-pair Δ Nash sign count: **+24 / −10 / 0: 2** of 36 pairs.

## Interpretation

**1. The primary resolves positive.** Δ Nash = +0.158 with CI (0.054, 0.263)
excluding zero. The pilot's K = 2 gap (+0.07, labelled suggestive) was real and
understated. Bayesian opponent inference buys genuinely more joint surplus than
the prior-averaged forward model — on top of the agreement the prior already
secures.

**2. The gain is in balance and efficiency, and it lands on the seller.**
min(G_B, G_S) and the Pareto rate both rise with CIs excluding zero: deals are
more balanced and more often efficient. Decomposed by side, **Δ G_S resolves
positive while Δ G_B does not** (its CI spans zero). Learning is helping the
seller realise more of its attainable gain; the buyer's change is not
distinguishable from zero at this scale. This asymmetry is **recorded, not
explained** — any mechanism offered now would be post hoc. It is the first thing
to investigate.

**3. Learning costs time.** Proposals to agreement rise by +0.580
(2.220 vs 1.640). Prior-only closes faster on the first acceptable
offer; the Bayes filter negotiates longer and lands better. The pilot saw the
same signature.

**4. Learning also nudges agreement at scale.** Prior-only lost a small but real
share of deals (0.9792); bayes lost none (1.0000), Δ CI excluding zero. The
pilot's 12/12 in both arms could not see this; 576 episodes can.

**5. The effect is heterogeneous across type pairs.** Learning raised Nash on
24 of 36 pairs and *lowered* it on 10. The pooled
CI is positive, but "learning helps" is a population statement, not a per-pair
guarantee. The per-pair deltas are in the frozen JSON (`per_pair_primary_delta`).

**6. Learning is real and imperfect — the §4.3 hazard observed.** Mean
improvement +0.851 bits; posterior argmax equals the true type 47.4%
of the time (chance 16.7%). But the range runs to **-4.94 bits**: at least
one seat finished very confidently wrong. The ε = 0.05 noise floor kept it from
collapsing to zero, as designed, but did not prevent it. A confidently-wrong
posterior is a live failure mode of this filter, not a theoretical one.

## What this changes in the story

- Agreement is unlocked by the **prior** forward model (mechanism controls).
- Quality — surplus, balance, efficiency — is improved by **learning**, at the
  cost of proposals, with the gain concentrated on the seller.
- For the BED question: EIG-style opponent inference is aimed at a margin that is
  **now shown to be real** at this scale, not merely directional.

## Limits

- One run; not independently replicated. The CI is cluster-bootstrapped over 36
  pairs, which is far more robust than the K = 2 pilot, but the Text-Nav program
  showed that single runs of `jev-latest` can be unrepresentative; a replication
  with a fresh seed would strengthen every claim here.
- Easy regime only. The hard regime (r = 0.55) removes the universal 0.5
  compromise and should sharpen the prior's discriminative power; untested.
- One prompt; no candidate-order randomization; β and ε not swept; the IR
  threshold response model is deliberately simple and optimistic.
- The seller asymmetry (finding 2) is unexplained.
- `jev-latest` is pinned by name, not version.
