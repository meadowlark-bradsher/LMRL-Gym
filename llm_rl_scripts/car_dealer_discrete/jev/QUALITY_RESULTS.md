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
distinguishable from zero at this scale. This asymmetry turned out to be the wrong cut: the exploratory section below
shows it is a **responder** effect (learning helps whoever moves second),
which pools to a slight seller tilt only through small role×start differences.

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

## Exploratory: the "seller asymmetry" is a responder effect

**Status: post hoc when written — since CONFIRMED in a pre-registered independent
replication (`RESPONDER_RESULTS.md`): P1–P3 held and the P4 validity gate passed.** Every number is rendered from
`results/jj_quality_20261025/exploratory_seller_asymmetry.json` (paired type-pair
cluster bootstrap, B = 5000, seed 0, 36 clusters). Regenerate:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_seller_asymmetry \
    results/jj_quality_20261025/bayes results/jj_quality_20261025/prior_only
```

The pre-registered analysis found Δ G_S resolved and Δ G_B not. Three lenses on
the same logs (no new API calls) show the role framing was the wrong cut.

### Lens 1 — starting side: learning helps whoever moves second

| Metric | bayes | prior-only | Δ | 95% CI | |
|---|---|---|---|---|---|
| G_B | start=buyer | 0.648 | 0.710 | **-0.062** | (-0.164, 0.042) | includes 0 |
| G_B | start=seller | 0.701 | 0.503 | **+0.198** | (0.096, 0.300) | excludes 0 ↑ |
| G_S | start=buyer | 0.748 | 0.530 | **+0.217** | (0.113, 0.324) | excludes 0 ↑ |
| G_S | start=seller | 0.679 | 0.692 | **-0.013** | (-0.089, 0.068) | includes 0 |
| G_starter (either role) | 0.664 | 0.701 | **-0.037** | (-0.091, 0.018) | includes 0 |
| G_responder (either role) | 0.724 | 0.517 | **+0.208** | (0.146, 0.272) | excludes 0 ↑ |

G_B rises only when the **seller** starts; G_S rises only when the **buyer**
starts. Role-agnostic: **Δ G_responder = +0.208 (0.146, 0.272); Δ G_starter = -0.037 (-0.091, 0.018)**,
flat. The pooled "seller" result is arithmetic: the start-split halves average
exactly to the pre-registered Δ G_S and Δ G_B (cross-checked above), and the
responder gain is slightly larger when the seller responds while the starter
penalty is slightly larger when the buyer starts — small role×start differences
that nudged G_S over the CI line and left G_B just short. The role difference
itself is not resolved.

### Lens 2 — mechanism: the responder counters instead of accepting

| Metric | bayes | prior-only | Δ | 95% CI | |
|---|---|---|---|---|---|
| starter's proposal was the accepted one (rate) | 0.387 | 0.711 | **-0.324** | (-0.408, -0.238) | excludes 0 ↓ |
| seller's proposal was the accepted one (rate) | 0.460 | 0.450 | **+0.010** | (-0.106, 0.128) | includes 0 |
| offers per episode: buyer | 1.130 | 0.892 | **+0.238** | (0.052, 0.434) | excludes 0 ↑ |
| offers per episode: seller | 1.090 | 0.839 | **+0.252** | (0.059, 0.434) | excludes 0 ↑ |
| split at agreement U_S - U_B | 0.004 | -0.007 | **+0.012** | (-0.033, 0.057) | includes 0 |

In prior-only the starter's proposal is the one accepted **0.711** of the
time — a proposer advantage: the responder takes the first acceptable offer. With
learning that falls to **0.387** (Δ -0.324 (-0.408, -0.238)). Both sides make more
offers; whose proposal closes is role-symmetric; the surplus split is ≈ 0 in both arms.

### Lens 2b — built-in placebo: why we can trust the mechanism

The **starter's** first offer is made under the uniform prior in *both* arms — the
annotated P(accept) numbers it sees are identical — so its Δ must be ≈ 0. The
**responder's** first offer is the only decision that differs between arms: it
has seen exactly one observation, the starter's opening concession.

| Metric | bayes | prior-only | Δ | 95% CI | |
|---|---|---|---|---|---|
| STARTER first offer: own utility  (placebo: expect Δ≈0) | 0.777 | 0.775 | **+0.002** | (-0.006, 0.011) | includes 0 |
| STARTER first offer: utility to recipient (placebo) | 0.493 | 0.490 | **+0.003** | (-0.004, 0.009) | includes 0 |
| STARTER first offer: aid P(accept) shown (placebo) | 0.638 | 0.641 | **-0.003** | (-0.010, 0.004) | includes 0 |
| STARTER first offer acceptable to recipient (placebo) | 0.658 | 0.649 | **+0.009** | (-0.009, 0.028) | includes 0 |
| RESPONDER first offer: own utility | 0.699 | 0.598 | **+0.101** | (0.045, 0.164) | excludes 0 ↑ |
| RESPONDER first offer: utility to recipient | 0.605 | 0.467 | **+0.138** | (0.099, 0.179) | excludes 0 ↑ |
| RESPONDER first offer: aid P(accept) shown | 0.883 | 0.831 | **+0.052** | (0.013, 0.089) | excludes 0 ↑ |
| RESPONDER first offer acceptable to recipient | 0.979 | 0.721 | **+0.258** | (0.138, 0.386) | excludes 0 ↑ |

The placebo passes on all four measures (tight CIs straddling zero). The
responder's first offer, after one observation, is better **for itself**
(+0.101) **and for the recipient** (+0.138), and is acceptable to the recipient
**0.979 vs 0.721** of the time. One concession reveals which attribute the
starter values least; the posterior sharpens; the annotated menu surfaces the
log-roll; the counter is a mutual improvement. This is the information doing
the work, and the starter row shows what happens when it is absent.

### Lens 3 — learning is symmetric by role

| Direction | bits (bayes) | 95% CI |
|---|---|---|
| seller learns buyer | 0.905 | (0.740, 1.072) |
| buyer learns seller | 0.797 | (0.495, 1.055) |
| asymmetry (seller − buyer) | +0.107 | (-0.204, 0.457) |
| argmax-correct: seller→buyer / buyer→seller | 0.432 / 0.516 | |

No resolved directional asymmetry; entropies match. This rules out "sellers read
buyers better" as the explanation.

### Pair level

Of 36 pairs: joint gain 12, transfer to seller 11, transfer to
buyer 10, joint loss 1, flat 2. Transfers run both ways
equally; a third of pairs gain jointly. No seller tilt at the pair level.

### The chain, as a hypothesis to pre-register

Starter concedes on its least-valued attribute → the responder's posterior
sharpens on one observation → the responder's annotated options surface a
mutual-improvement counter → the responder counters instead of accepting
(proposer-acceptance 0.711 → 0.387) → the responder captures the gain
(+0.21), the starter loses its proposer premium but faces a better counter (≈ flat)
→ joint surplus rises (the pre-registered Δ Nash). **Predictions for a fresh
seed:** Δ G_responder > 0 with CI excluding 0; Δ G_starter ≈ 0; the
starter-proposal-accepted rate falls; the starter-first-offer placebo stays ≈ 0.

## What this changes in the story

- Agreement is unlocked by the **prior** forward model (mechanism controls).
- Quality — surplus, balance, efficiency — is improved by **learning**, at the
  cost of proposals, with the gain landing on the **responder** (confirmed in replication).
- For the BED question: EIG-style opponent inference is aimed at a margin that is
  **now shown to be real** at this scale, not merely directional.

## Limits

- *Replicated:* an independent run at the same scale (`RESPONDER_RESULTS.md`)
  reproduced Δ Nash and every secondary with near-identical estimates. Numbers
  live there, not here.
- Easy regime here. The hard regime (r = 0.55) has since been run under its own
  pre-registration (`HARD_RESULTS.md`): the quality effect persists, smaller, and
  agreement becomes the main lever.
- One prompt; no candidate-order randomization; β and ε not swept; the IR
  threshold response model is deliberately simple and optimistic.
- The seller asymmetry (finding 2) is a responder effect, confirmed in a
  pre-registered independent replication (`RESPONDER_RESULTS.md`), where the
  pooled role gains come out equal.
- `jev-latest` is pinned by name, not version.
