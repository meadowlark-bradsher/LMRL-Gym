# Hard regime (r = 0.55): bayes vs prior-only — pre-registered results

**Date:** 2026-10-25 · **Pre-registration:** `PREREGISTRATION_HARD.md` (written 23:08:06Z, before any episode) · **Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest`

Every number is **rendered from frozen JSON** in `results/jj_hard_20261025/`:
`analysis_quality_feasible.json`, `responder_feasible.json`, `infeasible.json`, their `*_verdicts.json`
(rules applied by code before reading), and `a5_gate_diagnosis.json`. Regenerate:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
    results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --feasible-only --seed 3
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_seller_asymmetry \
    results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --feasible-only --seed 3
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_infeasible \
    results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --seed 3
```

Δ = bayes − prior-only; paired type-pair-clustered bootstrap 95% intervals (B = 5000, seed 3).

## Structure

30 / 36 ordered pairs admit an IR deal; **6 / 36 admit none** — and they are exactly the six
**identical-ranking** pairs (delivery>price>warranty / delivery>price>warranty; delivery>warranty>price / delivery>warranty>price; price>delivery>warranty / price>delivery>warranty; price>warranty>delivery / price>warranty>delivery; warranty>delivery>price / warranty>delivery>price; warranty>price>delivery / warranty>price>delivery): with complementary preferences and the same priority
order, both sides want the top attribute at the other's expense and nothing clears r = 0.55.
The universal compromise `P1-W1-D1` (0.5) is not IR. Feasible pairs are analysed with
`--feasible-only` (30 clusters); the infeasible six separately (6 clusters).

## Sanity gates (pre-registered; all passed in both arms)

| Gate | bayes | prior-only |
|---|---|---|
| Episodes attempted / failed | 576 / 0 | 576 / 0 |
| Feasibility split in logs | 30 pairs (480) + 6 pairs (96) | 30 pairs (480) + 6 pairs (96) |
| Ideal offers (must be 0) | 0/580 | 0/583 |
| bayes learning > 0 on feasible | +0.841 bits | — |
| B5 correct-no-deal on infeasible | 1.000 | 1.000 |
| B3 invalid-acceptance attempts | 0.000 | 0.000 |

## A. Feasible pairs (30 clusters)

| | Prediction | bayes | prior-only | Δ | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| **A1 primary** | Δ Nash Q_N > 0 | 0.709 | 0.608 | **+0.102** | [+0.029, +0.183] | **CONFIRMED** |
| A2 | agreement (reported) | 0.838 | 0.746 | +0.092 | [+0.021, +0.177] | excludes 0 ↑ |
| A3 | min(G_B, G_S) | 0.456 | 0.369 | +0.086 | [+0.027, +0.155] | excludes 0 ↑ |
| A3 | Pareto rate | 1.000 | 0.989 | +0.011 | [+0.000, +0.036] | includes 0 (lower bound exactly 0; saturated) |
| A3 | proposals to agreement | 1.249 | 1.050 | +0.199 | [+0.099, +0.305] | excludes 0 ↑ |
| A3 | G_B | 0.606 | 0.548 | +0.058 | [-0.004, +0.131] | includes 0 |
| A3 | G_S | 0.634 | 0.515 | +0.119 | [+0.042, +0.205] | excludes 0 ↑ |

Per-pair Δ Nash sign: +15 / −4 / 0: 11 (11 pairs tie exactly — pairs with
effectively one IR deal land on it in both arms). Learning: +0.841 bits, argmax-correct 0.338.

**The quality effect survives the loss of the safe compromise** (A1 confirmed), at roughly
two-thirds the easy-regime size. Two things changed character: **agreement is now a lever**
(both arms fall well below easy, and learning recovers +0.092 of it with a CI excluding 0),
and **deals close fast in both arms** (1.25 vs 1.05 proposals, against 2.2 / 1.6 in easy) — the
discriminating prior leaves few IR deals to search. Pareto is saturated (bayes 1.000); its Δ
lower bound is exactly 0 and is reported as borderline, not as a result.

## A5 validity gate — FAILED by the pre-registered rule, diagnosed as a false positive

| STARTER first offer (identical information in both arms) | Δ | 95% CI | includes 0? |
|---|---|---|---|
| own utility  | +0.0042 | [-0.0004, +0.0090] | yes |
| utility to recipient | -0.0048 | [-0.0146, +0.0046] | yes |
| aid P(accept) shown | -0.0069 | [-0.0134, -0.0009] | **no** |
| acceptable to recipient | +0.0042 | [-0.0083, +0.0167] | yes |

Three of four include 0; "aid P(accept) shown" excludes it by 0.0009. The rule says:
suspend interpretation of A4 pending diagnosis. **Diagnosis (frozen, `a5_gate_diagnosis.json`):**

- **No pipeline leak.** At the starter's first decision the aid showed exactly **one** acceptance
  dict per role in each arm ({'bayes/buyer': 1, 'bayes/seller': 1, 'prior_only/buyer': 1, 'prior_only/seller': 1}), **identical across arms**
  ({'buyer': True, 'seller': True}), with a uniform belief. The information was byte-for-byte the same.
- **The Δ is which deal Jev chose under identical information.** Both arms chose from the same
  eight deals with near-identical counts (bayes top: P2-W2-D0×101, P0-W0-D0×80, P0-W0-D2×78, P1-W0-D2×69; prior-only:
  P2-W2-D0×94, P0-W0-D0×80, P0-W0-D2×76, P2-W2-D2×75), moving the mean shown P(accept) from
  0.5894 to 0.5825 — a 0.007 shift. That is sampling noise.
- **Multiplicity.** 12 placebo tests across three runs, 1 marginal exclusion: the ≈0.6 false positives
  the null predicts. A per-run Bonferroni interval would include 0.

**Methodological finding against our own design:** a literal "all four CIs include 0" gate has a
**18.6% false-fail rate per run under the null**. It failed here as designed, on noise. The next
pre-registration will carry a multiplicity correction or a magnitude bound (as P2 does). The
gate's purpose — detecting a leak — was served by the diagnosis, which is now part of the record.

## A4. Responder effect on feasible pairs — interpreted with the gate caveat above

| | Rule | Δ | 95% CI | Verdict |
|---|---|---|---|---|
| P1 Δ G_responder > 0 | CI excludes 0 | **+0.119** | [+0.044, +0.199] | **CONFIRMED** |
| P2 Δ G_starter ≈ 0 | CI incl 0, \|Δ\| ≤ 0.10 | +0.058 | [-0.013, +0.138] | **SUPPORTED** |
| P3 starter-proposal-accepted falls | CI excludes 0, negative | -0.169 (0.950 → 0.781) | [-0.293, -0.055] | **CONFIRMED** |

Corollary — the responder's first offer after one observation: Δ own utility +0.047 [-0.027, +0.118]
(not resolved here), Δ utility to recipient **+0.236 [+0.099, +0.356]**, Δ acceptable-to-recipient
**+0.714 [+0.366, +0.982]**. In the hard regime the uninformed responder's counter is almost
never acceptable; the informed one's usually is. The mechanism is intact; its own-utility
component did not resolve at this scale.

| By starting side | Δ | 95% CI |
|---|---|---|
| G_S \| buyer starts (seller responds) | **+0.199** | [+0.082, +0.333] |
| G_B \| seller starts (buyer responds) | +0.039 | [-0.038, +0.130] |
| G_B \| buyer starts (buyer is starter) | +0.076 | [-0.011, +0.175] |
| G_S \| seller starts (seller is starter) | +0.039 | [-0.046, +0.134] |

**An asymmetry the easy regime did not show:** the responder gain is carried by the
*seller-as-responder* cell; the buyer-as-responder cell is positive but does not resolve. The
exploratory section below locates it: the two cells are indistinguishable under learning; it is the
**prior-only baseline** that differs by role.

## Exploratory: why the responder gain resolves only when the seller responds

**Status: post hoc, not pre-registered — hypothesis-generating.** Rendered from
`results/jj_hard_20261025/exploratory_role_asymmetry.json` (feasible pairs only; paired cluster
bootstrap over the 30 pairs, B = 5000, seed 3). Regenerate:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_hard_role_asymmetry \
    results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --seed 3
```

Cells: **seller_responds** (buyer starts) and **buyer_responds** (seller starts). Structural
checks first: the feasible pair set is closed under role swap (True) and the uniform-prior
P(accept) dicts shown to a buyer-starter and a seller-starter are exact mirror images
(True). The payoff structure cannot produce a role asymmetry; only Jev's behaviour or sampling can.

### Within each cell — Δ = bayes − prior-only

| | seller_responds: bayes / prior | Δ | buyer_responds: bayes / prior | Δ |
|---|---|---|---|---|
| G_responder | 0.776 / 0.577 | **+0.199** [+0.078, +0.333] | 0.729 / 0.690 | **+0.039** [-0.039, +0.135] |
| agreement | 0.846 / 0.704 | **+0.142** [+0.033, +0.271] | 0.829 / 0.787 | **+0.042** [-0.042, +0.133] |
| responder accepted the opening | 0.629 / 0.704 | **-0.075** [-0.175, +0.004] | 0.658 / 0.713 | **-0.054** [-0.117, -0.008] |
| responder countered | 0.217 / 0.104 | **+0.113** [-0.046, +0.267] | 0.171 / 0.158 | **+0.013** [-0.121, +0.133] |
| responder 1st counter: utility to recipient | 0.668 / 0.305 | **+0.363** [+0.324, +0.397] | 0.682 / 0.526 | **+0.156** [-0.062, +0.344] |
| responder 1st counter acceptable to recipient | 1.000 / 0.000 | **+1.000** [+1.000, +1.000] | 1.000 / 0.474 | **+0.526** [+0.000, +0.966] |
| starter's proposal was the one accepted | 0.744 / 1.000 | **-0.256** [-0.412, -0.116] | 0.819 / 0.905 | **-0.086** [-0.221, +0.049] |
| responder posterior entropy after 1 obs (bits) | 1.351 / 2.585 | **-1.234** [-1.281, -1.188] | 1.365 / 2.585 | **-1.220** [-1.292, -1.141] |
| responder learned about starter after 1 obs (bits) | 1.528 / 0.000 | **+1.528** [+1.440, +1.605] | 1.446 / 0.000 | **+1.446** [+1.397, +1.489] |

### Between cells, within each arm — Δ = seller_responds − buyer_responds

| | bayes | Δ | prior-only | Δ |
|---|---|---|---|---|
| G_responder | 0.776 vs 0.729 | +0.048 [-0.063, +0.156] | 0.577 vs 0.690 | -0.112 [-0.240, +0.010] |
| agreement | 0.846 vs 0.829 | +0.017 [-0.071, +0.096] | 0.704 vs 0.787 | -0.083 [-0.188, +0.000] |
| starter 1st offer: utility to recipient | 0.663 vs 0.656 | +0.007 [-0.061, +0.083] | 0.665 vs 0.664 | +0.001 [-0.069, +0.083] |
| starter 1st offer acceptable | 0.717 vs 0.708 | +0.008 [-0.104, +0.125] | 0.704 vs 0.713 | -0.008 [-0.138, +0.125] |
| responder posterior entropy after 1 obs | 1.351 vs 1.365 | -0.014 [-0.108, +0.069] | 2.585 vs 2.585 | +0.000 [+0.000, +0.000] |
| responder 1st counter: own utility | 0.791 vs 0.682 | +0.109 [+0.019, +0.190] | 0.702 vs 0.692 | +0.009 [-0.000, +0.061] |
| responder 1st counter: utility to recipient | 0.668 vs 0.682 | -0.014 [-0.080, +0.055] | 0.305 vs 0.526 | -0.222 [-0.462, -0.025] |
| responder 1st counter acceptable | 1.000 vs 1.000 | +0.000 [+0.000, +0.000] | 0.000 vs 0.474 | -0.474 [-1.000, -0.035] |
| starter's proposal was the one accepted | 0.744 vs 0.819 | -0.075 [-0.234, +0.085] | 1.000 vs 0.905 | +0.095 [+0.005, +0.224] |

### What the four candidate mechanisms got

- **H-a identifiability — rejected.** The responder's posterior after one observation has the same
  entropy whichever role responds (1.351 vs 1.365 bits, Δ -0.014 [-0.108, +0.069]).
  Buyer and seller openings reveal exactly as much type.
- **H-d opening quality — rejected.** Openings are identical across roles in both arms (utility to
  recipient 0.663 vs 0.656 in bayes; 0.665 vs 0.664 in prior-only).
- **H-c agreement drag — not within bayes** (0.846 vs 0.829, Δ [-0.071, +0.096]); but in **prior-only**
  agreement is lower when the seller responds (0.704 vs 0.787, Δ -0.083 [-0.188, +0.000], borderline).
- **H-b countering — the rate is similar, the *outcome* is not.** Responders counter at similar
  rates in both cells. But the **uninformed** seller's first counter is acceptable to the buyer
  **0.000** of the time against **0.474** for the uninformed buyer's (Δ -0.474 [-1.000, -0.035]),
  so in that cell the buyer-starter's proposal is the one accepted in **1.000** of agreements (vs 0.905).
  With learning both roles' counters are acceptable 1.000 / 1.000.

### The actual locus: the asymmetry is in the prior-only baseline, not in learning

Within **bayes** the two cells are statistically indistinguishable on every outcome
(G_responder 0.776 vs 0.729, Δ [-0.063, +0.156]). Within **prior-only** the seller-as-responder does worse
(G_responder 0.577 vs 0.690, Δ -0.112 [-0.240, +0.010], borderline), because its uninformed counters never
land. Since Δ = bayes − prior-only, a lower prior-only baseline in seller_responds yields a larger
Δ there. **Learning does not help sellers more; it lifts both cells to the same place from unequal
starting points.** The pre-registered "responder effect" (Δ G_responder > 0) is intact; what the
pooled role asymmetry reflected was a role-dependent *baseline*, not a role-dependent *gain*.

### Caveats, and the hypothesis this leaves

Prior-only responders rarely counter (0.104 / 0.158 of episodes), so the 0 %-vs-47 % contrast rests on
few counters and its interval barely excludes zero: **suggestive**. And *why* the uninformed seller's
counters are worse, under mirror-identical information, is unexplained — Jev's choice under the
seller framing, or selection into the pairs where countering happens at all. *Since tested* (`COUNTER_RESULTS.md`, pre-registered forced-counter probe, 1,440 decisions):
**not supported** — forced uninformed counters are equally acceptable across roles; the
natural 0 %-vs-47 % was selection into tough openings plus small n. No role effect survives.

## B. Infeasible pairs (6 clusters) — does learning help *recognise* incompatibility?

| | Prediction | bayes | prior-only | Δ | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| **B1** | fewer proposals before no-deal | 1.05 | 1.75 | **-0.698** | [-1.104, -0.292] | **CONFIRMED** |
| B2 | walk rate higher | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | saturated: both arms walk 100%, no timeouts |
| B3 | invalid acceptance ≈ 0 | 0.000 | 0.000 | — | — | OK |
| **B4 mechanism** | the aid's own verdict falls: max P(accept) over deals IR for me | 0.558 | 0.650 | **-0.092** | [-0.105, -0.079] | **CONFIRMED** |
| B5 | correct-no-deal ≈ 1.0 | 1.000 | 1.000 | — | — | engine check OK |
| | decisions before ending | 2.06 | 2.75 | -0.688 | [-1.083, -0.292] | |

Even at six clusters, **B1 and B4 resolved.** On a pair where no deal is individually rational for
both, the Bayes filter learns within a proposal that nothing it wants is anything the other side
would take (its own verdict falls from 0.650 to 0.558; the prior-only aid cannot update and sits at
0.650 forever) and walks 0.70 proposals sooner. B2 is not underpowered but at ceiling:
neither arm ever times out — everyone recognises no-deal; learning only changes *how fast*.

## What the hard regime adds to the story

1. The quality effect of learning **does not depend on the compromise deal** — it persists,
   smaller, when the only acceptable deals are genuine trades.
2. **Agreement becomes the main lever.** With few IR deals, the prior-only aid's first-acceptable
   strategy fails to find a deal a quarter of the time; learning recovers a third of that gap.
3. **Learning helps recognise incompatibility**, and the mechanism is the one predicted: the
   forward model's own estimate of "anything I want that they'd take" collapses after one
   observation.
4. A placebo-gate failure on noise, diagnosed and recorded, with the design fix it implies.

## Limits

One hard-regime run, not replicated; six infeasible clusters; the seller-as-responder asymmetry traced to the prior-only baseline and then, in
`COUNTER_RESULTS.md`, to selection plus small-n noise — no role effect; one prompt; no candidate-order randomization; β and ε
unswept; `jev-latest` pinned by name.
