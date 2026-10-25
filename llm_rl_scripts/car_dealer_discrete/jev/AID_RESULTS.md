# J/J policy aid — results (aid × salience 2×2)

**Date:** 2026-10-25 · **Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` (live) · **Status:** 2×2 + both mechanism controls complete; small-n

Every number below is regenerated from the committed logs by one command; the
machine-readable output is frozen at `results/jj_aid_20261025/analysis_2x2.json`.
Do not edit a figure here without re-running it:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_aid \
    results/jj_salience_20261025/baseline results/jj_salience_20261025/salience \
    results/jj_aid_20261025/aid results/jj_aid_20261025/aid_salience
```

Tags: **exact-on-logs** (deterministic function of the committed JSONL) or
**stochastic** (a draw from live `jev-latest`, reported with its spread, not a
rate). Floats from the analyzer are shown to 3 dp.

## Question

The salience probe (`SALIENCE_RESULTS.md`) showed the J/J failure is on the
**offer side**: in 130 offers neither Jev seat ever proposed a deal worth ≥ r to
the recipient. Diagnosis: Jev lacks the two things RL installs — (1) a
forward/acceptance model of the opponent, (2) internalized deadline cost.
Salience targets (2) and did not help. This experiment supplies (1) as an
**external policy aid** and tests it as a 2×2 with salience as an additive factor.

## The aid (what Jev is and is not given)

`jev/aid.py` — a finite Bayesian filter over the opponent's 6 rankings, uniform
prior, updated from the public history only (opponent offers via a softmax
likelihood over the 27 deals, β = 8; non-acceptance of our standing offer via the
public IR rule with noise floor ε = 0.05). For every candidate `OFFER` it returns
P(opponent accepts), which is appended to that option's text:
`"...; estimated chance the other party accepts: 0.83"`.

The aid **never** ranks options, computes expected value, recommends, prunes the
legal action set, or shows Jev the posterior. Jev sees every legal action
including its ideal (annotated 0.05) and still makes the Choice. One neutral
instruction sentence tells Jev what the annotation is. The aid reads only the
policy's own `Observation` and so cannot see the opponent's true ranking
(`test_aid.py::test_aid_api_is_leakage_safe`). The posterior is logged for the
evaluator, which unlocks the section-4.3 epistemic metrics for the first time.

## Design (matched 2×2)

Same 3 type pairs in all four cells (`--pairs 3 --seed 7`), easy regime
(r = 0.40), both starting sides, K = 2 → **12 episodes per cell**, 0 API errors.
Rows baseline / salience are the earlier probe's logs; rows aid / aid+salience
are new. Flags: `--regime easy --start both --repeats 2` plus `--aid` and/or
`--salience`.

## Results

### Outcome and offer behaviour

| | baseline | salience | **aid** | **aid+salience** | tag |
|---|---|---|---|---|---|
| Agreements | 0 / 12 | 0 / 12 | **12 / 12** | **12 / 12** | exact-on-logs |
| Wilson 95% (per cell) | [0, 0.243] | [0, 0.243] | [0.757, 1] | [0.757, 1] | |
| Pooled no-aid vs aid | 0 / 24 → [0, 0.138] | | 24 / 24 → [0.862, 1] | | non-overlapping |
| Offers made | 72 | 58 | 27 | 14 | exact-on-logs |
| Offers that were the offerer's ideal | 72 / 72 | 57 / 58 | **0 / 27** | **0 / 14** | exact-on-logs |
| Offers worth ≥ r to recipient | 0 / 72 | 0 / 58 | 23 / 27 | 12 / 14 | exact-on-logs |
| Aid-implied P(accept) of chosen offers, mean [min, max] | 0.05 [0.05, 0.05] | 0.05 [0.05, 0.05] | **0.765 [0.65, 0.95]** | **0.706 [0.65, 0.95]** | exact-on-logs (counterfactual for no-aid cells) |
| Share of chosen offers with P(accept) ≥ 0.5 | 0.0 | 0.0 | 1.0 | 1.0 | exact-on-logs |
| Recomputed vs logged aid value, max abs dev | — | — | 0.0 | 0.0 | cross-check |

### Deal quality (feasible episodes; all 12 per cell are feasible)

| | aid | aid+salience | tag |
|---|---|---|---|
| G_B (buyer attainable gain) | 0.745 | 0.729 | stochastic |
| G_S (seller attainable gain) | 0.697 | 0.572 | stochastic |
| min(G_B, G_S) | 0.666 | 0.572 | stochastic |
| Pareto-frontier rate | 0.833 (10/12) | 0.833 (10/12) | count exact-on-logs; rate stochastic |
| Nash ratio Q_N | 0.808 | 0.690 | stochastic |
| Proposals to agreement (mean) | 2.25 | 1.17 | stochastic |

Showcase (from `runs.jsonl`, `result.accepted_deal_id`): on the pair buyer
`price>delivery>warranty` / seller `warranty>delivery>price`, **all four**
rollouts in the aid cell agreed `P0-W0-D1` at utilities (0.808, 0.808) — each
side gets its top-ranked attribute. A genuine log-roll.

### Epistemic (section 4.3) — first belief metrics in this study

| | aid | aid+salience | tag |
|---|---|---|---|
| Belief reports scored (seats) | 24 | 24 | |
| Final log-loss of true type, bits, mean [min, max] | 1.566 [0.820, 3.292] | 2.039 [0.820, 3.483] | stochastic |
| Improvement over uniform prior, bits | +1.019 [−0.707, +1.765] | +0.546 [−0.898, +1.765] | stochastic |
| Final posterior entropy, bits (uniform = 2.585) | 1.718 [1.342, 2.585] | 2.025 [1.522, 2.585] | stochastic |
| Posterior argmax = true type (chance 0.167) | 0.417 | 0.292 | stochastic |

## Interpretation

**1. The forward model is sufficient to flip the outcome.** Supplying P(accept)
alone moves these pairs from 0/24 to 24/24 agreements. This is the decomposition
the design aimed at: of the two capabilities RL training installs, the
acceptance model — not deadline salience — is the one whose absence caused the
universal walk-away.

**2. Jev integrated the number; it was not merely present.** With the aid on,
Jev never once offered its ideal (0/41) even though the ideal remained a legal
option labelled "worth 1.000 to you". Every offer it made carried P(accept) ≥
0.65. It traded own utility for acceptance probability — the forward model being
*used* in the Choice. (The placebo control in §Mechanism controls refutes the
confound directly.)

**3. The agreements are efficient, not token.** 10/12 Pareto-efficient in each
aid cell; Nash ratio 0.81 in the aid cell; both sides well above reservation
(minG 0.67). Not compromise-at-0.5 deals.

**4. Most of the effect is the prior, not the learning.** The filter did learn
(+1.0 bit mean, argmax-correct 42% vs 17% chance) — but deals closed in ~2.25
proposals, so most offers were made under a near-uniform posterior. At the
uniform prior the aid already rates the ideal 0.05 and the universal compromise
0.95, and grades every deal by how many *types* find it IR — a public-information
computation. The value appears to come from having *any* acceptance model, more
than from identifying *this* opponent. This bears directly on whether EIG-style
opponent inference is the right investment; the prior-only control (§Mechanism
controls) confirms it — agreement is identical with learning switched off.

**5. Salience adds nothing to agreement and plausibly hurts quality.**
Agreement is 12/12 either way. Quality is lower with salience (minG 0.57 vs 0.67,
Nash 0.69 vs 0.81, G_S 0.57 vs 0.70) and deals close faster (1.17 vs 2.25
proposals), with two agreements landing a seller near its reservation (0.423).
This is consistent with the probe's mechanism — salience moves the quit/accept
side — now manifesting as accepting the *first* acceptable offer rather than
negotiating toward a better one. The epistemic numbers are also worse under
salience (less history → less learning), as expected mechanically. **All of
these quality gaps are K=2 stochastic differences: suggestive, not rates.**

**6. The aid is probabilistic and sometimes wrong, by design.** 4/27 and 2/14
aid-cell offers were below the recipient's reservation despite P(accept) ≥ 0.65 —
a deal IR for 4 of 6 types scores ≈ 0.65 but fails if the truth is one of the
other two. The recipients countered and the filter down-weighted. One seat
finished confidently wrong (improvement −0.71 bits), exactly the section-4.3
hazard; the noise floor kept it from collapsing to zero.

## What is robust vs suggestive

**Robust (saturated or exact-on-logs):** 0/24 vs 24/24 agreement; 0/41 ideal
offers under the aid vs 129/130 without; chosen-offer P(accept) ≥ 0.65 for 100%
of aid offers vs 0.05 for 100% of no-aid offers; recompute/log cross-check = 0;
10/12 Pareto counts.

**Suggestive (stochastic, K=2, 12/cell):** all deal-quality means; the
aid-vs-aid+salience quality gap; the epistemic means; the inference that
learning contributed little to agreement (needs the ablation below).

## Mechanism controls

Two controls isolate *what* in the aid caused the flip. Both are modes of the same
`AcceptanceAid`, so the policy/runner path and the instruction sentence are
byte-identical to the real aid; only the emitted numbers differ
(`test_aid.py::test_placebo_is_constant_and_prompt_identical_to_real_aid`).
Same 3 pairs (`--seed 7`), easy, both starts, K = 2 → 12 episodes per cell,
0 API errors. Regenerate (frozen at `results/jj_aid_20261025/analysis_controls.json`):

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_aid \
    results/jj_salience_20261025/baseline results/jj_salience_20261025/salience \
    results/jj_aid_20261025/aid results/jj_aid_20261025/aid_salience \
    results/jj_aid_20261025/placebo results/jj_aid_20261025/prior_only
```

- **Placebo** (`--aid-mode placebo`): every offer annotated with a constant 0.50.
  Separates "an acceptance field exists" from the information in it.
- **Prior-only** (`--aid-mode prior_only`): never updates; always the uniform-prior
  P(accept). Separates the prior-averaged forward model from opponent inference.

| | **aid (bayes)** | **prior-only** | **placebo** | tag |
|---|---|---|---|---|
| Agreements | 12 / 12 | **12 / 12** | **0 / 12** | exact-on-logs |
| Wilson 95% | [0.757, 1] | [0.757, 1] | [0, 0.243] | |
| Offers made | 27 | 14 | 72 | exact-on-logs |
| Offers that were the ideal | 0 / 27 | **0 / 14** | **72 / 72** | exact-on-logs |
| Offers worth ≥ r to recipient | 23 / 27 | 12 / 14 | 0 / 72 | exact-on-logs |
| Real-model P(accept) of chosen offers (counterfactual), mean [min, max] | 0.765 [0.65, 0.95] | 0.693 [0.65, 0.95] | 0.05 [0.05, 0.05] | exact-on-logs |
| G_B / G_S | 0.745 / 0.697 | 0.776 / 0.619 | 0 / 0 | stochastic |
| min(G_B, G_S) | 0.666 | 0.619 | 0 | stochastic |
| Pareto-frontier rate | 0.833 (10/12) | 0.833 (10/12) | — | count exact; rate stochastic |
| Nash ratio Q_N | 0.808 | 0.734 | 0 | stochastic |
| Proposals to agreement | 2.25 | 1.17 | — | stochastic |
| Learning (improvement over uniform, bits) | +1.019 | 0.0 | 0.0 | by construction for controls |

**Placebo reproduces baseline exactly.** Buyer `P0-W2-D2` ×36, seller `P2-W0-D0`
×36, walker = starter at decision #7 in all 12 — the same offer multiset and
walk pattern as the no-aid baseline. With the sentence and the field present but
the number uninformative, Jev behaves as if the aid were absent. **The effect is
the differential information across deals, not the presence of an acceptance
field or the instruction sentence.** Confound closed.

**Prior-only reproduces the real filter's outcome.** 12/12 agreements, 0/14 ideal
offers, 10/12 Pareto, and the same agreements — including `P0-W0-D1` at
(0.808, 0.808) in all four rollouts of the log-roll pair — with learning
switched off (improvement exactly 0.0 bits by construction). **The prior-averaged
forward model — which deals are IR for most *types*, a public-information
computation — is sufficient for agreement.** Opponent inference is not what
unlocks the deal.

**What learning buys is quality — suggestive at this scale, since resolved at scale in `QUALITY_RESULTS.md`.** The
Bayes filter's deals are more balanced (Nash 0.808 vs 0.734, minG 0.666 vs 0.619,
G_S 0.697 vs 0.619) and take longer to reach (2.25 vs 1.17 proposals); prior-only
closes faster on first-acceptable offers, the same signature salience produced.
These are K=2 gaps of the same size as the aid-vs-aid+salience gaps above: a
consistent direction, not a rate. *Since resolved:* the pre-registered 36-pair ×
K=8 run (`QUALITY_RESULTS.md`) confirms the direction with a cluster-bootstrap CI
excluding zero; the numbers live there, not here.

Artifact to ignore: under a uniform posterior, "argmax = true type" is a tie
broken by key order, so the 0.0 for the control cells is not "worse than chance".

**Robust:** placebo 0/12 with 72/72 ideal offers; prior-only 12/12 with 0/14 ideal
offers and 10/12 Pareto; zero learning in both controls by construction.
**Suggestive at this scale:** every bayes-vs-prior-only quality difference —
resolved in `QUALITY_RESULTS.md`.

**Implication for the BED question.** The capability Jev lacked was an acceptance
model over the public type space, not inference about the particular opponent.
EIG-style opponent inference would be aimed at the quality margin — shown real
at scale in `QUALITY_RESULTS.md` — not at the outcome.

## Limits

- **Both mechanism controls were run** (§Mechanism controls); the two bullets
  that previously stood here are answered: the effect is informational (placebo
  = baseline) and the prior suffices for agreement (prior-only = aid).

- **Confound: annotation vs instruction sentence.** The aid condition adds both
  the per-offer numbers *and* one sentence explaining them; they are inseparable
  here. The decisive control is a **placebo aid**: the same sentence with an
  uninformative constant (e.g. every offer annotated 0.50). If Jev still
  concedes, the effect is "an acceptance field exists", not the information in
  it. *Resolved:* the placebo control reproduces baseline exactly.
- **Prior vs learning.** *Resolved for agreement:* prior-only matches the aid.
  Quality resolved at scale: see `QUALITY_RESULTS.md`.
- n = 12 per cell, 3 pairs, easy regime only, K = 2. Per-cell rates need K ≈ 8–10
  across many pairs with seed-clustered intervals (Text-Nav main-run protocol).
- The response model (IR threshold) is deliberately simple and optimistic; a
  logistic/graded variant is untested. β and ε were not swept.
- Nothing here is a claim about Jev's negotiation ceiling under other prompts,
  or about the hard regime (r = 0.55), where the universal compromise is not IR.
