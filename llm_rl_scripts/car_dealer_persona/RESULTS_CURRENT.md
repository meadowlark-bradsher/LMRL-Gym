# CD-PERSONA-1 results: does a current model hold a planted Car Dealer persona?

_Rendered by `render_current_note.py` from `results/persona_current_20261107` written by `analyze_current.py`; every number is formatted from the result files, none is typed. Registered in `PREREGISTRATION_CURRENT.md`, frozen at tag `prereg-persona1-20261107` before any confirmatory conversation was generated. The 2023 baseline is CD-PERSONA-0 (`RESULTS_BASELINE2023.md`, tag `prereg-persona-20261106`)._

Two arms, one model (`claude-sonnet-5-5`, model-default sampling), 2250 conversations each, 250 per buyer × seller cell, ids fixed by the registered seed.

| Arm | Protocol | Refusals | Scorer `unclear` | Notes | Spend |
|---|---|---|---|---|---|
| A | buyer agent against content-independent scripted sellers | 0 | 67 | 265 decision-token timeouts (11.8%), 2 malformed tokens in 8323 replies | $27.05 |
| N | the 2023 single-narrator prompt, verbatim, `max_tokens` 2048 | 0 | 286 | 526 stories cut at the 2023 token cap (23.4%), 5 of them empty (thinking used the whole budget) | $39.45 |

## The 2×2: who leaked, the model or the protocol?

`eps_buyer` = audit-corrected P(sale) in the hard cells `H` = B-discount × {S-features, S-concise}, where no discount is on offer. In Arm A no discount was offered by construction, so this is the buyer's own violation rate; in Arm N, as in 2023, it is an upper bound.

| | 2023 narrator (GPT-3.5) | Current narrator (Arm N) | Current buyer agent (Arm A) |
|---|---|---|---|
| `eps_buyer` | 0.198 [0.174, 0.223] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| band | `SOFT` | `HELD` | `HELD` |

- Model effect, old protocol with the model swapped: **+0.198 [+0.173, +0.223]**.
- Protocol effect, new model with the protocol swapped: **+0.000 [+0.000, +0.000]**.

The registered estimator returns a `[0.000, 0.000]` interval when every audited scorer-sale is a false sale (precision 0) and no audited no-sale is a miss; it is not a certainty claim. The scorer's own raw rates with their Wilson intervals are 0.008 [0.003, 0.021] (4/497) in Arm A and 0.006 [0.002, 0.018] (3/482) in Arm N, and the audit found every one of those scorer-sales to be a no-sale.

## Predictions

| | Claim | Value | Verdict |
|---|---|---|---|
| P1 | Arm A: `eps_buyer` `HELD` (≤ 0.05) with S-concise and above 0.05 with S-features (raw, by seller) | S-features 0.000, S-concise 0.016; pooled audit-corrected 0.000 | REFUTED |
| P2 | Arm N: `eps_buyer` above 0.05 and at most the 2023 value 0.198 | 0.000 | REFUTED |
| P3 | protocol effect `eps_N − eps_A` at least 0.05 | +0.000 | REFUTED |
| P4 | B-impatient compatibility gap negative in both arms | A -0.193, N -0.055 | CONFIRMED |
| P5 | `eps_spec` (M1b) above 0.05 | 0.056 [0.033, 0.081], borderline | CONFIRMED |
| P6 | token agrees with dialogue ≥ 0.95 and timeout share < 0.1 | agreement 0.959, timeout 0.118 | REFUTED |
| P7 | M6 cumulative acceptance non-decreasing in the offer and ≥ 0.5 after 15% | 0% 0.000, 5% 0.000, 10% 0.140, 15% 0.384 | REFUTED |
| P8 | second buyer model | none registered | VOID |

P1, P2 and P3 were written expecting a non-zero leak in both arms and more of it under the narrator protocol; the leak is zero in both, so all three are refuted in the direction of the persona holding better than predicted. P6 fails on its timeout half only. P7 fails on its level, not its shape: acceptance is non-decreasing but reaches only 0.392 by the close (see M6).

## M1. Buyer discount constraint

**Arm A**: audit-corrected **0.000 [0.000, 0.000]**, band `HELD`.

| | Rate | |
|---|---|---|
| Raw scorer | 0.008 [0.003, 0.021] (4/497) | 3 scorer-unclear excluded, bounds [0.008, 0.014] |
| Audit-corrected (registered) | 0.000 [0.000, 0.000] | precision 0.000 on 4 audited sales, miss rate 0.000 on 30 audited no-sales |
| Sensitivity: audit `unclear` counted as a sale | 0.000 [0.000, 0.000] | |
| With S-features (raw) | 0.000 [0.000, 0.015] (0/249) | 1 unclear |
| With S-concise (raw) | 0.016 [0.006, 0.041] (4/248) | 2 unclear |

**Arm N**: audit-corrected **0.000 [0.000, 0.000]**, band `HELD`.

| | Rate | |
|---|---|---|
| Raw scorer | 0.006 [0.002, 0.018] (3/482) | 18 scorer-unclear excluded, bounds [0.006, 0.042] |
| Audit-corrected (registered) | 0.000 [0.000, 0.000] | precision 0.000 on 3 audited sales, miss rate 0.000 on 30 audited no-sales |
| Sensitivity: audit `unclear` counted as a sale | 0.000 [0.000, 0.000] | |
| With S-features (raw) | 0.009 [0.002, 0.031] (2/232) | 18 unclear |
| With S-concise (raw) | 0.004 [0.001, 0.022] (1/250) | 0 unclear |
| Text-level, exploratory: discount buyer, no discount detected, any seller | 0.006 [0.002, 0.018] (3/476) | |

## M1b. Buyer spec constraint (Arm A)

`eps_spec` = P(sale | B-spec, one required feature missing), audit-corrected: **0.056 [0.033, 0.081]**, band `SOFT`, borderline (the interval crosses a band edge).

| | Rate | |
|---|---|---|
| Raw scorer, missing-one | 0.087 [0.062, 0.120] (32/369) | 5 unclear, bounds [0.086, 0.099] |
| Audit-corrected (registered) | 0.056 [0.033, 0.081] | precision 0.643 on 28 audited sales, miss rate 0.000 on 30 audited no-sales |
| Sensitivity: audit `unclear` counted as a sale | 0.135 [0.062, 0.232] | |
| Raw scorer, all features present (complement) | 0.219 [0.179, 0.266] (75/342) | 34 unclear |

The audited spec-cell sales split three ways: a real purchase of a car the seller never confirmed has the missing feature, a conditional commitment ("if you confirm X, I'll sign") that the scorer reads as a sale and the labeller as `unclear`, and a refusal. Only the first counts against the persona; the registered estimator counts `unclear` as not-a-sale, the sensitivity row as a sale.

## M2. Seller discount constraint

- Arm A: the seller lines are scripted and verified against the script in every conversation (`seller_ok` in all 2250; violations 0), so no discount reached the buyer in `H` by construction.
- Arm N: **0.010 [0.006, 0.016] (15/1500)**, band `HELD`. From the detector, not audited; the detector misses dealers who counter in plain numbers without the word "discount", so this is understated.
  - S-features: 0.020 [0.012, 0.033] (15/750)
  - S-concise: 0.000 [0.000, 0.005] (0/750)

## M3. Compatibility: sale rate by persona pair

**Arm A**

| Buyer | S-discount | S-features | S-concise | Gap (compatible minus mean of others) |
|---|---|---|---|---|
| B-discount | **0.542 (130/240)** | 0.000 (0/249) | 0.016 (4/248) | +0.534 [+0.470, +0.597] |
| B-spec | 0.028 (7/249) | **0.264 (57/216)** | 0.175 (43/246) | +0.162 [+0.098, +0.227] |
| B-impatient | 0.082 (20/244) | 0.557 (137/246) | **0.127 (31/245)** | -0.193 [-0.248, -0.138] |

**Arm N**

| Buyer | S-discount | S-features | S-concise | Gap (compatible minus mean of others) |
|---|---|---|---|---|
| B-discount | **0.970 (226/233)** | 0.009 (2/232) | 0.004 (1/250) | +0.964 [+0.941, +0.987] |
| B-spec | 0.767 (145/189) | **0.756 (93/123)** | 0.775 (189/244) | -0.015 [-0.101, +0.071] |
| B-impatient | 0.966 (227/235) | 0.964 (217/225) | **0.910 (212/233)** | -0.055 [-0.096, -0.015] |

Bold marks the pairing the 2023 design calls compatible. Denominators exclude scorer-`unclear`; in Arm N the B-spec × S-features cell lost 127 of 250 to truncation, so its rate is the least certain in either table.

## M4. Budget

- Arm A, sales at list price 1.25 × budget: **0.630 [0.442, 0.785] (17/27)** closed above budget (0 sales without an extractable price). Every undiscounted sale there is over budget; a 15%-off S-discount sale is not. The scorer's price matched the decision token's price in 0.788 [0.747, 0.824] (338/429) of priced sales.

| List price / budget | Sale rate | Over budget among sales |
|---|---|---|
| 0.8 | 0.272 [0.241, 0.306] (198/727) | 0.000 [0.000, 0.019] (0/198) |
| 1.0 | 0.284 [0.252, 0.318] (204/718) | 0.000 [0.000, 0.018] (0/204) |
| 1.25 | 0.037 [0.025, 0.053] (27/738) | 0.630 [0.442, 0.785] (17/27) |

- Arm N, priced sales above the planted budget, as in 2023: **0.762 [0.738, 0.784] (997/1309)** (3 sales without a price).

| Budget | Share above budget |
|---|---|
| 10,000 | 0.960 [0.927, 0.978] (237/247) |
| 30,000 | 0.954 [0.920, 0.974] (229/240) |
| 50,000 | 0.796 [0.745, 0.840] (219/275) |
| 70,000 | 0.596 [0.536, 0.654] (155/260) |
| 90,000 | 0.547 [0.489, 0.604] (157/287) |

## M5. Decision channel against the dialogue

- Arm A: the decision token (`BUY` vs `LEAVE`/timeout) agrees with the scorer's dialogue outcome in **0.959 [0.950, 0.967] (2094/2183)**; 89 disagreements. Timeouts (five seller turns, no decision): **0.118 [0.105, 0.132] (265/2250)**. Malformed tokens: 0.000 [0.000, 0.001] (2/8323) of buyer replies.
- Arm N: a final self-report of exactly 0 or 1 agrees with the dialogue in **1.000 [0.944, 1.000] (65/65)**; 2178 stories (96.8%) end on a fractional one (dialogue outcomes 1312 sale, 587 no sale, 279 unclear); 7 have no parseable self-report.

## M6. Response to the discount schedule (Arm A, B-discount × S-discount)

250 conversations. Cumulative acceptance (token `BUY` by the end of the buyer's reply to that seller turn) and per-offer acceptance among buyers still present.

| Seller turn | Offer | Cumulative acceptance | Per-offer acceptance |
|---|---|---|---|
| 1 | no offer | 0.000 [0.000, 0.015] (0/250) | 0.000 [0.000, 0.015] (0/250) |
| 2 | 5% off | 0.000 [0.000, 0.015] (0/250) | 0.000 [0.000, 0.015] (0/250) |
| 3 | 10% off | 0.140 [0.102, 0.188] (35/250) | 0.140 [0.102, 0.188] (35/250) |
| 4 | 15% off | 0.384 [0.326, 0.446] (96/250) | 0.284 [0.228, 0.347] (61/215) |
| 5 | close at 15% | 0.392 [0.334, 0.454] (98/250) | 0.013 [0.004, 0.046] (2/154) |

Left without buying (token `LEAVE`): 0.032 [0.016, 0.062] (8/250); the rest of the non-buyers timed out still asking. Acceptance is non-decreasing and zero until a discount appears, as the persona requires, but it stays below 0.5 after the 15% offer: the agent conditions its purchase on feature confirmations that the scripted seller never gives.

## The hand-audit

**Arm A**: 238 items, labelled with cell and scorer call hidden (`audit.py`, seed 20261107).

| Stratum | Items | Labelled sale | Labelled no sale | Labelled unclear |
|---|---|---|---|---|
| Hard cells `H`, scorer said sale | 4 | 0 | 4 | 0 |
| Hard cells `H`, scorer said no sale | 30 | 0 | 30 | 0 |
| Spec cells `H_spec`, scorer said sale | 28 | 18 | 4 | 6 |
| Spec cells `H_spec`, scorer said no sale | 30 | 0 | 28 | 2 |
| Other cells, scorer said sale | 116 | 94 | 12 | 10 |
| Other cells, scorer said no sale | 30 | 0 | 24 | 6 |

**Arm N**: 180 items, labelled with cell and scorer call hidden (`audit.py`, seed 20261107).

| Stratum | Items | Labelled sale | Labelled no sale | Labelled unclear |
|---|---|---|---|---|
| Hard cells `H`, scorer said sale | 3 | 0 | 3 | 0 |
| Hard cells `H`, scorer said no sale | 30 | 0 | 30 | 0 |
| Other cells, scorer said sale | 117 | 110 | 0 | 7 |
| Other cells, scorer said no sale | 30 | 3 | 26 | 1 |

Registered as a check of the scorer outside `H`: in Arm A, 94 of 116 (0.810, interval 0.730 to 0.871) of the decoy scorer-sales were sales on the label (the rest are conditional commitments and walk-outs read as sales, the false-sale mode seen at the gate), and 0 of 30 (0.000, interval 0.000 to 0.114) of the decoy no-sales were sales. In Arm N, 110 of 117 (0.940, interval 0.882 to 0.971) of the decoy scorer-sales were sales and 3 of 30 (0.100, interval 0.035 to 0.256) of the decoy no-sales were sales. The scorer's errors run toward calling sales, so the raw rates in this note are upper bounds and the registered verdicts rest on the audit.

## Reading

- **The discount-only persona holds.** No violation survives the audit in 497 Arm A hard-cell conversations (the scorer called 4 sales, all false) or in 482 in Arm N (3, all false), against 20% in the 2023 data.
- **The change is the model, not the protocol.** Swapping GPT-3.5 for `claude-sonnet-5-5` under the 2023 narrator prompt removes the leak (+0.198); giving the same model its own seat against a scripted seller changes nothing further (+0.000). The registration predicted the opposite.
- **The spec persona is softer.** `eps_spec` is 0.056, `SOFT`, borderline (the interval crosses a band edge); 0.135 if the conditional commitments count as sales. The buyer told to need every feature sometimes buys a car whose missing feature was never confirmed.
- **Compatibility structure.** The discount buyer's gap is +0.534 (A) and +0.964 (N); the impatient buyer's is negative in both arms (-0.193, -0.055), as predicted; the spec buyer's is +0.162 (A) and -0.015 (N).
- **The agent is a harder counterparty than the narrator's buyer.** Arm A sale rates run from 0.000 to 0.557; Arm N from 0.004 to 0.970. The narrator writes buyers who cave to firm sellers; the agent walks or keeps asking.
- **For the oracle question.** As a noisy oracle of the discount rule, this model's noise rate is 0 in both protocols at this sample size; the spec rule leaks at about 6%. What the agent does not do is buy on the discount alone: it takes the 15% offer 38% of the time, holding out for answers the scripted seller cannot give.

## Limits

- One model, one buyer prompt, one narrator prompt. P8 is void; a second buyer model would be its own registration.
- The scripted seller ignores the buyer's words, so M6 and the Arm A sale rates measure the agent against a seller that never answers; 11.8% of Arm A conversations ended with the buyer still asking.
- The 2023 `max_tokens` of 2048 is not neutral for this model: 526 Arm N stories were cut, 5 before any text, and 286 score `unclear`. Rates exclude them and the bounds from counting them either way are in the tables.
- All labels (238 + 180 audit items, 160 gate items) come from one labeller, Claude, with cell and scorer call hidden but not the conversation's content; the labelling rules are those of `gold/README.md`.
- The `[0.000, 0.000]` intervals are the registered bootstrap on an audit with zero positives; they bound the estimator, not the quantity.
- M2 in Arm N and the text-level row depend on a discount detector that was gated but not audited.
