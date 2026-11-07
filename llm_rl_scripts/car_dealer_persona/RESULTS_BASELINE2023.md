# CD-PERSONA-0 results: did the 2023 Car Dealer personas govern behavior?

_Rendered by `render_baseline_note.py` from `results/persona_baseline2023_20261106` written by `analyze.py`; every number is formatted from the result files, none is typed. Registered in `PREREGISTRATION_BASELINE2023.md`, frozen at tag `prereg-persona-20261106` before the confirmatory split was read._

Data: the released LMRL-Gym Car Dealer conversations, each written by a single GPT-3.5 narrator (both sides). Confirmatory split: 3889 conversations; the persona-blind scorer returned `unclear` on 91 of them.

## Predictions

| | Claim | Value | Verdict |
|---|---|---|---|
| P1 | discount-only buyer constraint is not `HELD` (rate above 0.05) | 0.198 audit-corrected | CONFIRMED |
| P2 | no-discount seller constraint violated above 0.05 | 0.144 | CONFIRMED |
| P3 | compatibility gap at least 0.1 for the discount buyer, below it for the other two | B-discount: +0.756, B-spec: +0.047, B-impatient: -0.323 | CONFIRMED |
| P4 | more than 10% of priced sales above budget, highest at the 10k budget | 0.148 overall; highest at 30,000 | REFUTED |
| P5 | a self-report of exactly 0 or 1 agrees with the dialogue outcome at least 90% of the time | 0.988 | CONFIRMED |

## M1. Buyer hard constraint: does a discount-only buyer buy from a seller who never discounts?

Audit-corrected rate: **0.198 [0.174, 0.223]**, band `SOFT`, borderline (the interval crosses a band edge).

| | Rate | |
|---|---|---|
| Raw scorer | 0.201 [0.177, 0.227] (200/995) | band `NOT_HELD`; 15 scorer-unclear excluded, bounds [0.198, 0.213] |
| Audit-corrected (registered) | 0.198 [0.174, 0.223] | precision 0.983 on 60 audited sales, miss rate 0.000 on 30 audited no-sales |
| Sensitivity: audit `unclear` counted as a sale | 0.228 [0.182, 0.292] | |
| With S-features (raw) | 0.335 [0.296, 0.378] (170/507) | |
| With S-concise (raw) | 0.061 [0.043, 0.086] (30/488) | |
| Eval file only (raw) | 0.235 [0.158, 0.336] (20/85) | band `NOT_HELD` |
| Text-level, exploratory: discount buyer, no discount detected, any seller | 0.156 [0.133, 0.182] (134/858) | |

Sales in these cells follow a seller who was told never to discount, so this rate is an upper bound on how often the buyer broke its own persona; see the exploratory split below.

## M2. Seller hard constraint: does a no-discount seller offer a discount?

**0.144 [0.130, 0.158] (354/2463)**, band `SOFT`. Not hand-audited; on the labelled conversations the detector missed more offers than it invented, so this is more likely understated.

- S-features: 0.270 [0.246, 0.296] (330/1221)
- S-concise: 0.019 [0.013, 0.029] (24/1242)
- Eval file only: 0.132 [0.098, 0.175] (40/303)

## M3. Compatibility: sale rate by persona pair

| Buyer | S-discount | S-features | S-concise | Gap (compatible minus mean of others) |
|---|---|---|---|---|
| B-discount | **0.954 (479/502)** | 0.335 (170/507) | 0.061 (30/488) | +0.756 [+0.726, +0.785] |
| B-spec | 0.910 (444/488) | **0.600 (294/490)** | 0.197 (97/493) | +0.047 [-0.002, +0.095] |
| B-impatient | 0.928 (385/415) | 0.651 (136/209) | **0.466 (96/206)** | -0.323 [-0.400, -0.247] |

Bold marks the pairing the paper calls compatible. Eval file only, gaps: B-discount +0.693, B-spec +0.069, B-impatient -0.404.

## M4. Budget: share of priced sales above the planted budget

**0.148 [0.134, 0.164] (315/2127)** (4 sales had no extractable price). Eval file only: 0.153 [0.114, 0.202] (39/255).

| Budget | Share above budget |
|---|---|
| 10,000 | 0.198 [0.163, 0.239] (82/414) |
| 30,000 | 0.199 [0.164, 0.239] (85/428) |
| 50,000 | 0.145 [0.113, 0.183] (57/394) |
| 70,000 | 0.106 [0.080, 0.139] (44/417) |
| 90,000 | 0.099 [0.075, 0.129] (47/474) |

## M5. Self-report against the dialogue outcome

A final self-report of exactly 0 or 1 agrees with the dialogue outcome in **0.988 [0.983, 0.992] (2356/2384)**. Eval file only: 0.980 [0.954, 0.991] (247/252).

1344 conversations (34.6%) end on a fractional self-report; their dialogue outcomes are 740 sale, 559 no sale, 45 unclear. 125 have no parseable self-report.

## The hand-audit

180 items, labelled with cell and scorer call hidden (`audit.py`, seed 20261107).

| Stratum | Items | Labelled sale | Labelled no sale | Labelled unclear |
|---|---|---|---|---|
| Hard cells, scorer said sale | 60 | 59 | 0 | 1 |
| Hard cells, scorer said no sale | 30 | 0 | 29 | 1 |
| Other cells, scorer said sale | 60 | 60 | 0 | 0 |
| Other cells, scorer said no sale | 30 | 2 | 28 | 0 |

Not registered, but it bears on the verdict: among the scorer's no-sale calls outside the hard cells, 2 of 30 (0.067, interval 0.018 to 0.213) were labelled a sale, against 0 of 30 (0.000, interval 0.000 to 0.114) inside them. The registered correction uses only the hard-cell strata. If the hard cells had the outside miss rate, the corrected rate would be 0.251.

## Exploratory: who broke persona first in the hard cells?

Not registered. Scorer outcome by whether the discount detector fired, hard cells only (995 decided conversations).

| | Discount offered | No discount detected |
|---|---|---|
| Sale | 88 | 112 |
| No sale | 73 | 722 |

- The seller offered a discount it was told never to give in 161 of 995 (0.162, interval 0.140 to 0.186).
- The buyer bought with no discount detected in 112 of 995 (0.113, interval 0.094 to 0.134). The detector misses some offers, so this overstates the buyer's own violations somewhat.

## Reading

- **The personas strongly govern the story.** Sale rates run from 0.061 to 0.954 across the nine pairings.
- **The hard constraints are soft.** A discount-only buyer still buys from a no-discount seller about 20% of the time, and a no-discount seller offers a discount at least 14% of the time. As a noise level for an oracle, that is inside the range of answer noise the elicitation toy swept.
- **The verdict on M1 is borderline.** By the registered rule it is `SOFT`; the interval crosses a band edge, and counting audit `unclear` as a sale and applying the outside miss rate would put it above 0.2.
- **It depends on which no-discount seller it is.** Violations concentrate with the feature-talking seller (0.335) and are rare with the concise one (0.061).
- **The paper's compatibility structure holds for 1 of 3 buyers** (B-discount). The gap for B-impatient is -0.323: it buys less from the seller the design pairs it with than from the other two.
- **An unambiguous self-report can be trusted (0.988 agreement), but 35% of conversations end on a fractional one.**

## Limits

- One narrator wrote both sides. This is whether a narrated story respects two personas, not whether a buyer agent holds one against an independent seller.
- All labels (160 for the scorer gate, 180 for the audit) come from one labeler, Claude. The audit hid cell and scorer call, not the conversation's content.
- The released files may not be the exact data behind the paper's tables (`PERSONA_AUDIT_TARGET.md`).
- M2 and the text-level and exploratory rows depend on a discount detector that was gated but not audited.
