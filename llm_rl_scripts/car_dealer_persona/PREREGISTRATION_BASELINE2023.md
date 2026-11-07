# Pre-registration: CD-PERSONA-0 — did the 2023 Car Dealer personas govern behavior? (offline baseline)

**Registered:** 2026-11-06, before any outcome was tabulated by persona and before the
CONFIRMATORY split was read. **Frozen at tag:** `prereg-persona-20261106` (the commit that adds
this file; `data.py`, `scorer.py`, `validate.py`, `audit.py`, `analyze.py`, the tests and `gold/`
as of that commit; `scorer.py` sha256 `626833c6…0e7f26`). **Data:** the released LMRL-Gym Car
Dealer files (§Data). **Background and paper-vs-code line-up:**
[`PERSONA_AUDIT_TARGET.md`](PERSONA_AUDIT_TARGET.md).

## Why this study

The original Car Dealer task plants a persona (a strategy background, a brand, a feature list, a
budget) and expects the counterparty's behavior to follow from it. If planting is reliable, an
LLM counterparty is a noisy realization of a known type and can serve as a measured oracle for
elicitation studies (`elicit_negotiation`); the violation rate is the noise level. This study
measures that rate in the 2023 data, as the baseline against which current models will be
compared in a later, separately registered study.

## What has already been seen

- The field census of both files (keys, persona and budget marginals, record counts,
  duplicates) and the marginal distribution of the scraped final `purchase_prob`.
- The six conversations printed in the paper.
- The full text of the 360 conversations in the DEV, GOLD and RESERVE splits, read with
  `buyer_info` hidden, and the scorer's output on them.

**Not seen:** any outcome broken down by buyer or seller persona, by persona pair, or by
budget, on any split; and nothing at all from the CONFIRMATORY split beyond the field census.

## Data

`train.json` (sha256 `20c0ab1376d7867343bc330023d295a8a55aa41c9ef7107b6a82ded8e3d7e743`) and
`eval.json` (sha256 `bf244da9144529ff02ac77c9ee4b8abec60f20f0ad2ae748890c1d146d014eab`) from
`https://rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset/car-dealer/`. Each record is one
conversation written by a single GPT-3.5 narrator (both sides), with the planted persona in
`buyer_info`. A record's id is the sha1 of its `original_conversation`. Exact duplicates are
collapsed; a train record whose id also occurs in eval is dropped from train.

Splits (`data.py`, seed 20261106, drawn from the distinct train records only):

| split | size | use |
|---|---|---|
| DEV | 200 | scorer rule development; text read freely |
| GOLD | 80 | hand-labelled; gate set for scorer v1, then used to revise the rules |
| RESERVE | 80 | hand-labelled; gate set for the revised scorer |
| CONFIRMATORY | 3,889 (all remaining distinct train + all of eval) | the registered analysis |

## Outcome scoring

`scorer.py` is **persona-blind**: it receives only the conversation text, never `buyer_info`.

- **Dialogue outcome** (primary): `sale`, `no_sale` or `unclear`, decided from the dialogue
  turns alone. The narrator's probability/price annotations are stripped before scoring, so this
  outcome does not depend on the self-report.
- **Agreed price**: the dollar amount the sale closes at, when one is stated.
- **Discount offered**: whether the dealer offers any price reduction.
- **Self-report** (secondary): the narrator's final stated purchase probability, re-parsed from
  the raw text (percentages read as percentages), kept separate for the calibration measure.

Labels (`gold/`) are assigned by Claude reading each conversation in full with `buyer_info`
hidden and before the scorer is run on that split. They are committed with the text.

**Validity gate.** Sale outcome: accuracy at least 0.90 on items where both label and scorer
are decided, and the scorer returns `unclear` on at most 15% of label-decided items. Discount
detector: agreement at least 0.85.

**Gate history** (`validate.py`; outputs in `results/persona_scorer_validation_20261106*`).

| | gate set | sale accuracy | false sales | missed sales | discount | price exact |
|---|---|---|---|---|---|---|
| scorer v1 | GOLD | 75/77 (0.974), pass | 2 of 37 | 0 of 40 | 73/80 (0.912), pass | 38/39 |
| revised scorer | RESERVE | 76/77 (0.987), pass | 1 of 34 | 0 of 43 | 76/80 (0.950), pass | 41/41 |

Scorer v1 passed the gate, but both of its errors were a buyer deferring the decision scored as
a sale. A false-sale rate of that size (5.4%, interval 1.5% to 17.7%) is as large as the `HELD`
band for M1, which the gate as written does not test. The rules were therefore revised using
GOLD as development data (deferral phrases, conditional dealer closes, speaker labels with a
parenthetical, discount inquiries) and the revised scorer was validated once on RESERVE, which
had not been read before. The scorer is frozen as validated; one false sale in 34 (2.9%,
interval 0.5% to 14.9%) still cannot establish a rate below 5%, which is why M1 is audited by
hand (next section).

## Hand-audit of the hard cells

`audit.py`. Hard cells `H`: buyer `B-discount` with a seller instructed never to discount
(`S-features` or `S-concise`), within CONFIRMATORY.

- **Sample** (seed 20261107): 60 scorer-called sales from `H` (all, if fewer) and 30
  scorer-called no-sales from `H`.
- **Blinding:** the sample is topped up with conversations from outside `H` to a fixed 120
  scorer-sales and 60 scorer-no-sales, shuffled. The packet holds only an item number and the
  text. Cell membership and the scorer's call are in a key file that is not opened until all 180
  labels are written. The `packet` step prints no outcome counts.
- **Labels:** Claude, same definitions as `gold/README.md`, outcome only.
- **Corrected rate:** `eps_buyer = (S * precision + N * miss) / (S + N)`, with `S`, `N` the
  scorer's sale and no-sale counts in `H`, `precision` the share of audited `H` sales labelled
  `sale`, `miss` the share of audited `H` no-sales labelled `sale`. A label of `unclear` counts
  as not a sale; counting it as a sale is reported as a sensitivity. The 95% interval is a
  percentile bootstrap (B = 5000) over both the cell's scorer outcomes and the audit labels.

The M1 verdict and prediction P1 are judged on the audit-corrected rate. The raw scorer rate is
reported next to it.

## Measures

All on CONFIRMATORY, pooled; eval alone (balanced 3x3) is reported as a robustness check.
Rates exclude scorer-`unclear` outcomes; the `unclear` share is reported, with bounds from
counting every `unclear` as a sale and as no sale. Intervals are 95% Wilson unless stated.

Buyer backgrounds: `B-discount` ("never buy unless given a discount"), `B-spec` ("only buy if
it meets all specifications"), `B-impatient`. Seller backgrounds: `S-discount` (loves discounts,
never talks features), `S-features` (never discounts, talks features), `S-concise` (never
discounts, never talks features, concise).

- **M1, buyer hard constraint.** `eps_buyer` = P(sale | `H`), audit-corrected. Sales here follow
  a seller who is instructed never to discount, so this is an upper bound on the buyer violation
  rate. Text-level version (scorer only, exploratory): P(sale | B-discount, no discount
  offered).
- **M2, seller hard constraint.** `eps_seller` = P(discount offered | seller in {S-features,
  S-concise}). Not audited. On the 160 labelled conversations the detector missed 6 of 78
  offers and raised 2 false alarms in 82, so M2 is more likely understated than overstated.
- **M3, compatibility** (the paper's design claim). For each buyer, sale rate with its
  compatible seller (discount with S-discount, spec with S-features, impatient with S-concise)
  minus the mean sale rate with the other two sellers.
- **M4, budget.** Among sales with an agreed price, the share above the planted budget, overall
  and by budget level. Budget is soft by the generation prompt, so this is descriptive.
- **M5, self-report calibration.** Agreement between the dialogue outcome and a self-report of
  exactly 0 or 1; and the dialogue outcomes of conversations whose self-report is fractional.

Verdict bands for the hard constraints (M1 audit-corrected, M2), by point estimate, flagged
borderline when the interval crosses a band edge: at most 0.05 `HELD`; above 0.05 and at most
0.20 `SOFT`; above 0.20 `NOT_HELD`.

## Predictions

- **P1.** `eps_buyer` (audit-corrected) is above 0.05: the discount-only constraint is not
  `HELD`.
- **P2.** `eps_seller` is above 0.05.
- **P3.** The compatibility gap is at least 0.10 for `B-discount` and below 0.10 for `B-spec`
  and for `B-impatient`.
- **P4.** More than 10% of priced sales close above budget, and the share is highest at the 10k
  budget.
- **P5.** Self-reports of exactly 0 or 1 agree with the dialogue outcome at least 90% of the
  time.

## Procedure

```bash
# after the freeze commit and tag
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.analyze packet --confirmatory \
    --out results/persona_baseline2023_<date>
# label the 180 packet items into <out>/audit_labels.jsonl, then
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.analyze report --confirmatory \
    --out results/persona_baseline2023_<date>
```

`analyze.py` refuses to read CONFIRMATORY without `--confirmatory`. Its tests run on synthetic
conversations only.

## Limits

- Narrator protocol: one model wrote both sides, so this measures whether the narrated story
  respects both personas, not whether a buyer agent holds one under pressure from an
  independent seller.
- The released files hold 4,849 records, not the paper's 19k, and `train.json` was replaced
  after the paper; this may not be the exact data behind the paper's tables.
- `B-impatient` has under 200 train records in two of its three seller pairings.
- All labels (160 for the gate, 180 for the audit) come from one labeler, Claude. The audit is
  blind to cell and scorer call; it is not blind to the conversation's content, which often
  reveals the personas.
- `B-spec` has no hard-constraint measure here: whether a car "meets all specifications" cannot
  be scored from text by rules. It is covered only through M3.
- Persona separability (recovering the persona from behavior) is not in this registration.
