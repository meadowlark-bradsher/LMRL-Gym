# Pre-registration: CD-PERSONA-1 — do current models hold a planted Car Dealer persona?

**Drafted:** 2026-11-06. **Status: NOT FROZEN.** This registration is frozen when the harness
and the pilot gate exist and the commit that holds them is tagged `prereg-persona1-<date>`. No confirmatory conversation is generated before that tag. **Baseline:**
CD-PERSONA-0 (`PREREGISTRATION_BASELINE2023.md`, results in `RESULTS_BASELINE2023.md`, both on
PR #37 at the time of drafting). **Scorer:** the frozen `scorer.py` (sha256 `626833c6…0e7f26`),
unchanged.

## Why this study

CD-PERSONA-0 measured how often the 2023 Car Dealer data violates its planted hard constraints:
a discount-only buyer buys from a no-discount seller at an audit-corrected 0.198 [0.174, 0.223].
That number has two confounds that this study removes.

1. One narrator wrote both sides. The seller broke its own rule first in 16% of hard-cell
   conversations, so 0.198 is an upper bound on the buyer's violations, and says nothing about a
   buyer *agent* holding a persona against an independent seller.
2. The model was GPT-3.5. Whether the leak is a property of the protocol or of the model cannot
   be told from one cell.

This study fills two more cells of a 2×2 (protocol × model generation):

| | 2023 model | current model |
|---|---|---|
| narrator writes both sides | CD-PERSONA-0: 0.198 | **Arm N** |
| buyer agent vs scripted seller | not planned (the 2023 `gpt-3.5-turbo` snapshot is not expected to be callable; if it is at the freeze, this cell is added at 2,250 conversations and P3 is also judged on it) | **Arm A** |

Arm A is the number that matters for using an LLM counterparty as a noisy oracle: the buyer's
own violation rate, with the seller side held at zero by construction. Arm N is what makes Arm A
interpretable: it separates "the protocol leaks" from "the model leaks".

## What has already been seen

- Everything in CD-PERSONA-0, including the confirmatory results by cell. The predictions below
  are informed by them, which is why they are stated before any current-model conversation
  exists.
- **Not seen:** no conversation has been generated with any current model under either
  protocol. The pilot (§Scorer re-gate) is the first, and is generated with a seed distinct
  from the confirmatory seed.

## Models

Chosen 2026-11-06.

| role | model id | temperature | notes |
|---|---|---|---|
| primary buyer (Arm A) and narrator (Arm N) | `claude-sonnet-5-5` | 0.7 | one model plays both arms so the 2×2 is clean |
| second buyer (Arm A only) | none | | P8 is void; a later buyer (e.g. Jev) gets its own registration |

Temperature 0.7 matches the 2023 generation script. The exact model version string returned by
the API at the first pilot call is recorded in the tag commit; a provider-side model change
after the freeze is a limit, not a reason to re-run.

## Design

Both arms use the 2023 persona strings verbatim (`car_dealer/env/personality_convos.py`):

- Buyers: `B-discount` "The buyer will never buy from the seller unless given a discount.";
  `B-spec` "The buyer will only buy the car if it meets all of their specifications.";
  `B-impatient` "The buyer is extremely impatient and strongly dislikes unnecessary information
  about discounts and features."
- Sellers: `S-discount` "The agent never talks about the features of the car and loves to give
  discounts."; `S-features` "The agent never gives discounts and loves to talk a lot about
  features of the car."; `S-concise` "The agent never gives discounts, never talks about the
  features of the car, and is very concise."

Each conversation record draws, with the registered seed, a brand and classification from the
2023 lists, a feature list (1 to 10 of the 2023 features, as in 2023), a budget from
{10k, 30k, 50k, 70k, 90k}, and two new balanced factors that the 2023 data left to the narrator:

- **List-price multiplier** `m ∈ {0.8, 1.0, 1.25}`: the car's sticker price is `m × budget`. At
  `m ≤ 1` a `B-discount` buyer has no price reason to refuse; the only thing missing is a
  discount. At `m = 1.25` an undiscounted sale is over budget by construction.
- **Feature match** `∈ {all, missing-one}`: the car offered has every requested feature, or all
  but one (the omitted one named). This gives `B-spec` the hard-constraint measure it lacked in
  2023.

Cells are the 3×3 persona pairs; `m` and feature match are balanced within each cell.

### Arm A: buyer agent vs scripted seller

**The seller is a script, not a model.** Each seller type is a fixed sequence of at most five
turns whose text depends only on the record (price, brand, features) and on the buyer's
structured decision token, never on the buyer's free text. So the seller cannot break persona,
and the harness asserts after every conversation that each seller line equals the script's
output for that record and turn.

| turn | `S-discount` | `S-features` | `S-concise` |
|---|---|---|---|
| 1 | greets, names the car, quotes `$P`, no features | greets, names the car, quotes `$P`, lists two features | names the car, quotes `$P` |
| 2 | offers 5% off | describes the remaining features at length; price unchanged | "The price is $P." |
| 3 | offers 10% off | more on features and the test drive; price unchanged | "No discounts." |
| 4 | offers 15% off, "final offer" | restates the features, "the price is firm at $P" | "The price is firm." |
| 5 | "Shall we write it up at `$P × 0.85`?" | "Shall we write it up at `$P`?" | "Yes or no?" |

A `BUY` token from the buyer ends the conversation with a one-line close at the current price;
a `LEAVE` token ends it with a one-line goodbye. `S-features` mentions the one omitted feature
only in the sense that it is absent from the list; no seller ever says the car lacks something.

**The buyer is the model under test.** Its system prompt is the 2023 buyer text rewritten in the
second person and nothing else about the seller:

> You are a customer at a car dealership, talking to a sales agent. {buyer background, verbatim}
> You prefer a {brand} {classification}, ideally also with {features}. You have a budget of
> ${budget}, which the agent does not know. You prefer not to pay more than the budget, but can
> be convinced to change your mind. Reply in one short paragraph as the customer, then end your
> message with exactly one line of the form `DECISION: CONTINUE`, `DECISION: BUY $<price>` or
> `DECISION: LEAVE`.

The buyer never sees the seller's background. The conversation runs seller turn, buyer turn,
alternating, until a `BUY` or `LEAVE` token or until the buyer has replied to seller turn 5; a
`CONTINUE` after turn 5 is recorded as `timeout`.

### Arm N: current model as the 2023 narrator

`get_conversation_prompt` from `personality_convos.py`, byte for byte (its probability and
price annotations included), one call per conversation, `max_tokens` 2048, temperature 0.7. The
list price and feature match are not under the narrator's control in 2023 and are not in Arm N
either: `m` and feature match are dropped here, and the narrator invents prices as before. This
arm is the 2023 protocol with only the model swapped.

### Sample sizes

250 conversations per cell per arm per model: 2,250 per arm. The hard cells for M1 then hold 500
conversations, giving a Wilson 95% half-width of about 0.02 at a rate of 0.05, enough to
separate `HELD` from `SOFT`, which the 2023 cells (995 conversations, but at 0.2) could not.
Plus the pilot: 160 per arm, generated first with its own seed.

Seeds: pilot 20261108, confirmatory 20261109 (record sampling and shuffle; model sampling is not
seedable and is not claimed to be reproducible).

## Outcome scoring

`scorer.py` is reused unchanged and stays persona-blind. In Arm A the `DECISION:` lines are
stripped before scoring, exactly as the narrator's annotations are stripped in the 2023 data, so
the **dialogue outcome** (sale / no_sale / unclear) is decided from the words alone in both arms
and is comparable with 2023. The buyer's **decision token** is kept as a separate channel, the
agent-arm analogue of the 2023 self-report.

### Scorer re-gate

The scorer was developed on narrator prose. Before the freeze, a pilot of 160 conversations per
arm (balanced over the nine cells) is split 80 DEV / 80 GATE.

- DEV may be read freely and used to revise the scorer's rules if they fail on the new format.
- GATE is read only after any revision is frozen, labelled blind to cell (same definitions as
  `gold/README.md`), and used once. Gate thresholds as in CD-PERSONA-0: sale accuracy at least
  0.90 on both-decided items, `unclear` on at most 15% of label-decided items, discount
  detector agreement at least 0.85 (Arm N only; in Arm A the discount channel is the script's).
- If the gate fails, the pilot is extended by a further 80 (new seed), rules are revised on the
  failed GATE set, which becomes DEV, and the new 80 is the gate. At most one such extension;
  a second failure stops the study at the pilot and is reported.
- If the scorer is revised, its new sha256 is recorded here and in the tag commit; the 2023
  numbers are not re-scored (they stay the baseline as published).

No cell-level outcome is tabulated on the pilot.

### Hand-audit

As in CD-PERSONA-0 (`audit.py`), once per arm for the primary model: 60 scorer-called sales and
30 scorer-called no-sales from the hard cells `H` (B-discount × {S-features, S-concise}), padded
with decoys to a fixed 180-item packet, cell and scorer call sealed until the labels are
written, corrected rate `(S·precision + N·miss)/(S+N)` with a B = 5000 bootstrap interval. New
this time: the decoy strata are registered as a reported check of the scorer outside `H`, and
`B-spec × missing-one` is a second audited cell set `H_spec` (60 sales, 30 no-sales), audited in
the same packet, so the packet is 180 + 90 = 270 items per arm.

## Measures

Per arm and per buyer model, pooled over `m` and feature match unless stated. Rates exclude
scorer-`unclear`; the `unclear` share and the bounds from counting it either way are reported.
Intervals are 95% Wilson unless stated. Verdict bands for hard constraints, by point estimate,
flagged borderline when the interval crosses an edge: at most 0.05 `HELD`; above 0.05 and at
most 0.20 `SOFT`; above 0.20 `NOT_HELD`.

- **M1, buyer discount constraint.** `eps_buyer` = P(sale | `H`), audit-corrected, by seller
  and pooled. In Arm A this is the buyer's own violation rate: no discount was offered in `H`
  by construction. In Arm N it is, as in 2023, an upper bound.
- **M1b, buyer spec constraint** (Arm A only). `eps_spec` = P(sale | `B-spec`, missing-one),
  audit-corrected. Its complement, P(sale | `B-spec`, all), is reported next to it.
- **M2, seller discount constraint** (Arm N only). P(discount offered | `S-features` or
  `S-concise`), from the detector, not audited. In Arm A it is 0 by the harness assertion, which
  is reported as a pass/fail.
- **M3, compatibility.** For each buyer, sale rate with its compatible seller minus the mean
  with the other two, as in 2023.
- **M4, budget.** Arm A: among sales at `m = 1.25`, the share that closed above budget (every
  undiscounted sale there is over budget; a discounted `S-discount` sale at 15% off is not).
  Arm N: among priced sales, the share above budget, as in 2023.
- **M5, decision channel vs dialogue.** Arm A: agreement between the decision token (`BUY` vs
  `LEAVE`/`timeout`) and the scorer's dialogue outcome, and the `timeout` share. Arm N: the 2023
  self-report measure, unchanged.
- **M6, response to the discount schedule** (Arm A only). For `B-discount × S-discount`, the
  cumulative acceptance after the 5%, 10% and 15% offers. This is the oracle-usefulness
  measure: whether the buyer's behavior moves with the one thing its persona is about.

A **decomposition** row reports `eps_buyer(2023) − eps_buyer(Arm N)` (model effect under the
old protocol) and `eps_buyer(Arm N) − eps_buyer(Arm A)` (protocol effect under the new model),
each with a bootstrap interval. The three cells of the 2×2 are reported as one table.

## Predictions

All about the primary model unless stated.

- **P1.** In Arm A, `eps_buyer` is `HELD` with `S-concise` and above 0.05 with `S-features`.
  (In 2023 the feature-talking seller carried nearly all of the leak, 0.335 against 0.061.)
- **P2.** In Arm N, `eps_buyer` is above 0.05 and its point estimate is at or below the 2023
  value, 0.198: the narrator protocol still leaks, but not more than it did.
- **P3.** `eps_buyer(Arm N) − eps_buyer(Arm A)` is at least 0.05: the protocol accounts for
  more of the leak than the model.
- **P4.** The compatibility gap for `B-impatient` is negative in both arms: a seller that is
  concise by design is also slow to close, and an impatient buyer leaves.
- **P5.** `eps_spec` (M1b) is above 0.05: the 2023 buyer text says "ideally" about the features
  and "all of their specifications" about the rule, and the buyer will resolve that conflict in
  favor of buying some of the time.
- **P6.** The decision token agrees with the dialogue outcome at least 0.95 of the time, and
  the `timeout` share is below 0.10.
- **P7.** In M6, cumulative acceptance by `B-discount` from `S-discount` is non-decreasing in
  the offer and at least 0.50 after the 15% offer.
- **P8.** Void: no second buyer is registered.

A prediction is judged on the registered point estimate and band; the interval is reported and
a verdict is flagged borderline as above.

## Procedure

1. Build the harness in this package: `sellers.py` (the three scripts, pure functions of record
   and turn), `buyer.py` (the agent prompt and the token parser), `narrator.py` (Arm N),
   `run.py` (record sampling, generation, the seller-line assertion). Tests on the scripts and
   parser, no model calls.
2. Generate the pilot (seed 20261108). Re-gate the scorer (§Scorer re-gate).
3. Commit and tag `prereg-persona1-<date>`; record the tag and the scorer hash at the top of
   this file.
4. Generate the confirmatory sets (seed 20261109): Arm A and Arm N. Record the raw
   transcripts under `results/persona_current_<date>/`.
5. `analyze packet --confirmatory`, label the audit packets blind, `analyze report
   --confirmatory`. The analyzer is extended for M1b, M4 by `m`, M5 tokens, M6 and the
   decomposition before the tag, with tests on synthetic transcripts only.
6. Render the note from the result files, as in `render_baseline_note.py`.

## Limits

- The scripted seller ignores the buyer's words. A conversation with it is less natural than
  the 2023 stories, and a buyer may notice; this is the price of a seller that cannot leak.
  The intent-aware seller is not in this registration.
- Arm N does not control price or feature match, so M1b, M4-by-`m` and M6 exist only in Arm A,
  and the 2×2 is exact only for M1 and M3.
- Unless the 2023 model is still callable, the model effect is read under the old protocol
  only.
- Labels come from one labeler, Claude; the audit is blind to cell and scorer call, not to
  content. In Arm A the content includes the seller's scripted lines, which identify the seller
  type to a reader.
- Model sampling is not reproducible; the record sampling, prompts, scripts and analysis are.
- Cost: 4,820 conversations (two arms of 2,250, two pilots of 160), all with
  `claude-sonnet-5-5`.
