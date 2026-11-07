# Car Dealer persona planting: paper vs code, and the rebuild target

Status: reading notes + rebuild target for a persona-planting reliability audit.
Nothing here is pre-registered and nothing has been run. Open decisions are in §6.

Sources
- Paper: Abdulhai et al., *LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning
  with Language Models*, arXiv 2311.18232v1 (Nov 2023). Local copy:
  `2311.18232v1 (2).pdf` in the main checkout root (untracked). Page numbers below are
  PDF pages.
- Code: `llm_rl_scripts/car_dealer/` as inherited from upstream (last upstream change
  `911d726`); unchanged by this fork.

## 1. What the paper specifies

| Item | Paper | Where |
|---|---|---|
| Personas | 3 buyer x 3 seller "strategies". Buyers: loves discounts / wants features / impatient. Sellers: gives discounts / talks features / very concise | C.8, p19 |
| Intent | each seller sells best to one buyer persona and "often fails" with the others, so the seller must infer the persona | C.8, p19 |
| Hidden state | brand, feature list, budget (10k-90k in the samples) | C.8 samples, p19-21 |
| Data pipeline | GPT-3.5 seeds conversations; GPT2 models fine-tuned on each side; the two talk to each other to produce the dataset ("a similar process" to 20Qs) | §4.3, p6 |
| Dataset | 19k trajectories, avg length 16.5, success rate 0.53, avg return 0.562 (std 0.422) | Table 1, p7 |
| Model size | gpt2-xl for Car (BC, %BC top-10% filter, MC, ILQL, PPO) | Table 3, p22 |
| Reward | "defined in Appendix C" (Table 1 caption), but C.8 gives no reward function | p7, p19 |
| Outcome label | buyer/narrator text: "probability of buying: X%, purchase price: $Y" | samples, p19-21 |

Results as printed (Car row):

| | BC | %BC | MC | ILQL | PPO | Online %BC | GPT-4 |
|---|---|---|---|---|---|---|---|
| Table 2, "normalized" (p8) | 44.5 | 54.8 | 57.2 | 46.3 | 50.5 | 40.4 | 53.5 |
| Table 4, "raw" (p23) | 44.5 | 54.8 | 57.2 | 46.3 | 50.5 | blank | blank |

Table 5 (normalization constants, p23) gives Car min = 0 and leaves dataset-average
and max blank. Checked against the rendered page, not only the extracted text.

**Consequence: the Car row cannot be read as "50 = dataset average".** The raw and
normalized rows are identical, the constants are missing, and the values are on a
different scale from Table 1's avg return (0.562). They are consistent with mean
revenue in $k under the code's `revenue` mode (budgets average $50k), but the paper
does not say. Any statement about how far the 2023 methods sat above the dataset
average is unsupported.

## 2. What the code contains: two incompatible generations

**Gen A, the runnable environment** (`env/data.py`, `env/env.py`, `env/buyer.py`)
- `BuyerInfo` = `personality` (one of 10 tone adjectives: abusive, angry, insulting,
  polite, respectful, rude, sarcastic, talkative, toxic, uncommunicative),
  `preferred_brands`, `preferred_type`, 1-4 `preferred_features`, `budget` in
  {10k, 30k, 50k, 70k, 90k}.
- Planting = one header line fed to a fine-tuned GPT-2 buyer:
  `Personality: {p}. Prefers {brand} {type} with {features}. Budget: ${b}`.
- Outcome = buyer emits `Output: Decision=(Accept|Reject) MSRP=$.. [Buy Price=$..]`,
  regex-parsed; at `max_conversation_length` the env force-prompts `Output: Decision=`;
  an unparseable ending scores reward 0.
- Reward: `fancy` (default) = `buy_price / ((budget + msrp)/2)` if bought, else
  `-(budget - msrp)/msrp`; `revenue` = `buy_price/1000`.
- No strategy personas, no seller persona, anywhere in this path.

**Gen B, the paper's persona protocol** (`env/personality_convos.py`)
- The 3 buyer and 3 seller background strings match the paper's samples verbatim.
- One `gpt-3.5-turbo` call (temperature 0.7) writes the **whole conversation, both
  sides**, and is asked to print purchase probability and price after each buyer turn.
  There is no buyer agent holding a persona; a single narrator holds both.
- Budget is soft by instruction: "prefers not to pay more than the budget, but can be
  convinced to change their mind."
- Outcome = free-text regex scrape (`get_purchase_prob`, `get_purchase_price`).
- Not runnable as checked in: `return None` at module level (SyntaxError), `time` not
  imported. `classification` is drawn once, before `random.seed`, so every
  conversation in a run shares one unseeded body style. The 3x3 loop variables are
  overwritten by `random.choice`, so the grid is sampled, not balanced.

## 3. Paper vs code

| Topic | Paper | Code | Status |
|---|---|---|---|
| Personas | 3x3 strategy backgrounds | Gen B only; Gen A uses 10 tone adjectives | env that produced the scores does not carry the paper's personas |
| Seed generator | "GPT-3.5" (text-davinci-003 named for 20Qs) | gpt-3.5-turbo chat, single narrator | minor mismatch |
| Simulator buyer | GPT2 fine-tuned on buyer side | `BatchedGPT2BuyerPolicy`, gpt2-xl in the shell script | agrees; checkpoints not in repo |
| Outcome format | "probability of buying / purchase price" | Gen A `Decision/MSRP/Buy Price`; Gen B prob/price | the conversion between the two is not in the repo |
| Reward | unspecified; README says final purchase price | `fancy` default, `revenue` optional | which one produced Table 2 is unknown |
| Normalization | 0 / 50 / 100 | n/a | constants unpublished for Car |
| Dataset | 19k | released files hold 4,849 records in Gen B format (see §3a) | count does not match the paper |

### 3a. Released dataset, checked 2026-11-06

`https://rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset/car-dealer/` is live:
`train.json` (34,049,161 bytes, sha256 `20c0ab13…d7e743`, modified 2024-07-03),
`eval.json` (3,477,865 bytes, sha256 `bf244da9…014eab`, modified 2023-11-07), and
`simulator/model/` (a 5.9 GB `params.msgpack`, not downloaded).

- Records: train 4,399 (3,799 distinct; 600 exact duplicates), eval 450 (all distinct).
- Every record has `buyer_info` = {`background_buyer`, `background_seller`,
  `preferred_brands`, `preferred_features`, `budget`}, plus `original_conversation`
  and `lines`. All background strings are one of the paper's 3 + 3. No Gen A fields.
- These are the single-narrator GPT-3.5 conversations, not GPT-2 self-play.
- 3x3 coverage: eval is roughly balanced (43-61 per cell). Train is not: the impatient
  buyer has 789 records against ~1,800 for each other buyer, with 190 and 182 in the
  impatient x concise and impatient x features cells.
- Budgets 10k-90k roughly uniform; 1-10 features.
- Outcome labels are the regex-scraped self-report, per line. Final-line
  `purchase_prob` in train: 0 in 1,802, 1 in 1,246, strictly between in 1,245, above 1
  in 106 (scrape errors such as 9, 28, 75). Eval: 161 / 129 / 144 / 16.
- Against Table 1: eval mean lines 16.5 (paper 16.5) and share with final probability
  >= 0.5 is 0.521 (paper success rate 0.53). Mean final probability 0.50 (sd 0.45) is
  on the scale of the paper's avg return 0.562 (sd 0.422) but is not the same number.

## 4. What the paper's own samples show about adherence

Six printed conversations state a budget (p19-21, p25-26). Five end in a sale; two of
those five are above the stated budget (10k budget pays 15k to a seller who only
repeats the price; 70k budget pays 75k). The authors present the first as intended
behavior, and the Gen B prompt permits it. The one hard-constraint case shown (discount
-only buyer vs no-discount seller) holds: no sale. n = 6, author-selected, so this
fixes nothing except the definitions in §5: budget was never a hard constraint, the
background strings are.

## 5. Rebuild target

Goal: measure whether a planted persona governs buyer behavior, scored without the
buyer's self-report, for the paper's protocol and for current models.

Planted type (from Gen B): background string (3), brand, feature subset, budget.

Checkable predictions the background strings make under scripted sellers:
- discount buyer + seller that never discounts -> no sale
- specification buyer + car missing a listed feature -> no sale
- impatient buyer + verbose seller -> fewer sales / earlier exit than with a concise one
- budget -> acceptance falls as price rises past budget (a curve, since budget is soft)

Because the seller is scripted, "was a discount offered", "does the car meet the
spec" and "price vs budget" are known exactly. Ground truth never depends on the
buyer grading itself.

Measurements per (model, persona):
1. hard-constraint violation rate (first two predictions)
2. budget response curve: P(accept) vs price/budget
3. separability: can the persona be recovered from behavior under a fixed seller script
4. self-report calibration: stated probability/price vs the scored outcome
5. parse rate of the outcome line

Conditions that separate "old model" from "old protocol":
- narrator (one call writes both sides, as in Gen B) vs separated buyer agent
- free-text outcome vs structured outcome
- 2023 behavior vs current model

## 6. Open decisions

1. 2023 baseline source: DECIDED, the released dataset (§3a). It supports an offline
   narrator-protocol baseline only. Outcomes must be re-scored from the conversation
   text, because 28% of final self-reported probabilities are fractional and about 2.5%
   are corrupt. Where the files should live (they are not in the repo) is open.
2. Which current model(s) play the buyer, and the budget for calls.
3. Whether the Gen A tone personalities are in scope or only the paper's 3x3.
4. Thresholds for calling persona planting "reliable enough to serve as an oracle",
   to be fixed before any run.
