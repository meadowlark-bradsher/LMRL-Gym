# Experiment: Is RLCD buying anything? `Noul` vs LLM menu-probabilities on Chess

_Pre-registration, 2026-10-26. Predictions and the decision rule are fixed before any
new data are read._

## 1. Question
On the same 200 fixed positions (P1 `results/primary_20261025/`) and the same legal-move
menu, is Jev's `Noul` a **better confidence signal** — better calibrated, more
discriminative, or a better abstain gate — than the probability a menu-constrained LLM
already emits for free? If not, RLCD has not been shown to buy anything here.

## 2. Hypotheses
- **H1 (RLCD advantage):** `Noul` has lower calibration error *and* equal-or-higher
  discrimination than the LLM's menu probability, on the same event.
- **H0 (no advantage):** paired differences on both axes include zero.
- **H2 (LLM advantage):** the LLM probability is better on at least one axis with `Noul`
  no better on the other.
A split result is reported as such, not collapsed into H1.

## 3. Amendment (2026-10-26, before any data) — logprob feasibility
Verified against the live APIs before running:
- **`L-logprob` is impossible for `claude-opus-5-5`:** the Anthropic Messages API rejects
  `logprobs`/`top_logprobs` with HTTP 400 ("Extra inputs are not permitted"). There is no
  logprob surface on Claude.
- **OpenAI `top_logprobs` caps at 20;** single-token labels exist only for `A`–`Z`.
Resolution (chosen by the owner): **`L-logprob` = gpt-4o only** (the pre-reg's designated
*fair baseline*); Claude participates via `L-verbal` only. The headline H1 test is
**`Noul` vs gpt-4o menu-logprob**. For `L-logprob`, label legal moves `A`–`Z`, read the
top-20 logprobs, floor legal labels absent from the top-20, renormalize over legal
labels, and **report per-position logprob coverage**; positions with >20 (and >26) legal
moves are caveated, not dropped. This drops one model×readout cell and bounds another; the
core question (is `Noul` better than a free LLM menu-probability?) is unchanged because
gpt-4o logprob *is* the fair baseline.

## 4. Conditions
| id | model | readout |
|---|---|---|
| J-noul | `jev-latest` | native per-option `Noul` from `Choice`; **K=5** queries, mean per option (per-call SD recorded) |
| L-logprob | `gpt-4o` | one call, temp 0, `logprobs`, `top_logprobs=20`; renormalize legal labels (floor unlisted) |
| L-verbal-gpt | `gpt-4o` | chosen label + a probability in [0,1] it is engine-best; parse the number (top option only) |
| L-verbal-claude | `claude-opus-5-5` | same verbal readout |

One prompt family: FEN + enumerated legal moves (`A`–…) + the terse P1/P2 objective line.
Menu order is **randomized per position per model** with a fixed seed; the permutation is
logged.

## 5. Event (pre-registered, scored from the frozen Stockfish grader)
- **Primary — acceptable move:** `cp_lost ≤ 50`. A probability `p` on an option predicts it
  is acceptable.
- **Secondary — engine-best:** `cp_lost = 0` (match@1), for continuity with §3.2.
Sensitivity at 25 and 100 reported in an appendix, not used for the decision. The grader is
the frozen `results/primary_20261025/eval_cache.json` (per-move cp, depth 16, hash
`a9d18ee1…`), read-only.

## 6. Metrics (paired bootstrap over positions, B=5000, seed 0; 95% CIs)
- **Calibration:** ECE (10 equal-mass bins) and Brier over (option, p, event) triples;
  reliability diagram per condition. (Full-distribution ECE compares J-noul vs L-logprob;
  L-verbal has a probability only for the chosen option, so its calibration is over the
  top-option event.)
- **Discrimination:** AUROC of the **top option's** probability for predicting the chosen
  move is acceptable (comparable across all conditions); quintile lift (top-quintile
  acceptable-rate minus mean of the bottom four).
- **Gate utility:** selective accuracy at coverage τ ∈ {50,70,80,90} percentiles of the
  top-option probability, and its paired difference across conditions.
- **Secondary (recorded, not decisive):** wall-clock latency and billed cost per call.

## 7. Predictions and decision rule
- Under **H1**: `Noul` ECE lower than gpt-4o-logprob by ≥0.05 absolute (paired CI excludes
  0) **and** AUROC paired difference ≥0 (CI not excluding positive values).
- Under **H0**: both paired CIs include zero.
- `L-verbal` expected worst-calibrated (overconfident); if it beats `Noul`, reported as
  surprising.

**Decision rule for FINDINGS.md** (applied to the primary event; secondary checked
independently; primary decides on disagreement):
- **H1** → §4.5 keeps its claim and gains the comparison; §1 "confidence signal" becomes
  "a confidence signal better calibrated than a constrained LLM's".
- **H0** → §4.5 rewritten: `Noul` informative but no better than free LLM logprobs; Jev's
  case is cost/latency, now with measured numbers; §8 Chess row drops the `Noul` clause.
- **H2** → §4.5 reversed; abstain/escalate recommended on the LLM's logprob.

## 8. Power
200 positions fixed. Paired bootstrap detects ECE ≈0.05 and AUROC ≈0.07 at conventional
power given P1 variance; smaller true advantages return as H0 with a bounding CI. An RLCD
advantage too small to see on 200 positions is too small to change a recommendation.

## 9. Confounds / controls
Randomized menus (permutation logged; top-option-vs-position check); one prompt family;
K=5 averaging with within-position `Noul` SD reported; single-token label verification
(fallback noted); grader frozen before any readout; near-tie stratum (1 vs ≥2 acceptable
moves) in the appendix.

## 10. Outputs
`confidence_h2h/run.py` (→ `raw/<condition>.jsonl`), `analyze.py` (→ `results.json`,
`results.md`), `plot.py` (→ `figures/confidence_h2h.png`). FINDINGS.md patched per §7 from
`results.json`, never retyped. One PR: code, raw data, results, figure, doc edit.

## 11. Scope
Chess only; not a decision-quality test (P2 settled that); `Choice`+`Noul` only.
