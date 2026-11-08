# Open-world Twenty Questions — pre-registration

Written before the accumulation run. Records the design and the predictions so
the results note can be scored against them rather than rationalised after.

## Setup

The closed-world BED loop (`bed.py`) assumes the secret is one of a fixed 158
objects. Here we drop that: the **true** space is still the 158 (we have a Noul
row for each, so the matrix oracle can answer about any of them), but the
**known** hypothesis set `H_known` starts empty and grows only by expert
intervention. Each game:

1. A secret is drawn from a fixed **Zipf** frequency law over the 158 objects
   (exponent `s`, random object→rank assignment, fixed seed). The Zipf head
   stands in for "common species"; the long tail for the unbounded remainder.
2. The BED loop runs over `H_known ∪ {unknown}`:
   * `unknown` is a catch-all hypothesis carrying **Pitman–Yor** prior mass
     `p_new = (θ + Kσ)/(θ + n)` (K distinct seen, n games so far).
   * Its **background likelihood**: for an ATTRIBUTE question, the mean of the
     Noul rows over currently-known objects (the attribute marginal); for a
     CATEGORY question, the known-set fraction in that category; for a GUESS,
     ≈ε (you cannot guess an unknown by name). All ε-floored.
   * Real known objects use the exact closed-form (category/guess) and cached
     Noul (attribute) likelihoods, unchanged.
   * EIG argmax selects the question (the planner role that solved 158/158 —
     never the model-as-selector that scored 0/158).
3. Commit rule, unified through the existing confidence gate: if a real object
   reaches confidence → guess it (win iff it is the secret); if `unknown`
   reaches confidence (or is all that is left, or the budget ends with `unknown`
   on top) → **call the expert**, who names the secret and we add it to
   `H_known` (fill its Noul row from the matrix; K→K+1).

Everything runs offline on the committed `matrix_jev.json` (deterministic matrix
oracle), so the whole study regenerates with no API.

## Metrics

* **Discovery curve** `K_n` vs `n`; fit PY `(σ, θ)` by sequential-partition MLE;
  compare the fitted `p_new(t)` to the Good–Turing missing mass (singletons/t).
* **Detector ROC**: per game, label = secret-is-out-of-pool, score =
  `max_t p_unknown` over the episode; report AUC, and the TPR/FPR at the
  operational threshold (confidence = 0.9), and the earliness (question index at
  which `unknown` crosses the gate on true-novelty games).
* **Expert-call rate over the stream** (windowed), split into *necessary* calls
  (true novelty) and *false-positive* calls (in-pool game that fired `unknown`).
* **Favors-common**: autonomous-resolution rate (won without an expert) stratified
  by the secret's frequency rank.

## Predictions

* **P1 (heavy-tailed discovery).** `K_n` is sublinear and the PY MLE lands at
  `σ > 0` (not the Dirichlet `σ = 0`); the fitted `p_new(t)` tracks the
  Good–Turing missing mass within sampling noise.
* **P2 (the catch-all is a usable detector).** `max_t p_unknown` separates
  out-of-pool from in-pool with **AUC ≥ 0.90**; on true-novelty games it fires
  before the 20-question budget in the large majority of games (a novel secret
  matches no known row and every by-name guess fails, so mass flows to
  `unknown`).
* **P3 (expert load decays; coverage is size-biased).** The necessary-expert
  rate falls over the stream as the missing mass falls; autonomous-resolution
  rate is (weakly) monotone increasing in the secret's frequency — the common
  head is served without an expert, the rare tail keeps needing one.
* **P4 (false positives are an early, thin-`H_known` phenomenon).** In-pool games
  that wrongly fire `unknown` are rare and concentrated while `H_known` is small
  (the background marginal is estimated from few objects); the FP rate decays as
  `K` grows.

## What would falsify the design

P2 failing (AUC near 0.5, or novelty undetected until the budget) would mean the
attribute-marginal background is too permissive to notice novelty — the
load-bearing calibration choice, and the open-world analogue of the ε-clamp.
P1 landing at `σ ≈ 0` would say the stream is not actually heavy-tailed at this
horizon and "exhaust the space" is the real regime, not perpetual novelty.
