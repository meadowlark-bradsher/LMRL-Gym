# ST-4 Realistic answerer errors — pre-registration

**Status: DRAFT, not registered.** Frozen when tagged `prereg-stress-st4-<date>`.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.

## Question

Every noise result so far flips attribute answers independently, at a fixed rate, the
same for every object, while guesses and category answers stay exact. Real answerers
err in structured ways. They are unsure exactly where the planner's row is unsure. They
are wrong about one object consistently, so re-asking does not help. They have bad
games. They are a different model from the one that built the matrix. And sometimes
they mishear a name. This study replaces i.i.d. flips with each of these and asks which
of the earlier conclusions survive.

## Setup

The retraction study's setup, unchanged except for the answer channel (ST-0b) and the
outcome scoring (ST-0c): accumulation mode on the real matrix, Zipf 1.1, seed 7, 2000
games, budget 20, confidence 0.90, ε = 0.05. Backends `soft`, `hard`, `modeswitch`
(K = 3). 3 noise seeds per cell.

### Answerer arms

| arm | model | levels |
|---|---|---|
| A0 | i.i.d. attribute flips (anchor; the existing `noisy_matrix_oracle`) | 0.05, 0.10 |
| A1 | **cross-model**: answers are a 0.5 threshold of a second model's Noul matrix, built with `matrix.py`'s prompt on a different backbone, chosen at registration | one level, at its measured disagreement rate δ |
| A2 | **sticky**: per object, m cells chosen once (seeded) are always answered wrong | m = 1, 2, 4 |
| A3 | **sampled**: each attribute answer is drawn yes ~ Bernoulli(p) from the belief matrix | one level (the uniform-cell flip rate is 0.090) |
| A4 | **bursty**: each game is "confused" with probability b and flips at 0.30 inside it; clean otherwise | b chosen so the marginal rate matches A0's 0.05 and 0.10 (b = 1/6, 1/3) |
| G | **guess noise**: by-name guesses get false-yes g_fy (only when the guessed known is within Hamming 2 of the secret) and false-no g_fn (on the true name); attributes exact | (g_fy, g_fn) ∈ {(0.01, 0), (0.05, 0), (0, 0.05), (0.05, 0.05)} |

For each A-arm, the realised flip rate on the questions actually asked is recorded. An
**A0 cell at that same rate** is run as its matched control. The design compares each
structured arm against i.i.d. noise of the same realised strength, so "more noise" is
not mistaken for "differently shaped noise".

**A1 measurement first.** Before the A1 arm is registered, the second matrix is built
(~158 calls) and δ (the binarized disagreement with `matrix_jev.json`) is recorded,
together with how it splits across ambiguous and confident cells. A1's prediction is
written against the measured δ.

## Metrics

The shared scorecard, including `false_confirm`. In addition:

* the **realised** flip rate per arm and its matched-A0 delta for each metric
* the **per-object failure distribution** (Gini coefficient of in-pool failures across
  objects), for A2 and A1
* `modeswitch`'s recovery fraction, (fail_hard − fail_ms) / fail_hard, per arm

## Predictions

* **P1 — sampled answers hurt `hard` more than `soft` (A3).** At a matched realised
  rate, `soft`'s in-pool failure under A3 is lower than under A0: its graded update
  already discounts the ambiguous cells where A3's errors concentrate. `hard`'s is not
  lower, because it kills on the binarized side regardless of p. So the `soft`–`hard`
  gap is wider under A3 than under matched A0.
* **P2 — sticky errors are recoverable once, not twice (A2).** At m = 1, `modeswitch`'s
  recovery fraction is ≥ its matched-A0 value: a sticky cell gives the truth a single
  killer, and the exact by-name test rescues it. At m ≥ 2 its recovery is below 0.1 of
  the m = 1 value: two killers fail the ≤ 1-killer reinstate criterion. In-pool
  failures concentrate on objects (Gini at m = 1 exceeds matched A0's by ≥ 0.2), so
  the same object fails on every recurrence.
* **P3 — bursty noise is kinder to `hard` and harder on `modeswitch` (A4).** At a
  matched marginal rate, `hard`'s in-pool failure is lower than under A0, because fewer
  games are affected. `modeswitch`'s recovery fraction is also lower, because a
  confused game strikes the truth more than once.
* **P4 — a second model is worse than its rate suggests (A1).** The measured
  disagreement δ is between 0.05 and 0.15, well above Jev's 0.008 self-inconsistency.
  `hard`'s in-pool failure under A1 exceeds matched A0 at rate δ, because cross-model
  disagreements are sticky per object (P2's mechanism). `soft` lands within ±0.03 of
  matched A0.
* **P5 — any false-yes on a guess breaks zero-wrong (G, headline).** With g_fy = 0.01,
  `hard` and `modeswitch` both have `false_confirm` > 0. Their `false_confirm` rate is
  about g_fy × (mean number of by-name guesses per in-pool game of knowns within
  Hamming 2 of the secret). `modeswitch`'s is ≥ `hard`'s, because its reinstate tests
  are extra by-name guesses. The rescue mechanism is also the exposure.
* **P6 — false-no on the truth is recoverable by `modeswitch` only (G).** At
  g_fn = 0.05, `hard` turns those games into safe escalations. In `modeswitch` the
  struck truth has one killer (the guess), so it is a reinstate candidate and gets
  re-tested by name. Reinstate picks among single-killer tombstones in name order, so
  the truth is not always the first one probed within K = 3. Prediction:
  `modeswitch`'s in-pool failure is below `hard`'s by ≥ 0.5 × g_fn × P(truth guessed).
* **P7 — the detector ranking holds across arms.** AUC stays ≥ 0.90 for every backend
  in every arm except A3, where the floor is 0.85. The stress lands on the operating
  point (wrong outcomes, escalations), as it did for i.i.d. noise in §3.11 §4.

## What would falsify, and what it would mean

* P5 failing (no `false_confirm` at g_fy = 0.01): the discrete backends rarely guess a
  near neighbour of the truth by name, and their exposure to guess noise is smaller
  than the design note assumes.
* P2 failing at m ≥ 2 (recovery stays high): the reinstate criterion is looser in
  practice than its code reads, because games rarely ask both sticky cells. Recorded as
  a finding about question coverage, not about the criterion.
* P1 failing: graded beliefs do not buy robustness to answerer uncertainty. That would
  weaken the reading that `soft`'s advantage on the measured axes comes from its
  graded update.

## Out of scope

Adversarial answerers (lying to maximise damage) are not modelled. Each arm is a
plausible honest-but-imperfect answerer. An adversary study needs its own threat model.
