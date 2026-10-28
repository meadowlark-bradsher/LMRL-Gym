# Live Car-Dealer, Phase 0 — where is ASK optimal? (exact oracle, no model)

_Rendered by `render_phase0_note.py` from frozen JSON; do not edit numbers by hand._

Source: `results/live_phase0_sweep_20261027/sweep.json` (`sweep_ask_value.py`, env `live-car-dealer-v0.1`, regime easy, opponent shortlist 3, 1728 cells = 72 (kernel, H, c_q) aggregates × 2 focal roles × 2 openers × 6 focal types; 6166.2 s). Every number is **exact-on-model**: the oracle is solved, nothing is sampled.

Quantities: V\*_ask / V\*_noask = the focal negotiator's optimal expected net gain from the opening state with ASK allowed / forbidden at every decision; Δ\*_active = V\*_ask − V\*_noask (≥ 0 by construction); `ask-opt nodes` = focal decision nodes (ASK legal) where ASK is *strictly* optimal, out of all such nodes solved; `root ASK` = cells whose optimal opening move (when the focal opens) is a question.

## 1. Aggregates per (kernel, H, c_q)

| kernel (α₀, β, p_counter, p_hold) | H | c_q | mean Δ\*_active | max Δ\*_active | cells with Δ > 0 | ask-opt nodes | root ASK | mean V\*_ask |
|---|---|---|---|---|---|---|---|---|
| (0.60, 4, 0.5, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/324 | 0/24 | 0.2703 |
| (0.60, 4, 0.5, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/324 | 0/24 | 0.2703 |
| (0.60, 4, 0.5, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/324 | 0/24 | 0.2703 |
| (0.60, 4, 0.5, 1) | 3 | 0 | 0.0332 | 0.0435 | 1.00 | 96/5052 | 12/24 | 0.3583 |
| (0.60, 4, 0.5, 1) | 3 | 1/100 | 0.0232 | 0.0335 | 1.00 | 96/5052 | 12/24 | 0.3483 |
| (0.60, 4, 0.5, 1) | 3 | 3/100 | 0.0069 | 0.0139 | 0.50 | 60/5052 | 0/24 | 0.3321 |
| (0.60, 4, 0.5, 1) | 4 | 0 | 0.0197 | 0.0253 | 1.00 | 1344/83352 | 12/24 | 0.3796 |
| (0.60, 4, 0.5, 1) | 4 | 1/100 | 0.0127 | 0.0172 | 1.00 | 1296/83352 | 0/24 | 0.3727 |
| (0.60, 4, 0.5, 1) | 4 | 3/100 | 0.0040 | 0.0059 | 1.00 | 720/83352 | 0/24 | 0.3639 |
| (0.60, 4, 1, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2477 |
| (0.60, 4, 1, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2477 |
| (0.60, 4, 1, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2477 |
| (0.60, 4, 1, 1) | 3 | 0 | 0.0337 | 0.0492 | 1.00 | 96/1908 | 12/24 | 0.3583 |
| (0.60, 4, 1, 1) | 3 | 1/100 | 0.0237 | 0.0392 | 1.00 | 96/1908 | 12/24 | 0.3483 |
| (0.60, 4, 1, 1) | 3 | 3/100 | 0.0098 | 0.0195 | 0.50 | 60/1908 | 0/24 | 0.3344 |
| (0.60, 4, 1, 1) | 4 | 0 | 0.0189 | 0.0256 | 1.00 | 1128/17148 | 12/24 | 0.3796 |
| (0.60, 4, 1, 1) | 4 | 1/100 | 0.0125 | 0.0185 | 1.00 | 1092/17148 | 0/24 | 0.3732 |
| (0.60, 4, 1, 1) | 4 | 3/100 | 0.0051 | 0.0093 | 1.00 | 696/17148 | 0/24 | 0.3658 |
| (0.60, 8, 0.5, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/324 | 0/24 | 0.2689 |
| (0.60, 8, 0.5, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/324 | 0/24 | 0.2689 |
| (0.60, 8, 0.5, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/324 | 0/24 | 0.2689 |
| (0.60, 8, 0.5, 1) | 3 | 0 | 0.0326 | 0.0433 | 1.00 | 96/5052 | 12/24 | 0.3559 |
| (0.60, 8, 0.5, 1) | 3 | 1/100 | 0.0226 | 0.0333 | 1.00 | 96/5052 | 12/24 | 0.3459 |
| (0.60, 8, 0.5, 1) | 3 | 3/100 | 0.0067 | 0.0133 | 0.50 | 84/5052 | 0/24 | 0.3300 |
| (0.60, 8, 0.5, 1) | 4 | 0 | 0.0213 | 0.0255 | 1.00 | 1344/83352 | 12/24 | 0.3792 |
| (0.60, 8, 0.5, 1) | 4 | 1/100 | 0.0138 | 0.0170 | 1.00 | 1296/83352 | 0/24 | 0.3717 |
| (0.60, 8, 0.5, 1) | 4 | 3/100 | 0.0041 | 0.0055 | 1.00 | 900/83352 | 0/24 | 0.3620 |
| (0.60, 8, 1, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2468 |
| (0.60, 8, 1, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2468 |
| (0.60, 8, 1, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2468 |
| (0.60, 8, 1, 1) | 3 | 0 | 0.0346 | 0.0508 | 1.00 | 96/1908 | 12/24 | 0.3559 |
| (0.60, 8, 1, 1) | 3 | 1/100 | 0.0246 | 0.0408 | 1.00 | 96/1908 | 12/24 | 0.3459 |
| (0.60, 8, 1, 1) | 3 | 3/100 | 0.0104 | 0.0208 | 0.50 | 84/1908 | 0/24 | 0.3317 |
| (0.60, 8, 1, 1) | 4 | 0 | 0.0213 | 0.0272 | 1.00 | 1128/17148 | 12/24 | 0.3792 |
| (0.60, 8, 1, 1) | 4 | 1/100 | 0.0143 | 0.0196 | 1.00 | 1092/17148 | 0/24 | 0.3722 |
| (0.60, 8, 1, 1) | 4 | 3/100 | 0.0060 | 0.0100 | 1.00 | 876/17148 | 0/24 | 0.3639 |
| (0.80, 4, 0.5, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/372 | 0/24 | 0.2703 |
| (0.80, 4, 0.5, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/372 | 0/24 | 0.2703 |
| (0.80, 4, 0.5, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/372 | 0/24 | 0.2703 |
| (0.80, 4, 0.5, 1) | 3 | 0 | 0.0332 | 0.0435 | 1.00 | 96/7296 | 12/24 | 0.3583 |
| (0.80, 4, 0.5, 1) | 3 | 1/100 | 0.0232 | 0.0335 | 1.00 | 96/7296 | 12/24 | 0.3483 |
| (0.80, 4, 0.5, 1) | 3 | 3/100 | 0.0069 | 0.0139 | 0.50 | 60/7296 | 0/24 | 0.3321 |
| (0.80, 4, 0.5, 1) | 4 | 0 | 0.0330 | 0.0479 | 1.00 | 1308/119808 | 12/24 | 0.3796 |
| (0.80, 4, 0.5, 1) | 4 | 1/100 | 0.0206 | 0.0320 | 1.00 | 1284/119808 | 12/24 | 0.3673 |
| (0.80, 4, 0.5, 1) | 4 | 3/100 | 0.0034 | 0.0047 | 1.00 | 708/119808 | 12/24 | 0.3500 |
| (0.80, 4, 1, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2432 |
| (0.80, 4, 1, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2432 |
| (0.80, 4, 1, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2432 |
| (0.80, 4, 1, 1) | 3 | 0 | 0.0337 | 0.0492 | 1.00 | 96/1836 | 12/24 | 0.3583 |
| (0.80, 4, 1, 1) | 3 | 1/100 | 0.0237 | 0.0392 | 1.00 | 96/1836 | 12/24 | 0.3483 |
| (0.80, 4, 1, 1) | 3 | 3/100 | 0.0098 | 0.0195 | 0.50 | 60/1836 | 0/24 | 0.3344 |
| (0.80, 4, 1, 1) | 4 | 0 | 0.0360 | 0.0597 | 1.00 | 1020/16320 | 12/24 | 0.3796 |
| (0.80, 4, 1, 1) | 4 | 1/100 | 0.0252 | 0.0438 | 1.00 | 996/16320 | 12/24 | 0.3688 |
| (0.80, 4, 1, 1) | 4 | 3/100 | 0.0087 | 0.0165 | 1.00 | 624/16320 | 12/24 | 0.3523 |
| (0.80, 8, 0.5, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/372 | 0/24 | 0.2689 |
| (0.80, 8, 0.5, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/372 | 0/24 | 0.2689 |
| (0.80, 8, 0.5, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/372 | 0/24 | 0.2689 |
| (0.80, 8, 0.5, 1) | 3 | 0 | 0.0326 | 0.0433 | 1.00 | 96/7296 | 12/24 | 0.3559 |
| (0.80, 8, 0.5, 1) | 3 | 1/100 | 0.0226 | 0.0333 | 1.00 | 96/7296 | 12/24 | 0.3459 |
| (0.80, 8, 0.5, 1) | 3 | 3/100 | 0.0067 | 0.0133 | 0.50 | 84/7296 | 0/24 | 0.3300 |
| (0.80, 8, 0.5, 1) | 4 | 0 | 0.0351 | 0.0481 | 1.00 | 1308/119808 | 12/24 | 0.3792 |
| (0.80, 8, 0.5, 1) | 4 | 1/100 | 0.0223 | 0.0319 | 1.00 | 1284/119808 | 12/24 | 0.3664 |
| (0.80, 8, 0.5, 1) | 4 | 3/100 | 0.0042 | 0.0043 | 1.00 | 876/119808 | 0/24 | 0.3483 |
| (0.80, 8, 1, 1) | 2 | 0 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2423 |
| (0.80, 8, 1, 1) | 2 | 1/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2423 |
| (0.80, 8, 1, 1) | 2 | 3/100 | 0.0000 | 0.0000 | 0.00 | 0/96 | 0/24 | 0.2423 |
| (0.80, 8, 1, 1) | 3 | 0 | 0.0346 | 0.0508 | 1.00 | 96/1836 | 12/24 | 0.3559 |
| (0.80, 8, 1, 1) | 3 | 1/100 | 0.0246 | 0.0408 | 1.00 | 96/1836 | 12/24 | 0.3459 |
| (0.80, 8, 1, 1) | 3 | 3/100 | 0.0104 | 0.0208 | 0.50 | 84/1836 | 0/24 | 0.3317 |
| (0.80, 8, 1, 1) | 4 | 0 | 0.0383 | 0.0612 | 1.00 | 1020/16320 | 12/24 | 0.3792 |
| (0.80, 8, 1, 1) | 4 | 1/100 | 0.0269 | 0.0449 | 1.00 | 996/16320 | 12/24 | 0.3679 |
| (0.80, 8, 1, 1) | 4 | 3/100 | 0.0095 | 0.0170 | 1.00 | 804/16320 | 12/24 | 0.3504 |

## 2. Read-out (computed from the table)

- **H = 2:** Δ\*_active > 0 in 0/24 (kernel, c_q) aggregates; mean Δ ranges 0.0000–0.0000; ASK is strictly optimal in 0/5328 ASK-legal nodes across every kernel and cost, so no ask/offer trade-off exists at this horizon.
- **H = 3:** Δ\*_active > 0 in 24/24 (kernel, c_q) aggregates; mean Δ ranges 0.0067–0.0346; the largest is kernel `a0.60_b8_pc1_ph1_k3` at c_q = 0 (mean 0.0346, ASK strictly optimal in 96/1908 nodes).
- **H = 4:** Δ\*_active > 0 in 24/24 (kernel, c_q) aggregates; mean Δ ranges 0.0034–0.0383; the largest is kernel `a0.80_b8_pc1_ph1_k3` at c_q = 0 (mean 0.0383, ASK strictly optimal in 1020/16320 nodes).
- **Cost:** mean Δ\*_active is non-increasing in c_q in 48/48 adjacent (kernel, H) pairs (a structural expectation; violations would indicate a bug).
- **Horizon:** mean Δ\*_active increases with H in 29/48 adjacent (kernel, c_q) pairs over the horizons swept (H ∈ [2, 3, 4]). The design document's VOI-H2 predicts a *non-monotone* dependence once offers alone suffice. At c_q = 0 the decrease from H = 3 to H = 4 already appears for 5/8 kernels (`a0.60_b4_pc0.5_ph1_k3`, `a0.60_b4_pc1_ph1_k3`, `a0.60_b8_pc0.5_ph1_k3`, `a0.60_b8_pc1_ph1_k3`, `a0.80_b4_pc0.5_ph1_k3`); the remaining 3 (`a0.80_b4_pc1_ph1_k3`, `a0.80_b8_pc0.5_ph1_k3`, `a0.80_b8_pc1_ph1_k3`) still increase, so their peak lies at H ≥ 4 and is not located by this sweep.
- Best aggregate (`a0.80_b8_pc1_ph1_k3`, H = 4, c_q = 0), focal = buyer, opener = buyer: mean Δ\*_active over the 6 focal types 0.0612; optimal opening moves ['ASK top:delivery', 'ASK top:price', 'ASK top:warranty'].
- Best aggregate, focal = buyer, opener = seller: mean Δ\*_active over the 6 focal types 0.0154; the focal has no opening move (the opponent opens), so no root action is reported.
- Best aggregate, focal = seller, opener = buyer: mean Δ\*_active over the 6 focal types 0.0154; the focal has no opening move (the opponent opens), so no root action is reported.
- Best aggregate (`a0.80_b8_pc1_ph1_k3`, H = 4, c_q = 0), focal = seller, opener = seller: mean Δ\*_active over the 6 focal types 0.0612; optimal opening moves ['ASK top:delivery', 'ASK top:price', 'ASK top:warranty'].

## 3. What this settles for Phase 1

- Gate 7 of the design document is **passed**: 48/72 aggregates have states where ASK is strictly optimal, so a confirmatory `(H, c_q, r)` cell with a genuine ask/offer trade-off exists under the clean T1 kernel.
- The exact oracle is the planner at H ≤ 4, so the registered `C_plan − B_plan` contrast collapses to Δ\*_active in those cells; the budgeted-planner fairness questions of section 5 apply only to H = 6 and T2.
- Nothing here is a research claim about any model; no registration has been made and no Jev or LLM call has been placed.

## Regenerate

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.sweep_ask_value --out results/live_phase0_sweep_20261027/sweep.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase0_note --sweep results/live_phase0_sweep_20261027/sweep.json --out llm_rl_scripts/car_dealer_live/PHASE0_RESULTS.md
```
