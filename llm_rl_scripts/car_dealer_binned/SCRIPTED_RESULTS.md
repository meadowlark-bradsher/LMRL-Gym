# Binned Car Dealer, Condition B — scripted-arm results (no model)

_Rendered by `render_scripted_note.py` from frozen JSON; do not edit numbers by hand._

Every number here is **exact-on-logs** except the `random` arm (a seeded draw, 20 plays per cell).
Game `binned-car-dealer-b-v0.1`; design in [`DESIGN_B.md`](DESIGN_B.md). 72 cells = 2 roles × 6 own
types × 6 opponent types; the deterministic arms play each cell once.

## 1. Horizon × regime sweep — when does the ask-vs-act trade-off bind?

Source: `results/binned_b_sweep_20261026/sweep.json` (`sweep_horizon.py`, seed 0). V = exact reference value from the uniform prior,
averaged over the 72 cells; arms are realised means over the same cells. `greedy<exact` counts cells in which
greedy-EIG strictly underperforms the exact policy.

| regime | rounds | V_menu | V_full | exact | greedy_eig | myopic | random | exact asks | exact offers | greedy asks | greedy<exact cells | menu costless |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | 1 | 0.6855 | 0.7308 | 0.6855 | 0.6218 | 0.6487 | 0.6064 | 1.00 | 1.00 | 1.00 | 32/72 | no |
| easy | 2 | 0.7756 | 0.7756 | 0.7756 | 0.7171 | 0.7167 | 0.6622 | 1.25 | 1.33 | 2.00 | 16/72 | yes |
| easy | 3 | 0.7821 | 0.7821 | 0.7821 | 0.7821 | 0.7756 | 0.6849 | 0.67 | 2.00 | 3.00 | 0/72 | yes |
| easy | 4 | 0.7821 | 0.7821 | 0.7821 | 0.7821 | 0.7756 | 0.6863 | 0.00 | 2.67 | 4.00 | 0/72 | yes |
| easy | 5 | 0.7821 | 0.7821 | 0.7821 | 0.7821 | 0.7756 | 0.6863 | 0.00 | 2.67 | 5.00 | 0/72 | yes |
| easy | 6 | 0.7821 | 0.7821 | 0.7821 | 0.7821 | 0.7756 | 0.6863 | 0.00 | 2.67 | 6.00 | 0/72 | yes |
| hard | 1 | 0.7474 | 0.7474 | 0.7474 | 0.6932 | 0.7474 | 0.6829 | 0.00 | 1.67 | 1.00 | 16/72 | yes |
| hard | 2 | 0.7583 | 0.7583 | 0.7583 | 0.7340 | 0.7519 | 0.7194 | 1.00 | 1.50 | 2.00 | 8/72 | yes |
| hard | 3 | 0.7583 | 0.7583 | 0.7583 | 0.7583 | 0.7519 | 0.7259 | 0.00 | 2.50 | 3.00 | 0/72 | yes |
| hard | 4 | 0.7583 | 0.7583 | 0.7583 | 0.7583 | 0.7519 | 0.7259 | 0.00 | 2.67 | 4.00 | 0/72 | yes |
| hard | 5 | 0.7583 | 0.7583 | 0.7583 | 0.7583 | 0.7519 | 0.7259 | 0.00 | 2.83 | 5.00 | 0/72 | yes |
| hard | 6 | 0.7583 | 0.7583 | 0.7583 | 0.7583 | 0.7519 | 0.7259 | 0.00 | 3.00 | 6.00 | 0/72 | yes |

**Read-out (exact-on-logs; every list below is computed from the table).**
- Menu rule: V_menu = V_full in 11/12 rows; the exception(s) [('easy', 1)] — at a 1-decision-plus-final horizon the two-question menu can omit the single best probe.
- Easy regime: greedy-EIG underperforms the exact policy at rounds ∈ [1, 2] and equals it in every cell at rounds ∈ [3, 4, 5, 6]; the exact policy (ties → act) asks no questions at rounds ∈ [4, 5, 6]; the never-ask `myopic` arm equals the exact policy at rounds ∈ [].
- Hard regime: greedy-EIG underperforms the exact policy at rounds ∈ [1, 2] and equals it in every cell at rounds ∈ [3, 4, 5, 6]; the exact policy (ties → act) asks no questions at rounds ∈ [1, 3, 4, 5, 6]; the never-ask `myopic` arm equals the exact policy at rounds ∈ [1].
- Reading: once the horizon has slack, an offer is the better probe because it both tests a type set and can close, so greedy-EIG's extra questions cost nothing (easy, rounds ≥ 3) and the trade-off that Condition B exists to measure only binds at short horizons.
- Consequence for the document's **H2** ("greedy EIG can underperform the optimal policy in Condition B"): true at the 2-round horizon in both regimes, false at the document's 6-round horizon in both regimes. The primary pre-registered Jev horizon is therefore 2 rounds (`PREREGISTRATION_B.md`).

## 2. 2 rounds, easy (primary pre-registered horizon)

Source: `results/binned_b_scripted_r2_easy_20261026/analysis.json` (`analyze_condition_b.py`, B = 5000, seed 0); 1656 episodes.

| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret decisions | Σ EIG regret (bits) | questions | offers | C_H (bits) | final H (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | 72 | 0.7756 | 0.9779 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.3333 | 1.25 | 1.33 | 5.170 | 0.667 |
| `greedy_eig` | 72 | 0.7171 | 0.7778 | 0.778 | 0.0585 | 0.0585 | 0.741 | 0.0000 | 2.00 | 1.00 | 5.059 | 0.222 |
| `myopic` | 72 | 0.7167 | 0.6667 | 0.667 | 0.0590 | 0.0590 | 0.231 | 0.2075 | 0.00 | 2.17 | 5.377 | 0.667 |
| `random` | 1440 | 0.6556 | 0.6527 | 0.922 | 0.1201 | 0.1193 | 0.503 | 0.5650 | 0.85 | 1.25 | 5.305 | 1.249 |

Paired, cell-clustered bootstrap, arm − `exact` (for the deterministic arms the interval expresses variation across the type space, not sampling noise):

| arm − exact | Δ utility | Δ gain | Δ agreement | Δ Σ decision regret | Δ Σ EIG regret (bits) | Δ questions |
|---|---|---|---|---|---|---|
| `greedy_eig` | -0.0585 [-0.0923, -0.0293] | -0.2002 [-0.3036, -0.1067] | -0.2222 [-0.3194, -0.1250] | +0.0585 [+0.0415, +0.0763] | -0.3333 [-0.4444, -0.2222] | +0.7500 [+0.6528, +0.8472] |
| `myopic` | -0.0590 [-0.0799, -0.0386] | -0.3113 [-0.4173, -0.2093] | -0.3333 [-0.4444, -0.2361] | +0.0590 [+0.0502, +0.0677] | -0.1258 [-0.2264, -0.0260] | -1.2500 [-1.3472, -1.1528] |
| `random` | -0.1201 [-0.1351, -0.1050] | -0.3252 [-0.3584, -0.2909] | -0.0785 [-0.1069, -0.0521] | +0.1193 [+0.1111, +0.1273] | +0.2316 [+0.1363, +0.3269] | -0.4000 [-0.5028, -0.2951] |

By role (`exact` arm): buyer U = 0.7756, asks 1.25; seller U = 0.7756, asks 1.25.
Offer tags chosen by `exact`: {'AMBITIOUS': 54, 'SAFE': 42}. Deals accepted under `exact`: {'P0-W0-D0': 4, 'P0-W0-D1': 4, 'P0-W0-D2': 4, 'P0-W1-D0': 6, 'P0-W1-D1': 4, 'P0-W2-D0': 4, 'P1-W0-D1': 2, 'P1-W0-D2': 8, 'P1-W2-D0': 8, 'P1-W2-D1': 2, 'P2-W0-D2': 4, 'P2-W1-D1': 4, 'P2-W1-D2': 6, 'P2-W2-D0': 4, 'P2-W2-D1': 4, 'P2-W2-D2': 4}.

## 3. 6 rounds, easy (the document's horizon)

Source: `results/binned_b_scripted_r6_easy_20261026/analysis.json` (`analyze_condition_b.py`, B = 5000, seed 0); 1656 episodes.

| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret decisions | Σ EIG regret (bits) | questions | offers | C_H (bits) | final H (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | 72 | 0.7821 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.2075 | 0.00 | 2.67 | 5.377 | 0.333 |
| `greedy_eig` | 72 | 0.7821 | 1.0000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 0.0000 | 6.00 | 1.00 | 4.837 | 0.000 |
| `myopic` | 72 | 0.7756 | 0.9702 | 1.000 | 0.0064 | 0.0064 | 0.867 | 0.5409 | 0.00 | 2.50 | 5.711 | 0.667 |
| `random` | 1440 | 0.6767 | 0.7396 | 1.000 | 0.1053 | 0.1018 | 0.826 | 0.5228 | 1.51 | 1.31 | 5.230 | 1.125 |

Paired, cell-clustered bootstrap, arm − `exact` (for the deterministic arms the interval expresses variation across the type space, not sampling noise):

| arm − exact | Δ utility | Δ gain | Δ agreement | Δ Σ decision regret | Δ Σ EIG regret (bits) | Δ questions |
|---|---|---|---|---|---|---|
| `greedy_eig` | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.2075 [-0.2285, -0.1865] | +6.0000 [+6.0000, +6.0000] |
| `myopic` | -0.0064 [-0.0101, -0.0032] | -0.0298 [-0.0471, -0.0149] | +0.0000 [+0.0000, +0.0000] | +0.0064 [+0.0045, +0.0085] | +0.3333 [+0.2361, +0.4444] | +0.0000 [+0.0000, +0.0000] |
| `random` | -0.1053 [-0.1186, -0.0920] | -0.2604 [-0.2809, -0.2401] | +0.0000 [+0.0000, +0.0000] | +0.1018 [+0.0956, +0.1080] | +0.3153 [+0.2912, +0.3392] | +1.5104 [+1.4104, +1.6181] |

By role (`exact` arm): buyer U = 0.7821, asks 0.00; seller U = 0.7821, asks 0.00.
Offer tags chosen by `exact`: {'AMBITIOUS': 180, 'SAFE': 12}. Deals accepted under `exact`: {'P0-W0-D0': 4, 'P0-W0-D1': 4, 'P0-W0-D2': 4, 'P0-W1-D0': 4, 'P0-W1-D1': 4, 'P0-W2-D0': 4, 'P1-W0-D1': 4, 'P1-W0-D2': 4, 'P1-W1-D0': 4, 'P1-W1-D2': 4, 'P1-W2-D0': 4, 'P1-W2-D1': 4, 'P2-W0-D2': 4, 'P2-W1-D1': 4, 'P2-W1-D2': 4, 'P2-W2-D0': 4, 'P2-W2-D1': 4, 'P2-W2-D2': 4}.

## 4. 2 rounds, hard

Source: `results/binned_b_scripted_r2_hard_20261026/analysis.json` (`analyze_condition_b.py`, B = 5000, seed 0); 1656 episodes.

| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret decisions | Σ EIG regret (bits) | questions | offers | C_H (bits) | final H (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | 72 | 0.7583 | 1.0000 | 0.833 | 0.0000 | 0.0000 | 1.000 | 0.0000 | 1.00 | 1.50 | 4.837 | 0.333 |
| `greedy_eig` | 72 | 0.7340 | 0.8667 | 0.722 | 0.0244 | 0.0244 | 0.759 | 0.0000 | 2.00 | 1.00 | 5.059 | 0.222 |
| `myopic` | 72 | 0.7519 | 0.9701 | 0.833 | 0.0064 | 0.0064 | 0.167 | 0.0817 | 0.00 | 2.00 | 4.918 | 0.667 |
| `random` | 1440 | 0.7129 | 0.8074 | 0.792 | 0.0454 | 0.0450 | 0.568 | 0.1597 | 0.92 | 1.40 | 4.920 | 0.724 |

Paired, cell-clustered bootstrap, arm − `exact` (for the deterministic arms the interval expresses variation across the type space, not sampling noise):

| arm − exact | Δ utility | Δ gain | Δ agreement | Δ Σ decision regret | Δ Σ EIG regret (bits) | Δ questions |
|---|---|---|---|---|---|---|
| `greedy_eig` | -0.0244 [-0.0426, -0.0091] | -0.1333 [-0.2167, -0.0500] | -0.1111 [-0.1944, -0.0417] | +0.0244 [+0.0153, +0.0339] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+1.0000, +1.0000] |
| `myopic` | -0.0064 [-0.0101, -0.0032] | -0.0299 [-0.0448, -0.0149] | +0.0000 [+0.0000, +0.0000] | +0.0064 [+0.0061, +0.0067] | +0.0817 [+0.0817, +0.0817] | -1.0000 [-1.0000, -1.0000] |
| `random` | -0.0454 [-0.0547, -0.0367] | -0.1926 [-0.2186, -0.1669] | -0.0417 [-0.0618, -0.0250] | +0.0450 [+0.0418, +0.0483] | +0.1597 [+0.1356, +0.1869] | -0.0840 [-0.1264, -0.0403] |

By role (`exact` arm): buyer U = 0.7583, asks 1.00; seller U = 0.7583, asks 1.00.
Offer tags chosen by `exact`: {'AMBITIOUS': 54, 'SAFE': 54}. Deals accepted under `exact`: {'P0-W0-D0': 6, 'P0-W0-D1': 4, 'P0-W0-D2': 6, 'P0-W1-D0': 4, 'P0-W2-D0': 6, 'P1-W0-D2': 4, 'P1-W2-D0': 4, 'P2-W0-D2': 6, 'P2-W1-D2': 4, 'P2-W2-D0': 6, 'P2-W2-D1': 4, 'P2-W2-D2': 6}.

## Claims tagged

- exact-on-logs: every row of the `exact`, `greedy_eig`, `myopic` arms and every V; the sweep's cell counts.
- stochastic: the `random` arm (20 seeded plays per cell) and its intervals.

## Regenerate

```bash
sh results/binned_b_scripted_r2_easy_20261026/REGENERATE.sh
sh results/binned_b_scripted_r6_easy_20261026/REGENERATE.sh
sh results/binned_b_scripted_r2_hard_20261026/REGENERATE.sh
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.sweep_horizon --out results/binned_b_sweep_20261026/sweep.json --random-repeats 20 --seed 0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.render_scripted_note --out llm_rl_scripts/car_dealer_binned/SCRIPTED_RESULTS.md
```
