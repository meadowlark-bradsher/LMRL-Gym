# Live Car-Dealer, Phase 1 — scripted arms against the T1 kernel

_Rendered by `render_phase1_note.py` from `analysis.json` files written by `analyze_phase1.py`; every number is formatted from JSON, none is typed._

This is a scripted-arm note: it contains no model result and no registration has been made. Arms: exact_ask (C_plan, the menu-restricted exact oracle with ASK), exact_noask (B_plan, the same oracle with ASK forbidden), greedy_eig, random; any jev_* arm present ran against the offline mock backend unless the section's config says otherwise.

Result dirs:

- `results/live_phase1_scripted_20261029/h4_cq0` — env `live-car-dealer-v0.1`, H = 4, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.
- `results/live_phase1_scripted_20261029/h2_cq0` — env `live-car-dealer-v0.1`, H = 2, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.
- `results/live_phase1_scripted_20261029/h3_cq0` — env `live-car-dealer-v0.1`, H = 3, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.
- `results/live_phase1_scripted_20261029/h4_cq3` — env `live-car-dealer-v0.1`, H = 4, c_q = 3/100, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.

Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r), with U_max = 1 for every type; no deal gives Y = 0. Contrasts are paired over clusters (focal, opener, focal type, opponent type) with a percentile cluster bootstrap.

## H = 4, c_q = 0 — `results/live_phase1_scripted_20261029/h4_cq0`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 142.16 s. Bootstrap B = 2000, clusters = (focal, opener, focal_ranking, opp_ranking).

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.6322 | 0.3793 | 0.3793 | 1.000 | 1.517 | 1.000 | 0.0000 | 1.000 | 0.880 | 0.269 |
| exact_noask | 720 | 0.5519 | 0.3312 | 0.3312 | 0.912 | 0.000 | 1.478 | 0.0000 | 1.000 | 0.463 | 0.402 |
| greedy_eig | 720 | 0.3767 | 0.2260 | 0.2260 | 0.981 | 0.044 | 1.564 | 0.1566 | 0.236 | 0.000 | 0.393 |
| random | 720 | -0.0391 | -0.0234 | -0.0234 | 0.982 | 0.785 | 0.713 | 0.4071 | 0.326 | 0.647 | 0.706 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.6327 | 360 | 0.6318 |
| exact_noask | 360 | 0.5031 | 360 | 0.6008 |
| greedy_eig | 360 | 0.2361 | 360 | 0.5172 |
| random | 360 | 0.0098 | 360 | -0.0879 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0803 [+0.0660, +0.0961] | 144 | +0.1296 [+0.1079, +0.1493] | 72 | +0.0310 [+0.0205, +0.0440] | 72 |
| exact_noask-exact_ask | -0.0803 [-0.0960, -0.0660] | 144 | -0.1296 [-0.1493, -0.1078] | 72 | -0.0310 [-0.0440, -0.0204] | 72 |
| greedy_eig-exact_ask | -0.2556 [-0.3052, -0.2076] | 144 | -0.3965 [-0.4683, -0.3266] | 72 | -0.1146 [-0.1605, -0.0739] | 72 |
| random-exact_ask | -0.6713 [-0.7113, -0.6349] | 144 | -0.6229 [-0.6767, -0.5695] | 72 | -0.7197 [-0.7700, -0.6685] | 72 |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0383 | 0.0457 | 0.0000 | 0.0074 | 0.0000 | 0.0000 | 12 | 24 |
| focal_opens | 12 | 0.0612 | 0.0760 | 0.0000 | 0.0148 | 0.0000 | 0.0000 | 12 | 12 |
| opponent_opens | 12 | 0.0154 | 0.0154 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0803 [+0.0660, +0.0961] | 0.0762 | +1.0542 [+0.8669, +1.2618] | 0.0638 | +1.2582 [+1.0347, +1.5060] |
| exact_ask-exact_noask | focal_opens | +0.1296 [+0.1079, +0.1493] | 0.1266 | +1.0233 [+0.8523, +1.1791] | 0.1019 | +1.2712 [+1.0588, +1.4648] |
| exact_ask-exact_noask | opponent_opens | +0.0310 [+0.0205, +0.0440] | 0.0257 | +1.2065 [+0.8002, +1.7127] | 0.0257 | +1.2065 [+0.8002, +1.7127] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 1092/1092 | 1.000 | 0/1068 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/677 | 0.000 | 32/481 | 0.067 | -0.067 [-0.093, -0.041] |
| random | 307/797 | 0.385 | 258/692 | 0.373 | +0.012 [-0.043, +0.067] |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0803 [+0.0660, +0.0961] vs exact Δ\*_menu 0.0762 in Y units (0.0457 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.1296 [+0.1079, +0.1493] vs exact Δ\*_menu 0.1266 in Y units (0.0760 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0310 [+0.0205, +0.0440] vs exact Δ\*_menu 0.0257 in Y units (0.0154 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is NOT costless: 12/24 cells lose value at the root; mean loss with ASK allowed 0.0000, with ASK forbidden 0.0074. Attribution: the offer restriction with ASK forbidden (mean loss 0.0074). Δ\*_menu (0.0457 utility = 0.0762 in Y units) is the ceiling used for the captured fraction; Δ\*_full is 0.0383 utility. Losses in this bullet and the ceilings table are in utility units.
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.012); not more often: greedy_eig (-0.067).

## H = 2, c_q = 0 — `results/live_phase1_scripted_20261029/h2_cq0`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 4.18 s. Bootstrap B = 2000, clusters = (focal, opener, focal_ranking, opp_ranking).

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.3795 | 0.2277 | 0.2277 | 0.610 | 0.000 | 1.000 | 0.0000 | 1.000 | 0.302 | 1.160 |
| exact_noask | 720 | 0.3795 | 0.2277 | 0.2277 | 0.610 | 0.000 | 1.000 | 0.0000 | 1.000 | 0.292 | 1.160 |
| greedy_eig | 720 | -0.0296 | -0.0178 | -0.0178 | 0.856 | 0.199 | 0.801 | 0.2417 | 0.141 | 0.000 | 0.868 |
| random | 720 | -0.1804 | -0.1082 | -0.1082 | 0.932 | 0.393 | 0.461 | 0.3184 | 0.162 | 0.371 | 1.117 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.3248 | 360 | 0.4342 |
| exact_noask | 360 | 0.3248 | 360 | 0.4342 |
| greedy_eig | 360 | -0.0456 | 360 | -0.0136 |
| random | 360 | -0.2441 | 360 | -0.1167 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0000 [+0.0000, +0.0000] | 144 | +0.0000 [+0.0000, +0.0000] | 72 | +0.0000 [+0.0000, +0.0000] | 72 |
| exact_noask-exact_ask | +0.0000 [+0.0000, +0.0000] | 144 | +0.0000 [+0.0000, +0.0000] | 72 | +0.0000 [+0.0000, +0.0000] | 72 |
| greedy_eig-exact_ask | -0.4091 [-0.4821, -0.3335] | 144 | -0.3704 [-0.5025, -0.2409] | 72 | -0.4478 [-0.5105, -0.3829] | 72 |
| random-exact_ask | -0.5599 [-0.6118, -0.5100] | 144 | -0.5689 [-0.6296, -0.5023] | 72 | -0.5509 [-0.6230, -0.4718] | 72 |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0000 | 0.0000 | 0.0214 | 0.0214 | 0.0214 | 0.0000 | 12 | 0 |
| focal_opens | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| opponent_opens | 12 | 0.0000 | 0.0000 | 0.0428 | 0.0428 | 0.0428 | 0.0000 | 12 | 0 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |
| exact_ask-exact_noask | focal_opens | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |
| exact_ask-exact_noask | opponent_opens | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 0/0 | — | 0/720 | 0.000 | — |
| greedy_eig | 0/0 | — | 143/720 | 0.199 | — |
| random | 0/0 | — | 283/720 | 0.393 | — |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is NOT costless: 12/24 cells lose value at the root; mean loss with ASK allowed 0.0214, with ASK forbidden 0.0214. Attribution: the offer restriction (mean loss 0.0214 with ASK allowed); the offer restriction with ASK forbidden (mean loss 0.0214). Δ\*_menu (0.0000 utility = 0.0000 in Y units) is the ceiling used for the captured fraction; Δ\*_full is 0.0000 utility. Losses in this bullet and the ceilings table are in utility units.

## H = 3, c_q = 0 — `results/live_phase1_scripted_20261029/h3_cq0`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 21.21 s. Bootstrap B = 2000, clusters = (focal, opener, focal_ranking, opp_ranking).

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.5462 | 0.3277 | 0.3277 | 0.901 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.743 | 0.684 |
| exact_noask | 720 | 0.5091 | 0.3055 | 0.3055 | 0.728 | 0.000 | 1.333 | 0.0000 | 1.000 | 0.555 | 0.634 |
| greedy_eig | 720 | 0.1940 | 0.1164 | 0.1164 | 0.947 | 0.144 | 1.189 | 0.2135 | 0.120 | 0.000 | 0.503 |
| random | 720 | -0.0730 | -0.0438 | -0.0438 | 0.960 | 0.619 | 0.625 | 0.3697 | 0.179 | 0.604 | 0.851 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.5237 | 360 | 0.5688 |
| exact_noask | 360 | 0.5098 | 360 | 0.5084 |
| greedy_eig | 360 | 0.2573 | 360 | 0.1307 |
| random | 360 | -0.0480 | 360 | -0.0981 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0371 [+0.0070, +0.0654] | 144 | +0.0139 [-0.0323, +0.0634] | 72 | +0.0604 [+0.0273, +0.0950] | 72 |
| exact_noask-exact_ask | -0.0371 [-0.0653, -0.0068] | 144 | -0.0139 [-0.0631, +0.0325] | 72 | -0.0604 [-0.0950, -0.0272] | 72 |
| greedy_eig-exact_ask | -0.3522 [-0.4184, -0.2801] | 144 | -0.2664 [-0.3623, -0.1735] | 72 | -0.4381 [-0.5432, -0.3364] | 72 |
| random-exact_ask | -0.6193 [-0.6658, -0.5741] | 144 | -0.5717 [-0.6408, -0.4998] | 72 | -0.6669 [-0.7197, -0.6127] | 72 |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0346 | 0.0239 | 0.0294 | 0.0187 | 0.0000 | 0.0148 | 24 | 24 |
| focal_opens | 12 | 0.0184 | 0.0066 | 0.0388 | 0.0270 | 0.0000 | 0.0184 | 12 | 12 |
| opponent_opens | 12 | 0.0508 | 0.0412 | 0.0199 | 0.0103 | 0.0000 | 0.0112 | 12 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0371 [+0.0070, +0.0654] | 0.0398 | +0.9320 [+0.1766, +1.6418] | 0.0577 | +0.6438 [+0.1220, +1.1342] |
| exact_ask-exact_noask | focal_opens | +0.0139 [-0.0323, +0.0634] | 0.0109 | +1.2687 [-2.9504, +5.7903] | 0.0306 | +0.4540 [-1.0558, +2.0719] |
| exact_ask-exact_noask | opponent_opens | +0.0604 [+0.0273, +0.0950] | 0.0687 | +0.8784 [+0.3969, +1.3831] | 0.0847 | +0.7124 [+0.3219, +1.1217] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 720/720 | 1.000 | 0/720 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/720 | 0.000 | 104/240 | 0.433 | -0.433 [-0.487, -0.379] |
| random | 283/720 | 0.393 | 163/441 | 0.370 | +0.023 [-0.044, +0.086] |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0371 [+0.0070, +0.0654] vs exact Δ\*_menu 0.0398 in Y units (0.0239 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0139 [-0.0323, +0.0634] vs exact Δ\*_menu 0.0109 in Y units (0.0066 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0604 [+0.0273, +0.0950] vs exact Δ\*_menu 0.0687 in Y units (0.0412 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is NOT costless: 24/24 cells lose value at the root; mean loss with ASK allowed 0.0294, with ASK forbidden 0.0187. Attribution: the question restriction (mean loss 0.0148 with ASK allowed); the offer restriction with ASK forbidden (mean loss 0.0187). Δ\*_menu (0.0239 utility = 0.0398 in Y units) is the ceiling used for the captured fraction; Δ\*_full is 0.0346 utility. Losses in this bullet and the ceilings table are in utility units.
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.023); not more often: greedy_eig (-0.433).

## H = 4, c_q = 3/100 — `results/live_phase1_scripted_20261029/h4_cq3`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 138.42 s. Bootstrap B = 2000, clusters = (focal, opener, focal_ranking, opp_ranking).

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.5761 | 0.3457 | 0.3743 | 0.986 | 0.953 | 1.144 | 0.0000 | 1.000 | 0.619 | 0.236 |
| exact_noask | 720 | 0.5519 | 0.3312 | 0.3312 | 0.912 | 0.000 | 1.478 | 0.0000 | 1.000 | 0.463 | 0.402 |
| greedy_eig | 720 | 0.3744 | 0.2247 | 0.2260 | 0.981 | 0.044 | 1.564 | 0.1242 | 0.294 | 0.000 | 0.393 |
| random | 720 | -0.0783 | -0.0470 | -0.0234 | 0.982 | 0.785 | 0.713 | 0.3968 | 0.224 | 0.647 | 0.706 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.5425 | 360 | 0.6097 |
| exact_noask | 360 | 0.5031 | 360 | 0.6008 |
| greedy_eig | 360 | 0.2361 | 360 | 0.5128 |
| random | 360 | -0.0340 | 360 | -0.1226 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0242 [+0.0127, +0.0357] | 144 | +0.0395 [+0.0178, +0.0593] | 72 | +0.0089 [+0.0029, +0.0160] | 72 |
| exact_noask-exact_ask | -0.0242 [-0.0356, -0.0126] | 144 | -0.0395 [-0.0593, -0.0177] | 72 | -0.0089 [-0.0159, -0.0029] | 72 |
| greedy_eig-exact_ask | -0.2017 [-0.2488, -0.1564] | 144 | -0.3064 [-0.3759, -0.2391] | 72 | -0.0970 [-0.1407, -0.0569] | 72 |
| random-exact_ask | -0.6544 [-0.6960, -0.6151] | 144 | -0.5765 [-0.6314, -0.5220] | 72 | -0.7324 [-0.7821, -0.6803] | 72 |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0095 | 0.0118 | 0.0051 | 0.0074 | 0.0000 | 0.0012 | 12 | 24 |
| focal_opens | 12 | 0.0170 | 0.0216 | 0.0103 | 0.0148 | 0.0000 | 0.0023 | 12 | 12 |
| opponent_opens | 12 | 0.0020 | 0.0020 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0242 [+0.0127, +0.0357] | 0.0196 | +1.2331 [+0.6469, +1.8198] | 0.0158 | +1.5300 [+0.8026, +2.2580] |
| exact_ask-exact_noask | focal_opens | +0.0395 [+0.0178, +0.0593] | 0.0360 | +1.0964 [+0.4936, +1.6481] | 0.0284 | +1.3907 [+0.6261, +2.0904] |
| exact_ask-exact_noask | opponent_opens | +0.0089 [+0.0029, +0.0160] | 0.0033 | +2.7456 [+0.8794, +4.9097] | 0.0033 | +2.7456 [+0.8794, +4.9097] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 686/686 | 1.000 | 0/1144 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/534 | 0.000 | 32/624 | 0.051 | -0.051 [-0.070, -0.032] |
| random | 250/590 | 0.424 | 315/899 | 0.350 | +0.073 [+0.023, +0.126] |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0242 [+0.0127, +0.0357] vs exact Δ\*_menu 0.0196 in Y units (0.0118 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0395 [+0.0178, +0.0593] vs exact Δ\*_menu 0.0360 in Y units (0.0216 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0089 [+0.0029, +0.0160] vs exact Δ\*_menu 0.0033 in Y units (0.0020 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is NOT costless: 12/24 cells lose value at the root; mean loss with ASK allowed 0.0051, with ASK forbidden 0.0074. Attribution: the question restriction (mean loss 0.0012 with ASK allowed); the offer restriction with ASK forbidden (mean loss 0.0074). Δ\*_menu (0.0118 utility = 0.0196 in Y units) is the ceiling used for the captured fraction; Δ\*_full is 0.0095 utility. Losses in this bullet and the ceilings table are in utility units.
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.073); not more often: greedy_eig (-0.051).

## Menu size and costlessness — `results/live_phase1_scripted_20261029/costlessness_h4_cq0.json`

`costlessness_check.py`, H = 4, c_q = 0, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, 127.7 s. Root values of the exact oracle under the full action set and under the shortlist menu with n_offers × n_questions candidates (the offer rule yields at most three distinct candidates: SAFE, AMBITIOUS, BALANCED). Utility units.

| menu (offers × questions) | cells lossless | mean loss, ASK allowed | max loss, ASK allowed | mean loss, ASK forbidden | max loss, ASK forbidden | mean Δ\*_menu | mean Δ\*_full |
|---|---|---|---|---|---|---|---|
| 2x2 | 12/24 | 0.0000 | 0.0000 | 0.0074 | 0.0148 | 0.0457 | 0.0383 |
| 3x2 | 12/24 | 0.0000 | 0.0000 | 0.0023 | 0.0045 | 0.0405 | 0.0383 |
| 2x3 | 12/24 | 0.0000 | 0.0000 | 0.0074 | 0.0148 | 0.0457 | 0.0383 |
| 3x3 | 12/24 | 0.0000 | 0.0000 | 0.0023 | 0.0045 | 0.0405 | 0.0383 |
| 4x9 | 12/24 | 0.0000 | 0.0000 | 0.0023 | 0.0045 | 0.0405 | 0.0383 |

- Cells with any loss under any menu size: focal opens.

## Regenerate

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted_20261029/h4_cq0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted_20261029/h2_cq0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted_20261029/h3_cq0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted_20261029/h4_cq3
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note --dirs results/live_phase1_scripted_20261029/h4_cq0,results/live_phase1_scripted_20261029/h2_cq0,results/live_phase1_scripted_20261029/h3_cq0,results/live_phase1_scripted_20261029/h4_cq3 --out llm_rl_scripts/car_dealer_live/PHASE1_SCRIPTED_RESULTS.md --costlessness results/live_phase1_scripted_20261029/costlessness_h4_cq0.json
```
