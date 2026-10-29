# Live Car-Dealer, Phase 1 — scripted arms against the T1 kernel

_Rendered by `render_phase1_note.py` from `analysis.json` files written by `analyze_phase1.py`; every number is formatted from JSON, none is typed._

This is a scripted-arm note: it contains no model result and no registration has been made. Arms: exact_ask (C_plan, the menu-restricted exact oracle with ASK), exact_noask (B_plan, the same oracle with ASK forbidden), greedy_eig, random; any jev_* arm present ran against the offline mock backend unless the section's config says otherwise.

Result dirs:

- `results/live_phase1_jev_h4_cq0_20261029` — env `live-car-dealer-v0.1`, H = 4, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 5760 episodes.
- `results/live_phase1_jev_h2_cq0_20261029` — env `live-car-dealer-v0.1`, H = 2, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 4320 episodes.
- `results/live_phase1_jev_h3_cq0_20261029` — env `live-car-dealer-v0.1`, H = 3, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 4320 episodes.
- `results/live_phase1_jev_h4_cq3_20261029` — env `live-car-dealer-v0.1`, H = 4, c_q = 3/100, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 4320 episodes.
- `results/live_phase1_jev_h4_cq0_full_20261029` — env `live-car-dealer-v0.1`, H = 4, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `full`, 4320 episodes.
- `results/live_phase1_jev_h4_cq0_replicate_20261029` — env `live-car-dealer-v0.1`, H = 4, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 288 episodes.

Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r), with U_max = 1 for every type; no deal gives Y = 0. Contrasts are paired over clusters (focal, opener, focal type, opponent type) with a percentile cluster bootstrap.

## H = 4, c_q = 0 — `results/live_phase1_jev_h4_cq0_20261029`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random', 'jev_ask_plain', 'jev_ask_eig', 'jev_noask_plain', 'jev_noask_eig'], 5760 episodes (5 repeats per cell, seed 21), 0 error stubs, 912.85 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.6314 | 0.3788 | 0.3788 | 1.000 | 1.561 | 1.000 | 0.0000 | 1.000 | 0.900 | 0.281 |
| exact_noask | 720 | 0.5717 | 0.3430 | 0.3430 | 0.911 | 0.000 | 1.472 | 0.0000 | 1.000 | 0.416 | 0.359 |
| greedy_eig | 720 | 0.4647 | 0.2788 | 0.2788 | 0.951 | 0.000 | 1.615 | 0.1003 | 0.282 | 0.000 | 0.374 |
| random | 720 | -0.0207 | -0.0124 | -0.0124 | 0.990 | 0.861 | 0.728 | 0.3902 | 0.353 | 0.717 | 0.707 |
| jev_ask_plain | 720 | 0.4653 | 0.2792 | 0.2792 | 0.672 | 0.000 | 1.769 | 0.0979 | 0.335 | 0.327 | 0.404 |
| jev_ask_eig | 720 | 0.4682 | 0.2809 | 0.2809 | 0.686 | 0.000 | 1.751 | 0.1009 | 0.322 | 0.353 | 0.426 |
| jev_noask_plain | 720 | 0.4706 | 0.2824 | 0.2824 | 0.669 | 0.000 | 1.772 | 0.0589 | 0.389 | 0.315 | 0.391 |
| jev_noask_eig | 720 | 0.4708 | 0.2825 | 0.2825 | 0.674 | 0.000 | 1.772 | 0.0597 | 0.388 | 0.320 | 0.397 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.6305 | 360 | 0.6323 |
| exact_noask | 360 | 0.5329 | 360 | 0.6105 |
| greedy_eig | 360 | 0.3321 | 360 | 0.5972 |
| random | 360 | -0.0121 | 360 | -0.0292 |
| jev_ask_plain | 360 | 0.3540 | 360 | 0.5766 |
| jev_ask_eig | 360 | 0.3672 | 360 | 0.5691 |
| jev_noask_plain | 360 | 0.3644 | 360 | 0.5769 |
| jev_noask_eig | 360 | 0.3637 | 360 | 0.5778 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0597 [+0.0436, +0.0774] | 36 | +0.0976 [+0.0696, +0.1287] | 36 | +0.0218 [+0.0137, +0.0303] | 36 |
| jev_ask_plain-jev_noask_plain | -0.0053 [-0.0120, +0.0007] | 36 | -0.0104 [-0.0207, -0.0019] | 36 | -0.0002 [-0.0057, +0.0050] | 36 |
| jev_ask_eig-jev_noask_eig | -0.0026 [-0.0088, +0.0034] | 36 | +0.0035 [-0.0056, +0.0127] | 36 | -0.0088 [-0.0162, -0.0017] | 36 |
| exact_noask-exact_ask | -0.0597 [-0.0774, -0.0436] | 36 | -0.0976 [-0.1286, -0.0696] | 36 | -0.0218 [-0.0303, -0.0136] | 36 |
| greedy_eig-exact_ask | -0.1668 [-0.2244, -0.1165] | 36 | -0.2984 [-0.4008, -0.2087] | 36 | -0.0351 [-0.0513, -0.0203] | 36 |
| random-exact_ask | -0.6521 [-0.6950, -0.6098] | 36 | -0.6427 [-0.7033, -0.5806] | 36 | -0.6615 [-0.7206, -0.6058] | 36 |
| jev_ask_plain-exact_ask | -0.1661 [-0.1930, -0.1394] | 36 | -0.2765 [-0.3155, -0.2343] | 36 | -0.0557 [-0.0828, -0.0332] | 36 |
| jev_ask_eig-exact_ask | -0.1632 [-0.1900, -0.1372] | 36 | -0.2633 [-0.3026, -0.2213] | 36 | -0.0632 [-0.0892, -0.0417] | 36 |
| jev_noask_plain-exact_ask | -0.1608 [-0.1892, -0.1332] | 36 | -0.2661 [-0.3057, -0.2233] | 36 | -0.0554 [-0.0849, -0.0314] | 36 |
| jev_noask_eig-exact_ask | -0.1606 [-0.1866, -0.1345] | 36 | -0.2668 [-0.3059, -0.2247] | 36 | -0.0545 [-0.0801, -0.0325] | 36 |
| jev_ask_eig-jev_ask_plain | +0.0028 [-0.0032, +0.0088] | 36 | +0.0132 [+0.0041, +0.0233] | 36 | -0.0075 [-0.0162, +0.0005] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +1.5611 [+1.5056, +1.6139] | +1.8417 [+1.7694, +1.9111] | +1.2806 [+1.2194, +1.3444] |
| jev_ask_plain-jev_noask_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_eig-jev_noask_eig | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| exact_noask-exact_ask | -1.5611 [-1.6139, -1.5056] | -1.8417 [-1.9111, -1.7667] | -1.2806 [-1.3444, -1.2194] |
| greedy_eig-exact_ask | -1.5611 [-1.6139, -1.5056] | -1.8417 [-1.9111, -1.7667] | -1.2806 [-1.3444, -1.2194] |
| random-exact_ask | -0.7000 [-0.7833, -0.6153] | -0.8861 [-1.0056, -0.7694] | -0.5139 [-0.6278, -0.3972] |
| jev_ask_plain-exact_ask | -1.5611 [-1.6139, -1.5056] | -1.8417 [-1.9111, -1.7667] | -1.2806 [-1.3444, -1.2194] |
| jev_ask_eig-exact_ask | -1.5611 [-1.6139, -1.5056] | -1.8417 [-1.9111, -1.7667] | -1.2806 [-1.3444, -1.2194] |
| jev_noask_plain-exact_ask | -1.5611 [-1.6139, -1.5056] | -1.8417 [-1.9111, -1.7667] | -1.2806 [-1.3444, -1.2194] |
| jev_noask_eig-exact_ask | -1.5611 [-1.6139, -1.5056] | -1.8417 [-1.9111, -1.7667] | -1.2806 [-1.3444, -1.2194] |
| jev_ask_eig-jev_ask_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0383 | 0.0383 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 24 |
| focal_opens | 12 | 0.0612 | 0.0612 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |
| opponent_opens | 12 | 0.0154 | 0.0154 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0597 [+0.0436, +0.0774] | 0.0638 | +0.9354 [+0.6839, +1.2125] | 0.0638 | +0.9354 [+0.6839, +1.2125] |
| exact_ask-exact_noask | focal_opens | +0.0976 [+0.0696, +0.1287] | 0.1019 | +0.9572 [+0.6823, +1.2622] | 0.1019 | +0.9572 [+0.6823, +1.2622] |
| exact_ask-exact_noask | opponent_opens | +0.0218 [+0.0137, +0.0303] | 0.0257 | +0.8487 [+0.5325, +1.1788] | 0.0257 | +0.8487 [+0.5325, +1.1788] |
| jev_ask_plain-jev_noask_plain | all | -0.0053 [-0.0120, +0.0007] | 0.0638 | -0.0834 [-0.1884, +0.0103] | 0.0638 | -0.0834 [-0.1884, +0.0103] |
| jev_ask_plain-jev_noask_plain | focal_opens | -0.0104 [-0.0207, -0.0019] | 0.1019 | -0.1024 [-0.2030, -0.0185] | 0.1019 | -0.1024 [-0.2030, -0.0185] |
| jev_ask_plain-jev_noask_plain | opponent_opens | -0.0002 [-0.0057, +0.0050] | 0.0257 | -0.0083 [-0.2205, +0.1928] | 0.0257 | -0.0083 [-0.2205, +0.1928] |
| jev_ask_eig-jev_noask_eig | all | -0.0026 [-0.0088, +0.0034] | 0.0638 | -0.0413 [-0.1378, +0.0536] | 0.0638 | -0.0413 [-0.1378, +0.0536] |
| jev_ask_eig-jev_noask_eig | focal_opens | +0.0035 [-0.0056, +0.0127] | 0.1019 | +0.0342 [-0.0552, +0.1244] | 0.1019 | +0.0342 [-0.0552, +0.1244] |
| jev_ask_eig-jev_noask_eig | opponent_opens | -0.0088 [-0.0162, -0.0017] | 0.0257 | -0.3412 [-0.6296, -0.0652] | 0.0257 | -0.3412 [-0.6296, -0.0652] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 1124/1124 | 1.000 | 0/1036 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/663 | 0.000 | 0/500 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 327/824 | 0.397 | 293/692 | 0.423 | -0.027 [-0.080, +0.026] |
| jev_ask_plain | 0/857 | 0.000 | 0/424 | 0.000 | +0.000 [+0.000, +0.000] |
| jev_ask_eig | 0/857 | 0.000 | 0/417 | 0.000 | +0.000 [+0.000, +0.000] |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| exact_ask | {'ASK': 1124, 'PROPOSE': 720, 'REJECT': 316} | {'SAFE': 720} | {'accept': 720} | 0.000 | — | — | — |
| exact_noask | {'PROPOSE': 1060, 'REJECT': 514} | {'PLAN': 360, 'AMBITIOUS': 278, 'SAFE': 422} | {'walk': 64, 'accept': 656} | 0.089 | — | — | — |
| greedy_eig | {'PROPOSE': 1163, 'ACCEPT': 83} | {'AMBITIOUS': 737, 'PLAN': 143, 'SAFE': 283} | {'accept': 685, 'walk': 35} | 0.049 | — | — | — |
| random | {'ASK': 620, 'PROPOSE': 524, 'REJECT': 186, 'ACCEPT': 407} | {'SAFE': 247, 'PLAN_ASK': 54, 'AMBITIOUS': 157, 'PLAN': 66} | {'accept': 713, 'walk': 7} | 0.010 | — | — | — |
| jev_ask_plain | {'PROPOSE': 1274, 'WALK': 195, 'ACCEPT': 16} | {'PLAN_ASK': 360, 'AMBITIOUS': 698, 'SAFE': 213, 'PLAN': 3} | {'walk': 236, 'accept': 484} | 0.328 | 0.180 | 0.595 | 1.572 |
| jev_ask_eig | {'PROPOSE': 1261, 'WALK': 186, 'ACCEPT': 23} | {'PLAN_ASK': 360, 'AMBITIOUS': 678, 'SAFE': 220, 'PLAN': 3} | {'walk': 226, 'accept': 494} | 0.314 | 0.138 | 0.628 | 1.446 |
| jev_noask_plain | {'PROPOSE': 1276, 'WALK': 194, 'ACCEPT': 14} | {'PLAN_ASK': 360, 'AMBITIOUS': 708, 'SAFE': 208} | {'walk': 238, 'accept': 482} | 0.331 | — | 0.713 | 0.990 |
| jev_noask_eig | {'PROPOSE': 1276, 'WALK': 192, 'ACCEPT': 14} | {'PLAN_ASK': 360, 'AMBITIOUS': 704, 'SAFE': 212} | {'walk': 235, 'accept': 485} | 0.326 | — | 0.727 | 0.947 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 249, '1': 272, '2': 193, '3': 152, '4': 182, '5': 182, '6': 149, '7': 86, '8': 20} over 1485 decisions; well-formed share 1.000.
- jev_ask_eig: {'0': 281, '1': 287, '2': 181, '3': 159, '4': 161, '5': 161, '6': 143, '7': 81, '8': 16} over 1470 decisions; well-formed share 1.000.
- jev_noask_plain: {'0': 376, '1': 345, '2': 259, '3': 274, '4': 197, '5': 33} over 1484 decisions; well-formed share 1.000.
- jev_noask_eig: {'0': 378, '1': 333, '2': 272, '3': 280, '4': 186, '5': 33} over 1482 decisions; well-formed share 1.000.

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0597 [+0.0436, +0.0774] vs exact Δ\*_menu 0.0638 in Y units (0.0383 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0976 [+0.0696, +0.1287] vs exact Δ\*_menu 0.1019 in Y units (0.0612 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0218 [+0.0137, +0.0303] vs exact Δ\*_menu 0.0257 in Y units (0.0154 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000); not more often: greedy_eig (+0.000), random (-0.027), jev_ask_plain (+0.000), jev_ask_eig (+0.000).

## H = 2, c_q = 0 — `results/live_phase1_jev_h2_cq0_20261029`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random', 'jev_ask_plain', 'jev_noask_plain'], 4320 episodes (5 repeats per cell, seed 22), 0 error stubs, 198.5 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.4023 | 0.2414 | 0.2414 | 0.667 | 0.000 | 1.000 | 0.0000 | 1.000 | 0.286 | 1.206 |
| exact_noask | 720 | 0.4023 | 0.2414 | 0.2414 | 0.667 | 0.000 | 1.000 | 0.0000 | 1.000 | 0.275 | 1.206 |
| greedy_eig | 720 | -0.0573 | -0.0344 | -0.0344 | 0.872 | 0.229 | 0.771 | 0.2779 | 0.134 | 0.000 | 0.916 |
| random | 720 | -0.2025 | -0.1215 | -0.1215 | 0.958 | 0.460 | 0.379 | 0.3604 | 0.132 | 0.395 | 1.121 |
| jev_ask_plain | 720 | 0.3321 | 0.1993 | 0.1993 | 0.412 | 0.001 | 0.994 | 0.0420 | 0.385 | 0.042 | 0.961 |
| jev_noask_plain | 720 | 0.3331 | 0.1999 | 0.1999 | 0.404 | 0.000 | 1.000 | 0.0419 | 0.390 | 0.012 | 0.942 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.3248 | 360 | 0.4799 |
| exact_noask | 360 | 0.3248 | 360 | 0.4799 |
| greedy_eig | 360 | -0.0392 | 360 | -0.0754 |
| random | 360 | -0.2621 | 360 | -0.1428 |
| jev_ask_plain | 360 | 0.3095 | 360 | 0.3547 |
| jev_noask_plain | 360 | 0.3095 | 360 | 0.3567 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 |
| jev_ask_plain-jev_noask_plain | -0.0010 [-0.0063, +0.0045] | 36 | +0.0000 [+0.0000, +0.0000] | 36 | -0.0020 [-0.0126, +0.0089] | 36 |
| exact_noask-exact_ask | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 |
| greedy_eig-exact_ask | -0.4596 [-0.5407, -0.3805] | 36 | -0.3640 [-0.5196, -0.2165] | 36 | -0.5552 [-0.6541, -0.4591] | 36 |
| random-exact_ask | -0.6048 [-0.6685, -0.5450] | 36 | -0.5869 [-0.6592, -0.5178] | 36 | -0.6227 [-0.7085, -0.5377] | 36 |
| jev_ask_plain-exact_ask | -0.0702 [-0.1370, -0.0076] | 36 | -0.0152 [-0.1029, +0.0629] | 36 | -0.1252 [-0.1883, -0.0642] | 36 |
| jev_noask_plain-exact_ask | -0.0692 [-0.1353, -0.0069] | 36 | -0.0152 [-0.1029, +0.0629] | 36 | -0.1232 [-0.1847, -0.0629] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_plain-jev_noask_plain | +0.0014 [+0.0000, +0.0042] | +0.0000 [+0.0000, +0.0000] | +0.0028 [+0.0000, +0.0083] |
| exact_noask-exact_ask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| greedy_eig-exact_ask | +0.2292 [+0.2000, +0.2583] | +0.0000 [+0.0000, +0.0000] | +0.4583 [+0.4000, +0.5167] |
| random-exact_ask | +0.4597 [+0.4264, +0.4944] | +0.5944 [+0.5444, +0.6417] | +0.3250 [+0.2833, +0.3694] |
| jev_ask_plain-exact_ask | +0.0014 [+0.0000, +0.0042] | +0.0000 [+0.0000, +0.0000] | +0.0028 [+0.0000, +0.0083] |
| jev_noask_plain-exact_ask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| focal_opens | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| opponent_opens | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |
| exact_ask-exact_noask | focal_opens | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |
| exact_ask-exact_noask | opponent_opens | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |
| jev_ask_plain-jev_noask_plain | all | -0.0010 [-0.0063, +0.0045] | 0.0000 | — | 0.0000 | — |
| jev_ask_plain-jev_noask_plain | focal_opens | +0.0000 [+0.0000, +0.0000] | 0.0000 | — | 0.0000 | — |
| jev_ask_plain-jev_noask_plain | opponent_opens | -0.0020 [-0.0126, +0.0089] | 0.0000 | — | 0.0000 | — |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 0/0 | — | 0/720 | 0.000 | — |
| greedy_eig | 0/0 | — | 165/720 | 0.229 | — |
| random | 0/0 | — | 331/720 | 0.460 | — |
| jev_ask_plain | 0/0 | — | 1/720 | 0.001 | — |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| exact_ask | {'PROPOSE': 720, 'WALK': 120} | {'SAFE': 418, 'AMBITIOUS': 137, 'PLAN': 165} | {'walk': 240, 'accept': 480} | 0.333 | — | — | — |
| exact_noask | {'PROPOSE': 720, 'WALK': 120} | {'SAFE': 418, 'AMBITIOUS': 137, 'PLAN': 165} | {'walk': 240, 'accept': 480} | 0.333 | — | — | — |
| greedy_eig | {'PROPOSE': 555, 'ACCEPT': 405, 'ASK': 165} | {'AMBITIOUS': 555} | {'accept': 628, 'walk': 92} | 0.128 | — | — | — |
| random | {'ASK': 331, 'ACCEPT': 526, 'PROPOSE': 273, 'REJECT': 55} | {'SAFE': 126, 'AMBITIOUS': 125, 'PLAN': 22} | {'accept': 690, 'walk': 30} | 0.042 | — | — | — |
| jev_ask_plain | {'PROPOSE': 716, 'WALK': 227, 'ACCEPT': 17, 'ASK': 1} | {'AMBITIOUS': 702, 'SAFE': 14} | {'walk': 423, 'accept': 297} | 0.588 | 0.194 | 0.619 | 1.447 |
| jev_noask_plain | {'PROPOSE': 720, 'WALK': 226, 'ACCEPT': 14} | {'AMBITIOUS': 717, 'SAFE': 3} | {'walk': 429, 'accept': 291} | 0.596 | — | 0.739 | 0.863 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 234, '1': 221, '2': 105, '3': 110, '4': 109, '5': 100, '6': 40, '7': 23, '8': 19} over 961 decisions; well-formed share 1.000.
- jev_noask_plain: {'0': 345, '1': 266, '2': 194, '3': 67, '4': 67, '5': 21} over 960 decisions; well-formed share 1.000.

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).

## H = 3, c_q = 0 — `results/live_phase1_jev_h3_cq0_20261029`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random', 'jev_ask_plain', 'jev_noask_plain'], 4320 episodes (5 repeats per cell, seed 23), 0 error stubs, 244.18 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.5934 | 0.3560 | 0.3560 | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.942 | 0.824 |
| exact_noask | 720 | 0.5302 | 0.3181 | 0.3181 | 0.717 | 0.000 | 1.333 | 0.0000 | 1.000 | 0.511 | 0.621 |
| greedy_eig | 720 | 0.3028 | 0.1817 | 0.1817 | 0.864 | 0.000 | 1.333 | 0.1685 | 0.214 | 0.000 | 0.583 |
| random | 720 | -0.0718 | -0.0431 | -0.0431 | 0.960 | 0.696 | 0.569 | 0.4014 | 0.151 | 0.644 | 0.872 |
| jev_ask_plain | 720 | 0.4335 | 0.2601 | 0.2601 | 0.574 | 0.000 | 1.329 | 0.0947 | 0.272 | 0.052 | 0.635 |
| jev_noask_plain | 720 | 0.4348 | 0.2609 | 0.2609 | 0.565 | 0.000 | 1.332 | 0.0589 | 0.590 | 0.036 | 0.620 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.5890 | 360 | 0.5978 |
| exact_noask | 360 | 0.5447 | 360 | 0.5157 |
| greedy_eig | 360 | 0.5447 | 360 | 0.0609 |
| random | 360 | -0.0502 | 360 | -0.0933 |
| jev_ask_plain | 360 | 0.5063 | 360 | 0.3606 |
| jev_noask_plain | 360 | 0.5088 | 360 | 0.3609 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0632 [+0.0167, +0.1121] | 36 | +0.0443 [-0.0071, +0.0966] | 36 | +0.0820 [+0.0370, +0.1299] | 36 |
| jev_ask_plain-jev_noask_plain | -0.0014 [-0.0058, +0.0028] | 36 | -0.0025 [-0.0105, +0.0045] | 36 | -0.0002 [-0.0048, +0.0041] | 36 |
| exact_noask-exact_ask | -0.0632 [-0.1121, -0.0167] | 36 | -0.0443 [-0.0964, +0.0072] | 36 | -0.0820 [-0.1296, -0.0369] | 36 |
| greedy_eig-exact_ask | -0.2906 [-0.3713, -0.2096] | 36 | -0.0443 [-0.0964, +0.0072] | 36 | -0.5369 [-0.6503, -0.4211] | 36 |
| random-exact_ask | -0.6652 [-0.7170, -0.6161] | 36 | -0.6393 [-0.6974, -0.5797] | 36 | -0.6911 [-0.7509, -0.6353] | 36 |
| jev_ask_plain-exact_ask | -0.1599 [-0.2087, -0.1096] | 36 | -0.0827 [-0.1423, -0.0228] | 36 | -0.2371 [-0.2864, -0.1855] | 36 |
| jev_noask_plain-exact_ask | -0.1585 [-0.2068, -0.1080] | 36 | -0.0802 [-0.1391, -0.0212] | 36 | -0.2369 [-0.2858, -0.1852] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +1.0000 [+1.0000, +1.0000] | +1.0000 [+1.0000, +1.0000] | +1.0000 [+1.0000, +1.0000] |
| jev_ask_plain-jev_noask_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| exact_noask-exact_ask | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] |
| greedy_eig-exact_ask | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] |
| random-exact_ask | -0.3042 [-0.3569, -0.2528] | -0.1056 [-0.1889, -0.0250] | -0.5028 [-0.5694, -0.4361] |
| jev_ask_plain-exact_ask | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] |
| jev_noask_plain-exact_ask | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0346 | 0.0346 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 24 |
| focal_opens | 12 | 0.0184 | 0.0184 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |
| opponent_opens | 12 | 0.0508 | 0.0508 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0632 [+0.0167, +0.1121] | 0.0577 | +1.0952 [+0.2903, +1.9435] | 0.0577 | +1.0952 [+0.2903, +1.9435] |
| exact_ask-exact_noask | focal_opens | +0.0443 [-0.0071, +0.0966] | 0.0306 | +1.4480 [-0.2316, +3.1580] | 0.0306 | +1.4480 [-0.2316, +3.1580] |
| exact_ask-exact_noask | opponent_opens | +0.0820 [+0.0370, +0.1299] | 0.0847 | +0.9679 [+0.4362, +1.5327] | 0.0847 | +0.9679 [+0.4362, +1.5327] |
| jev_ask_plain-jev_noask_plain | all | -0.0014 [-0.0058, +0.0028] | 0.0577 | -0.0238 [-0.1007, +0.0479] | 0.0577 | -0.0238 [-0.1007, +0.0479] |
| jev_ask_plain-jev_noask_plain | focal_opens | -0.0025 [-0.0105, +0.0045] | 0.0306 | -0.0815 [-0.3445, +0.1467] | 0.0306 | -0.0815 [-0.3445, +0.1467] |
| jev_ask_plain-jev_noask_plain | opponent_opens | -0.0002 [-0.0048, +0.0041] | 0.0847 | -0.0029 [-0.0567, +0.0479] | 0.0847 | -0.0029 [-0.0567, +0.0479] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 720/720 | 1.000 | 0/720 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/720 | 0.000 | 0/240 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 317/720 | 0.440 | 184/437 | 0.421 | +0.019 [-0.038, +0.076] |
| jev_ask_plain | 0/720 | 0.000 | 0/240 | 0.000 | +0.000 [+0.000, +0.000] |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| exact_ask | {'ASK': 720, 'PROPOSE': 720} | {'SAFE': 720} | {'accept': 720} | 0.000 | — | — | — |
| exact_noask | {'PROPOSE': 960, 'REJECT': 360} | {'AMBITIOUS': 609, 'PLAN': 149, 'SAFE': 202} | {'walk': 204, 'accept': 516} | 0.283 | — | — | — |
| greedy_eig | {'PROPOSE': 960, 'ACCEPT': 206} | {'AMBITIOUS': 794, 'PLAN': 112, 'SAFE': 54} | {'walk': 98, 'accept': 622} | 0.136 | — | — | — |
| random | {'PROPOSE': 410, 'ASK': 501, 'ACCEPT': 419, 'REJECT': 110} | {'SAFE': 207, 'AMBITIOUS': 176, 'PLAN': 27} | {'accept': 691, 'walk': 29} | 0.040 | — | — | — |
| jev_ask_plain | {'PROPOSE': 957, 'WALK': 196, 'ACCEPT': 12} | {'AMBITIOUS': 889, 'SAFE': 67, 'PLAN': 1} | {'accept': 413, 'walk': 307} | 0.426 | 0.203 | 0.591 | 1.561 |
| jev_noask_plain | {'PROPOSE': 959, 'WALK': 197, 'ACCEPT': 10} | {'AMBITIOUS': 901, 'SAFE': 58} | {'walk': 313, 'accept': 407} | 0.435 | — | 0.735 | 0.918 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 221, '1': 272, '2': 128, '3': 144, '4': 139, '5': 143, '6': 73, '7': 23, '8': 22} over 1165 decisions; well-formed share 1.000.
- jev_noask_plain: {'0': 322, '1': 351, '2': 231, '3': 128, '4': 94, '5': 40} over 1166 decisions; well-formed share 1.000.

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0632 [+0.0167, +0.1121] vs exact Δ\*_menu 0.0577 in Y units (0.0346 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0443 [-0.0071, +0.0966] vs exact Δ\*_menu 0.0306 in Y units (0.0184 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0820 [+0.0370, +0.1299] vs exact Δ\*_menu 0.0847 in Y units (0.0508 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.019); not more often: greedy_eig (+0.000), jev_ask_plain (+0.000).

## H = 4, c_q = 3/100 — `results/live_phase1_jev_h4_cq3_20261029`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random', 'jev_ask_plain', 'jev_noask_plain'], 4320 episodes (5 repeats per cell, seed 24), 0 error stubs, 447.68 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.5856 | 0.3514 | 0.3702 | 0.990 | 0.626 | 1.150 | 0.0000 | 1.000 | 0.783 | 0.314 |
| exact_noask | 720 | 0.5702 | 0.3421 | 0.3421 | 0.906 | 0.000 | 1.483 | 0.0000 | 1.000 | 0.410 | 0.369 |
| greedy_eig | 720 | 0.4598 | 0.2759 | 0.2759 | 0.953 | 0.000 | 1.613 | 0.0742 | 0.337 | 0.000 | 0.399 |
| random | 720 | -0.0509 | -0.0305 | -0.0059 | 0.992 | 0.821 | 0.650 | 0.3832 | 0.190 | 0.671 | 0.712 |
| jev_ask_plain | 720 | 0.4922 | 0.2953 | 0.2955 | 0.789 | 0.006 | 1.596 | 0.0529 | 0.373 | 0.054 | 0.457 |
| jev_noask_plain | 720 | 0.4984 | 0.2990 | 0.2990 | 0.786 | 0.000 | 1.608 | 0.0419 | 0.376 | 0.041 | 0.449 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.5579 | 360 | 0.6134 |
| exact_noask | 360 | 0.5275 | 360 | 0.6129 |
| greedy_eig | 360 | 0.3225 | 360 | 0.5972 |
| random | 360 | -0.0009 | 360 | -0.1009 |
| jev_ask_plain | 360 | 0.4170 | 360 | 0.5675 |
| jev_noask_plain | 360 | 0.4220 | 360 | 0.5747 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0155 [-0.0025, +0.0349] | 36 | +0.0304 [-0.0052, +0.0682] | 36 | +0.0006 [-0.0029, +0.0047] | 36 |
| jev_ask_plain-jev_noask_plain | -0.0061 [-0.0120, -0.0008] | 36 | -0.0050 [-0.0124, +0.0017] | 36 | -0.0072 [-0.0190, +0.0028] | 36 |
| exact_noask-exact_ask | -0.0155 [-0.0348, +0.0025] | 36 | -0.0304 [-0.0681, +0.0052] | 36 | -0.0006 [-0.0047, +0.0029] | 36 |
| greedy_eig-exact_ask | -0.1258 [-0.1857, -0.0724] | 36 | -0.2354 [-0.3443, -0.1395] | 36 | -0.0163 [-0.0290, -0.0048] | 36 |
| random-exact_ask | -0.6365 [-0.6756, -0.5975] | 36 | -0.5588 [-0.6214, -0.4945] | 36 | -0.7143 [-0.7519, -0.6741] | 36 |
| jev_ask_plain-exact_ask | -0.0934 [-0.1203, -0.0684] | 36 | -0.1409 [-0.1788, -0.1066] | 36 | -0.0460 [-0.0675, -0.0269] | 36 |
| jev_noask_plain-exact_ask | -0.0873 [-0.1148, -0.0617] | 36 | -0.1358 [-0.1736, -0.1019] | 36 | -0.0387 [-0.0604, -0.0202] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +0.6264 [+0.5972, +0.6569] | +1.1750 [+1.1056, +1.2472] | +0.0778 [+0.0500, +0.1083] |
| jev_ask_plain-jev_noask_plain | +0.0056 [+0.0000, +0.0125] | +0.0111 [+0.0000, +0.0250] | +0.0000 [+0.0000, +0.0000] |
| exact_noask-exact_ask | -0.6264 [-0.6569, -0.5972] | -1.1750 [-1.2472, -1.1056] | -0.0778 [-0.1083, -0.0500] |
| greedy_eig-exact_ask | -0.6264 [-0.6569, -0.5972] | -1.1750 [-1.2472, -1.1056] | -0.0778 [-0.1083, -0.0500] |
| random-exact_ask | +0.1944 [+0.1083, +0.2833] | -0.2278 [-0.3861, -0.0722] | +0.6167 [+0.5500, +0.6861] |
| jev_ask_plain-exact_ask | -0.6208 [-0.6528, -0.5903] | -1.1639 [-1.2389, -1.0944] | -0.0778 [-0.1083, -0.0500] |
| jev_noask_plain-exact_ask | -0.6264 [-0.6569, -0.5972] | -1.1750 [-1.2472, -1.1056] | -0.0778 [-0.1083, -0.0500] |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0095 | 0.0095 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 24 |
| focal_opens | 12 | 0.0170 | 0.0170 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |
| opponent_opens | 12 | 0.0020 | 0.0020 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0155 [-0.0025, +0.0349] | 0.0158 | +0.9788 [-0.1550, +2.2046] | 0.0158 | +0.9788 [-0.1550, +2.2046] |
| exact_ask-exact_noask | focal_opens | +0.0304 [-0.0052, +0.0682] | 0.0284 | +1.0711 [-0.1823, +2.4038] | 0.0284 | +1.0711 [-0.1823, +2.4038] |
| exact_ask-exact_noask | opponent_opens | +0.0006 [-0.0029, +0.0047] | 0.0033 | +0.1730 [-0.8893, +1.4402] | 0.0033 | +0.1730 [-0.8893, +1.4402] |
| jev_ask_plain-jev_noask_plain | all | -0.0061 [-0.0120, -0.0008] | 0.0158 | -0.3881 [-0.7565, -0.0519] | 0.0158 | -0.3881 [-0.7565, -0.0519] |
| jev_ask_plain-jev_noask_plain | focal_opens | -0.0050 [-0.0124, +0.0017] | 0.0284 | -0.1777 [-0.4372, +0.0592] | 0.0284 | -0.1777 [-0.4372, +0.0592] |
| jev_ask_plain-jev_noask_plain | opponent_opens | -0.0072 [-0.0190, +0.0028] | 0.0033 | -2.2232 [-5.8483, +0.8652] | 0.0033 | -2.2232 [-5.8483, +0.8652] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 451/451 | 1.000 | 0/1377 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/531 | 0.000 | 0/630 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 245/559 | 0.438 | 346/920 | 0.376 | +0.062 [+0.011, +0.112] |
| jev_ask_plain | 4/528 | 0.008 | 0/632 | 0.000 | +0.008 [+0.000, +0.017] |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| exact_ask | {'ASK': 451, 'REJECT': 549, 'PROPOSE': 828} | {'SAFE': 670, 'AMBITIOUS': 158} | {'accept': 713, 'walk': 7} | 0.010 | — | — | — |
| exact_noask | {'PROPOSE': 1068, 'REJECT': 520} | {'PLAN': 360, 'SAFE': 430, 'AMBITIOUS': 278} | {'accept': 652, 'walk': 68} | 0.094 | — | — | — |
| greedy_eig | {'PROPOSE': 1161, 'ACCEPT': 88} | {'AMBITIOUS': 730, 'PLAN': 152, 'SAFE': 279} | {'accept': 686, 'walk': 34} | 0.047 | — | — | — |
| random | {'ASK': 591, 'REJECT': 202, 'ACCEPT': 405, 'PROPOSE': 468} | {'SAFE': 247, 'AMBITIOUS': 156, 'PLAN': 65} | {'accept': 714, 'walk': 6} | 0.008 | — | — | — |
| jev_ask_plain | {'PROPOSE': 1149, 'WALK': 111, 'ACCEPT': 6, 'ASK': 4, 'REJECT': 1} | {'AMBITIOUS': 849, 'SAFE': 293, 'PLAN': 7} | {'walk': 152, 'accept': 568} | 0.211 | 0.200 | 0.513 | 1.798 |
| jev_noask_plain | {'PROPOSE': 1158, 'WALK': 113} | {'AMBITIOUS': 866, 'SAFE': 292} | {'walk': 154, 'accept': 566} | 0.214 | — | 0.659 | 1.118 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 229, '1': 233, '2': 161, '3': 149, '4': 180, '5': 176, '6': 103, '7': 37, '8': 3} over 1271 decisions; well-formed share 1.000.
- jev_noask_plain: {'0': 290, '1': 348, '2': 249, '3': 256, '4': 109, '5': 19} over 1271 decisions; well-formed share 1.000.

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0155 [-0.0025, +0.0349] vs exact Δ\*_menu 0.0158 in Y units (0.0095 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0304 [-0.0052, +0.0682] vs exact Δ\*_menu 0.0284 in Y units (0.0170 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0006 [-0.0029, +0.0047] vs exact Δ\*_menu 0.0033 in Y units (0.0020 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.062), jev_ask_plain (+0.008); not more often: greedy_eig (+0.000).

## H = 4, c_q = 0 — `results/live_phase1_jev_h4_cq0_full_20261029`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `full` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random', 'jev_ask_plain', 'jev_noask_plain'], 4320 episodes (5 repeats per cell, seed 25), 0 error stubs, 509.18 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.6316 | 0.3790 | 0.3790 | 1.000 | 1.478 | 1.000 | 0.0000 | 1.000 | 1.018 | 0.304 |
| exact_noask | 720 | 0.5661 | 0.3397 | 0.3397 | 0.893 | 0.000 | 1.469 | 0.0000 | 1.000 | 0.461 | 0.359 |
| greedy_eig | 720 | 0.4656 | 0.2793 | 0.2793 | 0.940 | 0.000 | 1.618 | 0.0996 | 0.270 | 0.000 | 0.418 |
| random | 720 | -0.0508 | -0.0305 | -0.0305 | 0.803 | 0.728 | 1.317 | 0.4147 | 0.154 | 1.087 | 0.781 |
| jev_ask_plain | 720 | 0.0351 | 0.0211 | 0.0211 | 0.067 | 0.000 | 1.956 | 0.3593 | 0.201 | 1.594 | 1.285 |
| jev_noask_plain | 720 | 0.0250 | 0.0150 | 0.0150 | 0.051 | 0.000 | 1.974 | 0.3259 | 0.202 | 1.606 | 1.298 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.6314 | 360 | 0.6318 |
| exact_noask | 360 | 0.5267 | 360 | 0.6055 |
| greedy_eig | 360 | 0.3417 | 360 | 0.5894 |
| random | 360 | -0.0746 | 360 | -0.0269 |
| jev_ask_plain | 360 | 0.0468 | 360 | 0.0235 |
| jev_noask_plain | 360 | 0.0420 | 360 | 0.0081 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0655 [+0.0482, +0.0848] | 36 | +0.1047 [+0.0764, +0.1362] | 36 | +0.0262 [+0.0156, +0.0388] | 36 |
| jev_ask_plain-jev_noask_plain | +0.0101 [+0.0015, +0.0208] | 36 | +0.0048 [-0.0068, +0.0184] | 36 | +0.0154 [+0.0008, +0.0324] | 36 |
| exact_noask-exact_ask | -0.0655 [-0.0847, -0.0482] | 36 | -0.1047 [-0.1362, -0.0764] | 36 | -0.0262 [-0.0388, -0.0156] | 36 |
| greedy_eig-exact_ask | -0.1660 [-0.2250, -0.1153] | 36 | -0.2897 [-0.3910, -0.2016] | 36 | -0.0423 [-0.0604, -0.0256] | 36 |
| random-exact_ask | -0.6824 [-0.7251, -0.6389] | 36 | -0.7060 [-0.7579, -0.6516] | 36 | -0.6587 [-0.7177, -0.6033] | 36 |
| jev_ask_plain-exact_ask | -0.5964 [-0.6633, -0.5274] | 36 | -0.5846 [-0.6483, -0.5190] | 36 | -0.6083 [-0.6796, -0.5338] | 36 |
| jev_noask_plain-exact_ask | -0.6066 [-0.6764, -0.5349] | 36 | -0.5895 [-0.6549, -0.5232] | 36 | -0.6236 [-0.6978, -0.5467] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +1.4778 [+1.4306, +1.5292] | +1.6278 [+1.5389, +1.7167] | +1.3278 [+1.2722, +1.3833] |
| jev_ask_plain-jev_noask_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| exact_noask-exact_ask | -1.4778 [-1.5278, -1.4306] | -1.6278 [-1.7167, -1.5389] | -1.3278 [-1.3833, -1.2722] |
| greedy_eig-exact_ask | -1.4778 [-1.5278, -1.4306] | -1.6278 [-1.7167, -1.5389] | -1.3278 [-1.3833, -1.2722] |
| random-exact_ask | -0.7500 [-0.8292, -0.6722] | -0.8333 [-0.9556, -0.7083] | -0.6667 [-0.7639, -0.5667] |
| jev_ask_plain-exact_ask | -1.4778 [-1.5278, -1.4306] | -1.6278 [-1.7167, -1.5389] | -1.3278 [-1.3833, -1.2722] |
| jev_noask_plain-exact_ask | -1.4778 [-1.5278, -1.4306] | -1.6278 [-1.7167, -1.5389] | -1.3278 [-1.3833, -1.2722] |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0383 | 0.0383 | 0.0000 | 0.0000 | — | — | 0 | 24 |
| focal_opens | 12 | 0.0612 | 0.0612 | 0.0000 | 0.0000 | — | — | 0 | 12 |
| opponent_opens | 12 | 0.0154 | 0.0154 | 0.0000 | 0.0000 | — | — | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | all | +0.0655 [+0.0482, +0.0848] | 0.0638 | +1.0258 [+0.7554, +1.3283] | 0.0638 | +1.0258 [+0.7554, +1.3283] |
| exact_ask-exact_noask | focal_opens | +0.1047 [+0.0764, +0.1362] | 0.1019 | +1.0267 [+0.7497, +1.3362] | 0.1019 | +1.0267 [+0.7497, +1.3362] |
| exact_ask-exact_noask | opponent_opens | +0.0262 [+0.0156, +0.0388] | 0.0257 | +1.0221 [+0.6074, +1.5116] | 0.0257 | +1.0221 [+0.6074, +1.5116] |
| jev_ask_plain-jev_noask_plain | all | +0.0101 [+0.0015, +0.0208] | 0.0638 | +0.1585 [+0.0229, +0.3259] | 0.0638 | +0.1585 [+0.0229, +0.3259] |
| jev_ask_plain-jev_noask_plain | focal_opens | +0.0048 [-0.0068, +0.0184] | 0.1019 | +0.0475 [-0.0664, +0.1803] | 0.1019 | +0.0475 [-0.0664, +0.1803] |
| jev_ask_plain-jev_noask_plain | opponent_opens | +0.0154 [+0.0008, +0.0324] | 0.0257 | +0.5991 [+0.0305, +1.2620] | 0.0257 | +0.5991 [+0.0305, +1.2620] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 1064/1064 | 1.000 | 0/1096 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/688 | 0.000 | 0/477 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 329/909 | 0.362 | 195/639 | 0.305 | +0.057 [+0.002, +0.108] |
| jev_ask_plain | 0/880 | 0.000 | 0/555 | 0.000 | +0.000 [+0.000, +0.000] |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| exact_ask | {'ASK': 1064, 'REJECT': 376, 'PROPOSE': 720} | {'None': 720} | {'accept': 720} | 0.000 | — | — | — |
| exact_noask | {'PROPOSE': 1058, 'REJECT': 560} | {'None': 1058} | {'accept': 643, 'walk': 77} | 0.107 | — | — | — |
| greedy_eig | {'PROPOSE': 1165, 'ACCEPT': 78} | {'None': 1165} | {'accept': 677, 'walk': 43} | 0.060 | — | — | — |
| random | {'PROPOSE': 948, 'ACCEPT': 311, 'REJECT': 44, 'ASK': 524} | {'None': 948} | {'accept': 578, 'walk': 142} | 0.197 | — | — | — |
| jev_ask_plain | {'PROPOSE': 1408, 'WALK': 338, 'ACCEPT': 43, 'REJECT': 11} | {'None': 1408} | {'walk': 672, 'accept': 48} | 0.933 | 0.117 | 0.575 | 2.100 |
| jev_noask_plain | {'PROPOSE': 1421, 'WALK': 340, 'ACCEPT': 37, 'REJECT': 12} | {'None': 1421} | {'walk': 683, 'accept': 37} | 0.949 | — | 0.650 | 1.572 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 265, '1': 207, '2': 53, '3': 52, '4': 47, '5': 61, '6': 48, '7': 51, '8': 64, '9': 45, '10': 57, '11': 51, '12': 48, '13': 63, '14': 48, '15': 68, '16': 62, '17': 65, '18': 50, '19': 62, '20': 54, '21': 65, '22': 66, '23': 45, '24': 50, '25': 53} over 1800 decisions; well-formed share 1.000.
- jev_noask_plain: {'0': 266, '1': 234, '2': 67, '3': 88, '4': 72, '5': 62, '6': 95, '7': 78, '8': 87, '9': 71, '10': 77, '11': 88, '12': 73, '13': 89, '14': 80, '15': 86, '16': 73, '17': 66, '18': 58} over 1810 decisions; well-formed share 1.000.

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0655 [+0.0482, +0.0848] vs exact Δ\*_menu 0.0638 in Y units (0.0383 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.1047 [+0.0764, +0.1362] vs exact Δ\*_menu 0.1019 in Y units (0.0612 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0262 [+0.0156, +0.0388] vs exact Δ\*_menu 0.0257 in Y units (0.0154 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.057); not more often: greedy_eig (+0.000), jev_ask_plain (+0.000).

## H = 4, c_q = 0 — `results/live_phase1_jev_h4_cq0_replicate_20261029`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['jev_ask_plain', 'jev_noask_plain'], 288 episodes (1 repeats per cell, seed 31), 0 error stubs, 234.53 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_ask_plain | 144 | 0.4831 | 0.2899 | 0.2899 | 0.708 | 0.000 | 1.785 | 0.1016 | 0.312 | 0.338 | 0.380 |
| jev_noask_plain | 144 | 0.4904 | 0.2942 | 0.2942 | 0.701 | 0.000 | 1.792 | 0.0585 | 0.366 | 0.319 | 0.366 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| jev_ask_plain | 72 | 0.3948 | 72 | 0.5714 |
| jev_noask_plain | 72 | 0.4122 | 72 | 0.5686 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| jev_ask_plain-jev_noask_plain | -0.0073 [-0.0204, +0.0039] | 36 | -0.0175 [-0.0413, +0.0002] | 36 | +0.0028 [-0.0075, +0.0128] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| jev_ask_plain-jev_noask_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

### Exact ceilings and the costlessness check

| cell set | cells | mean Δ\*_full | mean Δ\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\*_menu > 0 |
|---|---|---|---|---|---|---|---|---|---|
| all | 24 | 0.0383 | 0.0383 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 24 |
| focal_opens | 12 | 0.0612 | 0.0612 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |
| opponent_opens | 12 | 0.0154 | 0.0154 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 |

Δ\*_full = V\*_ask − V\*_noask under the unrestricted action set; Δ\*_menu = the same under the shortlist menu (the ceiling for any arm that sees the menu). Losses are V\*(full) − V\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.

### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)

| contrast | cell set | paired ΔY | mean Δ\*_menu (Y units) | fraction of Δ\*_menu | mean Δ\*_full (Y units) | fraction of Δ\*_full |
|---|---|---|---|---|---|---|
| jev_ask_plain-jev_noask_plain | all | -0.0073 [-0.0204, +0.0039] | 0.0638 | -0.1144 [-0.3195, +0.0614] | 0.0638 | -0.1144 [-0.3195, +0.0614] |
| jev_ask_plain-jev_noask_plain | focal_opens | -0.0175 [-0.0413, +0.0002] | 0.1019 | -0.1712 [-0.4052, +0.0017] | 0.1019 | -0.1712 [-0.4052, +0.0017] |
| jev_ask_plain-jev_noask_plain | opponent_opens | +0.0028 [-0.0075, +0.0128] | 0.0257 | +0.1109 [-0.2912, +0.4993] | 0.0257 | +0.1109 [-0.2912, +0.4993] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| jev_ask_plain | 0/177 | 0.000 | 0/81 | 0.000 | +0.000 [+0.000, +0.000] |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| jev_ask_plain | {'PROPOSE': 257, 'WALK': 33, 'ACCEPT': 5} | {'PLAN_ASK': 72, 'AMBITIOUS': 144, 'SAFE': 40, 'PLAN': 1} | {'walk': 42, 'accept': 102} | 0.292 | 0.176 | 0.587 | 1.592 |
| jev_noask_plain | {'PROPOSE': 258, 'WALK': 32, 'ACCEPT': 5} | {'PLAN_ASK': 72, 'AMBITIOUS': 148, 'SAFE': 38} | {'walk': 43, 'accept': 101} | 0.299 | — | 0.701 | 1.016 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 56, '1': 57, '2': 24, '3': 34, '4': 43, '5': 34, '6': 26, '7': 19, '8': 2} over 295 decisions; well-formed share 1.000.
- jev_noask_plain: {'0': 59, '1': 80, '2': 59, '3': 40, '4': 48, '5': 9} over 295 decisions; well-formed share 1.000.

### Read-out (computed)

- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: none; not more often: jev_ask_plain (+0.000).

## Pre-registered verdicts — `results/live_phase1_jev_h4_cq0_20261029/verdicts.json`

Rules applied by `verdicts_live.py` (δ = 0.02, primary cells = focal_opens). Sanity gates pass: **True**.

| rule | quantity | prediction | estimate [95% CI] | verdict |
|---|---|---|---|---|
| P1 | Δ Y, jev_ask_plain-jev_noask_plain, focal_opens, H=4 c_q=0 (PRIMARY) | > 0; material if lower CI bound > 0.02 | -0.0104 [-0.0207, -0.0019] (n = 36) | **negative** |
| S1 | Δ Y, jev_ask_plain-jev_noask_plain, all cells, H=4 c_q=0 | > 0 | -0.0053 [-0.0120, +0.0007] (n = 36) | **unresolved** |
| S2 | Δ Y, jev_ask_eig-jev_noask_eig, focal_opens, H=4 c_q=0 | > 0 | +0.0035 [-0.0056, +0.0127] (n = 36) | **unresolved** |
| S3 | Δ questions asked / episode, jev_ask_eig-jev_ask_plain, all cells (EIG shown -> more asking) | > 0 | +0.0000 [+0.0000, +0.0000] (n = 36) | **not supported** |
| S4 | Δ Y, jev_ask_eig-jev_ask_plain, focal_opens (EIG shown does NOT raise net gain) | not > 0 | +0.0132 [+0.0041, +0.0233] (n = 36) | **not supported** |
| S5_jev_ask_plain | P(ask | ASK strictly optimal) − P(ask | ASK legal, not optimal), jev_ask_plain, H=4 c_q=0 | > 0 | +0.0000 [+0.0000, +0.0000] (n = 36) | **not supported** |
| S5_jev_ask_eig | P(ask | ASK strictly optimal) − P(ask | ASK legal, not optimal), jev_ask_eig, H=4 c_q=0 | > 0 | +0.0000 [+0.0000, +0.0000] (n = 36) | **not supported** |
| S6 | Δ Y, jev_ask_plain-exact_ask, focal_opens, H=4 c_q=0 (gap to the planner) | < 0 | -0.2765 [-0.3155, -0.2343] (n = 36) | **confirmed** |
| S7 | Δ Y, jev_ask_plain-jev_noask_plain, all cells, H=2 c_q=0 (negative control: ASK never optimal) | not > 0 | -0.0010 [-0.0063, +0.0045] (n = 36) | **confirmed** |
| S8 | Δ Y, jev_ask_plain-jev_noask_plain, all cells, H=3 c_q=0 (positive control across kernels) | > 0 | -0.0014 [-0.0058, +0.0028] (n = 36) | **unresolved** |
| S9 | questions / episode, jev_ask_plain, H=4 c_q=3/100 vs c_q=0 (does the policy respect question cost) | < at c_q=0 (CIs disjoint) | c_q=0: 0.000 [0.0, 0.0]; c_q=3/100: 0.006 [0.0, 0.013] | **not supported** |
| S10 | Δ Y, jev_ask_plain-jev_noask_plain, focal_opens, H=4 c_q=3/100 | > 0 | -0.0050 [-0.0124, +0.0017] (n = 36) | **unresolved** |
| S11 | Δ Y, jev_ask_plain-jev_noask_plain, focal_opens, H=4 c_q=0, FULL 27-deal menu (named secondary arm) | > 0 | +0.0048 [-0.0068, +0.0184] (n = 36) | **unresolved** |

- P1 captured fraction of the exact ceiling: -0.1024 [-0.2030, -0.0185] of Δ\*_menu (0.1019 in Y units); -0.1024 [-0.2030, -0.0185] of Δ\*_full (0.1019).
- gate `primary`: pass = True; error stubs 0, episodes per arm {'exact_ask': 720, 'exact_noask': 720, 'greedy_eig': 720, 'random': 720, 'jev_ask_plain': 720, 'jev_ask_eig': 720, 'jev_noask_plain': 720, 'jev_noask_eig': 720}, well-formed {'jev_ask_plain': 1.0, 'jev_ask_eig': 1.0, 'jev_noask_plain': 1.0, 'jev_noask_eig': 1.0}
- gate `h2`: pass = True; error stubs 0, episodes per arm {'exact_ask': 720, 'exact_noask': 720, 'greedy_eig': 720, 'random': 720, 'jev_ask_plain': 720, 'jev_noask_plain': 720}, well-formed {'jev_ask_plain': 1.0, 'jev_noask_plain': 1.0}
- gate `h3`: pass = True; error stubs 0, episodes per arm {'exact_ask': 720, 'exact_noask': 720, 'greedy_eig': 720, 'random': 720, 'jev_ask_plain': 720, 'jev_noask_plain': 720}, well-formed {'jev_ask_plain': 1.0, 'jev_noask_plain': 1.0}
- gate `cq3`: pass = True; error stubs 0, episodes per arm {'exact_ask': 720, 'exact_noask': 720, 'greedy_eig': 720, 'random': 720, 'jev_ask_plain': 720, 'jev_noask_plain': 720}, well-formed {'jev_ask_plain': 1.0, 'jev_noask_plain': 1.0}
- gate `full`: pass = True; error stubs 0, episodes per arm {'exact_ask': 720, 'exact_noask': 720, 'greedy_eig': 720, 'random': 720, 'jev_ask_plain': 720, 'jev_noask_plain': 720}, well-formed {'jev_ask_plain': 1.0, 'jev_noask_plain': 1.0}
- gate `random_matches_frozen`: pass = True; {"jev_run_random_mean_Y": -0.02067307692307691, "frozen_random_ci95": [-0.09834401709401708, 0.07282763532763534]}

## Regenerate

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h4_cq0_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h2_cq0_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h3_cq0_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h4_cq3_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h4_cq0_full_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_jev_h4_cq0_replicate_20261029
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note --dirs results/live_phase1_jev_h4_cq0_20261029,results/live_phase1_jev_h2_cq0_20261029,results/live_phase1_jev_h3_cq0_20261029,results/live_phase1_jev_h4_cq3_20261029,results/live_phase1_jev_h4_cq0_full_20261029,results/live_phase1_jev_h4_cq0_replicate_20261029 --out llm_rl_scripts/car_dealer_live/JEV_RESULTS_LIVE.md --verdicts results/live_phase1_jev_h4_cq0_20261029/verdicts.json
```
