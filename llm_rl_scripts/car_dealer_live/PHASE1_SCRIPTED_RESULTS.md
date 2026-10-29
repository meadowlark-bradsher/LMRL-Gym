# Live Car-Dealer, Phase 1 — scripted arms against the T1 kernel

_Rendered by `render_phase1_note.py` from `analysis.json` files written by `analyze_phase1.py`; every number is formatted from JSON, none is typed._

This is a scripted-arm note: it contains no model result and no registration has been made. Arms: exact_ask (C_plan, the menu-restricted exact oracle with ASK), exact_noask (B_plan, the same oracle with ASK forbidden), greedy_eig, random; any jev_* arm present ran against the offline mock backend unless the section's config says otherwise.

Result dirs:

- `results/live_phase1_scripted3_20261029/h4_cq0` — env `live-car-dealer-v0.1`, H = 4, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.
- `results/live_phase1_scripted3_20261029/h2_cq0` — env `live-car-dealer-v0.1`, H = 2, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.
- `results/live_phase1_scripted3_20261029/h3_cq0` — env `live-car-dealer-v0.1`, H = 3, c_q = 0, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.
- `results/live_phase1_scripted3_20261029/h4_cq3` — env `live-car-dealer-v0.1`, H = 4, c_q = 3/100, regime easy, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist`, 2880 episodes.

Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r), with U_max = 1 for every type; no deal gives Y = 0. Contrasts are paired over clusters (focal, opener, focal type, opponent type) with a percentile cluster bootstrap.

## H = 4, c_q = 0 — `results/live_phase1_scripted3_20261029/h4_cq0`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 178.79 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.6322 | 0.3793 | 0.3793 | 1.000 | 1.517 | 1.000 | 0.0000 | 1.000 | 0.880 | 0.269 |
| exact_noask | 720 | 0.5665 | 0.3399 | 0.3399 | 0.894 | 0.000 | 1.478 | 0.0000 | 1.000 | 0.408 | 0.349 |
| greedy_eig | 720 | 0.4632 | 0.2779 | 0.2779 | 0.953 | 0.000 | 1.608 | 0.1021 | 0.273 | 0.000 | 0.388 |
| random | 720 | -0.0132 | -0.0079 | -0.0079 | 0.986 | 0.794 | 0.733 | 0.3929 | 0.335 | 0.732 | 0.733 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.6327 | 360 | 0.6318 |
| exact_noask | 360 | 0.5323 | 360 | 0.6008 |
| greedy_eig | 360 | 0.3273 | 360 | 0.5991 |
| random | 360 | 0.0386 | 360 | -0.0650 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0657 [+0.0481, +0.0853] | 36 | +0.1004 [+0.0721, +0.1311] | 36 | +0.0310 [+0.0178, +0.0458] | 36 |
| exact_noask-exact_ask | -0.0657 [-0.0853, -0.0481] | 36 | -0.1004 [-0.1311, -0.0721] | 36 | -0.0310 [-0.0458, -0.0178] | 36 |
| greedy_eig-exact_ask | -0.1690 [-0.2255, -0.1190] | 36 | -0.3054 [-0.4038, -0.2167] | 36 | -0.0327 [-0.0534, -0.0154] | 36 |
| random-exact_ask | -0.6454 [-0.6748, -0.6138] | 36 | -0.5941 [-0.6484, -0.5402] | 36 | -0.6967 [-0.7419, -0.6528] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +1.5167 [+1.4722, +1.5639] | +1.8028 [+1.7167, +1.8833] | +1.2306 [+1.1556, +1.3056] |
| exact_noask-exact_ask | -1.5167 [-1.5639, -1.4722] | -1.8028 [-1.8833, -1.7139] | -1.2306 [-1.3056, -1.1556] |
| greedy_eig-exact_ask | -1.5167 [-1.5639, -1.4722] | -1.8028 [-1.8833, -1.7139] | -1.2306 [-1.3056, -1.1556] |
| random-exact_ask | -0.7222 [-0.8139, -0.6306] | -0.9972 [-1.1194, -0.8778] | -0.4472 [-0.5806, -0.3111] |

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
| exact_ask-exact_noask | all | +0.0657 [+0.0481, +0.0853] | 0.0638 | +1.0291 [+0.7543, +1.3364] | 0.0638 | +1.0291 [+0.7543, +1.3364] |
| exact_ask-exact_noask | focal_opens | +0.1004 [+0.0721, +0.1311] | 0.1019 | +0.9844 [+0.7074, +1.2863] | 0.1019 | +0.9844 [+0.7074, +1.2863] |
| exact_ask-exact_noask | opponent_opens | +0.0310 [+0.0178, +0.0458] | 0.0257 | +1.2065 [+0.6934, +1.7849] | 0.0257 | +1.2065 [+0.6934, +1.7849] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 1092/1092 | 1.000 | 0/1068 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/677 | 0.000 | 0/481 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 311/833 | 0.373 | 261/662 | 0.394 | -0.021 [-0.075, +0.036] |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0657 [+0.0481, +0.0853] vs exact Δ\*_menu 0.0638 in Y units (0.0383 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.1004 [+0.0721, +0.1311] vs exact Δ\*_menu 0.1019 in Y units (0.0612 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0310 [+0.0178, +0.0458] vs exact Δ\*_menu 0.0257 in Y units (0.0154 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000); not more often: greedy_eig (+0.000), random (-0.021).

## H = 2, c_q = 0 — `results/live_phase1_scripted3_20261029/h2_cq0`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 4.84 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.4051 | 0.2431 | 0.2431 | 0.667 | 0.000 | 1.000 | 0.0000 | 1.000 | 0.302 | 1.154 |
| exact_noask | 720 | 0.4051 | 0.2431 | 0.2431 | 0.667 | 0.000 | 1.000 | 0.0000 | 1.000 | 0.292 | 1.154 |
| greedy_eig | 720 | -0.0296 | -0.0178 | -0.0178 | 0.856 | 0.199 | 0.801 | 0.2610 | 0.141 | 0.000 | 0.868 |
| random | 720 | -0.2004 | -0.1202 | -0.1202 | 0.951 | 0.471 | 0.396 | 0.3585 | 0.134 | 0.377 | 1.090 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.3248 | 360 | 0.4854 |
| exact_noask | 360 | 0.3248 | 360 | 0.4854 |
| greedy_eig | 360 | -0.0456 | 360 | -0.0136 |
| random | 360 | -0.2370 | 360 | -0.1638 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 |
| exact_noask-exact_ask | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 | +0.0000 [+0.0000, +0.0000] | 36 |
| greedy_eig-exact_ask | -0.4347 [-0.5042, -0.3655] | 36 | -0.3704 [-0.5223, -0.2203] | 36 | -0.4990 [-0.5752, -0.4233] | 36 |
| random-exact_ask | -0.6055 [-0.6681, -0.5481] | 36 | -0.5618 [-0.6339, -0.4906] | 36 | -0.6492 [-0.7393, -0.5575] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| exact_noask-exact_ask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| greedy_eig-exact_ask | +0.1986 [+0.1722, +0.2250] | +0.0000 [+0.0000, +0.0000] | +0.3972 [+0.3444, +0.4500] |
| random-exact_ask | +0.4708 [+0.4389, +0.5028] | +0.5472 [+0.4972, +0.5972] | +0.3944 [+0.3417, +0.4472] |

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

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 0/0 | — | 0/720 | 0.000 | — |
| greedy_eig | 0/0 | — | 143/720 | 0.199 | — |
| random | 0/0 | — | 339/720 | 0.471 | — |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0000 [+0.0000, +0.0000] vs exact Δ\*_menu 0.0000 in Y units (0.0000 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).

## H = 3, c_q = 0 — `results/live_phase1_scripted3_20261029/h3_cq0`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 26.18 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.5952 | 0.3571 | 0.3571 | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000 | 0.896 | 0.768 |
| exact_noask | 720 | 0.5381 | 0.3228 | 0.3228 | 0.742 | 0.000 | 1.333 | 0.0000 | 1.000 | 0.504 | 0.570 |
| greedy_eig | 720 | 0.3441 | 0.2065 | 0.2065 | 0.882 | 0.000 | 1.333 | 0.1544 | 0.221 | 0.000 | 0.503 |
| random | 720 | -0.0683 | -0.0410 | -0.0410 | 0.944 | 0.694 | 0.586 | 0.3945 | 0.176 | 0.610 | 0.815 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.5887 | 360 | 0.6017 |
| exact_noask | 360 | 0.5575 | 360 | 0.5186 |
| greedy_eig | 360 | 0.5575 | 360 | 0.1307 |
| random | 360 | -0.0244 | 360 | -0.1122 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0571 [+0.0168, +0.0998] | 36 | +0.0312 [-0.0142, +0.0792] | 36 | +0.0831 [+0.0427, +0.1261] | 36 |
| exact_noask-exact_ask | -0.0571 [-0.0998, -0.0168] | 36 | -0.0312 [-0.0791, +0.0142] | 36 | -0.0831 [-0.1261, -0.0427] | 36 |
| greedy_eig-exact_ask | -0.2511 [-0.3282, -0.1768] | 36 | -0.0312 [-0.0791, +0.0142] | 36 | -0.4710 [-0.5810, -0.3629] | 36 |
| random-exact_ask | -0.6635 [-0.7053, -0.6207] | 36 | -0.6131 [-0.6873, -0.5411] | 36 | -0.7139 [-0.7653, -0.6591] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +1.0000 [+1.0000, +1.0000] | +1.0000 [+1.0000, +1.0000] | +1.0000 [+1.0000, +1.0000] |
| exact_noask-exact_ask | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] |
| greedy_eig-exact_ask | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] | -1.0000 [-1.0000, -1.0000] |
| random-exact_ask | -0.3056 [-0.3597, -0.2500] | -0.1694 [-0.2500, -0.0889] | -0.4417 [-0.5278, -0.3556] |

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
| exact_ask-exact_noask | all | +0.0571 [+0.0168, +0.0998] | 0.0577 | +0.9906 [+0.2915, +1.7313] | 0.0577 | +0.9906 [+0.2915, +1.7313] |
| exact_ask-exact_noask | focal_opens | +0.0312 [-0.0142, +0.0792] | 0.0306 | +1.0185 [-0.4644, +2.5876] | 0.0306 | +1.0185 [-0.4644, +2.5876] |
| exact_ask-exact_noask | opponent_opens | +0.0831 [+0.0427, +0.1261] | 0.0847 | +0.9805 [+0.5035, +1.4882] | 0.0847 | +0.9805 [+0.5035, +1.4882] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 720/720 | 1.000 | 0/720 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/720 | 0.000 | 0/240 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 331/720 | 0.460 | 169/463 | 0.365 | +0.095 [+0.039, +0.150] |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0571 [+0.0168, +0.0998] vs exact Δ\*_menu 0.0577 in Y units (0.0346 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0312 [-0.0142, +0.0792] vs exact Δ\*_menu 0.0306 in Y units (0.0184 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0831 [+0.0427, +0.1261] vs exact Δ\*_menu 0.0847 in Y units (0.0508 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.095); not more often: greedy_eig (+0.000).

## H = 4, c_q = 3/100 — `results/live_phase1_scripted3_20261029/h4_cq3`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'greedy_eig', 'random'], 2880 episodes (5 repeats per cell, seed 0), 0 error stubs, 176.78 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.5831 | 0.3498 | 0.3693 | 0.986 | 0.650 | 1.144 | 0.0000 | 1.000 | 0.784 | 0.324 |
| exact_noask | 720 | 0.5665 | 0.3399 | 0.3399 | 0.894 | 0.000 | 1.478 | 0.0000 | 1.000 | 0.408 | 0.349 |
| greedy_eig | 720 | 0.4632 | 0.2779 | 0.2779 | 0.953 | 0.000 | 1.608 | 0.0734 | 0.334 | 0.000 | 0.388 |
| random | 720 | -0.0447 | -0.0268 | -0.0031 | 0.992 | 0.790 | 0.708 | 0.3824 | 0.180 | 0.662 | 0.722 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.5564 | 360 | 0.6097 |
| exact_noask | 360 | 0.5323 | 360 | 0.6008 |
| greedy_eig | 360 | 0.3273 | 360 | 0.5991 |
| random | 360 | 0.0158 | 360 | -0.1053 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0165 [-0.0032, +0.0379] | 36 | +0.0241 [-0.0110, +0.0612] | 36 | +0.0089 [+0.0008, +0.0185] | 36 |
| exact_noask-exact_ask | -0.0165 [-0.0379, +0.0032] | 36 | -0.0241 [-0.0612, +0.0111] | 36 | -0.0089 [-0.0184, -0.0008] | 36 |
| greedy_eig-exact_ask | -0.1199 [-0.1778, -0.0685] | 36 | -0.2291 [-0.3320, -0.1365] | 36 | -0.0106 [-0.0258, +0.0023] | 36 |
| random-exact_ask | -0.6278 [-0.6612, -0.5914] | 36 | -0.5406 [-0.5959, -0.4838] | 36 | -0.7150 [-0.7624, -0.6690] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +0.6500 [+0.6167, +0.6847] | +1.1972 [+1.1194, +1.2806] | +0.1028 [+0.0639, +0.1472] |
| exact_noask-exact_ask | -0.6500 [-0.6833, -0.6167] | -1.1972 [-1.2806, -1.1167] | -0.1028 [-0.1472, -0.0639] |
| greedy_eig-exact_ask | -0.6500 [-0.6833, -0.6167] | -1.1972 [-1.2806, -1.1167] | -0.1028 [-0.1472, -0.0639] |
| random-exact_ask | +0.1403 [+0.0611, +0.2181] | -0.3778 [-0.5028, -0.2444] | +0.6583 [+0.5472, +0.7694] |

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
| exact_ask-exact_noask | all | +0.0165 [-0.0032, +0.0379] | 0.0158 | +1.0449 [-0.2037, +2.3962] | 0.0158 | +1.0449 [-0.2037, +2.3962] |
| exact_ask-exact_noask | focal_opens | +0.0241 [-0.0110, +0.0612] | 0.0284 | +0.8499 [-0.3860, +2.1589] | 0.0284 | +0.8499 [-0.3860, +2.1589] |
| exact_ask-exact_noask | opponent_opens | +0.0089 [+0.0008, +0.0185] | 0.0033 | +2.7456 [+0.2596, +5.6796] | 0.0033 | +2.7456 [+0.2596, +5.6796] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 468/468 | 1.000 | 0/1362 | 0.000 | +1.000 [+1.000, +1.000] |
| greedy_eig | 0/534 | 0.000 | 0/624 | 0.000 | +0.000 [+0.000, +0.000] |
| random | 223/558 | 0.400 | 346/908 | 0.381 | +0.019 [-0.036, +0.074] |

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0165 [-0.0032, +0.0379] vs exact Δ\*_menu 0.0158 in Y units (0.0095 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0241 [-0.0110, +0.0612] vs exact Δ\*_menu 0.0284 in Y units (0.0170 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0089 [+0.0008, +0.0185] vs exact Δ\*_menu 0.0033 in Y units (0.0020 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), random (+0.019); not more often: greedy_eig (+0.000).

## Menu size and costlessness — `results/live_phase1_scripted3_20261029/costlessness_h4_cq0.json`

`costlessness_check.py`, H = 4, c_q = 0, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, 105.3 s. Root values of the exact oracle under the full action set and under the shortlist menu with n_offers × n_questions candidates (the offer rule yields at most three distinct candidates: SAFE, AMBITIOUS, BALANCED). Utility units.

| menu (offers × questions) | cells lossless | mean loss, ASK allowed | max loss, ASK allowed | mean loss, ASK forbidden | max loss, ASK forbidden | mean Δ\*_menu | mean Δ\*_full |
|---|---|---|---|---|---|---|---|
| 2x2 | 12/24 | 0.0000 | 0.0000 | 0.0074 | 0.0148 | 0.0457 | 0.0383 |
| 3x2 | 24/24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0383 | 0.0383 |
| 3x2c | 24/24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0383 | 0.0383 |

- Cells with any loss under any menu size: focal opens.
- Stopping rule on the registered menu `3x2c` (threshold 0.005 on |mean Δ\*_menu − mean Δ\*_full|): all: artefact -0.0000; focal_opens: artefact -0.0000; opponent_opens: artefact +0.0000 → **PROCEED**.

## Menu size and costlessness — `results/live_phase1_scripted3_20261029/costlessness_h2_cq0.json`

`costlessness_check.py`, H = 2, c_q = 0, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, 1.9 s. Root values of the exact oracle under the full action set and under the shortlist menu with n_offers × n_questions candidates (the offer rule yields at most three distinct candidates: SAFE, AMBITIOUS, BALANCED). Utility units.

| menu (offers × questions) | cells lossless | mean loss, ASK allowed | max loss, ASK allowed | mean loss, ASK forbidden | max loss, ASK forbidden | mean Δ\*_menu | mean Δ\*_full |
|---|---|---|---|---|---|---|---|
| 2x2 | 12/24 | 0.0214 | 0.0428 | 0.0214 | 0.0428 | 0.0000 | 0.0000 |
| 3x2 | 24/24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3x2c | 24/24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- Cells with any loss under any menu size: opponent opens.
- Stopping rule on the registered menu `3x2c` (threshold 0.005 on |mean Δ\*_menu − mean Δ\*_full|): all: artefact +0.0000; focal_opens: artefact +0.0000; opponent_opens: artefact +0.0000 → **PROCEED**.

## Menu size and costlessness — `results/live_phase1_scripted3_20261029/costlessness_h3_cq0.json`

`costlessness_check.py`, H = 3, c_q = 0, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, 16.0 s. Root values of the exact oracle under the full action set and under the shortlist menu with n_offers × n_questions candidates (the offer rule yields at most three distinct candidates: SAFE, AMBITIOUS, BALANCED). Utility units.

| menu (offers × questions) | cells lossless | mean loss, ASK allowed | max loss, ASK allowed | mean loss, ASK forbidden | max loss, ASK forbidden | mean Δ\*_menu | mean Δ\*_full |
|---|---|---|---|---|---|---|---|
| 2x2 | 0/24 | 0.0294 | 0.0424 | 0.0187 | 0.0270 | 0.0239 | 0.0346 |
| 3x2 | 0/24 | 0.0148 | 0.0184 | 0.0000 | 0.0000 | 0.0198 | 0.0346 |
| 3x2c | 24/24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0346 | 0.0346 |

- Cells with any loss under any menu size: opponent opens, focal opens.
- Stopping rule on the registered menu `3x2c` (threshold 0.005 on |mean Δ\*_menu − mean Δ\*_full|): all: artefact +0.0000; focal_opens: artefact +0.0000; opponent_opens: artefact +0.0000 → **PROCEED**.

## Menu size and costlessness — `results/live_phase1_scripted3_20261029/costlessness_h4_cq3.json`

`costlessness_check.py`, H = 4, c_q = 3/100, kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, 104.2 s. Root values of the exact oracle under the full action set and under the shortlist menu with n_offers × n_questions candidates (the offer rule yields at most three distinct candidates: SAFE, AMBITIOUS, BALANCED). Utility units.

| menu (offers × questions) | cells lossless | mean loss, ASK allowed | max loss, ASK allowed | mean loss, ASK forbidden | max loss, ASK forbidden | mean Δ\*_menu | mean Δ\*_full |
|---|---|---|---|---|---|---|---|
| 2x2 | 12/24 | 0.0051 | 0.0103 | 0.0074 | 0.0148 | 0.0118 | 0.0095 |
| 3x2 | 12/24 | 0.0012 | 0.0023 | 0.0000 | 0.0000 | 0.0083 | 0.0095 |
| 3x2c | 24/24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0095 | 0.0095 |

- Cells with any loss under any menu size: focal opens.
- Stopping rule on the registered menu `3x2c` (threshold 0.005 on |mean Δ\*_menu − mean Δ\*_full|): all: artefact +0.0000; focal_opens: artefact +0.0000; opponent_opens: artefact +0.0000 → **PROCEED**.

## Regenerate

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted3_20261029/h4_cq0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted3_20261029/h2_cq0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted3_20261029/h3_cq0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_phase1_scripted3_20261029/h4_cq3
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note --dirs results/live_phase1_scripted3_20261029/h4_cq0,results/live_phase1_scripted3_20261029/h2_cq0,results/live_phase1_scripted3_20261029/h3_cq0,results/live_phase1_scripted3_20261029/h4_cq3 --out llm_rl_scripts/car_dealer_live/PHASE1_SCRIPTED_RESULTS.md --costlessness results/live_phase1_scripted3_20261029/costlessness_h4_cq0.json,results/live_phase1_scripted3_20261029/costlessness_h2_cq0.json,results/live_phase1_scripted3_20261029/costlessness_h3_cq0.json,results/live_phase1_scripted3_20261029/costlessness_h4_cq3.json
```
