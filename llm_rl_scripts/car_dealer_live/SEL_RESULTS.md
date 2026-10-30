# CD-SEL-1 — Jev shown what the history implies (live Car-Dealer, design Phase 2 selector arms)

_Rendered by `render_sel_note.py` from `analysis.json` (`analyze_phase1.py`) and `verdicts.json` (`verdicts_sel.py`) in `results/live_sel_jev_h4_cq0_20261030`; every number is formatted from JSON, none is typed. Rules registered in `PREREGISTRATION_SEL.md`._

Jev has seen the full event history (every proposal, answer, rejection, hold) in its state text since Phase 1. This study shows it, in separately named arms, what that history implies: the matched posterior over the opponent's six priority rankings in words (`belief`), the posterior probability that each offer is accepted (`pacc`, the B_passive selector), the same number under the never-updated uniform prior (`prior`, A_prior), a constant on every offer (`placebo`, P_placebo), and belief + posterior P(accept) together (`both`, with and without questions on the menu). The scripted `exact_ask` / `exact_noask` / `random` arms run in the same cells.

## What each arm is shown

| arm | questions on the menu | EIG per option | posterior over the opponent's rankings, in words | P(accept) printed on every offer |
|---|---|---|---|---|
| jev_ask_plain | yes | no | no | no |
| jev_ask_belief | yes | no | yes | no |
| jev_ask_pacc | yes | no | no | posterior |
| jev_ask_prior | yes | no | no | prior |
| jev_ask_placebo | yes | no | no | placebo |
| jev_ask_both | yes | no | yes | posterior |
| jev_noask_both | no | no | yes | posterior |

The placebo constant is 0.500 on every offer. Options are lettered and shuffled in every arm; no arm sees tags, Q\* or (except `eig` arms) EIG.

## Sanity gates and the drift check

- `pipeline`: **pass** — error_stubs 0, missing_arms [], episodes_per_arm {'exact_ask': 720, 'exact_noask': 720, 'random': 720, 'jev_ask_plain': 720, 'jev_ask_belief': 720, 'jev_ask_pacc': 720, 'jev_ask_prior': 720, 'jev_ask_placebo': 720, 'jev_ask_both': 720, 'jev_noask_both': 720}, well_formed {'jev_ask_plain': 1.0, 'jev_ask_belief': 1.0, 'jev_ask_pacc': 1.0, 'jev_ask_prior': 1.0, 'jev_ask_placebo': 1.0, 'jev_ask_both': 1.0, 'jev_noask_both': 1.0}
- `shuffle_active`: **pass** — positions_used_per_arm {'jev_ask_plain': 9, 'jev_ask_belief': 9, 'jev_ask_pacc': 8, 'jev_ask_prior': 8, 'jev_ask_placebo': 9, 'jev_ask_both': 8, 'jev_noask_both': 6}
- `random_matches_frozen`: **pass** — study_random_mean_Y -0.006303418803418786, frozen_random_ci95 [-0.09834401709401708, 0.07282763532763534]
- `exact_pair_reproduces_ceiling`: **pass** — contrast {'point': 0.0597934472934473, 'ci95': [0.0455840455840456, 0.07621082621082624], 'n_clusters': 36}, ceiling_delta_menu_Y 0.06381087747560826
- gates pass: **True**
- Drift check (reported, not a gate): the in-study `jev_ask_plain` mean Y 0.4578 vs the frozen Phase 1 run's 0.4653 [0.3729, 0.5582] — inside the Phase 1 CI; walk rate 0.342 vs 0.328; questions / episode 0.001 vs 0.000.

## H = 4, c_q = 0 — `results/live_sel_jev_h4_cq0_20261030`

Config: env `live-car-dealer-v0.1`, regime easy (r = 2/5), kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3, menu `shortlist` (3 offer candidates × 2 questions), arms ['exact_ask', 'exact_noask', 'random', 'jev_ask_plain', 'jev_ask_belief', 'jev_ask_pacc', 'jev_ask_prior', 'jev_ask_placebo', 'jev_ask_both', 'jev_noask_both'], 7200 episodes (5 repeats per cell, seed 51), 0 error stubs, 1020.45 s. Bootstrap B = 5000, clusters = (buyer ranking, seller ranking) type pair; 36 clusters.

### Per arm (all cells)

| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_ask | 720 | 0.6319 | 0.3792 | 0.3792 | 1.000 | 1.572 | 1.000 | 0.0000 | 1.000 | 0.913 | 0.306 |
| exact_noask | 720 | 0.5722 | 0.3433 | 0.3433 | 0.908 | 0.000 | 1.481 | 0.0000 | 1.000 | 0.440 | 0.379 |
| random | 720 | -0.0063 | -0.0038 | -0.0038 | 0.989 | 0.882 | 0.725 | 0.3865 | 0.350 | 0.745 | 0.695 |
| jev_ask_plain | 720 | 0.4578 | 0.2747 | 0.2747 | 0.658 | 0.001 | 1.776 | 0.1000 | 0.328 | 0.334 | 0.444 |
| jev_ask_belief | 720 | 0.4619 | 0.2771 | 0.2771 | 0.688 | 0.064 | 1.749 | 0.1008 | 0.334 | 0.362 | 0.454 |
| jev_ask_pacc | 720 | 0.4074 | 0.2444 | 0.2444 | 1.000 | 0.000 | 1.286 | 0.1350 | 0.140 | 0.522 | 1.222 |
| jev_ask_prior | 720 | 0.4090 | 0.2454 | 0.2454 | 0.994 | 0.001 | 1.261 | 0.1344 | 0.136 | 0.521 | 1.219 |
| jev_ask_placebo | 720 | 0.4546 | 0.2728 | 0.2728 | 0.658 | 0.001 | 1.749 | 0.1040 | 0.313 | 0.339 | 0.445 |
| jev_ask_both | 720 | 0.3999 | 0.2400 | 0.2400 | 1.000 | 0.014 | 1.276 | 0.1399 | 0.116 | 0.561 | 1.224 |
| jev_noask_both | 720 | 0.3978 | 0.2387 | 0.2387 | 1.000 | 0.000 | 1.314 | 0.1011 | 0.173 | 0.555 | 1.203 |

### Per arm, mean Y by who opens

| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |
|---|---|---|---|---|
| exact_ask | 360 | 0.6316 | 360 | 0.6323 |
| exact_noask | 360 | 0.5352 | 360 | 0.6091 |
| random | 360 | -0.0060 | 360 | -0.0066 |
| jev_ask_plain | 360 | 0.3508 | 360 | 0.5649 |
| jev_ask_belief | 360 | 0.3720 | 360 | 0.5518 |
| jev_ask_pacc | 360 | 0.3923 | 360 | 0.4225 |
| jev_ask_prior | 360 | 0.3946 | 360 | 0.4234 |
| jev_ask_placebo | 360 | 0.3626 | 360 | 0.5466 |
| jev_ask_both | 360 | 0.3875 | 360 | 0.4124 |
| jev_noask_both | 360 | 0.3894 | 360 | 0.4062 |

### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | n | focal opens | n | opponent opens | n |
|---|---|---|---|---|---|---|
| exact_ask-exact_noask | +0.0598 [+0.0456, +0.0762] | 36 | +0.0964 [+0.0703, +0.1243] | 36 | +0.0232 [+0.0146, +0.0334] | 36 |
| jev_ask_both-jev_noask_both | +0.0021 [-0.0085, +0.0128] | 36 | -0.0020 [-0.0134, +0.0098] | 36 | +0.0062 [-0.0126, +0.0255] | 36 |
| exact_noask-exact_ask | -0.0598 [-0.0762, -0.0456] | 36 | -0.0964 [-0.1243, -0.0702] | 36 | -0.0232 [-0.0334, -0.0146] | 36 |
| random-exact_ask | -0.6382 [-0.6794, -0.5962] | 36 | -0.6376 [-0.6950, -0.5793] | 36 | -0.6389 [-0.6887, -0.5885] | 36 |
| jev_ask_plain-exact_ask | -0.1741 [-0.2039, -0.1438] | 36 | -0.2808 [-0.3210, -0.2358] | 36 | -0.0674 [-0.0938, -0.0431] | 36 |
| jev_ask_belief-exact_ask | -0.1700 [-0.1996, -0.1403] | 36 | -0.2596 [-0.3001, -0.2161] | 36 | -0.0805 [-0.1088, -0.0546] | 36 |
| jev_ask_pacc-exact_ask | -0.2245 [-0.2613, -0.1868] | 36 | -0.2393 [-0.2717, -0.2058] | 36 | -0.2098 [-0.2536, -0.1652] | 36 |
| jev_ask_prior-exact_ask | -0.2230 [-0.2601, -0.1849] | 36 | -0.2370 [-0.2691, -0.2041] | 36 | -0.2089 [-0.2539, -0.1640] | 36 |
| jev_ask_placebo-exact_ask | -0.1773 [-0.2047, -0.1494] | 36 | -0.2689 [-0.3096, -0.2231] | 36 | -0.0857 [-0.1109, -0.0629] | 36 |
| jev_ask_both-exact_ask | -0.2320 [-0.2674, -0.1956] | 36 | -0.2441 [-0.2753, -0.2110] | 36 | -0.2199 [-0.2635, -0.1756] | 36 |
| jev_noask_both-exact_ask | -0.2342 [-0.2688, -0.1985] | 36 | -0.2422 [-0.2728, -0.2101] | 36 | -0.2261 [-0.2703, -0.1827] | 36 |
| jev_ask_belief-jev_ask_plain | +0.0041 [-0.0065, +0.0138] | 36 | +0.0212 [+0.0058, +0.0364] | 36 | -0.0131 [-0.0251, -0.0011] | 36 |
| jev_ask_pacc-jev_ask_plain | -0.0504 [-0.1106, +0.0093] | 36 | +0.0415 [-0.0282, +0.1051] | 36 | -0.1423 [-0.2064, -0.0765] | 36 |
| jev_ask_prior-jev_ask_plain | -0.0489 [-0.1117, +0.0117] | 36 | +0.0438 [-0.0250, +0.1080] | 36 | -0.1415 [-0.2054, -0.0773] | 36 |
| jev_ask_placebo-jev_ask_plain | -0.0032 [-0.0097, +0.0035] | 36 | +0.0119 [+0.0019, +0.0223] | 36 | -0.0183 [-0.0285, -0.0088] | 36 |
| jev_ask_both-jev_ask_plain | -0.0579 [-0.1174, -0.0003] | 36 | +0.0367 [-0.0307, +0.0987] | 36 | -0.1525 [-0.2155, -0.0878] | 36 |
| jev_ask_pacc-jev_ask_prior | -0.0016 [-0.0114, +0.0072] | 36 | -0.0023 [-0.0077, +0.0023] | 36 | -0.0008 [-0.0183, +0.0144] | 36 |
| jev_ask_both-jev_ask_pacc | -0.0075 [-0.0150, -0.0001] | 36 | -0.0048 [-0.0126, +0.0021] | 36 | -0.0101 [-0.0242, +0.0043] | 36 |
| jev_ask_both-jev_ask_belief | -0.0620 [-0.1184, -0.0076] | 36 | +0.0155 [-0.0486, +0.0730] | 36 | -0.1394 [-0.2004, -0.0761] | 36 |
| jev_ask_prior-jev_ask_placebo | -0.0457 [-0.1060, +0.0133] | 36 | +0.0319 [-0.0395, +0.0972] | 36 | -0.1233 [-0.1841, -0.0619] | 36 |

### Paired contrasts in questions asked per episode (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | +1.5722 [+1.5153, +1.6264] | +1.8194 [+1.7389, +1.8944] | +1.3250 [+1.2639, +1.3917] |
| jev_ask_both-jev_noask_both | +0.0139 [+0.0056, +0.0236] | +0.0222 [+0.0083, +0.0389] | +0.0056 [+0.0000, +0.0139] |
| exact_noask-exact_ask | -1.5722 [-1.6264, -1.5153] | -1.8194 [-1.8944, -1.7389] | -1.3250 [-1.3917, -1.2611] |
| random-exact_ask | -0.6903 [-0.7722, -0.6153] | -0.8778 [-0.9806, -0.7833] | -0.5028 [-0.6083, -0.3944] |
| jev_ask_plain-exact_ask | -1.5708 [-1.6250, -1.5139] | -1.8194 [-1.8944, -1.7389] | -1.3222 [-1.3889, -1.2583] |
| jev_ask_belief-exact_ask | -1.5083 [-1.5667, -1.4514] | -1.7278 [-1.8083, -1.6444] | -1.2889 [-1.3556, -1.2222] |
| jev_ask_pacc-exact_ask | -1.5722 [-1.6264, -1.5153] | -1.8194 [-1.8944, -1.7389] | -1.3250 [-1.3917, -1.2611] |
| jev_ask_prior-exact_ask | -1.5708 [-1.6250, -1.5139] | -1.8167 [-1.8917, -1.7361] | -1.3250 [-1.3917, -1.2611] |
| jev_ask_placebo-exact_ask | -1.5708 [-1.6250, -1.5139] | -1.8194 [-1.8944, -1.7389] | -1.3222 [-1.3889, -1.2583] |
| jev_ask_both-exact_ask | -1.5583 [-1.6139, -1.5014] | -1.7972 [-1.8722, -1.7139] | -1.3194 [-1.3861, -1.2583] |
| jev_noask_both-exact_ask | -1.5722 [-1.6264, -1.5153] | -1.8194 [-1.8944, -1.7389] | -1.3250 [-1.3917, -1.2611] |
| jev_ask_belief-jev_ask_plain | +0.0625 [+0.0389, +0.0889] | +0.0917 [+0.0556, +0.1333] | +0.0333 [+0.0139, +0.0528] |
| jev_ask_pacc-jev_ask_plain | -0.0014 [-0.0042, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0028 [-0.0083, +0.0000] |
| jev_ask_prior-jev_ask_plain | +0.0000 [-0.0042, +0.0042] | +0.0028 [+0.0000, +0.0083] | -0.0028 [-0.0083, +0.0000] |
| jev_ask_placebo-jev_ask_plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_both-jev_ask_plain | +0.0125 [+0.0028, +0.0222] | +0.0222 [+0.0083, +0.0389] | +0.0028 [-0.0056, +0.0139] |
| jev_ask_pacc-jev_ask_prior | -0.0014 [-0.0042, +0.0000] | -0.0028 [-0.0083, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_both-jev_ask_pacc | +0.0139 [+0.0056, +0.0236] | +0.0222 [+0.0083, +0.0389] | +0.0056 [+0.0000, +0.0139] |
| jev_ask_both-jev_ask_belief | -0.0500 [-0.0764, -0.0250] | -0.0694 [-0.1139, -0.0278] | -0.0306 [-0.0528, -0.0083] |
| jev_ask_prior-jev_ask_placebo | +0.0000 [-0.0042, +0.0042] | +0.0028 [+0.0000, +0.0083] | -0.0028 [-0.0083, +0.0000] |

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
| exact_ask-exact_noask | all | +0.0598 [+0.0456, +0.0762] | 0.0638 | +0.9370 [+0.7144, +1.1943] | 0.0638 | +0.9370 [+0.7144, +1.1943] |
| exact_ask-exact_noask | focal_opens | +0.0964 [+0.0703, +0.1243] | 0.1019 | +0.9457 [+0.6899, +1.2192] | 0.1019 | +0.9457 [+0.6899, +1.2192] |
| exact_ask-exact_noask | opponent_opens | +0.0232 [+0.0146, +0.0334] | 0.0257 | +0.9028 [+0.5672, +1.2995] | 0.0257 | +0.9028 [+0.5672, +1.2995] |

### Selectivity: P(ask) split by whether asking is strictly optimal in that state

| arm | asked / decisions where ASK optimal | P(ask \| optimal) | asked / decisions where ASK legal but not optimal | P(ask \| not optimal) | difference [CI] |
|---|---|---|---|---|---|
| exact_ask | 1132/1132 | 1.000 | 0/1028 | 0.000 | +1.000 [+1.000, +1.000] |
| random | 349/880 | 0.397 | 286/679 | 0.421 | -0.025 [-0.067, +0.020] |
| jev_ask_plain | 0/881 | 0.000 | 1/406 | 0.002 | -0.002 [-0.008, +0.000] |
| jev_ask_belief | 38/886 | 0.043 | 8/436 | 0.018 | +0.025 [+0.004, +0.045] |
| jev_ask_pacc | 0/582 | 0.000 | 0/360 | 0.000 | +0.000 [+0.000, +0.000] |
| jev_ask_prior | 1/581 | 0.002 | 0/361 | 0.000 | +0.002 [+0.000, +0.005] |
| jev_ask_placebo | 0/881 | 0.000 | 1/405 | 0.002 | -0.002 [-0.008, +0.000] |
| jev_ask_both | 10/595 | 0.017 | 0/358 | 0.000 | +0.017 [+0.007, +0.029] |

### What each arm chose

| arm | decisions by kind | offers by tag | episodes by terminal reason | walk rate | prob. mass on ASK when on the menu | mean confidence | mean choice entropy (bits) |
|---|---|---|---|---|---|---|---|
| exact_ask | {'ASK': 1132, 'PROPOSE': 720, 'REJECT': 308} | {'SAFE': 720} | {'accept': 720} | 0.000 | — | — | — |
| exact_noask | {'PROPOSE': 1066, 'REJECT': 562} | {'PLAN': 360, 'SAFE': 443, 'AMBITIOUS': 263} | {'accept': 654, 'walk': 66} | 0.092 | — | — | — |
| random | {'PROPOSE': 522, 'ASK': 635, 'ACCEPT': 400, 'REJECT': 218} | {'PLAN_ASK': 49, 'SAFE': 249, 'PLAN': 67, 'AMBITIOUS': 157} | {'accept': 712, 'walk': 8} | 0.011 | — | — | — |
| jev_ask_plain | {'PROPOSE': 1279, 'WALK': 196, 'ACCEPT': 20, 'ASK': 1} | {'PLAN_ASK': 360, 'AMBITIOUS': 717, 'SAFE': 199, 'PLAN': 3} | {'walk': 246, 'accept': 474} | 0.342 | 0.180 | 0.595 | 1.568 |
| jev_ask_belief | {'PROPOSE': 1259, 'WALK': 178, 'ACCEPT': 27, 'ASK': 46, 'REJECT': 1} | {'PLAN_ASK': 360, 'AMBITIOUS': 668, 'SAFE': 227, 'PLAN': 4} | {'walk': 225, 'accept': 495} | 0.312 | 0.273 | 0.452 | 1.894 |
| jev_ask_pacc | {'PROPOSE': 926, 'ACCEPT': 16} | {'SAFE': 874, 'AMBITIOUS': 52} | {'accept': 720} | 0.000 | 0.169 | 0.521 | 1.810 |
| jev_ask_prior | {'PROPOSE': 908, 'ACCEPT': 30, 'ASK': 1, 'WALK': 3, 'REJECT': 2} | {'SAFE': 856, 'AMBITIOUS': 51, 'PLAN': 1} | {'accept': 716, 'walk': 4} | 0.006 | 0.200 | 0.459 | 1.994 |
| jev_ask_placebo | {'PROPOSE': 1259, 'WALK': 195, 'ACCEPT': 39, 'ASK': 1} | {'PLAN_ASK': 360, 'AMBITIOUS': 715, 'SAFE': 182, 'PLAN': 2} | {'walk': 246, 'accept': 474} | 0.342 | 0.150 | 0.597 | 1.545 |
| jev_ask_both | {'PROPOSE': 919, 'ACCEPT': 24, 'ASK': 10} | {'SAFE': 852, 'AMBITIOUS': 66, 'PLAN': 1} | {'accept': 720} | 0.000 | 0.220 | 0.454 | 1.984 |
| jev_noask_both | {'PROPOSE': 946, 'ACCEPT': 22} | {'SAFE': 825, 'AMBITIOUS': 119, 'PLAN': 2} | {'accept': 720} | 0.000 | — | 0.509 | 1.437 |

### Jev chosen position (0 = first lettered option)

- jev_ask_plain: {'0': 286, '1': 266, '2': 196, '3': 166, '4': 151, '5': 172, '6': 129, '7': 109, '8': 21} over 1496 decisions; well-formed share 1.000.
- jev_ask_belief: {'0': 280, '1': 280, '2': 160, '3': 188, '4': 172, '5': 178, '6': 130, '7': 103, '8': 20} over 1511 decisions; well-formed share 1.000.
- jev_ask_pacc: {'0': 125, '1': 109, '2': 117, '3': 145, '4': 130, '5': 128, '6': 108, '7': 80} over 942 decisions; well-formed share 1.000.
- jev_ask_prior: {'0': 123, '1': 112, '2': 139, '3': 114, '4': 118, '5': 131, '6': 116, '7': 91} over 944 decisions; well-formed share 1.000.
- jev_ask_placebo: {'0': 286, '1': 281, '2': 158, '3': 167, '4': 179, '5': 171, '6': 139, '7': 92, '8': 21} over 1494 decisions; well-formed share 1.000.
- jev_ask_both: {'0': 108, '1': 122, '2': 113, '3': 128, '4': 127, '5': 151, '6': 110, '7': 94} over 953 decisions; well-formed share 1.000.
- jev_noask_both: {'0': 158, '1': 168, '2': 184, '3': 240, '4': 213, '5': 5} over 968 decisions; well-formed share 1.000.

### Read-out (computed)

- exact_ask − exact_noask (all): +0.0598 [+0.0456, +0.0762] vs exact Δ\*_menu 0.0638 in Y units (0.0383 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (focal_opens): +0.0964 [+0.0703, +0.1243] vs exact Δ\*_menu 0.1019 in Y units (0.0612 utility / 0.6000) — the ceiling is inside the CI.
- exact_ask − exact_noask (opponent_opens): +0.0232 [+0.0146, +0.0334] vs exact Δ\*_menu 0.0257 in Y units (0.0154 utility / 0.6000) — the ceiling is inside the CI.
- The shortlist menu is costless at the root in all 24 cells (every loss < 1e-9).
- Arms asking more often when asking is optimal than when it is not: exact_ask (+1.000), jev_ask_belief (+0.025), jev_ask_prior (+0.002), jev_ask_both (+0.017); not more often: random (-0.025), jev_ask_plain (-0.002), jev_ask_pacc (+0.000), jev_ask_placebo (-0.002).

### Paired contrasts in walk rate (episodes ending with a walk-out; cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| exact_ask-exact_noask | -0.0917 [-0.1486, -0.0431] | -0.1361 [-0.2306, -0.0556] | -0.0472 [-0.0806, -0.0222] |
| jev_ask_both-jev_noask_both | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| exact_noask-exact_ask | +0.0917 [+0.0431, +0.1486] | +0.1361 [+0.0556, +0.2306] | +0.0472 [+0.0222, +0.0806] |
| random-exact_ask | +0.0111 [+0.0042, +0.0194] | +0.0083 [+0.0000, +0.0167] | +0.0139 [+0.0028, +0.0306] |
| jev_ask_plain-exact_ask | +0.3417 [+0.2569, +0.4278] | +0.5417 [+0.4194, +0.6611] | +0.1417 [+0.0861, +0.2028] |
| jev_ask_belief-exact_ask | +0.3125 [+0.2389, +0.3903] | +0.4944 [+0.3861, +0.6000] | +0.1306 [+0.0806, +0.1861] |
| jev_ask_pacc-exact_ask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_prior-exact_ask | +0.0056 [+0.0000, +0.0125] | +0.0000 [+0.0000, +0.0000] | +0.0111 [+0.0000, +0.0250] |
| jev_ask_placebo-exact_ask | +0.3417 [+0.2583, +0.4264] | +0.5389 [+0.4167, +0.6583] | +0.1444 [+0.0889, +0.2056] |
| jev_ask_both-exact_ask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_noask_both-exact_ask | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_belief-jev_ask_plain | -0.0292 [-0.0514, -0.0097] | -0.0472 [-0.0806, -0.0194] | -0.0111 [-0.0361, +0.0083] |
| jev_ask_pacc-jev_ask_plain | -0.3417 [-0.4278, -0.2569] | -0.5417 [-0.6583, -0.4194] | -0.1417 [-0.2028, -0.0861] |
| jev_ask_prior-jev_ask_plain | -0.3361 [-0.4194, -0.2542] | -0.5417 [-0.6583, -0.4194] | -0.1306 [-0.1889, -0.0778] |
| jev_ask_placebo-jev_ask_plain | +0.0000 [-0.0069, +0.0069] | -0.0028 [-0.0167, +0.0083] | +0.0028 [+0.0000, +0.0083] |
| jev_ask_both-jev_ask_plain | -0.3417 [-0.4278, -0.2569] | -0.5417 [-0.6583, -0.4194] | -0.1417 [-0.2028, -0.0861] |
| jev_ask_pacc-jev_ask_prior | -0.0056 [-0.0125, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0111 [-0.0250, +0.0000] |
| jev_ask_both-jev_ask_pacc | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| jev_ask_both-jev_ask_belief | -0.3125 [-0.3903, -0.2375] | -0.4944 [-0.6000, -0.3861] | -0.1306 [-0.1861, -0.0806] |
| jev_ask_prior-jev_ask_placebo | -0.3361 [-0.4181, -0.2542] | -0.5389 [-0.6583, -0.4167] | -0.1333 [-0.1917, -0.0806] |

## Pre-registered rules and verdicts

Materiality δ = 0.02 Y. Primary cells: all. Verdict vocabulary as in `verdicts_live.py`.

| rule | quantity | prediction | estimate [95% CI] | verdict |
|---|---|---|---|---|
| P1 | Δ Y, jev_ask_both-jev_ask_plain, all cells (PRIMARY): belief in words + posterior P(accept) shown | > 0; material if lower CI bound > 0.02 | -0.0579 [-0.1174, -0.0003] | **negative** |
| S1 | Δ Y, jev_ask_belief-jev_ask_plain, all cells (the posterior in words alone) | > 0 | +0.0041 [-0.0065, +0.0138] | **unresolved** |
| S2 | Δ Y, jev_ask_pacc-jev_ask_plain, all cells (posterior P(accept) alone; B_passive selector) | > 0 | -0.0504 [-0.1106, +0.0093] | **unresolved** |
| S3 | Δ Y, jev_ask_pacc-jev_ask_prior, all cells (passive learning in the selector: B_passive − A_prior) | > 0 | -0.0016 [-0.0114, +0.0072] | **not supported** |
| S4 | Δ Y, jev_ask_placebo-jev_ask_plain, all cells (a constant annotation; salience check) | = 0 (CI includes 0) | -0.0032 [-0.0097, +0.0035] | **confirmed** |
| S5 | Δ questions asked / episode, jev_ask_belief-jev_ask_plain, all cells (seeing the uncertainty -> asking) | > 0 | +0.0625 [+0.0389, +0.0889] | **confirmed** |
| S5b | Δ questions asked / episode, jev_ask_both-jev_ask_plain, all cells | > 0 | +0.0125 [+0.0028, +0.0222] | **confirmed** |
| S6 | Δ Y, jev_ask_both-jev_noask_both, focal-opens cells (the value of having questions on the menu once the belief is shown) | > 0 | -0.0020 [-0.0134, +0.0098] | **unresolved** |
| S7 | Δ Y, jev_ask_both-exact_ask, all cells (the remaining gap to the planner) | < 0 | -0.2320 [-0.2674, -0.1956] | **confirmed** |
| S8 | Δ walk rate, jev_ask_pacc-jev_ask_plain, all cells (acceptance numbers -> fewer walk-outs) | < 0 | -0.3417 [-0.4278, -0.2569] | **confirmed** |
| S8b | Δ walk rate, jev_ask_both-jev_ask_plain, all cells | < 0 | -0.3417 [-0.4278, -0.2569] | **confirmed** |
| S9_jev_ask_belief | P(ask | ASK strictly optimal) − P(ask | ASK legal, not optimal), jev_ask_belief | > 0 | +0.0245 [+0.0043, +0.0452] | **confirmed** |
| S9_jev_ask_both | P(ask | ASK strictly optimal) − P(ask | ASK legal, not optimal), jev_ask_both | > 0 | +0.0168 [+0.0067, +0.0286] | **confirmed** |

## Read-out (computed)

- P1 `jev_ask_both-jev_ask_plain` (all cells): -0.0579 [-0.1174, -0.0003] → **negative**.
- Gap to the exact planner: plain -0.1741 [-0.2039, -0.1438], both -0.2320 [-0.2674, -0.1956]; the shown information closes -33.3% of the plain gap (point estimate; no interval on the ratio).
- Mean Y, all cells: exact_ask 0.6319, exact_noask 0.5722, random -0.0063, jev_ask_plain 0.4578, jev_ask_belief 0.4619, jev_ask_pacc 0.4074, jev_ask_prior 0.4090, jev_ask_placebo 0.4546, jev_ask_both 0.3999, jev_noask_both 0.3978.
- Walk rate: exact_ask 0.000, exact_noask 0.092, random 0.011, jev_ask_plain 0.342, jev_ask_belief 0.312, jev_ask_pacc 0.000, jev_ask_prior 0.006, jev_ask_placebo 0.342, jev_ask_both 0.000, jev_noask_both 0.000.
- Questions per episode: jev_ask_plain 0.001, jev_ask_belief 0.064, jev_ask_pacc 0.000, jev_ask_prior 0.001, jev_ask_placebo 0.001, jev_ask_both 0.014.
- jev_ask_prior − jev_ask_placebo (all cells, descriptive): -0.0457 [-0.1060, +0.0133].
- jev_ask_both − jev_ask_pacc (all cells, descriptive): -0.0075 [-0.0150, -0.0001].
- jev_ask_both − jev_ask_belief (all cells, descriptive): -0.0620 [-0.1184, -0.0076].

## Regenerate

```bash
bash results/live_sel_jev_h4_cq0_20261030/REGENERATE.sh
```
