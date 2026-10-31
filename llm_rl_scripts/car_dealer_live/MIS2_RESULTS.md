# Live Car-Dealer, CD-MIS-2 — severity sweep: how wrong can the opponent model be, and the evasive-answer channel

_Rendered by `render_mis2_note.py` from `analysis.json` written by `analyze_mis.py`; every number is formatted from JSON, none is typed._

No model is involved: every arm is scripted and every number is exact-on-model or Monte Carlo over the frozen seeds. The agent's ASSUMED behavioural kernel, planner, question bank and costs are fixed; the TRUE process, its severity, and the agent's answer model (the `C_ans` arms assume a flip rate) vary.

Source: `results/live_mis2_20261030` — env `live-car-dealer-v0.1`, design `CD-MIS-2`, regime easy (r = 2/5), H = 4, c_q = 0; assumed kernel alpha0=4/5, beta=8.0, p_counter=1.0, p_hold=1.0, shortlist=3, accept_noise=0.0; processes ['evasive@0', 'evasive@0.1', 'evasive@0.2', 'evasive@0.3', 'evasive@0.4', 'evasive@0.5', 'history@0.1', 'history@0.2', 'history@0.4', 'stochastic@0.15', 'stochastic@0.3', 'stochastic@0.5']; arms ['A_prior', 'B_naive', 'C_naive', 'C_mix0.2', 'C_ans0.1', 'C_ans0.2', 'C_ans0.3', 'O_true', 'B_mix0.2']; 56160 episodes (5 repeats per cell, seed 41), 0 error stubs, 6723.49 s. Clusters = (buyer ranking, seller ranking) type pair; 36 clusters; bootstrap B = 5000 (seed 41).

Hedged-answer arms (planner and filter both assume this flip rate; behavioural kernel = the registered T1): `C_ans0.1` → 1/10, `C_ans0.2` → 1/5, `C_ans0.3` → 3/10. `O_true` plans with the true process and the exact filter under it.

Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r); no deal gives Y = 0. 'Identical' = share of seed-paired episodes in which two arms made the same decision sequence and reached the same terminal.

## The evasive channel: true flip rate × assumed flip rate

Mean Y (all cells) by true flip rate (rows) and arm (columns). `C_naive` assumes truthful answers (assumed 0); `C_ans a` assumes a; `O_true` assumes the true rate (the diagonal); `B_naive` never asks.

| true flip rate | `B_naive` | `C_naive` | `C_ans0.1` | `C_ans0.2` | `C_ans0.3` | `O_true` |
|---|---|---|---|---|---|---|
| 0.00 (`evasive@0`) | 0.5648 | 0.6320 | 0.6229 | 0.6065 | 0.5762 | 0.6320 |
| 0.10 (`evasive@0.1`) | 0.5648 | 0.5736 | 0.6000 | 0.5908 | 0.5723 | 0.6000 |
| 0.20 (`evasive@0.2`) | 0.5648 | 0.5176 | 0.5752 | 0.5782 | 0.5658 | 0.5782 |
| 0.30 (`evasive@0.3`) | 0.5648 | 0.4780 | 0.5512 | 0.5619 | 0.5631 | 0.5631 |
| 0.40 (`evasive@0.4`) | 0.5648 | 0.4311 | 0.5277 | 0.5480 | 0.5597 | 0.5648 |
| 0.50 (`evasive@0.5`) | 0.5648 | 0.3841 | 0.4980 | 0.5313 | 0.5561 | 0.5648 |

Questions asked per episode, same layout:

| true flip rate | `B_naive` | `C_naive` | `C_ans0.1` | `C_ans0.2` | `C_ans0.3` | `O_true` |
|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 2.000 | 1.026 | 0.619 | 0.094 | 1.571 |
| 0.10 | 0.000 | 2.000 | 1.012 | 0.610 | 0.094 | 1.012 |
| 0.20 | 0.000 | 2.000 | 0.999 | 0.608 | 0.094 | 0.608 |
| 0.30 | 0.000 | 2.000 | 0.996 | 0.607 | 0.094 | 0.094 |
| 0.40 | 0.000 | 2.000 | 0.988 | 0.614 | 0.094 | 0.037 |
| 0.50 | 0.000 | 2.000 | 0.983 | 0.607 | 0.094 | 0.037 |

### Dose-response contrasts along the true flip rate (paired cluster bootstrap, 95% CI, all cells)

| true flip rate | `C_naive-B_naive` | `C_ans0.1-C_naive` | `C_ans0.2-C_naive` | `C_ans0.3-C_naive` | `O_true-B_naive` | `O_true-C_naive` | `O_true-C_ans0.2` |
|---|---|---|---|---|---|---|---|
| 0.00 | +0.0673 [+0.0522, +0.0841] | -0.0091 [-0.0128, -0.0057] | -0.0256 [-0.0344, -0.0175] | -0.0558 [-0.0651, -0.0467] | +0.0673 [+0.0522, +0.0841] | +0.0000 [+0.0000, +0.0000] | +0.0256 [+0.0175, +0.0344] |
| 0.10 | +0.0089 [-0.0110, +0.0311] | +0.0263 [+0.0156, +0.0382] | +0.0171 [+0.0013, +0.0328] | -0.0013 [-0.0191, +0.0151] | +0.0352 [+0.0176, +0.0544] | +0.0263 [+0.0156, +0.0382] | +0.0092 [-0.0002, +0.0189] |
| 0.20 | -0.0472 [-0.0713, -0.0217] | +0.0576 [+0.0416, +0.0742] | +0.0606 [+0.0372, +0.0841] | +0.0482 [+0.0257, +0.0692] | +0.0134 [-0.0048, +0.0327] | +0.0606 [+0.0372, +0.0841] | +0.0000 [+0.0000, +0.0000] |
| 0.30 | -0.0868 [-0.1089, -0.0626] | +0.0732 [+0.0545, +0.0923] | +0.0839 [+0.0625, +0.1059] | +0.0851 [+0.0637, +0.1050] | -0.0016 [-0.0094, +0.0066] | +0.0851 [+0.0637, +0.1050] | +0.0012 [-0.0161, +0.0183] |
| 0.40 | -0.1337 [-0.1585, -0.1064] | +0.0966 [+0.0760, +0.1176] | +0.1170 [+0.0936, +0.1411] | +0.1286 [+0.1046, +0.1509] | +0.0000 [+0.0000, +0.0000] | +0.1337 [+0.1064, +0.1586] | +0.0167 [-0.0006, +0.0332] |
| 0.50 | -0.1807 [-0.2124, -0.1468] | +0.1139 [+0.0896, +0.1391] | +0.1472 [+0.1182, +0.1765] | +0.1720 [+0.1405, +0.2019] | +0.0000 [+0.0000, +0.0000] | +0.1807 [+0.1469, +0.2125] | +0.0334 [+0.0147, +0.0512] |

Identical-trajectory share, same layout:

| true flip rate | `C_naive-B_naive` | `C_ans0.1-C_naive` | `C_ans0.2-C_naive` | `C_ans0.3-C_naive` | `O_true-B_naive` | `O_true-C_naive` | `O_true-C_ans0.2` |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.217 | 0.025 | 0.000 | 0.124 | 0.418 | 0.131 |
| 0.10 | 0.000 | 0.201 | 0.025 | 0.000 | 0.354 | 0.237 | 0.354 |
| 0.20 | 0.000 | 0.188 | 0.026 | 0.000 | 0.475 | 0.014 | 0.935 |
| 0.30 | 0.000 | 0.182 | 0.024 | 0.000 | 0.868 | 0.000 | 0.475 |
| 0.40 | 0.000 | 0.175 | 0.025 | 0.000 | 0.925 | 0.000 | 0.475 |
| 0.50 | 0.000 | 0.171 | 0.021 | 0.000 | 0.925 | 0.000 | 0.475 |

## Behavioural severity: `history`

Mean Y (all cells) by severity level and arm:

| level | `A_prior` | `B_mix0.2` | `B_naive` | `C_naive` | `O_true` |
|---|---|---|---|---|---|
| 0.10 (`history@0.1`) | 0.4386 | 0.5647 | 0.5647 | 0.6316 | 0.6316 |
| 0.20 (`history@0.2`) | 0.4375 | 0.5635 | 0.5635 | 0.6316 | 0.6343 |
| 0.40 (`history@0.4`) | 0.4394 | 0.5618 | 0.5618 | 0.6287 | 0.6287 |

| level | `B_naive-A_prior` | `B_mix0.2-B_naive` | `C_naive-B_naive` | `O_true-C_naive` |
|---|---|---|---|---|
| 0.10 | +0.1262 [+0.0981, +0.1546] | +0.0000 [+0.0000, +0.0000] | +0.0668 [+0.0518, +0.0833] | +0.0000 [+0.0000, +0.0000] |
| 0.20 | +0.1260 [+0.0983, +0.1540] | +0.0000 [+0.0000, +0.0000] | +0.0681 [+0.0526, +0.0850] | +0.0028 [+0.0012, +0.0047] |
| 0.40 | +0.1224 [+0.0949, +0.1500] | +0.0000 [+0.0000, +0.0000] | +0.0669 [+0.0514, +0.0837] | -0.0000 [-0.0022, +0.0022] |

Identical-trajectory share, same layout:

| level | `B_naive-A_prior` | `B_mix0.2-B_naive` | `C_naive-B_naive` | `O_true-C_naive` |
|---|---|---|---|---|
| 0.10 | 0.339 | 1.000 | 0.000 | 0.418 |
| 0.20 | 0.333 | 1.000 | 0.000 | 0.217 |
| 0.40 | 0.339 | 1.000 | 0.000 | 0.407 |

## Behavioural severity: `stochastic`

Mean Y (all cells) by severity level and arm:

| level | `A_prior` | `B_mix0.2` | `B_naive` | `C_naive` | `O_true` |
|---|---|---|---|---|---|
| 0.15 (`stochastic@0.15`) | 0.4430 | 0.5067 | 0.5067 | 0.6034 | 0.6333 |
| 0.30 (`stochastic@0.3`) | 0.4385 | 0.4925 | 0.4925 | 0.6034 | 0.6333 |
| 0.50 (`stochastic@0.5`) | 0.4359 | 0.4783 | 0.4783 | 0.6034 | 0.6333 |

| level | `B_naive-A_prior` | `B_mix0.2-B_naive` | `C_naive-B_naive` | `O_true-C_naive` |
|---|---|---|---|---|
| 0.15 | +0.0637 [+0.0309, +0.0939] | +0.0000 [+0.0000, +0.0000] | +0.0967 [+0.0761, +0.1183] | +0.0298 [+0.0180, +0.0433] |
| 0.30 | +0.0540 [+0.0173, +0.0885] | +0.0000 [+0.0000, +0.0000] | +0.1110 [+0.0916, +0.1320] | +0.0298 [+0.0180, +0.0433] |
| 0.50 | +0.0425 [+0.0044, +0.0787] | +0.0000 [+0.0000, +0.0000] | +0.1251 [+0.1047, +0.1475] | +0.0298 [+0.0180, +0.0433] |

Identical-trajectory share, same layout:

| level | `B_naive-A_prior` | `B_mix0.2-B_naive` | `C_naive-B_naive` | `O_true-C_naive` |
|---|---|---|---|---|
| 0.15 | 0.421 | 1.000 | 0.000 | 0.019 |
| 0.30 | 0.394 | 1.000 | 0.000 | 0.019 |
| 0.50 | 0.372 | 1.000 | 0.000 | 0.019 |

## Per process, per arm (all cells)

| process | arm | n | mean Y [95% CI] | agreement | walk rate | asks / episode | decision regret (true) | final p(true type) | final log-loss (bits) | argmax = true | predictive log-loss (bits) | Brier(accept) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `evasive@0` | A_prior | 720 | 0.4370 [0.3326, 0.5337] | 0.667 | 0.333 | 0.000 | 0.1153 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `evasive@0` | B_naive | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.000 | 0.0379 | 0.816 | 0.373 | 0.836 | 1.795 | 0.179 |
| `evasive@0` | C_naive | 720 | 0.6320 [0.5564, 0.7061] | 1.000 | 0.000 | 2.000 | 0.0000 | 0.853 | 0.293 | 0.854 | 0.705 | 0.000 |
| `evasive@0` | C_mix0.2 | 720 | 0.6284 [0.5536, 0.7019] | 1.000 | 0.000 | 2.000 | 0.0025 | 0.787 | 0.414 | 0.846 | 0.902 | 0.007 |
| `evasive@0` | C_ans0.1 | 720 | 0.6229 [0.5468, 0.6984] | 0.985 | 0.015 | 1.026 | 0.0043 | 0.831 | 0.339 | 0.871 | 1.288 | 0.046 |
| `evasive@0` | C_ans0.2 | 720 | 0.6065 [0.5321, 0.6784] | 0.976 | 0.024 | 0.619 | 0.0137 | 0.806 | 0.413 | 0.846 | 1.540 | 0.071 |
| `evasive@0` | C_ans0.3 | 720 | 0.5762 [0.4957, 0.6552] | 0.926 | 0.074 | 0.094 | 0.0297 | 0.802 | 0.397 | 0.836 | 1.719 | 0.173 |
| `evasive@0` | O_true | 720 | 0.6320 [0.5564, 0.7061] | 1.000 | 0.000 | 1.571 | 0.0000 | 0.853 | 0.293 | 0.854 | 0.972 | 0.024 |
| `evasive@0.1` | A_prior | 720 | 0.4370 [0.3326, 0.5337] | 0.667 | 0.333 | 0.000 | 0.0952 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `evasive@0.1` | B_naive | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.000 | 0.0178 | 0.816 | 0.373 | 0.836 | 1.795 | 0.179 |
| `evasive@0.1` | C_naive | 720 | 0.5736 [0.4970, 0.6484] | 0.912 | 0.087 | 2.000 | 0.0132 | 0.780 | 0.915 | 0.772 | 1.022 | 0.077 |
| `evasive@0.1` | C_mix0.2 | 720 | 0.5753 [0.5003, 0.6501] | 0.921 | 0.079 | 2.000 | 0.0121 | 0.712 | 0.685 | 0.760 | 0.977 | 0.063 |
| `evasive@0.1` | C_ans0.1 | 720 | 0.6000 [0.5201, 0.6774] | 0.944 | 0.056 | 1.012 | 0.0000 | 0.814 | 0.413 | 0.847 | 1.350 | 0.072 |
| `evasive@0.1` | C_ans0.2 | 720 | 0.5908 [0.5122, 0.6659] | 0.943 | 0.057 | 0.610 | 0.0013 | 0.803 | 0.428 | 0.839 | 1.573 | 0.091 |
| `evasive@0.1` | C_ans0.3 | 720 | 0.5723 [0.4907, 0.6521] | 0.919 | 0.081 | 0.094 | 0.0121 | 0.802 | 0.398 | 0.833 | 1.722 | 0.174 |
| `evasive@0.1` | O_true | 720 | 0.6000 [0.5201, 0.6774] | 0.944 | 0.056 | 1.012 | 0.0000 | 0.813 | 0.575 | 0.839 | 1.452 | 0.075 |
| `evasive@0.2` | A_prior | 720 | 0.4370 [0.3326, 0.5337] | 0.667 | 0.333 | 0.000 | 0.0857 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `evasive@0.2` | B_naive | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.000 | 0.0082 | 0.816 | 0.373 | 0.836 | 1.795 | 0.179 |
| `evasive@0.2` | C_naive | 720 | 0.5176 [0.4436, 0.5926] | 0.825 | 0.175 | 2.000 | 0.0355 | 0.692 | 1.793 | 0.692 | 1.295 | 0.165 |
| `evasive@0.2` | C_mix0.2 | 720 | 0.5261 [0.4503, 0.6013] | 0.839 | 0.161 | 2.001 | 0.0310 | 0.638 | 0.992 | 0.682 | 1.046 | 0.128 |
| `evasive@0.2` | C_ans0.1 | 720 | 0.5752 [0.4931, 0.6544] | 0.897 | 0.103 | 0.999 | 0.0062 | 0.807 | 0.453 | 0.842 | 1.415 | 0.110 |
| `evasive@0.2` | C_ans0.2 | 720 | 0.5782 [0.4958, 0.6570] | 0.915 | 0.085 | 0.608 | 0.0000 | 0.801 | 0.441 | 0.833 | 1.601 | 0.110 |
| `evasive@0.2` | C_ans0.3 | 720 | 0.5658 [0.4836, 0.6462] | 0.911 | 0.089 | 0.094 | 0.0050 | 0.800 | 0.406 | 0.826 | 1.725 | 0.176 |
| `evasive@0.2` | O_true | 720 | 0.5782 [0.4958, 0.6570] | 0.915 | 0.085 | 0.608 | 0.0000 | 0.783 | 0.616 | 0.811 | 1.786 | 0.123 |
| `evasive@0.3` | A_prior | 720 | 0.4370 [0.3326, 0.5337] | 0.667 | 0.333 | 0.000 | 0.0782 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `evasive@0.3` | B_naive | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.000 | 0.0008 | 0.816 | 0.373 | 0.836 | 1.795 | 0.179 |
| `evasive@0.3` | C_naive | 720 | 0.4780 [0.4012, 0.5547] | 0.757 | 0.243 | 2.000 | 0.0605 | 0.625 | 2.520 | 0.621 | 1.526 | 0.223 |
| `evasive@0.3` | C_mix0.2 | 720 | 0.4880 [0.4110, 0.5657] | 0.775 | 0.225 | 2.004 | 0.0529 | 0.578 | 1.252 | 0.615 | 1.104 | 0.173 |
| `evasive@0.3` | C_ans0.1 | 720 | 0.5512 [0.4684, 0.6297] | 0.856 | 0.144 | 0.996 | 0.0150 | 0.790 | 0.524 | 0.824 | 1.465 | 0.142 |
| `evasive@0.3` | C_ans0.2 | 720 | 0.5619 [0.4793, 0.6410] | 0.886 | 0.114 | 0.607 | 0.0013 | 0.795 | 0.464 | 0.826 | 1.628 | 0.125 |
| `evasive@0.3` | C_ans0.3 | 720 | 0.5631 [0.4814, 0.6440] | 0.910 | 0.090 | 0.094 | 0.0000 | 0.798 | 0.412 | 0.822 | 1.725 | 0.176 |
| `evasive@0.3` | O_true | 720 | 0.5631 [0.4814, 0.6440] | 0.910 | 0.090 | 0.094 | 0.0000 | 0.793 | 0.545 | 0.815 | 1.764 | 0.179 |
| `evasive@0.4` | A_prior | 720 | 0.4370 [0.3326, 0.5337] | 0.667 | 0.333 | 0.000 | 0.0774 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `evasive@0.4` | B_naive | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.000 | 0.0000 | 0.816 | 0.373 | 0.836 | 1.795 | 0.179 |
| `evasive@0.4` | C_naive | 720 | 0.4311 [0.3559, 0.5047] | 0.692 | 0.308 | 2.000 | 0.0889 | 0.544 | 3.565 | 0.553 | 1.708 | 0.286 |
| `evasive@0.4` | C_mix0.2 | 720 | 0.4482 [0.3710, 0.5257] | 0.715 | 0.285 | 2.008 | 0.0782 | 0.516 | 1.566 | 0.547 | 1.154 | 0.225 |
| `evasive@0.4` | C_ans0.1 | 720 | 0.5277 [0.4442, 0.6087] | 0.812 | 0.188 | 0.988 | 0.0297 | 0.778 | 0.585 | 0.811 | 1.510 | 0.173 |
| `evasive@0.4` | C_ans0.2 | 720 | 0.5480 [0.4652, 0.6272] | 0.863 | 0.138 | 0.614 | 0.0097 | 0.794 | 0.471 | 0.824 | 1.647 | 0.140 |
| `evasive@0.4` | C_ans0.3 | 720 | 0.5597 [0.4779, 0.6411] | 0.904 | 0.096 | 0.094 | 0.0017 | 0.797 | 0.414 | 0.819 | 1.727 | 0.177 |
| `evasive@0.4` | O_true | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.037 | 0.0000 | 0.813 | 0.378 | 0.836 | 1.810 | 0.180 |
| `evasive@0.5` | A_prior | 720 | 0.4370 [0.3326, 0.5337] | 0.667 | 0.333 | 0.000 | 0.0774 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `evasive@0.5` | B_naive | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.000 | 0.0000 | 0.816 | 0.373 | 0.836 | 1.795 | 0.179 |
| `evasive@0.5` | C_naive | 720 | 0.3841 [0.3119, 0.4554] | 0.628 | 0.372 | 2.000 | 0.1184 | 0.470 | 4.636 | 0.476 | 1.886 | 0.348 |
| `evasive@0.5` | C_mix0.2 | 720 | 0.4082 [0.3314, 0.4856] | 0.658 | 0.342 | 2.014 | 0.1030 | 0.455 | 1.904 | 0.478 | 1.201 | 0.281 |
| `evasive@0.5` | C_ans0.1 | 720 | 0.4980 [0.4140, 0.5790] | 0.771 | 0.229 | 0.983 | 0.0452 | 0.761 | 0.671 | 0.789 | 1.557 | 0.204 |
| `evasive@0.5` | C_ans0.2 | 720 | 0.5313 [0.4459, 0.6134] | 0.833 | 0.167 | 0.607 | 0.0186 | 0.788 | 0.491 | 0.815 | 1.673 | 0.156 |
| `evasive@0.5` | C_ans0.3 | 720 | 0.5561 [0.4739, 0.6372] | 0.900 | 0.100 | 0.094 | 0.0038 | 0.796 | 0.419 | 0.815 | 1.728 | 0.178 |
| `evasive@0.5` | O_true | 720 | 0.5648 [0.4797, 0.6484] | 0.894 | 0.106 | 0.037 | 0.0000 | 0.812 | 0.379 | 0.836 | 1.817 | 0.180 |
| `history@0.1` | A_prior | 720 | 0.4386 [0.3323, 0.5357] | 0.667 | 0.333 | 0.000 | 0.1151 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `history@0.1` | B_naive | 720 | 0.5647 [0.4788, 0.6494] | 0.892 | 0.108 | 0.000 | 0.0384 | 0.813 | 0.378 | 0.826 | 1.797 | 0.176 |
| `history@0.1` | C_naive | 720 | 0.6316 [0.5566, 0.7055] | 1.000 | 0.000 | 2.000 | 0.0000 | 0.849 | 0.302 | 0.850 | 0.704 | 0.000 |
| `history@0.1` | O_true | 720 | 0.6316 [0.5566, 0.7055] | 1.000 | 0.000 | 1.561 | 0.0000 | 0.849 | 0.302 | 0.850 | 0.980 | 0.023 |
| `history@0.1` | B_mix0.2 | 720 | 0.5647 [0.4788, 0.6494] | 0.892 | 0.108 | 0.000 | 0.0384 | 0.775 | 0.442 | 0.826 | 2.041 | 0.178 |
| `history@0.2` | A_prior | 720 | 0.4375 [0.3317, 0.5347] | 0.667 | 0.333 | 0.000 | 0.1189 | 0.167 | 2.585 | 0.167 | — | 0.271 |
| `history@0.2` | B_naive | 720 | 0.5635 [0.4761, 0.6488] | 0.887 | 0.113 | 0.000 | 0.0424 | 0.819 | 0.361 | 0.854 | 1.792 | 0.180 |
| `history@0.2` | C_naive | 720 | 0.6316 [0.5566, 0.7055] | 1.000 | 0.000 | 2.000 | 0.0015 | 0.849 | 0.302 | 0.850 | 0.704 | 0.000 |
| `history@0.2` | O_true | 720 | 0.6343 [0.5597, 0.7076] | 1.000 | 0.000 | 1.232 | 0.0000 | 0.898 | 0.198 | 0.924 | 1.107 | 0.029 |
| `history@0.2` | B_mix0.2 | 720 | 0.5635 [0.4761, 0.6488] | 0.887 | 0.113 | 0.000 | 0.0424 | 0.780 | 0.426 | 0.854 | 2.036 | 0.181 |
| `history@0.4` | A_prior | 720 | 0.4394 [0.3333, 0.5368] | 0.667 | 0.333 | 0.000 | 0.1131 | 0.167 | 2.585 | 0.167 | — | 0.270 |
| `history@0.4` | B_naive | 720 | 0.5618 [0.4739, 0.6474] | 0.883 | 0.117 | 0.000 | 0.0387 | 0.813 | 0.378 | 0.826 | 1.767 | 0.179 |
| `history@0.4` | C_naive | 720 | 0.6287 [0.5521, 0.7034] | 1.000 | 0.000 | 2.000 | 0.0000 | 0.755 | 0.491 | 0.760 | 0.629 | 0.000 |
| `history@0.4` | O_true | 720 | 0.6287 [0.5532, 0.7029] | 1.000 | 0.000 | 1.639 | 0.0000 | 0.755 | 0.491 | 0.760 | 0.820 | 0.018 |
| `history@0.4` | B_mix0.2 | 720 | 0.5618 [0.4739, 0.6474] | 0.883 | 0.117 | 0.000 | 0.0387 | 0.772 | 0.448 | 0.826 | 2.009 | 0.180 |
| `stochastic@0.15` | A_prior | 720 | 0.4430 [0.3361, 0.5413] | 0.667 | 0.333 | 0.000 | 0.1136 | 0.167 | 2.585 | 0.167 | — | 0.258 |
| `stochastic@0.15` | B_naive | 720 | 0.5067 [0.4164, 0.5938] | 0.790 | 0.210 | 0.000 | 0.0790 | 0.603 | 2.475 | 0.592 | 3.918 | 0.219 |
| `stochastic@0.15` | C_naive | 720 | 0.6034 [0.5226, 0.6828] | 0.942 | 0.058 | 2.000 | 0.0161 | 0.702 | 2.000 | 0.696 | 1.310 | 0.054 |
| `stochastic@0.15` | O_true | 720 | 0.6333 [0.5584, 0.7069] | 1.000 | 0.000 | 1.425 | 0.0000 | 0.728 | 1.552 | 0.714 | 2.428 | 0.082 |
| `stochastic@0.15` | B_mix0.2 | 720 | 0.5067 [0.4164, 0.5938] | 0.790 | 0.210 | 0.000 | 0.0790 | 0.571 | 1.483 | 0.588 | 2.892 | 0.218 |
| `stochastic@0.3` | A_prior | 720 | 0.4385 [0.3315, 0.5335] | 0.667 | 0.333 | 0.000 | 0.1167 | 0.167 | 2.585 | 0.167 | — | 0.243 |
| `stochastic@0.3` | B_naive | 720 | 0.4925 [0.4059, 0.5775] | 0.769 | 0.231 | 0.000 | 0.0891 | 0.585 | 2.643 | 0.564 | 4.131 | 0.216 |
| `stochastic@0.3` | C_naive | 720 | 0.6034 [0.5226, 0.6828] | 0.942 | 0.058 | 2.000 | 0.0161 | 0.702 | 2.000 | 0.696 | 1.310 | 0.054 |
| `stochastic@0.3` | O_true | 720 | 0.6333 [0.5584, 0.7069] | 1.000 | 0.000 | 1.425 | 0.0000 | 0.729 | 1.561 | 0.714 | 2.420 | 0.079 |
| `stochastic@0.3` | B_mix0.2 | 720 | 0.4925 [0.4059, 0.5775] | 0.769 | 0.231 | 0.000 | 0.0891 | 0.553 | 1.595 | 0.560 | 2.963 | 0.214 |
| `stochastic@0.5` | A_prior | 720 | 0.4359 [0.3298, 0.5308] | 0.667 | 0.333 | 0.000 | 0.1177 | 0.167 | 2.585 | 0.167 | — | 0.228 |
| `stochastic@0.5` | B_naive | 720 | 0.4783 [0.3960, 0.5605] | 0.754 | 0.246 | 0.000 | 0.1056 | 0.572 | 2.774 | 0.549 | 4.394 | 0.212 |
| `stochastic@0.5` | C_naive | 720 | 0.6034 [0.5226, 0.6828] | 0.942 | 0.058 | 2.000 | 0.0161 | 0.702 | 2.000 | 0.696 | 1.310 | 0.054 |
| `stochastic@0.5` | O_true | 720 | 0.6333 [0.5584, 0.7069] | 1.000 | 0.000 | 1.425 | 0.0000 | 0.729 | 1.561 | 0.714 | 2.420 | 0.079 |
| `stochastic@0.5` | B_mix0.2 | 720 | 0.4783 [0.3960, 0.5605] | 0.754 | 0.246 | 0.000 | 0.1056 | 0.539 | 1.685 | 0.544 | 3.062 | 0.209 |

## Pre-registered verdicts — `results/live_mis2_20261030/verdicts.json`

Confirmatory: `C_ans0.2-C_naive` under `evasive@0.3`. Sanity gates pass: **True**.

| rule | process | quantity | prediction | estimate [95% CI] | identical | verdict |
|---|---|---|---|---|---|---|
| E1 | `evasive@0.3` | Δ Y, C_ans0.2-C_naive, all cells | > 0 | +0.0839 [+0.0625, +0.1059] (n = 36) | 0.024 | **confirmed** |
| E2 | `evasive@0` | Δ Y, C_ans0.2-C_naive, all cells | <= 0 | -0.0256 [-0.0344, -0.0175] (n = 36) | 0.025 | **confirmed** |
| E3a | `evasive@0` | Δ Y, C_naive-B_naive, all cells | > 0 | +0.0673 [+0.0522, +0.0841] (n = 36) | 0.000 | **confirmed** |
| E3b | `evasive@0.5` | Δ Y, C_naive-B_naive, all cells | < 0 | -0.1807 [-0.2124, -0.1468] (n = 36) | 0.000 | **confirmed** |
| E4 | `evasive@0.1` | Δ Y, O_true-B_naive, all cells | > 0 | +0.0352 [+0.0176, +0.0544] (n = 36) | 0.354 | **confirmed** |
| E5 | `evasive@0.5` | Δ Y, O_true-B_naive, all cells | = 0 | +0.0000 [+0.0000, +0.0000] (n = 36) | 0.925 | **confirmed** |
| E6 | `evasive@0.3` | Δ Y, C_ans0.2-B_naive, all cells | >= 0 | -0.0028 [-0.0207, +0.0158] (n = 36) | 0.475 | **confirmed** |
| S1 | `history@0.4` | Δ Y, B_naive-A_prior, all cells | >= 0 | +0.1224 [+0.0949, +0.1500] (n = 36) | 0.339 | **confirmed** |
| S2 | `stochastic@0.5` | Δ Y, B_naive-A_prior, all cells | >= 0 | +0.0425 [+0.0044, +0.0787] (n = 36) | 0.372 | **confirmed** |
| S3a | `history@0.4` | Δ Y, B_mix0.2-B_naive, all cells | = 0 | +0.0000 [+0.0000, +0.0000] (n = 36) | 1.000 | **confirmed** |
| S3b | `stochastic@0.5` | Δ Y, B_mix0.2-B_naive, all cells | = 0 | +0.0000 [+0.0000, +0.0000] (n = 36) | 1.000 | **confirmed** |
| S4a | `history@0.4` | Δ Y, O_true-C_naive, all cells | = 0 | -0.0000 [-0.0022, +0.0022] (n = 36) | 0.407 | **confirmed** |
| S4b | `stochastic@0.5` | Δ Y, O_true-C_naive, all cells | > 0 | +0.0298 [+0.0180, +0.0433] (n = 36) | 0.019 | **confirmed** |
| G_evasive@0 | `evasive@0` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1278 [+0.0985, +0.1570] (n = 36) | 0.324 | **opposite** |
| G_evasive@0.1 | `evasive@0.1` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1278 [+0.0985, +0.1570] (n = 36) | 0.324 | **opposite** |
| G_evasive@0.2 | `evasive@0.2` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1278 [+0.0985, +0.1570] (n = 36) | 0.324 | **opposite** |
| G_evasive@0.3 | `evasive@0.3` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1278 [+0.0985, +0.1570] (n = 36) | 0.324 | **opposite** |
| G_evasive@0.4 | `evasive@0.4` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1278 [+0.0985, +0.1570] (n = 36) | 0.324 | **opposite** |
| G_evasive@0.5 | `evasive@0.5` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1278 [+0.0985, +0.1570] (n = 36) | 0.324 | **opposite** |
| G_history@0.1 | `history@0.1` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1262 [+0.0981, +0.1546] (n = 36) | 0.339 | **opposite** |
| G_history@0.2 | `history@0.2` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1260 [+0.0983, +0.1540] (n = 36) | 0.333 | **opposite** |
| G_history@0.4 | `history@0.4` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.1224 [+0.0949, +0.1500] (n = 36) | 0.339 | **opposite** |
| G_stochastic@0.15 | `stochastic@0.15` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.0637 [+0.0309, +0.0939] (n = 36) | 0.421 | **opposite** |
| G_stochastic@0.3 | `stochastic@0.3` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.0540 [+0.0173, +0.0885] (n = 36) | 0.394 | **opposite** |
| G_stochastic@0.5 | `stochastic@0.5` | Δ Y, B_naive-A_prior, all cells | < 0 | +0.0425 [+0.0044, +0.0787] (n = 36) | 0.372 | **opposite** |
| D_evasive@0 | `evasive@0` | Δ Y, C_naive-B_naive, all cells | > 0 | +0.0673 [+0.0522, +0.0841] (n = 36) | 0.000 | **confirmed** |
| D_evasive@0.1 | `evasive@0.1` | Δ Y, C_naive-B_naive, all cells | > 0 | +0.0089 [-0.0110, +0.0311] (n = 36) | 0.000 | **not supported** |
| D_evasive@0.2 | `evasive@0.2` | Δ Y, C_naive-B_naive, all cells | > 0 | -0.0472 [-0.0713, -0.0217] (n = 36) | 0.000 | **opposite** |
| D_evasive@0.3 | `evasive@0.3` | Δ Y, C_naive-B_naive, all cells | > 0 | -0.0868 [-0.1089, -0.0626] (n = 36) | 0.000 | **opposite** |
| D_evasive@0.4 | `evasive@0.4` | Δ Y, C_naive-B_naive, all cells | > 0 | -0.1337 [-0.1585, -0.1064] (n = 36) | 0.000 | **opposite** |
| D_evasive@0.5 | `evasive@0.5` | Δ Y, C_naive-B_naive, all cells | > 0 | -0.1807 [-0.2124, -0.1468] (n = 36) | 0.000 | **opposite** |

- gate `no_stubs`: PASS — {"error_stubs": 0, "pass": true}
- gate `balanced`: PASS — {"pass": true}
- gate `noask_invariant_to_answer_noise`: PASS — {"B_naive_mean_Y_by_level": {"evasive@0": 0.5647613960113961, "evasive@0.1": 0.5647613960113961, "evasive@0.2": 0.5647613960113961, "evasive@0.3": 0.5647613960113961, "evasive@0.4": 0.5647613960113961, "evasive@0.5": 0.5647613960113961}, "pass": true}
- gate `diagonal_zero_regret`: PASS — {"C_ans0.3 mean true-model decision regret at evasive@0.3": 4.163336342344337e-18, "tolerance": 0.001, "O_true-C_ans0.3 identical trajectories (ties may differ)": {"identical_share": 0.9097222222222222, "n_pairs": 720}, "pass": true}
- gate `replicates_mis1_exactly`: PASS — {"mean_Y_difference_by_process_arm": {"evasive@0.3:A_prior": 0.0, "evasive@0.3:B_naive": 0.0, "evasive@0.3:C_naive": 0.0, "evasive@0.3:C_mix0.2": 0.0, "evasive@0.3:O_true": 0.0, "history@0.1:A_prior": 0.0, "history@0.1:B_naive": 0.0, "history@0.1:B_mix0.2": 0.0, "history@0.1:C_naive": 0.0, "history@0.1:O_true": 0.0, "stochastic@0.15:A_prior": 0.0, "stochastic@0.15:B_naive": 0.0, "stochastic@0.15:B_mix0.2": 0.0, "stochastic@0.15:C_naive": 0.0, "stochastic@0.15:O_true": 0.0}, "pass": true}

## Read-out (computed)

- true flip 0.00: trusting answers (`C_naive − B_naive`) +0.0673 [+0.0522, +0.0841] (above 0); the 0.2 hedge over trusting (`C_ans0.2 − C_naive`) -0.0256 [-0.0344, -0.0175] (below 0); value of questions with the right answer model (`O_true − B_naive`) +0.0673 [+0.0522, +0.0841] (above 0); O_true asks 1.571 per episode.
- true flip 0.10: trusting answers (`C_naive − B_naive`) +0.0089 [-0.0110, +0.0311] (unresolved); the 0.2 hedge over trusting (`C_ans0.2 − C_naive`) +0.0171 [+0.0013, +0.0328] (above 0); value of questions with the right answer model (`O_true − B_naive`) +0.0352 [+0.0176, +0.0544] (above 0); O_true asks 1.012 per episode.
- true flip 0.20: trusting answers (`C_naive − B_naive`) -0.0472 [-0.0713, -0.0217] (below 0); the 0.2 hedge over trusting (`C_ans0.2 − C_naive`) +0.0606 [+0.0372, +0.0841] (above 0); value of questions with the right answer model (`O_true − B_naive`) +0.0134 [-0.0048, +0.0327] (unresolved); O_true asks 0.608 per episode.
- true flip 0.30: trusting answers (`C_naive − B_naive`) -0.0868 [-0.1089, -0.0626] (below 0); the 0.2 hedge over trusting (`C_ans0.2 − C_naive`) +0.0839 [+0.0625, +0.1059] (above 0); value of questions with the right answer model (`O_true − B_naive`) -0.0016 [-0.0094, +0.0066] (unresolved); O_true asks 0.094 per episode.
- true flip 0.40: trusting answers (`C_naive − B_naive`) -0.1337 [-0.1585, -0.1064] (below 0); the 0.2 hedge over trusting (`C_ans0.2 − C_naive`) +0.1170 [+0.0936, +0.1411] (above 0); value of questions with the right answer model (`O_true − B_naive`) +0.0000 [+0.0000, +0.0000] (unresolved); O_true asks 0.037 per episode.
- true flip 0.50: trusting answers (`C_naive − B_naive`) -0.1807 [-0.2124, -0.1468] (below 0); the 0.2 hedge over trusting (`C_ans0.2 − C_naive`) +0.1472 [+0.1182, +0.1765] (above 0); value of questions with the right answer model (`O_true − B_naive`) +0.0000 [+0.0000, +0.0000] (unresolved); O_true asks 0.037 per episode.
- `history@0.1`: learning (`B_naive − A_prior`) +0.1262 [+0.0981, +0.1546] (above 0); robust filter (`B_mix0.2 − B_naive`) +0.0000 [+0.0000, +0.0000] with identical trajectories 1.000; misspecification cost for the ASK planner (`O_true − C_naive`) +0.0000 [+0.0000, +0.0000] (unresolved).
- `history@0.2`: learning (`B_naive − A_prior`) +0.1260 [+0.0983, +0.1540] (above 0); robust filter (`B_mix0.2 − B_naive`) +0.0000 [+0.0000, +0.0000] with identical trajectories 1.000; misspecification cost for the ASK planner (`O_true − C_naive`) +0.0028 [+0.0012, +0.0047] (above 0).
- `history@0.4`: learning (`B_naive − A_prior`) +0.1224 [+0.0949, +0.1500] (above 0); robust filter (`B_mix0.2 − B_naive`) +0.0000 [+0.0000, +0.0000] with identical trajectories 1.000; misspecification cost for the ASK planner (`O_true − C_naive`) -0.0000 [-0.0022, +0.0022] (unresolved).
- `stochastic@0.15`: learning (`B_naive − A_prior`) +0.0637 [+0.0309, +0.0939] (above 0); robust filter (`B_mix0.2 − B_naive`) +0.0000 [+0.0000, +0.0000] with identical trajectories 1.000; misspecification cost for the ASK planner (`O_true − C_naive`) +0.0298 [+0.0180, +0.0433] (above 0).
- `stochastic@0.3`: learning (`B_naive − A_prior`) +0.0540 [+0.0173, +0.0885] (above 0); robust filter (`B_mix0.2 − B_naive`) +0.0000 [+0.0000, +0.0000] with identical trajectories 1.000; misspecification cost for the ASK planner (`O_true − C_naive`) +0.0298 [+0.0180, +0.0433] (above 0).
- `stochastic@0.5`: learning (`B_naive − A_prior`) +0.0425 [+0.0044, +0.0787] (above 0); robust filter (`B_mix0.2 − B_naive`) +0.0000 [+0.0000, +0.0000] with identical trajectories 1.000; misspecification cost for the ASK planner (`O_true − C_naive`) +0.0298 [+0.0180, +0.0433] (above 0).

## Regenerate

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir results/live_mis2_20261030 --B 5000 --seed 41
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis2_note --dir results/live_mis2_20261030 --out llm_rl_scripts/car_dealer_live/MIS2_RESULTS.md --verdicts results/live_mis2_20261030/verdicts.json
```
