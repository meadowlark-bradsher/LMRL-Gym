# CD-NOUL-1 Stage 1 — likelihood from the model, choice from the planner

_Rendered by `render_noul_note.py` from `analysis.json` (`analyze_noul.py`) and `verdicts.json` (`verdicts_noul.py`) in `results/noul_stage1_20261108`; every number is formatted from JSON, none is typed. Design: `DESIGN_NOUL.md`; rules: `PREREGISTRATION_NOUL.md`._

Env `live-car-dealer-v0.1`, regime easy (r = 2/5), H = 4, c_q = 0; kernel α₀=4/5, β=8, p_counter=1, p_hold=1, shortlist=3. Bank `results/noul_bank_20261108`. 18000 episodes (5 repeats per cell, seed 20261112), 0 error stubs, 11737.27 s. Bootstrap B = 5000 (seed 0); clusters = (buyer ranking, seller ranking) type pair; 36 clusters. Y = net gain / (U_max − r), scale 0.6000. Primary cells: focal_opens; primary hedge a = 1/5; agents' menu belief decoded by `noul_answer` at a = 1/5.

## Gates

- `pipeline`: **pass** — error_stubs 0, episodes_per_key {'jev_agent|oblique': 720, 'jev_agent|plain': 720, 'plan_ask-noul_answer-0|oblique': 720, 'plan_ask-noul_answer-0|plain': 720, 'plan_ask-noul_answer-0.2|oblique': 720, 'plan_ask-noul_answer-0.2|plain': 720, 'plan_ask-noul_type-0|oblique': 720, 'plan_ask-noul_type-0|plain': 720, 'plan_ask-noul_type-0.2|oblique': 720, 'plan_ask-noul_type-0.2|plain': 720, 'plan_ask-oracle-0|oblique': 720, 'plan_ask-oracle-0|plain': 720, 'plan_ask-oracle-0.2|oblique': 720, 'plan_ask-oracle-0.2|plain': 720, 'plan_ask-regex-0|oblique': 720, 'plan_ask-regex-0|plain': 720, 'plan_ask-regex-0.2|oblique': 720, 'plan_ask-regex-0.2|plain': 720, 'plan_ask-sonnet_answer-0|oblique': 720, 'plan_ask-sonnet_answer-0|plain': 720, 'plan_ask-sonnet_answer-0.2|oblique': 720, 'plan_ask-sonnet_answer-0.2|plain': 720, 'plan_noask|none': 720, 'sonnet_agent|oblique': 720, 'sonnet_agent|plain': 720}, well_formed {'jev_agent|oblique': 1.0, 'jev_agent|plain': 1.0, 'sonnet_agent|oblique': 1.0, 'sonnet_agent|plain': 1.0}
- `oracle_pair_reproduces_ceiling`: **pass** — contrast {'point': 0.06429843304843307, 'ci95': [0.04777421652421655, 0.08326210826210828], 'n_clusters': 36}, ceiling_delta_menu_Y 0.06381087747560826
- `oracle_decoder_text_invariant`: **pass** — contrast {'point': 0.0, 'ci95': [0.0, 0.0], 'n_clusters': 36}
- gates pass: **True**

## Exact ceilings (Y units)

| cell set | n cells | Δ\*_menu | Δ\*_full | V\*_menu ask | V\*_menu no-ask |
|---|---|---|---|---|---|
| all | 24 | 0.0638 | 0.0638 | 0.6321 | 0.5682 |
| focal_opens | 12 | 0.1019 | 0.1019 | 0.6321 | 0.5301 |
| opponent_opens | 12 | 0.0257 | 0.0257 | 0.6321 | 0.6064 |

## Per (arm, regime)

| arm | regime | n | mean Y (all) [95% CI] | mean Y (focal opens) | agreement | walk | questions / ep | regret (true) | zero-regret share |
|---|---|---|---|---|---|---|---|---|---|
| jev_agent | oblique | 720 | 0.4990 [0.4108, 0.5850] | 0.4248 | 0.787 | 0.212 | 0.001 | 0.0832 | 0.307 |
| jev_agent | plain | 720 | 0.5001 [0.4104, 0.5871] | 0.4235 | 0.786 | 0.214 | 0.003 | 0.0839 | 0.306 |
| plan_ask-noul_answer-0 | oblique | 720 | 0.5069 [0.4111, 0.5993] | 0.5059 | 0.776 | 0.224 | 1.660 | 0.0609 | 0.841 |
| plan_ask-noul_answer-0 | plain | 720 | 0.6100 [0.5283, 0.6859] | 0.6105 | 0.960 | 0.040 | 1.660 | 0.0066 | 0.975 |
| plan_ask-noul_answer-0.2 | oblique | 720 | 0.5720 [0.4809, 0.6587] | 0.5385 | 0.867 | 0.133 | 0.626 | 0.0315 | 0.639 |
| plan_ask-noul_answer-0.2 | plain | 720 | 0.5963 [0.5152, 0.6722] | 0.5822 | 0.938 | 0.062 | 0.621 | 0.0186 | 0.666 |
| plan_ask-noul_type-0 | oblique | 720 | 0.5586 [0.4678, 0.6464] | 0.5590 | 0.854 | 0.146 | 1.660 | 0.0338 | 0.890 |
| plan_ask-noul_type-0 | plain | 720 | 0.6105 [0.5256, 0.6898] | 0.6106 | 0.950 | 0.050 | 1.660 | 0.0055 | 0.980 |
| plan_ask-noul_type-0.2 | oblique | 720 | 0.5853 [0.4986, 0.6665] | 0.5662 | 0.901 | 0.099 | 0.674 | 0.0251 | 0.662 |
| plan_ask-noul_type-0.2 | plain | 720 | 0.6097 [0.5359, 0.6793] | 0.6090 | 0.982 | 0.018 | 0.532 | 0.0134 | 0.684 |
| plan_ask-oracle-0 | oblique | 720 | 0.6317 [0.5557, 0.7027] | 0.6325 | 1.000 | 0.000 | 1.560 | 0.0000 | 1.000 |
| plan_ask-oracle-0 | plain | 720 | 0.6317 [0.5557, 0.7027] | 0.6325 | 1.000 | 0.000 | 1.560 | 0.0000 | 1.000 |
| plan_ask-oracle-0.2 | oblique | 720 | 0.6097 [0.5359, 0.6793] | 0.6090 | 0.982 | 0.018 | 0.621 | 0.0134 | 0.684 |
| plan_ask-oracle-0.2 | plain | 720 | 0.6097 [0.5359, 0.6793] | 0.6090 | 0.982 | 0.018 | 0.621 | 0.0134 | 0.684 |
| plan_ask-regex-0 | oblique | 720 | 0.4881 [0.3831, 0.5881] | 0.4721 | 0.694 | 0.306 | 1.660 | 0.0589 | 0.839 |
| plan_ask-regex-0 | plain | 720 | 0.6317 [0.5557, 0.7027] | 0.6325 | 1.000 | 0.000 | 1.560 | 0.0000 | 1.000 |
| plan_ask-regex-0.2 | oblique | 720 | 0.5470 [0.4498, 0.6385] | 0.4936 | 0.806 | 0.194 | 0.697 | 0.0462 | 0.620 |
| plan_ask-regex-0.2 | plain | 720 | 0.6097 [0.5359, 0.6793] | 0.6090 | 0.982 | 0.018 | 0.621 | 0.0134 | 0.684 |
| plan_ask-sonnet_answer-0 | oblique | 720 | 0.5384 [0.4447, 0.6291] | 0.5421 | 0.812 | 0.188 | 1.660 | 0.0426 | 0.870 |
| plan_ask-sonnet_answer-0 | plain | 720 | 0.6306 [0.5548, 0.7015] | 0.6314 | 1.000 | 0.000 | 1.660 | 0.0000 | 1.000 |
| plan_ask-sonnet_answer-0.2 | oblique | 720 | 0.5802 [0.4918, 0.6624] | 0.5549 | 0.890 | 0.110 | 0.651 | 0.0268 | 0.649 |
| plan_ask-sonnet_answer-0.2 | plain | 720 | 0.6088 [0.5347, 0.6786] | 0.6072 | 0.982 | 0.018 | 0.621 | 0.0134 | 0.684 |
| plan_noask | none | 720 | 0.5674 [0.4812, 0.6489] | 0.5313 | 0.900 | 0.100 | 0.000 | 0.0081 | 0.731 |
| sonnet_agent | oblique | 720 | 0.4390 [0.3794, 0.4950] | 0.4189 | 0.931 | 0.069 | 0.003 | 0.1198 | 0.174 |
| sonnet_agent | plain | 720 | 0.4318 [0.3715, 0.4897] | 0.4225 | 0.912 | 0.087 | 0.008 | 0.1188 | 0.173 |

## Decoders in play (all cells)

| arm | regime | decodes | answer log-loss (bits) | mean P(true type) after decode | exact filter's | confident wrong | episodes with true type zeroed | final P(true type) |
|---|---|---|---|---|---|---|---|---|
| jev_agent | oblique | 1 | 1.344 | 0.140 | 0.500 | 0 | 0 | 0.794 |
| jev_agent | plain | 2 | 0.390 | 0.620 | 0.750 | 0 | 0 | 0.795 |
| plan_ask-noul_answer-0 | oblique | 1195 | 0.807 | 0.341 | 0.614 | 0 | 0 | 0.640 |
| plan_ask-noul_answer-0 | plain | 1195 | 0.238 | 0.518 | 0.628 | 0 | 0 | 0.786 |
| plan_ask-noul_answer-0.2 | oblique | 451 | 0.623 | 0.275 | 0.468 | 0 | 0 | 0.808 |
| plan_ask-noul_answer-0.2 | plain | 447 | 0.222 | 0.339 | 0.463 | 0 | 0 | 0.808 |
| plan_ask-noul_type-0 | oblique | 1195 | — | 0.412 | 0.613 | 0 | 0 | 0.693 |
| plan_ask-noul_type-0 | plain | 1195 | — | 0.528 | 0.626 | 0 | 0 | 0.783 |
| plan_ask-noul_type-0.2 | oblique | 485 | — | 0.372 | 0.498 | 0 | 0 | 0.824 |
| plan_ask-noul_type-0.2 | plain | 383 | — | 0.330 | 0.373 | 0 | 0 | 0.817 |
| plan_ask-oracle-0 | oblique | 1123 | 0.000 | 0.604 | 0.604 | 0 | 0 | 0.859 |
| plan_ask-oracle-0 | plain | 1123 | 0.000 | 0.604 | 0.604 | 0 | 0 | 0.859 |
| plan_ask-oracle-0.2 | oblique | 447 | 0.000 | 0.387 | 0.463 | 0 | 0 | 0.817 |
| plan_ask-oracle-0.2 | plain | 447 | 0.000 | 0.387 | 0.463 | 0 | 0 | 0.817 |
| plan_ask-regex-0 | oblique | 1195 | 1.678 | 0.293 | 0.610 | 51 | 51 | 0.615 |
| plan_ask-regex-0 | plain | 1123 | 0.000 | 0.604 | 0.604 | 0 | 0 | 0.859 |
| plan_ask-regex-0.2 | oblique | 502 | 2.067 | 0.262 | 0.522 | 33 | 0 | 0.821 |
| plan_ask-regex-0.2 | plain | 447 | 0.000 | 0.387 | 0.463 | 0 | 0 | 0.817 |
| plan_ask-sonnet_answer-0 | oblique | 1195 | 0.624 | 0.381 | 0.617 | 0 | 0 | 0.683 |
| plan_ask-sonnet_answer-0 | plain | 1195 | 0.091 | 0.580 | 0.628 | 0 | 0 | 0.837 |
| plan_ask-sonnet_answer-0.2 | oblique | 469 | 0.478 | 0.311 | 0.488 | 0 | 0 | 0.805 |
| plan_ask-sonnet_answer-0.2 | plain | 447 | 0.086 | 0.369 | 0.463 | 0 | 0 | 0.811 |
| sonnet_agent | oblique | 2 | 0.996 | 0.161 | 0.375 | 0 | 0 | 0.639 |
| sonnet_agent | plain | 6 | 0.217 | 0.233 | 0.333 | 0 | 0 | 0.636 |

## Selectivity against the evaluator's reference (ASK strictly optimal under the a = 0 oracle at the exact belief)

| arm | regime | asked / ASK-optimal | P(ask \| optimal) | asked / legal-not-optimal | P(ask \| not optimal) | difference [95% CI] |
|---|---|---|---|---|---|---|
| jev_agent | oblique | 1/701 | 0.001 | 0/459 | 0.000 | +0.001 [+0.000, +0.004] |
| jev_agent | plain | 1/700 | 0.001 | 1/461 | 0.002 | -0.001 [-0.003, +0.000] |
| plan_ask-noul_answer-0 | oblique | 1116/1116 | 1.000 | 79/1044 | 0.076 | +0.924 [+0.898, +0.950] |
| plan_ask-noul_answer-0 | plain | 1123/1123 | 1.000 | 72/1037 | 0.069 | +0.931 [+0.903, +0.957] |
| plan_ask-noul_answer-0.2 | oblique | 377/888 | 0.425 | 74/959 | 0.077 | +0.347 [+0.319, +0.377] |
| plan_ask-noul_answer-0.2 | plain | 377/888 | 0.425 | 70/959 | 0.073 | +0.352 [+0.319, +0.385] |
| plan_ask-noul_type-0 | oblique | 1111/1111 | 1.000 | 84/1049 | 0.080 | +0.920 [+0.890, +0.950] |
| plan_ask-noul_type-0 | plain | 1117/1117 | 1.000 | 78/1043 | 0.075 | +0.925 [+0.897, +0.953] |
| plan_ask-noul_type-0.2 | oblique | 393/880 | 0.447 | 92/967 | 0.095 | +0.351 [+0.321, +0.383] |
| plan_ask-noul_type-0.2 | plain | 377/888 | 0.425 | 6/959 | 0.006 | +0.418 [+0.393, +0.444] |
| plan_ask-oracle-0 | oblique | 1123/1123 | 1.000 | 0/1037 | 0.000 | +1.000 [+1.000, +1.000] |
| plan_ask-oracle-0 | plain | 1123/1123 | 1.000 | 0/1037 | 0.000 | +1.000 [+1.000, +1.000] |
| plan_ask-oracle-0.2 | oblique | 377/888 | 0.425 | 70/959 | 0.073 | +0.352 [+0.319, +0.385] |
| plan_ask-oracle-0.2 | plain | 377/888 | 0.425 | 70/959 | 0.073 | +0.352 [+0.319, +0.385] |
| plan_ask-regex-0 | oblique | 1123/1123 | 1.000 | 72/1037 | 0.069 | +0.931 [+0.903, +0.957] |
| plan_ask-regex-0 | plain | 1123/1123 | 1.000 | 0/1037 | 0.000 | +1.000 [+1.000, +1.000] |
| plan_ask-regex-0.2 | oblique | 377/875 | 0.431 | 125/972 | 0.129 | +0.302 [+0.277, +0.329] |
| plan_ask-regex-0.2 | plain | 377/888 | 0.425 | 70/959 | 0.073 | +0.352 [+0.319, +0.385] |
| plan_ask-sonnet_answer-0 | oblique | 1123/1123 | 1.000 | 72/1037 | 0.069 | +0.931 [+0.903, +0.957] |
| plan_ask-sonnet_answer-0 | plain | 1123/1123 | 1.000 | 72/1037 | 0.069 | +0.931 [+0.903, +0.957] |
| plan_ask-sonnet_answer-0.2 | oblique | 377/887 | 0.425 | 92/960 | 0.096 | +0.329 [+0.300, +0.361] |
| plan_ask-sonnet_answer-0.2 | plain | 377/888 | 0.425 | 70/959 | 0.073 | +0.352 [+0.319, +0.385] |
| sonnet_agent | oblique | 2/702 | 0.003 | 0/379 | 0.000 | +0.003 [+0.000, +0.007] |
| sonnet_agent | plain | 6/703 | 0.009 | 0/376 | 0.000 | +0.009 [+0.002, +0.017] |

## Agent arms

| arm | regime | decisions | well-formed | retried | positions chosen | tokens in / out |
|---|---|---|---|---|---|---|
| jev_agent | oblique | 1272 | 1.000 | 0 | {'0': 222, '1': 215, '2': 165, '3': 167, '4': 180, '5': 165, '6': 122, '7': 34, '8': 2} | — |
| jev_agent | plain | 1274 | 1.000 | 0 | {'0': 219, '1': 200, '2': 193, '3': 166, '4': 161, '5': 179, '6': 117, '7': 37, '8': 2} | — |
| sonnet_agent | oblique | 1126 | 1.000 | 160 | {'0': 173, '1': 171, '2': 152, '3': 140, '4': 184, '5': 170, '6': 101, '7': 35} | 936446 / 451199 |
| sonnet_agent | plain | 1131 | 1.000 | 137 | {'0': 168, '1': 191, '2': 168, '3': 166, '4': 132, '5': 159, '6': 115, '7': 31, '8': 1} | 940323 / 453302 |

## Paired contrasts in Y (cluster bootstrap, 95% percentile CI)

| contrast | all cells | focal opens | opponent opens | captured fraction of Δ\*_menu (focal opens) |
|---|---|---|---|---|
| ask_value|plan_ask-oracle-0|plain | +0.0643 [+0.0478, +0.0833] | +0.1012 [+0.0727, +0.1324] | +0.0274 [+0.0174, +0.0382] | +0.99 [+0.71, +1.30] |
| decoder|plan_ask-oracle-0|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-oracle-0|oblique | +0.0643 [+0.0478, +0.0833] | +0.1012 [+0.0727, +0.1324] | +0.0274 [+0.0174, +0.0382] | +0.99 [+0.71, +1.30] |
| decoder|plan_ask-oracle-0|oblique | +0.1436 [+0.1104, +0.1775] | +0.1604 [+0.1254, +0.1957] | +0.1268 [+0.0934, +0.1620] | — |
| ask_value|plan_ask-regex-0|plain | +0.0643 [+0.0478, +0.0833] | +0.1012 [+0.0727, +0.1324] | +0.0274 [+0.0174, +0.0382] | +0.99 [+0.71, +1.30] |
| vs_oracle|plan_ask-regex-0|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-regex-0|oblique | -0.0793 [-0.1098, -0.0494] | -0.0592 [-0.0964, -0.0232] | -0.0994 [-0.1303, -0.0687] | -0.58 [-0.95, -0.23] |
| vs_oracle|plan_ask-regex-0|oblique | -0.1436 [-0.1773, -0.1102] | -0.1604 [-0.1957, -0.1254] | -0.1268 [-0.1620, -0.0934] | — |
| ask_value|plan_ask-noul_answer-0|plain | +0.0426 [+0.0212, +0.0646] | +0.0792 [+0.0448, +0.1144] | +0.0061 [-0.0078, +0.0192] | +0.78 [+0.44, +1.12] |
| decoder|plan_ask-noul_answer-0|plain | -0.0217 [-0.0349, -0.0108] | -0.0220 [-0.0354, -0.0104] | -0.0214 [-0.0360, -0.0096] | — |
| vs_oracle|plan_ask-noul_answer-0|plain | -0.0217 [-0.0349, -0.0108] | -0.0220 [-0.0354, -0.0104] | -0.0214 [-0.0360, -0.0096] | — |
| ask_value|plan_ask-noul_answer-0|oblique | -0.0605 [-0.0918, -0.0300] | -0.0254 [-0.0648, +0.0142] | -0.0955 [-0.1266, -0.0664] | -0.25 [-0.64, +0.14] |
| decoder|plan_ask-noul_answer-0|oblique | +0.0188 [-0.0060, +0.0430] | +0.0338 [+0.0041, +0.0621] | +0.0038 [-0.0190, +0.0269] | — |
| vs_oracle|plan_ask-noul_answer-0|oblique | -0.1248 [-0.1564, -0.0943] | -0.1266 [-0.1593, -0.0954] | -0.1230 [-0.1562, -0.0910] | — |
| ask_value|plan_ask-noul_type-0|plain | +0.0431 [+0.0228, +0.0632] | +0.0793 [+0.0476, +0.1106] | +0.0070 [-0.0072, +0.0199] | +0.78 [+0.47, +1.09] |
| decoder|plan_ask-noul_type-0|plain | -0.0212 [-0.0369, -0.0083] | -0.0219 [-0.0393, -0.0077] | -0.0204 [-0.0366, -0.0073] | — |
| vs_oracle|plan_ask-noul_type-0|plain | -0.0212 [-0.0369, -0.0083] | -0.0219 [-0.0393, -0.0077] | -0.0204 [-0.0366, -0.0073] | — |
| ask_value|plan_ask-noul_type-0|oblique | -0.0088 [-0.0370, +0.0182] | +0.0277 [-0.0099, +0.0650] | -0.0453 [-0.0709, -0.0218] | +0.27 [-0.10, +0.64] |
| decoder|plan_ask-noul_type-0|oblique | +0.0705 [+0.0493, +0.0922] | +0.0869 [+0.0616, +0.1124] | +0.0541 [+0.0349, +0.0744] | — |
| vs_oracle|plan_ask-noul_type-0|oblique | -0.0731 [-0.1002, -0.0485] | -0.0735 [-0.1015, -0.0473] | -0.0727 [-0.1006, -0.0480] | — |
| ask_value|plan_ask-sonnet_answer-0|plain | +0.0632 [+0.0465, +0.0823] | +0.1001 [+0.0718, +0.1316] | +0.0264 [+0.0159, +0.0375] | +0.98 [+0.70, +1.29] |
| decoder|plan_ask-sonnet_answer-0|plain | -0.0011 [-0.0021, -0.0003] | -0.0011 [-0.0023, -0.0002] | -0.0011 [-0.0023, -0.0002] | — |
| vs_oracle|plan_ask-sonnet_answer-0|plain | -0.0011 [-0.0021, -0.0003] | -0.0011 [-0.0023, -0.0002] | -0.0011 [-0.0023, -0.0002] | — |
| ask_value|plan_ask-sonnet_answer-0|oblique | -0.0290 [-0.0596, +0.0009] | +0.0108 [-0.0305, +0.0515] | -0.0688 [-0.0960, -0.0427] | +0.11 [-0.30, +0.51] |
| decoder|plan_ask-sonnet_answer-0|oblique | +0.0503 [+0.0314, +0.0709] | +0.0700 [+0.0462, +0.0950] | +0.0306 [+0.0141, +0.0486] | — |
| vs_oracle|plan_ask-sonnet_answer-0|oblique | -0.0933 [-0.1221, -0.0659] | -0.0904 [-0.1196, -0.0620] | -0.0962 [-0.1259, -0.0684] | — |
| type_vs_answer|0|plain | +0.0005 [-0.0085, +0.0097] | +0.0001 [-0.0129, +0.0130] | +0.0009 [-0.0062, +0.0080] | — |
| chooser|jev_agent|0|plain | +0.1100 [+0.0872, +0.1347] | +0.1870 [+0.1608, +0.2154] | +0.0329 [+0.0106, +0.0582] | — |
| chooser|sonnet_agent|0|plain | +0.1782 [+0.1493, +0.2086] | +0.1880 [+0.1660, +0.2115] | +0.1684 [+0.1199, +0.2185] | — |
| type_vs_answer|0|oblique | +0.0517 [+0.0350, +0.0700] | +0.0531 [+0.0333, +0.0751] | +0.0503 [+0.0331, +0.0690] | — |
| chooser|jev_agent|0|oblique | +0.0079 [-0.0283, +0.0471] | +0.0811 [+0.0420, +0.1242] | -0.0654 [-0.1015, -0.0289] | — |
| chooser|sonnet_agent|0|oblique | +0.0679 [+0.0192, +0.1152] | +0.0870 [+0.0435, +0.1307] | +0.0488 [-0.0134, +0.1095] | — |
| ask_value|plan_ask-oracle-0.2|plain | +0.0423 [+0.0215, +0.0648] | +0.0777 [+0.0396, +0.1177] | +0.0070 [+0.0020, +0.0140] | +0.76 [+0.39, +1.15] |
| decoder|plan_ask-oracle-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-oracle-0.2|oblique | +0.0423 [+0.0215, +0.0648] | +0.0777 [+0.0396, +0.1177] | +0.0070 [+0.0020, +0.0140] | +0.76 [+0.39, +1.15] |
| decoder|plan_ask-oracle-0.2|oblique | +0.0627 [+0.0399, +0.0874] | +0.1154 [+0.0744, +0.1582] | +0.0100 [+0.0020, +0.0189] | — |
| ask_value|plan_ask-regex-0.2|plain | +0.0423 [+0.0215, +0.0648] | +0.0777 [+0.0396, +0.1177] | +0.0070 [+0.0020, +0.0140] | +0.76 [+0.39, +1.15] |
| vs_oracle|plan_ask-regex-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-regex-0.2|oblique | -0.0204 [-0.0402, -0.0008] | -0.0377 [-0.0773, +0.0007] | -0.0030 [-0.0080, +0.0000] | -0.37 [-0.76, +0.01] |
| vs_oracle|plan_ask-regex-0.2|oblique | -0.0627 [-0.0873, -0.0399] | -0.1154 [-0.1580, -0.0744] | -0.0100 [-0.0189, -0.0020] | — |
| ask_value|plan_ask-noul_answer-0.2|plain | +0.0290 [+0.0075, +0.0508] | +0.0509 [+0.0113, +0.0909] | +0.0070 [+0.0020, +0.0140] | +0.50 [+0.11, +0.89] |
| decoder|plan_ask-noul_answer-0.2|plain | -0.0134 [-0.0218, -0.0058] | -0.0267 [-0.0436, -0.0116] | +0.0000 [+0.0000, +0.0000] | — |
| vs_oracle|plan_ask-noul_answer-0.2|plain | -0.0134 [-0.0218, -0.0058] | -0.0267 [-0.0436, -0.0116] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-noul_answer-0.2|oblique | +0.0046 [-0.0159, +0.0246] | +0.0072 [-0.0324, +0.0457] | +0.0020 [-0.0040, +0.0080] | +0.07 [-0.32, +0.45] |
| decoder|plan_ask-noul_answer-0.2|oblique | +0.0250 [+0.0158, +0.0353] | +0.0449 [+0.0292, +0.0616] | +0.0050 [+0.0000, +0.0110] | — |
| vs_oracle|plan_ask-noul_answer-0.2|oblique | -0.0377 [-0.0569, -0.0201] | -0.0705 [-0.1052, -0.0385] | -0.0050 [-0.0130, +0.0000] | — |
| ask_value|plan_ask-noul_type-0.2|plain | +0.0423 [+0.0215, +0.0648] | +0.0777 [+0.0396, +0.1177] | +0.0070 [+0.0020, +0.0140] | +0.76 [+0.39, +1.15] |
| decoder|plan_ask-noul_type-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| vs_oracle|plan_ask-noul_type-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-noul_type-0.2|oblique | +0.0179 [-0.0012, +0.0369] | +0.0349 [-0.0023, +0.0708] | +0.0010 [-0.0020, +0.0040] | +0.34 [-0.02, +0.69] |
| decoder|plan_ask-noul_type-0.2|oblique | +0.0383 [+0.0254, +0.0524] | +0.0726 [+0.0489, +0.0983] | +0.0040 [+0.0000, +0.0090] | — |
| vs_oracle|plan_ask-noul_type-0.2|oblique | -0.0244 [-0.0390, -0.0109] | -0.0428 [-0.0679, -0.0191] | -0.0060 [-0.0120, -0.0010] | — |
| ask_value|plan_ask-sonnet_answer-0.2|plain | +0.0414 [+0.0205, +0.0639] | +0.0759 [+0.0376, +0.1165] | +0.0070 [+0.0020, +0.0140] | +0.74 [+0.37, +1.14] |
| decoder|plan_ask-sonnet_answer-0.2|plain | -0.0009 [-0.0017, -0.0002] | -0.0018 [-0.0034, -0.0004] | +0.0000 [+0.0000, +0.0000] | — |
| vs_oracle|plan_ask-sonnet_answer-0.2|plain | -0.0009 [-0.0017, -0.0002] | -0.0018 [-0.0034, -0.0004] | +0.0000 [+0.0000, +0.0000] | — |
| ask_value|plan_ask-sonnet_answer-0.2|oblique | +0.0128 [-0.0085, +0.0341] | +0.0236 [-0.0172, +0.0631] | +0.0020 [-0.0040, +0.0080] | +0.23 [-0.17, +0.62] |
| decoder|plan_ask-sonnet_answer-0.2|oblique | +0.0332 [+0.0203, +0.0470] | +0.0614 [+0.0381, +0.0862] | +0.0050 [+0.0000, +0.0110] | — |
| vs_oracle|plan_ask-sonnet_answer-0.2|oblique | -0.0295 [-0.0464, -0.0143] | -0.0541 [-0.0837, -0.0273] | -0.0050 [-0.0130, +0.0000] | — |
| type_vs_answer|1/5|plain | +0.0134 [+0.0058, +0.0218] | +0.0267 [+0.0116, +0.0437] | +0.0000 [+0.0000, +0.0000] | — |
| chooser|jev_agent|1/5|plain | +0.0963 [+0.0743, +0.1207] | +0.1587 [+0.1295, +0.1919] | +0.0338 [+0.0157, +0.0543] | — |
| chooser|sonnet_agent|1/5|plain | +0.1645 [+0.1349, +0.1957] | +0.1597 [+0.1296, +0.1907] | +0.1693 [+0.1213, +0.2198] | — |
| type_vs_answer|1/5|oblique | +0.0134 [+0.0069, +0.0206] | +0.0277 [+0.0142, +0.0429] | -0.0010 [-0.0040, +0.0020] | — |
| chooser|jev_agent|1/5|oblique | +0.0729 [+0.0478, +0.1005] | +0.1137 [+0.0774, +0.1536] | +0.0321 [+0.0127, +0.0533] | — |
| chooser|sonnet_agent|1/5|oblique | +0.1330 [+0.0948, +0.1708] | +0.1196 [+0.0740, +0.1658] | +0.1463 [+0.1011, +0.1931] | — |
| hedge|oracle|plain | -0.0220 [-0.0311, -0.0132] | -0.0235 [-0.0388, -0.0098] | -0.0204 [-0.0301, -0.0114] | — |
| hedge|oracle|oblique | -0.0220 [-0.0311, -0.0132] | -0.0235 [-0.0388, -0.0098] | -0.0204 [-0.0301, -0.0114] | — |
| hedge|regex|plain | -0.0220 [-0.0311, -0.0132] | -0.0235 [-0.0388, -0.0098] | -0.0204 [-0.0301, -0.0114] | — |
| hedge|regex|oblique | +0.0589 [+0.0433, +0.0743] | +0.0215 [+0.0085, +0.0359] | +0.0964 [+0.0670, +0.1255] | — |
| hedge|noul_answer|plain | -0.0137 [-0.0253, -0.0021] | -0.0283 [-0.0470, -0.0103] | +0.0009 [-0.0106, +0.0131] | — |
| hedge|noul_answer|oblique | +0.0651 [+0.0436, +0.0870] | +0.0326 [+0.0090, +0.0577] | +0.0975 [+0.0688, +0.1278] | — |
| hedge|noul_type|plain | -0.0008 [-0.0162, +0.0170] | -0.0016 [-0.0241, +0.0223] | +0.0000 [-0.0127, +0.0141] | — |
| hedge|noul_type|oblique | +0.0267 [+0.0125, +0.0431] | +0.0072 [-0.0047, +0.0202] | +0.0463 [+0.0233, +0.0710] | — |
| hedge|sonnet_answer|plain | -0.0218 [-0.0311, -0.0129] | -0.0242 [-0.0394, -0.0107] | -0.0194 [-0.0292, -0.0100] | — |
| hedge|sonnet_answer|oblique | +0.0418 [+0.0269, +0.0571] | +0.0128 [+0.0023, +0.0236] | +0.0708 [+0.0457, +0.0968] | — |
| ask_value|jev_agent|plain | -0.0673 [-0.0940, -0.0406] | -0.1078 [-0.1467, -0.0671] | -0.0269 [-0.0489, -0.0067] | -1.06 [-1.44, -0.66] |
| ask_value|jev_agent|oblique | -0.0683 [-0.0947, -0.0412] | -0.1066 [-0.1463, -0.0666] | -0.0301 [-0.0507, -0.0110] | -1.05 [-1.43, -0.65] |
| ask_value|sonnet_agent|plain | -0.1356 [-0.1712, -0.0995] | -0.1088 [-0.1462, -0.0678] | -0.1623 [-0.2149, -0.1132] | -1.07 [-1.43, -0.67] |
| ask_value|sonnet_agent|oblique | -0.1284 [-0.1614, -0.0940] | -0.1124 [-0.1495, -0.0715] | -0.1443 [-0.1919, -0.0990] | -1.10 [-1.47, -0.70] |
| text_invariance|plan_ask-oracle-0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |

### The same pairs in questions asked per episode

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| ask_value|plan_ask-oracle-0|plain | +1.5597 [+1.5083, +1.6097] | +1.8000 [+1.7167, +1.8806] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-oracle-0|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-oracle-0|oblique | +1.5597 [+1.5083, +1.6097] | +1.8000 [+1.7167, +1.8806] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-oracle-0|oblique | -0.1000 [-0.1417, -0.0597] | -0.2000 [-0.2833, -0.1194] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-regex-0|plain | +1.5597 [+1.5083, +1.6097] | +1.8000 [+1.7167, +1.8806] | +1.3194 [+1.2417, +1.3944] |
| vs_oracle|plan_ask-regex-0|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-regex-0|oblique | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| vs_oracle|plan_ask-regex-0|oblique | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_answer-0|plain | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-noul_answer-0|plain | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_answer-0|plain | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_answer-0|oblique | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-noul_answer-0|oblique | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_answer-0|oblique | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_type-0|plain | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-noul_type-0|plain | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_type-0|plain | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_type-0|oblique | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-noul_type-0|oblique | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_type-0|oblique | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-sonnet_answer-0|plain | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-sonnet_answer-0|plain | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-sonnet_answer-0|plain | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-sonnet_answer-0|oblique | +1.6597 [+1.6208, +1.6972] | +2.0000 [+2.0000, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| decoder|plan_ask-sonnet_answer-0|oblique | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-sonnet_answer-0|oblique | +0.1000 [+0.0597, +0.1417] | +0.2000 [+0.1194, +0.2833] | +0.0000 [+0.0000, +0.0000] |
| type_vs_answer|0|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| chooser|jev_agent|0|plain | +1.6569 [+1.6181, +1.6931] | +1.9972 [+1.9917, +2.0000] | +1.3167 [+1.2389, +1.3889] |
| chooser|sonnet_agent|0|plain | +1.6514 [+1.6139, +1.6889] | +1.9861 [+1.9694, +1.9972] | +1.3167 [+1.2389, +1.3917] |
| type_vs_answer|0|oblique | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| chooser|jev_agent|0|oblique | +1.6583 [+1.6194, +1.6958] | +1.9972 [+1.9917, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| chooser|sonnet_agent|0|oblique | +1.6569 [+1.6181, +1.6931] | +1.9944 [+1.9861, +2.0000] | +1.3194 [+1.2417, +1.3944] |
| ask_value|plan_ask-oracle-0.2|plain | +0.6208 [+0.5708, +0.6736] | +1.1944 [+1.1028, +1.2944] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-oracle-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-oracle-0.2|oblique | +0.6208 [+0.5708, +0.6736] | +1.1944 [+1.1028, +1.2944] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-oracle-0.2|oblique | -0.0764 [-0.1097, -0.0431] | -0.1528 [-0.2194, -0.0861] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-regex-0.2|plain | +0.6208 [+0.5708, +0.6736] | +1.1944 [+1.1028, +1.2944] | +0.0472 [+0.0194, +0.0778] |
| vs_oracle|plan_ask-regex-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-regex-0.2|oblique | +0.6972 [+0.6597, +0.7361] | +1.3472 [+1.2806, +1.4167] | +0.0472 [+0.0194, +0.0778] |
| vs_oracle|plan_ask-regex-0.2|oblique | +0.0764 [+0.0431, +0.1097] | +0.1528 [+0.0861, +0.2194] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_answer-0.2|plain | +0.6208 [+0.5708, +0.6736] | +1.1944 [+1.1028, +1.2944] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-noul_answer-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_answer-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_answer-0.2|oblique | +0.6264 [+0.5819, +0.6722] | +1.2056 [+1.1306, +1.2833] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-noul_answer-0.2|oblique | -0.0708 [-0.1042, -0.0389] | -0.1417 [-0.2083, -0.0778] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_answer-0.2|oblique | +0.0056 [-0.0264, +0.0347] | +0.0111 [-0.0528, +0.0694] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_type-0.2|plain | +0.5319 [+0.5167, +0.5500] | +1.0167 [+1.0000, +1.0444] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-noul_type-0.2|plain | -0.0889 [-0.1361, -0.0458] | -0.1778 [-0.2722, -0.0917] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_type-0.2|plain | -0.0889 [-0.1361, -0.0458] | -0.1778 [-0.2722, -0.0917] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-noul_type-0.2|oblique | +0.6736 [+0.6250, +0.7250] | +1.3000 [+1.2139, +1.3944] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-noul_type-0.2|oblique | -0.0236 [-0.0597, +0.0125] | -0.0472 [-0.1194, +0.0250] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-noul_type-0.2|oblique | +0.0528 [+0.0333, +0.0736] | +0.1056 [+0.0667, +0.1472] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-sonnet_answer-0.2|plain | +0.6208 [+0.5708, +0.6736] | +1.1944 [+1.1028, +1.2944] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-sonnet_answer-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-sonnet_answer-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ask_value|plan_ask-sonnet_answer-0.2|oblique | +0.6514 [+0.6042, +0.7014] | +1.2556 [+1.1722, +1.3444] | +0.0472 [+0.0194, +0.0778] |
| decoder|plan_ask-sonnet_answer-0.2|oblique | -0.0458 [-0.0736, -0.0194] | -0.0917 [-0.1472, -0.0389] | +0.0000 [+0.0000, +0.0000] |
| vs_oracle|plan_ask-sonnet_answer-0.2|oblique | +0.0306 [+0.0153, +0.0486] | +0.0611 [+0.0306, +0.0972] | +0.0000 [+0.0000, +0.0000] |
| type_vs_answer|1/5|plain | -0.0889 [-0.1361, -0.0458] | -0.1778 [-0.2722, -0.0917] | +0.0000 [+0.0000, +0.0000] |
| chooser|jev_agent|1/5|plain | +0.6181 [+0.5681, +0.6694] | +1.1917 [+1.1028, +1.2889] | +0.0444 [+0.0167, +0.0778] |
| chooser|sonnet_agent|1/5|plain | +0.6125 [+0.5611, +0.6667] | +1.1806 [+1.0861, +1.2833] | +0.0444 [+0.0194, +0.0722] |
| type_vs_answer|1/5|oblique | +0.0472 [+0.0236, +0.0764] | +0.0944 [+0.0472, +0.1528] | +0.0000 [+0.0000, +0.0000] |
| chooser|jev_agent|1/5|oblique | +0.6250 [+0.5806, +0.6694] | +1.2028 [+1.1278, +1.2806] | +0.0472 [+0.0194, +0.0778] |
| chooser|sonnet_agent|1/5|oblique | +0.6236 [+0.5806, +0.6681] | +1.2000 [+1.1278, +1.2778] | +0.0472 [+0.0194, +0.0778] |
| hedge|oracle|plain | -0.9389 [-1.0028, -0.8708] | -0.6056 [-0.7167, -0.4917] | -1.2722 [-1.3417, -1.1972] |
| hedge|oracle|oblique | -0.9389 [-1.0028, -0.8708] | -0.6056 [-0.7167, -0.4917] | -1.2722 [-1.3417, -1.1972] |
| hedge|regex|plain | -0.9389 [-1.0028, -0.8708] | -0.6056 [-0.7167, -0.4917] | -1.2722 [-1.3417, -1.1972] |
| hedge|regex|oblique | -0.9625 [-1.0125, -0.9111] | -0.6528 [-0.7194, -0.5833] | -1.2722 [-1.3417, -1.1972] |
| hedge|noul_answer|plain | -1.0389 [-1.0944, -0.9778] | -0.8056 [-0.8972, -0.7056] | -1.2722 [-1.3417, -1.1972] |
| hedge|noul_answer|oblique | -1.0333 [-1.0847, -0.9792] | -0.7944 [-0.8694, -0.7167] | -1.2722 [-1.3417, -1.1972] |
| hedge|noul_type|plain | -1.1278 [-1.1611, -1.0903] | -0.9833 [-1.0000, -0.9556] | -1.2722 [-1.3417, -1.1972] |
| hedge|noul_type|oblique | -0.9861 [-1.0389, -0.9264] | -0.7000 [-0.7861, -0.6056] | -1.2722 [-1.3417, -1.1972] |
| hedge|sonnet_answer|plain | -1.0389 [-1.0944, -0.9778] | -0.8056 [-0.8972, -0.7056] | -1.2722 [-1.3417, -1.1972] |
| hedge|sonnet_answer|oblique | -1.0083 [-1.0611, -0.9514] | -0.7444 [-0.8278, -0.6556] | -1.2722 [-1.3417, -1.1972] |
| ask_value|jev_agent|plain | +0.0028 [+0.0000, +0.0083] | +0.0028 [+0.0000, +0.0083] | +0.0028 [+0.0000, +0.0083] |
| ask_value|jev_agent|oblique | +0.0014 [+0.0000, +0.0042] | +0.0028 [+0.0000, +0.0083] | +0.0000 [+0.0000, +0.0000] |
| ask_value|sonnet_agent|plain | +0.0083 [+0.0014, +0.0167] | +0.0139 [+0.0028, +0.0306] | +0.0028 [+0.0000, +0.0083] |
| ask_value|sonnet_agent|oblique | +0.0028 [+0.0000, +0.0069] | +0.0056 [+0.0000, +0.0139] | +0.0000 [+0.0000, +0.0000] |
| text_invariance|plan_ask-oracle-0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

### Decoder pairs in in-episode answer log-loss (bits; lower is better)

| contrast | all cells | focal opens | opponent opens |
|---|---|---|---|
| decoder|plan_ask-oracle-0|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| decoder|plan_ask-oracle-0|oblique | -1.6615 [-2.0610, -1.3211] | -1.7524 [-2.1026, -1.4327] | -1.5422 [-2.0472, -1.1122] |
| decoder|plan_ask-noul_answer-0|plain | +0.2422 [+0.2214, +0.2631] | +0.2270 [+0.2054, +0.2481] | +0.2589 [+0.2379, +0.2799] |
| decoder|plan_ask-noul_answer-0|oblique | -0.8534 [-1.2073, -0.5525] | -0.9464 [-1.2547, -0.6578] | -0.7334 [-1.1990, -0.3344] |
| decoder|plan_ask-noul_type-0|plain | — | — | — |
| decoder|plan_ask-noul_type-0|oblique | — | — | — |
| decoder|plan_ask-sonnet_answer-0|plain | +0.0919 [+0.0867, +0.0972] | +0.0889 [+0.0838, +0.0941] | +0.0952 [+0.0896, +0.1010] |
| decoder|plan_ask-sonnet_answer-0|oblique | -1.0445 [-1.4189, -0.7268] | -1.1265 [-1.4540, -0.8252] | -0.9361 [-1.4195, -0.5289] |
| decoder|plan_ask-oracle-0.2|plain | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| decoder|plan_ask-oracle-0.2|oblique | -2.0532 [-2.7672, -1.4381] | -1.9762 [-2.6752, -1.3755] | -4.8564 [-9.4140, -1.0000] |
| decoder|plan_ask-noul_answer-0.2|plain | +0.2107 [+0.1869, +0.2361] | +0.2107 [+0.1865, +0.2366] | +0.2357 [+0.1677, +0.3133] |
| decoder|plan_ask-noul_answer-0.2|oblique | -1.4725 [-2.1028, -0.9257] | -1.3960 [-2.0137, -0.8560] | -4.1287 [-8.7269, -0.6988] |
| decoder|plan_ask-noul_type-0.2|plain | — | — | — |
| decoder|plan_ask-noul_type-0.2|oblique | — | — | — |
| decoder|plan_ask-sonnet_answer-0.2|plain | +0.0831 [+0.0762, +0.0905] | +0.0827 [+0.0757, +0.0901] | +0.0999 [+0.0713, +0.1298] |
| decoder|plan_ask-sonnet_answer-0.2|oblique | -1.6138 [-2.2769, -1.0419] | -1.5380 [-2.1880, -0.9748] | -4.2981 [-8.9145, -0.6742] |

## Pre-registered rules and verdicts

δ = 0.02 Y; captured minimum 0.5; selectivity maximum 0.2. Vocabulary in `verdicts_noul.py`.

| rule | quantity | prediction | estimate [95% CI] | verdict |
|---|---|---|---|---|
| P1 | Δ Y, plan_ask-noul_answer-0.2 (plain) − plan_noask, focal_opens (PRIMARY) | > 0 and captured fraction of Δ*_menu ≥ 0.5 | +0.0509 [+0.0113, +0.0909] | **positive but below captured minimum** |
| P2 | Δ Y, plan_ask-noul_answer-0.2 − plan_ask-regex-0.2, oblique, focal_opens | > 0 | +0.0449 [+0.0292, +0.0616] | **confirmed** |
| P3 | Δ Y, plan_ask-noul_answer-0.2 − plan_ask-regex-0.2, plain, focal_opens | within δ = 0.02 of 0 | -0.0267 [-0.0436, -0.0116] | **not supported** |
| P4 | Δ Y, plan_ask-noul_type-0.2 − plan_ask-noul_answer-0.2, oblique, focal_opens | ≥ 0 (exploratory) | +0.0277 [+0.0142, +0.0429] | **confirmed** |
| P5_noul_answer | Δ Y, plan_ask-noul_answer-0.2 − plan_ask-noul_answer-0, oblique, focal_opens | ≥ 0 | +0.0326 [+0.0090, +0.0577] | **confirmed** |
| P5_regex | Δ Y, plan_ask-regex-0.2 − plan_ask-regex-0, oblique, focal_opens | ≥ 0 | +0.0215 [+0.0085, +0.0359] | **confirmed** |
| P6_plain | P(ask | ASK optimal) − P(ask | legal, not optimal), sonnet_agent, plain | < 0.2 (upper CI bound) | +0.0085 [+0.0015, +0.0168] | **confirmed** |
| P6_oblique | P(ask | ASK optimal) − P(ask | legal, not optimal), sonnet_agent, oblique | < 0.2 (upper CI bound) | +0.0028 [+0.0000, +0.0072] | **confirmed** |
| P7_jev_agent_plain | Δ Y, plan_ask-noul_answer-0.2 − jev_agent, plain, focal_opens | > 0 | +0.1587 [+0.1295, +0.1919] | **confirmed** |
| P7_jev_agent_oblique | Δ Y, plan_ask-noul_answer-0.2 − jev_agent, oblique, focal_opens | > 0 | +0.1137 [+0.0774, +0.1536] | **confirmed** |
| P7_sonnet_agent_plain | Δ Y, plan_ask-noul_answer-0.2 − sonnet_agent, plain, focal_opens | > 0 | +0.1597 [+0.1296, +0.1907] | **confirmed** |
| P7_sonnet_agent_oblique | Δ Y, plan_ask-noul_answer-0.2 − sonnet_agent, oblique, focal_opens | > 0 | +0.1196 [+0.0740, +0.1658] | **confirmed** |
| P8 | Δ answer log-loss (bits), plan_ask-noul_answer-0.2 − plan_ask-regex-0.2, oblique, all cells | < 0 | -1.4725 [-2.1028, -0.9257] | **confirmed** |

## Read-out (computed)

- P1: +0.0509 [+0.0113, +0.0909] against a ceiling of 0.1019 → captured 50% [11%, 89%] → **positive but below captured minimum**.
- P2: +0.0449 [+0.0292, +0.0616] → **confirmed**.
- P3: -0.0267 [-0.0436, -0.0116] → **not supported**.
- P4: +0.0277 [+0.0142, +0.0429] → **confirmed**.
- P5_noul_answer: +0.0326 [+0.0090, +0.0577] → **confirmed**.
- P5_regex: +0.0215 [+0.0085, +0.0359] → **confirmed**.
- P8: -1.4725 [-2.1028, -0.9257] → **confirmed**.
- P6_plain: P(ask | optimal) 0.009, P(ask | not optimal) 0.000, difference +0.009 [+0.002, +0.017] → **confirmed**.
- P6_oblique: P(ask | optimal) 0.003, P(ask | not optimal) 0.000, difference +0.003 [+0.000, +0.007] → **confirmed**.
- P7_jev_agent_plain: +0.1587 [+0.1295, +0.1919] → **confirmed**.
- P7_jev_agent_oblique: +0.1137 [+0.0774, +0.1536] → **confirmed**.
- P7_sonnet_agent_plain: +0.1597 [+0.1296, +0.1907] → **confirmed**.
- P7_sonnet_agent_oblique: +0.1196 [+0.0740, +0.1658] → **confirmed**.
- The P1 contrast at the other hedge: +0.0792 [+0.0448, +0.1144] (descriptive).
- The Sonnet judge as decoder vs the parser, oblique: +0.0614 [+0.0381, +0.0862] (descriptive).

## Limits

- Stage 1's opponent is the registered T1 kernel: it never lies and its moves are typed; only its answers are text. The strategic responder, the realizer and a Noul over offers are Stage 2.
- The six-ranking type space makes the hypothesis bins free, as in 20Q; an open type space is not addressed here.
- The `oblique` verbalizer may let priorities show beyond the yes/no, so the binary-answer ceiling bounds the answer decoders only; `noul_type` can in principle exceed it there.
- The agents' menus are built from a belief decoded by `noul_answer`; the agent-vs-planner contrast isolates the chooser on identical menus, not an agent with no scaffolding.
