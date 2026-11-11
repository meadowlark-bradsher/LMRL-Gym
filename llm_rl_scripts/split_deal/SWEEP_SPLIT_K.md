# Split Deal v0.3 — B2, A3 and gate SD3-K

_Rendered by `sweep_k.py render` from `results/split_v03_k_20261110/sweep.json` (sweep `split-deal-v03-k-v0.1`, commit `2cb6606`, 3258.9 s). Exact on the model: no sampling, no API; every number is formatted from the JSON. Predictions as registered in `DESIGN_SPLIT_B2.md` §6; their operational reading is fixed in the docstring of `sweep_k.py`._

B2 is the exact best response to A1 (which best-responds to B0); A3 is the exact best response to B2. Protocol 4.2 on the v0.3 game, Q = 3; primary configuration: 13-question bank, A1 as built (ties to declining), π = 1/2. Independent uniform prior; Y on each side's own scale.

## Gate SD3-K

| prediction | reading | verdict |
|---|---|---|
| K1 | B2 reads A1 profitably (T3) | **holds** |
| K2 | against B2, a lo-level A1 does worse in T3 than in T2 | **holds** |
| K3 | a lo-level A3 still declines most of B2's HIGH, and hi answers more than lo | **fails** |
| K4 | hiding is worth less to a lo-level A against B2 than against B0 | **holds** |
| K5 | the 2 × 2 against B2 is still not a strict dilemma | **holds** |

- **K1** **holds**: T3, Y_B(A1, B2) 0.4754 against Y_B(A1, B0) 0.4077; with B2-d 0.4273.
- **K2** **holds**: a lo-level A1 against B2, Y_A^lo(T3) 0.4286 against Y_A^lo(T2) 0.4487.
- **K3** **fails**: at B2's HIGH in T3, a lo-level A3 declines 45.9 % of 582 decisions; answer rates hi 77.3 % (987 decisions), lo 54.1 %.
- **K4** **holds**: a lo-level A's gain from choosing over answering everything, Y_A^lo(T3) − Y_A^lo(T2): +0.0148 for (A3, B2) against +0.0323 for (A1, B0).
- **K5** **holds**: the 2 × 2 with (A1, B2): A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

Checks: B2's computed values are what its rollouts realize: True; A3's: True. The recomputed (A1, B0) values match the stored SD3-D ones (`results/split_v03_l2_20261110/sweep.json`): True.

## Registered consequences

**K3 fails.** Registered: “Disclosure already unravels at level 3. Split Deal's selective hiding was an artifact of a counterpart that does not read declines, and the two-sided live design carries no hiding treatment.”

**K5 holds.** Registered: “The all-or-nothing dilemma is absent in Split Deal against every counterpart tried (v0.2 with B0, v0.3 with B0, v0.3 with B2). The two-way motivation then rests on selective disclosure alone, and the write-up says so.”

## Sensitivities and the common scale (reported, not gating)

**A1′ (ties to answering)**

- **K1** **holds**: T3, Y_B(A1, B2) 0.4736 against Y_B(A1, B0) 0.4096; with B2-d 0.4286.
- **K2** **holds**: a lo-level A1 against B2, Y_A^lo(T3) 0.4300 against Y_A^lo(T2) 0.4487.
- **K3** **fails**: at B2's HIGH in T3, a lo-level A3 declines 12.4 % of 623 decisions; answer rates hi 98.7 % (1095 decisions), lo 87.6 %.
- **K4** **holds**: a lo-level A's gain from choosing over answering everything, Y_A^lo(T3) − Y_A^lo(T2): +0.0152 for (A3, B2) against +0.0323 for (A1, B0).
- **K5** **holds**: the 2 × 2 with (A1, B2): A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

**the 12-question bank (no HIGH)**

- **K1** **holds**: T3, Y_B(A1, B2) 0.4733 against Y_B(A1, B0) 0.4118; with B2-d 0.4291.
- **K2** **holds**: a lo-level A1 against B2, Y_A^lo(T3) 0.4275 against Y_A^lo(T2) 0.4537.
- **K3** **not testable**: at B2's HIGH in T3, a lo-level A3 declines — of 0 decisions; answer rates hi — (0 decisions), lo —.
- **K4** **holds**: a lo-level A's gain from choosing over answering everything, Y_A^lo(T3) − Y_A^lo(T2): +0.0152 for (A3, B2) against +0.0260 for (A1, B0).
- **K5** **holds**: the 2 × 2 with (A1, B2): A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

**π = 1/4**

- **K1** **holds**: T3, Y_B(A1, B2) 0.5293 against Y_B(A1, B0) 0.4432; with B2-d 0.4760.
- **K2** **holds**: a lo-level A1 against B2, Y_A^lo(T3) 0.3623 against Y_A^lo(T2) 0.3753.
- **K3** **holds**: at B2's HIGH in T3, a lo-level A3 declines 55.4 % of 769 decisions; answer rates hi 66.9 % (1047 decisions), lo 44.6 %.
- **K4** **holds**: a lo-level A's gain from choosing over answering everything, Y_A^lo(T3) − Y_A^lo(T2): +0.0162 for (A3, B2) against +0.0572 for (A1, B0).
- **K5** **holds**: the 2 × 2 with (A1, B2): A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

**π = 3/4**

- **K1** **holds**: T3, Y_B(A1, B2) 0.4022 against Y_B(A1, B0) 0.3677; with B2-d 0.3756.
- **K2** **holds**: a lo-level A1 against B2, Y_A^lo(T3) 0.5115 against Y_A^lo(T2) 0.5221.
- **K3** **fails**: at B2's HIGH in T3, a lo-level A3 declines 33.1 % of 740 decisions; answer rates hi 84.6 % (1082 decisions), lo 66.9 %.
- **K4** **holds**: a lo-level A's gain from choosing over answering everything, Y_A^lo(T3) − Y_A^lo(T2): +0.0096 for (A3, B2) against +0.0130 for (A1, B0).
- **K5** **holds**: the 2 × 2 with (A1, B2): A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

**Common scale, (U − own d)/10, primary runs**

- **K1** **holds**: T3, Y_B(A1, B2) 0.2520 against Y_B(A1, B0) 0.2155; with B2-d 0.2261.
- **K2** **holds**: a lo-level A1 against B2, Y_A^lo(T3) 0.2572 against Y_A^lo(T2) 0.2692.
- **K4** **holds**: a lo-level A's gain from choosing over answering everything, Y_A^lo(T3) − Y_A^lo(T2): +0.0089 for (A3, B2) against +0.0194 for (A1, B0).
- **K5** **holds**: the 2 × 2 with (A1, B2): A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

## With B2: D1′–D3′ readings and the 2 × 2 (primary)

| pairing | E(T0) | E(T1) | E(T2) | E(T3) | D1′ | Y_A(T1) | Y_A(T2) | Y_A^lo(T1) | Y_A^lo(T2) |
|---|---|---|---|---|---|---|---|---|---|
| (A1, B2) | 0.6008 | 0.8608 | 0.8869 | 0.8942 | True | 0.4311 | 0.4360 | 0.4573 | 0.4487 |
| (A3, B2) | 0.6008 | 0.8208 | 0.8871 | 0.8898 | True | 0.4965 | 0.5243 | 0.5329 | 0.5597 |
| (A1, B0) | 0.6008 | 0.7361 | 0.8299 | 0.8387 | True | 0.4210 | 0.4437 | 0.4549 | 0.4627 |

**2 × 2, (A1, B2)**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4311 | 0.4360 | 0.3536 | 0.3751 |
| B | 0.4210 | 0.4511 | 0.3536 | 0.4366 |

**2 × 2, (A3, B2)**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4965 | 0.5243 | 0.3536 | 0.4366 |
| B | 0.4311 | 0.3648 | 0.3536 | 0.3649 |

**2 × 2, (A1, B0)**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4210 | 0.4437 | 0.3536 | 0.3763 |
| B | 0.3973 | 0.4188 | 0.3536 | 0.3751 |

## By structure and pairing (primary)

Answers: answered / decisions at B's questions, HIGH and value questions apart. `asks HIGH`: episodes in which B asks HIGH. `misread`: episodes in which A's true type leaves B's belief.

| structure | pairing | A level | Y_A | Y_B | Y_A resp | Y_B resp | agree, A proposes | agree, B proposes | E | HIGH answered | value answered | asks HIGH | misread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | (A1, B0) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| T0 | (A1, B0) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 0 | 0 / 0 | 0 | 0 |
| T0 | (A1, B0) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 0 | 0 / 0 | 0 | 0 |
| T0 | (A1, B2) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| T0 | (A1, B2) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 0 | 0 / 0 | 0 | 0 |
| T0 | (A1, B2) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 0 | 0 / 0 | 0 | 0 |
| T0 | (A3, B2) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| T0 | (A3, B2) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 0 | 0 / 0 | 0 | 0 |
| T0 | (A3, B2) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 0 | 0 / 0 | 0 | 0 |
| T1 | (A1, B0) | all | 0.4210 | 0.3751 | 0.2831 | 0.3261 | 81.0 % | 55.8 % | 0.7361 | — | — | — | — |
| T1 | (A1, B0) | lo | 0.4549 | 0.4182 | 0.3143 | 0.3436 | 84.4 % | 64.9 % | 0.7571 | 0 / 0 | 0 / 0 | 0 | 0 |
| T1 | (A1, B0) | hi | 0.3872 | 0.3321 | 0.2519 | 0.3087 | 77.7 % | 46.8 % | 0.7128 | 0 / 0 | 0 / 0 | 0 | 0 |
| T1 | (A1, B2) | all | 0.4311 | 0.4366 | 0.3031 | 0.3261 | 81.0 % | 79.4 % | 0.8608 | — | — | — | — |
| T1 | (A1, B2) | lo | 0.4573 | 0.4879 | 0.3191 | 0.3436 | 84.4 % | 89.1 % | 0.8736 | 0 / 0 | 0 / 0 | 0 | 0 |
| T1 | (A1, B2) | hi | 0.4048 | 0.3854 | 0.2872 | 0.3087 | 77.7 % | 69.8 % | 0.8466 | 0 / 0 | 0 / 0 | 0 | 0 |
| T1 | (A3, B2) | all | 0.4965 | 0.3649 | 0.4536 | 0.3273 | 76.9 % | 74.7 % | 0.8208 | — | — | — | — |
| T1 | (A3, B2) | lo | 0.5329 | 0.3945 | 0.4919 | 0.3498 | 80.6 % | 80.9 % | 0.8269 | 0 / 0 | 0 / 0 | 0 | 4458 |
| T1 | (A3, B2) | hi | 0.4600 | 0.3352 | 0.4153 | 0.3048 | 73.1 % | 68.5 % | 0.8143 | 0 / 0 | 0 / 0 | 0 | 3901 |
| T2 | (A1, B0) | all | 0.4437 | 0.4188 | 0.3284 | 0.3261 | 81.0 % | 73.5 % | 0.8299 | — | — | — | — |
| T2 | (A1, B0) | lo | 0.4627 | 0.4667 | 0.3299 | 0.3436 | 84.4 % | 80.4 % | 0.8301 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| T2 | (A1, B0) | hi | 0.4247 | 0.3709 | 0.3269 | 0.3087 | 77.7 % | 66.5 % | 0.8299 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| T2 | (A1, B2) | all | 0.4360 | 0.4511 | 0.3130 | 0.3261 | 81.0 % | 84.6 % | 0.8869 | — | — | — | — |
| T2 | (A1, B2) | lo | 0.4487 | 0.5027 | 0.3020 | 0.3436 | 84.4 % | 91.4 % | 0.8811 | 455 / 455 | 1200 / 1200 | 455 | 0 |
| T2 | (A1, B2) | hi | 0.4232 | 0.3995 | 0.3240 | 0.3087 | 77.7 % | 77.7 % | 0.8936 | 420 / 420 | 1330 / 1330 | 420 | 0 |
| T2 | (A1, B2-d) | all | 0.4437 | 0.4188 | 0.3284 | 0.3261 | 81.0 % | 73.5 % | 0.8299 | — | — | — | — |
| T2 | (A1, B2-d) | lo | 0.4627 | 0.4667 | 0.3299 | 0.3436 | 84.4 % | 80.4 % | 0.8301 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| T2 | (A1, B2-d) | hi | 0.4247 | 0.3709 | 0.3269 | 0.3087 | 77.7 % | 66.5 % | 0.8299 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| T2 | (A3, B2) | all | 0.5243 | 0.3648 | 0.4805 | 0.3219 | 83.4 % | 81.4 % | 0.8871 | — | — | — | — |
| T2 | (A3, B2) | lo | 0.5597 | 0.3941 | 0.5136 | 0.3360 | 86.0 % | 87.4 % | 0.8842 | 777 / 777 | 6188 / 6188 | 777 | 3791 |
| T2 | (A3, B2) | hi | 0.4888 | 0.3355 | 0.4473 | 0.3078 | 80.8 % | 75.5 % | 0.8906 | 1534 / 1534 | 5585 / 5585 | 1534 | 3183 |
| T3 | (A1, B0) | all | 0.4684 | 0.4077 | 0.3869 | 0.3291 | 79.4 % | 74.9 % | 0.8387 | — | — | — | — |
| T3 | (A1, B0) | lo | 0.4950 | 0.4513 | 0.4068 | 0.3524 | 82.8 % | 81.5 % | 0.8421 | 2547 / 5644 | 8250 / 11420 | 5644 | 0 |
| T3 | (A1, B0) | hi | 0.4418 | 0.3640 | 0.3670 | 0.3058 | 76.0 % | 68.3 % | 0.8353 | 4790 / 5654 | 7651 / 11410 | 5654 | 0 |
| T3 | (A1, B2) | all | 0.4206 | 0.4754 | 0.3124 | 0.3743 | 81.0 % | 84.7 % | 0.8942 | — | — | — | — |
| T3 | (A1, B2) | lo | 0.4286 | 0.5362 | 0.3008 | 0.4096 | 84.7 % | 91.7 % | 0.8915 | 95 / 296 | 2097 / 3080 | 296 | 0 |
| T3 | (A1, B2) | hi | 0.4125 | 0.4146 | 0.3240 | 0.3390 | 77.2 % | 77.7 % | 0.8974 | 218 / 226 | 2245 / 3168 | 226 | 0 |
| T3 | (A1, B2-d) | all | 0.4455 | 0.4273 | 0.3411 | 0.3291 | 79.4 % | 77.2 % | 0.8433 | — | — | — | — |
| T3 | (A1, B2-d) | lo | 0.4607 | 0.4799 | 0.3382 | 0.3524 | 82.8 % | 84.7 % | 0.8451 | 2547 / 5644 | 8250 / 11420 | 5644 | 0 |
| T3 | (A1, B2-d) | hi | 0.4303 | 0.3747 | 0.3440 | 0.3058 | 76.0 % | 69.7 % | 0.8416 | 4790 / 5654 | 7651 / 11410 | 5654 | 0 |
| T3 | (A3, B2) | all | 0.5344 | 0.3524 | 0.4943 | 0.3163 | 84.6 % | 81.5 % | 0.8898 | — | — | — | — |
| T3 | (A3, B2) | lo | 0.5745 | 0.3734 | 0.5349 | 0.3271 | 86.9 % | 86.5 % | 0.8796 | 315 / 582 | 4541 / 8563 | 582 | 4563 |
| T3 | (A3, B2) | hi | 0.4943 | 0.3314 | 0.4537 | 0.3055 | 82.3 % | 76.5 % | 0.9013 | 763 / 987 | 4148 / 7844 | 987 | 3913 |
| game:all/all | (A1, B0) | all | 0.4437 | 0.4188 | 0.3284 | 0.3261 | 81.0 % | 73.5 % | 0.8299 | — | — | — | — |
| game:all/all | (A1, B0) | lo | 0.4627 | 0.4667 | 0.3299 | 0.3436 | 84.4 % | 80.4 % | 0.8301 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/all | (A1, B0) | hi | 0.4247 | 0.3709 | 0.3269 | 0.3087 | 77.7 % | 66.5 % | 0.8299 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/all | (A1, B2) | all | 0.4360 | 0.4511 | 0.3130 | 0.3261 | 81.0 % | 84.6 % | 0.8869 | — | — | — | — |
| game:all/all | (A1, B2) | lo | 0.4487 | 0.5027 | 0.3020 | 0.3436 | 84.4 % | 91.4 % | 0.8811 | 455 / 455 | 1200 / 1200 | 455 | 0 |
| game:all/all | (A1, B2) | hi | 0.4232 | 0.3995 | 0.3240 | 0.3087 | 77.7 % | 77.7 % | 0.8936 | 420 / 420 | 1330 / 1330 | 420 | 0 |
| game:all/all | (A1, B2-d) | all | 0.4437 | 0.4188 | 0.3284 | 0.3261 | 81.0 % | 73.5 % | 0.8299 | — | — | — | — |
| game:all/all | (A1, B2-d) | lo | 0.4627 | 0.4667 | 0.3299 | 0.3436 | 84.4 % | 80.4 % | 0.8301 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/all | (A1, B2-d) | hi | 0.4247 | 0.3709 | 0.3269 | 0.3087 | 77.7 % | 66.5 % | 0.8299 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/all | (A3, B2) | all | 0.5243 | 0.3648 | 0.4805 | 0.3219 | 83.4 % | 81.4 % | 0.8871 | — | — | — | — |
| game:all/all | (A3, B2) | lo | 0.5597 | 0.3941 | 0.5136 | 0.3360 | 86.0 % | 87.4 % | 0.8842 | 777 / 777 | 6188 / 6188 | 777 | 3791 |
| game:all/all | (A3, B2) | hi | 0.4888 | 0.3355 | 0.4473 | 0.3078 | 80.8 % | 75.5 % | 0.8906 | 1534 / 1534 | 5585 / 5585 | 1534 | 3183 |
| game:all/none | (A1, B0) | all | 0.3763 | 0.3973 | 0.3284 | 0.2831 | 55.8 % | 73.5 % | 0.6946 | — | — | — | — |
| game:all/none | (A1, B0) | lo | 0.3863 | 0.4487 | 0.3299 | 0.3077 | 59.7 % | 80.4 % | 0.6997 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/none | (A1, B0) | hi | 0.3662 | 0.3458 | 0.3269 | 0.2585 | 52.0 % | 66.5 % | 0.6891 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/none | (A1, B2) | all | 0.3751 | 0.4210 | 0.3261 | 0.2831 | 55.8 % | 81.0 % | 0.7361 | — | — | — | — |
| game:all/none | (A1, B2) | lo | 0.3844 | 0.4736 | 0.3261 | 0.3077 | 59.7 % | 88.1 % | 0.7399 | 3108 / 3108 | 10382 / 10382 | 3108 | 0 |
| game:all/none | (A1, B2) | hi | 0.3658 | 0.3685 | 0.3261 | 0.2585 | 52.0 % | 74.0 % | 0.7322 | 3108 / 3108 | 10368 / 10368 | 3108 | 0 |
| game:all/none | (A1, B2-d) | all | 0.3763 | 0.3973 | 0.3284 | 0.2831 | 55.8 % | 73.5 % | 0.6946 | — | — | — | — |
| game:all/none | (A1, B2-d) | lo | 0.3863 | 0.4487 | 0.3299 | 0.3077 | 59.7 % | 80.4 % | 0.6997 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/none | (A1, B2-d) | hi | 0.3662 | 0.3458 | 0.3269 | 0.2585 | 52.0 % | 66.5 % | 0.6891 | 5688 / 5688 | 11376 / 11376 | 5688 | 0 |
| game:all/none | (A3, B2) | all | 0.4366 | 0.4311 | 0.3261 | 0.3031 | 79.4 % | 81.0 % | 0.8608 | — | — | — | — |
| game:all/none | (A3, B2) | lo | 0.4540 | 0.4775 | 0.3261 | 0.3156 | 81.9 % | 88.1 % | 0.8551 | 3108 / 3108 | 10382 / 10382 | 3108 | 0 |
| game:all/none | (A3, B2) | hi | 0.4192 | 0.3846 | 0.3261 | 0.2907 | 77.0 % | 74.0 % | 0.8677 | 3108 / 3108 | 10368 / 10368 | 3108 | 0 |
| game:none/all | (A1, B0) | all | 0.4210 | 0.3751 | 0.2831 | 0.3261 | 81.0 % | 55.8 % | 0.7361 | — | — | — | — |
| game:none/all | (A1, B0) | lo | 0.4549 | 0.4182 | 0.3143 | 0.3436 | 84.4 % | 64.9 % | 0.7571 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/all | (A1, B0) | hi | 0.3872 | 0.3321 | 0.2519 | 0.3087 | 77.7 % | 46.8 % | 0.7128 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/all | (A1, B2) | all | 0.4311 | 0.4366 | 0.3031 | 0.3261 | 81.0 % | 79.4 % | 0.8608 | — | — | — | — |
| game:none/all | (A1, B2) | lo | 0.4573 | 0.4879 | 0.3191 | 0.3436 | 84.4 % | 89.1 % | 0.8736 | 0 / 0 | 0 / 0 | 0 | 0 |
| game:none/all | (A1, B2) | hi | 0.4048 | 0.3854 | 0.2872 | 0.3087 | 77.7 % | 69.8 % | 0.8466 | 0 / 0 | 0 / 0 | 0 | 0 |
| game:none/all | (A1, B2-d) | all | 0.4210 | 0.3751 | 0.2831 | 0.3261 | 81.0 % | 55.8 % | 0.7361 | — | — | — | — |
| game:none/all | (A1, B2-d) | lo | 0.4549 | 0.4182 | 0.3143 | 0.3436 | 84.4 % | 64.9 % | 0.7571 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/all | (A1, B2-d) | hi | 0.3872 | 0.3321 | 0.2519 | 0.3087 | 77.7 % | 46.8 % | 0.7128 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/all | (A3, B2) | all | 0.4965 | 0.3649 | 0.4536 | 0.3273 | 76.9 % | 74.7 % | 0.8208 | — | — | — | — |
| game:none/all | (A3, B2) | lo | 0.5329 | 0.3945 | 0.4919 | 0.3498 | 80.6 % | 80.9 % | 0.8269 | 0 / 0 | 0 / 0 | 0 | 4458 |
| game:none/all | (A3, B2) | hi | 0.4600 | 0.3352 | 0.4153 | 0.3048 | 73.1 % | 68.5 % | 0.8143 | 0 / 0 | 0 / 0 | 0 | 3901 |
| game:none/none | (A1, B0) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| game:none/none | (A1, B0) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/none | (A1, B0) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/none | (A1, B2) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| game:none/none | (A1, B2) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 0 | 0 / 0 | 0 | 0 |
| game:none/none | (A1, B2) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 0 | 0 / 0 | 0 | 0 |
| game:none/none | (A1, B2-d) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| game:none/none | (A1, B2-d) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/none | (A1, B2-d) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 5688 | 0 / 11376 | 5688 | 0 |
| game:none/none | (A3, B2) | all | 0.3536 | 0.3536 | 0.2831 | 0.2831 | 55.8 % | 55.8 % | 0.6008 | — | — | — | — |
| game:none/none | (A3, B2) | lo | 0.3785 | 0.4002 | 0.3143 | 0.3077 | 59.7 % | 64.9 % | 0.6268 | 0 / 0 | 0 / 0 | 0 | 0 |
| game:none/none | (A3, B2) | hi | 0.3287 | 0.3070 | 0.2519 | 0.2585 | 52.0 % | 46.8 % | 0.5720 | 0 / 0 | 0 / 0 | 0 | 0 |

Not run here: the CFR check on the smallest instance (§8 of `DESIGN_SPLIT_B2.md`).
