# Split Deal v0.3 — gate SD3-E, the CFR check on 2-2-3

_Rendered by `sweep_e.py render` from `results/split_v03_e_20261111/sweep.json` (sweep `split-deal-v03-e-v0.2`, commit `8f1e8722`, 594972.2 s). Floating point; every number is formatted from the JSON. Predictions as registered in `DESIGN_SPLIT_CFR.md` §5; their operational reading is fixed in the docstring of `sweep_e.py`. Run plan: §10 as amended by §11 (Amendment 1, settled 2026-11-14): CFR+ up to 128,000 iterations, no DCFR stage._

_Kept from the run under §10's plan (sweep `split-deal-v03-e-v0.1`, commit `84653a4`), whose computation the amendment leaves unchanged: ladder game:all/none, ladder game:none/none, game:all/none uniform, game:all/none warm, game:none/none uniform, game:none/none warm._

Game G: instance 2-2-3, Q = 2, merged bank, π = 1/2. Equilibria by CFR+ from a uniform start and from (A3, B2), certified when NashConv ≤ 0.001 Y.

## Gate SD3-E

| prediction | reading | verdict |
|---|---|---|
| E1 | the ladder has not converged: NashConv(A3, B2) in T3 > 0.01 Y | **holds** |
| E2 | B2’s loss to A3 is a level-k artifact: Y_B(σ*_T3) > Y_B(A3, B2) | **holds** |
| E3 | the level unravels at equilibrium (hi answers HIGH ≥ 0.95, P(lo | decline) ≥ 0.9) | **fails** |
| E4 | hiding is worth even less at equilibrium than at level 3 | **not certified** |
| E5 | no all-or-nothing dilemma at equilibrium | **not certified** |

- **E1** **holds**: NashConv(A3, B2) in T3 = 0.1729.
- **E2** (uniform start): Y_B(σ*_T3) 0.4545 against Y_B(A3, B2) 0.3527: holds.
- **E3** (uniform start): hi answers HIGH 92.8 %; P(lo | decline) 64.6 %; P(B asks HIGH) lo 23.2 %, hi 40.7 %: fails.
- **E4**: not certified (no certified equilibrium in a structure it needs).
- **E5**: not certified (no certified equilibrium in a structure it needs).

Checks: ladder profiles on the tree equal the exact rollouts: True; the best responses to B0, A1 and B2 have the values of A1, B2 and A3: True; a best response is never worse than its profile: True.

## Registered consequences

**E1 and E2 hold.** Registered: ““Reading backfires” is cited only as reading through a stale model. A live counterpart that reads silence needs a correct or adaptive model of the focal side, and NEG-33 inherits that constraint.”

**E3 fails.** Registered: “The level does not unravel at equilibrium, so K3’s consequence is specific to level 3. The hiding-treatment decision goes back to Meadowlark before NEG-33 uses it.”

**E4 not certified.** Registered: “The prediction is not settled. Nothing is inferred from it, and the write-up reports the profiles found.”

**E5 not certified.** Registered: “The prediction is not settled. Nothing is inferred from it, and the write-up reports the profiles found.”

## Solves

| structure | start | certified | stage | variant | iterations | NashConv | ε_A | ε_B | seconds | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|
| T2 | uniform | False | not certified | cfr+ | 128000 | 2.5e-02 | 1.3e-02 | 1.2e-02 | 96252.0 | 94462.2 |
| T2 | warm | False | not certified | cfr+ | 128000 | 4.5e-03 | 1.9e-03 | 2.6e-03 | 96106.9 | 94310.0 |
| T3 | uniform | True | extended | cfr+ | 62000 | 1.0e-03 | 4.9e-04 | 5.1e-04 | 79575.7 | 76845.6 |
| T3 | warm | False | not certified | cfr+ | 128000 | 4.0e-02 | 1.9e-02 | 2.1e-02 | 337782.6 | 202775.1 |
| game:all/none | uniform | True | budget | cfr+ | 500 | 6.5e-04 | 4.3e-04 | 2.2e-04 | 5.1 | 4.3 |
| game:all/none | warm | True | 4x iterations | cfr+ | 19000 | 9.5e-04 | 6.8e-04 | 2.8e-04 | 166.6 | 148.3 |
| game:none/all | uniform | True | extended | cfr+ | 30000 | 9.8e-04 | 3.7e-04 | 6.1e-04 | 91403.0 | 14258.1 |
| game:none/all | warm | False | not certified | cfr+ | 128000 | 5.5e-02 | 2.0e-02 | 3.4e-02 | 175901.1 | 62910.0 |
| game:none/none | uniform | True | budget | cfr+ | 500 | 3.7e-05 | 2.3e-05 | 1.4e-05 | 2.0 | 1.7 |
| game:none/none | warm | True | budget | cfr+ | 500 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 2.1 | 1.7 |

## Exploitability (reported)

| structure | profile | NashConv | Y_A | Y_B | A’s best response | B’s best response | B at A’s best response | A at B’s best response |
|---|---|---|---|---|---|---|---|---|
| T2 | A1/B0 | 5.0e-02 | 0.4498 | 0.4229 | 0.4498 | 0.4726 | 0.4229 | 0.4144 |
| T2 | A1/B2 | 1.2e-01 | 0.4391 | 0.4726 | 0.5573 | 0.4726 | 0.3546 | 0.4144 |
| T2 | A3/B2 | 1.8e-01 | 0.5573 | 0.3546 | 0.5573 | 0.5329 | 0.3546 | 0.3473 |
| T2 | σ* (uniform) | 2.5e-02 | 0.4351 | 0.4409 | 0.4482 | 0.4528 | 0.4567 | 0.4435 |
| T2 | σ* (warm) | 4.5e-03 | 0.4548 | 0.4603 | 0.4567 | 0.4629 | 0.4628 | 0.4389 |
| T3 | A1/B0 | 6.6e-02 | 0.4651 | 0.4047 | 0.4651 | 0.4703 | 0.4047 | 0.3880 |
| T3 | A1/B2 | 1.2e-01 | 0.4364 | 0.4703 | 0.5571 | 0.4703 | 0.3527 | 0.3880 |
| T3 | A3/B2 | 1.7e-01 | 0.5571 | 0.3527 | 0.5571 | 0.5256 | 0.3527 | 0.3518 |
| T3 | σ* (uniform) | 1.0e-03 | 0.4566 | 0.4545 | 0.4571 | 0.4550 | 0.4592 | 0.4499 |
| T3 | σ* (warm) | 4.0e-02 | 0.4349 | 0.4270 | 0.4537 | 0.4481 | 0.4467 | 0.4452 |
| game:all/none | A1/B0 | 3.1e-02 | 0.3861 | 0.4096 | 0.3861 | 0.4404 | 0.4096 | 0.3892 |
| game:all/none | A1/B2 | 6.2e-02 | 0.3899 | 0.4404 | 0.4520 | 0.4404 | 0.4403 | 0.3892 |
| game:all/none | A3/B2 | 4.4e-02 | 0.4520 | 0.4403 | 0.4520 | 0.4847 | 0.4403 | 0.3729 |
| game:all/none | σ* (uniform) | 6.5e-04 | 0.4213 | 0.4483 | 0.4218 | 0.4485 | 0.4537 | 0.4143 |
| game:all/none | σ* (warm) | 9.5e-04 | 0.4425 | 0.4459 | 0.4432 | 0.4462 | 0.4468 | 0.4102 |
| game:none/all | A1/B0 | 6.2e-02 | 0.4404 | 0.3899 | 0.4404 | 0.4520 | 0.3899 | 0.4103 |
| game:none/all | A1/B2 | 4.4e-02 | 0.4403 | 0.4520 | 0.4847 | 0.4520 | 0.3792 | 0.4103 |
| game:none/all | A3/B2 | 8.2e-02 | 0.4847 | 0.3792 | 0.4847 | 0.4612 | 0.3792 | 0.3904 |
| game:none/all | σ* (uniform) | 9.8e-04 | 0.4516 | 0.4406 | 0.4520 | 0.4412 | 0.4458 | 0.4497 |
| game:none/all | σ* (warm) | 5.5e-02 | 0.4121 | 0.3783 | 0.4325 | 0.4127 | 0.4047 | 0.4443 |
| game:none/none | A1/B0 | 0.0e+00 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3686 |
| game:none/none | A1/B2 | 0.0e+00 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3686 |
| game:none/none | A3/B2 | 0.0e+00 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3686 |
| game:none/none | σ* (uniform) | 3.7e-05 | 0.4003 | 0.3862 | 0.4004 | 0.3862 | 0.3933 | 0.3963 |
| game:none/none | σ* (warm) | 0.0e+00 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3686 |

## Behaviour (reported)

| structure | profile | A level | Y_A | Y_B | Y_A common | Y_B common | agree, A proposes | agree, B proposes | E |
|---|---|---|---|---|---|---|---|---|---|
| T2 | A1/B0 | all | 0.4498 | 0.4229 | 0.2384 | 0.2239 | 79.3 % | 67.3 % | 0.7533 |
| T2 | A1/B0 | lo | 0.4799 | 0.4518 | 0.2880 | 0.2392 | 81.5 % | 75.9 % | 0.7447 |
| T2 | A1/B0 | hi | 0.4198 | 0.3940 | 0.1889 | 0.2086 | 77.2 % | 58.6 % | 0.7633 |
| T2 | A1/B2 | all | 0.4391 | 0.4726 | 0.2306 | 0.2506 | 79.3 % | 87.0 % | 0.8525 |
| T2 | A1/B2 | lo | 0.4403 | 0.5189 | 0.2642 | 0.2750 | 81.5 % | 97.5 % | 0.8414 |
| T2 | A1/B2 | hi | 0.4379 | 0.4263 | 0.1971 | 0.2262 | 77.2 % | 76.5 % | 0.8653 |
| T2 | A3/B2 | all | 0.5573 | 0.3546 | 0.2938 | 0.1878 | 79.0 % | 79.3 % | 0.8195 |
| T2 | A3/B2 | lo | 0.5741 | 0.3877 | 0.3444 | 0.2052 | 82.7 % | 84.6 % | 0.8123 |
| T2 | A3/B2 | hi | 0.5405 | 0.3215 | 0.2432 | 0.1704 | 75.3 % | 74.1 % | 0.8278 |
| T2 | σ* (uniform), not certified | all | 0.4351 | 0.4409 | 0.2293 | 0.2329 | 77.5 % | 79.8 % | 0.7989 |
| T2 | σ* (uniform), not certified | lo | 0.4471 | 0.4795 | 0.2683 | 0.2532 | 80.0 % | 86.8 % | 0.7828 |
| T2 | σ* (uniform), not certified | hi | 0.4231 | 0.4023 | 0.1904 | 0.2127 | 74.9 % | 72.7 % | 0.8174 |
| T2 | σ* (warm), not certified | all | 0.4548 | 0.4603 | 0.2398 | 0.2432 | 81.5 % | 81.7 % | 0.8409 |
| T2 | σ* (warm), not certified | lo | 0.4690 | 0.4961 | 0.2814 | 0.2619 | 83.9 % | 87.8 % | 0.8170 |
| T2 | σ* (warm), not certified | hi | 0.4406 | 0.4244 | 0.1983 | 0.2244 | 79.1 % | 75.6 % | 0.8685 |
| T3 | A1/B0 | all | 0.4651 | 0.4047 | 0.2466 | 0.2146 | 76.9 % | 66.0 % | 0.7393 |
| T3 | A1/B0 | lo | 0.4974 | 0.4427 | 0.2985 | 0.2347 | 81.5 % | 73.5 % | 0.7477 |
| T3 | A1/B0 | hi | 0.4328 | 0.3666 | 0.1948 | 0.1944 | 72.2 % | 58.6 % | 0.7297 |
| T3 | A1/B2 | all | 0.4364 | 0.4703 | 0.2292 | 0.2500 | 77.5 % | 87.0 % | 0.8426 |
| T3 | A1/B2 | lo | 0.4372 | 0.5245 | 0.2623 | 0.2784 | 82.1 % | 97.5 % | 0.8451 |
| T3 | A1/B2 | hi | 0.4355 | 0.4160 | 0.1960 | 0.2216 | 72.8 % | 76.5 % | 0.8398 |
| T3 | A3/B2 | all | 0.5571 | 0.3527 | 0.2952 | 0.1857 | 85.8 % | 79.3 % | 0.8465 |
| T3 | A3/B2 | lo | 0.5936 | 0.3628 | 0.3562 | 0.1906 | 88.3 % | 84.0 % | 0.8232 |
| T3 | A3/B2 | hi | 0.5206 | 0.3426 | 0.2343 | 0.1809 | 83.3 % | 74.7 % | 0.8734 |
| T3 | σ* (uniform) | all | 0.4566 | 0.4545 | 0.2410 | 0.2402 | 80.6 % | 81.8 % | 0.8346 |
| T3 | σ* (uniform) | lo | 0.4736 | 0.4945 | 0.2842 | 0.2611 | 84.4 % | 87.7 % | 0.8196 |
| T3 | σ* (uniform) | hi | 0.4396 | 0.4145 | 0.1978 | 0.2194 | 76.8 % | 75.9 % | 0.8519 |
| T3 | σ* (warm), not certified | all | 0.4349 | 0.4270 | 0.2294 | 0.2255 | 75.5 % | 73.9 % | 0.7653 |
| T3 | σ* (warm), not certified | lo | 0.4494 | 0.4622 | 0.2696 | 0.2438 | 78.2 % | 80.5 % | 0.7512 |
| T3 | σ* (warm), not certified | hi | 0.4203 | 0.3918 | 0.1892 | 0.2072 | 72.9 % | 67.2 % | 0.7816 |
| game:all/none | A1/B0 | all | 0.3861 | 0.4096 | 0.2044 | 0.2171 | 58.0 % | 67.3 % | 0.6412 |
| game:all/none | A1/B0 | lo | 0.4090 | 0.4472 | 0.2454 | 0.2370 | 61.7 % | 75.9 % | 0.6399 |
| game:all/none | A1/B0 | hi | 0.3632 | 0.3721 | 0.1634 | 0.1971 | 54.3 % | 58.6 % | 0.6427 |
| game:all/none | A1/B2 | all | 0.3899 | 0.4404 | 0.2064 | 0.2336 | 58.0 % | 79.3 % | 0.7106 |
| game:all/none | A1/B2 | lo | 0.4126 | 0.4817 | 0.2475 | 0.2552 | 61.7 % | 87.7 % | 0.7094 |
| game:all/none | A1/B2 | hi | 0.3673 | 0.3992 | 0.1653 | 0.2119 | 54.3 % | 71.0 % | 0.7119 |
| game:all/none | A3/B2 | all | 0.4520 | 0.4403 | 0.2395 | 0.2330 | 78.1 % | 79.3 % | 0.8098 |
| game:all/none | A3/B2 | lo | 0.4815 | 0.4770 | 0.2889 | 0.2522 | 81.5 % | 87.7 % | 0.8087 |
| game:all/none | A3/B2 | hi | 0.4225 | 0.4036 | 0.1901 | 0.2139 | 74.7 % | 71.0 % | 0.8110 |
| game:all/none | σ* (uniform) | all | 0.4213 | 0.4483 | 0.2233 | 0.2380 | 70.4 % | 75.2 % | 0.7482 |
| game:all/none | σ* (uniform) | lo | 0.4488 | 0.4905 | 0.2693 | 0.2601 | 74.5 % | 82.5 % | 0.7503 |
| game:all/none | σ* (uniform) | hi | 0.3938 | 0.4060 | 0.1772 | 0.2159 | 66.2 % | 67.9 % | 0.7457 |
| game:all/none | σ* (warm) | all | 0.4425 | 0.4459 | 0.2349 | 0.2370 | 74.8 % | 73.5 % | 0.7740 |
| game:all/none | σ* (warm) | lo | 0.4768 | 0.4876 | 0.2861 | 0.2587 | 78.8 % | 82.2 % | 0.7860 |
| game:all/none | σ* (warm) | hi | 0.4082 | 0.4041 | 0.1837 | 0.2152 | 70.7 % | 64.8 % | 0.7602 |
| game:none/all | A1/B0 | all | 0.4404 | 0.3899 | 0.2336 | 0.2064 | 79.3 % | 58.0 % | 0.7106 |
| game:none/all | A1/B0 | lo | 0.4717 | 0.4168 | 0.2830 | 0.2204 | 81.5 % | 64.2 % | 0.7024 |
| game:none/all | A1/B0 | hi | 0.4091 | 0.3630 | 0.1841 | 0.1924 | 77.2 % | 51.9 % | 0.7200 |
| game:none/all | A1/B2 | all | 0.4403 | 0.4520 | 0.2330 | 0.2395 | 79.3 % | 78.1 % | 0.8098 |
| game:none/all | A1/B2 | lo | 0.4650 | 0.4880 | 0.2790 | 0.2583 | 81.5 % | 87.7 % | 0.8042 |
| game:none/all | A1/B2 | hi | 0.4156 | 0.4160 | 0.1870 | 0.2207 | 77.2 % | 68.5 % | 0.8162 |
| game:none/all | A3/B2 | all | 0.4847 | 0.3792 | 0.2566 | 0.2019 | 74.4 % | 71.0 % | 0.7475 |
| game:none/all | A3/B2 | lo | 0.5139 | 0.4331 | 0.3083 | 0.2306 | 77.8 % | 84.6 % | 0.7832 |
| game:none/all | A3/B2 | hi | 0.4554 | 0.3253 | 0.2049 | 0.1731 | 71.0 % | 57.4 % | 0.7064 |
| game:none/all | σ* (uniform) | all | 0.4516 | 0.4406 | 0.2400 | 0.2338 | 77.3 % | 74.0 % | 0.7861 |
| game:none/all | σ* (uniform) | lo | 0.4899 | 0.4860 | 0.2939 | 0.2576 | 81.4 % | 85.0 % | 0.8073 |
| game:none/all | σ* (uniform) | hi | 0.4134 | 0.3952 | 0.1860 | 0.2101 | 73.3 % | 63.0 % | 0.7617 |
| game:none/all | σ* (warm), not certified | all | 0.4121 | 0.3783 | 0.2187 | 0.1998 | 72.9 % | 58.4 % | 0.6727 |
| game:none/all | σ* (warm), not certified | lo | 0.4425 | 0.4130 | 0.2655 | 0.2182 | 75.5 % | 67.0 % | 0.6790 |
| game:none/all | σ* (warm), not certified | hi | 0.3818 | 0.3435 | 0.1718 | 0.1814 | 70.4 % | 49.8 % | 0.6654 |
| game:none/none | A1/B0 | all | 0.3766 | 0.3766 | 0.1995 | 0.1995 | 58.0 % | 58.0 % | 0.5985 |
| game:none/none | A1/B0 | lo | 0.4007 | 0.4122 | 0.2404 | 0.2182 | 61.7 % | 64.2 % | 0.5976 |
| game:none/none | A1/B0 | hi | 0.3525 | 0.3410 | 0.1586 | 0.1809 | 54.3 % | 51.9 % | 0.5995 |
| game:none/none | A1/B2 | all | 0.3766 | 0.3766 | 0.1995 | 0.1995 | 58.0 % | 58.0 % | 0.5985 |
| game:none/none | A1/B2 | lo | 0.4007 | 0.4122 | 0.2404 | 0.2182 | 61.7 % | 64.2 % | 0.5976 |
| game:none/none | A1/B2 | hi | 0.3525 | 0.3410 | 0.1586 | 0.1809 | 54.3 % | 51.9 % | 0.5995 |
| game:none/none | A3/B2 | all | 0.3766 | 0.3766 | 0.1995 | 0.1995 | 58.0 % | 58.0 % | 0.5985 |
| game:none/none | A3/B2 | lo | 0.4007 | 0.4122 | 0.2404 | 0.2182 | 61.7 % | 64.2 % | 0.5976 |
| game:none/none | A3/B2 | hi | 0.3525 | 0.3410 | 0.1586 | 0.1809 | 54.3 % | 51.9 % | 0.5995 |
| game:none/none | σ* (uniform) | all | 0.4003 | 0.3862 | 0.2120 | 0.2048 | 64.1 % | 56.8 % | 0.6282 |
| game:none/none | σ* (uniform) | lo | 0.4249 | 0.4151 | 0.2549 | 0.2199 | 65.3 % | 63.0 % | 0.6216 |
| game:none/none | σ* (uniform) | hi | 0.3758 | 0.3574 | 0.1691 | 0.1898 | 63.0 % | 50.6 % | 0.6359 |
| game:none/none | σ* (warm) | all | 0.3766 | 0.3766 | 0.1995 | 0.1995 | 58.0 % | 58.0 % | 0.5985 |
| game:none/none | σ* (warm) | lo | 0.4007 | 0.4122 | 0.2404 | 0.2182 | 61.7 % | 64.2 % | 0.5976 |
| game:none/none | σ* (warm) | hi | 0.3525 | 0.3410 | 0.1586 | 0.1809 | 54.3 % | 51.9 % | 0.5995 |

**A’s answering at B’s HIGH in T3** (masses from the prior; answer rate = answered / asked):

| profile | round | lo asked | lo answers | hi asked | hi answers |
|---|---|---|---|---|---|
| A1/B0 | 1 | 0.5000 | 70.4 % | 0.5000 | 64.2 % |
| A1/B0 | 2 | 0.0000 | — | 0.0000 | — |
| A1/B0 | pooled P(lo given decline) | 45.3 % | | | |
| A1/B2 | 1 | 0.0185 | 50.0 % | 0.0000 | — |
| A1/B2 | 2 | 0.0000 | — | 0.0000 | — |
| A1/B2 | pooled P(lo given decline) | 100.0 % | | | |
| A3/B2 | 1 | 0.1111 | 69.4 % | 0.0802 | 69.2 % |
| A3/B2 | 2 | 0.0062 | 0.0 % | 0.0185 | 83.3 % |
| A3/B2 | pooled P(lo given decline) | 59.1 % | | | |
| σ* (uniform) | 1 | 0.0596 | 98.2 % | 0.0977 | 100.0 % |
| σ* (uniform) | 2 | 0.0566 | 54.7 % | 0.1059 | 86.2 % |
| σ* (uniform) | pooled P(lo given decline) | 64.6 % | | | |
| σ* (warm) | 1 | 0.0252 | 95.5 % | 0.0718 | 99.0 % |
| σ* (warm) | 2 | 0.0530 | 18.9 % | 0.0790 | 71.2 % |
| σ* (warm) | pooled P(lo given decline) | 65.2 % | | | |

## Bridges (reported)

| structure | pairing | Y_A on G | Y_B on G | Y_A, full bank, Q = 2 | Y_B, full bank, Q = 2 | Y_A, SD3-K (full bank, Q = 3) | Y_B, SD3-K |
|---|---|---|---|---|---|---|---|
| T2 | A1/B0 | 0.4498 | 0.4229 | 0.4498 | 0.4229 | 0.4661 | 0.4292 |
| T2 | A1/B2 | 0.4391 | 0.4726 | 0.4391 | 0.4726 | 0.4462 | 0.4706 |
| T2 | A3/B2 | 0.5573 | 0.3546 | 0.5583 | 0.3587 | 0.5694 | 0.3561 |
| T3 | A1/B0 | 0.4651 | 0.4047 | 0.4651 | 0.4047 | 0.4941 | 0.4108 |
| T3 | A1/B2 | 0.4364 | 0.4703 | 0.4364 | 0.4703 | 0.4378 | 0.4793 |
| T3 | A3/B2 | 0.5571 | 0.3527 | 0.5574 | 0.3519 | 0.5832 | 0.3286 |
| game:all/none | A1/B0 | 0.3861 | 0.4096 | 0.3861 | 0.4096 | 0.3953 | 0.4179 |
| game:all/none | A1/B2 | 0.3899 | 0.4404 | 0.3899 | 0.4404 | 0.3879 | 0.4474 |
| game:all/none | A3/B2 | 0.4520 | 0.4403 | 0.4520 | 0.4403 | 0.4582 | 0.4511 |
| game:none/all | A1/B0 | 0.4404 | 0.3899 | 0.4404 | 0.3899 | 0.4474 | 0.3879 |
| game:none/all | A1/B2 | 0.4403 | 0.4520 | 0.4403 | 0.4520 | 0.4511 | 0.4582 |
| game:none/all | A3/B2 | 0.4847 | 0.3792 | 0.4852 | 0.3760 | 0.5419 | 0.3508 |
| game:none/none | A1/B0 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 |
| game:none/none | A1/B2 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 |
| game:none/none | A3/B2 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 | 0.3766 |

Not run here: the extension of these verdicts to the nine instances at Q = 3 (§7 of `DESIGN_SPLIT_CFR.md`).
