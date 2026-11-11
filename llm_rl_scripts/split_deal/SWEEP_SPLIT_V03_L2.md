# Split Deal v0.3 — L2 two-sided program and gate SD3-D

_Rendered by `sweep_l2_v03.py render` from `results/split_v03_l2_20261110/sweep.json` (sweep `split-deal-v03-l2-v0.1`, commit `3a9237f`, 51.7 s). Exact on the model: no sampling, no API; every number is formatted from the JSON. Predictions as registered in `DESIGN_SPLIT_V03.md` §5; their operational reading was fixed in the docstring of `sweep_l2_v03.py` before the sweep ran._

Protocol 4.2 on the v0.3 game, Q = 3 rounds (A then B), then the proposer is drawn (A with probability π) and makes one binding offer. Each side's type is its values and a private level (lo: d = 4, hi: d = 5.5). B is B0 (asks by expected information gain over the 13 questions, draws no inference from a decline, offers to maximise its own expected gain); A is A1, the exact best response to B0. Independent uniform prior. Y on each side's own scale; E is joint efficiency.

## Gate SD3-D (π = 1/2)

| prediction | reading | verdict |
|---|---|---|
| D1′ | E(T2) ≥ E(T1) ≥ E(T0) | **holds** |
| D2′ | Y_A(T2) < Y_A(T1): the information rent is positive | **fails** |
| D2′-lo | the same for a lo-level A | **fails** |
| D3′ | the 2 × 2 game is a strict prisoner's dilemma for both players (B0 reading) | **fails** |
| D4′ | in T3 at B0's HIGH: hi-level A answers more often than lo-level A, and lo-level A declines most | **holds** |
| D5′ | in T3, A1 answers value questions at a higher rate than HIGH | **holds** |

- **D1′** **holds**: E(T0) 0.6008, E(T1) 0.7361, E(T2) 0.8299 (E(T3) 0.8387).
- **D2′** **fails**: Y_A(T1) 0.4210, Y_A(T2) 0.4437; A's information rent as responder -0.0453.
- **D2′-lo** **fails**: Y_A(T1) 0.4549, Y_A(T2) 0.4627; A's information rent as responder -0.0156.
- **hi-level A (reported, not predicted)**: Y_A(T1) 0.3872, Y_A(T2) 0.4247; A's information rent as responder -0.0750.
- **D3′ (B0 asks by EIG, the registered reading)** **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D3′, second reading (B asks by the exact program)** **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D3′ on the common scale (reported)** **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D4′** **holds**: at B0's HIGH in T3, hi-level A answers 4790 / 5654 (84.7 %), lo-level A answers 2547 / 5644 (45.1 %); lo-level A declines 54.9 % (609 of its declines on a tie).
- **D5′** **holds**: A1 answers value questions 15901 / 22830 (69.6 %) and HIGH 7337 / 11298 (64.9 %).
- Choosing what to answer (T3) against always answering (T2): Y_A +0.0247, Y_B -0.0111, E +0.0088.

Against the control (descriptive): being known is worth +0.0078 Y to a lo-level A with HIGH in the bank and +0.0353 Y without it. Adding HIGH lowers it by 0.0274 Y, in the direction the mechanism predicts (the difference also includes B0 spending a question on the level rather than on values), but not below zero: the value information disclosed alongside still outweighs it, so the rent stays negative.

## Registered consequences

**D2′-lo fails.** Registered: "a private d does not make disclosure costly against B0, and the next rung is B2 before any change to the game."

**D3′ fails.** Registered: "Making the reservation private is not enough against B0, which reads nothing into a decline, and the next rung is B2 before any other change to the game."

**D4′ holds.** Registered: "Against B0, a low-level A hides by declining at no cost. That finding is specific to B0. A counterpart that reads declines (B2) would remove it."

## D4′ by round

Answered / decisions at B0's HIGH in T3, by A's level. First by the round in which B0 asks HIGH on its T2 path (the instance's column in `DESIGN_SPLIT_V03.md` §3), then by the round of the decision on the actual T3 path.

| split | round | hi-level A | lo-level A |
|---|---|---|---|
| T2-path round | 1 | 2576 / 3008 (85.6 %) | 1610 / 3008 (53.5 %) |
| T2-path round | 2 | 2065 / 2392 (86.3 %) | 918 / 2392 (38.4 %) |
| T2-path round | 3 | 149 / 254 (58.7 %) | 19 / 244 (7.8 %) |
| on-path round | 1 | 2576 / 3008 (85.6 %) | 1610 / 3008 (53.5 %) |
| on-path round | 2 | 1897 / 2074 (91.5 %) | 921 / 2132 (43.2 %) |
| on-path round | 3 | 317 / 572 (55.4 %) | 16 / 504 (3.2 %) |

| instance | hi-level A | lo-level A |
|---|---|---|
| 1-1-3 | 1120 / 1352 (82.8 %) | 491 / 1352 (36.3 %) |
| 1-2-2 | 840 / 882 (95.2 %) | 298 / 882 (33.8 %) |
| 1-1-4 | 530 / 882 (60.1 %) | 550 / 882 (62.4 %) |
| 1-2-3 | 342 / 392 (87.2 %) | 212 / 392 (54.1 %) |
| 2-2-2 | 882 / 882 (100.0 %) | 574 / 882 (65.1 %) |
| 1-1-5 | 603 / 648 (93.1 %) | 215 / 648 (33.2 %) |
| 1-2-4 | 149 / 254 (58.7 %) | 19 / 244 (7.8 %) |
| 1-3-3 | 180 / 200 (90.0 %) | 86 / 200 (43.0 %) |
| 2-2-3 | 144 / 162 (88.9 %) | 102 / 162 (63.0 %) |

## The 2 × 2 disclosure game (π = 1/2)

Temptation = withhold against a discloser, reward = both disclose, punishment = both withhold, sucker = disclose to a withholder.

**B0 asks by EIG**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4210 | 0.4437 | 0.3536 | 0.3763 |
| B | 0.3973 | 0.4188 | 0.3536 | 0.3751 |

**B asks by the exact program**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4210 | 0.4427 | 0.3536 | 0.3753 |
| B | 0.4210 | 0.4427 | 0.3536 | 0.3753 |

**B0 asks by EIG, common scale**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.2236 | 0.2344 | 0.1875 | 0.1983 |
| B | 0.2109 | 0.2210 | 0.1875 | 0.1976 |

## Control: the same game without HIGH (no prediction)

E(T0) 0.6008, E(T1) 0.7257, E(T2) 0.8309, E(T3) 0.8392. Y_A(T1) 0.4104, Y_A(T2) 0.4367; lo-level A: Y_A(T1) 0.4431, Y_A(T2) 0.4784.

- 2 × 2, B0 asks by EIG: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- 2 × 2, B asks by the exact program: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4104 | 0.4367 | 0.3536 | 0.3799 |
| B | 0.4006 | 0.4281 | 0.3536 | 0.3811 |

## Structures by π and A's level

Y_A and Y_B are each side's expected Y (own scale; `c` the common scale); `prop` and `resp` a side's Y when it proposes and when it responds; agreement shares by who proposed.

| bank | π | structure | A level | Y_A | Y_B | Y_A c | Y_B c | Y_A prop | Y_A resp | Y_B prop | Y_B resp | agree, A proposes | agree, B proposes | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | 1/2 | T0 no queries | all | 0.3536 | 0.3536 | 0.1875 | 0.1875 | 0.4241 | 0.2831 | 0.4241 | 0.2831 | 55.8 % | 55.8 % | 0.6008 |
| high | 1/2 | T0 no queries | lo | 0.3785 | 0.4002 | 0.2271 | 0.2120 | 0.4427 | 0.3143 | 0.4928 | 0.3077 | 59.7 % | 64.9 % | 0.6268 |
| high | 1/2 | T0 no queries | hi | 0.3287 | 0.3070 | 0.1479 | 0.1630 | 0.4055 | 0.2519 | 0.3555 | 0.2585 | 52.0 % | 46.8 % | 0.5720 |
| high | 1/2 | T1 private (A asks) | all | 0.4210 | 0.3751 | 0.2236 | 0.1976 | 0.5590 | 0.2831 | 0.4241 | 0.3261 | 81.0 % | 55.8 % | 0.7361 |
| high | 1/2 | T1 private (A asks) | lo | 0.4549 | 0.4182 | 0.2729 | 0.2202 | 0.5955 | 0.3143 | 0.4928 | 0.3436 | 84.4 % | 64.9 % | 0.7571 |
| high | 1/2 | T1 private (A asks) | hi | 0.3872 | 0.3321 | 0.1742 | 0.1751 | 0.5224 | 0.2519 | 0.3555 | 0.3087 | 77.7 % | 46.8 % | 0.7128 |
| high | 1/2 | T2 public (both ask, both answer) | all | 0.4437 | 0.4188 | 0.2344 | 0.2210 | 0.5590 | 0.3284 | 0.5114 | 0.3261 | 81.0 % | 73.5 % | 0.8299 |
| high | 1/2 | T2 public (both ask, both answer) | lo | 0.4627 | 0.4667 | 0.2776 | 0.2461 | 0.5955 | 0.3299 | 0.5898 | 0.3436 | 84.4 % | 80.4 % | 0.8301 |
| high | 1/2 | T2 public (both ask, both answer) | hi | 0.4247 | 0.3709 | 0.1911 | 0.1959 | 0.5224 | 0.3269 | 0.4331 | 0.3087 | 77.7 % | 66.5 % | 0.8299 |
| high | 1/2 | T3 reciprocal (A chooses) | all | 0.4684 | 0.4077 | 0.2479 | 0.2155 | 0.5499 | 0.3869 | 0.4863 | 0.3291 | 79.4 % | 74.9 % | 0.8387 |
| high | 1/2 | T3 reciprocal (A chooses) | lo | 0.4950 | 0.4513 | 0.2970 | 0.2384 | 0.5832 | 0.4068 | 0.5503 | 0.3524 | 82.8 % | 81.5 % | 0.8421 |
| high | 1/2 | T3 reciprocal (A chooses) | hi | 0.4418 | 0.3640 | 0.1988 | 0.1926 | 0.5166 | 0.3670 | 0.4223 | 0.3058 | 76.0 % | 68.3 % | 0.8353 |
| high | 1/4 | T0 no queries | all | 0.3184 | 0.3889 | 0.1692 | 0.2058 | 0.4241 | 0.2831 | 0.4241 | 0.2831 | 55.8 % | 55.8 % | 0.6008 |
| high | 1/4 | T0 no queries | lo | 0.3464 | 0.4465 | 0.2078 | 0.2360 | 0.4427 | 0.3143 | 0.4928 | 0.3077 | 59.7 % | 64.9 % | 0.6405 |
| high | 1/4 | T0 no queries | hi | 0.2903 | 0.3312 | 0.1306 | 0.1756 | 0.4055 | 0.2519 | 0.3555 | 0.2585 | 52.0 % | 46.8 % | 0.5565 |
| high | 1/4 | T1 private (A asks) | all | 0.3521 | 0.3996 | 0.1873 | 0.2108 | 0.5590 | 0.2831 | 0.4241 | 0.3261 | 81.0 % | 55.8 % | 0.6685 |
| high | 1/4 | T1 private (A asks) | lo | 0.3846 | 0.4555 | 0.2307 | 0.2400 | 0.5955 | 0.3143 | 0.4928 | 0.3436 | 84.4 % | 64.9 % | 0.7057 |
| high | 1/4 | T1 private (A asks) | hi | 0.3196 | 0.3438 | 0.1438 | 0.1817 | 0.5224 | 0.2519 | 0.3555 | 0.3087 | 77.7 % | 46.8 % | 0.6269 |
| high | 1/4 | T2 public (both ask, both answer) | all | 0.3861 | 0.4651 | 0.2034 | 0.2459 | 0.5590 | 0.3284 | 0.5114 | 0.3261 | 81.0 % | 73.5 % | 0.8091 |
| high | 1/4 | T2 public (both ask, both answer) | lo | 0.3963 | 0.5282 | 0.2378 | 0.2790 | 0.5955 | 0.3299 | 0.5898 | 0.3436 | 84.4 % | 80.4 % | 0.8151 |
| high | 1/4 | T2 public (both ask, both answer) | hi | 0.3758 | 0.4020 | 0.1691 | 0.2129 | 0.5224 | 0.3269 | 0.4331 | 0.3087 | 77.7 % | 66.5 % | 0.8026 |
| high | 1/4 | T3 reciprocal (A chooses) | all | 0.4296 | 0.4432 | 0.2273 | 0.2344 | 0.5394 | 0.3930 | 0.4818 | 0.3274 | 77.3 % | 75.4 % | 0.8279 |
| high | 1/4 | T3 reciprocal (A chooses) | lo | 0.4535 | 0.4950 | 0.2721 | 0.2615 | 0.5678 | 0.4154 | 0.5430 | 0.3509 | 79.9 % | 81.8 % | 0.8348 |
| high | 1/4 | T3 reciprocal (A chooses) | hi | 0.4056 | 0.3915 | 0.1825 | 0.2074 | 0.5109 | 0.3705 | 0.4207 | 0.3039 | 74.7 % | 69.0 % | 0.8205 |
| high | 3/4 | T0 no queries | all | 0.3889 | 0.3184 | 0.2058 | 0.1692 | 0.4241 | 0.2831 | 0.4241 | 0.2831 | 55.8 % | 55.8 % | 0.6008 |
| high | 3/4 | T0 no queries | lo | 0.4106 | 0.3540 | 0.2464 | 0.1880 | 0.4427 | 0.3143 | 0.4928 | 0.3077 | 59.7 % | 64.9 % | 0.6131 |
| high | 3/4 | T0 no queries | hi | 0.3671 | 0.2827 | 0.1652 | 0.1504 | 0.4055 | 0.2519 | 0.3555 | 0.2585 | 52.0 % | 46.8 % | 0.5875 |
| high | 3/4 | T1 private (A asks) | all | 0.4900 | 0.3506 | 0.2599 | 0.1844 | 0.5590 | 0.2831 | 0.4241 | 0.3261 | 81.0 % | 55.8 % | 0.8038 |
| high | 3/4 | T1 private (A asks) | lo | 0.5252 | 0.3809 | 0.3151 | 0.2003 | 0.5955 | 0.3143 | 0.4928 | 0.3436 | 84.4 % | 64.9 % | 0.8086 |
| high | 3/4 | T1 private (A asks) | hi | 0.4548 | 0.3204 | 0.2047 | 0.1686 | 0.5224 | 0.2519 | 0.3555 | 0.3087 | 77.7 % | 46.8 % | 0.7987 |
| high | 3/4 | T2 public (both ask, both answer) | all | 0.5013 | 0.3725 | 0.2653 | 0.1961 | 0.5590 | 0.3284 | 0.5114 | 0.3261 | 81.0 % | 73.5 % | 0.8507 |
| high | 3/4 | T2 public (both ask, both answer) | lo | 0.5291 | 0.4051 | 0.3175 | 0.2133 | 0.5955 | 0.3299 | 0.5898 | 0.3436 | 84.4 % | 80.4 % | 0.8451 |
| high | 3/4 | T2 public (both ask, both answer) | hi | 0.4735 | 0.3398 | 0.2131 | 0.1790 | 0.5224 | 0.3269 | 0.4331 | 0.3087 | 77.7 % | 66.5 % | 0.8572 |
| high | 3/4 | T3 reciprocal (A chooses) | all | 0.5113 | 0.3677 | 0.2708 | 0.1938 | 0.5566 | 0.3754 | 0.4913 | 0.3265 | 80.5 % | 74.1 % | 0.8529 |
| high | 3/4 | T3 reciprocal (A chooses) | lo | 0.5421 | 0.3987 | 0.3253 | 0.2101 | 0.5923 | 0.3914 | 0.5578 | 0.3457 | 83.9 % | 80.4 % | 0.8486 |
| high | 3/4 | T3 reciprocal (A chooses) | hi | 0.4805 | 0.3368 | 0.2162 | 0.1776 | 0.5209 | 0.3593 | 0.4248 | 0.3074 | 77.2 % | 67.7 % | 0.8581 |
| no_high | 1/2 | T0 no queries | all | 0.3536 | 0.3536 | 0.1875 | 0.1875 | 0.4241 | 0.2831 | 0.4241 | 0.2831 | 55.8 % | 55.8 % | 0.6008 |
| no_high | 1/2 | T0 no queries | lo | 0.3785 | 0.4002 | 0.2271 | 0.2120 | 0.4427 | 0.3143 | 0.4928 | 0.3077 | 59.7 % | 64.9 % | 0.6268 |
| no_high | 1/2 | T0 no queries | hi | 0.3287 | 0.3070 | 0.1479 | 0.1630 | 0.4055 | 0.2519 | 0.3555 | 0.2585 | 52.0 % | 46.8 % | 0.5720 |
| no_high | 1/2 | T1 private (A asks) | all | 0.4104 | 0.3811 | 0.2179 | 0.2028 | 0.5376 | 0.2831 | 0.4241 | 0.3381 | 78.0 % | 55.8 % | 0.7257 |
| no_high | 1/2 | T1 private (A asks) | lo | 0.4431 | 0.4278 | 0.2659 | 0.2273 | 0.5720 | 0.3143 | 0.4928 | 0.3627 | 81.6 % | 64.9 % | 0.7492 |
| no_high | 1/2 | T1 private (A asks) | hi | 0.3776 | 0.3345 | 0.1699 | 0.1782 | 0.5032 | 0.2519 | 0.3555 | 0.3134 | 74.3 % | 46.8 % | 0.6997 |
| no_high | 1/2 | T2 public (both ask, both answer) | all | 0.4367 | 0.4281 | 0.2324 | 0.2279 | 0.5376 | 0.3357 | 0.5180 | 0.3381 | 78.0 % | 74.6 % | 0.8309 |
| no_high | 1/2 | T2 public (both ask, both answer) | lo | 0.4784 | 0.4767 | 0.2870 | 0.2533 | 0.5720 | 0.3848 | 0.5907 | 0.3627 | 81.6 % | 86.2 % | 0.8625 |
| no_high | 1/2 | T2 public (both ask, both answer) | hi | 0.3950 | 0.3794 | 0.1777 | 0.2025 | 0.5032 | 0.2867 | 0.4454 | 0.3134 | 74.3 % | 62.9 % | 0.7957 |
| no_high | 1/2 | T3 reciprocal (A chooses) | all | 0.4641 | 0.4118 | 0.2467 | 0.2191 | 0.5338 | 0.3944 | 0.4886 | 0.3349 | 76.7 % | 76.8 % | 0.8392 |
| no_high | 1/2 | T3 reciprocal (A chooses) | lo | 0.5044 | 0.4529 | 0.3026 | 0.2405 | 0.5684 | 0.4404 | 0.5445 | 0.3614 | 80.8 % | 86.3 % | 0.8620 |
| no_high | 1/2 | T3 reciprocal (A chooses) | hi | 0.4239 | 0.3706 | 0.1907 | 0.1977 | 0.4992 | 0.3485 | 0.4328 | 0.3084 | 72.6 % | 67.3 % | 0.8139 |
| no_high | 1/4 | T0 no queries | all | 0.3184 | 0.3889 | 0.1692 | 0.2058 | 0.4241 | 0.2831 | 0.4241 | 0.2831 | 55.8 % | 55.8 % | 0.6008 |
| no_high | 1/4 | T0 no queries | lo | 0.3464 | 0.4465 | 0.2078 | 0.2360 | 0.4427 | 0.3143 | 0.4928 | 0.3077 | 59.7 % | 64.9 % | 0.6405 |
| no_high | 1/4 | T0 no queries | hi | 0.2903 | 0.3312 | 0.1306 | 0.1756 | 0.4055 | 0.2519 | 0.3555 | 0.2585 | 52.0 % | 46.8 % | 0.5565 |
| no_high | 1/4 | T1 private (A asks) | all | 0.3467 | 0.4026 | 0.1844 | 0.2134 | 0.5376 | 0.2831 | 0.4241 | 0.3381 | 78.0 % | 55.8 % | 0.6633 |
| no_high | 1/4 | T1 private (A asks) | lo | 0.3787 | 0.4603 | 0.2272 | 0.2436 | 0.5720 | 0.3143 | 0.4928 | 0.3627 | 81.6 % | 64.9 % | 0.7017 |
| no_high | 1/4 | T1 private (A asks) | hi | 0.3148 | 0.3450 | 0.1416 | 0.1832 | 0.5032 | 0.2519 | 0.3555 | 0.3134 | 74.3 % | 46.8 % | 0.6204 |
| no_high | 1/4 | T2 public (both ask, both answer) | all | 0.3862 | 0.4731 | 0.2062 | 0.2511 | 0.5376 | 0.3357 | 0.5180 | 0.3381 | 78.0 % | 74.6 % | 0.8211 |
| no_high | 1/4 | T2 public (both ask, both answer) | lo | 0.4316 | 0.5337 | 0.2589 | 0.2826 | 0.5720 | 0.3848 | 0.5907 | 0.3627 | 81.6 % | 86.2 % | 0.8717 |
| no_high | 1/4 | T2 public (both ask, both answer) | hi | 0.3408 | 0.4124 | 0.1534 | 0.2197 | 0.5032 | 0.2867 | 0.4454 | 0.3134 | 74.3 % | 62.9 % | 0.7644 |
| no_high | 1/4 | T3 reciprocal (A chooses) | all | 0.4306 | 0.4475 | 0.2293 | 0.2372 | 0.5276 | 0.3982 | 0.4860 | 0.3317 | 75.6 % | 77.1 % | 0.8389 |
| no_high | 1/4 | T3 reciprocal (A chooses) | lo | 0.4739 | 0.4942 | 0.2843 | 0.2613 | 0.5612 | 0.4448 | 0.5391 | 0.3596 | 80.0 % | 86.3 % | 0.8715 |
| no_high | 1/4 | T3 reciprocal (A chooses) | hi | 0.3873 | 0.4007 | 0.1743 | 0.2131 | 0.4940 | 0.3517 | 0.4330 | 0.3039 | 71.2 % | 67.9 % | 0.8024 |
| no_high | 3/4 | T0 no queries | all | 0.3889 | 0.3184 | 0.2058 | 0.1692 | 0.4241 | 0.2831 | 0.4241 | 0.2831 | 55.8 % | 55.8 % | 0.6008 |
| no_high | 3/4 | T0 no queries | lo | 0.4106 | 0.3540 | 0.2464 | 0.1880 | 0.4427 | 0.3143 | 0.4928 | 0.3077 | 59.7 % | 64.9 % | 0.6131 |
| no_high | 3/4 | T0 no queries | hi | 0.3671 | 0.2827 | 0.1652 | 0.1504 | 0.4055 | 0.2519 | 0.3555 | 0.2585 | 52.0 % | 46.8 % | 0.5875 |
| no_high | 3/4 | T1 private (A asks) | all | 0.4740 | 0.3596 | 0.2514 | 0.1921 | 0.5376 | 0.2831 | 0.4241 | 0.3381 | 78.0 % | 55.8 % | 0.7882 |
| no_high | 3/4 | T1 private (A asks) | lo | 0.5076 | 0.3952 | 0.3045 | 0.2110 | 0.5720 | 0.3143 | 0.4928 | 0.3627 | 81.6 % | 64.9 % | 0.7967 |
| no_high | 3/4 | T1 private (A asks) | hi | 0.4404 | 0.3239 | 0.1982 | 0.1732 | 0.5032 | 0.2519 | 0.3555 | 0.3134 | 74.3 % | 46.8 % | 0.7791 |
| no_high | 3/4 | T2 public (both ask, both answer) | all | 0.4871 | 0.3831 | 0.2586 | 0.2047 | 0.5376 | 0.3357 | 0.5180 | 0.3381 | 78.0 % | 74.6 % | 0.8408 |
| no_high | 3/4 | T2 public (both ask, both answer) | lo | 0.5252 | 0.4197 | 0.3151 | 0.2240 | 0.5720 | 0.3848 | 0.5907 | 0.3627 | 81.6 % | 86.2 % | 0.8533 |
| no_high | 3/4 | T2 public (both ask, both answer) | hi | 0.4491 | 0.3464 | 0.2021 | 0.1854 | 0.5032 | 0.2867 | 0.4454 | 0.3134 | 74.3 % | 62.9 % | 0.8271 |
| no_high | 3/4 | T3 reciprocal (A chooses) | all | 0.4999 | 0.3759 | 0.2653 | 0.2008 | 0.5365 | 0.3902 | 0.4895 | 0.3380 | 77.7 % | 76.3 % | 0.8447 |
| no_high | 3/4 | T3 reciprocal (A chooses) | lo | 0.5374 | 0.4087 | 0.3224 | 0.2180 | 0.5711 | 0.4362 | 0.5476 | 0.3623 | 81.4 % | 86.1 % | 0.8533 |
| no_high | 3/4 | T3 reciprocal (A chooses) | hi | 0.4625 | 0.3431 | 0.2081 | 0.1836 | 0.5020 | 0.3443 | 0.4315 | 0.3137 | 73.9 % | 66.4 % | 0.8354 |

## D3′ by π (B0 reading)

| bank | π | A in dilemma | B in dilemma |
|---|---|---|---|
| high | 1/2 | False | False |
| high | 1/4 | False | False |
| high | 3/4 | False | False |
| no_high | 1/2 | False | False |
| no_high | 1/4 | False | False |
| no_high | 3/4 | False | False |

Not built here: B2 (designed after these sweeps, its predictions registered before it runs; decision 7) and the CFR exploitability check.
