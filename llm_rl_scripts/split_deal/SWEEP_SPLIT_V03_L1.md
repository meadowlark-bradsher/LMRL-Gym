# Split Deal v0.3 — L1 words-value sweep and gate SD3-W

_Rendered by `sweep_l1_v03.py render` from `results/split_v03_l1_20261110/sweep.json` (sweep `split-deal-v03-l1-v0.1`, commit `3a9237f`, 74.6 s). Exact on the model: no sampling, no API; every number is formatted from the JSON. Gate as registered in `DESIGN_SPLIT_V03.md` §5 before this ran; its operational reading is fixed in the docstring of `sweep_l1_v03.py`._

Protocol 4.1 on the v0.3 game: each side's type is its item values and a private outside-option level (lo: d = 4, hi: d = 5.5, ½ each). A asks up to Q of the 13 bank questions (HIGH included), answered truthfully, then makes one binding offer or walks; B accepts iff its utility is at least its own d. Y is net gain over one's own d as a share of 10 − one's own d, averaged over both types uniformly within an instance, then over the nine instances with equal weight.

## Gate SD3-W

W = D_words(3) = **0.1348** Y (exact 7814985461/57953750400).

Bars: go at W ≥ 0.10 Y, conditional from 0.08 Y.

**Verdict: go** — v0.3 keeps a words value worth a live study (its registration still needs its own power calculation, NEG-33).

Re-offer check (H = 4, h = 0): asks-and-offers minus offers-only = 0.0056 Y against the 0.04 Y line. Below it, and the gate did not fail: as in v0.2, the words matter *because* offers are scarce, so any live protocol keeps the single-offer rule and any write-up says the result is conditional on it.

Diagnostics, deciding nothing: on the common scale (U − own d)/10, W = 0.0721; in the control bank without HIGH, W = 0.1135. v0.2 gave 0.1659 (easy) and 0.1063 (hard) (`SWEEP_SPLIT_L1.md`).

## W by level pair (diagnostic)

From the exact policy's realized gains against each B type, Q = 3 minus Q = 0, ask cost 0, independent prior.

| level pair | V_none | W (own scale) | W (common scale) | W without HIGH |
|---|---|---|---|---|
| A lo, B lo | 0.5091 | 0.1626 | 0.0976 | 0.1303 |
| A lo, B hi | 0.3763 | 0.1430 | 0.0858 | 0.1283 |
| A hi, B lo | 0.4764 | 0.1309 | 0.0589 | 0.1035 |
| A hi, B hi | 0.3347 | 0.1029 | 0.0463 | 0.0919 |

## By Q, ask cost 0 — 13-question bank, independent uniform prior

V_reveal = 0.5773.

| Q | V_words | D_words | D_words common scale | D_rest | questions / ep | asks HIGH | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4241 | 0.0000 | 0.0000 | 0.1532 | 0.00 | 0.0 % | 0.0 % | 55.8 % | 0.601 | 0.643 |
| 1 | 0.5046 | 0.0804 | 0.0427 | 0.0727 | 0.86 | 0.0 % | 0.0 % | 67.1 % | 0.735 | 0.654 |
| 2 | 0.5398 | 0.1157 | 0.0619 | 0.0375 | 1.74 | 21.6 % | 0.0 % | 77.4 % | 0.840 | 0.653 |
| 3 | 0.5590 | 0.1348 | 0.0721 | 0.0183 | 2.56 | 50.9 % | 0.0 % | 81.1 % | 0.872 | 0.668 |
| 4 | 0.5692 | 0.1451 | 0.0776 | 0.0081 | 3.22 | 52.8 % | 0.1 % | 83.3 % | 0.894 | 0.677 |
| 5 | 0.5700 | 0.1459 | 0.0781 | 0.0073 | 3.52 | 38.5 % | 0.7 % | 83.6 % | 0.897 | 0.678 |
| 6 | 0.5700 | 0.1459 | 0.0781 | 0.0073 | 3.67 | 31.7 % | 0.9 % | 83.6 % | 0.897 | 0.677 |

## By Q, ask cost 0 — 13-question bank, value-pair constraints (sensitivity prior; never gates)

V_reveal = 0.6513.

| Q | V_words | D_words | D_words common scale | D_rest | questions / ep | asks HIGH | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4754 | 0.0000 | 0.0000 | 0.1759 | 0.00 | 0.0 % | 0.0 % | 59.5 % | 0.586 | 0.695 |
| 1 | 0.5667 | 0.0913 | 0.0486 | 0.0846 | 0.89 | 6.4 % | 0.0 % | 73.8 % | 0.735 | 0.693 |
| 2 | 0.6065 | 0.1311 | 0.0702 | 0.0448 | 1.75 | 18.8 % | 0.0 % | 84.9 % | 0.843 | 0.684 |
| 3 | 0.6307 | 0.1553 | 0.0832 | 0.0205 | 2.62 | 53.6 % | 0.0 % | 89.3 % | 0.879 | 0.702 |
| 4 | 0.6423 | 0.1670 | 0.0894 | 0.0089 | 3.19 | 57.1 % | 0.0 % | 91.9 % | 0.903 | 0.711 |
| 5 | 0.6432 | 0.1678 | 0.0899 | 0.0081 | 3.50 | 45.6 % | 0.1 % | 92.2 % | 0.907 | 0.711 |
| 6 | 0.6432 | 0.1678 | 0.0899 | 0.0081 | 3.61 | 39.5 % | 0.3 % | 92.3 % | 0.907 | 0.710 |

## By Q, ask cost 0 — control, 12 questions without HIGH, independent uniform prior

V_reveal = 0.5773.

| Q | V_words | D_words | D_words common scale | D_rest | questions / ep | asks HIGH | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4241 | 0.0000 | 0.0000 | 0.1532 | 0.00 | — | 0.0 % | 55.8 % | 0.601 | 0.643 |
| 1 | 0.5046 | 0.0804 | 0.0427 | 0.0727 | 0.86 | — | 0.0 % | 67.1 % | 0.735 | 0.654 |
| 2 | 0.5320 | 0.1079 | 0.0578 | 0.0453 | 1.70 | — | 0.0 % | 76.2 % | 0.832 | 0.644 |
| 3 | 0.5376 | 0.1135 | 0.0608 | 0.0397 | 2.46 | — | 0.2 % | 77.9 % | 0.850 | 0.647 |
| 4 | 0.5383 | 0.1142 | 0.0611 | 0.0390 | 2.88 | — | 0.3 % | 78.2 % | 0.854 | 0.646 |
| 5 | 0.5383 | 0.1142 | 0.0611 | 0.0390 | 3.07 | — | 0.4 % | 78.3 % | 0.855 | 0.646 |
| 6 | 0.5383 | 0.1142 | 0.0611 | 0.0390 | 3.10 | — | 0.4 % | 78.4 % | 0.856 | 0.646 |

## By Q, ask cost 0 — control, 12 questions without HIGH, value-pair constraints (sensitivity prior; never gates)

V_reveal = 0.6513.

| Q | V_words | D_words | D_words common scale | D_rest | questions / ep | asks HIGH | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4754 | 0.0000 | 0.0000 | 0.1759 | 0.00 | — | 0.0 % | 59.5 % | 0.586 | 0.695 |
| 1 | 0.5610 | 0.0856 | 0.0456 | 0.0903 | 0.86 | — | 0.0 % | 73.0 % | 0.729 | 0.689 |
| 2 | 0.5918 | 0.1164 | 0.0624 | 0.0595 | 1.67 | — | 0.0 % | 83.0 % | 0.829 | 0.670 |
| 3 | 0.5984 | 0.1230 | 0.0660 | 0.0529 | 2.32 | — | 0.0 % | 85.1 % | 0.851 | 0.672 |
| 4 | 0.5992 | 0.1239 | 0.0664 | 0.0520 | 2.69 | — | 0.0 % | 85.4 % | 0.853 | 0.673 |
| 5 | 0.5992 | 0.1239 | 0.0664 | 0.0520 | 2.84 | — | 0.1 % | 85.5 % | 0.855 | 0.672 |
| 6 | 0.5992 | 0.1239 | 0.0664 | 0.0520 | 2.87 | — | 0.1 % | 85.5 % | 0.855 | 0.672 |

`asks HIGH` is the share of episodes in which the exact policy asks HIGH. `walk` and `agree` are shares of episodes; `mean E` is joint efficiency S(o)/S* over pairs with S* > 0 (no deal counts 0) and `mean C_A` A's share of the surplus over agreements. Ties go to not asking.

## Ask cost (13-question bank, independent prior)

| cost (Y) | D_words(3) | questions / ep at Q = 3 | D_words(6) |
|---|---|---|---|
| 0 | 0.1348 | 2.56 | 0.1459 |
| 1/100 | 0.1126 | 2.16 | 0.1224 |
| 3/100 | 0.0753 | 1.61 | 0.0811 |

## Greedy gap, ask cost 0 (13-question bank, independent prior)

| Q | exact | EIG-greedy | value-greedy | gap EIG | gap value |
|---|---|---|---|---|---|
| 1 | 0.5046 | 0.4453 | 0.5046 | 0.0593 | 0.0000 |
| 2 | 0.5398 | 0.4797 | 0.5316 | 0.0601 | 0.0082 |
| 3 | 0.5590 | 0.5114 | 0.5463 | 0.0475 | 0.0126 |
| 4 | 0.5692 | 0.5417 | 0.5597 | 0.0275 | 0.0095 |
| 5 | 0.5700 | 0.5659 | 0.5671 | 0.0041 | 0.0029 |
| 6 | 0.5700 | 0.5692 | 0.5692 | 0.0008 | 0.0008 |

## Re-offer variant (H = 4, ask cost 0)

| bank | prior | h | offers only | asks and offers | D |
|---|---|---|---|---|---|
| high | independent | 0 | 0.5687 | 0.5743 | 0.0056 |
| high | independent | 1/10 | 0.5415 | 0.5678 | 0.0264 |
| high | independent | 1/4 | 0.5089 | 0.5631 | 0.0542 |
| high | constrained | 0 | 0.6389 | 0.6473 | 0.0083 |
| high | constrained | 1/10 | 0.6122 | 0.6402 | 0.0279 |
| high | constrained | 1/4 | 0.5773 | 0.6352 | 0.0578 |
| no_high | independent | 0 | 0.5687 | 0.5743 | 0.0056 |
| no_high | independent | 1/10 | 0.5415 | 0.5647 | 0.0232 |
| no_high | independent | 1/4 | 0.5089 | 0.5552 | 0.0463 |
| no_high | constrained | 0 | 0.6389 | 0.6473 | 0.0083 |
| no_high | constrained | 1/10 | 0.6122 | 0.6348 | 0.0226 |
| no_high | constrained | 1/4 | 0.5773 | 0.6222 | 0.0448 |

## By instance (ask cost 0, independent prior)

| instance | types | W | W without HIGH | D_rest(3) | asks HIGH at Q = 3 |
|---|---|---|---|---|---|
| 1-1-3 | 52 | 0.1216 | 0.1136 | 0.0168 | 26.0 % |
| 1-2-2 | 42 | 0.1432 | 0.1235 | 0.0241 | 65.3 % |
| 1-1-4 | 42 | 0.1134 | 0.0919 | 0.0129 | 59.1 % |
| 2-2-2 | 42 | 0.1542 | 0.1325 | 0.0402 | 70.5 % |
| 1-1-5 | 36 | 0.0971 | 0.0708 | 0.0123 | 37.3 % |
| 1-2-3 | 28 | 0.1427 | 0.1233 | 0.0150 | 57.9 % |
| 1-2-4 | 24 | 0.1496 | 0.1134 | 0.0083 | 71.2 % |
| 1-3-3 | 20 | 0.1503 | 0.1300 | 0.0117 | 49.5 % |
| 2-2-3 | 18 | 0.1416 | 0.1224 | 0.0237 | 21.0 % |

Spearman rank correlation of W with type count over the nine instances: -0.22 (descriptive).
