# Split Deal — L1 words-value sweep and gate SD-W

_Rendered by `sweep_l1.py render` from `results/split_l1_20261110/sweep.json` (sweep `split-deal-l1-v0.1`, commit `9125f52`, 13.5 s). Exact on the model: no sampling, no API; every number is formatted from the JSON. Gate and configuration as registered in `DESIGN_SPLIT.md` §7 (spec v0.2) before this ran._

Protocol 4.1: A asks up to Q bank questions, answered truthfully, then makes one binding offer or walks; B accepts iff its utility is at least d. Values in Y (net gain over d as a share of 10 − d), averaged over the joint prior of the two types within an instance, then over the nine instances with equal weight. `D_words(Q) = V_words(Q) − V_none`; `D_rest = V_reveal − V_words(Q)`.

## Gate SD-W

| regime | W = D_words(3) | exact | tier |
|---|---|---|---|
| easy | 0.1659 | 801278921/4829479200 | go |
| hard | 0.1063 | 164943061/1552332600 | go |

Bars: go at W ≥ 0.10 Y, conditional from 0.08 Y. Primary regime: **easy**.

**Verdict: go** — a live study on Split Deal is worth designing (its registration still needs its own power calculation, NEG-33).

Re-offer check (H = 4, h = 0, easy): asks-and-offers minus offers-only = 0.0035 Y against the 0.04 Y line. Below it, and the gate did not fail: the words matter *because* offers are scarce, so the live protocol must keep the single-offer rule and any write-up must say the result is conditional on it.

## By Q, ask cost 0 — independent uniform prior (primary)

**easy** (V_reveal = 0.7016)

| Q | V_words | D_words | D_rest | questions / ep | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5177 | 0.0000 | 0.1838 | 0.00 | 0.0 % | 68.0 % | 0.652 | 0.673 |
| 1 | 0.6219 | 0.1041 | 0.0797 | 0.87 | 0.0 % | 80.1 % | 0.789 | 0.700 |
| 2 | 0.6717 | 0.1539 | 0.0299 | 1.75 | 0.0 % | 89.7 % | 0.875 | 0.716 |
| 3 | 0.6837 | 0.1659 | 0.0179 | 2.55 | 0.0 % | 90.7 % | 0.880 | 0.732 |
| 4 | 0.6844 | 0.1667 | 0.0172 | 2.98 | 0.2 % | 90.8 % | 0.881 | 0.733 |
| 5 | 0.6844 | 0.1667 | 0.0172 | 3.13 | 0.2 % | 90.7 % | 0.881 | 0.733 |
| 6 | 0.6844 | 0.1667 | 0.0172 | 3.16 | 0.2 % | 90.7 % | 0.880 | 0.734 |

**hard** (V_reveal = 0.4479)

| Q | V_words | D_words | D_rest | questions / ep | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3413 | 0.0000 | 0.1066 | 0.00 | 0.0 % | 46.8 % | 0.611 | 0.606 |
| 1 | 0.4145 | 0.0732 | 0.0334 | 0.85 | 0.0 % | 60.3 % | 0.785 | 0.611 |
| 2 | 0.4384 | 0.0971 | 0.0095 | 1.63 | 0.0 % | 71.3 % | 0.919 | 0.595 |
| 3 | 0.4475 | 0.1063 | 0.0003 | 2.30 | 0.5 % | 74.5 % | 0.958 | 0.597 |
| 4 | 0.4479 | 0.1066 | 0.0000 | 2.67 | 1.3 % | 74.8 % | 0.961 | 0.597 |
| 5 | 0.4479 | 0.1066 | 0.0000 | 2.83 | 1.8 % | 74.8 % | 0.960 | 0.597 |
| 6 | 0.4479 | 0.1066 | 0.0000 | 2.85 | 1.9 % | 74.8 % | 0.960 | 0.597 |

## By Q, ask cost 0 — value-pair constraints (sensitivity prior; never gates)

**easy** (V_reveal = 0.7809)

| Q | V_words | D_words | D_rest | questions / ep | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6053 | 0.0000 | 0.1755 | 0.00 | 0.0 % | 73.5 % | 0.674 | 0.733 |
| 1 | 0.7013 | 0.0960 | 0.0795 | 0.81 | 0.0 % | 87.0 % | 0.806 | 0.747 |
| 2 | 0.7516 | 0.1462 | 0.0293 | 1.57 | 0.0 % | 96.2 % | 0.892 | 0.746 |
| 3 | 0.7632 | 0.1578 | 0.0177 | 2.18 | 0.0 % | 96.6 % | 0.888 | 0.769 |
| 4 | 0.7638 | 0.1584 | 0.0171 | 2.54 | 0.0 % | 96.4 % | 0.885 | 0.772 |
| 5 | 0.7638 | 0.1584 | 0.0171 | 2.68 | 0.0 % | 96.3 % | 0.884 | 0.773 |
| 6 | 0.7638 | 0.1584 | 0.0171 | 2.70 | 0.0 % | 96.3 % | 0.884 | 0.774 |

**hard** (V_reveal = 0.5058)

| Q | V_words | D_words | D_rest | questions / ep | walk | agree | mean E | mean C_A |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3714 | 0.0000 | 0.1344 | 0.00 | 0.4 % | 50.6 % | 0.558 | 0.651 |
| 1 | 0.4606 | 0.0892 | 0.0452 | 0.87 | 0.4 % | 66.6 % | 0.744 | 0.651 |
| 2 | 0.4946 | 0.1232 | 0.0113 | 1.71 | 0.4 % | 81.0 % | 0.897 | 0.628 |
| 3 | 0.5055 | 0.1341 | 0.0004 | 2.34 | 0.4 % | 85.4 % | 0.946 | 0.630 |
| 4 | 0.5058 | 0.1344 | 0.0000 | 2.74 | 1.4 % | 85.8 % | 0.950 | 0.629 |
| 5 | 0.5058 | 0.1344 | 0.0000 | 2.84 | 1.4 % | 85.8 % | 0.949 | 0.629 |
| 6 | 0.5058 | 0.1344 | 0.0000 | 2.86 | 1.9 % | 85.8 % | 0.949 | 0.629 |

`walk` and `agree` are shares of episodes under the exact policy; `mean E` is joint efficiency S(o)/S* over pairs with S* > 0 (no deal counts 0) and `mean C_A` A's share of the surplus over agreements, both from the L0 oracle (`frontier.py`). Questions per episode: ties go to not asking.

## Ask cost (independent prior)

| regime | cost (Y) | D_words(3) | questions / ep at Q = 3 | D_words(6) |
|---|---|---|---|---|
| easy | 0 | 0.1659 | 2.55 | 0.1667 |
| easy | 1/100 | 0.1481 | 1.77 | 0.1491 |
| easy | 3/100 | 0.1146 | 1.59 | 0.1159 |
| hard | 0 | 0.1063 | 2.30 | 0.1066 |
| hard | 1/100 | 0.0898 | 1.62 | 0.0906 |
| hard | 3/100 | 0.0602 | 1.32 | 0.0611 |

## Greedy gap, ask cost 0 (independent prior)

EIG-greedy (T4) asks the question with the most expected information about B's type; value-greedy the one with the largest one-step gain in offer value. Both ask while a question splits the posterior and then make the optimal offer. The gap is exact (T5) minus greedy.

| regime | Q | exact | EIG-greedy | value-greedy | gap EIG | gap value |
|---|---|---|---|---|---|---|
| easy | 1 | 0.6219 | 0.5796 | 0.6219 | 0.0423 | 0.0000 |
| easy | 2 | 0.6717 | 0.6192 | 0.6643 | 0.0524 | 0.0074 |
| easy | 3 | 0.6837 | 0.6520 | 0.6785 | 0.0316 | 0.0051 |
| easy | 4 | 0.6844 | 0.6799 | 0.6828 | 0.0045 | 0.0016 |
| easy | 5 | 0.6844 | 0.6833 | 0.6843 | 0.0012 | 0.0001 |
| easy | 6 | 0.6844 | 0.6844 | 0.6844 | 0.0000 | 0.0001 |
| hard | 1 | 0.4145 | 0.3748 | 0.4145 | 0.0397 | 0.0000 |
| hard | 2 | 0.4384 | 0.4005 | 0.4351 | 0.0379 | 0.0033 |
| hard | 3 | 0.4475 | 0.4263 | 0.4425 | 0.0212 | 0.0050 |
| hard | 4 | 0.4479 | 0.4448 | 0.4466 | 0.0031 | 0.0012 |
| hard | 5 | 0.4479 | 0.4474 | 0.4476 | 0.0004 | 0.0003 |
| hard | 6 | 0.4479 | 0.4479 | 0.4479 | 0.0000 | 0.0000 |

## Re-offer variant (H = 4, ask cost 0)

Each move is an ask, a binding offer or a walk; a rejected offer continues with probability 1 − h. `D` is asks-and-offers minus offers-only, the words value when rejections are tests.

| prior | regime | h | offers only | asks and offers | D |
|---|---|---|---|---|---|
| independent | easy | 0 | 0.6975 | 0.7009 | 0.0035 |
| independent | easy | 1/10 | 0.6640 | 0.6963 | 0.0323 |
| independent | easy | 1/4 | 0.6238 | 0.6920 | 0.0682 |
| independent | hard | 0 | 0.4443 | 0.4479 | 0.0035 |
| independent | hard | 1/10 | 0.4249 | 0.4477 | 0.0228 |
| independent | hard | 1/4 | 0.4015 | 0.4476 | 0.0461 |
| constrained | easy | 0 | 0.7752 | 0.7800 | 0.0047 |
| constrained | easy | 1/10 | 0.7465 | 0.7750 | 0.0285 |
| constrained | easy | 1/4 | 0.7094 | 0.7706 | 0.0612 |
| constrained | hard | 0 | 0.5011 | 0.5058 | 0.0048 |
| constrained | hard | 1/10 | 0.4798 | 0.5057 | 0.0259 |
| constrained | hard | 1/4 | 0.4517 | 0.5056 | 0.0538 |

## Words value against type count (easy, ask cost 0, independent prior)

| instance | types | W = D_words(3) | W hard | D_rest(3) easy |
|---|---|---|---|---|
| 1-1-3 | 26 | 0.1568 | 0.1049 | 0.0067 |
| 1-2-2 | 21 | 0.1780 | 0.1136 | 0.0257 |
| 1-1-4 | 21 | 0.1534 | 0.0814 | 0.0064 |
| 2-2-2 | 21 | 0.1772 | 0.1293 | 0.0608 |
| 1-1-5 | 18 | 0.1301 | 0.0645 | 0.0082 |
| 1-2-3 | 14 | 0.1743 | 0.1162 | 0.0034 |
| 1-2-4 | 12 | 0.1887 | 0.1065 | 0.0046 |
| 1-3-3 | 10 | 0.1700 | 0.1289 | 0.0000 |
| 2-2-3 | 9 | 0.1646 | 0.1111 | 0.0453 |

Spearman rank correlation of W with type count over the nine instances: -0.15. W does not rise with type count, so by the reading registered in §7 type count was not what held the words back.

## L0 frontier by instance

Per pair of types with an acceptable allocation: S* in units of 10 − d (so a joint value; 2 is the most), the size of the Pareto set among acceptable allocations, the number of Nash points, and A's share of the surplus at the Nash point (the lower one if tied) over pairs with S* > 0.

| regime | instance | pairs with a deal | mean S* | mean Pareto size | mean Nash points | Nash share C_A |
|---|---|---|---|---|---|---|
| easy | 1-1-3 | 0.93 | 1.163 | 2.46 | 1.67 | 0.478 |
| easy | 1-2-2 | 0.98 | 1.168 | 2.93 | 1.65 | 0.465 |
| easy | 1-1-4 | 0.93 | 1.184 | 3.13 | 2.17 | 0.480 |
| easy | 1-2-3 | 0.95 | 1.171 | 3.52 | 2.04 | 0.469 |
| easy | 2-2-2 | 1.00 | 1.154 | 3.79 | 1.65 | 0.484 |
| easy | 1-1-5 | 0.90 | 1.218 | 3.72 | 2.82 | 0.477 |
| easy | 1-2-4 | 0.97 | 1.193 | 4.37 | 2.64 | 0.457 |
| easy | 1-3-3 | 0.91 | 1.190 | 4.24 | 2.40 | 0.461 |
| easy | 2-2-3 | 1.00 | 1.156 | 5.95 | 3.21 | 0.471 |
| hard | 1-1-3 | 0.72 | 1.046 | 1.81 | 1.43 | 0.484 |
| hard | 1-2-2 | 0.78 | 1.044 | 1.70 | 1.31 | 0.485 |
| hard | 1-1-4 | 0.66 | 1.102 | 2.17 | 1.79 | 0.489 |
| hard | 1-2-3 | 0.79 | 1.025 | 2.08 | 1.56 | 0.485 |
| hard | 2-2-2 | 0.80 | 1.028 | 1.85 | 1.34 | 0.489 |
| hard | 1-1-5 | 0.63 | 1.115 | 2.44 | 2.30 | 0.491 |
| hard | 1-2-4 | 0.78 | 1.079 | 2.34 | 1.79 | 0.491 |
| hard | 1-3-3 | 0.84 | 0.995 | 2.50 | 1.83 | 0.471 |
| hard | 2-2-3 | 0.74 | 1.156 | 2.73 | 2.07 | 0.489 |
