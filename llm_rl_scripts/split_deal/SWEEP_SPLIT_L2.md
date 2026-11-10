# Split Deal — L2 two-sided program and gate SD-D

_Rendered by `sweep_l2.py render` from `results/split_l2_20261110/sweep.json` (sweep `split-deal-l2-v0.1`, commit `b816d2c`, 11.0 s). Exact on the model: no sampling, no API; every number is formatted from the JSON. Predictions as registered in `DESIGN_SPLIT.md` §7; their operational reading (Q, prior, how D3 and D4 are scored, tie-breaks) was fixed in the docstring of `twosided.py` before the sweep ran._

Protocol 4.2 with Q = 3 rounds (A then B), then the proposer is drawn (A with probability π) and makes one binding offer. B is the rule counterpart B0 (asks by expected information gain, draws no inference from a decline, offers to maximise its own expected gain); A is A1, the exact best response to B0. Independent uniform prior. Y in units of 10 − d for each side; E is joint efficiency from the L0 oracle.

## Gate SD-D (π = 1/2)

| prediction | reading | verdict |
|---|---|---|
| D1 | E(T2) ≥ E(T1) ≥ E(T0), easy | **holds** |
| D2 | Y_A(T2) < Y_A(T1), easy | **fails** |
| D3 | strict prisoner's dilemma for both players, in some regime | **fails** |
| D4 | in T3, A1 answers some questions and declines others, by type, easy | **holds** |

### easy

- **D1** **holds**: E(T0) 0.6518, E(T1) 0.7652, E(T2) 0.8731 (E(T3) 0.8714).
- **D2** **fails**: Y_A(T1) 0.4985, Y_A(T2) 0.5153; A's lost information rent as responder -0.0338, and the difference equals (1 − π) times it exactly.
  The rent is negative: being known *raises* A's value as responder. An informed B0 makes offers A accepts more often (agreement when B proposes 68.0 % in T1, 89.4 % in T2), and that outweighs what B0 extracts.
- **D3** (B0 asks by EIG, the registered reading) **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D3, second reading** (B asks by the exact program) **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D4** **holds**: 8517 on-path decisions at B's questions; answered 5144, declined 1938 strictly and 1435 on a tie; 61 of 69 (instance, question) pairs are answered by some A types and declined by others. Choosing what to answer (T3) against always answering (T2): Y_A +0.0403, Y_B -0.0306, E -0.0017.

### hard

- **D1** **holds**: E(T0) 0.6108, E(T1) 0.7843, E(T2) 0.9236 (E(T3) 0.9354).
- **D2** **fails**: Y_A(T1) 0.3518, Y_A(T2) 0.3793; A's lost information rent as responder -0.0550, and the difference equals (1 − π) times it exactly.
  The rent is negative: being known *raises* A's value as responder. An informed B0 makes offers A accepts more often (agreement when B proposes 46.8 % in T1, 68.9 % in T2), and that outweighs what B0 extracts.
- **D3** (B0 asks by EIG, the registered reading) **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D3, second reading** (B asks by the exact program) **fails**: A is not in a dilemma, B is not; withholding dominant for A False, for B False; mutual disclosure beats mutual withholding for both True.
- **D4** **holds**: 8532 on-path decisions at B's questions; answered 5409, declined 966 strictly and 2157 on a tie; 59 of 69 (instance, question) pairs are answered by some A types and declined by others. Choosing what to answer (T3) against always answering (T2): Y_A +0.0215, Y_B -0.0118, E +0.0119.

**Registered consequence of D3 failing in both regimes** (§7): "this game has no negotiator's dilemma under its rules, and the two-way motivation would need a different mechanism (for example a private outside option) before a live study is worth designing."

One qualification, from D4 rather than D3: the dilemma that fails is the all-or-nothing one. Question by question, A1 does withhold selectively, and in easy that is worth +0.0403 Y to A and -0.0306 Y to B against always answering.

## The 2 × 2 disclosure game (π = 1/2)

Rows are A's answering, columns B's; each cell is (Y_A, Y_B). Temptation = withhold against a discloser, reward = both disclose, punishment = both withhold, sucker = disclose to a withholder.

**easy, B0 asks by EIG**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4985 | 0.5153 | 0.4155 | 0.4324 |
| B | 0.4827 | 0.4853 | 0.4155 | 0.4182 |

**easy, B asks by the exact program**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.4985 | 0.5015 | 0.4155 | 0.4185 |
| B | 0.4985 | 0.5015 | 0.4155 | 0.4185 |

**hard, B0 asks by EIG**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.3518 | 0.3793 | 0.2986 | 0.3261 |
| B | 0.3411 | 0.3695 | 0.2986 | 0.3270 |

**hard, B asks by the exact program**

| | temptation | reward | punishment | sucker |
|---|---|---|---|---|
| A | 0.3518 | 0.3801 | 0.2986 | 0.3270 |
| B | 0.3518 | 0.3801 | 0.2986 | 0.3270 |

## Structures by π

Y_A and Y_B are each side's expected Y; `prop` and `resp` are a side's Y when it proposes and when it responds; agreement shares by who proposed.

| regime | π | structure | Y_A | Y_B | Y_A prop | Y_A resp | Y_B prop | Y_B resp | agree, A proposes | agree, B proposes | E |
|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | 1/2 | T0 no queries | 0.4155 | 0.4155 | 0.5177 | 0.3133 | 0.5177 | 0.3133 | 68.0 % | 68.0 % | 0.6518 |
| easy | 1/2 | T1 private (A asks) | 0.4985 | 0.4182 | 0.6837 | 0.3133 | 0.5177 | 0.3186 | 90.6 % | 68.0 % | 0.7652 |
| easy | 1/2 | T2 public (both ask, both answer) | 0.5153 | 0.4853 | 0.6837 | 0.3470 | 0.6520 | 0.3186 | 90.6 % | 89.4 % | 0.8731 |
| easy | 1/2 | T3 reciprocal (A chooses) | 0.5557 | 0.4547 | 0.6774 | 0.4339 | 0.5869 | 0.3225 | 90.0 % | 87.8 % | 0.8714 |
| easy | 1/4 | T0 no queries | 0.3644 | 0.4666 | 0.5177 | 0.3133 | 0.5177 | 0.3133 | 68.0 % | 68.0 % | 0.6518 |
| easy | 1/4 | T1 private (A asks) | 0.4059 | 0.4680 | 0.6837 | 0.3133 | 0.5177 | 0.3186 | 90.6 % | 68.0 % | 0.7085 |
| easy | 1/4 | T2 public (both ask, both answer) | 0.4312 | 0.5687 | 0.6837 | 0.3470 | 0.6520 | 0.3186 | 90.6 % | 89.4 % | 0.8703 |
| easy | 1/4 | T3 reciprocal (A chooses) | 0.4970 | 0.5162 | 0.6662 | 0.4406 | 0.5816 | 0.3201 | 88.5 % | 88.1 % | 0.8699 |
| easy | 3/4 | T0 no queries | 0.4666 | 0.3644 | 0.5177 | 0.3133 | 0.5177 | 0.3133 | 68.0 % | 68.0 % | 0.6518 |
| easy | 3/4 | T1 private (A asks) | 0.5911 | 0.3684 | 0.6837 | 0.3133 | 0.5177 | 0.3186 | 90.6 % | 68.0 % | 0.8220 |
| easy | 3/4 | T2 public (both ask, both answer) | 0.5995 | 0.4020 | 0.6837 | 0.3470 | 0.6520 | 0.3186 | 90.6 % | 89.4 % | 0.8759 |
| easy | 3/4 | T3 reciprocal (A chooses) | 0.6185 | 0.3878 | 0.6826 | 0.4262 | 0.5919 | 0.3198 | 90.5 % | 87.7 % | 0.8752 |
| hard | 1/2 | T0 no queries | 0.2986 | 0.2986 | 0.3413 | 0.2560 | 0.3413 | 0.2560 | 46.8 % | 46.8 % | 0.6108 |
| hard | 1/2 | T1 private (A asks) | 0.3518 | 0.3270 | 0.4475 | 0.2560 | 0.3413 | 0.3127 | 74.5 % | 46.8 % | 0.7843 |
| hard | 1/2 | T2 public (both ask, both answer) | 0.3793 | 0.3695 | 0.4475 | 0.3110 | 0.4263 | 0.3127 | 74.5 % | 68.9 % | 0.9236 |
| hard | 1/2 | T3 reciprocal (A chooses) | 0.4007 | 0.3577 | 0.4448 | 0.3566 | 0.4036 | 0.3118 | 73.5 % | 70.8 % | 0.9354 |
| hard | 1/4 | T0 no queries | 0.2773 | 0.3200 | 0.3413 | 0.2560 | 0.3413 | 0.2560 | 46.8 % | 46.8 % | 0.6108 |
| hard | 1/4 | T1 private (A asks) | 0.3039 | 0.3341 | 0.4475 | 0.2560 | 0.3413 | 0.3127 | 74.5 % | 46.8 % | 0.6976 |
| hard | 1/4 | T2 public (both ask, both answer) | 0.3451 | 0.3979 | 0.4475 | 0.3110 | 0.4263 | 0.3127 | 74.5 % | 68.9 % | 0.9064 |
| hard | 1/4 | T3 reciprocal (A chooses) | 0.3798 | 0.3805 | 0.4403 | 0.3596 | 0.4041 | 0.3098 | 72.2 % | 71.4 % | 0.9317 |
| hard | 3/4 | T0 no queries | 0.3200 | 0.2773 | 0.3413 | 0.2560 | 0.3413 | 0.2560 | 46.8 % | 46.8 % | 0.6108 |
| hard | 3/4 | T1 private (A asks) | 0.3996 | 0.3199 | 0.4475 | 0.2560 | 0.3413 | 0.3127 | 74.5 % | 46.8 % | 0.8711 |
| hard | 3/4 | T2 public (both ask, both answer) | 0.4134 | 0.3411 | 0.4475 | 0.3110 | 0.4263 | 0.3127 | 74.5 % | 68.9 % | 0.9407 |
| hard | 3/4 | T3 reciprocal (A chooses) | 0.4236 | 0.3350 | 0.4470 | 0.3534 | 0.4059 | 0.3113 | 74.1 % | 70.8 % | 0.9453 |

## D3 by π (B0 reading)

| regime | π | A in dilemma | B in dilemma |
|---|---|---|---|
| easy | 1/2 | False | False |
| easy | 1/4 | False | False |
| easy | 3/4 | False | False |
| hard | 1/2 | False | False |
| hard | 1/4 | False | False |
| hard | 3/4 | False | False |

Not built here: B2 (a counterpart that reads A's declines and question choices) and the CFR exploitability check on the smallest instance (§5).
