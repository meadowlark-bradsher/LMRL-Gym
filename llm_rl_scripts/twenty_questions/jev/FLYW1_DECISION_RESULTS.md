# FLYW-1: aim at a decision, not an identity — results

* **Prereg:** `FLYW1_DECISION_PREREG.md`, tag `prereg-flyw1-decision-20261118` (1762523), and Addendum 1, tag
  `prereg-flyw1-addendum1-20261119` (35f9826).
* **Build:** `FLYW1_BUILD.md`.
* **Jira:** FLYW-1 (epic); FLYW-4 (Part 1, arms), FLYW-5 (Part 2, escalation trigger), FLYW-6 (remediation label).
* **Runs** (all on 6 workers, `PYTHONHASHSEED=0`, BLAS pinned to 1 thread):
  * Part 2: jobq 64, 2026-11-18, 5 seeds, 1,023 jobs, 30 min. Output `results/flyw1_part2_20261118`.
  * Part 1: jobq 65, 2026-11-18 to 19, 66 seeds, 30,590 jobs. 15.2 h wall clock, including about 6.7 h while the
    Mac slept. Output `results/flyw1_part1_20261118`.
  * DE1 extension (Addendum 1): jobq 68, 2026-11-19, arms A and B on seeds 67..S, 2,704 jobs, 45 min. Output
    `results/flyw1_part1ext_20261119`.
  * Part 2's own seed count (Addendum 1, item 3): `flyw1_seed_rule_p2.py` gave S = 5 for every test, so Part 2 was
    not extended (`results/flyw1_part2_20261118/seed_rule_p2.json`).
* **Scoring:** jobq 69, `analyze_flyw1 --part1 … --part1-ext … --part2 … --gate … --seed-rule …` (the command is in
  `FLYW1_BUILD.md`), which wrote `results/flyw1_analysis_20261119.json`.
* **Reported tables:** `flyw1_report --analysis results/flyw1_analysis_20261119.json --part2
  results/flyw1_part2_20261118`, which wrote `results/flyw1_report_20261119.json`. Every number below that is not a
  verdict comes from that file. Its construction tables pool the analyzer's cells exactly as the analyzer pools them.

## In one paragraph

**The question.** An SRE agent does not need to know *which* fault it faces, only *what to do* about it. Faults that
share a remedy form a class. Should the agent stop questioning as soon as it knows the class, rather than the exact
fault? And does it pay to choose questions for what they say about the class? On the 20Q matrix, objects stand in for
faults and a class map stands in for the remedies. P0a, the hard-mask questioner the stress tests settled on, is
today's loop.

**The registered answer is no.** The decision rule says "class targeting does not pay on P0a", because DE1 fails.
B (P0a's questions, stopping at the class) is not cheaper than A (P0a as it is) in probes everywhere:
* on maps whose classes follow P0a's own questions (`align`, `cat17`), B saves 0.5–1.2 probes;
* on maps whose classes cut across them (`mix15`, `rand`), B needs 0.6–1.3 probes *more*.

**What the probe count hides.** A reaches the right class early by guessing objects by name, and every wrong guess is
a wrong *action* (in SRE terms, a remediation applied to the wrong fault). The prereg's reported total cost charges
each wrong action 50 probes at the primary cost ratio. On that measure B is cheaper than A in all 15
construction × flip cells. A costs from 8% more (`cat17` at flip 0, where A makes no wrong actions) to 5.8× as much
(`rand` at flip 0); under noise, 2.0–3.1× as much. Wherever A makes wrong actions, B cuts them by 60–95%; DE1(c) only
required B to make no more than A.

**So what.** The scored test asked whether a class stopping rule saves *questions*. It does where the classes line up
with the questions, and it does not elsewhere. The larger effect, fewer wrong actions, was measured but not scored.
That effect was flagged as a sensitivity before registration ("A's early guessing has a cost that probes alone do
not show"). The registered verdict stands. Whether the total-cost reading justifies the remediation label (FLYW-6)
is a decision this run cannot make on its own, and is left to the user. The clean way to settle it is a new prereg
with total cost as the primary measure.

**Escalation (Part 2).** The proposed T-loss trigger changes nothing at the primary ratio: it calls the expert
exactly as often as today's trigger. The reason is that at the moment either trigger is consulted, the class belief is
flat. On aligned maps the median belief in the best class is 0.11 for B and 0.24 for D, against the 0.8 needed to
act. Costed at the same ratio, T-loss is never cheaper than T-mass at any of the three ratios. Separately, every
arm takes wrong actions on *every* new-class incident before escalation is consulted at all (B about 3 per incident,
A 5.5–7.7). That is the design gap the prereg named: escalation is checked only when the live set empties or the probes
run out, not before each action.

**Distance from the main question.** This is a side branch. The main question is why the negotiating AI never asks
and what fixes it. FLYW-1 belongs to the SRE Flywheel; it fixes what the SRE system's belief store targets (the
remedy class, not the identity). Its one link back is the cost lesson: scoring a questioner on questions alone
rewards guessing, which is the same blind spot as scoring a negotiator on price alone.

## Terms

* **Probe:** one question or one action. **Probes to decide:** probes until the first right action.
* **Action:** applying a class's remedy. **Wrong action:** applying one to a secret outside that class. For A, a
  by-name guess of an object outside the secret's class is a wrong action; A is scored at its first right-class guess.
* **Arms** (all use P0a, the `modeswitch_guard` hard-mask belief, as the belief):
  * **A:** P0a as it is, guessing objects by name.
  * **B:** P0a's own questions; act on the class as soon as every live object shares it.
  * **C:** questions chosen for class information, with B's stopping rule. **C-bin:** C on yes/no answers (reported).
  * **D:** EC², a question score built to reach a class decision, with B's stopping rule.
  * **Fallback:** B, C or D finds no question that separates the live classes and acts on the largest live class.
* **Construction:** how a class map was built against P0a's own question tree. `align` follows it, `mix05` and `mix15`
  mix in 5% and 15% random reassignment, `rand` is random, `category` is `cat17` (the matrix's 17 categories).
* **Flip:** the share of attribute answers flipped by noise (0, 0.05, 0.10).
* **Total cost (`cost_r0.2`):** 1 per probe, 10 per expert call, 50 per wrong action (c_expert ÷ c_wrong = 0.2).
* **T-mass:** today's escalation; call the expert when P0a's live set empties or the probes run out.
  **T-loss:** at those same moments, act on the best class X\* if 1 − P(X\*) ≤ 0.2, else call the expert.

## Verdicts

| test | verdict | measured | bar |
|---|---|---|---|
| DE0 gate | **PASS** | every part holds | all parts |
| DE1 B beats A cheaply | **FAIL** | (a) fails on `mix05` flip 0, `mix15`, `rand`; (b) fails on `mix05`; (c) passes everywhere | all three |
| DE2 gain follows alignment | **PASS** | Spearman 0.82 over 56 maps | ≥ 0.5 |
| DE3(a) C myopic on `rand` | **PASS** | C − B = +3.17 probes | ≥ 1.0 |
| DE3(b) C worse under noise | **FAIL** | C − B wrong actions = −0.099 (SE 0.0024) | ≥ +0.02 |
| DE4 D ≤ C on `rand` | **PASS** | 16 of 16 maps | all |
| DE5 new question score not worth it | **FAIL** | D saves 1.14 probes over B at flip 0.05 (SE 0.0023); C costs 1.05 more | neither saves ≥ 0.5 |
| ET1 T-loss cuts spurious calls | **FAIL** | B 0.0274 vs 0.0274; D 0.0435 vs 0.0435 | ≤ 0.7 × T-mass |
| ET2 T-loss rarely acts on new-class novels | **PASS** | worst map 0.093 (B, `k8_bal_rand_s1`); D worst 0.053 | ≤ 0.10 per map |
| ET3 T-loss neutral on `rand` | **PASS** | B 0.0254 vs 0.0260; D 0.0367 vs 0.0378 | within 0.02 |

No test landed in the 2-SE inconclusive band. **Decisions:** Part 1, "class targeting does not pay on P0a; FLYW-6
records why no label". Part 2, T-loss is not adopted for B or D (ET1 fails, and its total cost is not lower: B 31.40
vs 31.36, D 37.22 vs 37.19).

## Part 1

### DE1: where B saves probes and where it does not

Mean probes to decide, A against B, pooled per construction (the analyzer's DE1(a) values). The bar is B ≤ A + 0.1.

| construction | flip 0 | flip 0.05 | flip 0.10 |
|---|---|---|---|
| `align` | 7.20 → 6.48 ✓ | 7.70 → 6.52 ✓ | 8.18 → 6.53 ✓ |
| `mix05` | 7.38 → 7.58 ✗ | 7.83 → 7.64 ✓ | 8.25 → 7.67 ✓ |
| `mix15` | 7.70 → 8.76 ✗ | 8.00 → 8.84 ✗ | 8.26 → 8.87 ✗ |
| `rand` | 8.52 → 9.80 ✗ | 8.66 → 9.92 ✗ | 8.75 → 9.95 ✗ |
| `cat17` | 7.06 → 6.55 ✓ | 7.68 → 6.81 ✓ | 8.29 → 7.05 ✓ |

DE1(b) needed B to save at least 0.5 probe on `align` and 0.3 on `mix05` at flips 0 and 0.05. `align` saved 0.72 and
1.18; `mix05` saved −0.20 and 0.19.

**Why B costs more on cut-across maps.** The basis (the class-map file's `p0a_pure_cost`, a tree model of P0a's
questions stopped at the class) predicted B's flip-0 probes. It was exact on `cat17` and increasingly optimistic as
classes cut across the questions:

| construction | basis | B observed | gap | B fallbacks per game |
|---|---:|---:|---:|---:|
| `cat17` | 6.55 | 6.55 | 0.00 | 0 |
| `align` | 6.11 | 6.48 | +0.37 | 0.017 |
| `mix05` | 6.70 | 7.58 | +0.88 | 0.038 |
| `mix15` | 7.26 | 8.76 | +1.51 | 0.078 |
| `rand` | 7.98 | 9.80 | +1.82 | 0.162 |

The fallback rate rises with the gap, but too few games fall back to account for 1.8 probes. The cause of the
remainder was not diagnosed. A plausible one is that P0a's live set in play is larger than the tree model's at the
same depth. Since the prereg's DE1 bars were set from this basis, the cut-across cells were never going to pass.

**The extension changed no verdict.** Addendum 1 ran A and B on more seeds in the four cells whose blind seed rule
asked for more than 66: `align` 71 and 119, `cat17` 465 and 787 seeds. Every one passed at 66 seeds and passes at S.
For example, `cat17` flip 0.10 DE1(c): B − A = −0.835 wrong actions, SE 0.0069 at 787 seeds (0.024 at 66). The 66-seed
results are kept in the JSON as `primary_protocol`.

### Total cost: what probes alone do not show

Reported, not scored. Mean total cost per game at ratio 0.2.

| construction | flip | A | B | C | C-bin | D |
|---|---:|---:|---:|---:|---:|---:|
| `align` | 0 | 14.5 | 7.0 | 7.0 | 6.3 | 6.4 |
| | 0.05 | 40.9 | 14.9 | 15.2 | 19.0 | 20.4 |
| | 0.10 | 68.2 | 22.7 | 23.0 | 30.0 | 31.7 |
| `mix05` | 0 | 23.3 | 8.6 | 8.8 | 8.0 | 7.6 |
| | 0.05 | 47.1 | 17.2 | 17.7 | 21.3 | 22.2 |
| | 0.10 | 71.6 | 25.6 | 26.3 | 32.9 | 34.1 |
| `mix15` | 0 | 39.4 | 10.7 | 11.5 | 10.2 | 9.0 |
| | 0.05 | 55.4 | 20.9 | 21.7 | 25.8 | 25.2 |
| | 0.10 | 71.5 | 30.2 | 30.9 | 38.4 | 38.0 |
| `rand` | 0 | 80.0 | 13.8 | 17.5 | 13.5 | 11.8 |
| | 0.05 | 87.9 | 28.7 | 29.4 | 37.1 | 33.2 |
| | 0.10 | 95.4 | 41.0 | 39.2 | 53.3 | 48.7 |
| `cat17` | 0 | 7.1 | 6.6 | 6.6 | 5.9 | 6.5 |
| | 0.05 | 42.1 | 20.7 | 20.2 | 33.1 | 33.5 |
| | 0.10 | 77.8 | 34.9 | 33.7 | 57.0 | 56.7 |

Wrong actions per game drive it. At flip 0.10, A makes 1.19 (`align`) to 1.73 (`rand`); B makes 0.32 to 0.61. Even at
flip 0, A makes 1.43 wrong actions per game on `rand` maps, because its by-name guesses land in the wrong class.

What else the table shows:
* **At flip 0, D and C-bin are cheapest.** D asks the fewest questions on cut-across maps (`rand` 7.87 probes against
  B's 9.80), which is DE5's failure.
* **Under noise, D and C-bin cost more than B** everywhere: they act sooner, with more of the live set still in play,
  and so follow a flip-struck answer to the wrong class more often (`align` flip 0.05: D 0.28 wrong actions against
  B's 0.16).
* **C makes fewer wrong actions than B on `rand` under noise** (DE3(b)'s wrong sign), but calls the expert about twice
  as often (0.117 against 0.056 at flip 0.10). It calls the expert in 11% of `rand` games even at flip 0, where answers
  are exact and the live set cannot empty, so those calls come from the 20-probe budget running out: C's myopic
  questions take too long. Its total cost on `rand` ends up slightly below B's at flip 0.10 (39.2 against 41.0).

### DE2–DE5 in brief

* **DE2:** B's saving over A rises with the map's alignment (Spearman 0.82), so where classes follow the questions is
  what decides whether stopping at the class saves probes.
* **DE3(a):** C, asking only for immediate class information, needs 3.17 more probes than B on `rand` maps. This is the
  myopia that motivated EC².
* **DE4:** D ≤ C on all 16 `rand` maps. D's saving over B is larger on cut-across maps (1.82) than on aligned maps
  (0.53), reported.
* **DE5:** D saves 1.14 probes over B at flip 0.05, more than the 0.5 bar, so by the prereg the question score
  matters too. But the prereg's branch for that case requires DE1 to pass, and D's extra wrong actions under noise
  make it costlier than B in total.

## Part 2: when to call the expert

### The verdicts

At the primary ratio T-loss calls the expert on exactly the same in-pool games as T-mass, on aligned maps for both B
and D, so ET1 fails. It rarely acts on new-class novels (ET2 passes) and changes nothing on `rand` maps (ET3 passes).

### Why the triggers agree

Both triggers are consulted at the same moment: when P0a's live set has emptied after its reinstates, or when the
probes run out. By then the answers contradict every known object, so the class belief is flat. At ratio 0.2 T-loss
needs P(X\*) ≥ 0.8 to act.

| arm, maps, games | trigger points | T-loss acts | of those, right | median P(X\*) | 90th pct. | share ≥ 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| B, `align`+`cat17`, in-pool | 2,278 | 0.18% | 100% | 0.11 | 0.18 | 0.18% |
| B, `rand`, in-pool | 1,733 | 3.0% | 75% | 0.28 | 0.62 | 3.0% |
| D, `align`+`cat17`, in-pool | 3,617 | 0.06% | 100% | 0.24 | 0.43 | 0.06% |
| D, `rand`, in-pool | 2,525 | 3.3% | 93% | 0.36 | 0.65 | 3.3% |
| B, `rand`, new-class | 4,080 | 3.1% | 0% | 0.26 | 0.71 | — |
| D, `rand`, new-class | 4,080 | 1.8% | 0% | 0.36 | 0.68 | — |

(Noisy flips pooled, weighted; P(X\*) at T-loss's first decision. On new-class novels, P(X\*) is missing at 55% of
B's and 64% of D's `rand` trigger points, because the stopping rule's wrong actions had already struck every known
class.)

ET1's basis was that a flip-struck secret's classmates would still fit the answers, so its class would keep most of
the belief. On aligned maps it keeps a median 0.11 (B) or 0.24 (D). This does not say a loss-based trigger cannot help. It says that
consulting one only where T-mass is consulted gives it nothing to work with. A trigger checked before each action, while
the class belief is still informative, is the version worth testing.

### Every new-class incident takes wrong actions first

On the prereg's original missed-call measure (any wrong action before the expert, or no expert call), every arm scores
**1.000** at every ratio and flip: every new-class novel takes at least one wrong action before escalation is consulted.
Wrong actions before the trigger point per new-class game (the analyzer's reported block):
* `align`+`cat17`: A 5.49–5.64, B 3.24–3.25, D 2.97–2.99;
* `rand`: A 7.41–7.74, B 2.87–2.92, D 3.03.

These come from the stopping rule, not the trigger, and are identical under both triggers. In SRE terms: a novel fault
gets about three remediations applied to it before anyone is paged.

### Other ratios and the max shadow (reported)

Total cost per game, T-mass against T-loss, both costed at the same ratio (noisy flips pooled):

| arm, maps | ratio 0.5 | ratio 0.2 | ratio 0.05 |
|---|---|---|---|
| B, `align`+`cat17` | 14.35 / 14.35 | 25.61 / 25.61 | 81.95 / 81.95 |
| B, `rand` | 21.53 / 21.69 | 38.46 / 38.55 | 123.10 / 123.23 |
| D, `align`+`cat17` | 17.16 / 17.22 | 32.74 / 32.75 | 110.65 / 110.65 |
| D, `rand` | 22.35 / 22.47 | 42.69 / 42.74 | 144.39 / 144.45 |

T-loss never lowers cost. At ratio 0.5 it acts more often, and on `rand` maps it acts on 12% (B) and 9% (D) of
new-class novels. The max-likelihood shadow is worse: it acts on 25% (B) and 20% (D) of them on `rand`, and on 12% of
D's on `align`+`cat17`. This is the risk the prereg named for the max version: the best member of a large class looks
like any novel.

**A correction to the analyzer's reported block.** `analysis.json`'s `part2.reported` costs T-mass at ratio 0.2 only,
while each T-loss entry is costed at its own ratio. Its ratio-0.5 and ratio-0.05 T-loss costs are therefore not
comparable with the T-mass cost next to them. The table above, from `flyw1_report` (`triggers_by_ratio`), costs both
at the same ratio. No verdict uses the reported block: ET1–ET3 and the adoption rule are at 0.2 throughout.

## What this changes

* **FLYW-6 (remediation label):** the registered rule records why no label is needed. The total-cost table argues the
  other way: B, which needs the label, is cheaper than A in every cell. This is the user's call. If it is reopened, the
  clean route is a new prereg with total cost (probes + c_wrong × wrong actions, at Part 2's ratios) as the primary
  measure, A against B and D, with bases computed from P0a's played live sets rather than the tree model.
* **Escalation:** T-loss at today's trigger points is not adopted. The open item is consulting escalation before each
  action (the original missed-call finding), which neither trigger here does.
* **Question score:** D asks fewer questions but acts into more noise. If it is revisited, it needs a noise-aware
  stopping threshold, not just a better question score.

## Caveats

* **Fixed pool.** The SEs are within-secret: they count only answer noise over this pool of 158 objects, not
  variation across pools (the scope the seed rule states).
* **Basis miss.** DE1's and DE5's bars were derived from a tree model of B that undercounts B's real probes on
  cut-across maps by up to 1.8. A rerun on corrected bases would not change the direction of DE1(a) on `mix15` and
  `rand`, where B is above A, but the bars as registered were not well calibrated there.
* **Standard errors on multi-map cells are understated (found 2026-11-19, `FLYW1_COST_BUILD.md`).** The
  within-secret SE treats each (map, secret) as an independent unit, but A plays each (secret, seed) game once and
  is scored on every map, and the keyed noise is shared by every arm's game of that secret and seed. On `align` at
  flip 0.05 (A − B total cost, 20 seeds) it gives 0.53, against 1.42 with the secret as the unit and 1.60 measured
  across seeds; on multi-map cells the understatement is 2.1–3.5×. No verdict changes: the closest scored test,
  DE1(b) on `mix05` at flip 0.05 (0.186 against 0.3), stays about 7 corrected SEs from its bar. The SEs quoted
  above and in `flyw1_analysis_20261119.json` for multi-map cells, and the Addendum 1 seed counts derived from
  them, carry this understatement; `cat17` (one map) is unaffected. The total-cost follow-up uses the secret unit.
* **Part 2 at 5 seeds.** Its own seed rule asked for no more, and its SEs are well inside its margins.
* **Synthetic classes.** Except `cat17`, the class maps are constructed; real remediation classes may sit anywhere on
  the alignment axis. DE2 says that position decides whether a class stopping rule saves questions.
