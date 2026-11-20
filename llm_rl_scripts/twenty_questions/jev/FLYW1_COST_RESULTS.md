# Does a class stopping rule pay in total cost? — results (FLYW-1, follow-up)

Results of `FLYW1_COST_PREREG.md`, registered as tag `prereg-flyw1-cost-20261119` (on fork main b407cf63). Scored
once, after the run had completed, by `analyze_flyw1_cost.py` at the registered commit.

## In plain terms

* **Question.** FLYW-1 found that stopping at the remediation class (arm B) does not save *questions* over P0a as
  it is (arm A). But A gets to the right class by guessing objects by name, and each wrong guess is a wrong
  *remediation*. Once a wrong remediation is charged what it costs, is B cheaper overall, on class maps and noise
  nobody had seen?
* **Finding.** Yes, everywhere tested. Charging a wrong action 50 probes, B is cheaper than A by 15–64 probes per
  incident in all 13 scored cells, and on all 144 fresh map × flip cells. It still wins when a wrong action costs
  only 5 probes. B takes 61–75% fewer wrong actions than A under noise, and 94–95% fewer without noise.
* **So what.** Every registered test passed, so the registered decision applies: **adopt**. FLYW-6 adds a
  remediation label to fault-graph's failure modes, and B is the default policy for the noisy regime. The
  saving over A holds whenever a wrong remediation costs more than 5 diagnostic probes.
* **One limit on "B is the default".** B beats the other class-aware policy, D (EC²), only when wrong actions are
  expensive. With no noise D is cheaper than B everywhere. Under noise B wins when a wrong action costs more than
  about 3–15 probes, depending on the cell. The registered comparison (S1) was at 50, where B wins clearly; at 5,
  D is cheaper in 4 of the 10 noisy cells. This is reported, not scored, and does not change the decision.
* **Distance from the main question** (why the AI never asks, and what fixes it): one step removed. This is about
  what the asker should aim at once it is asking (a decision, not an identity), not about whether it asks.

## Terms

* **Arms.** A = P0a, scored at its first right-class guess. B = P0a's questions with a class stopping rule: stop
  and act once the live set sits in one remediation class. D = EC², a class-aware question selector.
* **Total cost** per incident = probes + 10 per expert call + c_wrong per wrong action, in probe-equivalents.
  c_wrong = 50 is primary; 5 is the cheap case (P2); 20 and 200 are reported.
* **Flip** = the probability that each answer is flipped (noise). Noise is keyed on (seed, game, question), so all
  arms see the same flips.
* **Constructions.** How a class map relates to P0a's questions: `align` (classes follow the questions), `cat17`
  (the 17 real categories), `mix05`/`mix15` (5% / 15% of objects moved to a random class), `rand` (random classes).
* **SE** is per secret: the paired difference is averaged over a cell's maps for each (secret, seed), then the
  variance is taken across seeds (the unit the user chose before registration).
* **Verdicts.** PASS when estimate − 2 SE clears the bar, FAIL when estimate + 2 SE misses it, otherwise
  inconclusive (counts as not passed). Flip-0 cells are deterministic, so their SE is 0.

## Data

* **Fresh class maps:** `class_maps --seeds 3 4`, generated after the tag (`results/flyw1_cost_maps_20261119`, 57
  maps: 48 fresh `mix05`/`mix15`/`rand` maps plus FLYW-1's 8 `align` maps and `cat17`).
* **Fresh noise seeds:** 1001–1020 at flips 0.05 and 0.10, plus flip 0. Arms A, B, D.
* **Run:** jobq 79, 4,715 jobs, 57.2 min on 6 workers (projection 0.95 h), exit 0
  (`results/flyw1_cost_20261119`).
* **Analysis:** `results/flyw1_cost_analysis_20261119.json`.

## Scored tests

All five pass. Each table entry is the estimate (SE); per-secret SE.

| cell | P1: A − B at c_wrong 50 (bar 2) | P2: A − B at c_wrong 5 (bar 0) | P3: wrong A − wrong B (bar 0) | basis A − B at 50 |
|---|---:|---:|---:|---:|
| `mix05` flip 0 | 15.52 | 1.24 | 0.317 | 14.7 |
| `mix05` 0.05 | 30.97 (1.23) | 3.19 (0.15) | 0.618 (0.024) | 29.8 |
| `mix05` 0.10 | 44.01 (1.61) | 4.81 (0.19) | 0.871 (0.032) | 46.0 |
| `mix15` flip 0 | 28.34 | 2.04 | 0.584 | 28.7 |
| `mix15` 0.05 | 36.17 (0.82) | 3.12 (0.10) | 0.735 (0.016) | 34.5 |
| `mix15` 0.10 | 42.99 (1.09) | 4.02 (0.13) | 0.866 (0.021) | 41.3 |
| `rand` flip 0 | 63.52 | 5.19 | 1.296 | 66.3 |
| `rand` 0.05 | 56.80 (0.54) | 4.49 (0.07) | 1.162 (0.011) | 59.2 |
| `rand` 0.10 | 51.98 (0.68) | 3.98 (0.08) | 1.067 (0.013) | 54.4 |
| `align` 0.05 | 29.93 (1.67) | 4.25 (0.21) | 0.571 (0.033) | 26.0 |
| `align` 0.10 | 48.99 (2.20) | 6.65 (0.27) | 0.941 (0.043) | 45.5 |
| `cat17` 0.05 | 25.06 (1.73) | 3.45 (0.21) | 0.480 (0.034) | 21.4 |
| `cat17` 0.10 | 46.15 (2.30) | 5.96 (0.27) | 0.893 (0.045) | 43.0 |

* **P1 (primary) PASS** in all 13 cells. The closest is `mix05` at flip 0, 15.5 against a bar of 2; among noisy
  cells, `cat17` at 0.05, whose lower 2-SE bound is 21.6. Every estimate is within 4 of its FLYW-1 basis.
* **P2 PASS** in all 13 cells. Closest: `mix05` at flip 0, 1.24 (deterministic); among noisy cells, `mix15` at
  0.05, lower bound 2.92.
* **P3 PASS** in all 13 cells (at flip 0 the rule is wrong_B ≤ wrong_A).
* **P4 PASS:** B's point estimate is cheaper than A's at c_wrong 50 on all 144 fresh map × flip cells. Smallest
  gap 5.68 (`k2_skew_mix05_s4`, flip 0); largest 121.75 (`k16_bal_rand_s4`, flip 0).
* **S1 PASS:** pooled over flips 0.05 and 0.10, D − B at c_wrong 50 is +5.33 (0.46) `mix05`, +6.02 (0.47) `mix15`,
  +5.59 (0.50) `rand`, +6.42 (0.45) `align`, +15.16 (1.17) `cat17`. D is not cheaper beyond 2 SE anywhere.

**Decision (registered rule):** P1, P3 and P4 pass, and P2 passes, so **ADOPT**: FLYW-6 adds a remediation label;
B is the default for the noisy regime; the result holds whenever a wrong remediation costs more than 5 probes. The
kill criterion (P1 or P3 failing anywhere) did not fire.

## Reported, not scored

### Cost per incident by arm

Probes, wrong actions per incident, expert share, and total cost at c_wrong 5 / 50 / 200.

| cell | A | B | D |
|---|---|---|---|
| `mix05` 0 | 7.40, 0.34, 0 → 9.1 / 24.3 / 75.1 | 7.74, 0.02, 0 → 7.9 / 8.8 / 12.0 | 6.52, 0.02, 0 → 6.6 / 7.6 / 10.8 |
| `mix05` 0.05 | 7.88, 0.83, .029 → 12.3 / 49.7 / 174.3 | 7.87, 0.21, .021 → 9.1 / 18.7 / 50.7 | 6.79, 0.30, .026 → 8.5 / 21.9 / 66.3 |
| `mix05` 0.10 | 8.26, 1.24, .052 → 15.0 / 70.8 / 256.8 | 7.95, 0.37, .037 → 10.2 / 26.8 / 82.1 | 7.01, 0.54, .054 → 10.2 / 34.3 / 114.5 |
| `mix15` 0 | 7.68, 0.62, 0 → 10.8 / 38.4 / 130.7 | 8.56, 0.03, 0 → 8.7 / 10.1 / 14.7 | 7.07, 0.03, 0 → 7.2 / 8.6 / 13.2 |
| `mix15` 0.05 | 8.04, 0.98, .027 → 13.2 / 57.3 / 204.3 | 8.67, 0.25, .020 → 10.1 / 21.1 / 57.9 | 7.37, 0.34, .029 → 9.4 / 24.6 / 75.6 |
| `mix15` 0.10 | 8.32, 1.29, .048 → 15.2 / 73.1 / 265.8 | 8.74, 0.42, .036 → 11.2 / 30.1 / 93.0 | 7.61, 0.61, .060 → 11.2 / 38.6 / 129.7 |
| `rand` 0 | 8.46, 1.38, 0 → 15.3 / 77.2 / 283.5 | 9.75, 0.08, 0 → 10.1 / 13.7 / 25.5 | 7.87, 0.08, 0 → 8.3 / 11.8 / 23.6 |
| `rand` 0.05 | 8.62, 1.55, .023 → 16.6 / 86.1 / 317.9 | 9.87, 0.38, .031 → 12.1 / 29.3 / 86.8 | 8.21, 0.49, .042 → 11.1 / 32.9 / 105.6 |
| `rand` 0.10 | 8.75, 1.69, .040 → 17.6 / 93.7 / 347.5 | 9.94, 0.63, .056 → 13.6 / 41.8 / 135.5 | 8.47, 0.80, .081 → 13.3 / 49.4 / 169.8 |
| `align` 0 * | 7.20, 0.15, 0 → 7.9 / 14.5 / 36.3 | 6.48, 0.01, 0 → 6.5 / 7.0 / 8.4 | 5.96, 0.01, 0 → 6.0 / 6.4 / 7.9 |
| `align` 0.05 | 7.80, 0.76, .033 → 11.9 / 46.0 / 159.5 | 6.55, 0.19, .018 → 7.7 / 16.0 / 44.0 | 6.23, 0.28, .025 → 7.9 / 20.2 / 61.5 |
| `align` 0.10 | 8.28, 1.27, .060 → 15.2 / 72.4 / 262.9 | 6.59, 0.33, .034 → 8.6 / 23.4 / 72.8 | 6.45, 0.50, .053 → 9.5 / 32.0 / 107.2 |
| `cat17` 0 * | 7.06, 0, 0 → 7.1 / 7.1 / 7.1 | 6.55, 0, 0 → 6.6 / 6.6 / 6.6 | 6.47, 0, 0 → 6.5 / 6.5 / 6.5 |
| `cat17` 0.05 | 7.79, 0.79, .079 → 12.5 / 48.1 / 166.8 | 6.86, 0.31, .067 → 9.1 / 23.1 / 69.7 | 6.98, 0.50, .087 → 10.4 / 32.9 / 108.2 |
| `cat17` 0.10 | 8.38, 1.45, .139 → 17.0 / 82.0 / 298.9 | 7.08, 0.55, .119 → 11.0 / 35.9 / 118.8 | 7.41, 0.94, .174 → 13.9 / 56.3 / 197.9 |

\* not scored: deterministic and already seen in FLYW-1. All 27 of their arm × map results match FLYW-1 exactly
(the reproducibility check passes).

### Break-even cost c\* of a wrong action

c\* is the wrong-action cost above which one arm is cheaper than the other. "Dominates" means no more probes, no
more expert calls and no more wrong actions, so it is cheaper at any cost.

| cell | B against A | B against D |
|---|---|---|
| `mix05` 0 / 0.05 / 0.10 | 1.10 / dominates / dominates | D always cheaper / 12.3 / 4.7 |
| `mix15` 0 / 0.05 / 0.10 | 1.50 / 0.76 / 0.35 | D always cheaper / 12.8 / 4.7 |
| `rand` 0 / 0.05 / 0.10 | 1.00 / 1.13 / 1.27 | D always cheaper / 15.2 / 6.9 |
| `align` 0 / 0.05 / 0.10 | dominates / dominates / dominates | D always cheaper / 2.8 / dominates |
| `cat17` 0 / 0.05 / 0.10 | dominates / dominates / dominates | D always cheaper / dominates / dominates |

* **Against A**, per fresh map: B dominates on 30 of 144; on the rest c\* is at most 4.95
  (`k2_skew_mix15_s3`, flip 0.10). So on every fresh map B is cheaper than A once a wrong action costs 5 probes,
  though barely on that map.
* **Against D**, under noise, per fresh map: B is cheaper above c\* on 91 of 96, dominates on 4, and is never
  cheaper on 1 (`k16_skew_mix15_s3`, flip 0.05). The per-map c\* runs from 0.09 to 110 (median 13.2). At flip 0, D
  is cheaper than B on every map: with no noise its better questions win, and neither makes wrong actions.

### Alignment and the size of B's lead

Spearman correlation, over the 48 fresh maps, between a map's alignment index and A − B at c_wrong 50: −0.84 at
flip 0, −0.58 at 0.05, −0.29 at 0.10. B's lead is largest on maps whose classes cut across P0a's questions, where
A's by-name guesses go wrong most often. Noise weakens the relation, because under noise both arms take wrong
actions on every kind of map.

## What this does and does not show

* **A registered replication.** The arms, belief, oracle, pool and cost model are FLYW-1's. The new evidence is
  that B's total-cost lead holds on 48 class maps and 20 noise seeds no one had seen, survives a cheap wrong
  action, and faced an explicit kill criterion that did not fire.
* **This pool, synthetic classes, bitmask form, no escalation trigger.** The prereg's scope holds: 158 fixed
  objects; constructed classes except `cat17`; independent elimination masks, not a constraint graph where one
  elimination cascades; escalation consulted only when the live set empties or probes run out. Each would need its
  own prereg.
* **The default policy depends on the price of a mistake.** B over A needs a wrong action to cost more than about
  5 probes. B over D needs more: up to about 15 at flip 0.05. Domains with cheap remediations may do better with D.

## Disclosure

* Scored once, after the run had completed. No interim outcome was read.
* The map generator prints each map's flip-0 descriptives (alignment, P0a's pure cost, class cost, EC² cost) when it
  writes the maps. Those lines were on screen before the run. They are map properties computed by the registered
  generator, not run outcomes, and were not used.
* Run, maps, analysis: `results/flyw1_cost_20261119`, `results/flyw1_cost_maps_20261119`,
  `results/flyw1_cost_analysis_20261119.json`.
