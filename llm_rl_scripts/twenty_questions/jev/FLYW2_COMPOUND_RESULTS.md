# FLYW-7: more than one cause — results

* **Prereg:** `FLYW2_COMPOUND_PREREG.md`, tag `prereg-flyw2-compound-20261119` (008e6da).
* **Build:** `FLYW2_BUILD.md`. CX0 gate PASS (jobq 77, `results/flyw2_gate_20261119/gate.json`).
* **Jira:** FLYW-7 (epic FLYW-2). FLYW-8 (fixes) and FLYW-9 (the two-fault composition study) are decided from it.
* **Run:** jobq 80, 2026-11-19, `flyw2_run --run --seeds 5` on 6 workers (`PYTHONHASHSEED=0`, BLAS pinned to 1
  thread). It ran flip 0 once, then seeds 1–5 at flips 0.05 and 0.10: 154 jobs in 13.3 min. Output:
  `results/flyw2_run_20261119`.
* **Scoring:** jobq 81, `analyze_flyw2` (the command is in `FLYW2_BUILD.md`), which wrote
  `results/flyw2_analysis_20261119.json`. Every number below comes from that file. No code changed between the
  tag and the run; only the prereg's status line did.

## In one paragraph

**The question.** Today's loop, P0a, assumes an incident has one cause. When it has two, does single-cause
elimination fail badly enough to need new modelling?

**The registered answer is no, and the kill criterion applies.** CX1, CX2 and CX3 all pass, at both noise levels:

* **Two causes do not make P0a give up more often** (CX1).
* **The second cause costs about one expert call** (CX2): +12.4 cost units against a bar of 35.
* **The failure takes one form** (CX3): P0a names one of the two causes and stops, without a warning.

So **FLYW-8's modelling fixes are not built.** What carries forward is the check itself: **verify after every
remediation, and escalate if the incident has not cleared.** Without that check, the second cause costs about one
wrong action per incident instead of a fraction of an expert call.

**What the run cannot do: tell "new", "two causes" and "noise" apart (README gap 10).** TW1 and TW2 both fail.
Without noise, two-cause and new-object exhaustions were separable (AUC 0.94). With noise they are not (0.41,
worse than chance). The three-way classifier calls almost every exhaustion "one cause plus noise". On P0a's own
question trail, an empty survivor set does not say which of the three happened. Telling them apart needs questions
chosen for it (FLYW-8 arm DP).

**Distance from the main question** (why the AI never asks, and what fixes it): one step. A hard gate plus a check
after every fix is enough to stay safe with two causes. Only deliberately chosen questions could tell new, two
causes and noise apart, and that is a case where asking pays.

## Terms

| term | meaning |
|---|---|
| **P0a** | Today's loop: the `modeswitch_guard` hard mask, reinstate budget 3, no repeats, ε = 0.05, confidence 0.9, 20 probes, end rule "always" |
| **flip** | The chance that an attribute answer is flipped (the noise). Category and by-name answers are not flipped |
| **single / novel / KK / KN** | A game whose secret is one known object / one held-out object / a pair of known objects / a known and a held-out object |
| **exhaustion escalation** | P0a ran out of candidates (after its reinstate budget) and called the expert |
| **partial / silent partial** | P0a named one of the two secrets. Silent: it never ran out of candidates and the catch-all never fired, so nothing signalled the second cause |
| **catch-all fires** | The soft shadow's P(unknown) reached 0.9 after some answer |
| **compound penalty** | A KK game's cost minus the mean cost of its two members played alone, at the same flip and seed |
| **verified** | After a right fix, the incident is checked. If it has not cleared, the expert is called (+c_expert). Unverified: the uncleared second cause is a wrong action (+c_wrong) |
| **cost** | probes + 50 × wrong actions + 10 × expert calls (the FLYW-1 convention; a wrong by-name guess is a wrong remediation) |
| **SE** | Within-unit standard error (within pairs for KK, within secrets for singles). TW1 and TW2: a bootstrap over seeds |

## Scored tests

A result within 2 SE of its bar is inconclusive, and inconclusive counts as not passed. None was inconclusive.

| test | flip | estimate | SE | bar | verdict |
|---|---|---|---|---|---|
| CX1(a) excess exhaustion, KK − single | 0.05 | 0.0966 − 0.0851 = **+0.012** | 0.011 | < 0.10 | PASS |
| | 0.10 | 0.1749 − 0.1816 = **−0.007** | 0.014 | < 0.10 | PASS |
| CX1(b) excess catch-all firing | 0.05 | 0 − 0 = **0** | 0 | < 0.10 | PASS |
| | 0.10 | 0 − 0 = **0** | 0 | < 0.10 | PASS |
| CX2 compound penalty, verified, (10, 50) | pooled | **+12.43** (16.93 and 7.93) | 0.38 | ≤ 35 | PASS |
| CX3(a) KK names one secret | 0.05 | **0.903** | 0.001 | ≥ 0.80 | PASS |
| | 0.10 | **0.825** | 0.002 | ≥ 0.80 | PASS |
| CX3(b) KK silent partial | 0.05 | **0.888** | 0.001 | ≥ 0.70 | PASS |
| | 0.10 | **0.804** | 0.002 | ≥ 0.70 | PASS |
| TW1 AUC, compound vs novel | pooled | **0.413** | 0.013 | ≥ 0.85 | FAIL |
| TW2 three-way macro recall at w = 0.25 | pooled | **0.303** | 0.008 | ≥ 0.60 | FAIL |

**The registered decision** (as the analyzer printed it):

1. CX1 passes: the recommendation stands on exhaustion.
2. **Kill:** CX2 and CX3 pass. FLYW-8's modelling fixes are not built; carry "verify after every remediation;
   escalate if not cleared".
3. Gap 10: TW1 FAIL, TW2 FAIL. Separating novel, error and compound needs probes chosen for it (FLYW-8 arm DP).

## What happens in a two-cause game

| | flip | n games | named one / none / both | exhaustion esc. | silent partial | probes | wrong actions | cost (10, 50) verified | unverified |
|---|---|---|---|---|---|---|---|---|---|
| single | 0.05 | 705 | 0.915 / 0.085 / – | 0.085 | – | 10.00 | 3.11 | 166.5 | 166.5 |
| KK | 0.05 | 49,350 | 0.903 / 0.097 / 0 | 0.097 | 0.888 | 9.84 | 3.27 | 183.5 | 219.6 |
| single | 0.10 | 705 | 0.818 / 0.182 / – | 0.182 | – | 10.31 | 3.60 | 192.3 | 192.3 |
| KK | 0.10 | 49,350 | 0.825 / 0.175 / 0 | 0.175 | 0.804 | 9.97 | 3.60 | 200.2 | 233.2 |
| novel | 0.05 | 85 | – / 1.0 / – | 1.0 | – | 14.46 | 8.69 | 459.2 | 459.2 |
| novel | 0.10 | 85 | – / 1.0 / – | 1.0 | – | 13.99 | 8.61 | 454.6 | 454.6 |

* **P0a never named both causes.** Its loop ends at the first confirmed name, and that is the point of the
  verification step.
* **It never named an object that wasn't a secret** (CX0(c), checked again on the whole run).
* **The catch-all never fired on a single or a KK game.** It fired on 45% of novel games at flip 0.05 and on 40%
  at 0.10.
* **Wrong actions barely change.** The penalty's wrong-action part is +0.16 per game at flip 0.05 and +0.00 at
  0.10. The second cause hides behind the first fix, not behind more wrong guesses.

### The penalty under other readings

| reading | flip 0.05 | flip 0.10 |
|---|---|---|
| **verified, (10, 50): primary** | **+16.9** | **+7.9** |
| unverified, (10, 50) | +53.1 | +40.9 |
| verified, (10, 20) | +12.2 | +7.9 |
| verified, (10, 200) | +40.8 | +8.2 |
| verified, (30, 150) | +51.1 | +24.5 |
| verified, (100, 500) | +170.8 | +82.4 |
| cat17 decisions, (10, 50) | +2.3 | −7.2 |
| guesses as diagnostics, c_expert 10 | +9.0 | +7.8 |
| two separate incidents, (10, 50) | −149.6 | −184.3 |

* **The penalty is smaller at 10% noise** because the one-cause baseline is already worse there. Singles run out
  of candidates in 18% of games, about as often as pairs do.
* **Skipping verification raises the penalty to +53 and +41**, about one wrong action per incident.
* **The penalty scales with the prices.** At (30, 150) and (100, 500) it is about 1.7 expert calls at flip 0.05,
  not one. The bar was set at the primary costs only.
* **A pair is much cheaper than two separate incidents** (−150 and −184). That is the FLYW-1 cost scale again: one
  autonomous incident costs 17–19 expert calls, so a compound saves one whole autonomous run. This is why the
  primary measure compares against the mean member, not against two incidents.

## Overlap between the two causes (reported)

d_min is the number of questions on which the better-fitting member is contradicted by the pair's OR answers.

| stratum | games per flip (5 seeds) | exhaustion 0.05 / 0.10 | penalty 0.05 / 0.10 |
|---|---|---|---|
| d0 | 890 | 0.097 / 0.167 | −45.3 / −42.8 |
| d1 | 1,550 | 0.102 / 0.186 | −16.1 / −17.5 |
| d2 | 2,890 | 0.104 / 0.195 | +11.9 / +12.4 |
| d3–4 | 13,695 | 0.099 / 0.187 | +30.9 / +22.0 |
| d5+ | 30,325 | 0.095 / 0.167 | +14.6 / +3.9 |
| same category | 2,785 | 0.098 / 0.178 | −32.4 / −34.6 |
| cross category | 46,565 | 0.097 / 0.175 | +19.9 / +10.5 |

* **Overlap does not predict exhaustion or the penalty.** The Spearman of d_min is −0.008 (0.05) and −0.027 (0.10)
  against exhaustion, and +0.004 and −0.018 against the penalty.
* **Pairs that nearly coincide (d0, d1, same category) cost less than one cause alone.** The OR answers look like
  one of the members, and P0a reaches a right fix sooner. The middle strata (d3–4) cost the most.

## A known and a held-out cause (KN, reported)

| flip (11,985 games each) | named the known one | exhaustion esc. | silent partial | catch-all fires | cost (10, 50) |
|---|---|---|---|---|---|
| 0.05 | 0.463 | 0.538 | 0.443 | 0.227 | 317.7 |
| 0.10 | 0.424 | 0.576 | 0.403 | 0.214 | 325.3 |

**The new cause hides behind a known one in 40–44% of KN incidents.** P0a fixes the known cause and stops, and
nothing signals the new one. In the flywheel, verification is what catches it: the expert is called because the
incident has not cleared. Without verification, the new cause is never seen and so never admitted.

## The three-way test (gap 10)

| | flip 0 (basis) | noisy flips | noisy, ε = true flip |
|---|---|---|---|
| events: single / novel / compound | 0 / 17 / 153 | 188 / 170 / 13,398 | same |
| TW1 AUC, compound vs novel | 0.938 | **0.413** | 0.452 |
| TW2 macro recall at w = 0.25 | 0.020 | **0.303** | 0.329 |
| recall: single / novel / compound | – / 1/17 / 0/153 | 161/188 / 9/170 / 0/13,398 | 180/188 / 5/170 / 0/13,398 |
| AUC, single vs novel | – | 0.277 | |
| AUC, compound vs single | – | 0.607 | |

* **The flip-0 separability does not survive noise.** Without noise, a pair of known objects explained a two-cause
  exhaustion better than "something new" did (0.94). With noise, a held-out object's answers are explained by
  *some* pair of known objects better than a real pair's answers are (0.41). Out of 9,870 pairs, some pair always
  fits a new object's short, noisy question trail.
* **The classifier calls almost everything "one cause plus noise".** At w = 0.25, it recalled 0 of 13,398 compound
  exhaustions and 9 of 170 novel ones. At w = 0.5, with ε set to the true flip, it calls more compounds (7,489 of
  13,398), but takes 83 of 170 novels and 72 of 188 errors with them.
* **Knowing the true noise level does not rescue it** (0.45 and 0.33).

**What the flywheel would see.** Among exhaustion escalations at w = 0.25, these are the shares of compounds,
errors and novels:

| | compound | error | novel |
|---|---|---|---|
| flip 0.05 | 0.149 | 0.352 | 0.499 |
| flip 0.10 | 0.178 | 0.494 | 0.328 |

At 10% noise, only a third of what reaches the expert by exhaustion is actually new. Admitting from exhaustion
alone would admit two non-new causes for every new one.

## What this changes

* **Production recommendation:** add "verify after every remediation; if the incident has not cleared, escalate".
  The hard gate stays first. An empty survivor set still escalates, but it is not evidence of a new cause by
  itself.
* **FLYW-8:** pairs-as-hypotheses, noisy-OR and linked causes are not built. If anything is built, it is the
  discriminating probe (arm DP), and only to answer gap 10.
* **FLYW-9:** needed only to learn the likelihood, not to rescue elimination.
* **README gap 10:** answered "not on P0a's own answers". Exhaustion under noise cannot be split into new, error
  and compound without extra probes.

## Caveats

* **This is the 20Q matrix, not SRE.** Objects stand in for faults, and OR over attributes stands in for "either
  fault shows the symptom".
* **The bar holds at the primary costs only.** At higher prices the penalty is 1.7 expert calls, not one.
* **Verification is assumed perfect.** A check that sometimes reports "cleared" when it is not would move the
  verified penalty toward the unverified one.
* **TW1 and TW2 rest on 170 novel events** (17 novels × 5 seeds × 2 flips). Their bootstrap SEs are small (0.013
  and 0.008), and both are more than 30 SE below their bars.
