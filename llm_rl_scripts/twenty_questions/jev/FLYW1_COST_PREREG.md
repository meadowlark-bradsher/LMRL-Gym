# Does a class stopping rule pay in total cost? — pre-registration (FLYW-1, follow-up)

**Status: DRAFT, not registered.** Nothing in this document has been run. It follows FLYW-1
(`FLYW1_DECISION_PREREG.md`, tags `prereg-flyw1-decision-20261118` and `prereg-flyw1-addendum1-20261119`;
results in `FLYW1_DECISION_RESULTS.md`), whose results doc names this prereg as "the clean route" to settling
FLYW-6.

Jira: epic FLYW-1. Decides FLYW-6 (whether fault-graph's failure modes get a remediation label).

## Why this prereg exists

FLYW-1 asked whether stopping at the remediation class (arm B) beats P0a as it is (arm A), and scored it on
**probes**. The registered answer is no: B saves probes where classes follow P0a's questions (`align`, `cat17`)
and needs 0.6–1.3 more where they cut across them (`mix15`, `rand`).

Probes count a wrong action the same as a question. A reaches the right class early by guessing objects by name,
and its wrong guesses are wrong *remediations*. FLYW-1's reported (not scored) total cost charged each wrong
action 50 probes. On it, B was cheaper than A in all 15 construction × flip cells, and wherever A made wrong
actions B cut them by 60–95%. That reading was made after the results were seen, so it cannot decide FLYW-6.

This prereg makes total cost the primary measure and tests it on data no one has seen: **fresh class maps** and
**fresh noise seeds**.

**What is not new.** The arms, belief, oracle, pool and cost model are FLYW-1's, unchanged. The predictions
below come from FLYW-1's Part 1, which has been fully read. So this is a registered replication with the
scoring changed, not a test of a new idea. The new information it can produce is:

1. whether B's total-cost advantage holds on class maps it was not tuned or observed on;
2. whether it survives a *cheap* wrong action (5 probes), which is where it would first break;
3. an explicit kill criterion, which FLYW-1 did not have.

## Terms

Unchanged from FLYW-1 except where noted.

| term | meaning |
|---|---|
| **A** | P0a as it is (`modeswitch_guard`, reinstate K = 3, `no_repeat`), guessing objects by name. Scored at its first right-class guess; every by-name guess outside the secret's class is a wrong action |
| **B** | P0a's own questions; acts on the class as soon as every live object shares it (fallback: largest live class) |
| **D** | EC² question score with B's stopping rule |
| **wrong action** | An action on any class other than the secret's |
| **expert call** | The game ends without a right action (T-mass, today's escalation) |
| **total cost** | Per game: 1 per probe + c_expert per expert call + c_wrong per wrong action. c_expert = 10 throughout |
| **c_wrong** | 50 (FLYW-1's primary ratio 0.2), **primary**. Also 5 (P2), and 20 and 200 (FLYW-1's ratios 0.5 and 0.05, reported) |
| **cell** | One construction × flip, pooled over its maps with FLYW-1's game-weighted pooling |
| **SE** | FLYW-1's within-secret SE: per (map, secret) unit, the variance of the paired difference across noise seeds, pooled; SE of a cell mean = √(Σ σᵢ²) ÷ (N · √S). Flip-0 cells are deterministic (SE 0) |

C and C-bin are not run. Neither is a candidate policy: C is myopic (FLYW-1 DE3(a)), and C-bin was a diagnostic.

## Data

### Fresh class maps

* **Constructions `mix05`, `mix15`, `rand`:** generate new maps with `class_maps.py` (generator version 3,
  unchanged) at **generator seeds 3 and 4**. FLYW-1 used seeds 1 and 2. K ∈ {2, 4, 8, 16} × profile ∈ {`bal`,
  `skew`} × 2 seeds × 3 constructions = **48 fresh maps**. The seeds are fixed here, before any map is generated;
  no fresh map is dropped or regenerated for any reason other than a generator error, which is reported.
* **`align` (8 maps) and `cat17` (1 map)** are deterministic constructions, so they cannot be refreshed. They run
  with fresh noise seeds only.
* Each fresh map's alignment index is computed by the generator and reported.

### Fresh noise seeds

* **Noisy flips 0.05 and 0.10:** noise seeds **1001–1020** (S = 20). FLYW-1 used seeds 1–66, and the DE1
  extension 67–787, so these streams are unseen. The keyed oracle (seed, secret, question) is unchanged, so arms
  stay paired on the same noise.
* **Flip 0:** one deterministic game per secret, as in FLYW-1.

### Scored cells

| construction | maps | flip 0 | flips 0.05, 0.10 |
|---|---|---|---|
| `mix05`, `mix15`, `rand` | 16 fresh each | scored (fresh maps) | scored |
| `align` | FLYW-1's 8 | **not scored**: deterministic, already seen | scored (fresh seeds) |
| `cat17` | FLYW-1's 1 | **not scored**: deterministic, already seen | scored (fresh seeds) |

That is **13 scored cells**: 3 constructions × 3 flips on fresh maps, plus `align` and `cat17` × 2 noisy flips.
The two unscored flip-0 cells are reported.

### Seed count

S = 20 per noisy flip, fixed now. Rule: FLYW-1's, S = max(5, ⌈5 × (SE₅ ÷ ((G − m) ÷ 3))²⌉), where G is the basis gap
and m the margin (below). With SE₅ computed from FLYW-1 Part 1's seeds 1–10
(`results/flyw1_cost_basis_20261119/cost_var.json`, from `flyw1_cost_basis.py`, disclosed below), it asks for 5
seeds in every scored noisy cell, for P1 and P2 alike. The tightest are both on `cat17`: P1 at flip 0.10, SE₅ 4.17
against a basis gap of 43.0; P2 at flip 0.05, SE₅ 0.36 against 3.02.
S = 20 is four times that, to cover the shift to fresh maps, where Part 1's variances are only an estimate.

**Budget:** about 2 × 20 × (2 arms × 57 maps + A) ≈ 4,600 noisy jobs plus flip 0, roughly 15% of FLYW-1 Part 1's
job count, without C's long games. Project hours from Part 1's per-job `secs` before queueing; expected well
under 2 h on 6 workers through `jobq`.

## Predictions

The basis for every prediction is FLYW-1 Part 1 (66 seeds, the analyzer's cells, game-weighted per construction).

| construction | flip | A − B at c_wrong 50 | A − B at c_wrong 5 | wrong actions A → B | D − B at 50 | SE₅ of A − B at 50 |
|---|---|---:|---:|---|---:|---:|
| `align` | 0 | 7.5 | 1.40 | 0.15 → 0.01 | −0.5 | 0 |
| | 0.05 | 26.0 | 3.77 | 0.66 → 0.16 | +5.5 | 1.07 |
| | 0.10 | 45.5 | 6.25 | 1.19 → 0.32 | +9.0 | 1.47 |
| `mix05` | 0 | 14.7 | 1.30 | 0.32 → 0.02 | −1.0 | 0 |
| | 0.05 | 29.8 | 3.25 | 0.78 → 0.19 | +5.0 | 0.71 |
| | 0.10 | 46.0 | 5.31 | 1.26 → 0.35 | +8.5 | 0.99 |
| `mix15` | 0 | 28.7 | 1.91 | 0.63 → 0.04 | −1.7 | 0 |
| | 0.05 | 34.5 | 2.76 | 0.94 → 0.24 | +4.3 | 0.56 |
| | 0.10 | 41.3 | 3.69 | 1.25 → 0.42 | +7.8 | 0.76 |
| `rand` | 0 | 66.3 | 5.47 | 1.43 → 0.08 | −1.9 | 0 |
| | 0.05 | 59.2 | 4.73 | 1.58 → 0.37 | +4.5 | 0.47 |
| | 0.10 | 54.4 | 4.25 | 1.73 → 0.61 | +7.7 | 0.61 |
| `cat17` | 0 | 0.5 | 0.51 | 0.00 → 0.00 | −0.1 | 0 |
| | 0.05 | 21.4 | 3.02 | 0.67 → 0.27 | +12.8 | 3.02 |
| | 0.10 | 43.0 | 5.60 | 1.36 → 0.53 | +21.8 | 4.17 |

`align` and `cat17` at flip 0 are shown for completeness and are not scored. Per map, B was cheaper than A in all
171 FLYW-1 map × flip cells at c_wrong 50, 20 and even 5 (smallest per-map gaps: 0.51 at 50, on `cat17` flip 0;
0.03 at 5, on `k2_skew_mix05_s2` flip 0).

**Break-even.** B is cheaper than A whenever a wrong action costs more than c\* probes, with
c\* = (probes_B − probes_A − c_expert · (expert_A − expert_B)) ÷ (wrong_A − wrong_B). On the basis, c\* ≤ 1.8 in
every cell where B does not simply dominate (B dominates, no more probes and no more wrong actions, in 8 of 15).

### Scored tests

Each test passes, fails or is **inconclusive** by FLYW-1's rule: inconclusive when the estimate is within 2 SE of
the bar. **An inconclusive test counts as not passed.**

* **P1 (primary): B is cheaper than A in total cost.** At c_wrong = 50, in every scored cell,
  A − B ≥ **2** (probe-equivalents per game), with A − B − 2 SE > 2. Basis: 14.7–66.3 in the scored cells.
  The margin of 2 is about a seventh of the smallest basis gap on a scored cell, and two questions per incident
  is a saving worth a label.
* **P2: it survives a cheap wrong action.** At c_wrong = 5, in every scored cell, A − B > 0 (A − B − 2 SE > 0).
  Basis: 1.30–6.25. This is where the claim would break first: on the fresh flip-0 maps, B's lead is only its
  avoided wrong actions minus its extra questions.
* **P3: B makes fewer wrong actions than A.** In every scored noisy cell, wrong_B < wrong_A (2-SE). At flip 0 on
  the fresh maps, wrong_B ≤ wrong_A. Basis: 60–95% fewer.
* **P4: no fresh map goes the other way.** At c_wrong = 50, B's point estimate is cheaper than A's on **every**
  fresh map at every flip (48 maps × 3 flips = 144). Basis: 171 of 171.

### Secondary: which class-aware policy

* **S1: under noise, B is cheaper than D.** At c_wrong = 50, pooled over flips 0.05 and 0.10, per construction,
  D − B > 0 (2-SE). Basis: +4.3 to +21.8. At flip 0 D is cheaper (basis −0.1 to −1.9); reported.
* Reported: the break-even c\* of D against B per cell (basis 1.7–12.7 under noise; D always cheaper at flip 0).

### Reported, not scored

* every cell's probes, wrong actions, expert share, and total cost at c_wrong 5, 20, 50 and 200, for A, B and D;
* c\* (B against A, and D against B) per cell and per fresh map;
* per-map A − B against the map's alignment index (FLYW-1's DE2 relation, now on cost);
* the two unscored flip-0 cells (`align`, `cat17`), checked against FLYW-1 as a reproducibility check.

## Decision rule and kill criterion

* **Kill.** If P1 or P3 **fails** (the estimate is beyond 2 SE on the wrong side of the bar) in any scored cell,
  class targeting does not pay on P0a even in total cost. FLYW-6 closes with no remediation label, and the
  decision-goal proposal (sre-flywheel README, Experiment 1) is dropped for P0a-style policies.
* **Adopt.** If P1, P3 and P4 pass: FLYW-6 adds a remediation label to fault-graph's failure modes, and the class
  stopping rule (B) is the default for the noisy regime.
  * If P2 also passes, the result holds whenever a wrong remediation costs more than 5 diagnostic probes.
  * If P2 fails or is inconclusive in some cells, the label is still adopted, and FLYW-6 records the cells, their
    c\*, and that domains with remediations cheaper than about c\* probes may not benefit.
* **Not decided.** If any of P1, P3 or P4 is inconclusive, or P4 fails on some map while P1 passes, the label is
  not adopted on this run. Report the cells or maps, their alignment, and the SEs; a follow-up needs its own
  prereg.
* **S1** chooses between B and D for the noisy regime: if it passes, B; if D − B < 0 beyond 2 SE in some
  construction, D is noted there. It does not change the FLYW-6 decision.

## What has been seen, and disclosure

* FLYW-1's Part 1, its DE1 extension, Part 2, the analysis and `FLYW1_DECISION_RESULTS.md` have all been read.
  Every basis number above comes from them.
* `cost_var.json` (Part 1 seeds 1–10, arms A, B and D at flips 0.05 and 0.10, c_wrong 5, 20, 50 and 200) was
  computed while drafting this document, to size S. It holds means and SEs of the paired cost differences. No
  fresh map or fresh seed has been generated or played.
* The margins, c_wrong values and S were chosen after seeing FLYW-1. P2's c_wrong = 5 stands for a cheap
  remediation, a tenth of the primary cost. On the basis, B still wins every cell and every map there, but by as
  little as 0.03 on one map, so it is the bar most likely to fail on fresh maps.

## Scope

* **This pool.** The 158 objects are fixed, as in FLYW-1. Generalisation comes from fresh class maps and noise,
  not from sampling objects.
* **Synthetic classes.** Except `cat17`, the class maps are constructed. Real remediation classes may sit anywhere
  on the alignment axis; the per-map cost against alignment is reported for that reason.
* **Bitmask form only.** P0a eliminates objects one by one with independent masks. The sre-flywheel README
  ("Bitmask or graph", under the meet-semilattice claims) notes that a constraint-graph form, where one
  elimination cascades to dependent hypotheses, is closer to diagnosis. This run says nothing about that form: a
  cascade could change B's question count and A's guessing alike.
* **Escalation is out of scope.** FLYW-1 Part 2 showed that escalation is consulted only once the live set
  empties or the probes run out, and that every new-class incident takes wrong actions first. A trigger checked
  before each action is a separate prereg.

## Build needed

1. `class_maps.py`: a `--seeds` option, so seeds 3 and 4 write a new map set
   (`results/flyw1_cost_maps_<date>/`) without touching FLYW-1's.
2. `flyw1_run.py`: `--seed-start` (noise seeds 1001–1020), `--arms A,B,D`, and `--maps-dir`.
3. An analyzer, `analyze_flyw1_cost.py`, reusing `analyze_flyw1`'s loaders, `score_a_record` and pooling, that
   scores P1–P4 and S1 with the within-secret SE and the 2-SE inconclusive rule.
4. **Gate**, before the run: B, D and A on `align` and `cat17` at flip 0 reproduce FLYW-1's cells exactly, and a
   regenerated seed-1 map from the new `--seeds` path matches FLYW-1's file byte for byte.
5. Projection from Part 1's per-job timings, then `jobq -N 6`.
