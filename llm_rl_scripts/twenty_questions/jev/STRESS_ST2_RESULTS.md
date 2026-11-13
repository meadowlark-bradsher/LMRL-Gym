# ST-2 Look-alike tail — results

Scored against `STRESS_LOOKALIKE_PREREG.md` as tagged `prereg-stress-st2-20261112`
(commit 629b540). Build note: `STRESS_ST2_BUILD.md`. Run directory:
`results/stress_st2_20261112` (732 jobs × 2500 games, one segment, 386 min on 5
workers under `taskpolicy -c utility`; provenance clean: registered commit, no dirty
files, Python 3.9.21, `PYTHONHASHSEED=0`). Registered scoring: `analyze_st2` →
`analysis.json` and two figures. Post-hoc (not registered, labelled as such below):
`analyze_st2_posthoc` → `posthoc.json`.

All numbers are deterministic replays of seeded jobs; a re-run reproduces them
exactly. A cell pools its 3 look-alike seeds (and, under noise or in the G arm, its 3
noise seeds). "P0a" is `modeswitch_guard`.

## Headline

1. **Under exact guesses, look-alikes are safe and cheap for P0a at every distance.**
   `modeswitch_guard` and `hard` end no look-alike game `wrong` at any d or flip rate,
   and fire on every look-alike first appearance (1.000 in all 21 cells each). A
   d ≤ 2 look-alike costs about 0.7 extra questions over an ordinary first appearance
   (P1 PASS). Unguarded `modeswitch` fails P1: 8–12 look-alike `wrong` per cell at
   d ≤ 4 with clean answers, up to 46 under noise. These are the ST-1 path
   (budget-unaware reinstatement). The guard removes every one.
2. **Name confusion hits every backend about equally, so the ranking does not
   reverse (P4 headline: P4a PASS, P4b FAIL).** In the G arm, `false_confirm` on
   look-alike games with the parent known is ≈ g × P(parent guessed by name) for all
   four backends (ratio 0.75–1.43). The predicted mechanism holds, but `soft` guesses
   the parent by name *more* often (0.99) than the discrete backends (0.84–0.91). The
   discrete backends' `false_confirm` is above `soft`'s at d = 1 and below it at d = 2.
   Every difference is inside ±0.04 sampling error. `soft` also ends 14–20% of
   look-alike novel games `wrong`, so P0a stays the safer backend overall: at g = 0.20,
   d = 1, the unsafe rate (`wrong` + `false_confirm`) on look-alike novel games is
   0.105 for P0a against 0.226 for `soft`. The exposure is real for P0a, just not
   specific to it.
3. **Admitted look-alikes do not tax their parents; at d ≤ 2 they make the parent's
   games shorter (P5 FAIL, wrong sign).** A post-hoc trajectory read (labelled below)
   shows why. In the parent's own games the near twin is never split off. The game
   ends on a by-name guess of the parent, which has the higher weight, and the guess
   is right. A far look-alike shares the parent's category but sits off its row, and
   costs the parent's games extra questions to rule out. The cost of a near
   look-alike falls on the look-alike's own games (headline 1), not on its parent's.
4. **The detector does see far look-alikes sooner, but AUC cannot show it (P2 FAIL
   in 10 of 12 backend × flip cells).** On the look-alike side
   the prediction holds: TPR rises with d (`hard`, 10% flips: 0.946 → 0.982) and the
   median fire turn falls (11 → 9). With clean answers, discrete AUC sits at the
   ceiling (0.9988–0.9998), so it has no room to rise. Under noise the in-pool
   false-escalation rate rises with d (`hard`, 10% flips: 0.22 → 0.31). That pulls
   look-alike AUC *and* ordinary-novelty AUC down in step. So the AUC trend measures
   the in-pool side, not look-alike detectability.

## Scorecard

| prediction | verdict | reading |
|---|---|---|
| P1 look-alikes cheap, not dangerous (flip 0) | `hard` PASS, `modeswitch_guard` PASS, `modeswitch` FAIL | (a) first-appearance fire rate 1.000 at every d for all three. (b) look-alike `wrong` = 0 for `hard` and the guard; `modeswitch` 12/9/9/8/2/0/0 over d = 1…12 (of 264–266 novel look-alike games per cell). (c) extra questions, parent known, d = 1/2: `hard` 0.73/0.73, guard 0.72/0.71, `modeswitch` 0.66/0.67, all in [0.5, 1.5]. |
| P2 AUC lowest at d = 1, non-decreasing, fire turn falls | PASS only for `modeswitch` at flip 0 and 0.05; FAIL in the other 10 backend × flip cells | (c) the fire turn falls in every cell, all 12 pass. (a)/(b): see headline 4. With clean answers `hard` and the guard are flat at the ceiling, and d = 2 undercuts d = 1 by 0.00003, which fails (b). Under noise all four backends trend *down* in d (`fig_auc_by_d.png`). Unguarded `modeswitch` with clean answers rises from 0.962 to 0.9998 in step with its ordinary-novelty AUC (0.928 → 0.9998). Near look-alikes in `H_known` also hurt its other games: 44 non-look-alike `wrong` at d = 1, 0 at d = 12. |
| P3 `modeswitch` spends probes on the parent (noise, d ≤ 2) | guard PASS, `modeswitch` FAIL | Fire-turn delay over `hard` is 2–3 in all four cells for both variants. Unguarded `modeswitch` has 24–36 look-alike `wrong` per cell. The delay is **not** noise-driven: with clean answers it is the same (look-alike median fire turn 14 vs `hard` 11). It comes from re-guessing the struck parent, 0.97 re-guesses of an already-refuted name per look-alike novel game at d ≤ 2 (the path in the prereg's *Notes*). |
| P4a `false_confirm` ≈ g × P(parent guessed) | PASS, all four backends | Ratios 0.75–1.43, all within [2/3, 3/2]. P(parent guessed \| parent known) is 0.84–0.91 for the discrete backends (≥ 0.8 holds) and 0.99–1.00 for `soft`. |
| P4b discrete `false_confirm` ≥ `soft`'s | FAIL, all three discrete | Holds at d = 1 in both g, fails at d = 2 in both g. Table below. |
| P5 parent tax +0.5–1.5 at d ≤ 2, none at d ≥ 6 | FAIL, all four | d ≤ 2: registered value is negative for the discrete backends (`hard` −1.58 [−2.42, −0.73] at d = 1, −0.80 [−1.53, −0.06] at d = 2) and ≈ 0 for `soft` (−0.29, −0.10). d ≥ 6: the interval contains 0 everywhere except `soft` at d = 6 (+1.09 [+0.17, +2.00]). |

The prereg's falsification clause for P4 fired. P4 failing means the discrete
backends' exposure to name confusion is not worse than `soft`'s, so P0a's safety
claim needs a weaker caveat than predicted: "as exposed as `soft`", not "more
exposed".

## P4 — the G arm

`false_confirm` on look-alike novel games with the parent known; ± is the 95%
binomial half-width. Unsafe = `wrong` + `false_confirm` on all look-alike novel games.

| g | d | backend | n | `false_confirm` | P(parent guessed) | unsafe, all look-alike novel |
|---:|---:|---|---:|---:|---:|---:|
| 0.05 | 1 | `soft` | 389 | 0.041 ± 0.020 | 0.997 | 0.165 |
| | | `hard` | 341 | 0.044 ± 0.022 | 0.877 | 0.019 |
| | | `modeswitch_guard` | 344 | 0.052 ± 0.024 | 0.895 | 0.022 |
| 0.05 | 2 | `soft` | 436 | 0.055 ± 0.021 | 0.986 | 0.226 |
| | | `hard` | 340 | 0.038 ± 0.020 | 0.835 | 0.016 |
| | | `modeswitch_guard` | 340 | 0.035 ± 0.020 | 0.865 | 0.015 |
| 0.20 | 1 | `soft` | 442 | 0.172 ± 0.035 | 0.995 | 0.226 |
| | | `hard` | 380 | 0.197 ± 0.040 | 0.889 | 0.089 |
| | | `modeswitch_guard` | 390 | 0.228 ± 0.042 | 0.905 | 0.105 |
| 0.20 | 2 | `soft` | 485 | 0.194 ± 0.035 | 0.986 | 0.264 |
| | | `hard` | 366 | 0.153 ± 0.037 | 0.842 | 0.068 |
| | | `modeswitch_guard` | 376 | 0.184 ± 0.039 | 0.883 | 0.083 |

Unguarded `modeswitch` sits with the guard (0.033–0.210) and adds 27–33 `wrong` per
cell. Reported, not scored: an *admitted* look-alike stays exposed in its later
in-pool games. Its parent is guessed and confused there too, at ≈ 0.55 g for every
backend (0.106–0.119 at g = 0.20, 0.027–0.030 at g = 0.05).

## P5 — why the parent pays nothing (post-hoc, not registered)

`analyze_st2_posthoc`. The stream is the same in every job, so a parent's game g can
be paired across d: same secret, same index. For each P5 event, the paired
difference in questions (d minus d = 12) is averaged before and after the
look-alike's admission.

| backend | d | events | paired before | paired after | change [95%] |
|---|---:|---:|---:|---:|---:|
| `hard` / guard | 1 | 44 | −0.18 | −1.13 | −0.95 [−1.60, −0.29] |
| `hard` / guard | 2 | 44 | −0.31 | −0.95 | −0.64 [−1.13, −0.15] |
| `hard` / guard | 6 | 44 | +0.17 | +0.21 | +0.04 [−0.53, +0.62] |
| `soft` | 1 | 37 | −0.48 | −0.96 | −0.47 [−1.35, +0.40] |
| `soft` | 2 | 34 | +0.04 | −0.62 | −0.66 [−1.47, +0.15] |

(`hard` and the guard play identically on these games.) After admission the parent's
games at d = 1 make fewer by-name guesses than at d = 12 (2.35 vs 2.95) and ask fewer
questions (7.64 vs 8.86). Replaying one pair (`--replay`; look-alike seed 1, `hard`)
shows it directly. In the d = 1 job, Tree's games after its twin's admission run
`manmade, Vegetables, Nature, plant, guess Bush, guess Tree` and win. The twin is
never split off. In the d = 12 job the same games also ask `animal` and `Fruits` or
`food` before reaching Tree.

The paired check doesn't depend on the control games and gives the same sign as the
registered value at d ≤ 2 for the discrete backends. The registered value is about
1.6× larger (−1.58 vs −0.95 at d = 1), because the control drift at d = 1 (+1.19
questions) is larger than at d = 12 (+0.51). The drift difference is **not
explained**; it is reported in `posthoc.json` (`drift`).

## P2 — what moves with d

Exact arm, `hard`. With clean answers the guard matches it except that the fire turn
is +3. Under noise the guard's AUC is 0.004–0.018 higher, with the same trend.

| flip | d = 1 → 12: look-alike TPR | median fire turn | in-pool false escalation | look-alike AUC | ordinary AUC |
|---:|---|---|---|---|---|
| 0 | 0.939 → 0.981 | 11 → 9 | 0 → 0 | 0.9988 → 0.9998 | 0.9991 → 0.9998 |
| 0.05 | 0.939 → 0.979 | 11 → 9 | 0.118 → 0.166 | 0.935 → 0.913 | 0.937 → 0.915 |
| 0.10 | 0.946 → 0.982 | 11 → 9 | 0.222 → 0.306 | 0.879 → 0.841 | 0.882 → 0.843 |

Ordinary-novelty AUC moves with look-alike AUC to within 0.003, so the downward trend
belongs to the in-pool games. A plausible reading: near look-alikes keep a
consistent known alive when a noisy answer strikes the truth, which delays the empty
live set. This is **not checked** against trajectories.

`soft` moves the other way on the look-alike side. Its look-alike TPR *falls* with d
(clean answers: 0.837 → 0.587) and look-alike `wrong` rises (41 → 96). A far
look-alike is usually nearer some other object than its parent (census: 74–81 of 100
at d = 12), and `soft` guesses it as that object.

## Reported, not scored

* **Re-guessing a refuted name.** Both `modeswitch` variants re-guess the struck
  parent in nearly every look-alike novel game (0.97 per game at d ≤ 2, 0.55 at d =
  12, clean answers) and retract about 2.7 times per game. That is their whole 3-turn
  fire delay over `hard`. `hard` and `soft` never re-guess.
* **Repeat-question rate** (any game): discrete 0.029 at d = 1 → 0.007 at d = 12; `soft`
  0.11–0.31, rising with flip rate.
* **`soft` first-appearance fire rate** is 0.81–0.91 (not scored for `soft`), and its
  extra questions with the parent known are 2.4–3.0, about 4× the discrete backends'.
* **Census** (`census.json`): as in the build note. Distinct parents per seed 100;
  parent not nearest 2–6 of 100 at d ≤ 2, 74–81 at d = 12.

## What this means for P0a

* The look-alike tail adds no new failure path for P0a under exact guesses. No
  `wrong` at any d or flip rate, every first appearance escalates, and the cost is
  about one question plus a 3-turn later fire. Unguarded `modeswitch` would not pass.
  This is a second, independent reason (after ST-1) to keep the guard as P0a.
* P0a's safety rests on exact by-name tests, and a confusable name breaks that. On a
  look-alike whose parent is known, P0a false-confirms at ≈ 0.9 g, and an admitted
  look-alike keeps a ≈ 0.55 g exposure in its later games. `soft` is no better on
  this, and is worse overall once its `wrong` is counted. The caveat to carry
  forward: "P0a's zero-`wrong` record assumes exact by-name answers; with near-name
  confusion rate g it false-confirms look-alikes at about g, as every backend here
  does."
* The rates g = 0.05 and 0.20 are registered placeholders (prereg *Notes*). The size
  of this exposure in practice waits on ST-6 Stage 0's measured near-name false-yes
  rates.

## Possible follow-ups (not done)

* Don't reinstate a known struck only by its own exact by-name guess (prereg
  *Notes*). With exact guesses this would remove the 0.97 re-guesses and most of the
  3-turn delay. It is a real design choice only once guesses are noisy, which is
  exactly the G arm here. Under confusion, re-testing a refuted name is what a
  false "yes" exploits.
* Score P2-style claims on TPR and fire turn, or on AUC against a fixed in-pool
  baseline, rather than on pooled AUC.
* The d = 1 control drift in P5.

## Reproduce

```
# registered run (after the tag)
jobq -N 5 -L lmrl-20q-st2 taskpolicy -c utility env PYTHONHASHSEED=0 PYTHONPATH=. \
    OPENBLAS_NUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_st2 --run \
    --prereg-tag prereg-stress-st2-20261112 \
    --outdir llm_rl_scripts/twenty_questions/jev/results/stress_st2_20261112 --workers 5

# registered scoring (about 15 s)
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.analyze_st2 \
    --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st2_20261112

# post-hoc reads (about 2 min with --replay, one core)
PYTHONHASHSEED=0 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.analyze_st2_posthoc \
    --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st2_20261112 --replay
```

The build note's 20 h projection was an upper bound measured under `taskpolicy -b`;
the registered run took 6.4 h.
