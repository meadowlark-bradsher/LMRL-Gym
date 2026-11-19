# FLYW-1 build: arms, runner, analyzer, gate and smoke

Builds what `FLYW1_DECISION_PREREG.md` registers (Jira FLYW-1; Part 1 = FLYW-4, Part 2 = FLYW-5). The
prereg is **not registered yet**.

## What is built

| file | what it does |
|---|---|
| `flyw1_decision.py` | The game engine: the keyed (paired) oracle, arm A as a P0a record, arms B, C, D and C-bin on the P0a belief with exact class actions, the class belief and T-loss continuation, and Part 2's rotated hold-outs |
| `flyw1_run.py` | The runner: Part 1 and Part 2 job lists, one gzip file per finished job (resumable), the DE0 gate, the smoke |
| `analyze_flyw1.py` | Scores DE0–DE5 and ET1–ET3 and the decision rules; writes the reported tables |
| `class_maps.py` (v3) | The maps, now cut from P0a's own question tree; see `FLYW1_CLASS_MAPS.md` |
| `flyw1_a_calibration.py` | The disclosed control-arm run and the bases |
| `flyw1_seed_rule.py` | The seed rule: within-secret SE from the 5-seed smoke, and the seed count |

**How the pieces fit.**

* **Arm A** is `run_openworld_episode` unchanged (`modeswitch_guard`, K = 3, `no_repeat`, `always`). Its steps,
  with answers, are stored and scored against any map afterwards, so one A game serves all 57 maps.
* **Arms B, C, D** share one loop (`ClassGame`) on a `ModeSwitchGuarded` belief:
  * act on a class as soon as every live object is in it;
  * otherwise ask the best-scoring *splitting* question;
  * a failed action strikes the whole class exactly, with no reinstate, and then P0a's bounded reinstate runs
    as it does after an answer.
* **Part 2.** A B/D game is played once, up to its first trigger point (live set empty after reinstates, or
  20 probes). The game is identical up to that point under every trigger, so T-mass, T-loss at all three
  ratios, the max-likelihood shadow and the p_U sensitivity values are all read off the same trajectory.
* **Paired noise.** A flip is a hash of (seed, game, question). Game = secret plus repeat index, so in-pool
  repeats in Part 2 differ, while every arm asking q in the same game gets the same answer.

## Readings fixed in the build

These are where the prereg's wording needed a concrete rule. `analyze_flyw1.py` is the reference.

1. **A splitting question.** "No question scores above zero" is read on P0a's hard belief: a candidate must
   *split* the live set, meaning some answer strikes at least one live object and leaves at least one (kill
   rule: likelihood of the answer below 0.5). Without this, the graded EIG stays slightly positive on
   questions that cannot change the live set, and B would spend its budget on unresolvable objects.
2. **Unresolvable (DE0(c)) uses the kill rule,** not the binarized table. 17 matrix cells are exactly 0.5. An
   object there survives either answer, so P0a cannot separate, for example, Pendant (0.50 on "makes
   sound") from Brooch (0.40). `kill_rule_twins` lists each object's survivors.
3. **Flip 0 in Part 2.** In-pool repeats are identical at flip 0, so each is played once with weight 4.
4. **The p_U sensitivity** changes only T-loss's class belief (π(new)), not the play. It answers whether
   T-loss's results depend on being told the true novelty share.
5. **T-loss actions at a trigger point** come after the 20-probe budget when the trigger is the budget end
   (at most 2). They are counted as probes.

## Gate (DE0)

`results/flyw1_gate_20261118/gate.json`: **PASS** on the v3 maps.

* **(a)** A at flip 0 reproduces the calibration run game for game. At flip 0.10 the runner's A equals a direct
  `run_openworld_episode` with a fresh keyed oracle (30 secrets). Keyed flips are order-free; the flip rate at
  0.1 is 0.098.
* **(b)** On a one-class map, B, C and D act at the first probe and are right in all 158 games each.
* **(c)** At flip 0, B, C and D take wrong actions only on kill-rule-unresolvable objects. This was checked on
  6 maps here; the analyzer checks every flip-0 job.
* **(d)** All 57 maps match their registered hash and generator version 3.
* **Part 2:** 31 rotations, and every class keeps at least one known member.

## Smoke

`results/flyw1_smoke_20261118/smoke.json` (jobq 55).

* **Scope.** Part 1: one map per construction, flips 0.05 and 0.10, seeds 1–2. Part 2: two rotations at
  flip 0.10.
* **What was read.** Only the costs and the spread of the paired per-game differences. Its arm-versus-arm
  means were not read.
* **Cost:** 5.6 s per 158-game job. That projects to the following:

| run | hours on 6 workers |
|---|---:|
| Part 1, 5 seeds, all 57 maps | 0.66 |
| Part 1, 20 seeds | 2.46 |
| Part 2, 5 seeds, 4 in-pool repeats | 0.58 |

Neither the trimming rule (> 8 h) nor Part 2's cut (> 4 h) applies at 5 seeds.

* **Per-game spread of the paired A − B difference:** 1.9–2.8 probes and 1.9–2.4 wrong actions.

## Seed rule (user decision 2026-11-18)

The per-game SE in the first draft counted the spread *between* secrets as sampling error. The secrets are
the whole pool, so the only randomness is the answer noise. The rule registered instead:

* **Estimate within secrets.** Take the variance of the paired difference across seeds for each secret, and
  pool over secrets (one-way ANOVA within-secret mean square, 158 × 4 degrees of freedom from a 5-seed smoke).
* **SE of a pooled mean** over N (map, secret) pairs = √(Σ σᵢ²) ÷ (N √S).
* **Seed count:** S = max(5, ⌈5 (SE₅ ÷ (margin ÷ 3))²⌉) per pooled test. The run uses the largest S, capped at
  8 h on 6 workers. If the cap binds, a result within 2 SE of its bar is inconclusive.
* **Blind:** `flyw1_seed_rule.py` forms only variances and SEs. The smoke code no longer writes paired means.
  The first smoke's summary did contain them; they were never read.

**Result** (`results/flyw1_smoke_20261118/seed_rule.json`; 5-seed smoke, jobq 58). The cap binds:

| test | units | seeds needed |
|---|---|---:|
| DE1(a) probes, margin 0.1 | synthetic constructions | 5–6 |
| DE1(a) probes, margin 0.1 | `cat17` | 17–28 |
| DE1(c) wrong actions, margin 0.02 | synthetic constructions | 20–119 |
| DE1(c) wrong actions, margin 0.02 | `cat17` | 465–787 |
| DE3(b), margin 0.02 | `rand` | 16 |
| DE1(b), DE5 | all | 5 |

* **The largest need is 787 seeds** (DE1(c) on `cat17`, a single map at flip 0.10). The 8-hour cap allows 66,
  so Part 1 runs **66 seeds: 7.98 h on 6 workers**. Tests whose SE at 66 seeds leaves the result within 2 SE of
  the bar are scored inconclusive.
* **What drives the need.** Wrong actions vary a lot from seed to seed within a secret: the within-secret
  variance of the B − A difference is 2.2–6.6. DE1(c)'s margin is small (0.02 per game).
* **Part 2 at 66 seeds** would take about 7 h, past its 4-hour budget. So **Part 2 runs 5 seeds** (0.58 h), with
  SE reported.

## Commands

```bash
# gate, smoke
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --gate --out llm_rl_scripts/twenty_questions/jev/results/flyw1_gate_20261118
jobq -N 6 -L lmrl-flyw1-smoke taskpolicy -c utility /usr/bin/env PYTHONHASHSEED=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --smoke --workers 6 --out $PWD/llm_rl_scripts/twenty_questions/jev/results/flyw1_smoke_20261118
# full runs (after registration; same command resumes)
jobq -N 6 -L lmrl-flyw1-part1 taskpolicy -c utility /usr/bin/env PYTHONHASHSEED=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --part 1 --seeds 66 --workers 6 --out $PWD/llm_rl_scripts/twenty_questions/jev/results/flyw1_part1_<date>
jobq -N 6 -L lmrl-flyw1-part2 ... --part 2 --seeds 5 --workers 6 --out .../flyw1_part2_<date>
# analysis
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw1 --part1 .../flyw1_part1_<date> --part2 .../flyw1_part2_<date> --gate .../flyw1_gate_20261118/gate.json --out .../flyw1_analysis_<date>.json
```

## Part 2's own seed count (Addendum 1, item 3; 2026-11-19)

`flyw1_seed_rule_p2.py` applies the within-secret rule to Part 2's own 5 seeds (jobq 64). It was run on
2026-11-19 at about 06:00, after the addendum's tag and before any ET result was read. It writes
`results/flyw1_part2_20261118/seed_rule_p2.json`.

* **Unit:** one game, meaning (map, rotation, flip, secret, repeat, kind). It pools the noisy flips at ratio 0.2
  with the true p_U.
* **Margins:**
  * ET3: 0.02 on the difference T-loss − T-mass.
  * ET2: 0.10 on the per-map missed-call rate.
  * ET1: 0.3 on the ratio T-loss ÷ T-mass. Its SE comes from the delta method at the bar (R = 0.7). That
    needs T-mass's pooled spurious rate, which the tool forms internally and does not print.

**Result: every test needs 5 seeds, so Part 2 is not extended.**

| test | SE at 5 seeds, B | SE at 5 seeds, D | margin ÷ 3 |
|---|---:|---:|---:|
| ET1 (ratio) | 0.0059 | 0.0045 | 0.10 |
| ET3 | 0.0001 | 0.0001 | 0.0067 |
| ET2, largest of 9 maps | 0.0073 | 0.0121 | 0.033 |

An extension would have fitted: 0.059 h per extra seed for B and D, up to 64 seeds within the 4-hour budget.
Several ET2 maps have SE 0: on those, every novel game's act-or-call is the same under all 5 seeds. The 2-SE band
applies as registered, and it cannot make a zero-SE result inconclusive.

**Disclosure.** ET1's SE is scaled by T-mass's level, so a reader who knew the variance could infer that level
roughly. The level itself was not printed or read.

## DE1 extension: run and merge (Addendum 1, items 1 and 2; 2026-11-19)

* **Run:** `flyw1_run --extend .../seed_rule.json` (jobq 68, `results/flyw1_part1ext_20261119`). It runs A and B on
  seeds 67…S in the four cells whose blind DE1 S is above 66:

  | construction | flip | S |
  |---|---|---:|
  | `align` | 0.05 | 71 |
  | `align` | 0.10 | 119 |
  | `cat17` | 0.05 | 465 |
  | `cat17` | 0.10 | 787 |

  The jobs use Part 1's job ids. Regenerated seed-41 jobs equal the stored ones. Projected 0.73 h on 6 workers.
* **Merge:** `analyze_flyw1 --part1-ext <dir> --seed-rule <seed_rule.json>`.
  * In each extended cell, DE1(a) and DE1(c) are pooled over seeds 1…S, and the SE is taken at S seeds. The
    registered 66-seed entry is kept under `primary_protocol`.
  * A cell missing any seed is marked `INCOMPLETE` and keeps its 66-seed score.
  * The extension may hold only A and B jobs, none overlapping the registered run (asserted).
* **Item 1:** `DE1.INCONCLUSIVE` is set when no DE1 entry fails and at least one is inconclusive. The decision
  then reads "B is not recommended; more seeds are needed".
* **Test** (blind; only equality was printed). The 5-seed smoke was split into seeds 1–3 plus an "extension" of
  seeds 4–5. The merged cells equal the full 5-seed scores, and `primary_protocol` equals the 3-seed run. A cell
  without its B extension is flagged `INCOMPLETE` and left unchanged. No other result changes.

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw1 --part1 .../flyw1_part1_20261118 --part1-ext .../flyw1_part1ext_20261119 --part2 .../flyw1_part2_20261118 --gate .../flyw1_gate_20261118/gate.json --seed-rule .../flyw1_smoke_20261118/seed_rule.json --out .../flyw1_analysis_20261119.json
```

## Reported tables (`flyw1_report.py`, after scoring)

`flyw1_report.py` writes the numbers `FLYW1_DECISION_RESULTS.md` quotes beyond the verdicts. It reads the
analyzer's output and the Part 2 jobs, and scores nothing:

* Part 1 cells pooled per construction with the analyzer's own pooling, plus the class-map basis beside B's
  observed flip-0 probes;
* how often T-loss acts at Part 2's trigger points, and P(X\*) there;
* T-mass against T-loss costed at the *same* ratio. The analyzer's `part2.reported` block costs T-mass at 0.2
  only, so its T-loss costs at 0.5 and 0.05 are not comparable with the T-mass cost beside them. No verdict uses
  that block.

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_report --analysis .../flyw1_analysis_20261119.json --part2 .../flyw1_part2_20261118 --out .../flyw1_report_20261119.json
```
