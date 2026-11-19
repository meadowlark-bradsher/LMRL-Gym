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
