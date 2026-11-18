# FLYW-1 build: arms, runner, analyzer, gate and smoke

Builds what `FLYW1_DECISION_PREREG.md` registers (Jira FLYW-1; Part 1 = FLYW-4, Part 2 = FLYW-5). The
prereg is **not registered yet**. One choice is still open: the seed rule (below).

## What is built

| file | what it does |
|---|---|
| `flyw1_decision.py` | The game engine: the keyed (paired) oracle, arm A as a P0a record, arms B, C, D and C-bin on the P0a belief with exact class actions, the class belief and T-loss continuation, and Part 2's rotated hold-outs |
| `flyw1_run.py` | The runner: Part 1 and Part 2 job lists, one gzip file per finished job (resumable), the DE0 gate, the smoke |
| `analyze_flyw1.py` | Scores DE0–DE5 and ET1–ET3 and the decision rules; writes the reported tables |
| `class_maps.py` (v3) | The maps, now cut from P0a's own question tree; see `FLYW1_CLASS_MAPS.md` |
| `flyw1_a_calibration.py` | The disclosed control-arm run and the bases |

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

**The open choice: the seed rule.** The prereg says to raise seeds until the paired SE is at most a third
of the margin. Measured per game, that needs 2–20 seeds for DE1(a) but 40–512 for DE1(c), whose margin is
0.02 wrong actions. The 512 is for `cat17`, a single map. That is roughly 14,000 jobs per noisy flip, about
7 h on 6 workers.

Most of that per-game spread is *between secrets*. The 158 secrets are the whole pool, not a sample, so it is
not noise that more seeds would average away. The options are:

* the rule as written (about 7 h);
* SE from seed-to-seed variation of the pooled difference, which treats the secrets as the population and
  the noise as the only sampling error. It needs 5 seeds to estimate;
* 5 seeds, with SE reported only. The user's call on choice 2 offered pooled scoring *or* SE-driven seeds,
  and pooled scoring is already in.

## Commands

```bash
# gate, smoke
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --gate --out llm_rl_scripts/twenty_questions/jev/results/flyw1_gate_20261118
jobq -N 6 -L lmrl-flyw1-smoke taskpolicy -c utility /usr/bin/env PYTHONHASHSEED=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --smoke --workers 6 --out $PWD/llm_rl_scripts/twenty_questions/jev/results/flyw1_smoke_20261118
# full runs (after registration; same command resumes)
jobq -N 6 -L lmrl-flyw1-part1 taskpolicy -c utility /usr/bin/env PYTHONHASHSEED=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --part 1 --seeds 5 --workers 6 --out $PWD/llm_rl_scripts/twenty_questions/jev/results/flyw1_part1_<date>
jobq -N 6 -L lmrl-flyw1-part2 ... --part 2 --seeds 5 --workers 6 --out .../flyw1_part2_<date>
# analysis
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw1 --part1 .../flyw1_part1_<date> --part2 .../flyw1_part2_<date> --gate .../flyw1_gate_20261118/gate.json --out .../flyw1_analysis_<date>.json
```
