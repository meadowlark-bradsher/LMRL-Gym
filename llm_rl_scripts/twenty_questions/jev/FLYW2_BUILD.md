# FLYW-7 build notes

Prereg: `FLYW2_COMPOUND_PREREG.md`. Jira: FLYW-7 (epic FLYW-2).

## Files

| file | what it does |
|---|---|
| `flyw2_compound.py` | Engine: `KeyedSetOracle` (OR over the secrets, attribute flips keyed on (`flyw2`, seed, game, question)), `play_s` (today's P0a, `run_openworld_episode` with `modeswitch_guard`, unchanged, plus a soft shadow), `score` / `score_class` / `score_diagnostic` (the primary cost and the two reported readings), `pair_overlap` (d_min), `ThreeWay` (the gap-10 classifier) |
| `flyw2_calibration.py` | The disclosed flip-0 basis (jobq 72) |
| `flyw2_run.py` | Runner: `--gate` (CX0), `--smoke` (5 seeds at flips 0.05 and 0.10), `--run --seeds S` (flip 0 once, then seeds 1..S at each noisy flip). Resumable |
| `flyw2_seed_rule.py` | Blind seed tool: reads the smoke and writes only within-unit variances, SEs, seed counts and timings |
| `analyze_flyw2.py` | Analyzer: CX1–CX3, TW1–TW2, the decision rule, and the reported tables |

## Jobs

One job is one block of games at one (flip, seed):

* the 141 singles plus the 17 novels;
* one of 10 chunks of the 9,870 KK pairs (pair i goes to chunk i mod 10);
* one of 3 chunks of the 2,397 KN pairs.

That is 14 jobs per (flip, seed). Each job writes `jobs/<job_id>.json.gz` when it finishes, so a stopped run
resumes by rerunning the same command. The smoke's seeds 1–5 are the same keyed streams as the run's seeds 1–5, so
the run replays them exactly.

## Commands

```bash
W=$(pwd)   # the repo root
jobq -N 6 -L lmrl-flyw2-gate taskpolicy -c utility /usr/bin/env PYTHONHASHSEED=0 PYTHONPATH=$W \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.flyw2_run \
  --gate --workers 6 --out $W/llm_rl_scripts/twenty_questions/jev/results/flyw2_gate_20261119
```

* **Smoke:** the same command with `--smoke` and `-L lmrl-flyw2-smoke`, out `results/flyw2_smoke_20261119`.
* **Seed rule:** `flyw2_seed_rule --smoke results/flyw2_smoke_20261119`. Single core, seconds.
* **Run, after the tag:** `--run --seeds S`, out `results/flyw2_run_<date>`.
* **Scoring:**

  ```bash
  analyze_flyw2 --run results/flyw2_run_<date> \
    --gate results/flyw2_gate_20261119/gate.json \
    --seed-rule results/flyw2_smoke_20261119/seed_rule.json \
    --out results/flyw2_analysis_<date>.json
  ```

  Single core. It runs the three-way classifier on every exhaustion escalation, twice: with ε = 0.05, and with ε
  set to the true flip (reported).

## CX0 gate (jobq 77, 2026-11-19): PASS

`results/flyw2_gate_20261119/gate.json`:

* **(a)** All 158 objects played as a single secret through `KeyedSetOracle` at flip 0 match
  `run_openworld_episode` with `matrix_oracle` step for step (0 mismatches).
* **(b)** The runner's flip-0 output reproduces the calibration's `games.json.gz` record for record: 12,425
  games, 0 mismatches, 0 extra.
* **(c)** No game confirmed a name outside its secrets.
* **(d)** The keyed noise is pure, a pair's answers do not depend on member order, and the `flyw2` namespace
  differs from FLYW-1's.

## Testing the analyzer without reading outcomes

The analyzer's noisy-flip code paths were tested on a synthetic run directory: the gate's flip-0 games
relabelled as flips 0.05 and 0.10, seeds 1–2. These are disclosed calibration data, so no noisy outcome was read.
On that input it reproduces the calibration's bases:

* penalty +20.044;
* KK exhaustion escalations 0.0155, partials 0.9845, silent partials 0.9758;
* TW1 AUC 0.9379; TW2 macro recall 0.0196;
* the incident mix at w = 0.25.

It also runs the decision rule end to end. On the real flip-0 output alone it gives the same reported tables. The
synthetic directory lived in the session's temp space and is not committed.

## Clarifications fixed at build (the code is the reference)

* **Exhaustion event:** the live set is empty after an answer, or a reinstate fired on that answer (the set would
  have emptied). **Exhaustion escalation:** `end == "exhausted"`, meaning the episode's expert decision fired
  because the live set was empty after the reinstate budget.
* **Catch-all fires:** the soft shadow's P(UNKNOWN) ≥ 0.9 after some answer. The shadow observes every answer P0a
  observes, including by-name "no"s. It does not see the final "yes" of a confirmed guess, because the episode ends
  there.
* **CX1 and CX3** pool every unit and seed at a flip. **CX2** is the mean of the two flips' penalty means, with SE
  √(SE₁² + SE₂²) ÷ 2.
* **TW1 and TW2:** the SE is the standard deviation of 200 bootstrap resamples of the seeds (RNG seed 7), with
  both flips' events for a seed kept together. The inconclusive band (within 2 SE of the bar) applies to them too.
* **TW2's macro recall** divides by 3 even if a class has no events at all, so a missing class counts as recall 0.
