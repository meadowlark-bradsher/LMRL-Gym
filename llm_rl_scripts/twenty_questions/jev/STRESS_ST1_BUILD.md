# ST-1 pre-tag tooling — build note

The tooling `STRESS_CROWDING_PREREG.md` lists under *Built before tagging*. Nothing
here plays a game on the real matrix in an ST-1 configuration. The parity check uses
pools outside the ST-1 grid seeds. The smoke test and the timing use the synthetic
matrix only. ST-1 has **not been tagged or run**.

Everything runs under `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9) with
`PYTHONHASHSEED=0`.

## What was built

| piece | where |
|---|---|
| numpy planner: vectorised EIG ranking, soft update, per-pool `like` and bank; `tie_sorted` | `stress_fastplan.py` |
| opt-in `planner=` hook (default None = the untouched Python path) | `openworld.run_openworld_episode` |
| repeat counter `n_repeat_questions`; rescue diagnostic `RescueObserver`; scorecard fields `repeat_rate`, `rescue_trigger_rate`, `guess_share`, `q95_questions` | `stress_harness.py` |
| vectorised census (exact; identical to the pairwise version on 3 pools) | `stress_census.py` |
| ST-1 runner: `--plan`, `--smoke` (synthetic matrix), `--run` (registration-checked) | `stress_st1.py` |
| ST-1 analyzer: the prereg's *Scoring* rules, `analysis.json`, 4 figures with provenance captions | `analyze_st1.py` |

## Finding — ties in the Python ranking are decided by float summation order

`bed.rank_questions` sorts by (−EIG, qid), intending ties to go to the qid. But
candidates of mathematically equal EIG come out of the sequential float sum a few
ulp apart. The common case is guesses of two hypotheses with equal posterior mass,
such as a generated object and the parent it copies. So in practice the summation
order decides, not the qid.

On the first parity run (40 games, generated pool of 500), the numpy and Python
EIGs agreed to 1.3e-13. Even so, `soft` parted at such near-ties in 54 of 636
rankings (largest gap 8.9e-16), and only 12 of 40 games matched end to end. A BLAS
matrix product's summation order can differ between machines, so ranking ST-1 by
raw floats would make the study machine-dependent.

The numpy path therefore rounds EIG to 9 decimals before the sort (`tie_sorted`),
so a tie goes to the qid as intended. The parity reference (`SnapPythonPlanner`) is
the Python EIG and update under the same rule. The default Python path is unchanged:
committed results do not move.

## Parity (ST-0f): 200 games

Run on 2026-11-11 under Python 3.9.21, `PYTHONHASHSEED=0`, numpy 1.26.4, OpenBLAS
0.3.21, arm64. Output: `results/stress_st0f_20261111/parity.json`. **PARITY PASS,
12/12 cells.**

The cells are 2 pools × 3 backends × flip {0, 0.10}, with 200 games each. The pools
are the real 158 objects and a generated pool of 500 at ρ_real, from seed 9001,
outside the ST-1 grid. Each game is played under a 0.091 prior, with 20 held-out
novelty secrets per pool.

| pool | worst max \|ΔEIG\| | worst max \|Δposterior\| | argmax mismatches | identical games |
|---|---:|---:|---:|---:|
| real 158 | 3.8e-14 | 7.8e-16 | 0 of 12,819 rankings | 1200 / 1200 |
| generated 500 | 1.3e-13 | 1.6e-15 | 0 of 18,091 rankings | 1200 / 1200 |

The tolerance was 1e-9. The posterior gap is 0 for the survivor backends, which
don't update a posterior. Under the tie rule, the numpy path plays the same game as
the Python reference in every one of the 2400 games.

## Timing and the runtime-cap decision

Output: `results/stress_st0f_20261111/timing.json`. These are seconds per game on
pools built from the **synthetic** matrix, so no ST-1 outcome was produced; 5 games
per row (2 on the Python path). The figures are indicative, not precise.

| N | backend | numpy s/game | Python s/game | ≈ minutes per 1100-game job |
|---:|---|---:|---:|---:|
| 500 | soft | 0.043 | 0.265 | 0.8 |
| 2000 | soft | 0.110 | 1.337 | 2.0 |
| 5000 | soft | 0.365 | (not run) | 6.7 |
| 5000 | hard | 0.104 | 0.936 | 1.9 |
| 5000 | modeswitch | 0.132 | 0.933 | 2.4 |

**Runtime-cap decision (recorded before the grid, as the prereg requires):** a
`soft` job at N = 5000 is estimated at about 7 minutes, far under the 2 h cap, so
`soft` runs the full grid. Use `--soft-max-n 5000`.

The full grid of 1785 jobs comes to about 45 CPU-hours, roughly 4 h of wall time on
12 cores. Real-matrix games can run longer than synthetic ones, so allow for more.

## Regression gates

`stress_regression` under Python 3.9.21 and `PYTHONHASHSEED=0`, A/B against
`origin/main` d2a258b, with the `planner=` hook, the repeat counter, the rescue
observer and the vectorised census in place: **REGRESSION PASS, 92 of 92.**

* **committed, 46/46:** the clean arm plus every retraction cell × noise seed,
  exactly against the committed record. All cells are checked since #72.
* **ab, 46/46:** game by game.

The default Python path is unchanged. `stress_harness --self-test` and
`eval_openworld --self-test` pass.

## Smoke test

`stress_st1 --smoke` on the synthetic matrix covered 5 pools and 50 jobs, then
`analyze_st1` on its output. It exercises the census pass, the resumable per-job
store, the summary, every scoring rule and all four figures. The verdicts it prints
mean nothing: the matrix is synthetic, and the tiny 30 + 10 game mix puts the prior
at 0.25, inside the repeat loop. The smoke output is not committed.

## Reproduce

```
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.stress_harness --self-test
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.stress_fastplan --parity --games 200 \
    --out llm_rl_scripts/twenty_questions/jev/results/stress_st0f_20261111/parity.json
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.stress_fastplan --timing \
    --out llm_rl_scripts/twenty_questions/jev/results/stress_st0f_20261111/timing.json
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.stress_regression
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.stress_st1 --plan
```

## Remaining before the run

1. You read and approve `STRESS_CROWDING_PREREG.md`.
2. Tag `prereg-stress-st1-<date>` on the approved commit.
3. Run `stress_st1 --run --prereg-tag ... --soft-max-n 5000`.

## Portability notes

* **Recorded in every run.** `provenance()` records Python, numpy, scipy, BLAS,
  platform and machine alongside the git commit and hash seed, so a cross-machine
  difference can be traced. Here that is Python 3.9.21, numpy 1.26.4, scipy 1.11.1,
  OpenBLAS 0.3.21 and arm64 macOS.
* **Only tested on this machine.** No run on another machine or BLAS yet. The tie
  rule makes rankings independent of summation order, except when two EIGs within
  about 1e-13 of each other land on opposite sides of a 1e-9 rounding boundary.
  Whether that ever happens in practice is unmeasured.
* **Heavy imports.** The stress code imports torch and transformers through
  `bed.py` → `twenty_questions/env/data.py` (for the object list), although it never
  uses them. A fresh environment therefore needs `requirements.txt`'s stack, not
  just numpy, scipy and matplotlib.
* **Out of date specs.** `requirements.txt` pins scipy 1.9.0 while `LLM_RL` has
  1.11.1, and numpy isn't pinned.
