# FLYW-1 total-cost follow-up — build notes

Build for `FLYW1_COST_PREREG.md` (draft, **not registered**). Nothing has been run beyond the gate and a projection.
No fresh class map (generator seeds 3, 4) has been generated and no fresh noise seed (1001–1020) has been played.

## What was built

| file | change |
|---|---|
| `class_maps.py` | `--seeds` (default `1 2`): the generator seeds of the `mix05`/`mix15`/`rand` maps. `cat17` and the `align` maps are always written; they do not depend on the seeds. Refuses to write into a directory that already holds a map set |
| `flyw1_run.py` | Part 1 options `--seed-start` (first noise seed), `--arms` (e.g. `A,B,D`; no identity job), `--maps-dir` (stored in each job relative to `jev/`, so the workers load the same set), `--project-only PART1_DIR` (print hours, run nothing). With none of them, Part 1's job list is FLYW-1's, unchanged. The projection borrows a fresh seed-3/4 map's timings from the FLYW-1 map of the same K, profile and construction at seed 1/2 |
| `analyze_flyw1_cost.py` | Scores P1–P4 and S1, applies the decision rule and kill criterion, and writes the reported tables (every cell's probes, wrong actions, expert share and cost at c_wrong 5/20/50/200; c\* of B against A and of B against D, per cell and per fresh map; per-map A − B against alignment; the unscored flip-0 cells checked against FLYW-1). The margins and costs the user may still change are named constants at the top (`P1_MARGIN`, `P2_C_WRONG`, `P2_BAR`, `C_WRONG_PRIMARY`, `C_EXPERT`, `SE_UNIT`) |
| `flyw1_cost_gate.py` | The build gate and the projection (below) |

## Gate: PASS

jobq 73, `results/flyw1_cost_gate_20261119/gate.json`.

* **Maps.** `build_all(seeds=(1,))` and `(seeds=(2,))` through `write_all` reproduce all 33 FLYW-1 map files of each
  seed **byte for byte**, including `cat17` and the 8 `align` maps. So the `--seeds` path changes nothing but the seeds.
* **Flip 0.** A, B and D on the 8 `align` maps and `cat17` at flip 0, built by `part1_jobs(arms=("A","B","D"),
  maps_dir=<the regenerated set>)` and played by `play_part1`, reproduce FLYW-1 Part 1's games exactly: 19 of 19 jobs,
  every record field.
* **Jobs.** With no new options, `part1_jobs(66)` lists exactly FLYW-1 Part 1's 30,590 jobs. The cost run has 4,715 jobs
  (A 41, B 2,337, D 2,337; seeds 0 and 1001–1020), all unique, all carrying `maps_dir`. No noisy job shares an id or
  a noise seed with FLYW-1. The 115 flip-0 jobs keep FLYW-1's ids: they are deterministic replays, in the run's own
  directory.

## Projection

**0.95 h on 6 workers** (5.72 CPU-hours) from FLYW-1 Part 1's per-job `secs`, with wall time = CPU ÷ 6. FLYW-1's own
map ids stand in for the fresh set (same count, seed 1/2 in place of 3/4). Nothing is queued.

## Analyzer check on FLYW-1 Part 1 (seeds 1–20)

jobq 75, `results/flyw1_cost_analyzer_check_20261119/analysis_on_flyw1_part1_seeds1-20.json`. The analyzer was run on
FLYW-1 Part 1 with FLYW-1's maps standing in for the fresh ones (`--fresh-seeds 1 2 --max-seed 20`). This is the
prereg's basis data, already read, so it checks the code, not the hypothesis.

* **Flip 0 reproduces the basis table:** A − B at c_wrong 50 is 14.7 (`mix05`), 28.7 (`mix15`), 66.3 (`rand`); at c_wrong
  5, 1.30, 1.91, 5.47. All 27 unscored-cell checks (`align` and `cat17`, A/B/D, flip 0) match FLYW-1's cells.
* Every test passes on the basis data, as the prereg's basis table implies.

### A correction to the SE

The noisy means on seeds 1–20 sat about 2 below the 66-seed basis (`align` at flip 0.05: 23.7 against 26.0). The
cause is seed-to-seed variation that FLYW-1's SE understates:

* **The units are not independent.** FLYW-1's within-secret SE treats every (map, secret) as an independent unit. But
  A plays each (secret, seed) game once and is scored on every map, and the keyed noise is shared by every arm's game of
  that secret and seed. So the maps of one cell move together.
* **Measured on `align`, flip 0.05,** A − B at c_wrong 50, FLYW-1 Part 1:

  | seeds | SE per (map, secret) | SE per secret (maps averaged) | SD of per-seed means ÷ √S |
  |---:|---:|---:|---:|
  | 20 | 0.53 | 1.42 | 1.60 |
  | 66 | 0.31 | 0.84 | 0.75 |

  The A − B means by seed block (1–10, 11–20, 21–40, 41–66) are 24.3, 23.2, 27.8 and 26.4.
* **The analyzer therefore uses the secret as the unit** (`SE_UNIT = "secret"`). For each secret it averages the paired
  difference over the cell's maps at each seed, then takes the variance across seeds. It reports FLYW-1's
  (map, secret) SE beside it as `se_map_secret`. The corrected SE is 2.1–3.5× larger on the multi-map cells and
  unchanged on `cat17`.
* **S = 20 is still ample.** With the corrected SE, the prereg's seed rule asks for 5 seeds in every scored noisy cell,
  for P1 and P2.
* **The prereg is amended to the secret unit** (user, 2026-11-19, before registration): its Terms table, seed-count
  note and basis SE column. `flyw1_cost_basis.py` now writes both SEs to `cost_var.json` (jobq 76; the earlier
  fields are unchanged), and with the secret unit the seed rule still asks for 5 seeds everywhere.
* **FLYW-1's own verdicts do not change.** Its reported SEs on multi-map cells are understated by the same factor, but
  every scored FLYW-1 test sits far beyond its bar; the closest is DE1(b) on `mix05` at flip 0.05, 0.186 against 0.3,
  about 7 corrected SEs away.

## Run commands (after registration)

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.class_maps --seeds 3 4 --out llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_maps_<date>
```

```bash
jobq -N 6 -L lmrl-flyw1-cost taskpolicy -c utility /usr/bin/env PYTHONHASHSEED=0 PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python -m llm_rl_scripts.twenty_questions.jev.flyw1_run --part 1 --arms A,B,D --seeds 20 --seed-start 1001 --maps-dir <repo>/llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_maps_<date> --workers 6 --out <repo>/llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_<date>
```

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw1_cost --run .../flyw1_cost_<date> --maps .../flyw1_cost_maps_<date> --flyw1-analysis .../flyw1_analysis_20261119.json --out .../flyw1_cost_analysis_<date>.json
```

## Decisions (user, 2026-11-19)

All five open choices are settled and recorded in the prereg's "Decisions" section: P1 margin 2, P2 c_wrong 5, C
dropped, P4 strict, SE per secret. The analyzer's constants already match. What remains is registration: merge,
tag, generate the fresh maps, then queue the run.
