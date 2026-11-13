# ST-3 pre-tag tooling — build note

The tooling `STRESS_TAIL_PREREG.md` lists under *Built before tagging*. Nothing here
plays a game on the real matrix. The self-test, the smoke run and the timing use the
synthetic matrix only. The stream census below plays no games: it is a property of
the Pitman–Yor streams, not an ST-3 outcome.

**Status (2026-11-13):** built and gated; not tagged, not run.

Everything runs under `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9) with
`PYTHONHASHSEED=0` and one BLAS thread per worker.

## What was built

| piece | where |
|---|---|
| ST-3 runner: `--plan`, `--stream-census`, `--self-test`, `--smoke` (synthetic matrix), `--run` (registration-checked, resumable per job) | `stress_st3.py` |
| Pitman–Yor stream with species numbered by first appearance (`py_stream`); real objects first in a seeded order, then generator children at ρ_real (`build`) | `stress_st3.py` |
| priors `true` / `fixed` / `online` (refit every 100 games on the loop's own admissions) | `stress_st3.Prior` |
| per-game columns, the census of `H_known` every 250 games, the online fit path | `stress_st3.play` |
| ST-3 analyzer: the prereg's *Scoring* rules, `analysis.json`, 2 figures with provenance captions | `analyze_st3.py` |
| two opt-in hooks in `run_accumulation`: `prior_fn` and `planner_factory` (both default off) | `openworld.py` |

The only change to shared code is the two hooks. With both off, `run_accumulation`
takes exactly the path it took before. The regression gate below checks that.

## Gates

**Regression (committed record).** `stress_regression --committed-only` replays the
§3.11 clean arm and every cell of the retraction grid, all five backends, against
the committed records. **PASS**: every record is identical (jobq 41, 4 workers,
587 s).

**Planner parity.** The numpy path is needed for speed (Prereg Decision 3). The
self-test plays 600 games on a σ = 0.5 synthetic stream under the `online` prior at
5% flips. It plays each of `soft`, `hard` and `modeswitch_guard` (with `no_repeat`)
twice: once through `FastPlanner`, and once through the Python EIG and update with
the same `tie_sorted` rule (`SnapPythonPlanner`). **600/600 games identical** for
each backend (outcome, questions, fire turn, |H_known|). The largest difference in
`p_unknown_max` is 5.8e-16 for `soft` and 0 for the others.

Self-test (synthetic matrix, about 50 s):

```
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_st3 --self-test
```

Besides parity, it checks:

* the stream is seeded and numbered by first appearance;
* σ = 0 reproduces the expected species count (108 over 20 seeds, against 111 from
  Σ θ/(θ + n));
* the species count rises with σ;
* the real objects come first, in a seeded and shuffled order, followed by
  deterministic generated rows;
* the online prior refits exactly at games 100, 200, 300 and uses the fixed values
  before that.

## Grid

`stress_st3 --plan`: **648 jobs of 5000 games**, 108 per arm-backend (4 σ × 3 priors
× (3 stream seeds at flip 0 + 3 × 2 noise seeds at flip 0.05)).

## Stream census (no games)

`stress_st3 --stream-census`, written to
`results/stress_st3_stream_census_20261113/stream_census.json`. "True p_new" is
(θ + Kσ)/(θ + n) entering each game, K the distinct species so far. It is the mass a
correctly specified gate would put on "unknown".

| σ | stream seed | species (game 2500 / 5000) | true p_new, 1st → 2nd half | games with true prior > 0.15 |
|---:|---:|---:|---|---:|
| 0 | 1 / 2 / 3 | 97 / 116, 87 / 101, 97 / 111 | 0.039 → 0.006 (all three) | 114 |
| 0.25 | 1 / 2 / 3 | 233 / 296, 197 / 239, 208 / 250 | 0.082 → 0.024, 0.076 → 0.020, 0.078 → 0.021 | 250, 229, 241 |
| 0.5 | 1 / 2 / 3 | 427 / 619, 367 / 535, 337 / 507 | 0.170 → 0.077, 0.145 → 0.068, 0.154 → 0.062 | 957, 637, 729 |
| 0.75 | 1 / 2 / 3 | 897 / 1536, 1124 / 1858, 1045 / 1817 | 0.364 → 0.250, 0.443 → 0.306, 0.412 → 0.296 | 5000 (every game) |

Two consequences shaped the revised pre-registration:

* **The drafted P1 numbers are out of reach.** The missing mass falls by only about
  0.033 between halves at σ = 0, and by 0.11–0.14 at σ = 0.75. The draft asked for an
  autonomous-share rise of more than 0.092 at σ = 0 and less than 0.05 at σ = 0.75
  (Prereg Decision 6).
* **The repeat loop is on for the whole σ = 0.75 stream** under the true prior, and
  for the first 640–960 games at σ = 0.5. Hence the `norepeat` arm, and P2 scored on
  it (Prereg Decision 2).

## Smoke and timing (synthetic matrix)

`stress_st3 --smoke` ran the full 5000-game length at σ ∈ {0, 0.75}, under the
`true` and `online` priors, at flips {0, 0.05}, with 1 stream seed and every
arm-backend: 48 jobs. Under jobq 42 on 4 workers it took 15 min wall (3490
job-seconds). Each worker kept one core at about 93% and about 200 MB.

| σ | arm-backend | mean (max) seconds per job |
|---:|---|---|
| 0 | every arm-backend | 19–28 (31) |
| 0.75 | primary `soft` | 75 (82) |
| 0.75 | norepeat `hard` / `modeswitch_guard` | 98 (127) / 126 (176) |
| 0.75 | primary `hard` / `modeswitch` / `modeswitch_guard` | 122 (184) / 169 (261) / 157 (212) |

The numpy path makes σ = 0.75 play in about two minutes per job, at pools of
~1500 objects. Interpolating σ = 0.25 and 0.5 between the two measured levels, the
648-job grid is about 11 CPU-hours: **about 2 h on 6 workers**, or 3 h on 4. The
real matrix may be slower or faster than the synthetic one. Every job writes its
result when it finishes, so a stop loses at most the jobs in flight.

## Before the tag

* Registration needs the user's go on the decisions in the revised
  pre-registration, above all the rewritten P1 and the new P2b.
* Run plan, after the tag: one jobq submission at `-N 6` with `--workers 6`,
  `taskpolicy -c utility`, BLAS pinned to 1 thread. The same command resumes after
  a stop.
