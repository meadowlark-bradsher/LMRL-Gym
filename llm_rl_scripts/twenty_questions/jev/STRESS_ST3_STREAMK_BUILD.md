# ST-3 stream-K follow-up — build note

This is the tooling for `STRESS_ST3_STREAMK_PREREG.md`. No game on the real matrix is
played under the new prior here. The self-test and the smoke run use the synthetic
matrix only. The regression gate replays committed ST-3 jobs, whose outcomes are
already published.

**Status (2026-11-13):** built and gated; not tagged, not run.

The environment is ST-3's: `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9),
`PYTHONHASHSEED=0`, and one BLAS thread per worker.

## What was built

| piece | where |
|---|---|
| prior kind `stream`: the generating (σ, θ), with K the stream's distinct species before game g; 1.0 while \|H_known\| = 0 | `stress_st3.Prior` |
| `--streamk` with `--plan`, `--self-test`, `--smoke` and `--run`. `--run` checks the registration against this pre-registration; the grid is the `stream` prior × `STREAMK_ARMS` | `stress_st3.py` |
| `--regress JOB_ID... --run-dir <dir>`: replays committed jobs on the real matrix, and requires every per-game column and the scorecard to match | `stress_st3.regress` |
| `check_prereg(tag, prereg)`: the path is now a parameter, and the file must exist at the tag | `stress_st3.py` |
| analyzer: pairs each stream-K job with ST-3's `true` job, scores SK0–SK3, writes `analysis.json` and `fig_streamk_vs_true.png` | `analyze_st3_streamk.py` |

No code shared with ST-3's games changed. The prior kinds `true`, `online` and
`fixed` take exactly the path they took before; the regression gate below checks it.

## Gates

* **Regression (committed ST-3 record): PASS.** Six committed jobs of
  `results/stress_st3_20261113` replay identically: every per-game column and the
  scorecard (jobq 53, 4 min). They cover all three ST-3 priors, σ = 0.25 to 0.75,
  both flip rates, and every arm-backend this follow-up uses, plus norepeat `hard`.

  ```
  ... stress_st3 --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_20261113 \
      --regress s25_true_f0_ss1_ns0_primary_soft \
                s25_true_f0.05_ss2_ns1_norepeat_modeswitch_guard \
                s50_true_f0_ss3_ns0_primary_modeswitch_guard \
                s25_online_f0.05_ss1_ns2_primary_soft \
                s50_fixed_f0_ss2_ns0_norepeat_hard \
                s75_true_f0_ss1_ns0_primary_soft
  ```

* **Self-test: PASS.** It runs on the synthetic matrix in about 30 s:

  ```
  ... stress_st3 --self-test --streamk
  ```

  It checks four things:
  * K before game g equals the stream's distinct species so far, and the prior is
    the PY formula with that K.
  * A `stream` prior without a stream is refused.
  * At σ = 0, 1500 games each of `soft` and norepeat P0a under the `stream` prior are
    identical to the `true` prior's, at 5% flips. The PY prior ignores K at σ = 0.
  * At σ = 0.75 the stream's K is ≥ |H_known| in every game, with a mean gap of 48
    species over 800 `soft` games.

## Grid

`stress_st3 --plan --streamk` gives **108 jobs of 5000 games**, 36 per arm-backend
(4 σ × (3 stream seeds at flip 0 + 6 stream × noise seeds at flip 0.05)). Job ids
replace `_true_` with `_stream_`, so the analyzer finds each pair by name. It also
asserts that the pair shares the stream and the oracle seed.

## Smoke and timing (synthetic matrix)

`stress_st3 --smoke --streamk` played the 3 arm-backends under the `stream` and
`true` priors at σ ∈ {0, 0.75}, flips {0, 0.05}, stream seed 1: 24 jobs. Under
jobq 54 on 4 workers it took 6 min.

The analyzer ran on it against itself, as a plumbing check:
* all 12 σ = 0 jobs identical;
* SK0 reads FAIL only because the smoke has 12 σ = 0 jobs, not the grid's 27;
* SK1 reads *incomplete*, since there is no σ = 0.5.

The stream-K prior costs up to about 20% more time per job at σ = 0.75.

The paired ST-3 `true` jobs took 2.0 CPU-hours on the real matrix, with a longest job
of 245 s. The stream-K grid should take about **25–30 min on 6 workers**.

## Before the tag

* **Registration needs the user's go** on the pre-registration: above all SK1's
  thresholds (0.05 and 0.15) and its registered expectation (*likelihood*), and
  SK2's wrong-rate floor (0.05).
* **Run plan after the tag:** one jobq submission at `-N 6`, `--workers 6`,
  `taskpolicy -c utility`, BLAS pinned to 1 thread, out
  `results/stress_st3_streamk_<date>`. Then
  `analyze_st3_streamk --run-dir <out> --base-dir results/stress_st3_20261113`.
