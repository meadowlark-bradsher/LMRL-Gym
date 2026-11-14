# ST-3 end-game follow-up — build note

This is the tooling for `STRESS_ST3_ENDGAME_PREREG.md`. No game on the real matrix is
played under a new rule here. The self-test and the smoke run use the synthetic
matrix only. The regression gate replays committed ST-3 jobs, whose outcomes are
already published.

**Status (2026-11-14):** built and gated; not registered.

The environment is ST-3's: `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9),
`PYTHONHASHSEED=0`, and one BLAS thread per worker.

## What was built

| piece | where |
|---|---|
| `end_rule` for a game that reaches the budget end: `top` (default, the committed rule), `half`, `conf`, `always` | `openworld.end_rule_escalates`, `openworld.run_openworld_episode` |
| the belief at the budget end on the episode: `end_budget`, `end_u_leads`, `end_top`, `end_p_top`, `end_p_unknown` | `openworld.OWEpisode` |
| `end_rule` passed to every game of a stream | `openworld.run_accumulation` |
| per-game columns `end_budget`, `end_u_leads`, `end_p_top`, `end_p_unknown`, `end_top_is_secret`, written only for jobs that name an `end_rule` | `stress_st3.play`, `stress_st3.Trace` |
| `--endgame` with `--plan`, `--self-test`, `--smoke` and `--run`; the grid is ST-3's `true` jobs for `ENDGAME_ARMS` × `ENDGAME_RULES`, with ids `<st3 id>_end<rule>` | `stress_st3.endgame_jobs`, `stress_st3.main` |
| analyzer: pairs each job with its ST-3 `true` job, scores EG0–EG5, writes `analysis.json` and `fig_endgame_rules.png` | `analyze_st3_endgame.py` |

* With the default `end_rule="top"` the budget-end code takes the committed
  decision: expert iff UNKNOWN is on top. It only also records the belief.
  ST-3's job records do not get the new columns, so committed jobs replay byte for
  byte. The regression gate below checks this.
* The prereg names four new columns. The build adds a fifth, `end_u_leads`, so the
  analyzer can tell, in every game of a rule job, what `top` would have done
  without recomputing a tie-break.
* Outside the budget end, `end_p_top` and `end_p_unknown` are −1 and
  `end_top_is_secret` is −1.

## Gates

* **Regression (committed ST-3 record): PASS.** Seven committed jobs of
  `results/stress_st3_20261113` replay identically, in every per-game column and the
  scorecard (jobq 75, 4 min):
  * the six of the stream-K gate;
  * norepeat P0a at σ = 0.75, the end-game study's main cell.

  ```
  ... stress_st3 --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_20261113 \
      --regress s25_true_f0_ss1_ns0_primary_soft \
                s25_true_f0.05_ss2_ns1_norepeat_modeswitch_guard \
                s50_true_f0_ss3_ns0_primary_modeswitch_guard \
                s25_online_f0.05_ss1_ns2_primary_soft \
                s50_fixed_f0_ss2_ns0_norepeat_hard \
                s75_true_f0_ss1_ns0_primary_soft \
                s75_true_f0_ss1_ns0_norepeat_modeswitch_guard
  ```

* **ST-0 open-world regression (`stress_regression`): running** (jobq 78, 6 workers).
  It covers the committed §3.11 and retraction records and the A/B against
  `origin/main` for every cell.

* **Self-test: PASS.** It runs on the synthetic matrix (jobq 76, 4.5 min):

  ```
  ... stress_st3 --self-test --endgame
  ```

  It checks:
  * **The decision on the survivor backends' shares** (1 − u)/n_live, for
    u ∈ {0.02, …, 0.3} and n_live ∈ {1, 2, 3, 6}:
    * the rules are nested;
    * `top` escalates iff n_live > (1 − u)/u;
    * `half` iff UNKNOWN leads or n_live ≥ 2;
    * `conf` also escalates a sole survivor once u > 0.1;
    * `always` always escalates.

    An unknown rule is refused.
  * **`top` given explicitly** plays exactly the default path: every ST-3 column and
    the scorecard.
  * **Pairing**, for `soft`, published P0a and P0a with `no_repeat`, at σ = 0.75, flips
    0 and 0.05, over 1200 games:
    * every rule plays games identical to `top`'s up to the first one it ends
      differently;
    * in that game only `outcome` differs, and it is a budget-end `wrong` that the
      rule escalated.
  * **In every rule job:**
    * a budget-end game with UNKNOWN on top escalates;
    * no game is `wrong` before the budget end;
    * `always` leaves no `wrong`, and sends every novel game to the expert.

## Grid

`stress_st3 --plan --endgame` gives **324 jobs of 5000 games**: 36 per arm-backend
and rule, over 3 arm-backends × 3 rules. Each 36 is 4 σ × (3 stream seeds at flip 0
+ 6 stream × noise seeds at flip 0.05). The `top` cell is ST-3's committed `true`
cell and is not re-run. The ST-3 grid (648) and the stream-K grid (108) are
unchanged.

## Smoke and timing (synthetic matrix)

`stress_st3 --smoke --endgame` played 48 jobs (jobq 77; 4 workers; 14 min wall;
55 CPU-min):
* the 3 arm-backends × 3 rules at σ ∈ {0, 0.75}, flips {0, 0.05}, stream seed 1;
* their 12 ST-3-style `true` pairs, in the same directory.

The analyzer ran on it with `--base-dir` set to the smoke directory, as a plumbing
check:
* **Pairing:** all 36 rule jobs pass EG0's pairing and budget-end checks.
* **Published P0a at σ = 0.75:** all 6 jobs are identical to their pairs.
  EG0(d) reads FAIL only because the smoke has 6 such jobs, not the grid's 27.
* **EG1, EG4 and EG5** read FAIL only on the missing σ = 0.5 cells.
* **The other verdicts are synthetic-matrix outcomes**, not results. Synthetic
  pools pack differently from the real matrix, so they say nothing about the
  registered predictions.

Cost per σ = 0.75 cell, rule job against its `true` pair (synthetic):
* `soft` is about 3× slower (325 s against 112 s), because `always` and `conf` admit
  nearly every novel object and the pool grows from about 250 to about 1500.
* The P0a arms are about 1.1× slower.

Real-matrix estimate:
* ST-3's paired `true` jobs took about 2 CPU-hours per 108.
* With the slowdowns above, the 324-job grid is about 10 CPU-hours, so **about
  1.5–2 h on 6 workers**.
* The longest job should be a `soft` job at σ = 0.75, around 4 min.
* Workers stayed near 180 MB each.
* Every job writes its own record, so a stopped run resumes.

## Before the tag

* **Registration needs the user's go** on the pre-registration, above all:
  * the definitions *autonomous* (FPR ≤ 0.20) and *safe* (wrong ≤ 0.05);
  * EG3's ±0.05;
  * EG4's 0.02;
  * EG5's 0.01 tolerance;
  * whether published P0a stays in the grid. EG0(d) makes its σ = 0.75 jobs
    identical to ST-3 by construction.
* **Run plan after the tag:** one jobq submission at `-N 6`, `--workers 6`,
  `taskpolicy -c utility`, BLAS pinned to 1 thread, out
  `results/stress_st3_endgame_<date>`. Then
  `analyze_st3_endgame --run-dir <out> --base-dir results/stress_st3_20261113`.
