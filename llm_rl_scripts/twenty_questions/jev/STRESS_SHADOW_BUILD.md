# Shadow-mode `soft` beside P0a — build note

This is the tooling for `STRESS_SHADOW_PREREG.md` (Jira EE-40, ENV-705).

* No shadow measurement on the real matrix has been read.
* The identity gate replays real committed jobs with the shadow attached. It prints only "identical" and the
  timing, and discards the shadow's output.
* The self-test and the smoke run use the synthetic matrix.

**Status (2026-11-15):** built and gated; not registered.

The environment is the stress tests' own: `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9),
`PYTHONHASHSEED=0`, and one BLAS thread per worker.

## What was built

| piece | where |
|---|---|
| an opt-in `shadow` hook on the two job runners; with it unset (the default), the committed path is unchanged | `stress_st2.play`, `stress_st3.play`, `_observers` in both |
| the passive observer. It wraps each job's own trace and keeps a `SoftPosterior` built from P0a's own likelihood, `by_name`, known pool and starting novelty prior. It records the per-game `sh_*` columns | `stress_shadow.ShadowObserver` |
| the grid: the 255 committed jobs, read from the committed run directories | `stress_shadow.plan` |
| replay with an identity check against the committed record; each job writes its own record, so a stopped run resumes | `stress_shadow.replay`, `replay_job`, `run` |
| `--plan`, `--self-test`, `--regress`, `--smoke` and `--run`. `--run` needs `--prereg-tag` and checks it | `stress_shadow.main` |
| scoring of SH0–SH5, the follow-on decision, the reported cells, and `fig_shadow.png` with a provenance caption | `analyze_shadow.py` |

* **The leader.** It is the argmax of `soft` over known objects, with ties in `known_hyps` order.
  * `soft` treats category and by-name answers as exact, so every known can reach mass 0. Then there is no
    leader, and the turn cannot disagree.
  * The smoke run found this case: object 164 "led" unrelated games because it came first in the list.
  * The prereg now states the rule.
* **Revivals** happen inside P0a's update, when the live set empties. The observer reads them from `n_revives`
  and the sole survivor afterwards. It pairs each revival with `soft`'s leader after the same answer.
* **SH0(b) runs inside every job.** For the first 50 games, a standalone `SoftPosterior` is rebuilt from the
  game's own steps, with an independently built likelihood and question bank, and must equal the shadow
  exactly. The record carries the counts.
* **The near-copy reference** is the known with the fewest attribute cells on opposite sides of 0.5. It is
  computed only for novel games.

## Prereg wording changed during the build (untagged, needs the user's go)

* **SH2 was replaced** (user: "Replace SH2"; see the next section). It now predicts that P0a guesses the recorded
  parent by name no later than the first disagreeing turn, in ≥ 0.7 of novel look-alike games whose parent is
  known, at d ≤ 2. The draft's measure is reported, not scored. A new per-game column, `sh_guesses`, records
  every by-name guess.
* **SH5 compares** `soft`'s leader *after the same answer* P0a revived on, so both views have the same
  information. The draft said "the turn before".
* **The per-game column list** now matches the build: objects are stored as indices.
* **The no-leader rule** above.

The thresholds are unchanged.

## Gates

* **Self-test: PASS** (synthetic matrix; jobq 4, rerun after the no-leader fix as jobq 7 and after the SH2
  replacement as jobq 10):
  * the shadow changes no game, on ST-3 (σ = 0.75, `no_repeat`, `always`, flip 0.05, 400 games) or ST-2 (d = 1,
    flips 0 and 0.10, 600 games);
  * every shadow posterior re-derives exactly;
  * the shadow's starting prior equals the job's committed prior column;
  * revival records match `n_retractions` game by game;
  * at flip 0 no in-pool game has a struck true object;
  * the grid has 255 jobs (147 ST-2, 72 ST-3, 36 end-game).
* **Identity on the real committed record: PASS.** Five committed jobs replay identically with the shadow on, in
  every per-game column and the scorecard, and every rederived posterior matches (jobq 5):

  | source | job | seconds |
  |---|---|---:|
  | ST-3 | `s75_true_f0_ss1_ns0_norepeat_modeswitch_guard` | 120 |
  | end-game | `s75_true_f0_ss1_ns0_norepeat_modeswitch_guard_endalways` | 151 |
  | ST-3 | `s50_true_f0.05_ss1_ns1_primary_modeswitch_guard` | 55 |
  | ST-2 | `exact_d1_la1_f0.1_s1_g0_modeswitch_guard` | 55 |
  | ST-2 | `exact_d12_la2_f0_s0_g0_modeswitch_guard` | 52 |

  After the no-leader fix, two of them were rerun (jobq 8). After the SH2 replacement, two more were replayed
  (jobq 11): ST-2 `exact_d2_la3_f0.05_s2_g0_modeswitch_guard` (105 s) and ST-3
  `s25_true_f0.05_ss2_ns2_norepeat_modeswitch_guard` (40 s). Both were identical.
* **Smoke** (synthetic matrix, 13 jobs at the grid's extremes, 4 workers; jobq 9, rerun as jobq 12 after the
  SH2 replacement). The analyzer ran on it as a plumbing check:
  * SH0(a) reads FAIL only because the smoke has 13 fresh jobs with nothing committed to compare;
  * SH0(b) and SH0(c) hold;
  * the other verdicts are synthetic-matrix outcomes, not results.

## Why SH2 was replaced (synthetic matrix only)

* On the smoke, SH2 read 0.017: `soft`'s leader at the first disagreement was almost never the parent. The
  reference equalled the parent in every game, so the mapping is right.
* A synthetic-only probe covered ST-2 at d ∈ {1, 2}, flips 0 and 0.10, look-alike seed 1, 2500 games each. It
  took 136 novel look-alike games whose parent was known:
  * 111 had a disagreeing turn. In 92 of those (83%) the parent's `soft` mass was already exactly 0 at the first
    disagreement.
  * In 89 of them, P0a had guessed the parent by name before that turn.
* So on near copies, **P0a's ordinary guess of a sole survivor already runs the exact test on the parent.** The
  "no" removes the parent from both views before they ever disagree. SH2's premise, that `soft` still leans on
  the parent while the hard view has struck it, mostly never arises.
* This is a property of the design, not of the matrix, but it has only been seen on the synthetic matrix. The
  replacement SH2 measures it directly on the real matrix.

## Timing

* Real-matrix jobs took 52–151 s each with the shadow on.
* The grid is about 4 CPU-hours: 147 ST-2 jobs at about 55 s, and 108 ST-3 jobs at 30–150 s by σ. **That is about
  45 min on 6 workers.** The prereg's estimate of 2 h was conservative.
* Workers stay under 200 MB.

## Before the tag

* **The user's go** on the thresholds and on the wording changes above. The thresholds are SH1 ≥ 0.5, SH2 ≥ 0.7
  (replaced measure), SH3 ≥ 0.5, SH4 ≤ 0.3 and SH5 ≥ +0.10.
* **Run plan after the tag:**
  * one jobq submission at `-N 6`, `--workers 6`, under `taskpolicy -c utility`, with BLAS pinned to 1 thread;
  * output to `results/stress_shadow_<date>`;
  * then `analyze_shadow --run-dir <out>`.
