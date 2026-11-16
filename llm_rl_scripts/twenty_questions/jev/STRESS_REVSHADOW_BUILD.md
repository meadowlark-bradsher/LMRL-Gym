# Reverse shadow: a hard survivor set beside a playing `soft` — build note

This is the tooling for `STRESS_REVSHADOW_PREREG.md` (Jira EE-41, ENV-706).

* No reverse-shadow measurement on the real matrix has been read.
* The identity gate replays real committed jobs with the observer attached. It prints only "identical" and
  the timing, and discards the observer's output.
* The self-test and the smoke run use the synthetic matrix.

**Status (2026-11-16):** built and gated; not registered. The thresholds and the wording points below need the
user's go before the tag.

The environment is the stress tests' own: `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9),
`PYTHONHASHSEED=0`, and one BLAS thread per worker.

## What was built

| piece | where |
|---|---|
| the passive hard observer. It wraps each job's own trace and keeps a hard survivor set over `H_known` with `HardMonotone`'s rule (struck when an answer's likelihood is below 0.5, never revived), using the game's own likelihood. A tolerance-1 set (struck on a second contradicting answer) is kept beside it. It records the per-game `rh_*` columns | `stress_revshadow.HardObserver` |
| the grid: the 219 committed `soft` jobs, read from the committed run directories | `stress_revshadow.plan` |
| replay with an identity check against the committed record; each job writes its own record, so a stopped run resumes | `stress_revshadow.replay`, `replay_job`, `run` |
| `--plan`, `--self-test`, `--regress`, `--smoke` and `--run`. `--run` needs `--prereg-tag` and checks it | `stress_revshadow.main` |
| scoring of RS0–RS4, the follow-on decision, the reported cells, and `fig_revshadow.png` with a provenance caption | `analyze_revshadow.py` |

It reuses the loop's existing `shadow` hook on `stress_st2.play` and `stress_st3.play` (built for the shadow
study) and that study's job loader. No committed code path changed.

* **Per-game columns.** Objects are stored as indices into the job's object list.
  * `rh_n_obs`: answers the observer saw.
  * `rh_alarm` and `rh_alarm1`: the turn the hard set (tolerance-1 set) emptied, or −1.
  * `rh_surv_end` and `rh_surv1_end`: their sizes at the end of the game.
  * `rh_end_top`: `soft`'s leading known at the budget end, or −1.
  * `rh_end_struck`: whether the hard view had struck it (−1 if no budget end).
  * `rh_end_contra`: the number of answers that contradicted it.
  * `rh_secret_contra`: the number of answers that contradicted the secret (in-pool games; −1 for novel).
  * `rh_u_leads`: whether UNKNOWN led at the budget end (−1 if no budget end).
  * Each record also carries the committed `obj`, `in_pool`, `outcome`, `n_questions`, `p_unknown_max`,
    `fire_turn` and `k_before`. ST-2 records also carry every object's parent.
* **The counterfactual outcomes** (`analyze_revshadow.outcomes`):
  * `base`: `soft` under the committed `top` rule. At the budget end it answers its leading known unless
    UNKNOWN leads. In the `always` regime this re-applies `top` to the escalated budget-end games.
  * `veto`: as `base`, but a budget-end answer becomes `expert` when the hard view had struck the leading known.
  * `alarm`: `expert` whenever the hard set emptied during the game. The observer sees only the answers before
    the game's last decision, so an alarm always comes before the end.
  * `both`: the alarm, then the veto.
  * `veto1`, `alarm1`, `both1`: the same with the tolerance-1 view (reported only).
  * `committed`: the committed outcome, with its budget-end answers split into wrong and right-unconfirmed.
    This lets the references in the prereg be reproduced from the run itself.

## Wording points (untagged; need the user's go)

* **RS4's measure is read literally.** It counts the novel games in which the hard set emptied during the game,
  over all novel games. A novel game that `soft` sent to the expert before the set emptied counts as a miss,
  because the game ended first. Two readings are reported beside it:
  * the alarm's rate among novel games `soft` did not send to the expert itself (its added value);
  * the share of novel games sent to the expert by either.
* **An empty `H_known`** (the first games of a stream) is an empty hard set from the start: alarm at turn 0.
* **Wrong and FPR.** *Wrong* is an unconfirmed answer that is not the secret, over all games, as the prereg
  defines it. *FPR* is in-pool games that end with the expert, over in-pool games, as in the end-game study. The
  end-game study's `wrong` also counted right-unconfirmed answers; that version is reported as
  `wrong_incl_right_unconf`, so the prereg's references can be checked against the `committed` outcome.
* **The tolerance-1 variant.** The veto fires when at least two answers contradicted the leading known; the
  alarm fires when no known has fewer than two contradicting answers.
* **RS0(c)** also checks that, at flip 0, the veto never fires on a right-unconfirmed answer, as the prereg
  states.
* **The decision rule** is coded as the prereg states it: build the played arm iff RS1 and RS3 pass, or RS1
  passes and RS3 holds on one bracket only.

The thresholds are unchanged: RS1 ≥ 0.5, RS2 ≥ 0.10, RS3 wrong ≤ 0.05 and FPR ≤ 0.20, RS4 ≥ 0.6.

## Gates

* **Self-test: PASS** (synthetic matrix; jobq 16):
  * the observer changes no game on ST-3 (σ = 0.75, flip 0.05, 400 games), the end-game `always` regime
    (σ = 0.75, flip 0, 400 games) or ST-2 (d = 1, flips 0 and 0.10, 600 games each);
  * every game's hard set re-derives exactly from a standalone `HardMonotone` built with its own likelihood and
    question bank (same survivors, same alarm turn);
  * the budget-end record agrees with the committed end columns, and every committed `wrong` is a budget-end
    answer of a known;
  * at flip 0 no in-pool secret is contradicted and no in-pool game raises the alarm;
  * the tolerance-1 view is never stricter than the hard view;
  * the grid has 219 jobs (147 ST-2, 36 ST-3, 36 end-game).
* **Identity on the real committed record: PASS.** Five committed jobs replay identically with the observer on,
  in every per-game column and the scorecard, and every re-derived hard set matches (jobq 17):

  | source | job | seconds |
  |---|---|---:|
  | ST-3 | `s75_true_f0_ss1_ns0_primary_soft` | 43 |
  | end-game | `s75_true_f0_ss1_ns0_primary_soft_endalways` | 179 |
  | ST-3 | `s50_true_f0.05_ss1_ns1_primary_soft` | 61 |
  | ST-2 | `exact_d1_la1_f0.1_s1_g0_soft` | 199 |
  | ST-2 | `exact_d12_la2_f0_s0_g0_soft` | 131 |

* **Smoke** (synthetic matrix, 9 jobs at the grid's extremes, 4 workers; jobq 18, 6 min). The analyzer ran on
  it as a plumbing check:
  * RS0(a) reads FAIL only because the smoke has fresh jobs with nothing committed to compare;
  * RS0(b) and RS0(c) hold (450 games re-derived; 15,673 flip-0 in-pool games, no violation);
  * the other verdicts are synthetic-matrix outcomes, not results.

## Timing

* Real-matrix jobs took 43–199 s each with the observer on: ST-2 about 130–200 s, ST-3 45–60 s at σ = 0.75,
  end-game about 180 s.
* The grid is about 10 CPU-hours: 147 ST-2 jobs, 36 ST-3 and 36 end-game. **That is about 1.75–2 h on 6
  workers**, under the prereg's estimate of 2.5–3 h.
* Workers stay under 200 MB.

## Before the tag

* **The user's go** on the thresholds and on the wording points above.
* **Run plan after the tag:**
  * one jobq submission at `-N 6`, `--workers 6`, under `taskpolicy -c utility`, with BLAS pinned to 1 thread;
  * output to `results/stress_revshadow_<date>`;
  * then `analyze_revshadow --run-dir <out>`.
