# Shadow-mode `soft` beside P0a — results

* **Prereg:** `STRESS_SHADOW_PREREG.md`, tag `prereg-stress-shadow-20261115` (850967d).
* **Build:** `STRESS_SHADOW_BUILD.md`.
* **Jira:** EE-40 (ENV-705).
* **Run:** jobq 13, 2026-11-15. 255 jobs on 6 workers in 68 min. Output in `results/stress_shadow_20261115`.
* **Scoring:** `analyze_shadow --run-dir results/stress_shadow_20261115` produced `analysis.json` and
  `fig_shadow.png`.

## In one paragraph

**The question.** P0a decides from a hard list of surviving candidates. Could it borrow the *ranking* of `soft`,
a soft-belief view running silently beside it, to pick which struck candidate to re-test by name?

**The answer is no.** Under noisy answers, `soft`'s top pick is the secret only about 1 time in 10. The
prediction was at least half. At P0a's revivals `soft` beats P0a's own pick by only 4 points; 10 were needed.
By the pre-registered decision rule, the follow-on arm is **not built**.

**Near copies.** The two predictions about near copies held. On look-alikes, P0a has already tested the parent by
name before the two views ever disagree. On the crowded tail there is little to win back.

**So what.** Running `soft` beside the hard gate buys no better re-test target in these regimes. `soft`'s known
advantage under noise came when it chose its own questions. Riding along on P0a's questions, it has the same
information and cannot use it better.

## Terms

* **P0a**: the adopted backend (`modeswitch_guard`).
  * It keeps a hard survivor set: a candidate is *struck* when one answer contradicts it.
  * When the set empties, it revives the candidate with the fewest contradicting answers, at most 3 times per game,
    and tests it by name. A by-name guess is answered exactly, with no noise.
  * It calls the expert when nothing is left.
* **`soft`**: the soft posterior with ε = 0.05. A contradicting answer multiplies a candidate's weight by a small
  factor instead of striking it.
* **The shadow**: a `soft` posterior updated on exactly P0a's questions and answers. It changes nothing in the game.
* **Leader**: `soft`'s top-weighted *known* object.
* **Disagreement**: a turn on which the leader is an object P0a has struck.
* **Noisy ST-2**: the look-alike stress test with 5% or 10% of attribute answers flipped, pooled over look-alike
  distance d.
* **In-pool**: the secret is one of the known objects, so the right outcome is to name it.

## Verdicts

| | prediction | scope | measured | n | verdict |
|---|---|---|---:|---:|---|
| SH0 | mechanics hold | all 255 jobs | all identical, all re-derived, truth never struck at flip 0 | 255 | **PASS** |
| SH1 | disagreeing leader is the secret ≥ 0.5 | noisy ST-2, in-pool games that disagree | **0.092** | 56,611 | **FAIL** |
| SH2 | P0a tests the parent by name by the first disagreement ≥ 0.7 | ST-2 d ≤ 2, novel look-alikes with a known parent | **0.732** | 1,554 | **PASS** |
| SH3 | leader at the decision is the secret ≥ 0.5 | noisy ST-2, in-pool games ending with the expert or at the budget | **0.108** | 56,337 | **FAIL** |
| SH4 | leader at the decision is the secret ≤ 0.3 | ST-3 σ = 0.75, flip 0, `no_repeat` + `always`, in-pool expert games | **0.120** | 967 | **PASS** |
| SH5 | `soft` beats P0a's revival pick by ≥ +0.10 | noisy ST-2, every revival, paired | **+0.042** (0.064 vs 0.022) | 243,773 | **FAIL** |

**Follow-on** (build only if SH3 and SH5 pass): **do not build** the arm in which P0a tests `soft`'s leader at a
revival.

### SH0 in detail

* Every replayed job matched its committed record in every per-game column and in the scorecard.
* In each job, the shadow's first 50 games were re-derived from the game's own steps by a separately built
  `SoftPosterior`. They matched bit for bit, with no failures.
* At flip 0, no in-pool game had a disagreeing leader that was the secret.

## What happened

### Under noise, `soft`'s ranking does not find the flip victim (SH1, SH3, SH5)

* **The rates.**
  * When the views first disagree, the leader is the secret in 0.092 of games: 0.094 at flip 0.05 and 0.090 at
    flip 0.10.
  * In games P0a sends to the expert or ends at the budget, the leader is the secret in 0.108: 0.115 and 0.104.
* **No dependence on look-alike distance.** Both rates are nearly flat in d, rising only from about 0.06 to 0.13
  between d = 1 and d = 12 (`fig_shadow.png`, left panel). So the miss is not a look-alike effect.
* **At revivals.** The leader is the secret in 0.064 of revivals, against 0.022 for P0a's fewest-contradictions
  pick. `soft` is about 3 times better, but both are low. The gap of +0.042 is under the +0.10 bar.

**Why.** This comes from an exploratory replay, not a registered measurement: one job,
`exact_d1_la1_f0.1_s1_g0_modeswitch_guard`, via `shadow_why_leader.py`.

* There were 450 in-pool games with a disagreement. The leader was the secret in 26 of them (0.058, in line with the
  pooled 0.092).
* In the other 424, the leader that beat the secret had itself been struck by **exactly one** answer, in 99.3% of
  games. That answer was a confident contradiction: median answer likelihood 0.08, and only 6.7% at 0.3 or above.
  It was not a borderline cell.
* The secret was never at mass 0. It sat at median rank 6 (0 = the leader), at about 0.63 of the leader's weight.

A flipped answer leaves the true object looking exactly like any other object that one confident answer
contradicted. Many known objects are in that position, so the secret is one of a crowd, and `soft`'s ordering
inside that crowd comes from the remaining answers. Those answers were P0a's questions, chosen to split P0a's
survivors, not to separate this crowd.

**This is where the prediction went wrong.** Its basis was the retraction study, where `soft` recovered
0.136 more in-pool games than `hard` at 10% flips. But there `soft` *chose its own questions*. Riding on P0a's
questions, its ranking carries no such advantage. This reading is an interpretation of the two runs, not something
this study tested directly.

### On near copies, P0a already runs the test (SH2)

* In 0.732 of novel look-alike games whose parent is known (d ≤ 2), P0a guesses the parent by name no later than
  the first disagreeing turn. It guesses the parent at some point in 0.769 of them.
* The exact "no" removes the parent from both views. That is why the draft's measure, whether the leader at the
  first disagreement is the parent, is only 0.040 (n = 1,180; reported, not scored).
* Across all flips and all d in noisy ST-2, the share is lower: 0.546 (0.577 at flip 0.05 and 0.516 at flip
  0.10). The scored scope is d ≤ 2.

### On the crowded tail, there is little to win back (SH4)

* At σ = 0.75 under `no_repeat` + `always`, the configuration that is both autonomous and safe, the leader is the
  secret in 0.120 of the 967 in-pool expert calls. Those calls are the 0.099 false-escalation rate from the
  end-game study.
* At smaller σ the share is higher (0.40 at σ = 0.5 and 0.67 at σ = 0.25), but only 161 and 6 such games remain
  there.

### Reported, not scored

* **Cost of a `soft`-chosen test.** In novel games that P0a sent to the expert (the right outcome), 0.725 had a
  disagreeing leader and a turn left in noisy ST-2. A follow-on would mostly spend that turn on a "no" and then
  escalate anyway.
* **Wrong answers under ST-3's original end rule.** At σ = 0.75 with `no_repeat`, the wrong rate is 0.115.
  `soft`'s leader is the secret in 0.059 of those games. So at most about 6% of them could be turned into wins
  this way.
* **How often the views disagree.** In noisy ST-2, at least one turn disagrees in 0.25 of games, and the first
  disagreement comes at turn 9 (median; quartiles 8 and 11).
* Per-cell tables are in `analysis.json`.

## What this means

* **For the dual design.** The SH1 and SH5 failure branches in the prereg apply. `soft`'s ranking adds little
  to P0a's own pick under noise. The remaining lever is **planning**: questions that separate the
  struck-once crowd, which is where `soft`'s own gains came from. Testing is not that lever. This study did not
  build or measure planning.
* **For the hard gate.** Nothing here weakens it. The shadow found no `soft`-recoverable value that the hard gate is
  leaving behind in these regimes. On near copies, P0a's ordinary by-name guess already runs the exact test.
* **Distance from the main question.** This is a 20 Questions side branch about how the defer-or-answer gate
  should be built. It does not bear on why the negotiating AI never asks. It does sharpen one design point that
  carries over: a soft view's advantage comes from the questions it chooses, not from a ranking laid over someone
  else's questions.

## Provenance

* Every number in this note except the "Why" paragraph comes from
  `results/stress_shadow_20261115/analysis.json`, written by `analyze_shadow.py`.
* The "Why" numbers come from
  `python -m llm_rl_scripts.twenty_questions.jev.shadow_why_leader exact_d1_la1_f0.1_s1_g0_modeswitch_guard`.
* Environment for both: `PYTHONHASHSEED=0` and BLAS pinned to 1 thread.
