# ST-3 Endless tail — pre-registration

**Status: REVISED DRAFT, not registered.** Frozen when tagged
`prereg-stress-st3-<date>`. This revision settles what building the runner left open:
the backend list now that P0a is `modeswitch_guard`, the repeat loop, the planner
path, how the stream and the online prior are made, and the *Scoring* rules. It also
rewrites P1, whose drafted numbers the stream's own arithmetic rules out (Decision 6).
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`. Build notes and
timing: `STRESS_ST3_BUILD.md`.

## Question

§3.11's heavy tail (σ = 0.200) came from the draw order of a Zipf law over a *finite*
158 objects. Given enough games, novelty would run out. And the live gate used PY
parameters fitted on the very stream it was scored on. This study gives the loop a
genuinely unbounded supply of species and tails heavier than anything measured. It
asks whether expert load still tracks the missing mass, and whether the gate survives
a prior that is estimated online or simply wrong.

## Decisions taken in this revision

**1. Backends.** P0a became `modeswitch_guard` on 2026-11-12. The grid runs `soft`,
`hard`, `modeswitch` (unguarded, K = 3, as drafted) and `modeswitch_guard` (K = 3).
Where `modeswitch` and the guard differ, the guard speaks for P0a.

**2. The repeat loop.** `hard` and `modeswitch` hold the catch-all's mass at its
prior, so above a prior of about 0.15 they re-ask one question until the budget runs
out (`STRESS_ST0.md`, Finding 2). The stream census (`STRESS_ST3_BUILD.md`) puts the
true prior above 0.15 for the first 114 games at σ = 0, 229–250 at σ = 0.25,
637–957 at σ = 0.5, and **all 5000 games at σ = 0.75**. Without a guard, the discrete
backends' results at heavy tails would mostly measure the loop, as ST-0 warned. So:

* the **primary arm** is the published architecture (`no_repeat` off) for all four
  backends;
* a **`norepeat` arm** runs `hard` and `modeswitch_guard` with `no_repeat` on.
  `soft` does not loop and is not run there;
* P2 (the discrete gate ignores the prior) is scored on the `norepeat` arm, where the
  claim is about the gate rather than the loop;
* a new **P2b** registers what the loop does to the published arm: there, the prior
  *does* reach the discrete gate.

**3. Planner path.** At σ = 0.75 the pool can reach 1500–1850 objects, which the Python
planner cannot play in useful time. Games use the numpy planner
(`stress_fastplan.FastPlanner`, the `tie_sorted` rule, as in ST-1), rebuilt each time
`H_known` grows, through a new opt-in `planner_factory` hook in `run_accumulation`.
The self-test plays 600 games per backend through both paths with the same tie rule:
every game is identical (`STRESS_ST3_BUILD.md`). With the hooks off, the committed
§3.11 and retraction records reproduce bit for bit (regression gate).

**4. The stream.** Game n opens a new species with probability (θ + Kσ)/(θ + n), K
the distinct species so far. Otherwise it repeats seen species j with weight
(count_j − σ). Species are numbered in order of first appearance. Species 0–157 are
the 158 real objects in a seeded random order. Later ones come from the ST-0a
generator at ρ_real: a child of a real parent, every real object equally likely to be
the parent. The stream depends only on (σ, stream seed), so every prior, backend and
flip rate plays the same stream: common random numbers along those axes.

**5. The online prior.** It uses the §3.11 values (σ = 0.2, θ = 20) until game 100.
Then it refits the PY MLE (`fit_pitman_yor`, its default grid) every 100 games on the
loop's **own** history: a game counts as "new" iff it ended in an admission. So the K
in the fit is |H_known|, the same K the gate reads, and a missed novelty is invisible
to the estimate as it is to the loop. The fit never sees a future game.

**6. P1 rewritten.** The draft predicted an autonomous-share rise of more than 0.092
at σ = 0 and of less than 0.05 at σ = 0.75. The streams rule out both before any game
is played. Averaged over the three stream seeds, the true missing mass falls between
halves by about 0.033 at σ = 0 (0.039 → 0.006), and by 0.11–0.14 at σ = 0.75
(0.36–0.44 → 0.25–0.31). An autonomous-share rise is bounded by roughly that drop. The
σ = 0 clause could only pass through effects unrelated to the tail, and the σ = 0.75
clause could only pass if detection collapsed. The draft's §3.11 comparison came from
a Zipf stream over 158 objects, which is not this stream. P1 now tests what its title
says: the expert's necessary-call rate follows the stream's own missing mass.

**7. σ = 0.75 is out of P1.** At σ = 0.75 the pool passes 1000 objects, where ST-1
measured the detector near chance in snapshot mode. Whether detection holds there is
P5's question, and P1 would only repeat it. σ = 0.75 is reported for P1, not scored.

**8. Seeds.** 3 stream seeds, and under noise 2 noise seeds per stream seed. Flip 0
is deterministic and has one run per stream seed. All seeds derive from fixed
integers.

## Setup

Accumulation mode with the streaming generator (Decision 4). 5000 games per run.
Budget 20, confidence 0.90, ε = 0.05. An admitted species joins `H_known` with its
correct row. Wrong rows on admission are ST-6's subject.

| axis | levels |
|---|---|
| true tail σ | 0 (Dirichlet), 0.25, 0.5, 0.75 (θ = 20 throughout) |
| planner prior | `true` (the generating σ, θ); `online` (Decision 5); `fixed` (σ = 0.2, θ = 20, the §3.11 fit) |
| attribute-flip rate | 0, 0.05 |
| arm × backend | primary: `soft`, `hard`, `modeswitch`, `modeswitch_guard`; norepeat: `hard`, `modeswitch_guard` |
| seeds | 3 stream seeds (× 2 noise seeds at flip 0.05) |

Grid: 4 σ × 3 priors × 9 runs × 6 arm-backends = 648 jobs.

σ = 0.9 is excluded. K would pass several thousand within 5000 games, which turns this
into the crowding study (ST-1).

**Interpreter, hash seed and BLAS.** Python 3.9 (`LLM_RL`), `PYTHONHASHSEED=0`, and
one BLAS thread per worker. The runner refuses otherwise and records all three in the
summary, along with the git commit and the registration tag.

### Built before tagging

None of these change the design above. Status is in `STRESS_ST3_BUILD.md`.

* **The runner** (`stress_st3.py`): `--plan`, `--stream-census`, `--self-test` and
  `--smoke` (synthetic matrix only), and `--run` (registration-checked, resumable per
  job). Each job stores per-game columns (outcome, questions, `p_unknown_max`, fire
  turn, |H_known|, the prior used, repeat questions), the census of `H_known` every
  250 games, and the online fit path.
* **The analyzer** (`analyze_st3.py`): implements *Scoring* below. Its figure
  captions name it and the result directory.
* **Two opt-in hooks** in `openworld.run_accumulation`: `prior_fn` and
  `planner_factory`. Both default to off, and the regression gate holds with them off.

## Metrics

* Per 250-game window: the expert-call rate, split into necessary (true novelty) and
  false-positive calls; the autonomous share; novelty TPR/FPR at the gate; AUC; the
  mean prior; and the census of `H_known` (size, distinct rows, `nn_within_1`). The
  census separates the effects of tail shape from those of pool size.
* Per half (games 0–2499 and 2500–4999): the same, and the stream's own mean true
  p_new.
* **Prior calibration:** the gate's prior against the realized novelty frequency, per
  window (a reliability curve).
* The online estimate (σ̂, θ̂) over time.

## Predictions

* **P1 — expert load tracks the missing mass.** For σ ≤ 0.5 and the loop-free
  backends (`soft`, and `hard` and `modeswitch_guard` with `no_repeat`), under the
  true prior:
  * (a) the gate's TPR in the second half is within 0.05 of the first half;
  * (b) the necessary-call rate falls between halves by the same factor as the
    stream's true p_new, within ±25%.
* **P2 — with the loop removed, the discrete gate ignores the prior.** For `hard` and
  `modeswitch_guard` with `no_repeat`, novelty TPR and FPR at the gate differ by at
  most 0.02 across the `true`, `online` and `fixed` priors, at every σ. Their expert
  call fires when the live set empties. The prior only matters when it exceeds 0.90
  itself, which happens only in the first few games.
* **P2b — in the published arm, the prior reaches the discrete gate through the
  loop.** At σ = 0.75 the true prior stays above 0.15 for the whole stream, and the
  fixed prior (about 0.2 · K/n, roughly 0.07) stays below it. So `hard` and
  `modeswitch_guard` with `no_repeat` off escalate in-pool games at a rate at least
  0.10 higher under `true` than under `fixed`.
* **P3 — the soft gate depends on the prior.** At σ = 0.75 under `fixed`, which
  under-predicts novelty, `soft`'s TPR at the gate is more than 0.05 lower than under
  `true`. At σ = 0 under `fixed`, which over-predicts, `soft`'s FPR is higher than
  under `true`. If both hold, that is another argument for P0a: the discrete gate is
  robust to a misspecified novelty prior once the loop is removed, and the soft gate
  is not.
* **P4 — the online estimate is usable.** For σ ≥ 0.25, the online σ̂ at game 2000
  is within ±0.05 of the true σ, mean over seeds. After game 2000, the `online`
  arm's soft TPR is within 0.02 of the `true` arm's.
* **P5 — the ranking falls with σ, but pool size is what hurts it.** AUC falls with
  σ. Matched on |H_known|, the AUC difference between σ levels is at most 0.02. Any
  loss of separation at heavy tails comes from the larger, more crowded pool, not
  from the tail shape itself.

## Scoring

`analyze_st3.py` implements exactly these rules.

**Cells and populations.**

* A **cell** is (σ, prior, flip, arm, backend). It pools its stream seeds and, under
  noise, its noise seeds.
* *Novel* games are games whose secret was not in `H_known` when the game started.
  A **necessary call** is a novel game that ends with an expert call. A **false
  escalation** is an in-pool game that ends with one.
* **TPR** = necessary calls / novel games. **FPR** = false escalations / in-pool
  games. **AUC** is the detector AUC of `p_unknown_max`, novel games against in-pool
  games, ties at 0.5. *Autonomous* = the game ended `won`.
* **Halves** are games 0–2499 and 2500–4999. The stream's true p_new per half is the
  mean, over a σ's three stream seeds, of (θ + Kσ)/(θ + n) entering each game, K the
  distinct species so far (`stream_census.json`).
* **Loop-free arm-backends:** primary `soft`, norepeat `hard`, norepeat
  `modeswitch_guard`.

**Verdicts.** Each prediction is PASS or FAIL, and it passes only if every listed part
holds. A NaN or a missing cell counts as not holding.

**P1** is scored per loop-free arm-backend × flip ∈ {0, 0.05} × σ ∈ {0, 0.25, 0.5},
under the true prior.
* (a) |TPR(second half) − TPR(first half)| ≤ 0.05.
* (b) Let r = (necessary-call rate, second half) / (first half), and
  s = (true p_new, second half) / (first half). Then |r / s − 1| ≤ 0.25.

**P2** is scored per norepeat backend × σ, at flip 0. Across the three priors,
max − min of TPR ≤ 0.02, and max − min of FPR ≤ 0.02.

**P2b** is scored for primary `hard` and primary `modeswitch_guard`, at σ = 0.75,
flip 0: FPR(true) − FPR(fixed) ≥ 0.10.

**P3** is scored on primary `soft`, per flip ∈ {0, 0.05}.
* (a) At σ = 0.75: TPR(true) − TPR(fixed) > 0.05.
* (b) At σ = 0: FPR(fixed) > FPR(true).

**P4** is scored per σ ∈ {0.25, 0.5, 0.75} × flip ∈ {0, 0.05}.
* (a) For primary `soft` and norepeat `modeswitch_guard`: the online σ̂ at game 2000,
  averaged over the cell's runs, is within ±0.05 of σ.
* (b) For primary `soft`: |TPR(online) − TPR(true)| ≤ 0.02, over games 2000–4999.

**P5** is scored per loop-free arm-backend, under the true prior, at flip 0.
* (a) AUC is non-increasing in σ, allowing an adjacent rise of up to 0.01.
* (b) The pool-size bins are |H_known| entering the game ∈ [100, 200), [200, 400),
  [400, 800) and [800, 1600). A σ level *qualifies* for a bin when its games in the
  bin include at least 30 novel and 200 in-pool games. In each bin where at least
  two σ levels qualify, max − min AUC across them is ≤ 0.02. At least one bin must
  qualify.

**Reported, not scored.** Every metric per window and per half, for every cell
(σ = 0.75 included for P1). The reliability curve of the prior. The online fit path.
The census of `H_known` per window. The repeat-question rate. `modeswitch` against
`modeswitch_guard`. Wrong-answer rates. The autonomous share per half, against the
stream's missing-mass drop.

## What would falsify, and what it would mean

* **P1 failing:** the expert's load does not follow the missing mass at a pool the
  detector can handle. Detection drifts as the pool grows even below 600 objects,
  which would bring ST-1's size effect into the accumulation setting earlier than
  expected.
* **P2 failing:** the discrete gate depends on the prior even with the loop removed.
  That would have to be traced in the backend code (the commit share (1 − u)/n, or
  the EIG choice) before trusting P0a's "no tuning needed" reading.
* **P2b failing:** the loop does not engage at a prior of ~0.3 in accumulation mode.
  That contradicts ST-0's snapshot finding and would need a trajectory read.
* **P5 failing** (AUC differs at matched pool size): the tail shape itself degrades
  detection, perhaps through how often the background marginal is refreshed. That
  would be a new effect worth its own write-up.

## Note on what stays fixed

The expert still supplies a correct row on admission. Wrong rows on admission are
ST-6's subject. Keeping them out of this study keeps the tail effect clean.
