# `modeswitch` budget guard — design and gates

**Status (2026-11-12):** built; gate criteria fixed below before any gate run (commit
5ab0e9d); gates run and **all pass** (see *Results*). **Adopted as P0a** (2026-11-12,
see *Decision*): the production rule is now `modeswitch_guard` (`P0A_BACKEND`).
`modeswitch` itself is unchanged and stays unguarded, so every committed result
regenerates as before.

## Why

ST-1's trajectory read (`STRESS_ST1_RESULTS.md` §4) found a failure path specific to
`modeswitch`. When the live set empties on a game's last answer, `modeswitch` still
reinstates a candidate. No turn is left to test it, so the game ends with an
untested known on top and is scored `wrong`. `hard` in the same position has an empty
set, so UNKNOWN is on top and it escalates. In ST-1 this turned 210 novel-secret
escalations into `wrong` in the primary arm (613 with `no_repeat`), all at flip 0;
under noise the same path also reaches in-pool games.

## The change

`ModeSwitchedRetraction` takes `budget_guard` (default False). With it on, a reinstate
is skipped when no turn remains after the current answer; the live set stays empty,
and the game escalates as `hard` would. `modeswitch_guard` is `modeswitch` with the
guard on, registered under its own name so every runner can select it.

The episode loop (`openworld.run_openworld_episode`) tells the backend how many turns
remain by setting `turns_left` before each `observe`, on backends that have the
attribute (only the mode-switch backends). A caller that never sets it leaves the
guard inert.

**Why it can only change one kind of game.** The guard acts only on the final answer
of a game that runs to the budget, after which the loop asks nothing more. So the
questions, the answers, and the answerer's random stream are the same with and
without it. A game changes only if `modeswitch` reinstated on its final answer; such
a game ends with a sole reinstated known on top (`wrong`, one live known), and with
the guard it ends with an empty set (`expert`, zero live). Its `p_unknown_max` can
rise, because the empty set puts UNKNOWN at 1.

## Gates

Run: `stress_guard_gate --out results/stress_reinstate_guard_20261112` (Python 3.9,
`PYTHONHASHSEED=0`, BLAS pinned). All must hold:

| gate | criterion |
|---|---|
| G1 reproduction | (a) the unguarded `modeswitch` cells of the published retraction grid (stream mode, 2000 games, K ∈ {1, 3}, flips 0 / 0.05 / 0.10, 2 noise seeds) equal the committed run `openworld_retraction_20261108_213525` in every per-seed metric; (b) every flip-0 single-live `wrong` game of ST-1's `modeswitch` jobs replays to its stored outcome and `p_unknown_max` |
| G2 the guard changes only what it should | (a) in those replays, the question/answer sequence is identical under both backends; (b) the outcome changes exactly when the unguarded game reinstated on its final answer; (c) replaying every ST-1 `modeswitch` job (714 jobs, both arms) whole under `modeswitch_guard`, every game equals the stored record in every column, except games stored as `wrong` at turn 20 with one live known, which may change only to `expert` with zero live (and a changed `p_unknown_max`) |
| G3 no-op at the published anchor | (a) in the retraction grid, `modeswitch_guard` equals `modeswitch` in every per-seed metric; (b) no ST-1 game at N = 158 changes |

Reported, with no threshold (it is the measurement the guard exists for): the guarded
ST-1 cell metrics (won / safe / wrong / AUC) against the stored ones, and the number of
games changed, in-pool and novel.

Not decided here: whether `modeswitch_guard` replaces `modeswitch` as P0a. That is a
production decision for after the gates. (Decided afterwards; see *Decision*.)

## Results

Run `results/stress_reinstate_guard_20261112/` (`stress_guard_gate`, 6 workers). Gate
verdicts are in `gates.json`; every number below is in `summary.json` (rebuild with
`stress_guard_gate --out <dir> --summarize`), `trace.json`, `retraction.json` or
`compare.json` of that directory.

### Gates

| gate | measured | verdict |
|---|---|---|
| G1(a) | 6/6 unguarded retraction cells equal the committed run in every per-seed metric | pass |
| G1(b) | 1225/1225 flip-0 single-live `wrong` games replay to their stored outcome and `p_unknown_max` | pass |
| G2(a) | 1225/1225 have identical question/answer sequences under both backends | pass |
| G2(b) | 823 reinstated on the final answer; exactly those 823 changed (1225/1225 agree) | pass |
| G2(c) | 785,400 games in 714 jobs; 16,821 changed, all of them allowed (0 disallowed) | pass |
| G3(a) | 6/6 guarded retraction cells equal the unguarded ones | pass |
| G3(b) | 0 games at N = 158 changed | pass |

The 823 changes at flip 0 are the 210 (primary) + 613 (`no_repeat`) novel-secret games
that ST-1 §4 attributed to the reinstate-overrun path, so the trace and the
whole-grid replay agree on that count.

### What the guard changes in ST-1

Of 21,750 candidate games (stored `wrong` at turn 20 with one live known), 16,821
change to `expert`:

| arm | flips | in-pool changed | novel changed |
|---|---|---|---|
| primary | 0 | 0 | 210 |
| primary | 0.05 | 1225 | 627 |
| primary | 0.10 | 2234 | 636 |
| no_repeat | 0 | 0 | 613 |
| no_repeat | 0.05 | 2898 | 1765 |
| no_repeat | 0.10 | 4942 | 1671 |

The guard does not change `won` in any cell (`won_changed` = 0). It never changes a
flip-0 in-pool game. Under noise it reaches in-pool games, because a flipped answer
can eliminate the true secret and leave the live set to empty on the last answer.

**In-pool `wrong`, max over cells with N ≥ 500 (the cell that attains it):**

| arm | flips | unguarded | guarded |
|---|---|---|---|
| primary | 0 | 0.013 (500 floor) | 0.013 (unchanged) |
| primary | 0.05 | 0.038 (500 real) | 0.022 (1000 real) |
| primary | 0.10 | 0.058 (500 real) | 0.030 (500 crowded) |
| no_repeat | 0 | 0.119 (1000 twins) | 0.119 (unchanged) |
| no_repeat | 0.05 | 0.177 (5000 floor) | 0.152 (5000 floor) |
| no_repeat | 0.10 | 0.203 (5000 floor) | 0.167 (5000 floor) |

Per cell, the drop in in-pool `wrong` is 0 to 0.036 (primary, 10% flips) and 0 to 0.053
(`no_repeat`, 10% flips). The primary arm's noisy worst case is roughly halved. The
`no_repeat` arm's in-pool `wrong` is mostly not this path: its flip-0 `wrong` (up to
0.119) is untouched, and at 10% flips it falls only to 0.167.

**Novelty AUC** (min to max over the four crowding levels of a size) rises at every
N from 500 to 2000, largest at N = 500, flip 0:

| arm | N | unguarded | guarded |
|---|---|---|---|
| primary | 500 | 0.63 to 0.65 | 0.68 to 0.70 |
| primary | 1000 | 0.55 to 0.60 | 0.58 to 0.63 |
| no_repeat | 500 | 0.71 to 0.73 | 0.82 to 0.84 |
| no_repeat | 1000 | 0.59 to 0.67 | 0.65 to 0.74 |

At N = 5000 it moves by at most 0.032 and stays near 0.5. In three noisy N = 5000 cells
it falls slightly (by at most 0.004, `no_repeat` crowded at 10% flips), because
changed in-pool games also get `p_unknown_max` = 1. Every N = 500 to 2000 cell rises
(by 0.002 at least; per-cell values in `compare.json`). The rise is mechanical: a changed game ends with an empty set,
so UNKNOWN reaches 1. The guard does not stop the detector's collapse with N
(ST-1 §3). It removes one way in which the collapse showed up as confident errors.

### Reading

Within ST-1's scoring the guard has no cost. It cannot change a won game, and it
moves budget-end `wrong` to `expert` only in games where the final known was never
tested. In the primary arm it removes the flip-0 novel overrun (210 games) and roughly
halves the noisy in-pool `wrong` worst case (0.058 → 0.030). It does not touch the
repeat loop or the size effect, which remain the main ST-1 findings.

Adoption as P0a is still not decided here. What would argue against it is outside
ST-1: a deployment that rewards a final untested best guess over an escalation.

## Decision

**2026-11-12: `modeswitch_guard` is P0a.** Taken after the gates above, on their
result: the guard is free within ST-1's scoring and removes a confident-error path.
How it is applied:

* `PRODUCTION_RECOMMENDATIONS.md` P0a names `modeswitch_guard`, with the precondition
  above (an escalation must cost less than a wrong answer) stated next to it.
* `openworld_backends.P0A_BACKEND = "modeswitch_guard"` is the name new work uses
  for "P0a".
* The backend `modeswitch` keeps its unguarded behaviour (`budget_guard` still
  defaults to False). Changing that default would change stored ST-1 outcomes at
  N ≥ 500 and break their regeneration; G3 shows it would change nothing in the
  published retraction grid. Preregistered studies keep the backend names they
  registered, so ST-1 and earlier results still read `modeswitch` as unguarded.

