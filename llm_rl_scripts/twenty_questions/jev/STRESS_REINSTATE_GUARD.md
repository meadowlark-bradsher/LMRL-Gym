# `modeswitch` budget guard — design and gates

**Status (2026-11-12):** built; gate criteria fixed below before any gate run.
Not adopted: `modeswitch` (P0a) is unchanged, and the guard is a separate backend,
`modeswitch_guard`.

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
production decision for after the gates.
