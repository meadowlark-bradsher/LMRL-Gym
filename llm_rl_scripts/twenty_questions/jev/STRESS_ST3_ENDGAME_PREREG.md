# ST-3 follow-up: the end-of-game rule — pre-registration

**Status (2026-11-14):** draft, untagged. Not built. Nothing here has been played on
the real matrix. The numbers under *Why* are read from ST-3's committed per-game
columns (`st3_budget_end.py`, `st3_composition.py`); no game was replayed for them.

## Question

ST-3 (`STRESS_ST3_RESULTS.md`, FINDINGS §3.14) and its stream-K follow-up
(`STRESS_ST3_STREAMK_RESULTS.md`) found:

* At σ = 0.75 no arm is both autonomous and safe:
  * `soft` is wrong in 0.44 of games;
  * P0a with `no_repeat` is wrong in 0.115;
  * published P0a escalates almost every in-pool game.
* The prior is not the fix. Even the oracle stream-K prior leaves `soft` at 0.45 wrong
  and P0a with `no_repeat` at 0.11.
* The stream-K results named the lever as the decision at the end of a game: whether
  a known survivor or UNKNOWN wins.

This follow-up changes only that decision and asks two questions:

* Does a rule that escalates more often at the budget end make an arm both
  autonomous and safe at σ = 0.75?
* What does it cost once the extra admissions feed back into the pool?

## Why this decision, read from ST-3's committed games

* **Every ST-3 `wrong` is a budget-end game.**
  * The loop only ends a game as `wrong` after the 20th answer, with no confirmed
    guess, when the leading entry of the belief is a known object rather than UNKNOWN
    (`openworld.run_openworld_episode`).
  * A by-name guess that is answered "no" does not end the game.
  * So the label means "the budget ran out and the loop answered with an unconfirmed
    known".
* **Those games are nearly all novel.** At flip 0, as shares of all games
  (`results/stress_st3_20261113/budget_end.json`, `true` prior):

  | σ | backend | wrong, novel | wrong, in-pool | novel wrong by live set at the end: 1 / 2 / ≥ 3 |
  |---:|---|---:|---:|---|
  | 0.5 | `soft` | 0.100 | 0.001 | (no live set) |
  | 0.5 | P0a, norepeat | 0.030 | 0.004 | 0.019 / 0.007 / 0.003 |
  | 0.75 | `soft` | 0.435 | 0.004 | (no live set) |
  | 0.75 | P0a, norepeat | 0.105 | 0.010 | 0.071 / 0.027 / 0.007 |
  | 0.75 | P0a, published | 0.000 | 0.000 | none |

  * On P0a with `no_repeat` at σ = 0.75, two thirds of the novel `wrong` games end
    with exactly one live known. That is presumably the near-copy parent, whose
    by-name test needed a 21st turn.
  * The loop escalated only when UNKNOWN's prior mass beat that survivor's share.
* **`soft` leaves repeats unadmitted.** At σ = 0.75, 0.146 of `soft`'s games are
  repeats of a species it never admitted. They are novel every time they come back,
  and are wrong almost every time (`composition.json`: TPR 0.02).

What this means for the design:

* A rule that escalates every budget-end game sets `wrong` to 0 *by construction*.
  Under it, a novel game can only end with the expert, so TPR is 1.
* That rule's real content is its cost:
  * the in-pool games it escalates (FPR);
  * the unconfirmed answers it gives up that were right;
  * what the extra admissions do to later games. A larger pool of near copies may
    make in-pool games harder. The `true` prior, which reads |H_known|, also rises.
* The rules in between test *which* budget-end games need the expert.

## Design

* **Rules.** These apply at the budget end: 20 questions asked, no confirmed guess,
  and no expert call earlier in the game.
  * Let *t* be the leading known in the backend's distribution, with ties broken as
    in ST-3, and *p_t* its mass.
  * Every rule escalates when UNKNOWN leads, as ST-3 does.
  * Otherwise each rule either answers *t*, scored `wrong` as in ST-3, or escalates:

  | rule | answers *t* only if | on the survivor backends (*p_t* = (1 − u)/n_live) |
  |---|---|---|
  | `top` (ST-3) | *t* leads | escalates iff n_live > (1 − u)/u |
  | `half` | *p_t* ≥ 0.5: *t* beats UNKNOWN and every other known combined | escalates whenever ≥ 2 knowns are live (u < 0.5) |
  | `conf` | *p_t* ≥ 0.9: the mass a by-name guess needs mid-game (CONFIDENCE) | also escalates a sole survivor once u > 0.1 |
  | `always` | never | escalates every budget-end game |

  * Per game the rules are nested: `top` ⊆ `half` ⊆ `conf` ⊆ `always`.
  * Each uses only the backend's own distribution, so each is a deployable rule, not
    an oracle.
  * A rule acts only after the 20th answer. It never changes a question asked in that
    game. It changes later games only through admissions, and through the prior,
    which reads |H_known|.
* **Why a threshold on *p_t*, not a margin between *t* and UNKNOWN.**
  * On the survivor backends every live known has the same mass, so "the leading
    known" is a tie-break.
  * A near copy comes with its parent and its siblings.
  * Requiring *t* to carry a given share of the whole belief covers both cases. A
    two-way margin against UNKNOWN would ignore the sibling knowns.
* **Grid:**
  * rules `half`, `conf`, `always`;
  * arm-backends: primary `soft`, primary `modeswitch_guard` (published P0a) and
    norepeat `modeswitch_guard` (P0a with `no_repeat`);
  * σ ∈ {0, 0.25, 0.5, 0.75} × flip ∈ {0, 0.05} × ST-3's stream and noise seeds;
  * the `true` prior.

  That is **324 jobs of 5000 games**, with ST-3's budget, confidence, ε, K, streams,
  real matrix and build order. `top` is not re-run: it is ST-3's committed `true`
  cell.
* **Pairing.**
  * Job ids are ST-3's `true` ids plus `_end<rule>`, e.g.
    `s75_true_f0_ss1_ns0_norepeat_modeswitch_guard_endconf`.
  * Each job shares its pair's stream, objects, oracle seed and backend.
  * Games are identical until the first game the rule ends differently. After that,
    the pools differ.
* **New per-game columns:**
  * `end_budget`: the game reached the budget end;
  * `end_p_top` and `end_p_unknown`;
  * `end_top_is_secret`: whether *t* was the secret.

  With these, every job also records what `top` would have done in each of its own
  games.

**Definitions** (used for the headline):

* **Autonomous:** FPR ≤ 0.20, where FPR is the share of in-pool games that end with
  the expert.
* **Safe:** a wrong rate ≤ 0.05 of all games. This is SK2's floor.

On ST-3's σ = 0.75 cells these reproduce §3.14's verdict:

| backend | FPR | wrong | autonomous? | safe? |
|---|---:|---:|---|---|
| `soft` | 0.002 | 0.439 | yes | no |
| P0a, norepeat | 0.068 | 0.115 | yes | no |
| P0a, published | 0.998 | 0.000 | no | yes |

At σ = 0.5, P0a with `no_repeat` meets both (FPR 0.000, wrong 0.034). So §3.14's
"past σ = 0.5" means σ = 0.75 under these thresholds.

## Predictions

* **EG0 — mechanical gate.** All parts must hold.
  * (a) In every job, games before the first difference from the ST-3 pair are
    identical in every ST-3 column.
  * (b) The first differing game is one that ST-3 scored `wrong` at the budget end
    and the rule escalated.
  * (c) Under `always`, every job has wrong = 0 and TPR = 1, at both flips. A novel
    game cannot be won, and guess answers are never flipped.
  * (d) Published P0a at σ = 0.75 is identical to ST-3 under every rule, because ST-3
    has no budget-end `wrong` there.
* **EG1 — escalating at the budget end makes an arm autonomous and safe at σ = 0.75.**
  * Under `always` at flip 0, FPR ≤ 0.20 for both `soft` and P0a with `no_repeat`, at
    both σ = 0.5 and σ = 0.75. Safety holds by EG0(c).
  * The registered expectation is PASS. First-order, escalating every ST-3 in-pool
    `wrong` gives FPR 0.010 (`soft`) and 0.085 (P0a with `no_repeat`) at σ = 0.75.
  * The risk is second-order: the extra admissions crowd the pool until in-pool
    games run out of budget.
* **EG2 — P0a with `no_repeat` needs the survivor-against-prior rule, not just the
  ambiguity rule.** At σ = 0.75, flip 0:
  * under `half` it stays unsafe (wrong > 0.05);
  * under `conf` it is safe and autonomous (wrong ≤ 0.05, FPR ≤ 0.20).

  Basis: `half` catches only the live sets of ≥ 2, which leaves wrong ≈ 0.076
  first-order. The sole-survivor games (0.071 novel) need `conf`, which escalates
  them once the prior passes 0.1. ST-3's prior ended near 0.20.
* **EG3 — admission pays for `soft`.**
  * Gain = won(`always`) − won(`top`) for `soft` at σ = 0.75, flip 0.
  * Readings:
    * **pays** if the gain is ≥ +0.05;
    * **crowding costs** if it is ≤ −0.05;
    * **neutral** otherwise.
  * PASS only on *pays*.
  * Basis: the 0.146 of games that are repeats of unadmitted species become in-pool
    once their first appearance is escalated. ST-3's crowded pools still win about
    0.9 of in-pool games (P0a with `no_repeat`, σ = 0.75: 0.578 won of 0.632
    in-pool).
* **EG4 — `always` gives up few right answers.**
  * Count the games where the budget ran out with the secret leading. These are
    correct, unconfirmed answers that `top` would have given and ST-3 scored `wrong`.
  * Under `always`, at σ ∈ {0.5, 0.75} and flip 0, for `soft` and P0a with
    `no_repeat`, they are ≤ 0.02 of all games.
  * Basis: ST-3's in-pool `wrong` is ≤ 0.010 of games in these cells.
* **EG5 — the rules order the outcomes.**
  * Scope: `soft` and P0a with `no_repeat`, σ ∈ {0.5, 0.75}, flip 0.
  * wrong(`top`) ≥ wrong(`half`) ≥ wrong(`conf`) ≥ wrong(`always`), and FPR in the
    reverse order.
  * Each step may break the order by up to 0.01, because admissions change the later
    games.
  * `top` is ST-3's committed cell.

## Scoring

The analyzer to be built, `analyze_st3_endgame.py`, implements these rules. Its code
is the reference where this wording is ambiguous.

* **Cells and rates:**
  * A cell is (σ, flip, arm, backend, rule), pooled over its stream seeds and, under
    noise, its noise seeds.
  * Novel game, TPR, FPR, wrong rate, autonomous share and won are as in
    `analyze_st3`, over all 5000 games.
  * Each rule's cell is compared with ST-3's `true` cell of the same key.
* **Verdicts:**
  * EG0, EG1, EG2, EG4 and EG5 are PASS only if every part holds.
  * EG3 reports its reading, and is PASS only on *pays*.
  * A missing cell or base job counts as not holding.
* **Reported, not scored:**
  * every cell at both flips and every σ, by half;
  * flip 0.05. First-order, P0a with `no_repeat` is already at FPR 0.28 at σ = 0.75
    there, so it is not autonomous under noise whatever the end rule.
  * published P0a at σ ≤ 0.5;
  * admissions and the final |H_known|;
  * the prior in the last window;
  * the unconfirmed-correct share under every rule;
  * the per-job range of every scored change.
  * For `soft`, the distribution of `end_p_top` at the budget end, split into novel
    and in-pool games. This is the near-copy margin itself, which ST-3 did not
    record.
* **Figure:** `fig_endgame_rules.png`, with wrong rate, FPR and won against the rule
  (`top` to `always`) at σ = 0.75 and 0.5, flip 0, one panel per arm-backend. The
  caption names the analyzer and both run directories.

## What would falsify, and what it would mean

* **EG1 fails** (FPR > 0.20 under `always`): escalating at the end is not free. Each
  admitted near copy makes later in-pool games run out of budget. The lever then
  moves to the question budget and the planner, not the end decision.
* **EG2's `half` part fails** (`half` is safe): the wrong answers were mostly
  ambiguous live sets. Weighing a sole survivor against the prior is unnecessary.
* **EG2's `conf` part fails:** sole survivors are wrong early in the stream, while
  the prior is still below 0.1. On a growing stream, a prior-weighted rule only
  works once the stream has shown how fast it grows; only `always` is safe from the
  start.
* **EG3 reads *crowding costs*:** admitting near copies makes their parents' games
  harder. `soft`'s low TPR was partly protecting its won rate.
* **EG4 fails:** unconfirmed answers at the budget end were often right. Then ST-3's
  `wrong` overstated unsafe answers, and `always` has a real cost that `conf`
  avoids.

## What stays fixed, and what this is not

* Everything ST-3 fixed stays fixed:
  * the `true` prior;
  * budget 20, CONFIDENCE 0.9, ε 0.05, K 3;
  * the answerer, the generator and ρ_real;
  * the build order (real objects first);
  * the admitted rows, which are always correct.
* The rule does not change when the loop escalates *during* a game, or which
  questions it asks. A rule that escalates earlier on near copies, or a planner that
  spends its last questions on the survivor against UNKNOWN, is a separate study.
* No API calls. Expected CPU time is to be measured by the build's smoke run. A first
  guess, from ST-3's paired jobs (about 2 CPU-hours per 108 jobs) plus larger pools
  under `always`, is 1–1.5 h on 6 workers.
