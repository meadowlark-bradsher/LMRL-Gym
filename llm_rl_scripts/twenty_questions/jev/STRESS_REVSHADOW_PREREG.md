# Reverse shadow: a hard survivor set beside a playing `soft` — pre-registration

**Status (2026-11-16):** registered at tag `prereg-stress-revshadow-20261116` with the thresholds as drafted and
the build note's wording points as built (user: "Register and queue"). Built and gated
(`STRESS_REVSHADOW_BUILD.md`). Jira: EE-41 (ENV-706, under EE-33).
**Run done 2026-11-16** (jobq 26): RS0, RS2 and RS4 pass; RS1 and RS3 fail, so the played arm is not
built. See `STRESS_REVSHADOW_RESULTS.md`. The predictions and thresholds above are unchanged from the tag.

## Question

The shadow study (`STRESS_SHADOW_RESULTS.md`, §3.14) ran `soft` passively beside P0a.

* `soft`'s ranking, laid over P0a's questions, pointed at the true object only about one time in ten under noise.
* Its known advantage under noise came from the retraction study, where `soft` *chose its own questions*: in-pool
  failure 0.039 for `soft` against 0.175 for `hard` at 10% flips.

This study reverses the roles. **`soft` plays**: it asks, guesses and ends the game as it did in the committed
runs. **A hard survivor set rides along passively.** The question is whether that hard view could be the gate,
the part that decides "answer or call the expert", without giving back `soft`'s gains. That is the rule the
retraction study arrived at: keep the view that gates the human hard.

**Where a gate can act.** A by-name guess is answered exactly. A "yes" is a confirmed win and a "no" removes the
name. So every `soft` `wrong` is a game that used all 20 questions and then answered with an unconfirmed known
(§3.14 end-game). That leaves the hard view two places to act:

* **The veto.** At the budget end, answer `soft`'s leading known only if no answer has contradicted it. Otherwise
  call the expert.
* **The alarm.** Call the expert as soon as the hard survivor set empties ("nothing I know fits"). `soft` itself
  flags only 0.10 of near-copy children at σ = 0.75, because it keeps naming the parent.

**The risk.**

* `soft`'s questions are not aimed at emptying the hard set. On a near copy, the hard view strikes the parent only
  if one of `soft`'s questions hits a cell where the copy differs. If that rarely happens, both the veto and the
  alarm stay silent exactly where `soft` goes wrong.
* Under noise, a flipped answer strikes the true object. The alarm then fires on an in-pool game that `soft`
  would have won. That is why `hard` failed in-pool at 0.175.

## Design

* **Shadow only. No game changes.**
  * Each job replays a committed `soft` job with the same stream, objects, oracle seed, prior and end rule.
  * A passive observer keeps a hard survivor set over `H_known`. It applies `HardMonotone`'s rule: a known is
    struck when an answer's likelihood under it is below 0.5, and it is never revived.
  * The observer updates on every question and answer the `soft` game produces, including by-name guesses
    answered "no", which strike the named object.
  * It uses the loop's existing `shadow` hook (`stress_st2.play`, `stress_st3.play`). It reads only; the game cannot
    see it.
* **Regimes** (committed records, replayed):

  | regime | source | jobs | what it stresses |
  |---|---|---:|---|
  | look-alikes with noise | ST-2 `results/stress_st2_20261112`, exact answerer, `soft`: d ∈ {1, 2, 3, 4, 6, 8, 12} × look-alike seeds 1–3 × flip {0, 0.05, 0.10} with noise seeds | 147 | near copies at distance d; flip-struck truths, which set off false alarms |
  | endless tail | ST-3 `results/stress_st3_20261113`, `true` prior, `soft`: σ ∈ {0, 0.25, 0.5, 0.75} × flip {0, 0.05} × seeds | 36 | near copies that accumulate; `soft`'s 0.44 wrong rate at σ = 0.75 |
  | endless tail, end rule `always` | ST-3 end-game `results/stress_st3_endgame_20261114`, `soft` + `always` | 36 | the same streams with every budget-end game escalated and admitted (crowding) |

  That is **219 jobs**. ST-2's name-confusion arm (`G_*`) is excluded: there a by-name guess is not exact, and
  both gates depend on that.
* **The two tail regimes bracket the pool.** A gated `soft` escalates more than ST-3's `soft` and less than
  `always`. So it would admit a pool between the two: ST-3 under-admits and `always` over-admits. Every tail
  measure is reported on both.
* **Per-turn record.** After each answer, the observer records:
  * the size of the hard survivor set S_t;
  * the first turn S_t is empty (the *alarm turn*), or none;
  * for every known, the number of answers that contradicted it. This is reported only, for a tolerance-1
    variant.
* **Budget-end record.** For a game that reaches the budget end without a confirmed guess:
  * `soft`'s leading known (`end_top`, as the committed record defines it);
  * whether the hard view has struck it;
  * the number of answers that contradicted it;
  * |S| at the end.
* **Counterfactual outcomes per game.** Each is exact for that game, because a gate only ends a game earlier or
  changes its last decision. Each ignores what later games would see. In ST-2 and ST-3 a gated escalation admits
  an object the committed run did not, and that is what the bracket covers.
  * **Veto (`G-end`):** at the budget end, if `soft` answered (UNKNOWN did not lead) and the hard view had struck
    `end_top`, the outcome becomes `expert`.
  * **Alarm (`G-alarm`):** if the alarm turn comes before the committed game ended, the outcome becomes `expert`
    at that turn.
  * **Both (`G-both`):** the alarm first, then the veto.
  * In the `always` regime every budget-end game was escalated. There the veto acts the other way round: it
    *answers* `end_top` when the hard view has *not* struck it. That is the outcome a gated `soft` would have
    reached on that game.
* **Scoring an unconfirmed answer.** The committed record calls every budget-end answer `wrong`. Here an
  unconfirmed answer whose `end_top` is the secret is counted separately as *right-unconfirmed*. It is neither a
  confirmed win nor a wrong answer; it is reported, as in the end-game study.
* **Per-game columns** (new; the committed columns are unchanged). Objects are stored as indices into the job's
  object list:
  * `rh_n_obs`: answers the observer saw;
  * `rh_alarm`: the alarm turn, or −1;
  * `rh_surv_end`: |S| at the end of the game;
  * `rh_end_top`: `end_top` at the budget end, or −1;
  * `rh_end_struck`: whether it was struck (−1 if no budget end);
  * `rh_end_contra`: the number of answers that contradicted it;
  * `rh_secret_contra`: the number of answers that contradicted the secret, for in-pool games.

  Each record also carries the committed `obj`, `in_pool`, `outcome` and `n_questions`. ST-2 records also carry
  every object's parent.

## Predictions

"Noisy ST-2" means the ST-2 `soft` jobs at flip 0.05 and 0.10, pooled over d. "Tail" means σ = 0.75, flip 0.
*Autonomous* means FPR ≤ 0.20 and *safe* means wrong ≤ 0.05, as in the end-game study.

* **RS0: mechanical gate.** All parts must hold.
  * (a) Every replayed job matches its committed record in every committed per-game column and in the scorecard.
  * (b) For the first 50 games of every job, the shadow's survivor set equals a standalone `HardMonotone` fed the
    game's own steps.
  * (c) At flip 0, the secret of an in-pool game is never struck. So no in-pool game raises the alarm, and the veto
    never fires on a right-unconfirmed answer.
* **RS1: the veto catches `soft`'s wrong answers.**
  * Scope: `soft`'s budget-end answers whose `end_top` is not the secret, on the tail (both brackets) and in ST-2 at
    d ≤ 2 (all flips).
  * Measure: the share in which the hard view had struck `end_top`.
  * Prediction: **≥ 0.5 in each scope.**
  * Basis: `soft` answers at the budget end only when some known still leads UNKNOWN. On a near copy every known is
    wrong somewhere. Twenty questions aimed at the leading candidates should hit at least one cell where the leader
    differs, and the hard rule strikes on one contradiction. This is the prediction the risk above threatens.
* **RS2: the raw alarm is too costly under noise.**
  * Scope: noisy ST-2, in-pool games that `soft` won with a confirmed guess.
  * Measure: the share in which the alarm turn comes before `soft`'s winning guess.
  * Prediction: **≥ 0.10 at flip 0.10.**
  * Basis: `hard`'s in-pool failure of 0.175 at 10% flips (retraction study). If this holds, a played arm needs the
    veto alone, or the alarm behind P0a's bounded reinstate, not the raw alarm.
* **RS3: on the tail, `soft` with the veto is both autonomous and safe.**
  * Scope: the tail, `soft` under `G-end`, on each bracket.
  * Measure: wrong (answered and not the secret) and FPR (in-pool games that end with the expert), over all games.
  * Prediction: **wrong ≤ 0.05 and FPR ≤ 0.20 on both brackets.**
  * For reference:
    * `soft` under ST-3's rule is wrong in 0.439, with FPR 0.002;
    * under `always` it is wrong in 0, with FPR 0.245;
    * P0a with `no_repeat` under `always` is wrong in 0, with FPR 0.099.
  * Basis: RS1 for the wrong side. For the FPR side, at flip 0 the veto never strikes a right answer (RS0(c)). It
    only turns in-pool wrong answers into expert calls, which the end-game study bounds below `always`'s 0.245.
* **RS4: the alarm sees novel objects that `soft` names.**
  * Scope: the tail, ST-3 bracket, novel games.
  * Measure: the share in which the alarm fires within the 20 questions.
  * Prediction: **≥ 0.6.**
  * Basis: P0a with `no_repeat`, asking its own questions, catches 0.716 of novel games at σ = 0.75. `soft`'s
    questions are not aimed at the hard set, so a somewhat lower rate is expected. Below 0.6 the alarm is not worth
    building into a played arm.

## Scoring

The analyzer to be built, `analyze_revshadow.py`, implements these rules. Its code is the reference where this
wording is ambiguous.

* RS0 is PASS only if every part holds. RS1 to RS4 are each PASS or FAIL on the threshold, and RS1 and RS3 must
  hold in every scope. A missing cell counts as FAIL.
* **Decision rule for the follow-on.** The played arm, where `soft` chooses the questions and a hard view gates the
  answer, is worth building **iff RS1 and RS3 both pass**.
  * If RS4 also passes, the played arm includes the alarm. If RS2 passes too, the alarm sits behind P0a's bounded
    reinstate (the `modeswitch_guard` view) rather than raw.
  * If RS3 passes on one bracket only, the played arm is still built, because playing it settles the pool. The
    note says which bracket held.
* **Reported, not scored:**
  * the veto's and alarm's cells by σ, d and flip;
  * the alarm turn against `soft`'s own end turn, which is the questions an early escalation saves;
  * the right-unconfirmed answers the veto would give up, under noise;
  * the tolerance-1 variant: strike on two contradicting answers, not one;
  * `G-both` on every regime;
  * |S| at the budget end in `soft`'s wrong games. A live set of more than one known means the hard view also
    could not decide.

## What would falsify, and what it would mean

* **RS1 fails.** `soft`'s questions leave its wrong answer standing in the hard view. Its near-copy misses are
  invisible to a hard check made on its own questions. The gate then needs questions of its own, which is P0a's
  design.
* **RS2 fails.** The raw alarm is cheap under noise, and a played arm can use it as is.
* **RS3 fails on both brackets.** Even with the veto, `soft` cannot be both autonomous and safe on the tail. P0a
  with `no_repeat` under `always` stays the only arm that is.
* **RS4 fails.** The alarm is too late or too rare on `soft`'s questions. Only the veto is worth building.

## What stays fixed, and what this is not

* **Every game is the committed game.** The shadow cannot change a question, a decision, an admission or the
  prior.
* **This measures a gate's signals and per-game outcomes, not a backend.** The played arm is not in this study.
  The per-game counterfactuals ignore admission feedback; the two tail brackets bound it.
* **The answerer is the self-consistent matrix oracle** with attribute flips, and by-name guesses are exact.
* **Cost.** No API calls. The replay re-runs `soft`'s planner, which is most of the time. The committed `soft` jobs
  took about 15 CPU-hours: ST-2 13.0, ST-3 0.6, end-game 1.1. The hard observer adds little. That is **about
  2.5–3 h on 6 workers**, to be pinned by the build's smoke run. Trimming ST-2 to d ∈ {1, 2, 4, 12} cuts about
  40%.
