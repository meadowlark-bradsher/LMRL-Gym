# Shadow-mode `soft` beside P0a — pre-registration

**Status (2026-11-15):** draft, not built, untagged. Jira: EE-40 (ENV-705, under EE-33). The thresholds below are
proposals and need the user's go before the tag.

## Question

P0a (`modeswitch_guard`) decides whether to answer or call the expert from a hard survivor set. `soft` fails
in different places:

* **On a pool with noisy answers, `soft` recovers more.** At 10% flips, in-pool failure is 0.039 for `soft`, 0.175
  for `hard` and 0.149 for `modeswitch` (`OPENWORLD_RETRACTION_RESULTS.md`).
* **On near copies, `soft` names the parent.** ST-2 found this on look-alikes (§3.13). ST-3 found it on the
  endless tail, where `soft` is wrong in 0.44 of games at σ = 0.75 (§3.14).

The case for the hard gate is *expert deferral*. That case needs only the defer-or-answer decision to be hard,
not the whole belief. So `soft` might still contribute its ranking.

This study runs `soft` passively beside P0a on the same questions and answers, and asks two things:

* **Is the disagreement informative?** Where `soft`'s leading known has already been struck by the hard view, the
  leader should usually be the secret when a flipped answer struck it. When the secret is a near copy, the leader
  should usually be the parent.
* **How much could `soft` win back?** In the games P0a ends with the expert or at the budget end, how often is
  `soft`'s leader the secret? That bounds what a follow-on backend could gain. Such a backend would test `soft`'s
  leader by name behind the hard gate.

A by-name guess is an exact test, so it settles either case in one question:

* the true object answers yes;
* a near copy's parent answers no, is struck again, and the game goes to the expert.

The retraction study showed the belief alone cannot tell the two cases apart (ENV-702). This study measures
whether `soft`'s ranking points the test at the right object.

## Design

* **Shadow only. No game changes.**
  * Each job replays a committed P0a job with the same stream, objects, oracle seed, prior and end rule.
  * A passive observer keeps a `soft` posterior (`SoftPosterior`) over the same `H_known ∪ {UNKNOWN}`. It starts
    from the same novelty prior, uses ε = 0.05, and is updated on every question and answer P0a's game produces,
    including by-name guesses answered "no".
  * The observer uses the loop's existing `observer` hook, the one the auditability study used. It reads only;
    the game cannot see it.
* **Regimes** (committed records, replayed):

  | regime | source | jobs | what it stresses |
  |---|---|---:|---|
  | look-alikes with noise | ST-2 `results/stress_st2_20261112`, exact answerer, P0a: d ∈ {1, 2, 3, 4, 6, 8, 12} × look-alike seeds 1–3 × flip {0, 0.05, 0.10} with noise seeds | 147 | flip-struck true objects (flips), near copies (small d), and both |
  | endless tail | ST-3 `results/stress_st3_20261113`, `true` prior: published P0a and P0a with `no_repeat` × σ ∈ {0, 0.25, 0.5, 0.75} × flip {0, 0.05} × seeds | 72 | near copies that accumulate, crowding |
  | endless tail, end rule `always` | ST-3 end-game `results/stress_st3_endgame_20261114`: P0a with `no_repeat`, `always` | 36 | the configuration that is both autonomous and safe at σ = 0.75 |

  That is **255 jobs**. ST-2's name-confusion arm (`G_*`) is excluded: there a by-name guess is not an exact
  test, which is the premise of the follow-on.
* **Per-turn record.** After each answer, the observer records:
  * the hard survivor set S_t;
  * `soft`'s leading known L_t. It is the argmax of `soft` over known objects only, with ties broken in
    `known_hyps` order, as the backends do;
  * π_t(L_t) and π_t(UNKNOWN);
  * whether L_t ∈ S_t.
* **Disagreement.** A turn *disagrees* when L_t ∉ S_t: `soft`'s leader is an object the hard view has struck.
* **Decision turn.** This is the turn on which P0a's game ends:
  * the escalation turn;
  * the last by-name guess, if it was answered yes;
  * or the budget end.

  "Leader at decision" is L at that turn.
* **The near-copy reference** for a novel game is the known object closest to the secret:
  * closest means the fewest attribute cells on opposite sides of 0.5 in the committed matrix;
  * ties are broken in `known_hyps` order;
  * the comparison is against the pool at the start of the game.

  In ST-2 the build checks that this is the recorded look-alike parent. ST-3's generator does not record parents,
  so there the reference is defined only by distance.
* **P0a's revivals.** When the survivor set grows during a turn, that is a revival. At each revival the observer
  records:
  * the object P0a revived;
  * L at the turn before;
  * whether each is the secret.
* **Per-game columns** (new; the ST-2 and ST-3 columns are unchanged):
  * `sh_disagree_turns`;
  * `sh_first_disagree`: the turn, or −1;
  * `sh_leader_at_decision_is_secret`;
  * `sh_leader_at_decision_is_ref`;
  * `sh_leader_at_decision_struck`;
  * `sh_p_leader_at_decision`;
  * `sh_p_unknown_at_decision`;
  * `sh_turns_left_at_decision`;
  * `sh_revivals`;
  * `sh_revived_is_secret`;
  * `sh_leader_before_revival_is_secret`.

## Predictions

Rates are pooled over seeds within a cell. "Noisy ST-2" means the ST-2 exact-answerer jobs at flip 0.05 and 0.10,
pooled over d.

* **SH0: mechanical gate.** All parts must hold.
  * (a) Every replayed job matches its committed record in every committed per-game column and in the scorecard.
  * (b) The shadow posterior equals a standalone `SoftPosterior` fed the same questions and answers, bit for bit.
    This is checked in the self-test.
  * (c) At flip 0, no in-pool game has a disagreeing turn whose leader is the secret. With consistent answers the
    true object can never be struck.
* **SH1: under noise, a disagreeing `soft` leader is usually the true object.**
  * Scope: noisy ST-2, in-pool games with at least one disagreeing turn.
  * Measure: the share in which L at the first disagreement is the secret.
  * Prediction: **≥ 0.5.**
  * Basis: `soft` keeps 0.136 more in-pool games than `hard` at 10% flips. That is games in which one flipped
    answer barely moves it.
* **SH2: on look-alikes, a disagreeing `soft` leader is usually the parent.**
  * Scope: ST-2 at d ≤ 2, all flips, novel look-alike games with at least one disagreeing turn.
  * Measure: the share in which L at the first disagreement is the near-copy reference.
  * Prediction: **≥ 0.7.**
  * Basis: `soft` guessed the parent 0.99 of the time in ST-2's look-alike games.
* **SH3: the headroom under noise is large.**
  * Scope: noisy ST-2, P0a's in-pool games that end with the expert or at the budget end.
  * Measure: the share in which the leader at decision is the secret.
  * Prediction: **≥ 0.5.**
  * This bounds what a `soft`-chosen by-name test could win back on noise.
* **SH4: the headroom on the crowded tail is small.**
  * Scope: ST-3 σ = 0.75, flip 0, P0a with `no_repeat` under `always`. In-pool games that end with the expert.
    These are the 0.099 FPR, mostly crowding.
  * Measure: the share in which the leader at decision is the secret.
  * Prediction: **≤ 0.3.**
  * Basis: in these games `soft`'s own belief is nearly flat at the budget end. In the end-game run, the median
    leading mass in its in-pool budget-end games was 0.058. A sibling or parent is as likely to lead as the
    secret.
* **SH5: `soft` points the revival test better than P0a's own rule.**
  * Scope: P0a's revivals in noisy ST-2, paired per revival.
  * Measure: the share in which L at the turn before is the secret, minus the share in which P0a's revived object
    is the secret.
  * Prediction: **≥ +0.10.**
  * Basis: P0a revives the candidate struck by the fewest answers. At K = 1 that barely recovered anything (in-pool
    failure 0.173 against `hard`'s 0.175), so the fewest-killer pick is rarely the flip victim.

## Scoring

The analyzer to be built, `analyze_shadow.py`, implements these rules. Its code is the reference where this
wording is ambiguous.

* SH0 is PASS only if every part holds. SH1 to SH5 are each PASS or FAIL on the threshold. A missing cell counts
  as FAIL.
* **Decision rule for the follow-on.** The arm in which P0a tests `soft`'s leader at a revival, behind the
  unchanged hard gate, is worth building **iff SH3 and SH5 both pass**.
  * If SH4 also fails (headroom on the tail > 0.3), the follow-on also targets crowding.
  * Otherwise the follow-on is a noise fix only.
* **Reported, not scored:**
  * the disagreement rate per game, by regime, σ, d and flip;
  * the distribution of the first-disagreement turn;
  * π(L) at disagreement;
  * **the cost side of a `soft`-chosen test:** in novel games P0a escalated, the share with a disagreeing leader
    and a turn left. Such a test would spend that turn and then escalate anyway;
  * published P0a against `no_repeat`;
  * SH1 and SH3 at each flip separately.
  * **Upper bound on wrong answers in ST-3 under the original end rule:** among budget-end `wrong` games, how
    often `soft`'s leader is the secret.

## What would falsify, and what it would mean

* **SH1 fails.** Under noise, `soft`'s disagreeing leader is rarely the true object. Its ranking adds nothing to
  P0a's own pick, and the dual design has no lever on noise.
* **SH2 fails.** On look-alikes, `soft` does not lean on the parent at the moment of disagreement. Then §3.13's
  "`soft` names the parent" is an end-of-game effect, not a turn-by-turn one.
* **SH3 fails.** There is little to win back on noise. Do not build the follow-on for noise.
* **SH4 fails** (headroom > 0.3 on the tail). `soft`'s ranking still picks out the secret in crowded games. A
  `soft`-chosen test could cut the crowding cost the end-game study found, not only the noise cost.
* **SH5 fails.** P0a's fewest-killer rule already captures what `soft` knows at a revival. The dual design's only
  remaining lever would be planning, not testing.

## What stays fixed, and what this is not

* **Every game is the committed game.** The shadow cannot change a question, a decision, an admission or the
  prior.
* **This measures signals, not a backend.** Neither follow-on arm is in this study:
  * P0a testing `soft`'s leader at a revival;
  * `soft` choosing the questions behind the hard gate.
* **The answerer is the self-consistent matrix oracle** with attribute flips. By-name guesses are exact, so a
  real answerer that confuses names (ST-2's `G` arm) would break the follow-on's premise.
* **The ST-3 near-copy reference is the nearest known by matrix distance,** not the recorded parent.
* **Cost.** No API calls. Rough estimate from the committed runs:
  * ST-2's P0a jobs are about 6.5 CPU-hours;
  * ST-3's 108 jobs about 2.5;
  * the shadow's Python posterior over pools of up to about 1700 knowns about 2–3 more.

  That is **about 11–12 CPU-hours, about 2 h on 6 workers.** Restricting ST-2 to d ∈ {1, 2, 4, 12} cuts the run to
  about 1.5 h. The build's smoke run will pin this.
