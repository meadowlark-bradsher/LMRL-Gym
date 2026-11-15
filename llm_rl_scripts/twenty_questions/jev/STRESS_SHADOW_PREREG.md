# Shadow-mode `soft` beside P0a — pre-registration

**Status (2026-11-15):** built and gated (`STRESS_SHADOW_BUILD.md`), untagged. Jira: EE-40 (ENV-705, under EE-33). The thresholds below are
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
    `known_hyps` order, as the backends do. `soft` treats category and by-name answers as exact, so every known
    can reach mass 0. Then there is no leader: L_t is undefined, and the turn cannot disagree;
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

  ST-2 records each look-alike's parent, and SH2 uses that; the nearest-known reference is reported.
  ST-3's generator does not record parents, so there the reference is defined only by distance.
* **P0a's revivals.** A revival happens inside P0a's update, when the live set empties. At each revival the
  observer records:
  * the object P0a revived;
  * L after the same answer;
  * whether each is the secret.
* **Per-game columns** (new; the ST-2 and ST-3 columns are unchanged). Objects are stored as indices into the
  job's object list, and the analyzer derives "is the secret", "is the parent" and "is the reference" from them:
  * `sh_n_obs`: answers the shadow saw;
  * `sh_disagree`: disagreeing turns;
  * `sh_first_disagree`: the first disagreeing turn, or −1;
  * `sh_leader_first`: L at that turn, or −1;
  * `sh_leader_dec`: L at the decision, or −1 if no answer was seen;
  * `sh_struck_dec`: whether that L is struck;
  * `sh_p_leader_dec` and `sh_p_unknown_dec`;
  * `sh_turns_left_dec`;
  * `sh_ref`: the near-copy reference for a novel game, or −1;
  * `sh_revivals`: one pair per P0a revival, the revived object and L after the same answer;
  * `sh_guesses`: every by-name guess P0a made, as a turn and the object named.

  Each record also carries the committed `obj`, `in_pool`, `outcome`, `n_questions` and `n_retractions`. ST-2
  records also carry every object's parent.

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
* **SH2: on look-alikes, P0a already tests the parent by name before the two views disagree.**
  * Scope: ST-2 at d ≤ 2, all flips, novel look-alike games whose parent is already in `H_known`.
  * Measure: the share in which P0a guesses the recorded parent by name no later than the first disagreeing
    turn. In a game with no disagreeing turn, any by-name guess of the parent counts.
  * Prediction: **≥ 0.7.**
  * Why this replaced the draft's SH2. The draft predicted that `soft`'s leader at the first disagreement would
    be the parent. On the synthetic smoke that almost never happened (0.017), because the parent was usually
    already gone from both views:
    * P0a guesses a sole survivor by name, and on a near copy that is usually the parent;
    * the "no" sets the parent to 0 in both views.

    In a synthetic-only probe, the parent had `soft` mass 0 at the first disagreement in 83% of disagreeing
    look-alike games. P0a had guessed it by name first in 89 of 111. If this holds on the real matrix, the exact
    test the dual design would add on near copies is one P0a already runs.
  * Basis: ST-2 found P0a re-guesses a refuted parent about 0.97 times per game (§3.13).
  * The draft's measure (leader at the first disagreement is the parent) is reported, not scored.
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
  * Measure: the share in which L after the same answer is the secret, minus the share in which P0a's revived
    object is the secret. P0a revives right after that answer, so both views have the same information.
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
* **SH2 fails.** P0a often reaches a disagreement before it has tested the parent by name. Then near copies are
  a second place where a `soft`-chosen by-name test could act, besides noise, and the follow-on should target
  them too.
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

  That was the draft's estimate: **about 11–12 CPU-hours, about 2 h on 6 workers.** The build measured 52–151 s
  per real job with the shadow on, which puts the grid at about 4 CPU-hours, **about 45 min on 6 workers**
  (`STRESS_SHADOW_BUILD.md`).
