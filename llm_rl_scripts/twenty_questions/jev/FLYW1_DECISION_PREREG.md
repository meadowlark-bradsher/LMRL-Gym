# Aim at a decision, not an identity — pre-registration (Flywheel experiment 1)

**Status (2026-11-18): REGISTERED** at tag `prereg-flyw1-decision-20261118` (commit 1762523, the merge of PR #100;
user: "Tag, and queue both parts"). Thresholds, seed counts and rules are as written below; nothing has
changed since the tag except **Addendum 1** (end of this file; tag `prereg-flyw1-addendum1-20261119`, Jira
FLYW-1 comment 10593), which was tagged while Part 1 was still running, before any Part 1 or Part 2 outcome was
read. **Run and scored 2026-11-19**: DE1 fails, so the registered decision is "class targeting does not pay on
P0a"; T-loss is not adopted. Results in `FLYW1_DECISION_RESULTS.md`. Before registration it was revised twice
the same day:

* after the user's calls on the four open choices (Jira FLYW-1, comment 10584) and a disclosed calibration run
  of the control arm (§Basis);
* during the build, after two design problems surfaced (§Build amendments): the maps were rebuilt on P0a's
  own question tree (generator version 3), the bars were rescaled to the new basis, and ET2 was redefined.

The arms, runner, analyzer and DE0 gate are built (`FLYW1_BUILD.md`). Nothing has been run beyond the A
calibration, the gate, the smoke, and one unplanned look at B that is disclosed below.

Jira: epic FLYW-1. Part 1 is FLYW-4, Part 2 is FLYW-5, and FLYW-6 (whether fault-graph needs a remediation
label) is decided from the result. Design source: sre-flywheel README, "First experiments", branch
`claude/experiment-designs`.

## The question in plain words

Today's 20q loop (P0a) keeps asking until it can name the secret. In SRE, the job ends when the right
*action* is known. Two causes with the same fix do not need to be told apart.

**Does aiming at the action instead of the name reach the right action in fewer probes, without more wrong
actions? And does it matter how the action classes line up with the questions that are naturally
informative?**

**Why it comes first.** It decides what the system is trying to learn. How finely to cut "species" for
stopping, what the likelihood should predict, and what the fault-graph → diagnosis contract calls a
hypothesis all depend on it.

## Terms

| term | meaning |
|---|---|
| **class** | A group of secrets that share a remediation. Taken from a class map (`results/flyw1_class_maps_20261118/maps/`, generator version 3) |
| **question** | A category or attribute question. Attribute answers can be noisy; category answers are always exact |
| **action** | Applying the remediation for a class. Its outcome is exact: it works if and only if the secret is in that class. A by-name guess is the action for the guessed object's class |
| **right action / wrong action** | An action on the secret's class / on any other class |
| **probe** | A question or an action. **Probes to decide** = probes used until the first right action, or until the game ends without one |
| **expert call** | The loop gives up and asks a human (P0a's escalation) |
| **alignment** | The class-map index from `FLYW1_CLASS_MAPS.md`: the share of *P0a's own* questions that stopping at the class saves, on the question tree P0a follows at flip 0. Constructions: `align` (highest), `mix05`, `mix15`, `rand` (near 0), and `cat17` |
| **construction × flip** | The scoring unit: all maps of one construction at one flip level, pooled |

## Arms

Every arm uses the **P0a belief**: `modeswitch_guard`, with reinstate budget K = 3, `no_repeat` on, the
graded matrix likelihood (ε = 0.05), confidence 0.9 and 20 probes. Only the question score and the stopping
rule change.

| arm | question score | stops and acts when |
|---|---|---|
| **A (today)** | P0a's EIG about *which object* | P0a's rule: guess by name when one object is left, or when a guess wins the EIG ranking |
| **B** | Same as A | Every live object is in one class: act on that class |
| **C** | EIG about *which class*: the same graded likelihood, with entropy taken over the class of the live objects | Same as B |
| **D** | EC² (Golovin, Krause & Ray 2010) on P0a's live set: the expected weight of cross-class edges cut, uniform weights, and an object kept by an answer exactly when P0a's kill rule keeps it (likelihood of the answer ≥ 0.5) | Same as B |

**Rules every arm shares**

* **What can be asked.** A keeps P0a's candidates: every category and attribute question, plus by-name
  guesses of the leading live objects. In B–D the candidates are the category and attribute questions only,
  and every action comes from the stopping rule.
* **Scoring A as a decision.** A plays exactly as P0a does today. Its first by-name guess of an object in the
  secret's class is scored as the right action, even when the name is wrong: in SRE the fix would have worked
  and the incident would end there. Every guess before it, of an object in another class, is a wrong action.
  **A-strict** (the game run until the name is confirmed) is reported for reference.
* **A failed action.** In B–D, an action that fails strikes every member of that class. The strike is exact,
  so reinstate never revives them. In A, a guess answered "no" strikes the named object, as today.
* **Reinstate.** As in P0a: when the live set would empty, up to K = 3 times per game, P0a revives the object
  with the fewest contradicting answers. B–D then act on that object's class, which tests it, just as A's
  by-name guess does.
* **When no question separates the live objects.** If the live objects span several classes but no question
  scores above zero (the unresolvable pairs of `FLYW1_CLASS_MAPS.md`), B–D act on the class with the most live
  members (ties: lowest class id).
* **Running out of probes.** At 20 probes with no right action, the game ends with an expert call. This is the
  `always` end rule, the one under which norepeat P0a was both autonomous and safe (§3.14 end-game).
* **Paired noise.** Attribute flips come from a *keyed* oracle: whether the answer to question q about secret s
  is flipped under noise seed k is a fixed function of (k, s, q), a hash, not a draw from a running random
  stream. With `no_repeat` on, every arm that asks q about s under seed k gets the same answer, so arms are
  paired game for game. (`noisy_matrix_oracle` draws flips in sequence, so arms asking in a different order
  would get different flips.) DE0(a)'s reference run uses the same keyed oracle.

## Basis: where the thresholds come from

* **B: P0a's own question tree.** B asks P0a's questions, so its noise-free cost is read off the tree those
  questions form in the control arm's flip-0 run: questions until the object's branch holds one class, plus
  the actions (`p0a_pure_cost`). Objects that A told apart only by guessing are taken in the plurality order.
* **C and D: the class maps' noise-free trees** on the binarized answer table (`class_cost`, `ec2_cost`),
  questions plus actions, including failed actions on unresolvable objects. These model C's and D's own
  question choice. They do not model the graded likelihood's preference for exact category questions, which
  C shares with P0a; C-bin is reported to separate the two.
* **A: a disclosed calibration run.** The identity tree assumes A narrows to one object before it guesses, but
  P0a does not do that. So A's cost was measured: the 158 deterministic games of arm A at flip 0
  (`flyw1_a_calibration.py`, `results/flyw1_a_calibration_20261118/`). Only the control arm was played, and
  nothing was learned about B, C or D. All 158 games were won. At flip 0, A:
  * asks about 6.1 questions and then makes about 4.1 by-name guesses (up to 11), the first at probe 7.1 on
    average;
  * takes 10.16 probes to confirm the name (A-strict) and 7.76 probes to decide, averaged over the v3 maps (8.15 on the first draft's maps);
  * makes 0 to 2.6 wrong actions per game, depending on the map. On `cat17` it makes none, because its wrong
    guesses always fall in the right category.

The table gives the bases on the v3 maps, for the full run and for the trimmed run (seed-1 maps plus `align`
and `cat17`, 33 maps), both registered now. A's numbers are the same 158 games scored on the v3 maps.

| basis | full (57 maps) | trimmed (33 maps) | bar |
|---|---:|---:|---:|
| DE1(a): B's saving over A, smallest single map | 0.27 | 0.28 | ≥ −0.1 pooled |
| DE1(b): B's saving over A, `align` (lowest map) | 1.09 (0.38) | 1.09 (0.38) | ≥ 0.5 |
| DE1(b): B's saving over A, `mix05` (lowest map) | 0.68 (0.30) | 0.66 (0.39) | ≥ 0.3 |
| DE2: Spearman, alignment vs B's saving | 0.55 | 0.58 | ≥ 0.5 |
| DE3(a): C − B on `rand` (lowest map) | 2.87 (0.33) | 3.26 (0.33) | ≥ 1.0 |
| DE4(a), reported: B − D on `mix15` + `rand`, against `align` | 0.20 against 0.17 | 0.17 against 0.17 | none |
| DE4(b): `rand` maps with D ≤ C | 16 of 16 | 8 of 8 | all |
| DE5: D's saving over B pooled (C's) | 0.19 (−1.42) | 0.18 (−1.32) | < 0.5 |
| (reported) DE5 ratio, (A − B) ÷ (A − best of B, C, D per map) | 0.72 | 0.73 | none |

The first draft's bases, on the binarized identity tree, were 2–3 times larger (B saved 3.3 probes on aligned
maps). P0a does not follow that tree: it asks the exact category questions one at a time (602 category and
363 attribute questions over A's 158 games), so class-pure branches come late.

## Part 1: probes, accuracy and expert calls (FLYW-4)

**Design**

* **Closed world.** Every secret is one of the 158 known objects; the catch-all prior is 0.
* **Maps.** All 57. A's play does not depend on the map, so A is played once per secret and noise seed and then
  scored on every map.
* **Answers.** The keyed matrix oracle with attribute flips at 0, 0.05 and 0.10. By-name guesses, category
  questions and actions are exact.
* **Games per cell.** Each of the 158 objects is the secret once at flip 0 (the game is deterministic), and
  once per noise seed 1–5 at each noisy level, so 790 games.
* **Size.** B, C and D: 57 maps × 3 arms × (158 + 2 × 790) = **297,198 games**, plus 1,738 for A.
* **Trimming rule, fixed now.** If the smoke run projects more than 8 hours on 6 workers, drop the seed-2
  maps (24 maps) and keep seed 1, `align` and `cat17`. **A trim redefines every prediction over the maps
  actually run**: "all maps", "the synthetic maps" and "every `rand` map" then mean the trimmed set, and the
  trimmed bases in the table apply.
* **Checkpointing.** The runner writes each (map, arm, flip) cell to disk as it finishes and resumes from
  those files after a stop, as the ST and Split Deal runners do.
* **Seeds (rule set 2026-11-18, user).** The quantity each cell measures is the mean arm difference over this
  fixed pool of 158 secrets, so the only randomness is the answer noise. Its standard error is estimated
  within secrets, which is a one-way ANOVA within-secret mean square:
  * for each secret, take the variance of the paired difference across its seeds, and pool over the secrets
    (158 × (S − 1) degrees of freedom);
  * SE of a pooled mean over N (map, secret) pairs = √(Σ σᵢ²) ÷ (N √S);
  * flip-0 cells are deterministic, with SE 0.

  The per-game SE of the first draft counted the spread *between* secrets as sampling error, and it asked for
  40–512 seeds that could not reduce that spread.

  **The rule.**
  * For each pooled Part 1 test at a noisy flip (DE1(a), (b), (c), DE3(b), DE5),
    S = max(5, ⌈5 × (SE₅ ÷ (margin ÷ 3))²⌉). SE₅ comes from a 5-seed smoke: one map per construction, whose
    within-secret variance stands in for its construction's maps.
  * The run uses the largest S over the tests, **capped at 8 hours on 6 workers**.
  * If the cap binds, the analyzer reports SE for every scored test and calls a result within 2 SE of its bar
    **inconclusive** rather than pass or fail.
  * Part 2 uses the same seed count, unless that projects past its 4-hour budget; in that case it uses
    5 seeds, and SE is reported.

  **Blind.** SE₅ is computed by `flyw1_seed_rule.py`, which forms and prints only variances, standard errors and
  seed counts, never the arms' mean differences. This step used the smoke's spread and nothing else. The
  first smoke's summary file did contain the paired means, written by the code; they were never read, and the
  code no longer writes them.

**Per-game record:** map, arm, flip, noise seed, secret, secret's class, probes to decide, questions,
actions, wrong actions, outcome (`right` / `expert`), reinstates used, and the live-set size and class count
at each turn.

**Per cell:** mean probes to decide, decision accuracy (right actions ÷ all actions), wrong actions per game,
share of games ending with an expert call, and the share decided (ending in a right action).

### Predictions

* **DE0: mechanical gate.** Every part must hold.
  * (a) A matches a direct `run_openworld_episode` with backend `modeswitch_guard`, `no_repeat` and the keyed
    oracle, game for game, on the same seeds. At flip 0 it reproduces the calibration run exactly.
  * (b) On a one-class map (every object in the same class), B, C and D act at the first probe and are right
    in every game.
  * (c) At flip 0, B, C and D never take a wrong action, except on unresolvable objects.
  * (d) Every map loaded matches its registered `assignment_sha256_12`, and the map files are generator
    version 3.
* **DE1: B beats A cheaply and almost everywhere.** Scored per construction × flip.
  * (a) In every construction × flip, B's mean probes to decide ≤ A's + 0.1. Basis: B saves at least 0.27 on
    every single map.
  * (b) At flip 0 and at 0.05, B saves **≥ 0.5 probe** on `align` maps and **≥ 0.3** on `mix05` maps, on
    average. Basis: 1.09 and 0.68. The bars keep the drafted share of their basis.
  * (c) In every noisy construction × flip, B's wrong actions per game ≤ A's + 0.02. Basis: at flip 0, A makes
    0–2.6 wrong actions per game and B none, so this is expected to pass with room to spare. It is kept because
    noise is where B, acting while several objects are still live, can follow a flip-struck secret to a wrong
    class.
* **DE2: the gain follows alignment.** Over the synthetic maps at flip 0, the Spearman correlation between
  alignment and B's saving (A − B probes) is **≥ 0.5**. Basis: 0.55. A's own savings vary by map, because its
  first right-class guess comes earlier on some maps than others, which weakens the correlation.
* **DE3: C can do badly.**
  * (a) On `rand` maps at flip 0, C needs **≥ 1.0 more probe** than B. Basis: 2.87, smallest single map 0.33.
    This is the myopia that motivated EC²: a question can say nothing about the class on its own and still be
    needed to get there.
  * (b) On `rand` maps at flip 0.10, C's wrong actions per game ≥ B's + 0.02. Weak basis: the design text expects
    C to suffer under noise, but no calibration predicts the size.
* **DE4: EC² beats greedy class information where classes cut across the natural splits.**
  * (a) Reported, not scored: B − D (the probes D saves over B), averaged over `mix15` and `rand` maps against
    `align` maps, at flip 0. Its basis on P0a's tree is 0.20 against 0.17, a difference inside the noise.
  * (b) At flip 0, D ≤ C on every `rand` map. Basis: 16 of 16. DE4 is scored on (b).
* **DE5: changing the question score is not worth it.** At flip 0.05, pooled over all maps, neither C nor D
  saves **0.5 probe or more** over B. Basis: D saves 0.19 and C costs 1.42 more. The bar is set by what a new
  question score is worth, not by a ratio; the ratio is reported only.

### Scoring and decision rule

* The analyzer to be built (`analyze_flyw1.py`) implements these rules; where this wording is ambiguous, its
  code is the reference. DE0 passes only if every part holds. DE1–DE5 each pass or fail on their threshold,
  over the maps actually run.
* **If DE1 and DE5 pass:** only the stopping rule needs to change. Recommend B for the production loop, and
  FLYW-6 specifies a remediation label on fault-graph failure modes.
* **If DE1 passes and DE5 fails:** the question score matters too. Recommend whichever of C and D saves 0.5 or
  more over B at flip 0.05, provided its wrong actions stay within DE1(c)'s bound against A.
* **If DE1 fails:** class targeting does not pay on P0a. FLYW-6 records why no label is needed.

**Reported, not scored:**

* every cell's measures by K, profile and alignment, including the per-map DE1(a) and DE1(c) differences;
* `cat17` on its own: category questions test the class exactly and P0a asks them first. At flip 0 the basis
  has A at 7.06 probes and B at 6.55;
* A-strict;
* a total cost per game that charges wrong actions (1 per probe plus c_wrong per wrong action, at Part 2's
  ratios). A's early guessing has a cost that probes alone do not show;
* **C-bin**, a variant of C on binarized answers, which separates "class objective" from "graded likelihood"
  in the C–D comparison;
* B on the *identity map* (every object its own class) against A-strict;
* probes against alignment for every arm, and how often the unresolvable-pair rule fires;
* the ratio (A − B) ÷ (A − best of B, C, D). Read it as inconclusive within 0.8 ± 0.05.

## Part 2: when to call the expert (FLYW-5)

Today the expert is called when the catch-all wins: P0a's live set empties after its reinstates, or the probes
run out. Under a decision goal an unknown secret matters only if it would change the action.

**The two triggers.** Both act at the same moments, where today's rule would call the expert.

| trigger | at those moments |
|---|---|
| **T-mass (today)** | Call the expert |
| **T-loss** | Work out the class belief P(X) (below), take the best class X\*, and call the expert only if **1 − P(X\*) > c_expert ÷ c_wrong**. Otherwise act on X\*. If that fails, strike X\* and decide again, at most twice per game; after that, call the expert |

**Class belief at a trigger point.** P(X) ∝ π(X) · L(X), over the known classes plus one *new-class* option.

* π(X) = (1 − p_U) × X's share of the known objects; π(new) = p_U, the catch-all prior P0a already uses.
* L(X) = the **mean**, over X's known members, of the product of their ε-floored likelihoods of every answer so
  far (ε = 0.05). L(new) uses the catch-all's background likelihood (`background_marginals`).
* A class already struck by a failed action gets P(X) = 0.
* **Why the mean.** With π(X) proportional to class size and members equally likely, π(X) × the mean
  likelihood is exactly the class's share of the object-level posterior. L(new) is a typical draw's
  likelihood, the same kind of quantity. The best of n members grows with n, and one lookalike would beat the
  background and absorb new-class novels, which is ET2's risk.
* **Shadow.** At every trigger point the runner also logs the decision a *max* version (L(X) = the best member)
  would have taken. It is reported, not scored, and never acts.

**Design**

* **Maps:** `cat17`, plus `align` and `rand` (seed 1) at K = 4 and 8, both profiles. That is 9 maps.
* **Novel secrets, two kinds, rotated.** Each map is run once per held-out class:
  * **K = 4 maps: every class in turn** (4 rotations each).
  * **K = 8 maps and `cat17`: 3 classes.** Sort the classes by size, ties by id, and take those at positions
    0.25 (n − 1), 0.5 (n − 1) and 0.75 (n − 1), rounded half up, where n is the number of classes
    (counting from 0).
  * In each rotation the held-out class's members are *new-class* novels, which acting cannot fix.
  * **Same-class novels:** from every other class, hold out 10% of its size, rounded half up, by seed, keeping at
    least one known member in every class. A class of 2 gives none. They are drawn once per (map, rotation) with
    a fixed seed, not per noise seed, so the known pools are fixed.
  * That is 31 rotations.
* **Alignment of each reduced known pool** is recomputed and reported with the hold-out files before the run.
  It is not used for any threshold.
* **Games.** Snapshot mode against the rotation's known pool: 4 in-pool games per known object plus one game
  per novel. The catch-all prior is the novelty share (as in `run_snapshot`). If the Part 2 smoke run projects
  more than 4 hours on 6 workers, cut in-pool games to 2 per known object.
* **Flips** 0, 0.05 and 0.10 with the keyed oracle; noise seeds as in Part 1.
* **Arms:** B and D, each under both triggers. A under T-mass is the reference.
* **Cost ratio** c_expert ÷ c_wrong ∈ {0.5, 0.2, 0.05}. **0.2 is primary.**

**Measures**

* **Spurious call:** an expert call on an in-pool game or a same-class novel.
* **Missed call:** T-loss *acting* at the trigger point on a new-class novel. Such an action is always wrong,
  because no known class contains the secret. T-mass never acts there, so its rate is 0 by construction.
* **Wrong actions before the trigger point** on new-class novels are the stopping rule's, not the trigger's,
  and are the same under both triggers. They are reported per arm.
* **The original missed-call measure,** reported and not scored: a new-class novel game with any wrong action
  before the expert, or no expert call. It is kept as a finding for SRE. A novel incident costs wrong actions
  before the escalation logic is even consulted, because both triggers fire only when the live set empties
  or the probes run out. That points to a gap in the design: escalation should probably be considered at every
  action, not only at those points.
* **Total cost per game:** 1 per probe, c_expert = 10 per expert call, and c_wrong = 10 ÷ ratio per wrong
  action.

### Predictions

All three are pooled over each map's rotations.

* **ET1:** at ratio 0.2, pooled over flips 0.05 and 0.10, on `align` maps and `cat17`, T-loss cuts spurious calls
  by **≥ 30% relative to T-mass**, for both B and D. Basis: under noise, P0a's in-pool calls come from
  flip-struck secrets. When classes are aligned, the struck secret's classmates still fit the answers, so its
  class keeps most of the belief.
* **ET2:** at ratio 0.2, pooled over flips 0.05 and 0.10, on every map, T-loss's missed-call rate is
  **≤ 0.10** for both B and D. Basis: a new-class novel should fit the background better than any class
  average. This is the prediction most at risk
  on `rand` maps, where classes carry no answer signal, so every class average looks like the background.
* **ET3:** at ratio 0.2, pooled over flips 0.05 and 0.10, on `rand` maps, T-loss's spurious-call rate is within
  **0.02** of T-mass's. When classes
  carry no answer signal, the class belief should stay spread, so T-loss calls the expert just as often.

**Decision rule.** Adopt T-loss as the escalation rule for the chosen arm if ET1 and ET2 both pass and T-loss's
total cost is lower than T-mass's at ratio 0.2 pooled over the noisy flips. If ET1 passes and ET2 fails, T-loss
is adopted only together with an alignment check on the class map; the note says which maps failed.

**Reported, not scored:**

* all three ratios, and flip 0;
* spurious and missed calls broken down by kind of novel and by rotation;
* the turn at which T-loss acts or calls;
* P(X\*) at the trigger point, for right and wrong actions;
* the max-likelihood shadow's decisions;
* a sensitivity run with p_U halved and doubled. Setting p_U to the true novelty share flatters T-loss; this
  shows how much ET1 and ET2 depend on it.

## What stays fixed, and what this is not

* **The belief is P0a in every arm.** The soft posterior is not an arm; the shadow studies (§3.14) found its
  top pick unreliable under noise.
* **Answers come from the self-consistent matrix oracle** with keyed attribute flips. Category questions,
  by-name guesses and actions are exact. This is not Jev answering live; there are no API calls.
* **Classes are synthetic stand-ins for remediations.** Except for `cat17`, they are not meaningful groups of
  objects. What carries over to SRE is the alignment measure, not a particular map.
* **One cause per game.** Two-cause games are Experiment 2 (FLYW-2).
* **Scope: this pool.** The secrets are treated as fixed: they are the whole pool, not a sample, and the only
  randomness modelled is the answer noise. So the conclusions hold for this 158-object pool. Generalisation
  comes from the variation across maps (alignment, K, size profile), not from sampling secrets.
* **Compute** goes through `jobq` with `OMP/OPENBLAS/MKL/VECLIB_MAXIMUM_THREADS=1` per worker. BLAS pinning was
  the whole speed fix in ST-1.

## Decisions taken 2026-11-18 (user, Jira FLYW-1 comment 10584)

1. **Noise-free basis: accepted, with two fixes.** A's cost is measured by the disclosed flip-0 run, not taken
   from the identity tree. B, C and D's costs count their actions, including failed ones on unresolvable
   objects. DE5 is now an absolute bar, a 0.5-probe saving at flip 0.05, and the ratio is reported only.
2. **Trimming and seeds: kept, with fixes.**
   * A trim redefines the predictions over the maps run, and the trimmed bases are registered.
   * Noise is keyed on (seed, secret, question).
   * The DE1 margins are scored per construction × flip and reported per cell.
   * The smoke run sets the seed count from the paired SE.
   * The runner checkpoints per cell.
3. **T-loss class belief: mean likelihood**, with the max version logged as a shadow.
4. **Part 2 hold-outs: rotated.**
   * Every class on K = 4 maps, and 3 classes by size position on K = 8 maps and `cat17`.
   * Same-class novels are stratified by class, never take a class's last known member, and are fixed per
     rotation.
   * Each reduced pool's alignment is recomputed.
   * A p_U sensitivity run is reported.

## Build amendments (2026-11-18, user)

Two design problems surfaced while the analyzer was being tested on a two-map trial run, before the smoke.

**1. The maps did not follow P0a's questions.** The aligned maps had been cut from a balanced tree on the
binarized answers. P0a's graded EIG prefers the exact category questions and asks them one at a time, so it
never follows that tree.

* **The unplanned look.** The trial run played B, C and D at flip 0 and 0.05 on the old `k4_bal_align` and
  `k4_bal_rand_s1` maps; it was meant only to exercise the analyzer. On old `k4_bal_align` at flip 0, B took
  9.0 probes against a basis of 4.5 and A's 8.0. That is what exposed the problem. Those maps no longer exist
  in this form, but the look is disclosed because it informed the change below.
* **The change (user: "Rebuild on P0a's tree").** The `align` and `mix` maps are cut from P0a's own question
  tree, read off the control arm's flip-0 run. `rand` and `cat17` keep their assignments. Alignment and B's
  basis are measured on that tree (generator version 3).
* **Bars (user: "Rescale by the drafted rule").** Each bar keeps the share of its basis it had in the draft:
  DE1(b) 1.0 → 0.5 (`align`) and 0.3 (`mix05`); DE2 0.8 → 0.5. DE4(a)'s new basis difference, 0.04, is inside
  the noise, so it is reported only. DE1(a), DE1(c), DE3, DE4(b) and DE5 are unchanged.
* **What this does to the test.** B's flip-0 basis is now read off the same questions B asks, so the flip-0
  predictions for B are close to mechanical. The noisy predictions (DE1(b) at 0.05, DE1(c), DE5) carry the
  test.

**2. ET2 counted wrong actions the trigger cannot affect.** In a trial rotation, every new-class novel took a
wrong action before any trigger point, under every arm (A 7.1 per game, B 2.7, D 3.0). So no trigger could
meet a 0.10 bar.

* **The look, disclosed.** The change followed a look at Part 2 output from the same trial run: one rotation of
  `k4_bal_align` (the old, version-2 map) at flip 0.10, seed 1, for arms A, B and D. The analyzer's printout
  showed, for that one rotation:
  * spurious-call rates (B 0.000 under both triggers, D 0.009 under both);
  * per-game costs, also equal under both triggers;
  * the old missed-call rate of 1.0;
  * a follow-up count showing that every new-class game had a wrong action before any trigger point, under
    every arm.

  No other Part 2 output was looked at.
* **Why the change is not outcome-driven.** The reason is structural. Wrong actions before the trigger point
  happen before either trigger is consulted, so they are identical under T-mass and T-loss, and the original
  ET2 could not tell the triggers apart, whatever the results. That holds by construction of the arms, not
  because of the trial's numbers.
* **The change (user: "Score T-loss's own acts").** A missed call is T-loss acting at the trigger point on a
  new-class novel. Wrong actions before the trigger point are reported per arm. ET2 and ET3 pool flips 0.05
  and 0.10, as ET1 does. The original measure is kept as reported (Part 2, Measures).

**3. The seed rule** was set by the user after the first smoke (Part 1, "Seeds"). It replaced a per-game SE
rule that treated the spread between secrets as noise.

**The seed count.** The 5-seed smoke (jobq 58, `results/flyw1_smoke_20261118/seed_rule.json`) gives:

* DE1(c) on `cat17` needs 787 seeds, and DE1(c) on the synthetic constructions needs 20–119;
* every other test needs 5–28.

The 8-hour cap allows 66, so **the cap binds**. Part 1 runs **66 noise seeds** at each noisy flip (7.98 h on 6
workers), and the 2-SE inconclusive band applies. Part 2 at 66 seeds would pass its 4-hour budget, so **Part 2
runs 5 seeds** with SE reported.

**Before registering:** this draft merges and the prereg is tagged, as the ST studies were.

## Addendum 1: seed allocation (2026-11-19, user, Jira FLYW-1 comment 10593)

**When.** This addendum was written and tagged (`prereg-flyw1-addendum1-20261119`) on 2026-11-19 at about 05:45
PST. Part 1 (jobq 65) was still running, at 21,050 of 30,590 jobs. It was paused while the Mac slept overnight,
which is why it ran past the 01:00 estimate. Part 2 (jobq 64) had finished on 2026-11-18. **No Part 1 or Part 2
outcome had been read.** The addendum rests only on the smoke's variances
(`results/flyw1_smoke_20261118/seed_rule.json`), never on an arm-against-arm mean. jobq 65 finishes as
registered.

**Why it is needed.** At 66 seeds, DE1(c) cannot be decided on `cat17` (it needs 465–787 seeds), nor on the
synthetic cells that need more than 66 (up to 119). The registered decision rule says each of DE1–DE5 passes or
fails, but never says what an inconclusive DE1 means. The 0.02 margin cannot be loosened instead. At Part 2's
primary ratio a wrong action costs 50 probes, and B saves 0.5–1.1 probes, so break-even is 0.01–0.022 wrong
actions per game. The margin sits at break-even, so the precision is needed.

**1. An inconclusive DE1 counts as not passed.** If DE1 is inconclusive (within the registered 2-SE band), B is
not recommended. The report gives the SEs and says that more seeds are needed. This only connects the
registered 2-SE band to the decision rule (§Scoring and decision rule).

**2. A DE1 extension, for arms A and B only.** Only DE1 needs high precision, and DE1 compares A with B. C needs
16 seeds (DE3(b) on `rand`) and D needs 5 (DE5), so 66 already covers them.

* **Which cells.** Every construction × flip whose blind-tool S is above 66 runs A and B on seeds 67…S. The
  cells are chosen by S alone, whatever Part 1 shows. From `seed_rule.json`, the cells are:

  | construction | flip | S (DE1(c)) | B runs | A runs |
  |---|---|---:|---|---|
  | `align` (8 maps) | 0.05 | 71 | seeds 67–71 on 8 maps | seeds 67–71 |
  | `align` (8 maps) | 0.10 | 119 | seeds 67–119 on 8 maps | seeds 67–119 |
  | `category` (`cat17`) | 0.05 | 465 | seeds 67–465 | seeds 67–465 |
  | `category` (`cat17`) | 0.10 | 787 | seeds 67–787 | seeds 67–787 |

  DE1(a) needs at most 28 seeds anywhere, so it adds no cells.
* **The streams are paired.** The noise is keyed on (seed, secret, question), so seeds 67…S extend the same
  paired streams, and nothing already run changes. A's play does not depend on the map, so one A game per
  (flip, seed) serves every map.
* **Scoring.** In those cells, DE1(a) and DE1(c) are scored on seeds 1…S. The registered 66-seed result is
  reported beside it as the primary-protocol result. DE1's verdict uses the extended cells, and the
  inconclusive band applies to their SE at S seeds.
* **Cost**, in arm × map × seed units:
  * B: 8 × (5 + 53) + (399 + 721) = 1,584;
  * A: 1,120;
  * total: 2,704 against Part 1's ≈ 11,286, about 24%, or roughly 2 h on 6 workers if cost per game is equal.
  The comment's upper bound was ≤ 39% (≤ 3 h). The blind tool projects hours before anything is queued, and
  the runs go through jobq.

**3. Part 2's own seed count.** Part 2 inherited Part 1's maximum and fell back to 5 seeds.
* **Run the blind tool on ET1–ET3** directly, computing only variances and SEs, to get Part 2's own S.
* **If S is above 5** and the extension fits the 4-hour budget, Part 2 extends on seeds 6…S, the same way.
* **Otherwise** the 2-SE inconclusive band applies to ET1–ET3 as well.
* **This holds only while Part 2's ET results are unread.** At this tag they are unread. If they are read before
  the extension, any extension is reported as exploratory.

**Disclosure.** The addendum was made after registration and after the smoke, from variances only, before any
Part 1 or Part 2 outcome was read.
