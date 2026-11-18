# Aim at a decision, not an identity — pre-registration (Flywheel experiment 1)

**Status (2026-11-18): DRAFT, not registered.** Revised the same day after the user's calls on the four
open choices (Jira FLYW-1, comment 10584) and a disclosed calibration run of the control arm (§Basis). No
arm except A at flip 0 has been run, and B, C and D are not built. The class maps are built
(`FLYW1_CLASS_MAPS.md`).

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
| **class** | A group of secrets that share a remediation. Taken from a class map (`results/flyw1_class_maps_20261118/maps/`, generator version 2) |
| **question** | A category or attribute question. Attribute answers can be noisy; category answers are always exact |
| **action** | Applying the remediation for a class. Its outcome is exact: it works if and only if the secret is in that class. A by-name guess is the action for the guessed object's class |
| **right action / wrong action** | An action on the secret's class / on any other class |
| **probe** | A question or an action. **Probes to decide** = probes used until the first right action, or until the game ends without one |
| **expert call** | The loop gives up and asks a human (P0a's escalation) |
| **alignment** | The class-map index from `FLYW1_CLASS_MAPS.md`: the share of the identity tree's questions that stopping at the class saves. Constructions: `align` (high), `mix05`, `mix15`, `rand` (near 0), and `cat17` |
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

* **B, C and D: the class maps' noise-free trees.** These are built from the answer table alone and never from
  an arm's output. Each arm's cost is `pure_cost`, `class_cost` or `ec2_cost`: questions plus actions,
  including the failed actions an arm takes on unresolvable objects.
* **A: a disclosed calibration run.** The identity tree assumes A narrows to one object before it guesses, but
  P0a does not do that. So A's cost was measured: the 158 deterministic games of arm A at flip 0
  (`flyw1_a_calibration.py`, `results/flyw1_a_calibration_20261118/`). Only the control arm was played, and
  nothing was learned about B, C or D. All 158 games were won. At flip 0, A:
  * asks about 6.1 questions and then makes about 4.1 by-name guesses (up to 11), the first at probe 7.1 on
    average;
  * takes 10.16 probes to confirm the name (A-strict) and 8.15 probes to decide, averaged over the maps;
  * makes 0 to 2.6 wrong actions per game, depending on the map. On `cat17` it makes none, because its wrong
    guesses always fall in the right category.

The table gives the bases, for the full run and for the trimmed run (seed-1 maps plus `align` and `cat17`,
33 maps), both registered now.

| basis | full (57 maps) | trimmed (33 maps) | bar |
|---|---:|---:|---:|
| DE1(a): B's saving over A, `rand` maps pooled (lowest single map) | 0.48 (−0.11) | 0.51 (−0.01) | ≥ −0.1 pooled |
| DE1(b): B's saving over A, `align` (lowest map) | 3.32 (1.94) | 3.32 (1.94) | ≥ 1.0 |
| DE1(b): B's saving over A, `mix05` (lowest map) | 1.98 (1.35) | 1.88 (1.35) | ≥ 1.0 |
| DE2: Spearman, alignment vs B's saving | 0.86 | 0.88 | ≥ 0.8 |
| DE3(a): C − B on `rand` (lowest map) | 2.82 (1.18) | 3.21 (1.18) | ≥ 1.0 |
| DE4(a): B − D, `mix15` + `rand` | 0.30 | 0.28 | ≥ `align` + 0.2 |
| DE4(a): B − D, `align` | −0.10 | −0.10 | |
| DE4(b): `rand` maps with D ≤ C | 16 of 16 | 8 of 8 | all |
| DE5: B − D pooled (B − C) | 0.30 (−0.82) | 0.25 (−0.73) | < 0.5 |
| (reported) DE5 ratio, (A − B) ÷ (A − best of B, C, D per map) | 0.81 | 0.83 | none |

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
* **Seeds.** The smoke run also measures the standard error of the paired A − B difference per construction ×
  flip. If any SE is above a third of that unit's margin (0.1 for DE1(a), 0.02 for DE1(c)), raise the noise
  seeds from 5 until it is not, before the main run. The seed count actually used is recorded.

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
    version 2.
* **DE1: B beats A cheaply and almost everywhere.** Scored per construction × flip.
  * (a) In every construction × flip, B's mean probes to decide ≤ A's + 0.1. Basis: the smallest pooled saving
    is `rand` at 0.48. Two single `rand` maps sit just past the margin (−0.11), which is why this is pooled.
  * (b) For `align` and for `mix05`, at flip 0 and at 0.05, B saves **≥ 1.0 probe** on average. Basis: 3.32 and
    1.98, smallest single map 1.35.
  * (c) In every noisy construction × flip, B's wrong actions per game ≤ A's + 0.02. Basis: at flip 0, A makes
    0–2.6 wrong actions per game and B none, so this is expected to pass with room to spare. It is kept because
    noise is where B, acting while several objects are still live, can follow a flip-struck secret to a wrong
    class.
* **DE2: the gain follows alignment.** Over the synthetic maps at flip 0, the Spearman correlation between
  alignment and B's saving (A − B probes) is **≥ 0.8**. Basis: 0.86.
* **DE3: C can do badly.**
  * (a) On `rand` maps at flip 0, C needs **≥ 1.0 more probe** than B. Basis: 2.82, smallest single map 1.18.
    This is the myopia that motivated EC²: a question can say nothing about the class on its own and still be
    needed to get there.
  * (b) On `rand` maps at flip 0.10, C's wrong actions per game ≥ B's + 0.02. Weak basis: the design text expects
    C to suffer under noise, but no calibration predicts the size.
* **DE4: EC² matters most when classes cut across the natural splits.**
  * (a) At flip 0, B − D (the probes D saves over B) averaged over `mix15` and `rand` maps is **≥ 0.2 above** its
    average over `align` maps. Basis: +0.30 against −0.10.
  * (b) At flip 0, D ≤ C on every `rand` map. Basis: 16 of 16.
* **DE5: changing the question score is not worth it.** At flip 0.05, pooled over all maps, neither C nor D
  saves **0.5 probe or more** over B. Basis: D saves 0.30 and C costs 0.82 more. The bar is set by what a new
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
* `cat17` on its own: category questions test the class exactly, and at flip 0 A already decides there as
  cheaply as B (7.06 against 7.09 probes);
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
* **Missed call:** a new-class novel game with at least one wrong action before the expert is called, or with
  no expert call at all.
* **Total cost per game:** 1 per probe, c_expert = 10 per expert call, and c_wrong = 10 ÷ ratio per wrong
  action.

### Predictions

All three are pooled over each map's rotations.

* **ET1:** at ratio 0.2, pooled over flips 0.05 and 0.10, on `align` maps and `cat17`, T-loss cuts spurious calls
  by **≥ 30% relative to T-mass**, for both B and D. Basis: under noise, P0a's in-pool calls come from
  flip-struck secrets. When classes are aligned, the struck secret's classmates still fit the answers, so its
  class keeps most of the belief.
* **ET2:** at ratio 0.2, on every map, T-loss's missed-call rate is **≤ 0.10** for both B and D. Basis: a
  new-class novel should fit the background better than any class average. This is the prediction most at risk
  on `rand` maps, where classes carry no answer signal, so every class average looks like the background.
* **ET3:** at ratio 0.2, on `rand` maps, T-loss's spurious-call rate is within **0.02** of T-mass's. When classes
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

**Before registering:** this draft merges, then the arms, the keyed oracle and the analyzer are built and gated
(DE0 and the smoke run). After that the prereg is tagged, as the ST studies were.
