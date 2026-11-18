# Aim at a decision, not an identity — pre-registration (Flywheel experiment 1)

**Status (2026-11-18): DRAFT, not registered.** No arm has been built or run. The class maps are built
(`FLYW1_CLASS_MAPS.md`). Jira: epic FLYW-1. Part 1 is FLYW-4, Part 2 is FLYW-5, and FLYW-6 (whether
fault-graph needs a remediation label) is decided from the result. Design source: sre-flywheel README,
"First experiments", branch `claude/experiment-designs`.

## The question in plain words

Today's 20q loop (P0a) keeps asking until it can name the secret. In SRE, the job ends when the right
*action* is known. Two causes with the same fix do not need to be told apart.

**Does aiming at the action instead of the name reach the right action in fewer questions, without more
wrong actions? And does it matter how the action classes line up with the questions that are naturally
informative?**

**Why it comes first.** It decides what the system is trying to learn. How finely to cut "species" for
stopping, what the likelihood should predict, and what the fault-graph → diagnosis contract calls a
hypothesis all depend on it.

## Terms

| term | meaning |
|---|---|
| **class** | A group of secrets that share a remediation. Taken from a class map (`results/flyw1_class_maps_20261118/maps/`) |
| **question** | A category or attribute question. Attribute answers can be noisy; category answers are always exact |
| **action** | Applying the remediation for a class. Its outcome is exact: it works if and only if the secret is in that class. A by-name guess is the action for the guessed object's class |
| **right action / wrong action** | An action on the secret's class / on any other class |
| **probe** | A question or an action. **Probes to decide** = probes used until the first right action, or until the game ends without one |
| **expert call** | The loop gives up and asks a human (P0a's escalation) |
| **alignment** | The class-map index from `FLYW1_CLASS_MAPS.md`: the share of the identity tree's questions that stopping at the class saves. Constructions: `align` (high), `mix05`, `mix15`, `rand` (near 0), and `cat17` |

## Arms

Every arm uses the **P0a belief**: `modeswitch_guard`, with reinstate budget K = 3, `no_repeat` on, the
graded matrix likelihood (ε = 0.05), confidence 0.9 and 20 probes. Only the question score and the stopping
rule change.

| arm | question score | stops and acts when |
|---|---|---|
| **A (today)** | P0a's EIG about *which object* | One object is left: guess it by name |
| **B** | Same as A | Every live object is in one class: act on that class |
| **C** | EIG about *which class*: the same graded likelihood, with entropy taken over the class of the live objects | Same as B |
| **D** | EC² (Golovin, Krause & Ray 2010) on P0a's live set: the expected weight of cross-class edges cut, uniform weights, and an object kept by an answer exactly when P0a's kill rule keeps it (likelihood of the answer ≥ 0.5) | Same as B |

**Rules every arm shares**

* **What can be asked.** A keeps P0a's candidates: every category and attribute question, plus by-name
  guesses of the leading live objects, so A can guess before one object is left. In B–D the candidates are the
  category and attribute questions only, and every action comes from the stopping rule.
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
  members (ties: lowest class id). A keeps P0a's behaviour and guesses by name.
* **Running out of probes.** At 20 probes with no right action, the game ends with an expert call. This is the
  `always` end rule, the one under which norepeat P0a was both autonomous and safe (§3.14 end-game).

## Part 1: probes, accuracy and expert calls (FLYW-4)

**Design**

* **Closed world.** Every secret is one of the 158 known objects; the catch-all prior is 0.
* **Maps.** All 57. A's play does not depend on the map, so A is played once per secret and noise seed and then
  scored on every map.
* **Answers.** The matrix oracle with attribute flips at 0, 0.05 and 0.10. By-name guesses, category questions
  and actions are exact.
* **Games per cell.** Each of the 158 objects is the secret once at flip 0 (the game is deterministic), and
  once per noise seed 1–5 at each noisy level, so 790 games.
* **Size.** B, C and D: 57 maps × 3 arms × (158 + 2 × 790) = **297,198 games**, plus 1,738 for A. Cost is to be
  pinned by the build's smoke run. **Trimming rule, fixed now:** if the smoke projects more than 8 hours on
  6 workers, drop the seed-2 maps (24 maps) and keep seed 1, `align` and `cat17`.

**Per-game record:** map, arm, flip, noise seed, secret, secret's class, probes to decide, questions,
actions, wrong actions, outcome (`right` / `expert`), reinstates used, and the live-set size and class count
at each turn.

**Per cell:** mean probes to decide, decision accuracy (right actions ÷ all actions), wrong actions per game,
share of games ending with an expert call, and the share decided (ending in a right action).

### Predictions

The noise-free basis is the tree measures in `FLYW1_CLASS_MAPS.md`. They use the binarized table, so the run's
graded planner may differ.

* **DE0: mechanical gate.** Every part must hold.
  * (a) A matches a direct `run_openworld_episode` with backend `modeswitch_guard` and `no_repeat`, game for
    game, on the same seeds.
  * (b) On a one-class map (every object in the same class), B, C and D act at the first probe and are right
    in every game.
  * (c) At flip 0, B, C and D never take a wrong action, except on unresolvable objects.
  * (d) Every map loaded matches its registered `assignment_sha256_12`.
* **DE1: B beats A cheaply and almost everywhere.**
  * (a) In every cell, B's mean probes to decide ≤ A's + 0.1.
  * (b) On `align` and `mix05` maps, at flip 0 and at 0.05, B saves **≥ 1.0 probe** on average over each
    construction. Basis: noise-free savings 3.63 (`align`) and 2.22 (`mix05`), smallest single map 1.13.
  * (c) In every noisy cell, B's wrong actions per game ≤ A's + 0.02. B acts while several objects are still
    live, so a flip-struck secret can lead it to a wrong class sooner. That is the risk this part checks.
* **DE2: the gain follows alignment.** Over the 56 synthetic maps at flip 0, the Spearman correlation between
  alignment and B's saving (A − B probes) is **≥ 0.8**. Noise-free, the two are the same quantity, so this tests
  whether P0a's graded planner behaves like the identity tree.
* **DE3: C can do badly.**
  * (a) On `rand` maps at flip 0, C needs **≥ 1.0 more probe** than B. Basis: 2.82 noise-free, smallest single
    map 1.18. This is the myopia that motivated EC²: a question can say nothing about the class on its own and
    still be needed to get there.
  * (b) On `rand` maps at flip 0.10, C's wrong actions per game ≥ B's + 0.02. Weak basis: the design text expects
    C to suffer under noise, but no calibration predicts the size.
* **DE4: EC² matters most when classes cut across the natural splits.**
  * (a) At flip 0, B − D (the probes D saves over B) averaged over `mix15` and `rand` maps is **≥ 0.2 above** its
    average over `align` maps. Basis: +0.30 against −0.10 noise-free.
  * (b) At flip 0, D ≤ C on every `rand` map (noise-free: all 16).
* **DE5: B captures most of the gain.** At flip 0 and at 0.05, summed over all 57 maps,
  (A − B) ÷ (A − the best of B, C, D) is **≥ 0.8**. Basis: 0.82 noise-free, close to the bar.

### Scoring and decision rule

* The analyzer to be built (`analyze_flyw1.py`) implements these rules; where this wording is ambiguous, its
  code is the reference. DE0 passes only if every part holds. DE1–DE5 each pass or fail on their threshold, and
  a part scored "in every cell" fails on any missing cell.
* **If DE1 and DE5 pass:** only the stopping rule needs to change. Recommend B for the production loop, and
  FLYW-6 specifies a remediation label on fault-graph failure modes.
* **If DE1 passes and DE5 fails:** the question score matters too. Recommend whichever of C and D has fewer
  probes to decide pooled over noisy cells, as long as its wrong actions stay within DE1(c)'s bound against A.
* **If DE1 fails:** class targeting does not pay on P0a. FLYW-6 records why no label is needed.

**Reported, not scored:** every cell's measures by K, profile and alignment; `cat17` on its own (category
questions test the class exactly, so C may win there with low alignment); A-strict; **C-bin**, a variant of C
on binarized answers, which separates "class objective" from "graded likelihood" in the C–D comparison;
B on the *identity map* (every object its own class) against A-strict, where the only difference left is
that B cannot guess early;
probes against alignment for every arm; and how often the unresolvable-pair rule fires.

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
* L(X) = the mean, over X's known members, of the product of their ε-floored likelihoods of every answer so far
  (ε = 0.05). L(new) uses the catch-all's background likelihood (`background_marginals`).
* A class already struck by a failed action gets P(X) = 0.

**Design**

* **Maps:** `cat17`, plus `align` and `rand` (seed 1) at K = 4 and 8, both profiles. That is 9 maps.
* **Novel secrets, two kinds.** For each map, hold out:
  * **one whole class**: its members are *new-class* novels, and acting cannot fix them. Use the class closest to
    15 objects, ties to the lower id. For `cat17` that is a 14-object category;
  * **10% of the remaining objects**, chosen by seed: these are *same-class* novels. Their class has known
    members, so acting on it would fix them.
* **Games.** Snapshot mode against the remaining known pool: 4 in-pool games per known object plus one game per
  novel, with the catch-all prior set to the novelty share (as in `run_snapshot`).
* **Flips** 0, 0.05 and 0.10, noise seeds 1–5.
* **Arms:** B and D, each under both triggers. A under T-mass is the reference.
* **Cost ratio** c_expert ÷ c_wrong ∈ {0.5, 0.2, 0.05}. **0.2 is primary.**

**Measures**

* **Spurious call:** an expert call on an in-pool game or a same-class novel.
* **Missed call:** a new-class novel game with at least one wrong action before the expert is called, or with
  no expert call at all.
* **Total cost per game:** 1 per probe, c_expert = 10 per expert call, and c_wrong = 10 ÷ ratio per wrong
  action.

### Predictions

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

**Reported, not scored:** all three ratios; flip 0; spurious and missed calls broken down by kind of novel; the
turn at which T-loss acts or calls; and P(X\*) at the trigger point for right and wrong actions.

## What stays fixed, and what this is not

* **The belief is P0a in every arm.** The soft posterior is not an arm; the shadow studies (§3.14) found its
  top pick unreliable under noise.
* **Answers come from the self-consistent matrix oracle** with attribute flips. Category questions, by-name
  guesses and actions are exact. This is not Jev answering live; there are no API calls.
* **Classes are synthetic stand-ins for remediations.** Except for `cat17`, they are not meaningful groups of
  objects. What carries over to SRE is the alignment measure, not a particular map.
* **One cause per game.** Two-cause games are Experiment 2 (FLYW-2).
* **Compute** goes through `jobq` with `OMP/OPENBLAS/MKL/VECLIB_MAXIMUM_THREADS=1` per worker. BLAS pinning was
  the whole speed fix in ST-1.

## Open choices to settle at registration

1. **Is the noise-free basis acceptable** as the source of the thresholds? It is calibration on the answer
   table, not a peek at any arm's results.
2. **The trimming rule** (8 h on 6 workers) and the 5 noise seeds per cell.
3. **The T-loss class belief** uses the mean likelihood of each class's members. A "nearest member" (max)
   version is the obvious alternative; pick one before registering.
4. **Part 2's hold-out sizes:** one whole class plus 10% of the rest.
