# ST-7 Open-side EIG fidelity — pre-registration

**Status: DRAFT, not registered.** Frozen when tagged `prereg-stress-st7-<date>`.
Nothing here has been built or run. Shared infrastructure, scorecard and mechanisms:
`STRESS_DESIGN.md`. The choices still open before registration are listed under
*Decisions for review*.

## Question

The open-world loop folds every not-yet-known object into one catch-all hypothesis,
`__unknown__`. Its likelihood is a stand-in: the per-attribute average over the
**known** objects, floored at ε = 0.05 (`openworld.make_openworld_likelihood`). The
EIG planner then scores questions over `H_known ∪ {UNKNOWN}` as if that stand-in were
the truth.

When the open side carries little mass, the stand-in barely matters and the loop
behaves like the closed-world planner of §3.5. That is the reason a missing-mass
estimate is worth having: it says when a mixed open-closed set can be planned on as if
it were closed. This study asks **how far that holds as the open side grows**. At what
novelty share does the lumped catch-all start choosing different questions from an
exact treatment of the open side, how large is the gap in bits, which way does it
point, and what does it cost in outcomes?

Nothing measured so far answers this. ST-1 fixed the catch-all prior at 0.091, and
ST-0's prior sweep recorded wins only. The only EIG checks to date are numpy-vs-Python
parity checks, which test the implementation, not the approximation.

## What "exact" means here

The target variable is the one the loop decides on: **Z**, which is either one of the
known objects or "unknown". Telling unknown objects apart from each other is worth
nothing to the loop, so the reference does not reward it.

* **Lumped (L), the published loop.** Belief over the K knowns plus one seat `U`. On
  an attribute question, `U` says yes with probability clamp(mean of the knowns' rows,
  ε, 1 − ε). On a category question it says yes with the clamped share of knowns in
  that category, and on a by-name guess with ε. Each answer is scored independently of
  the others.
* **Exact (X).** The unknown class is a uniform mixture over a **reference sample R**
  of whole rows, drawn from the same generator that produces the novel secrets but
  from a disjoint seed. Belief is kept over the K knowns plus the R members. Category
  questions use each member's own category, and a by-name guess (which only ever names
  a known) has likelihood 0 for every member. It is then collapsed to Z:
  P(Z = unknown) is the sum over members. EIG_X(q) = H(Z) − E_a[H(Z | a)], with the
  predictive taken from the expanded belief.

X is Bayes-correct about the population of novel objects without knowing which one is
the secret, so it is not clairvoyant. Everything else is shared with L: the `Noul`
rows, the attribute clamp in the update (unclamped rows in EIG, as `bed.py` does), the
candidate set and the commit rule. So **the L–X gap isolates the lumping
approximation and nothing else.**

What X is *not*: the true data-generating process. Two approximations sit under both
models and are out of scope. The oracle answers by thresholding a row at 0.5, while
both models score answers with the graded row. And both treat a repeated question's
answers as independent draws. *Realized* information gain is therefore not used as the
reference. Its mean equals EIG only when the model generates the data, which neither
model does here.

The gap has two sources, reported separately:

1. **Belief:** after t answers, the lump's P(Z = unknown) differs from X's, because
   a product of marginals is not a mixture of whole rows.
2. **Predictive:** at equal belief, the lump's P(yes | U) is a fixed average. X's
   sharpens toward the members still consistent with the answers so far.

## Setup

Snapshot mode (`stress_harness.run_snapshot`, ST-0d): `H_known` is a fixed pool and
the game mix sets the novelty share. The catch-all prior equals the true novelty share
π, as in ST-1. Misspecified priors are ST-3's question.

| axis | levels |
|---|---|
| known pool K | 158 (the real matrix), 500 (real + 342 generated at ρ_real, `build_pool`) |
| novelty share π (= prior) | 0.02, 0.05, 0.091, 0.15, 0.25, 0.40, 0.60 |
| attribute-flip rate | 0, 0.05 |
| arm | see below |
| replicates | 3 (novel set, reference sample and game order; K = 500 also redraws the pool) |

**Games.** 1000 in-pool games (secrets uniform over the pool) plus
n_novel = round(1000 · π / (1 − π)) novel games: 20, 53, 100, 176, 333, 667 and 1500.
Each novel secret is a distinct object from `generate_objects` at ρ_real = 0.2345, so
it is a child of a real parent, as in ST-1. The prior is the realized share
n_novel / (1000 + n_novel), which is `run_snapshot`'s default. π labels the level, so
"0.091" is 100/1100. Budget 20, confidence 0.90, ε = 0.05.

**Reference sample.** R = 1000 objects from `generate_objects` at ρ_real, on a seed
disjoint from the novel set. Any coincidental identical rows are legitimate: they are
the population. The R-convergence gate below can raise R.

**Arms.**

| arm | belief and commit | planner | `no_repeat` |
|---|---|---|---|
| L-soft | soft (published) | lumped EIG | off |
| L-hard | hard | lumped EIG | off |
| L-hard-nr | hard | lumped EIG | on |
| L-P0a | `modeswitch_guard`, K = 3 | lumped EIG | off |
| L-P0a-nr | `modeswitch_guard`, K = 3 | lumped EIG | on |
| X | exact soft, collapsed to Z; soft commit rule on Z | exact EIG over its own top-5 candidates | off |

**The shadow.** Every L game carries a non-invasive shadow. It keeps the X belief,
updated on the same answers. Before each question it records:

* EIG_L(q*), the planner's value for the question it chose;
* EIG_X(q*);
* EIG_X(q_X*), the best question under X **from the same candidate menu L chose
  from**;
* P(Z = unknown) under both beliefs;
* P(yes | U) for q* under both models.

The shadow never touches the loop. The X arm is the outcome counterpart: the same loop
driven by X belief and exact EIG through the existing `planner` and backend seams,
with no change to `run_openworld_episode`.

Grid: 2 K × 7 π × 6 arms × (3 flip-0 replicates + 3 flip-0.05 replicates with one
noise seed each) = 504 jobs.

**Interpreter, hash seed and BLAS.** Python 3.9 (`LLM_RL`), `PYTHONHASHSEED=0`, one
BLAS thread per worker (`pin_blas`), and the numpy planner path with the `tie_sorted`
rule, as in ST-1. The runner refuses otherwise and records each setting in the
summary. Jobs go through `jobq` with `--workers` equal to `-N`.

### Built and gated before tagging

* **Runner** (`stress_st7.py`): `--plan`, `--self-test`, `--smoke` (synthetic matrix
  only), and `--run` (registration-checked, resumable per job). It stores per-turn
  shadow columns and per-game outcomes.
* **Exact model** (`openside_exact.py`): FastPlanner's attribute and category arrays
  with the single `U` column replaced by R member columns, plus the collapse to Z.
* **Analyzer** (`analyze_st7.py`): implements *Scoring*. Its figure captions name it
  and the result directory.

Gates (each recorded in `STRESS_ST7_BUILD.md`):

* **G1, non-invasive:** with the shadow on, L-arm records match shadow-off records
  bit for bit, on 2 pools × 3 backends × 200 games.
* **G2, degenerate reference:** with R replaced by one member whose likelihoods *are*
  the lump's (clamped attribute averages, clamped category shares, guess = ε),
  EIG_X equals EIG_L
  and the beliefs match to 1e-9 on L-soft trajectories. This shows the reference
  sample is the only difference between the two models.
* **G3, R convergence:** on the K = 158 and K = 500 cells at π = 0.25, flip 0, 100
  games, mean |EIG_X(R = 1000) − EIG_X(R = 4000)| on the chosen questions is ≤ 0.01
  bits. If not, R doubles until it is, and the R used is the one registered.
* **G4, timing:** per-job wall time measured on the largest cell (K = 500, π = 0.60,
  X arm). The registered grid must finish within 6 h on the slots `jobq` grants.

## Metrics

Per turn, from the shadow (L arms):

* **Agreement:** q* is a tie-tolerant argmax under X, meaning EIG_X(q*) ≥
  EIG_X(q_X*) − 1e-6.
* **Regret:** EIG_X(q_X*) − EIG_X(q*), in bits, which is ≥ 0. **Relative regret** of
  a cell is Σ regret / Σ EIG_X(q_X*).
* **Signed EIG error:** EIG_L(q*) − EIG_X(q*). Positive means the lump overstates what
  the question is worth.
* **Belief gap:** P_X(Z = unknown) − P_L(Z = unknown).
* **Predictive gap:** P_X(yes | U) − P_L(yes | U) on q*.

Per game (all arms): outcome (won / expert / wrong / false_confirm), questions,
detector score `p_unknown_max` (under the arm's own belief), and AUC per cell.

## Predictions

The thresholds below are proposals for review (*Decisions for review*, item 1).

* **P1 — closed-world limit.** At π = 0.02, L-soft agrees with X on ≥ 0.90 of turns
  with relative regret ≤ 0.05, at both K and both flip rates.
* **P2 — the gap grows with the open side.** For L-soft, per K × flip: relative
  regret and the mean |signed EIG error| are non-decreasing in π (allowing an adjacent
  drop of up to 0.005), and both are strictly larger at π = 0.60 than at π = 0.02.
* **P3 — the published operating point is closed-like, and the regime ends inside
  the grid.** Define π\* per K × flip as the largest π such that every π ≤ π\* meets
  P1's two conditions. For L-soft at K = 158, flip 0: π\* ≥ 0.091 (ST-1's prior) and
  π\* < 0.60.
* **P4 — crowding, not lumping, blinds the detector at scale.** ST-1 found AUC falling
  from 1.000 at N = 158 to 0.57–0.70 at N = 500. If lumping caused that, the exact
  model should recover it.
  * (a) At K = 158, π ≤ 0.15, flip 0: both X and L-soft have AUC ≥ 0.98.
  * (b) At K = 500, π = 0.091, flip 0: X's AUC − L-soft's AUC ≤ 0.10.
* **P5 — the exact planner is never a worse detector.** In every cell, X's AUC ≥
  L-soft's AUC − 0.02.

**No direction is registered for the signed EIG error or the belief gap.** Both can
change sign within a game. Early on, X's unknown class is the population average and
fits an in-pool secret worse than the lump's ε-floored average does. Later, the few
reference members close to the secret dominate, and X's unknown class fits better than
the lump. Which effect wins on the chosen questions is part of what is measured, not
predicted. Both are reported per cell and by turn.

## Scoring

`analyze_st7.py` implements exactly these rules.

* A **cell** is (K, π, flip, arm). It pools its 3 replicates.
* Turn-level metrics pool every non-guess turn of every game in the cell. Guess turns
  are reported separately, because the lump's ε guess likelihood for `U` is a
  separate stand-in.
* A NaN, such as an empty population, counts as not holding.
* **P1 and P3** are scored on L-soft turn metrics. **P2** is scored per K × flip
  across the 7 π levels in increasing order.
* **P4 and P5** are scored on per-cell AUC of `p_unknown_max`: novel games against
  in-pool games, ties at 0.5 (`analyze_openworld.auc`). P5 covers all 28 K × π × flip
  cells.
* **Discrete arms are not scored.** Their shadow metrics are reported (next list),
  because their belief is not a Bayesian posterior, and an L-vs-X gap there mixes
  lumping with elimination.

**Reported, not scored.**

* Every turn metric by turn index (1–20) and by in-pool vs novel games.
* The belief and predictive gaps.
* Turn metrics for the discrete arms.
* Outcome splits and questions per game for all arms.
* For L-hard and L-P0a with `no_repeat` off: EIG_L and EIG_X on repeated questions,
  split by kind. On a repeated category question EIG_X is 0 by construction (every
  surviving hypothesis already agrees with the answer). The figure of interest is how
  much EIG_L still assigns it, which is the mechanism behind ST-0's repeat loop.

## What would falsify, and what it would mean

* **P1 fails** (the lump disagrees with X even at π = 0.02): the catch-all distorts
  planning even when the open side is negligible. That would most likely come through
  the ε guess likelihood or the category stand-in rather than the attribute average,
  and the per-kind split would show which.
* **P3 fails low** (π\* < 0.091): ST-1's results were obtained outside the closed-like
  regime, and its in-pool numbers carry a lumping cost that ST-1 attributed to size.
* **P4(b) fails** (X recovers much of the K = 500 AUC): lumping, not crowding, is the
  main reason the detector collapses at scale. A better model of the open side, built
  from a generator or from expert-supplied rows, becomes the fix, ahead of anything
  about pool size.
* **P5 fails:** the exact model is a worse detector somewhere. Its commit rule or
  question choice, not its belief, would then need a trajectory read before anything
  else.

## Decisions for review

1. **Thresholds.** The 0.90 agreement, 0.05 relative-regret and 0.10 / 0.02 AUC
   margins are first proposals. They are not derived from data, and nothing has been
   run on the real matrix.
2. **What X knows.** X knows the generator: novel objects are children of real
   parents at ρ_real. That is the right reference for this simulation, but a real
   deployment has no generator. A weaker reference (R drawn at ρ = 1, or from a
   different ρ than the secrets) would measure the cost of a *misspecified* open-side
   model. It is left out to keep the grid at 504 jobs. It could be added as one
   misspecified-R arm at K = 158.
3. **K = 1000.** It is left out because ST-1's AUC there is already near 0.5 and the
   shadow cost scales with K + R. It could be added if P4(b) turns out interesting.
4. **Link to the missing-mass certificate** (`mm_certificate.py`). π\* is the novelty
   share at which planning on the set as if closed costs ≤ 5% of the exact EIG. That
   makes it a candidate for the certificate's ε: once the certificate says the missing
   mass is ≤ π\*, the loop could drop the catch-all from planning (not from the gate).
   That is a design consequence for a follow-up, not something this study tests.

## Notes

* The lumped `U` uses the known-set average, not the novel-set average. In this
  simulation the two are close, because novel objects are children of known parents.
  The predictive gap therefore mostly measures the independence approximation (whole
  rows versus marginals), not a wrong average. A misspecified-R arm (*Decisions for
  review*, item 2) would separate the two.
* At K = 158 with R = 1000, each real parent has on average about 6 reference children.
  `generate_objects` picks a category in proportion to its size and then a parent
  uniformly within it, so every real object is equally likely to be a parent, and
  1000 / 158 ≈ 6.3. X therefore always
  holds a few near-copies of any known object in its unknown class. That is the
  honest version of the look-alike risk ST-2 studied directly, and it is why the
  belief gap can change sign during a game.
