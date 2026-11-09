# Production guardrails for gated identification

A synthesis across the 20Q Bayesian-experimental-design (BED) line — ENV-701 through
ENV-704-B, epic EE-33 — aimed at one question: **when a system identifies a hidden thing
by asking questions and spends scarce, expensive effort (a human, a costly oracle) to do
it, what should you actually build?** The recommendations are ranked by what a production
deployment punishes, not by how interesting each finding was.

**Scope.** A planner runs a BED loop over a pool of hypotheses, answered by a cheap source,
with the option to pay for a truthful one. Every finding cited here regenerates from a
committed run; the source study is named on each rule. Read the preconditions (bottom)
before trusting the cost numbers.

## The one decision

Choose the **discrete, hard-gated backend with an append-only op-log.** It is safe *and*
auditable in a single architectural move — P0a and P0b below are not two features, they are
one choice paying off twice. Everything after it is tuning. Get this wrong and no amount of
cost optimization matters, because the failures it prevents are the silent, expensive ones.

## Ranked recommendations

### P0a — Hard-gate elimination; re-probe behind the gate with a bounded exact test
*(safety-critical)*

Use hard elimination to gate the expensive action. To recover from a wrong elimination,
re-probe the hypothesis **directly** with a bounded exact test behind the gate — never
soften the belief that gates the human. That is the `modeswitch` rule: naive reversibility
(scalar `retract`, CMBS-aligned `tombstone`) recovers in-pool loss but corrupts everything
downstream.

* **Prevents:** confident wrong guesses, and a blinded novelty detector.
* **Evidence:** `modeswitch` recovers in-pool (0.149 < hard 0.175 @10% noise), keeps the
  detector (AUC 0.926 ≥ hard), and manufactures zero wrong-guesses — whereas naive
  reversibility drives AUC to chance and invents wrong-guesses.
* **Source:** ENV-702 (`OPENWORLD_RETRACTION_RESULTS.md`).

### P0b — Keep an append-only, hash-chained op-log
*(free with the gate)*

Record every `ELIMINATE` / `REINSTATE` as an op in a SHA-256 hash-chained log. An external
auditor then reconstructs the run, replays it, and answers "why was this eliminated" — all
at 1.000. This costs almost nothing because the discrete backend chosen for safety is
exactly what makes it auditable. A soft posterior scores zero here: no discrete survivor
set, nothing to log (its ~1104-cell dense trace is not an audit trail).

* **Prevents:** unexplainable decisions; undetectable corruption of the trace.
* **Source:** ENV-703 (`OPENWORLD_AUDIT_RESULTS.md`).

### P1 — Spend the expensive source early and sparingly
*(opex knob)*

Front-load one or two truthful queries on the earliest in-pool questions, then stop. The
dominant lever is **how many** expensive queries you buy, not **which** ones — so ship a
simple budget policy (`first@1`–`first@2`) and skip the fragile per-question scorer. One
well-placed query captures ~70–73% of the fidelity gap; two (`first@2`) reach the ceiling
exactly. Threshold policies go bang-bang because the asked questions cluster at nearly the
same EIG, so there is little to target.

* **Prevents:** overspending on a truthful oracle; over-engineering a targeting heuristic.
* **Source:** ENV-704-B (`TWOORACLE_RESULTS.md`).

### P2 — Run a catch-all novelty detector when the catalog is incomplete
*(conditional)*

Add a Pitman-Yor catch-all hypothesis; posterior mass flowing to it flags a secret that
isn't in your pool — which, in production, it usually isn't (AUC ≈ 1.000 clean, 0.997 at
10% noise). Note the dependency: this is the thing the P0a hard-gate protects. Naive
recovery blinds it, so it earns its place **on top of** the gated backend, not instead of
it.

* **Prevents:** silently forcing an out-of-catalog secret onto a wrong in-pool guess.
* **Source:** open-world extension (`OPENWORLD_RESULTS.md`).

### × — Skip up-front attribute onboarding
*(don't build)*

When the question budget is loose relative to the pool, eliciting an attribute row before
the game doesn't pay — optimal effort is zero (E\* = 0), and a *partial* diagnostic row is
actively harmful (7–10% wrong at E=4/8). Useful mainly as a "don't invest here" result:
save the engineering time. Revisit only if your budget/pool ratio is genuinely tight.

* **Prevents:** wasted build effort on a feature that nets negative at low effort.
* **Source:** ENV-704-A (`OPENWORLD_ONBOARDING_RESULTS.md`).

## Before you trust the numbers

* **Budget-to-pool ratio is the hinge.** "Spend sparingly" (P1) and "skip onboarding" both
  hold because the budget was loose relative to the pool. Verify your real ratio matches
  before importing the cost conclusions — a tight-budget / large-pool regime is where
  fidelity starts to matter (ENV-704-B's hard regime opens a 0.11–0.19 gap).
* **The targeting edge is small and grows slowly.** Front-loading beats random spend by only
  a little (+0.008 at N=40), widening to ~+0.03 at N=120 as decision points multiply, but it
  stays modest. Don't build infrastructure to chase it.
* **Study assumptions.** Findings come from closed-pool, soft-posterior simulations with
  cheap noise on attribute answers only, pools subsampled from the 158 objects, and carry
  ~±0.02 sampling noise at the quoted sample sizes. Read the trends, not the third decimal.
* **P0 is not regime-dependent.** Unlike the cost findings, the hard-gate and op-log
  recommendations are architectural invariants — they hold regardless of budget, pool size,
  or noise.

## Bottom line

Ship the discrete, hard-gated backend with the append-only op-log — safe and auditable in
one move. Spend your expensive-oracle budget early and sparingly. Add a catch-all if your
catalog is incomplete. Don't build onboarding. And confirm your budget-to-pool ratio matches
the studies before trusting the cost knobs.
