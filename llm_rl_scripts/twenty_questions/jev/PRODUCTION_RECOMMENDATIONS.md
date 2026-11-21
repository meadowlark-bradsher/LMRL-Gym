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
soften the belief that gates the human. Make the test **budget-aware**: reinstate only
when a turn is left to test the candidate, else escalate. That is the `modeswitch_guard`
backend (`P0A_BACKEND`; `modeswitch` with the budget guard on). Naive reversibility
(scalar `retract`, CMBS-aligned `tombstone`) recovers in-pool loss but corrupts everything
downstream.

* **Prevents:** confident wrong guesses, a blinded novelty detector, and a game that ends
  on a reinstated known nobody tested.
* **Evidence:** `modeswitch` recovers in-pool (0.149 < hard 0.175 @10% noise), keeps the
  detector (AUC 0.926 ≥ hard), and manufactures zero wrong-guesses — whereas naive
  reversibility drives AUC to chance and invents wrong-guesses. The guard changes none of
  these numbers (identical in every per-seed metric of that grid). At larger pools
  (N ≥ 500, ST-1) unguarded `modeswitch` can reinstate on the final answer and end
  `wrong`; the guard turns those games into escalations, roughly halving the primary
  arm's noisy in-pool `wrong` worst case (0.058 → 0.030 at 10% flips) with no change to
  any cell's `won`.
* **Precondition:** an escalation must cost less than a wrong answer. A deployment that
  prefers a final untested best guess to an escalation should use unguarded `modeswitch`.
* **Source:** ENV-702 (`OPENWORLD_RETRACTION_RESULTS.md`); guard: `STRESS_REINSTATE_GUARD.md`
  (run `results/stress_reinstate_guard_20261112/`, gates G1–G3 all pass).

### P0a′ — Verify after every remediation; if the incident has not cleared, escalate
*(safety-critical, cheap; the condition that keeps P0a safe when there is more than one cause)*

The gated loop assumes one hidden cause. It stops at the first one it can confirm, and it
never confirms a wrong one. When two causes are present, the usual result is a **silent
partial answer**: one cause is fixed, the other is never mentioned, and nothing in the loop
signals it. So after acting on the confirmed hypothesis, check whether the incident has
actually cleared. If it hasn't, escalate to the expert. Do not resume elimination on your
own (explain-away and pairs as hypotheses were not built; FLYW-8 is not needed).

* **Prevents:** an incident left half-fixed because the loop's answer explained only some
  of the symptoms. Without the check, the missed second cause becomes a wrong action.
* **Evidence (FLYW-7, registered):** P0a against a two-secret OR oracle, all 9,870 pairs of
  known objects, at 5% and 10% answer noise.
  * It named exactly one of the two causes in 0.90 and 0.83 of games, never both, and never
    an object outside the pair. 0.89 and 0.80 ended as silent partials.
  * Two causes emptied the survivor set about as often as one did (excess +0.012 and
    −0.007), and the catch-all never fired on them. The second cause added almost no wrong
    guesses (+0.16 and +0.00 per incident).
  * With the check, the second cause costs about one expert call: a compound penalty of
    +12.4 at 1 per probe, 10 per expert call and 50 per wrong action. Without it, the
    penalty is +53 and +41, about one wrong action per incident.
* **Caveats:**
  * **Verification was assumed perfect.** A check that sometimes reports "cleared" when the
    incident hasn't cleared moves the cost toward the unverified figure.
  * **The cost bar holds at the primary costs only.** At (30, 150) and (100, 500) the
    verified penalty is about 1.7 expert calls at 5% noise, not one.
  * **An empty survivor set still escalates, but it isn't evidence of a new cause by
    itself.** Under noise, the loop's own answers can't tell a new cause from two known
    ones or from an answer error (P2).
* **Source:** FLYW-7 (`FLYW2_COMPOUND_PREREG.md`, tag `prereg-flyw2-compound-20261119`;
  `FLYW2_COMPOUND_RESULTS.md`, run `results/flyw2_run_20261119/`); FINDINGS §3.16.

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
* **Don't admit from exhaustion alone.** An empty survivor set is a reason to escalate, not
  proof of a new cause. In FLYW-7, with a quarter of incidents having two causes, only
  33–50% of exhaustion escalations came from a truly new cause; the rest were answer errors
  and compounds. A classifier on the loop's own answers could not separate the three (AUC
  0.41, macro recall 0.30). A new cause can also hide behind a known one: in 40–44% of
  incidents with one known and one new cause, the loop fixed the known cause and stopped.
  Only P0a′'s check sends the new one to the expert.
* **Source:** open-world extension (`OPENWORLD_RESULTS.md`); exhaustion and compounds:
  FLYW-7 (`FLYW2_COMPOUND_RESULTS.md`).

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
* **P0a′'s cost is regime-dependent.** What holds at every price tested: the loop never
  confirmed a cause outside the pair, and a second cause added almost no wrong guesses. How
  much the second cause costs depends on the price of an expert call and a wrong action, and
  on the check being reliable.

## Bottom line

Ship the discrete, hard-gated backend with the append-only op-log — safe and auditable in
one move. Verify after every remediation, and escalate if the incident hasn't
cleared. Spend your expensive-oracle budget early and sparingly. Add a catch-all if your
catalog is incomplete. Don't build onboarding. And confirm your budget-to-pool ratio matches
the studies before trusting the cost knobs.
