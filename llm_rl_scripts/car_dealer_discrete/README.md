# Discretized Car Dealer × Jev — study index

A finite, exactly-scored bilateral negotiation game *inspired by* the LMRL-Gym Car Dealer task,
used to ask whether a non-generative typed-decision model (TypeSafe **Jev**, `Choice` primitive)
can negotiate — and what it is missing when it cannot. Game version `car-dealer-discrete-v0.1`.
This directory is self-contained: the specification, the engine, the Jev policy, every experiment's
pre-registration, logs, frozen analyses and results notes, and a verifier that regenerates all of it.

**Start here:** [`discretized_car_dealer_experiment.md`](discretized_car_dealer_experiment.md) is the
design specification every note cites as "the spec" (§ numbers refer to it). It was written before any
code and is versioned here verbatim.

## Chronology and chain of findings

Each step was run live against `jev-latest` on 2026-10-25. "Pre-reg" = rules fixed before data,
verdicts applied by code and frozen before being read. All intervals are paired, type-pair-clustered
bootstraps; the unit of inference is the ordered type pair.

| # | Step | Result (one line) | Where |
|---|---|---|---|
| 0 | Exact engine (Milestone 1) | 27 deals, 6 rankings, `Fraction`-exact utilities; reproduces spec feasibility 36/36 easy, 30/36 hard | `game.py`, `scoring.py`, `protocol.py`, `test_game.py` |
| 1 | J/J wired; first live smoke | 3/3 walk — each seat offers only its own ideal, never concedes | `jev/policy.py`, `jev/run_jj.py`, `jev/SMOKE_RESULTS.md` |
| 2 | Salience probe | Deadline/offer framing does **not** fix it (0/12 both); 0/130 offers ever acceptable → failure is on the **offer side** | `jev/SALIENCE_RESULTS.md` |
| 3 | Policy aid (forward model) + 2×2 | P(accept) annotation alone: **0/24 → 24/24** agreements, efficient deals; Jev uses the number | `jev/aid.py`, `jev/AID_RESULTS.md` |
| 4 | Mechanism controls | Placebo (constant) = baseline → it is the *information*; prior-only = aid for agreement → the **prior** suffices | `AID_RESULTS.md` §Mechanism controls |
| 5 | Scaled quality test (pre-reg) | **Learning improves quality**: Δ Nash +0.158 (0.054, 0.263) | `jev/PREREGISTRATION_QUALITY.md`, `jev/QUALITY_RESULTS.md` |
| 6 | Seller asymmetry → responder effect (exploratory) | Gain goes to whoever moves **second**, not a role | `QUALITY_RESULTS.md` §Exploratory |
| 7 | Responder effect replication (pre-reg) | P1–P3 confirmed, P4 gate passed, Δ Nash replicates; seller tilt was noise | `jev/PREREGISTRATION_RESPONDER.md`, `jev/RESPONDER_RESULTS.md` |
| 8 | Hard regime r = 0.55 (pre-reg) | Quality effect persists; agreement becomes the lever; learning recognises infeasibility sooner; placebo-gate false positive diagnosed | `jev/PREREGISTRATION_HARD.md`, `jev/HARD_RESULTS.md` |
| 9 | Hard-regime role asymmetry (exploratory) | Lives in the prior-only baseline, not in learning | `HARD_RESULTS.md` §Exploratory |
| 10 | Forced-counter probe (pre-reg) | **Null, well powered**: uninformed counters equally acceptable across roles; natural gap was selection + small n | `jev/PREREGISTRATION_COUNTER.md`, `jev/COUNTER_RESULTS.md` |

**The story in four lines.** Jev fails the game not for lack of preference knowledge but for lack of a
*forward model* of what the other side will accept (1–2). Supplying that model externally unlocks
agreement; its *prior* — which deals most types find acceptable — is enough for that (3–4). Bayesian
*learning* about the specific opponent buys quality, and the gain goes to the second mover, who alone
has an observation to learn from (5–7). This holds in the hard regime, where learning also recognises
when no deal exists; every apparent role asymmetry along the way was sampling noise (8–10).

## Where things live

- **Engine:** `game.py` (deals, rankings, utilities), `scoring.py` (feasibility, U*, G_i, Pareto, Nash),
  `protocol.py` (OFFER/ACCEPT/REJECT/WALK state machine), `test_game.py` (10 tests).
- **Jev:** `../../LLM_RL/jev/client.py` (Choice client; HTTP + mock), `jev/policy.py`, `jev/aid.py`
  (`AcceptanceAid`: bayes / prior_only / placebo), `jev/test_aid.py` (12 tests).
- **Runners:** `jev/run_jj.py` (episodes; `--self-test` offline), `jev/run_counter_probe.py` (single decisions).
- **Analyzers:** `jev/analyze_salience.py`, `analyze_aid.py`, `analyze_quality.py`, `analyze_infeasible.py`,
  `analyze_counter_probe.py`, `explore_seller_asymmetry.py`, `explore_hard_role_asymmetry.py`.
- **Logs and frozen analyses:** `../../results/jj_*_20261025/` (each note names its JSON and its
  regeneration command; pooled logs are derived — `jj_quality_pooled_20261025/REGENERATE.sh`).
- **Figures:** `figures/` (PNG, SVG, and a values sidecar per figure), produced by `jev/plot_*.py` from frozen JSON.
- **Verifier:** `jev/verify_cited.py` regenerates every frozen analysis, re-applies every pre-registered
  rule and re-derives the gate diagnosis, then diffs. Run it after any analyzer change.

## Reproduce

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.test_game           # engine, no API
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.test_aid        # aid + policy, no API
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.run_jj --self-test   # full loop, no API
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.verify_cited    # every frozen number
```

Live runs need `TYPESAFE_API_KEY`. `jev-latest` is stochastic server-side; `--seed` is provenance only.

## Conventions

- Results notes are **rendered from frozen JSON** — no number is typed by hand. Claims are tagged
  *exact-on-logs* (deterministic function of the committed logs) or *stochastic* (a draw, with spread).
- Confirmatory programs are **pre-registered**; verdicts are applied by code and frozen before reading.
  Exploratory sections say so in their first line and never promote a finding.
- Bootstrap unit = ordered type pair, paired across conditions. With 36 pairs (the whole type space)
  pair-clustered intervals have a precision ceiling that more repeats cannot lower.
- A placebo gate stated as "all four CIs include 0" has an 18.5 % false-fail rate per run under the null
  (observed in step 8); future gates carry a multiplicity correction or magnitude bound.
- **Figures are regenerable like numbers.** A plot reads frozen JSON only, writes a `<stem>.values.json`
  sidecar of what it drew, and its caption names, per panel, the analyzer and result directory it came
  from plus the plotting script. Figures live in `figures/`; scripts in `jev/plot_*.py`.

## The exact Jev interface

Generated from `policy.py` by `jev/show_prompt.py` — rerun it rather than editing this block.

```text
Canonical state: buyer, ranking price > warranty > delivery, r = 0.40, after seller opened P2-W0-D0.

==============================================================================
CONDITION: baseline
==============================================================================
--- instructions ---
You are the buyer negotiating the sale of one car. A deal sets three attributes — price, warranty, delivery — each at one of three levels. As the buyer you prefer lower price, longer warranty, and faster delivery. Your PRIVATE priority ranking, most to least important, is: price > warranty > delivery. Every deal is worth a score between 0 (worst for you) and 1 (best for you). Both you and the other party have a reservation value of 0.400: a deal only holds if it is worth at least 0.400 to BOTH sides, and if no deal is reached you each receive exactly 0.400. You have a limited number of proposals. Choose the single action giving the best outcome for YOU — you do not know the other party's private ranking.
--- state ---
Proposals used: 1 of 6.
There is an offer on the table from the other party: price=high, warranty=short, delivery=slow [P2-W0-D0] (worth 0.000 to you).
Negotiation so far:
  - seller proposed P2-W0-D0
--- example options (of 30) ---
  [ACCEPT P2-W0-D0] Accept the offer on the table (price=high, warranty=short, delivery=slow), worth 0.000 to you. This ends the negotiation with an agreed deal.
  [OFFER P1-W1-D1] Propose price=medium, warranty=medium, delivery=medium [P1-W1-D1]; this deal is worth 0.500 to you.
  [OFFER P0-W2-D2] Propose price=low, warranty=long, delivery=fast [P0-W2-D2]; this deal is worth 1.000 to you.
  [WALK] Walk away with no deal. You then receive only your reservation payoff (0.400).

==============================================================================
CONDITION: salience
==============================================================================
--- instructions ---
You are the buyer negotiating the sale of one car. A deal sets three attributes — price, warranty, delivery — each at one of three levels. As the buyer you prefer lower price, longer warranty, and faster delivery. Your PRIVATE priority ranking, most to least important, is: price > warranty > delivery. Every deal is worth a score between 0 (worst for you) and 1 (best for you). Both you and the other party have a reservation value of 0.400: a deal only holds if it is worth at least 0.400 to BOTH sides, and if no deal is reached you each receive exactly 0.400. You have a limited number of proposals. Choose the single action giving the best outcome for YOU — you do not know the other party's private ranking.
--- state ---
Proposals used: 1 of 6.
There is an offer on the table from the other party: price=high, warranty=short, delivery=slow [P2-W0-D0] (worth 0.000 to you).
Proposals remaining for both sides combined: 5. If they run out with no accepted deal, the negotiation ends and you receive exactly 0.400 — your no-deal payoff.
The best offer the seller has made to you so far is price=high, warranty=short, delivery=slow [P2-W0-D0], worth 0.000 to you — below your no-deal payoff of 0.400.
Accepting the offer on the table now gives you 0.000; walking away or running out of proposals gives you 0.400.
Negotiation so far:
  - seller proposed P2-W0-D0
--- example options (of 30) ---
  [ACCEPT P2-W0-D0] Accept the offer on the table (price=high, warranty=short, delivery=slow), worth 0.000 to you. This ends the negotiation with an agreed deal.
  [OFFER P1-W1-D1] Propose price=medium, warranty=medium, delivery=medium [P1-W1-D1]; this deal is worth 0.500 to you.
  [OFFER P0-W2-D2] Propose price=low, warranty=long, delivery=fast [P0-W2-D2]; this deal is worth 1.000 to you.
  [WALK] Walk away with no deal. You then receive only your reservation payoff (0.400).

==============================================================================
CONDITION: aid (bayes)
==============================================================================
--- instructions ---
You are the buyer negotiating the sale of one car. A deal sets three attributes — price, warranty, delivery — each at one of three levels. As the buyer you prefer lower price, longer warranty, and faster delivery. Your PRIVATE priority ranking, most to least important, is: price > warranty > delivery. Every deal is worth a score between 0 (worst for you) and 1 (best for you). Both you and the other party have a reservation value of 0.400: a deal only holds if it is worth at least 0.400 to BOTH sides, and if no deal is reached you each receive exactly 0.400. You have a limited number of proposals. Choose the single action giving the best outcome for YOU — you do not know the other party's private ranking. Each proposal also shows an estimate of how likely the other party is to accept it, based on their behaviour so far.
--- state ---
Proposals used: 1 of 6.
There is an offer on the table from the other party: price=high, warranty=short, delivery=slow [P2-W0-D0] (worth 0.000 to you).
Negotiation so far:
  - seller proposed P2-W0-D0
--- example options (of 30) ---
  [ACCEPT P2-W0-D0] Accept the offer on the table (price=high, warranty=short, delivery=slow), worth 0.000 to you. This ends the negotiation with an agreed deal.
  [OFFER P1-W1-D1] Propose price=medium, warranty=medium, delivery=medium [P1-W1-D1]; this deal is worth 0.500 to you; estimated chance the other party accepts: 0.95.
  [OFFER P0-W2-D2] Propose price=low, warranty=long, delivery=fast [P0-W2-D2]; this deal is worth 1.000 to you; estimated chance the other party accepts: 0.05.
  [WALK] Walk away with no deal. You then receive only your reservation payoff (0.400).
```

## Open questions

Hard regime and the forced-counter probe are single runs. One prompt throughout; candidate-order
randomization (spec §5.1.6) not run; β and ε of the aid unswept. Why Jev-as-seller and Jev-as-buyer pick
*different specific deals of equal quality* under mirror-identical information is recorded, not explained.
