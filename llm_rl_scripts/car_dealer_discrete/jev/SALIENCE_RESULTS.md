# J/J salience probe — results

**Date:** 2026-10-25 · **Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` (live) · **Status:** probe complete; small-n

Every number below is regenerated from the committed logs by one command. Do not
edit a figure here without re-running it:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_salience \
    results/jj_salience_20261025/baseline results/jj_salience_20261025/salience
```

Claims are tagged **exact-on-logs** (a deterministic function of the committed
JSONL — any drift means the logs or analyzer changed) or **stochastic** (a draw
from live `jev-latest`; the number is a sample, reported with its spread, not a
rate). The pilot-A non-replication in the Text-Nav program is why this
distinction is enforced.

## Question

Jev-only J/J walks away in every episode. The diagnosis: Jev lacks the two things
RL training installs — (1) a forward/acceptance model of the opponent and
(2) internalized deadline cost. The **salience** condition targets (2) only: the
state restates the deadline and the standing/best-seen offer in *value* terms
relative to the no-deal payoff. No strategic advice. Instructions, own-utility
reveal, and the action set are identical to baseline, so the state text is the
only factor.

## Design (matched)

Same 3 type pairs in both conditions (`--pairs 3 --seed 7`), easy regime
(r = 0.40, every pair has an IR deal), both starting sides, K = 2 independent
rollouts → **12 episodes per condition**, 0 API errors. Runner flags:
`--regime easy --start both --repeats 2`, plus `--salience` for the treatment.

## Results

| Claim | Baseline | Salience | Tag |
|---|---|---|---|
| Agreements | 0 / 12 | 0 / 12 | exact-on-logs; Wilson 95% [0.000, 0.243] each |
| Outcome of every episode | walk | walk | exact-on-logs |
| Offers made (total) | 72 | 58 | exact-on-logs |
| Offers that were the offerer's own ideal (own utility = 1.0) | 72 / 72 | 57 / 58 | exact-on-logs |
| **Offers worth ≥ r to the recipient** | **0 / 72** | **0 / 58** | exact-on-logs; pooled 0/130, Wilson 95% [0.000, 0.029] |
| Distinct deals offered — buyer | `P0-W2-D2` ×36 | `P0-W2-D2` ×23 | exact-on-logs |
| Distinct deals offered — seller | `P2-W0-D0` ×36 | `P2-W0-D0` ×34, `P1-W0-D0` ×1 | exact-on-logs |
| Who walked | starter, always, at decision #7 (6+6) | starter 9, non-starter 3; at decisions #6, #7, #9 | exact-on-logs |
| Confidence at first offer (mean [min, max], n=24) | 0.724 [0.54, 0.92] | 0.646 [0.31, 0.93] | stochastic — ranges fully overlap |
| Confidence at last offer (n=24) | 0.399 [0.27, 0.56] | 0.370 [0.22, 0.71] | stochastic — ranges overlap |
| Confidence at walk (n=12) | 0.747 [0.66, 0.83] | 0.578 [0.30, 0.69] | stochastic — ranges nearly separated (overlap only 0.66–0.69) |

## Interpretation

**Salience did not fix the outcome.** 0/12 agreements in both conditions. At n=12
this rules out a large effect, not a modest one (upper bound ≈ 24%).

**The decisive finding is the confound check, and it is tight.** Across both
conditions, **0 of 130 offers were worth ≥ r to the recipient** (upper bound
≈ 3%). So the salience text "the best offer so far is worth 0.000 — below your
no-deal payoff" was *true in every episode*, and WALK was the correct response at
every deadline. The accept-side mechanism salience targets never had a decent
offer to act on. **The binding constraint is on the offer side:** neither agent
ever concedes enough to put a mutually-acceptable deal on the table. That is
gap (1), the forward model — not gap (2), deadline salience. An accept-side
intervention cannot repair a failure to generate acceptable offers.

**Salience was not inert — it moved the reactive axis, as in Text-Nav.** The
baseline is perfectly rigid: 72/72 own-ideal offers, and the walker is always
whoever started, always at decision #7. Under salience: the first concession
observed in any run (`P1-W0-D0`, once — still worth < r to the buyer), fewer
offers before quitting (58 vs 72), three non-starter walks, and walks at
decisions #6 and #9 (an early quit, and a REJECT-extended game). Walk confidence
is the one confidence shift with nearly separated ranges. These are consistent
with Jev *recognizing* the no-deal payoff as real and taking it earlier — but
not with it *composing* a concession that crosses the other side's reservation.

**What is robust vs suggestive.** Robust: 0/12 outcome; 0/130 decent offers;
baseline rigidity (72/72). Suggestive only (single-draw-scale at K=2, do not
cite as rates): the one concession, 58-vs-72, the three non-starter walks, the
walk-confidence shift. A correction to the earlier verbal summary: it said
confidence was "lower everywhere"; the spread shows only walk confidence
plausibly shifted — first- and last-offer ranges overlap fully.

## Implication for the next remedy

The remedy must act on **offer generation**, not acceptance: an external
acceptance-probability / belief model that annotates each candidate offer (the
design note's "Jev + explicit belief filter" row), leaving Jev the decision.
Two mutually-blind BED-style learners are one way to build that filter. Salience
should be retained as a cheap additive factor in that design, since it appears
to move the quit/accept side.

## Scope limits

n = 12 per condition, 3 pairs, easy regime only, K = 2. jev-latest is stochastic
at near-tie decisions; per-cell rates would need K ≈ 8–10 across many pairs with
seed-clustered intervals, as in the Text-Nav main run. Nothing here is a claim
about Jev's negotiation ceiling under other prompts.
