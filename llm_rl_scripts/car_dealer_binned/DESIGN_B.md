# Binned Car Dealer, Condition B — design note (`binned-car-dealer-b-v0.1`)

**Source document:** [`binned_car_dealer_experiment.md`](binned_car_dealer_experiment.md) (copied
verbatim from the pasted design, 2026-10-26). This note records how that document was turned
into an exactly solvable simulator, every place the implementation departs from it, and why.
Numbers live in [`SCRIPTED_RESULTS.md`](SCRIPTED_RESULTS.md), rendered from frozen JSON.

## 1. What was kept from the document

- **Condition B first** (§7, §10): one learner may ask or offer each turn; a deal closes the game.
- **Truthful questions with known response distributions** (§2, "a crucial experimental control").
- **Exact EIG for every option at every decision**, EIG regret, decision regret against a
  dynamic-programming reference, and the best-minus-second-best Q gap for H3 (§3, §4, §8).
- **Labelled options in randomised order** for the Jev arm, with Jev's full option-probability
  vector logged so action-selection entropy is one call, not repeated samples (§2, §6).
- The three scoring dimensions (§4) and the entropy trajectory / cumulative exposure (§5).
- No reward shaping (§9).

## 2. Departures, each with the reason

| Document | Implementation | Why |
|---|---|---|
| 4 label types per side ("lowest price", …) | the **6 strict priority rankings** of `car_dealer_discrete` (prior entropy log2 6 ≈ 2.585 bits) | §1's own warning: label types reward *identifying* the opponent, not negotiating. Ranking types already carry a complete exact utility over price, warranty, delivery. |
| both sides infer (bilateral) | learner vs a **frozen scripted opponent** (public model, hidden type; truthful answers, deterministic individually-rational acceptance, never counters, holds no belief) | Q*(b, a) is only defined against a fixed opponent policy; with two adapting sides "optimal" means an equilibrium of a Bayesian game, not a DP. The bilateral entropy tracking of §5 is therefore degenerate here: only the learner's entropy moves. Self-play is a later, exploratory cell. |
| offers carry 0 bits (§2 table) | offers are **informative**: a rejection eliminates every type for which the deal was acceptable, an acceptance is likewise a likelihood | follows from the public acceptance rule; the EIG of an offer at the uniform prior is the binary entropy of the acceptor share (checked by test). The document's illustrative "0.000 bits" is not a property of this game. |
| four candidate actions | **2 highest-EIG unasked questions + 2 offers** named by a public rule over the deals worth ≥ r to the learner itself (SAFE = most likely accepted, AMBITIOUS = best own utility among deals some supported type accepts; BALANCED fills only if they coincide); the final decision is offers-only | a menu that is a deterministic function of the public state is what lets the reference optimise *within* the menu (§4, Q* over a ∈ A_t). The unconstrained game is also solved to bound what the rule costs; under zero noise the bound is tight at every horizon tested (see results). |
| six rounds then a final offer | configurable `rounds` (+1 offers-only decision); **primary pre-registered horizon = 2 rounds**, 6 rounds kept as a secondary condition | with nine ≥0.9-bit questions, six rounds leave information free: the horizon never binds, greedy-EIG equals the exact policy, and the exact policy stops asking altogether because an offer both probes and can close. The trade-off that Condition B exists to measure binds only at 2 rounds (see sweep). Discounting per turn would be the alternative lever; the document frames cost as turns, so the horizon is the knob used. |
| "exact Bayesian policy" | **exact DP over (t, belief, informative-asked-set)** in Fractions; the identity *mean realised utility of the exact arm over a uniform opponent = V(1, uniform)* holds to equality | a reference that is exact in Fractions can be checked as an identity rather than approximately. |

## 3. The game, precisely

- Learner role ρ ∈ {buyer, seller}, private ranking θ_L (known to it); opponent ranking θ_O hidden,
  uniform prior over the 6 rankings. Reservation r: easy 2/5, hard 11/20 (as in `car_dealer_discrete`).
- Decisions t = 1 … rounds: menu = questions + offers. Decision t = rounds + 1: offers only.
- `ASK q`: opponent answers truthfully (optional noise ε, default 0); belief updates exactly.
- `OFFER d`: accepted iff U_O(d; θ_O) ≥ r (optional noise floor, default 0). Acceptance ends the game
  at U_L(d). Rejection consumes the turn and updates the belief. A rejected final offer ends at r.
- Question bank (9, binary): TOP(a) and BOTTOM(a) for each attribute (2/6 yes → 0.918 bits from
  the prior), ABOVE(a, b) for the three pairs (3/6 yes → exactly 1 bit). Three ABOVE answers identify
  the type. Zero-EIG questions stay in the bank and surface as the natural distractors.
- Per decision the log holds: the annotated menu (EIG, P(accept), own utility, myopic value),
  Q* of every option, the choice and its menu position, the outcome, EIG regret, decision regret,
  the Q gap, and for Jev the letter map, probability vector, entropy and confidence.
- Scoring per episode: learner utility, gain G = (U − r)/(U* − r) with U* the learner's best
  individually-rational deal for the true pair, agreement, Nash ratio and Pareto flag from
  `car_dealer_discrete.scoring`, entropy trajectory, cumulative entropy, both regrets summed.

## 4. Arms

| Arm | Rule | Stochastic? |
|---|---|---|
| `exact` | argmax Q*, ties → act (OFFER), then menu order | no (zero noise) |
| `greedy_eig` | argmax EIG, ties → menu order | no |
| `myopic` | never asks; offer with the highest P(accept)·U + (1 − P(accept))·r | no |
| `random` | uniform over the menu | yes (seeded) |
| `jev_plain` | one Choice over lettered options; own utility of offers shown; **no** EIG, P(accept), tags or posterior shown | yes (`jev-latest`) |
| `jev_eig` | as above plus each option's exact EIG in bits | yes |

Deterministic arms are played once per cell; stochastic arms `--repeats` times. All arms face the
identical menu at the identical public state.

## 5. What the simulator already decides without any model (exact-on-logs)

See `SCRIPTED_RESULTS.md`. In short: the menu rule is costless (V_menu = V_full) at every horizon
and regime tested; greedy-EIG is optimal whenever the horizon does not bind and loses to the exact
policy only when it does; the exact policy prefers probing offers to questions once it has slack,
and the never-ask policy is close to optimal in the hard regime. Those are properties of the game, and they are why the
document's H2 ("greedy EIG can underperform in Condition B") is *true but horizon-dependent*.

## 6. Not built here

Condition A (questions only), C (strategic disclosure; no exact reference without an equilibrium),
D (natural language), the conventional-LLM and BED-LLM arms, self-play, reward shaping, and the
entropy trajectory of a believing opponent. The hooks exist (`entropy_trajectory_bits` is per
learner); the opponent is the piece that would change.
