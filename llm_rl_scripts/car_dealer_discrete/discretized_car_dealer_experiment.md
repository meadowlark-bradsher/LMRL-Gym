# Discretized Car Dealer: Bilateral Information Gain and Negotiation

**Design note / experiment specification — v0.1**  
**Date:** 2026-10-25  
**Status:** Proposed experiment; no model results yet  
**Candidate agents:** Jev, conventional LLMs, BED-LLM or a clearly labeled BED-style implementation

## 1. Research question

Can two agents, each with private preferences, learn enough about the other through a sequence of structured proposals to reach an individually acceptable, Pareto-efficient agreement? Does explicitly optimizing information acquisition improve the *eventual deal*, and under what opponent policies?

This is a **discretized redesign inspired by LMRL-Gym Car Dealer**, not a drop-in replication of its environment or reward. The original task is seller-centered: the dealer negotiates with a buyer and is rewarded for the final sale price. Here **both agents are decision-makers and both receive separate scores**. The finite type and action spaces make exhaustive outcome scoring possible.

**Primary outcomes:** (i) buyer's realized benefit, (ii) seller's realized benefit, (iii) efficiency and fairness of the agreement, and (iv) what each agent learned about the other's private type. Keep these distinct; do not declare an agent the winner on information gain alone.

## 2. Minimal finite game

### 2.1 Public deal space

A proposed car sale is a tuple `d = (price, warranty, delivery)`:

| Attribute | Three possible levels | Buyer generally prefers | Seller generally prefers |
|---|---|---|---|
| Price | low / medium / high | lower | higher |
| Warranty | short / medium / long | longer | shorter |
| Delivery | fast / medium / slow | faster | slower |

There are **3³ = 27 legal deals**. A deal has a stable ID, and all agents see the same deal catalog. Monetary amounts are intentionally abstract: these are preference dimensions, not a literal model of real automobile retail economics.

### 2.2 Private agent types

Each agent privately receives a strict ranking of the three attributes: **highest, middle, lowest priority**. There are `3! = 6` possible rankings per agent and **36 buyer–seller type pairs** in the basic factorial design.

Example: `buyer = price > warranty > delivery`; `seller = delivery > price > warranty`.

In v0, only the **ranking** is hidden. Both agents know the public attribute direction, utility formula, reservation rule, legal deals, turn budget, and prior distribution over types. Sample the two rankings independently and uniformly, unless testing a deliberately nonuniform prior.

The distinction matters: independent initial types have zero mutual information **with each other**. The experiment concerns the information that **each player learns from the other's behavior**, not increasing `I(type_buyer; type_seller)`.

### 2.3 Exact, inspectable utilities

Encode satisfaction on each attribute as `0`, `0.5`, or `1`, with the directions above. Assign public priority weights:

- highest: `9`
- middle: `3`
- lowest: `1`

For player `i` with ranked attributes `(a1, a2, a3)`, define

\[
U_i(d;\theta_i)=\frac{9s_i(a_1,d)+3s_i(a_2,d)+s_i(a_3,d)}{13},\quad 0\le U_i\le 1.
\]

On these three-level attributes, moving the highest-priority attribute by one level outweighs the maximum possible combined change in the other two (`9 × 0.5 > 3 + 1`); moving the middle attribute by one level outweighs the full change in the lowest (`3 × 0.5 > 1`). Thus the weighted sum implements the stated priority ordering without an ambiguous natural-language rubric.

Use a **public outside option** `r_i = r` for each side. A deal is individually rational (IR) only when both utilities meet their outside options:

\[
F(\theta_B,\theta_S)=\{d:\ U_B(d)\ge r_B\ \text{and}\ U_S(d)\ge r_S\}.
\]

Run two initially useful regimes (checked by exhaustive enumeration of the 36 type pairs and 27 deals):

| Regime | Outside option for each side | Type pairs admitting at least one IR deal | Why it matters |
|---|---:|---:|---|
| Easy | `r = 0.40` | `36 / 36` | Every episode has a possible agreement; failure to agree is avoidable. |
| Hard | `r = 0.55` | `30 / 36` | Some pairs should rationally end with no deal; tests recognition of incompatibility. |

These counts apply **only to the exact v0 utility definition**. Any changes to weights, levels, or reservation rules require recomputing them.

### 2.4 A worked deal

For the example rankings above, take `d = (medium price, long warranty, slow delivery)`:

- Buyer: `(9×0.5 + 3×1 + 1×0)/13 = 7.5/13 ≈ 0.577`.
- Seller: `(9×1 + 3×0.5 + 1×0)/13 = 10.5/13 ≈ 0.808`.

Both acceptability thresholds are passed even in the hard regime (`r = 0.55`). That **does not** establish that this is the best attainable deal for either side, or Pareto-efficient; the evaluator determines those separately by exhaustive search.

## 3. Two-agent protocol

### 3.1 What each agent sees

Each participant receives:

1. Its **own private ranking**, never the opponent's ranking.
2. The public rulebook and complete legal deal catalog.
3. The public sequence of proposals and responses so far.
4. Its own remaining turn/compute budget.

The evaluator retains both hidden rankings but **must not expose the opponent's true utility, feasibility table, optimal deal, or evaluator labels** through prompts, tools, action descriptions, logs, or examples.

### 3.2 Core action space: proposals, not free-form persuasion

Use the cooperative **proposal-only** interpretation first; the agents need not generate sales rhetoric.

- `OFFER(deal_id)`: propose one of 27 deals; this becomes the outstanding offer.
- `ACCEPT`: accept the outstanding offer and end the episode.
- `REJECT`: decline it without agreement.
- `WALK`: end without an agreement.

At the receiving turn, the agent can `ACCEPT`, `REJECT`, `WALK`, or reply with a **counteroffer**, represented by `OFFER(new_deal_id)`. A counteroffer replaces the outstanding offer. Turns alternate, and the starting side is counterbalanced. Set **six total proposal opportunities** in v0; record total decisions separately, so counteroffers do not silently extend the budget.

The environment verifies that an accepted deal exists and that **both parties** meet their reservation utility. Acceptance of a non-IR deal is an invalid action (logged as such, with no enforceable agreement), not a profitable sale. A participant can reject an IR deal to seek a better offer; rejection **does not prove** that the offered utility was below its reservation value.

At the deadline, an outstanding but unaccepted offer does not count as a deal. The no-deal payoff equals each player's outside option.

**Variant:** compare `reject-only` against `reject-with-counteroffer` responses. The latter makes communication more informative but changes the action budget and must be reported as a separate protocol condition.

### 3.3 Why this is different from ordinary 20 Questions

In 20 Questions, a fixed secret is typically queried through an oracle with a known response mechanism. Here **both agents choose actions** while simultaneously adapting to what they have learned. Offers may express preferences, test acceptance, anchor negotiations, or trade information against concessions. The observation model is therefore **policy-dependent and endogenous**. A rejection may be strategic; a proposal is not a truthful ranking report.

This is a finite **partially observable stochastic game / bilateral Bayesian negotiation game**, not automatically a single-agent POMDP with a stationary oracle. Given a fixed opponent policy it can be analyzed as a POMDP from one participant's perspective.

## 4. How to score success: keep four score families separate

### 4.1 Individual buyer and seller outcome scores

For an actual accepted IR deal `d`, calculate the private utility `U_i(d)`. If there is no deal, define the player's realized payoff as the outside option `r_i`.

For each **feasible** type pair, compute the best individually attainable utility while still respecting the opponent's reservation:

\[
U_i^*=\max_{d\in F(\theta_B,\theta_S)} U_i(d).
\]

Define each agent's **normalized attainable-gain score**:

\[
G_i=\frac{U_i(\mathrm{outcome})-r_i}{U_i^*-r_i},\qquad i\in\{B,S\}.
\]

For feasible v0 cases, `0 ≤ G_i ≤ 1`; `G_i=0` for no agreement, and `G_i=1` means the player obtained its *own* best outcome among mutually acceptable deals. **Both sides generally cannot reach 1 in the same transaction.** Publish `G_B` and `G_S` separately; also report raw `U_B` and `U_S`. If the denominator is ever zero after a rule change, report the case as undefined for this metric rather than divide by zero.

This answers **“How did each side do?”** without declaring that the seller wins whenever the buyer loses. The score is relative to what was possible *given both actual preferences*, which the evaluator knows but the participants do not.

For a pair with `F = ∅`, **no deal is the correct outcome**. Mark the individual `G_i` as not applicable and score the episode under *correct-no-deal rate*, not as two failures. Report feasible and infeasible episodes separately so a policy cannot inflate success by always walking away.

### 4.2 Joint deal efficiency and fairness

Agreement alone is inadequate. For every accepted deal, compute:

**Individual rationality:** Was `U_B(d) ≥ r_B` **and** `U_S(d) ≥ r_S`? Invalid acceptances cannot earn a deal reward.

**Pareto efficiency:** Is there **no** other feasible deal `d'` with both agents at least as well off and at least one strictly better off? Report **fraction of accepted deals on the Pareto frontier**.

**Normalized Nash surplus ratio:**

\[
N(d)=\big(U_B(d)-r_B\big)\big(U_S(d)-r_S\big),\qquad
Q_N=\frac{N(d)}{\max_{d'\in F}N(d')}.
\]

Set `Q_N=0` for no deal when `F ≠ ∅`; mark it `N/A` when `F = ∅`. This is a useful **joint reference criterion**, not the only acceptable definition of a good negotiation. It rewards joint improvement relative to outside options and avoids equating high sale price with success. In the degenerate case of zero optimal product, define and report a separate convention before running tests.

Also report `min(G_B,G_S)` as a transparent **balanced individual attainment** measure. Do not silently combine the Pareto indicator, Nash ratio, and individual scores into a single unexamined number.

### 4.3 Information gained by *each* agent

Let `θ_B` and `θ_S` denote the private preference rankings. At step `t`, the buyer holds an estimated distribution `q^B_t(θ_S)` over the seller's six possible rankings, and the seller holds `q^S_t(θ_B)`.

The **prospective** expected information gain (EIG), when a defensible response model exists, is

\[
\operatorname{EIG}_B(a\mid h_t)=
I(\theta_S;Y_{t+1}\mid h_t,a,\theta_B,\pi_S),
\]

with an analogous expression for the seller. Here `Y` is the response/observation and `π_S` specifies the opponent's response policy or generative model. Depending on the protocol, the history and candidate action must be conditioned upon appropriately. This is the **information gained about another agent**, not the initial mutual information between independently sampled private rankings.

**Important:** a model's self-reported reduction in entropy does not establish accurate learning. A model can become confidently wrong. Require agents to provide a normalized six-category belief report at specified checkpoints (or maintain an explicit external posterior on their behalf), then evaluate:

- **Log loss of the true opponent type:** `−log₂ q_t(θ_true)`; lower is better.
- **Log-score improvement:** `log₂ q_T(θ_true) − log₂ q_0(θ_true)`; higher is better, can be negative.
- **Multiclass Brier score** and calibration across episodes.
- **Entropy trajectory:** `H(q_t)`; descriptive, **not** proof of correctness.
- **Action costs per meaningful improvement:** turns, model calls, tokens, latency, and dollars.

With an initially uniform six-type prior, entropy starts at `log₂(6) ≈ 2.585 bits` per side. If eliciting belief vectors changes the policy, log it as part of the agent interface and use the same checkpoint protocol across comparisons. Never treat Jev `Choice` probabilities as a calibrated posterior over private rankings without validating that interpretation.

**Reference-posterior caveat:** Exact Bayesian updates based on accept/reject observations require an explicit model of the other agent's decision policy. In the fully interactive setting a rejection is *not* a deterministic preference constraint. A known-rule response oracle can provide an exact-posterior control condition; live Jev/LLM opponents will generally require an estimated policy model or evaluation based on proper scoring rules instead of a purported oracle posterior.

### 4.4 Operational episode success

For a **feasible** type pair, count *deal success* only when a valid agreement is reached. Then report its two individual scores and joint efficiency. For an **infeasible** pair, count *correct no-deal* when the agents end without an invalid agreement. Scores for information acquisition are reported separately in both cases.

This prevents three misleading outcomes: selling at a very high price while harming the buyer, extracting information without reaching an available deal, and forcing an agreement when no acceptable agreement exists.

## 5. Agents and comparisons

Use **one shared interface and one deterministic evaluator** across all policies. The models may differ in their internal decision mechanisms, but none gets to change the legal actions, private information, turn limits, or scoring code.

| Policy condition | Proposed implementation | Scientific distinction |
|---|---|---|
| Random / rule heuristic | Random legal offer; fixed concession strategy | Checks whether the task is nontrivial. |
| Myopic utility | Propose the deal maximizing own utility under a simple public guess about opponent; no exploration | Measures the value of seeking information. |
| **Jev-only** | Supply public history, own private ranking, and enumerated legal actions to `Choice`; choose a typed response | Can a structured, non-generative decision model negotiate? |
| **LLM-only** | Conventional chat LLM, forced to return a legal structured action ID | Baseline language-model reasoning under the same action restrictions. |
| **Jev + explicit belief filter** | Jev picks offers/actions; external finite Bayesian/approximate belief state tracks opponent | Separates decision selection from belief maintenance. |
| **LLM + explicit belief filter** | Same posterior and legal actions, conventional LLM selects | Tests whether representation or policy drives gains. |
| **BED-style policy** | Select proposals according to expected information gain, with a specified response model | Tests information-seeking independently of model branding. |
| **BED-LLM adaptation** | Faithful adaptation of the published BED-LLM method: candidate proposal generation, EIG estimation, and explicit sequential belief updates | Tests the published method when made bilateral. Clearly document divergences. |
| Oracle benchmarks | Exhaustive efficient-deal selector with **both** hidden rankings; optional exact Bayesian planning oracle in a fully specified responder setting | Upper bounds, **not eligible agents**. |

**Jev-specific:** `Choice` selects among finite actions; `Score` and `Noul` may be useful *ablation inputs* (e.g., estimated deal acceptability) but are not proof of Bayesian beliefs. Do not silently repair invalid model actions. Record model and API version, prompt, candidate order, and whether actions were constrained by code.

**BED-LLM-specific:** BED-LLM is principally a sequential experimental-design strategy using candidate queries and EIG about a variable of interest. A straightforward greedy-EIG algorithm is a **BED-style baseline**, not automatically a reproduction of BED-LLM. EIG-only policies may gather accurate information yet miss the transaction deadline; test a utility-aware acquisition rule separately, e.g.

\[
\operatorname{score}(a)=\mathbb E[\text{future negotiated surplus}\mid a]
+\lambda\operatorname{EIG}(a)-c(a),
\]

and vary `λ`. This is a **proposed experimental objective**, not an attributed BED-LLM formula. A policy may seek information through *offers* rather than direct questions.

### 5.1 The distinctive two-agent experiment: cross-play

Evaluate the **ordered 3×3 buyer–seller matrix**, starting with `Jev`, `LLM`, and `BED-LLM adaptation`:

| Buyer ↓ / Seller → | Jev | LLM | BED-LLM adaptation |
|---|---|---|---|
| Jev | J/J | J/L | J/BED |
| LLM | L/J | L/L | L/BED |
| BED-LLM adaptation | BED/J | BED/L | BED/BED |

Run every pair against the **same sampled private-type pairs** and counterbalance the side that starts. Do not infer symmetric performance: `Jev buyer / BED seller` is not the same condition as `BED buyer / Jev seller`. A policy's behavior can change the other's information opportunities. Report **compatibility and cross-play performance**, not just individual-model leaderboards.

Additional controlled comparisons:

1. **One learner versus fixed rule-based responder:** establishes whether belief learning works with a known stable observation model.
2. **Both adaptive agents:** tests reciprocal learning, coordination, and strategic effects.
3. **Proposal-only versus explicit question actions:** tests whether preference discovery through offers differs from direct interrogation. Treat direct questions as a separate protocol, not a hidden privilege.
4. **Truthful response oracle versus freely strategic responses:** isolates information acquisition from opponent-policy inference.
5. **Hard versus easy reservation threshold:** tests whether agents identify when agreement is impossible.
6. **Candidate-order randomization:** checks sensitivity of Jev and LLM action selection to list order.

## 6. Experimental design and reporting

### 6.1 Pilot first

- **Profiles:** all `6 × 6 = 36` ordered pairs of preference rankings.
- **Initial regimes:** `r = 0.40` and `r = 0.55`.
- **Role/turn control:** both starting sides; roles remain buyer/seller according to utility direction.
- **Pilot seeds:** 3 independent runs per combination, then increase to at least 20 for stochastic policies as budgets permit.
- **Turn budget:** 6 total proposals per episode, plus clearly accounted response turns.
- **Budget matching:** report both equal-action-budget and cost-matched comparisons; match underlying LLM used by LLM-only and BED-LLM where possible.
- **Paired design:** reuse preference pairs, start order, and random seeds across policies when applicable.
- **Uncertainty:** paired bootstrap confidence intervals over episode clusters; stratify by type pair and feasibility class, with caution about repeated model samples.

### 6.2 Required leaderboard columns

| Dimension | Metrics (report buyer and seller separately where relevant) |
|---|---|
| Feasible-episode outcome | Agreement rate when `F ≠ ∅`; valid agreement rate; timeout/walk-away rate |
| Infeasible-episode outcome | Correct-no-deal rate when `F = ∅`; invalid-acceptance attempts |
| Individual payoff | `U_B`, `U_S`, `G_B`, `G_S`; distribution, not just the mean |
| Joint deal quality | Pareto-frontier rate, Nash surplus ratio `Q_N`, `min(G_B,G_S)` |
| Epistemic quality | True-type log loss, log-score improvement, Brier score, calibration, entropy over turns — for **both directions** |
| Efficiency | Proposals/decisions to agreement, model calls, token/cost usage, wall-clock latency |
| Cross-play | Results conditional on opponent type/policy and starting side |

Always show a **scatterplot of buyer versus seller realized gains** and the attainable frontier for representative type pairs. This reveals whether a model is consistently extracting surplus from one side rather than improving mutually feasible outcomes.

### 6.3 Minimum sanity checks

- Exhaustively enumerate all 27 deals and verify the feasible set, Pareto frontier, `U_i*`, and Nash optimum for every hidden type pair.
- Verify that no player's prompt or observations expose the opponent's rank or the evaluator's answer keys.
- Verify complementarity of per-attribute satisfaction and that scores lie in `[0,1]`.
- Verify correct handling of no-deal, rejected IR offers, proposals at the deadline, and invalid actions.
- Check role swapping and candidate-order permutation without changing ground-truth payoff tables.
- Compare posterior reports with uniform baselines; check Brier and log scores on known diagnostic examples.
- Validate that any exact posterior calculation uses a specified response model rather than assuming strategic rejection is truthful.
- Freeze protocol and evaluation code before inspecting final model comparisons.

## 7. Recommended implementation order

**Milestone 1 — exact game engine.** Implement immutable type assignment, 27-deal catalog, full truth-table scoring, legal action/turn state machine, and evaluation tests. Confirm the `36/36` and `30/36` feasibility counts.

**Milestone 2 — deterministic agents and controlled inference.** Run random/myopic/oracle baselines and a known-rule response oracle. Implement six-category belief logging and proper scoring. Verify that useful offers can genuinely distinguish otherwise plausible opponent types; do not assume all types are identifiable within six offers.

**Milestone 3 — one adaptive agent.** Compare Jev-only, constrained LLM-only, and BED-style information acquisition against an identical fixed responder. Hold the responder and seed constant.

**Milestone 4 — bilateral cross-play.** Run the ordered 3×3 Jev/LLM/BED-LLM matchups, including homogeneous pairings. Measure utility, truthful learning, Pareto efficiency, and interaction effects. This is the experiment's most distinctive extension.

**Milestone 5 — robustness.** Add varying outside options, hidden urgency/patience bins, noise in controlled responses, altered candidate order, and possibly a free-language negotiation condition. Every added private dimension expands the type space and must be versioned independently.

## 8. Hypotheses and falsification tests

- **H1 (information acquisition):** EIG-directed policies improve proper belief scores about their counterpart compared with myopic and random policies. **Falsified** if log/Brier improvement disappears or comes only from overconfidence.
- **H2 (instrumental usefulness):** Better opponent inference increases feasible agreement or joint surplus at a fixed turn/cost budget. **Falsified** if agents learn more but fail to close comparable deals.
- **H3 (bilateral adaptation):** Two adaptive agents can achieve higher Pareto-efficient agreement rates than one adaptive agent paired with a fixed policy under selected regimes. **Falsified** if joint adaptation destabilizes coordination or creates poor strategic signaling.
- **H4 (representation versus algorithm):** An explicit finite belief state can improve Jev and standard LLM performance even without full BED-LLM. **Falsified** if the benefit is absent after matching budgets.
- **H5 (typed decisions):** Jev can be competitive in a finite action game without generating text. **Falsified** if structured-action LLMs substantially outperform it on buyer/seller utility and efficiency at comparable budgets.

These are **questions to test**, not findings.

## 9. Interpretation limits

1. **Negotiation success is not information gain.** The scores must be published separately.
2. **Two agents make the observation process endogenous.** Never interpret a rejection as a clean noisy-oracle answer unless the responder was explicitly programmed that way.
3. **Bilateral IG is not `I(θ_B; θ_S)`.** The two private rankings may remain statistically independent even as both participants learn the other's ranking.
4. **Finite actions are not a neutral accommodation only for Jev.** Require the same constrained-interface comparator for ordinary LLMs; free-text negotiation belongs in a later, separately labeled condition.
5. **Cooperative does not mean identical interests.** Each agent protects its reservation value and seeks a good individual outcome; the joint evaluator records Pareto efficiency and fairness.
6. **A BED-LLM adaptation is not automatically the published implementation.** Track which components are faithfully reproduced and what has been modified to handle strategic opponents and offers.
7. **This is not the original LMRL-Gym seller-price benchmark.** Direct numerical comparison to its reported seller rewards would be misleading.

## 10. Suggested data contract

For each episode, retain:

```json
{
  "game_version": "car-dealer-discrete-v0.1",
  "seed": 42,
  "reservation_regime": "hard",
  "private_truth_evaluator_only": {
    "buyer_rank": ["price", "warranty", "delivery"],
    "seller_rank": ["delivery", "price", "warranty"]
  },
  "policies": {"buyer": "jev", "seller": "bed-llm-adapted"},
  "starting_side": "buyer",
  "turns": [
    {"actor": "buyer", "action": "OFFER", "deal_id": "P1-W2-D0"},
    {"actor": "seller", "action": "ACCEPT"}
  ],
  "belief_reports": {
    "buyer_about_seller": [],
    "seller_about_buyer": []
  },
  "result": {
    "agreement": true,
    "accepted_deal_id": "P1-W2-D0",
    "buyer_utility": 0.576923,
    "seller_utility": 0.807692,
    "individually_rational": true
  }
}
```

Interpret each deal ID according to a **versioned** canonical index (`P0=low, P1=medium, P2=high`; `W0=short, W1=medium, W2=long`; `D0=slow, D1=medium, D2=fast`). The example utilities above correspond to `P1-W2-D0`. Keep evaluator-only fields inaccessible to the agents. In a real implementation, store private truth in a separate object from policy-visible observations to reduce leakage risk.

## 11. Sources and provenance

- **LMRL-Gym, Car Dealer:** [official repository and task summary](https://github.com/abdulhaim/LMRL-Gym). The finite mutual-preference protocol and scores here are *new proposed modifications*.
- **BED-LLM:** Choudhury et al., *BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design*, [arXiv:2508.21184](https://arxiv.org/abs/2508.21184). It motivates EIG-based adaptive querying; bilateral strategic negotiation is a proposed extension, not a reported BED-LLM result.
- **Jev:** [TypeSafe Jev documentation](https://docs.typesafe.ai/), particularly the `Choice`, `Score`, and `Noul` typed decision interfaces. No empirical Jev Car Dealer result is assumed.
- **Design history:** Earlier exploratory notes called for private highest/middle/lowest rankings, joint preference discovery, a cooperative proposal-only game, and explicit belief/entropy trajectories. This document converts those ideas into a scored, falsifiable benchmark.

---

**Decision to preserve:** Score **both agents** and the **agreement**, not merely the sale price or information gain. The most informative study is the *cross-play interaction*: how one learner changes the other's ability to discover preferences and reach a good deal.
