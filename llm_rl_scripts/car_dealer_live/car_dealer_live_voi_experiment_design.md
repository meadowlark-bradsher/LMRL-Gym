# Car-Dealer: The Decision Value of Information in Live Negotiation

**Experiment design:** `CD-VOI-1` (primary) and `CD-MIS-1` (robustness)  
**Status:** Proposed / not executed  
**Date:** 2026-10-27  
**Source baseline:** `FINDINGS.md`, §§3.4, 3.6, A.4–A.5  
**Research question:** When is deliberate information acquisition worth its opportunity cost in bilateral negotiation, and when does an incorrect opponent model make learning worse than retaining the prior?

## 0. Executive design

This is an extension of the completed Car-Dealer studies, **not a replication of their original comparisons**. The completed bilateral game shows that an acceptance-probability forward model can unlock agreement even if its prior is never updated, while posterior updating improves agreement quality. The completed binned game shows that information gain and optimal decision value are distinct, especially when the horizon binds. The missing experiment connects **deliberate probing**, **live counteroffers**, **finite bargaining time**, and **belief-model errors**.

Two studies share one environment, logging format, and evaluation harness:

1. **`CD-VOI-1`:** Compare a prior-only forward model (A), passive posterior learning (B), and active information seeking (C), first against a history-responsive counterparty and then with two learning negotiators. An exact/near-exact planner in a small state space anchors the decision-value comparison.
2. **`CD-MIS-1`:** Hold the learner and its assumed likelihood fixed while systematically perturbing the *true* counterparty behavior. Determine where updates cease to help, where they damage negotiation, and whether robust filtering recovers value.

**Primary empirical contrast:** A planning-matched, **budgeted** offer-only policy versus the *same planning algorithm* with explicit probing actions: `C_plan − B_plan`. The exact `C_oracle` versus restricted exact oracle comparison identifies the theoretical opportunity, but is not itself evidence that an implementable agent can exploit it. A secondary, architecture-matched contrast tests whether Jev or a general LLM can use the same information without an external planner.

**Key distinction:** Information gain (bits) is not bargaining value (utility). An informative offer may be accepted and terminate the episode; a question may reveal a hidden preference while forfeiting a valuable offer opportunity. Both are part of the outcome distribution, not nuisances to strip away.

## 1. Established baseline; do not claim it as a new result

The following are reported in the existing `FINDINGS.md`, not produced by this design:

| Completed result | Implication for the next experiment |
|---|---|
| Unaided Jev/Jev: 0/12 agreements; 0/130 initial offers cleared the recipient threshold | A suitable forward model matters more initially than sophisticated inference. |
| Acceptance-probability aid: 24/24 agreements across aided sampled conditions | A prior-averaged acceptance model can alter offer selection decisively. |
| Prior-only filter: 12/12 agreements in the initial control | Learning the *specific* opponent is not required for the initial agreement outcome. |
| Bayesian versus prior-only: normalized Nash improvement +0.158 [0.054, 0.263], with +0.58 proposals | Learning appears to buy agreement quality at a cost in bargaining rounds. |
| Binned, 2-round: exact 0.7756; Jev+EIG 0.7248; greedy EIG 0.7171; myopic 0.7167 | EIG maximization is not equivalent to sequential decision optimization. |
| Binned, 6-round: Jev matched exact in a regime where exact never asked | High utility does not necessarily demonstrate information-acquisition competence. |

**Scope of evidence:** The bilateral study had six private preference rankings per role and 27 possible deals. Its type-pair census is finite (36 pairs). The binned problem had a *scripted, non-countering* opponent, so its exact POMDP does not yet certify decision quality in a genuinely interactive game. The old Bayesian filter was sometimes confidently wrong. These limits motivate the experiments below.

## 2. Hypotheses and falsification criteria

| ID | Prediction | What would disconfirm it? |
|---|---|---|
| **VOI-H1** | In at least one predeclared binding-horizon/low-cost condition, the active planner yields a **material positive improvement** in the focal negotiator's *net* normalized gain over the identical passive planner. | The paired improvement is indistinguishable from zero, or the lower confidence bound does not clear the prespecified material-effect threshold. |
| **VOI-H2** | Optimal active probing becomes less attractive as its direct cost rises, and can exhibit a **non-monotone** dependence on horizon: too little time prevents exploitation, while very long horizons may make offers sufficient. | No systematic relation between oracle ask advantage and these factors. |
| **VOI-H3** | A policy maximizing raw EIG sometimes chooses a lower-value action than an expected-return planner, including when an informative offer has termination value. | No EIG-versus-decision-value discordances occur in the enumerated test states. |
| **VOI-H4** | EIG annotation alone changes question frequency more reliably than it reduces decision regret for a Jev selector. | EIG annotation consistently closes the planner regret gap, beyond matched presentation and position controls. |
| **MIS-H1** | Under sufficiently misspecified behavioral likelihoods, posterior updates can make a policy worse than its prior-only version. | No negative prior-versus-update contrast anywhere in the prespecified mismatch grid. |
| **MIS-H2** | Tempered/mixture likelihoods mitigate the worst miscalibration and some negative learning effects, at a possible cost under correct specification. | Robust methods fail to improve worst-case utility or calibration while preserving acceptable matched-model performance. |
| **BIL-H1** | When both sides learn, the value of an inquiry depends on the information it reveals to the *other* party as well as what it reveals to the asker. | Bilateral inquiry effects are fully predicted by the unilateral model and show no opponent-learning interaction. |

Only `VOI-H1` and the prespecified `MIS-H1` contrast are proposed as confirmatory at the outset. The others are mechanism checks or follow-on preregistrations unless a multiple-testing procedure is locked before data collection. A **zero** effect is scientifically informative: an active-information wrapper is not justified merely because it is available.

## 3. Environment: continuity with the existing Car-Dealer game

### 3.1 State and utilities

- Retain the **27 deals**, **six private rankings per role**, and **36 buyer–seller type pairs** from the current code. Do not silently change the deal-to-utility mapping.
- Retain an individually rational (IR) agreement requirement: `U_i(d, θ_i) >= r_i` for both parties. Baseline reservation regimes are **easy `r=0.40`** (36/36 pairs have an IR deal) and **hard `r=0.55`** (30/36 do).
- No deal gives each player the reservation payoff `r_i`; the *gain above outside option* is zero. An accepted deal transfers the exact existing utility of that deal to each side. Explicit inquiry costs, if enabled, are charged separately.
- Each actor knows its own ranking, the legal action menu, remaining budget, standing offer, and entire **public** transcript. It never sees the other actor's actual type, posterior, or private random draws.
- Counterbalance the opener (buyer first/seller first), and later counterbalance which seat receives the experimental policy.

### 3.2 Turns and actions

Use the prior bilateral game as the compatibility baseline: **PROPOSE** any legal deal, **ACCEPT** a standing offer, **REJECT**, and **WALK**. Add **ASK(q)** from the established nine binary TOP / ABOVE / BOTTOM questions. For the primary study:

1. A proposal or question spends **one proposal-equivalent opportunity** from a common episode budget. Therefore questions are not cost-free or extra turns.
2. An `ASK` elicits an immediate answer (under the designated response policy), is appended to the **public** transcript, and passes initiative to the next bargainer. The respondent observes which question was asked.
3. A proposal may be accepted, rejected, or countered; a counteroffer replaces the standing offer and itself consumes an opportunity. A rejection is evidence about the responder's type when the response model supports that inference.
4. An existing standing offer remains accept-able at the deadline; no late new proposal or question is allowed. Acceptance does not consume a proposal opportunity, consistent with the earlier game.
5. A question does not by itself cancel a standing offer; the transcript and standing-offer state must be included in planning. If this creates undesirable action combinations, revise **before** preregistration and version the rules.

**Budget conditions:** `H ∈ {2, 4, 6}` proposal-equivalent opportunities, with no additional free question turns. `H=6` is the legacy reference; `H=2` makes acquisition expensive in a way the prior binned study suggests will be informative. Label this new clock carefully: it is **not** numerically identical to binned `rounds + final offer`.

**Direct inquiry fee:** In addition to consuming an opportunity, set `c_q ∈ {0, 0.01, 0.03}` in each asker's *raw utility* units. These are design levels, not empirically justified thresholds; report them as such. Proposal and counteroffer fees default to zero, with a sensitivity analysis charging the same per-message fee for *all* outbound actions. An optional noisy-answer factor is introduced only in the relevant tier.

### 3.3 Two counterparty tiers

**Tier T1 — interactive but modelable (primary causal test).** The opponent is a **history-responsive scripted negotiator** that can accept, reject, propose counteroffers, and walk as a function of its own type, visible history, and budget. It may make conditional concessions. Its behavior-generating probabilities are known to the experimental oracle and held fixed across treatment arms. It is not a frozen threshold responder. Truthful, deterministic answers to explicit preference questions are provided by a rule-governed mediator in the clean primary condition. This is deliberately artificial but enables causal isolation and finite-state solution.

**Tier T2 — genuinely bilateral (extension).** Both parties maintain beliefs about the other's type and may ask, counter, withhold, or react to revealed preferences. Use finite-depth models (e.g., each agent models the other as a specified level-0 or level-1 policy) rather than claiming an exact solution to unrestricted recursive beliefs. Compare `(B,B)`, `(C,B)`, `(B,C)`, and `(C,C)` with seats and types counterbalanced. At this tier, a question is strategically observable and its response may be truthful, noisy, evasive, or strategically chosen according to a documented protocol.

**Do not pool T1 and T2 effects:** T1 measures decision value under a controlled, known process; T2 measures how well the mechanism transfers when the opponent also infers and acts strategically.

## 4. Decision-theoretic quantities

Let `θ_j` be the opponent's private type and `h_t` the public history. For negotiator `i`, the belief state is

\[
b^i_t(\theta_j)=P(\theta_j\mid h_t,\theta_i,\mathcal M_i).
\]

On observing `o` following action `a`, update with the **action- and history-conditional** likelihood:

\[
b^i_{t+1}(\theta_j)
=\frac{b^i_t(\theta_j)P_{\mathcal M_i}(o\mid \theta_j,h_t,a)}
{\sum_{\theta'}b^i_t(\theta')P_{\mathcal M_i}(o\mid\theta',h_t,a)}.
\]

Here `o` includes accept/reject/counter/walk, answer text, and terminal agreement. Critically, the focal party must **not update from its own emitted proposal alone**; information is obtained through the opponent's subsequent response. In T2, opponent proposals and questions are actions chosen under the opponent policy and carry likelihood information of their own.

**Expected information gain** (diagnostic, not the objective):

\[
\operatorname{EIG}(h,a)= H(b_t) - \mathbb E_{o\mid h,a}[H(b_{t+1})].
\]

For a terminated negotiation, posterior entropy may still be evaluated for diagnosis, but **there are no future decisions in which to exploit that information**. Report information obtained from offers as well as questions. Use survival/termination-aware statistics to avoid measuring EIG only for histories that happened not to end.

**Decision value** (objective for the oracle and planner):

\[
Q^{\pi}(h,a)=\mathbb E\!\left[ (U_i(d_{\rm terminal})-r_i)
-\sum_{s=t}^{T}c_i(a_s)\;\middle|\;h,a,\pi_i,\pi_{-i}\right].
\]

Set `U_i(d_terminal) − r_i = 0` on no deal. The reference Bellman recursion is `V(h)=max_{a∈A(h)}Q(h,a)` with a fully specified finite-horizon opponent policy and terminal rules. The state must include the standing offer, remaining budget, both players' relevant public actions, and the current posterior. Exact dynamic programming is required only for tractable T1 slices; use documented approximate rollouts elsewhere.

**Value of active inquiry (global policy comparison):**

\[
\Delta_{\rm active}^*(h)=V^*_{\rm ASK\ allowed}(h)-V^*_{\rm ASK\ forbidden}(h)\geq 0.
\]

Both values are recomputed from the same starting state using the same reward and transition model; `ASK forbidden` removes inquiry at **all future decisions**, not only the current move. To identify *local* ask-versus-offer choices, separately log

\[
A^*_{\rm ask}(h)=\max_{a\in A_{\rm ask}}Q^*_{\rm full}(h,a)
-\max_{a\in A_{\rm no\ ask}}Q^*_{\rm full}(h,a),
\]

which may be positive or negative (and is undefined when ASK is illegal). `A_no ask` retains accept, reject, counter, offer and walk when legal. At a matched state, an **optimal** policy with an expanded action set cannot perform worse than an optimal restricted policy. This is a **structural fact**, not an experimental discovery. The empirical questions are how large the gain can be, whether a practical budgeted planner or a selector realizes it, and whether belief-model errors create **negative realized** gains.

**Do not infer value from entropy alone.** For heuristic diagnosis only, compare

\[
\operatorname{score}_{\lambda}(a)=\mathbb E[\text{immediate surplus}\mid h,a]
+\lambda\operatorname{EIG}(h,a)-c(a).
\]

Choose `λ` on a separate calibration split, never on test outcomes. The primary causal claim must not rely on finding a convenient `λ` post hoc.

## 5. Experimental arms and what each isolates

| Arm | Opponent model | Belief updates from live history | Deliberate ASK | Decision policy | Role |
|---|---|---|---|---|---|
| `J0` | None | No | No | Original unaided Jev | Historical anchor only |
| `A_prior` | Prior acceptance model | **No** | No | Same proposer/selector interface as B | Forward-model baseline |
| `B_passive` | Posterior acceptance model | **Yes** | No | Same proposer/selector as A | Passive-learning effect, `B−A` |
| `B_plan` | Posterior + full action outcome model | **Yes** | No | Belief-space planner, **ASK disabled** | Planning-matched primary control |
| `C_plan` | Identical model to `B_plan` | **Yes** | **Yes** | Same belief-space planner, ASK enabled | Primary causal contrast, `C_plan−B_plan` |
| `C_eig` | Posterior + EIG annotations | **Yes** | **Yes** | Jev/LLM selector, matched menu framing | EIG-number integration |
| `C_greedy` | Posterior + EIG | **Yes** | **Yes** | Maximum-EIG legal action, with explicit tie-break rule | Information-only heuristic |
| `C_bonus` | Posterior + estimated surplus/EIG | **Yes** | **Yes** | Tuned `λ` heuristic | Practical inexpensive heuristic |
| `C_oracle` | True generative model | **Yes** | **Yes** | Exact finite-state dynamic program where tractable | Upper bound / regret reference |
| `P_placebo` | Constant accept annotation | No | No | Same interface as A/B | Prompt/salience check |

**Important fairness constraints:** `B_plan` and `C_plan` are **practical, budgeted belief-space planners**, not the exact oracle; they differ **only by the availability of ASK**. They have the same environment model, planning algorithm/depth, reward function, deal menu, observation rules, and compute allocation. If ASK increases the branching factor and forces a shorter search, equalize achieved search accuracy or separately report compute-matched and accuracy-matched results. `B_passive` versus `C_eig` is **not** a clean acquisition-value estimate because its menus and selector behavior also differ.

For Jev/LLM arms, randomize option order and letter assignments, keep formatting and token budgets matched, and include a non-informative annotation placebo. Do not present the posterior itself unless that is a separately named treatment.

## 6. CD-VOI-1: primary protocol

### Phase 0 — implementation and oracle readiness (no research claims)

1. Load the existing 27-deal utility tables and six private rankings unchanged; reproduce the reported legacy metrics with the old environment adapter.
2. Add public `ASK(q)` actions and immediate, typed answer events; assert that the same budget accounting is applied to all policies.
3. Add a fixed, history-dependent opponent policy with accept, counter, reject, and walk paths; expose its true transition kernel only to the simulator/oracle, never to the agent.
4. Validate probabilities sum to one, terminal behavior and reservation payoffs are correct, every action has a valid counterparty response, and the posterior stays normalized (including noisy likelihoods).
5. For a tiny finite horizon, compare exact DP to exhaustive tree enumeration and Monte Carlo within numerical tolerance. **Required gate:** oracle-value discrepancy < `1e-6` in deterministic/exhaustive cases; Monte Carlo checks use prespecified standard-error bounds.
6. Audit that a candidate's own proposal does not update beliefs before a response, and that an accepted informative proposal ends the episode.
7. Inspect the number of states with `Q*(ASK) > Q*(best non-ASK)` at each `(H,c_q)`; report **all** cells. If this is zero everywhere, do not claim to have tested meaningful inquiry value. Amend the model *before* a locked confirmatory run, with changes recorded.

### Phase 1 — controlled, reactive-counterparty T1 (confirmatory)

**Sampling grid:** complete six-by-six type-pair census, two opening seats, two assignments of which seat is focal, and `K=5` stochastic replications per cell, for **720 episodes per arm per prespecified `(H,c_q,r)` condition**. Use the same type-pair and exogenous-randomness seeds across arms where possible; a common random-number stream must not inadvertently change with a different number of prior actions.

**Primary regime:** use an independently chosen, documented `H=2` or `H=4` condition with `r=0.40`, `c_q=0` and a pre-pilot-tested transition model exhibiting genuine ask/offer tradeoffs. Lock this one condition before the confirmatory evaluation. Evaluate the complete secondary `(H,c_q,r)` grid without selectively reporting the best cells. `r=0.55` includes infeasible pairs, which must be identified in analysis.

**Primary estimand:** role-balanced paired difference in **focal normalized net gain** between `C_plan` and `B_plan`:

\[
Y_i=\frac{U_i(d_{\rm terminal})-r_i-\sum c_i}{U_i^{\max}-r_i},
\quad \Delta_{\rm VOI}=\mathbb E[Y_i(C_{\rm plan})-Y_i(B_{\rm plan})],
\]

where `U_i^max` is the maximum over the 27 deals for that type as in the legacy normalization. Require positive denominator; flag any exception. Values can be **negative** when a negotiator pays to inquire but walks away. Report *gross* gain and action costs separately too.

**Materiality threshold:** pre-register `δ = +0.02` normalized-gain units as a provisional *design* threshold, not a discovered scientific constant. Call the result (i) positive if the 95% CI is entirely above zero; (ii) materially positive if its lower bound also exceeds `δ`; otherwise report unresolved/no material benefit. Replace `δ` before running if a domain-relevant economic margin can be justified, and record why.

**Secondary outcomes:** agreement rate; both parties' normalized gain `G_i`; normalized Nash quality `Q_N`; minimum gain; Pareto-efficient agreement fraction; number of offers/counteroffers/questions; direct cost; time-to-deal; decision regret against the oracle; EIG acquired by action type; and belief quality. For Nash normalization, use the established definition `(U_B-r_B)(U_S-r_S)` divided by its maximum over feasible deals, and define `Q_N=0` for no-deal *in feasible pairs*. When the maximizing feasible Nash product is **zero** (including pairs with no IR deal), report feasibility separately and mark the normalized Nash metric undefined rather than dividing by zero. Show gross and net quantities distinctly rather than quietly changing the legacy metric.

**Oracle decomposition:** On each reached decision state, label `(i)` EIG of ASK vs OFFER, `(ii)` `Q*` of ASK vs OFFER, `(iii)` whether ASK is optimal, `(iv)` whether the chosen action is optimal, `(v)` terminal-event probability, and `(vi)` expected remaining opportunities. Classify especially useful discordances:

- **High EIG, low decision value**: useful facts arrive too late or at excessive cost.
- **Moderate EIG, high decision value**: a question resolves a pivotal acceptance uncertainty.
- **Information-bearing offer**: rejection teaches; acceptance produces payoff and terminates.
- **Uninformative but profitable offer**: immediate surplus beats further inquiry.

### Phase 2 — selectors and practical heuristics (secondary)

Against the same T1 seeds and episode protocol, evaluate `A_prior`, `B_passive`, `C_eig`, `C_greedy`, and `C_bonus`, plus Jev or general-LLM controls on identical action menus. Preserve the exact planner's action-values as a *grader* even if not shown to the selector.

For each observed state report (a) exact/estimated EIG rank, (b) exact/estimated `Q*` rank, (c) per-decision regret, (d) question propensity, (e) first-position bias, and (f) calibration if a confidence score is exposed. **Question count alone is not an improvement criterion.**

### Phase 3 — adaptive bilateral T2 (transfer test)

Run the four crossed assignments `(B,B)`, `(C,B)`, `(B,C)`, `(C,C)`, with buyer/seller and opener counterbalanced. Each agent holds a posterior over the other actor's preferences; observed counteroffers and questions may be evidence. Use a bounded model of opponent reasoning, publish its assumptions, and avoid calling its decisions exact global equilibrium solutions.

For each combination report **both** negotiators' net utility, joint welfare, normalized Nash, agreement frequency, and the difference between unilateral and mutual access to probing. Compare the focal effect of `C` when facing `B` versus `C`. Because questioning may reveal the asker's strategic interests, also log the *recipient's* posterior change after observing a question. Analyze all episodes, including early terminations.

## 7. CD-MIS-1: learning with the wrong opponent model

### 7.1 What varies and what stays fixed

Fix the filtering agent's prior, its assumed six-type softmax offer-likelihood model (legacy reference `β=8`, non-acceptance floor `ε=0.05`), action set, and decision policy. Change **only** the opponent's true response process along prespecified axes, then repeat with robust filters. Because behavior changes the histories visited, use paired initial types/seeds, but assess effects on full episodes rather than requiring identical counteroffer traces.

| True opponent process | Perturbation | Failure mechanism |
|---|---|---|
| Matched reference | Generate from the assumed response kernel | Sanity check; posterior should be calibrated on average. |
| Deterministic threshold | Accept strictly at reservation; rigid proposal rule | Assumed response noise creates distorted likelihoods. |
| Stochastic acceptance | Additional temperature/noise relative to assumed kernel | Observed acceptance/rejection is misinterpreted. |
| Anchoring/concession | Offers depend on initial anchor and remaining time | Nonstationary behavior is wrongly attributed to type. |
| Strategic withholding | Suppress otherwise acceptable early offers; stall | False rejection-as-type evidence. |
| History-contingent countering | Counter selection depends on the agent's previous acts | Conditional dependence omitted from filter. |
| Question evasiveness | Noisy or strategically selected responses | Explicit inquiries become misleading. |

The labels above are **experimental mechanisms**, not claims that real negotiators follow these rules. Choose and freeze the parameters before the confirmatory run; don't find the most pathological mismatch and present it as a typical result.

### 7.2 Robustness arms and contrasts

- `A_prior`: prior-only, no updates.
- `B_naive`: standard Bayesian posterior under the **incorrect assumed** kernel.
- `B_tempered`: likelihood tempering, `b' ∝ b · p(o|θ,h,a)^τ`, with `τ ∈ {0.5, 0.75, 1}`; `τ=1` recovers naïve updating and `τ=0` would recover prior-only behavior.
- `B_mixture`: outlier-robust mixture `(1−η)p_model + η p_background` with preselected `η` levels; specify `p_background` for every observation category.
- `C_naive`, `C_robust`: active-information versions that also select questions under naïve or robust beliefs.
- `O_true`: oracle that knows the true response mechanism (upper reference, never available to a deployed agent).

**Confirmatory mismatch contrast:** Within a **preselected history-dependent counteroffer or withholding condition**, measure `B_naive − A_prior` in focal net normalized gain. The negative-learning hypothesis is supported only if the paired 95% CI lies below zero. Report the entire mismatch grid and the matched-model cost of robustness.

**Diagnostics:** posterior probability of the true type and its log-loss (bits), Brier score for predicted acceptance, predictive log-loss for *all* observed response categories, posterior entropy, posterior concentration on an incorrect type, proposal quality, and decision regret. Separate **belief error** from **decision loss**: a confidently wrong posterior is important only insofar as it changes downstream choices or their reliability.

## 8. Analysis plan and statistical validity

- **Unit of dependence:** the **36 private-type pairs** form the finite census and the natural outer cluster. Both opener orders, focal-seat assignments, and repeated stochastic rollouts within a type pair are correlated; never treat individual turns as independent observations.
- **Paired comparisons:** pair treatment and control on type pair, opener, focal seat, and stochastic-replication index. Use paired cluster bootstrap over type pairs (e.g. 5,000 resamples) and report the estimated difference plus percentile 95% CI. If seeds are not meaningfully paired after divergent trajectories, preserve baseline paired initial conditions and estimate the episode-level difference.
- **Finite-census limitation:** repeating stochastic rollouts reduces Monte Carlo uncertainty but **cannot create new independent private-type configurations**. Report both the exact 36-pair finite-population mean (where possible) and uncertainty over stochastic rollouts. Treat type-pair resampling as a sensitivity analysis for generalization beyond this closed world, **not** as proof of an unlimited type population.
- **Pre-registration:** lock primary environment, policy versions, true-response parameters, action costs, one confirmatory VOI regime, one confirmatory mismatch regime, estimands, materiality criterion, and random seeds *before* evaluating the primary treatment arms. Pilots used to ensure the VOI tradeoff is nondegenerate must be segregated from confirmatory runs.
- **Multiplicity:** one primary VOI comparison and one primary misspecification comparison; all other comparisons clearly marked exploratory or adjusted as an explicitly pre-registered family.
- **Model/version control:** pin Jev/LLM model IDs, prompting, candidate menu generation, stochasticity, question-bank IDs, solver tolerance, and code commit. Repeat at least one held-out replicate with a fresh model call/sample to check reproducibility.
- **Checks against misleading selection:** report disagreement by starter/responder and buyer/seller separately; include unsuccessful episodes in aggregates; stratify by whether an IR deal exists; and show agreement-quality metrics both unconditional and conditional on agreement (conditional estimates are selected).

## 9. Minimal run schema, artifacts, and implementation sequence

Each episode should record at least:

```json
{
  "design": "CD-VOI-1",
  "env_version": "live-car-dealer-v1",
  "tier": "T1",
  "condition": {"horizon": 4, "r": 0.40, "ask_cost": 0.01},
  "true_types": {"buyer": "internal_id", "seller": "internal_id"},
  "role_assignment": {"focal": "buyer", "opener": "seller"},
  "arms": {"focal": "C_plan", "opponent": "reactive_script_v1"},
  "seeds": {"instance": 1001, "environment": 2001, "selector": 3001},
  "turns": [
    {
      "actor": "buyer",
      "action": {"kind": "ASK", "question_id": "q3"},
      "observation": {"kind": "ANSWER", "value": true},
      "belief_before": [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1665],
      "belief_after": [0.0, 0.0, 0.3333, 0.3333, 0.3334, 0.0],
      "eig_bits": null,
      "q_estimate": null,
      "q_oracle": null,
      "cost": 0.01,
      "remaining_opportunities": 3
    }
  ],
  "result": {"deal_id": null, "agreed": false, "gross_utility": {}, "net_utility": {}}
}
```

All example numbers/IDs in the JSON above are **schema illustrations**, not real episode data; the posterior vector should be stored in lossless numerical form in actual logs, and the logged true type must not be accessible to the agent. Include an explicit mechanism for reproducible replay of exogenous response randomness despite variable action histories (e.g., draw keys indexed by episode, actor, opportunity, response category, and action identity rather than simple position in a shared RNG stream).

**Suggested files/outputs:**

- `design/CD_VOI_1.md`, `design/CD_MIS_1.md`, `design/PREREGISTRATION.md`
- `env/live_car_dealer.py`, `env/question_bank.py`, `env/opponent_policies.py`
- `belief/finite_filter.py`, `planning/exact_t1.py`, `planning/rollout_t2.py`
- `policies/prior.py`, `policies/passive.py`, `policies/active.py`, `policies/jev_adapter.py`
- `tests/test_ir_and_deadline.py`, `tests/test_bayes_consistency.py`, `tests/test_oracle.py`, `tests/test_no_leakage.py`, `tests/test_action_costs.py`
- `results/<run_id>/episodes.jsonl`, `manifest.json`, `summary.json`, `analysis.md`, `figures/`

**Recommended build order:** (1) env compatibility; (2) T1 response policies; (3) ASK/question bank and action-budget tests; (4) posterior likelihoods; (5) exact planner and oracle checks; (6) A/B/C scripted policies; (7) preregister and run T1; (8) add Jev/LLM adapters; (9) run mismatch sweep; (10) introduce T2 bilateral dynamics.

## 10. Minimum success criteria and interpretation matrix

| Observed outcome | Defensible conclusion | Conclusion **not** justified |
|---|---|---|
| `C_plan > B_plan` net in binding states | Deliberate acquisition has exploitable **decision value** here. | EIG itself should be maximized; every negotiation benefits. |
| `C_plan = B_plan` with no oracle-optimal ASKs | Active questioning is unnecessary under these constraints. | Information is useless; offer/rejection learning may still matter. |
| Oracle gain positive, Jev/LLM `C_eig` regret high | The decision opportunity exists, but the selector does not realize it reliably. | The selector cannot learn any relevant information. |
| `B_passive > A_prior`, but `C_plan = B_plan` | Incidental learning is valuable, deliberate questioning adds little. | Bayesian inference and active probing are interchangeable. |
| `B_naive < A_prior` under mismatch | Updating a wrong behavioral model can be harmful. | Bayesian inference is generally harmful. |
| Robust filter reduces downside but loses matched-model value | Robustness imposes a measurable bias/variance or caution tradeoff. | The robust filter dominates universally. |
| `(C,C)` differs from `(C,B)` | Reciprocal learning changes inquiry incentives. | A general equilibrium is established without a specified game-theoretic solution. |

## 11. What this would contribute to the wider harness program

The scientific unit is a **decision-relevant belief intervention**, not a particular model provider or prompt. The harness makes three separations measurable:

1. **Forward-model availability:** Can the agent anticipate acceptable deals from public structure and a prior?
2. **Passive epistemic updating:** Can it revise beliefs about a specific counterpart from negotiated evidence?
3. **Active sequential control:** Can it decide when information is worth acquiring, considering cost, bargaining leverage, and the possibility of immediate termination?

`CD-MIS-1` adds a fourth: **belief-model reliability**. A robust harness should not only maintain a posterior; it should know when posterior-driven action is worth more than a prior-based fallback.

The intended deliverable is a set of **state-level, falsifiable decision certificates**: posterior, assumed likelihood, EIG, offer acceptance value, acquisition cost, expected return, and realized decision regret. These measurements can be compared across Jev, normal LLMs, BED-style wrappers, and exact planners while keeping the negotiation environment fixed.

**Bottom line:** The next experiment is not “does asking questions increase information?” It is **“does a negotiator know when it is worth paying for information, and does that judgment survive a wrong model of the opponent?”**
