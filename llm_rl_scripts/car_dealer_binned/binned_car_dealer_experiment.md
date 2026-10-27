# Experiment: Binned Car Dealer — Bilateral Information Gain

**Objective:** Determine whether an agent can select information-gathering actions that improve its negotiation outcome, and whether its action-selection uncertainty predicts the quality of those decisions.

I suggest starting with a **small, exactly solvable negotiation game** rather than immediately reproducing the full LMRL-Gym Car Dealer environment.

The first experiment should establish whether the information-theoretic measurements work. We can introduce natural-language negotiation afterward.

## 1. Environment

Two agents negotiate the sale of a car.

- **Buyer:** Wants a low price and favorable contract terms.
- **Seller:** Wants a high price and favorable contract terms.
- **Private information:** Each agent has a hidden preference type.
- **Actions:** Each agent chooses from four labeled options, `{A, B, C, D}`.
- **Horizon:** Six rounds, followed by a final offer.

Each agent knows its own preferences but must infer the opponent's.

### Private preference types

For the initial experiment, each player has one of four possible types.

| Type | Buyer preference | Seller preference |
|---|---|---|
| 0 | Lowest price | Highest price |
| 1 | Long warranty | Low warranty liability |
| 2 | Low financing rate | Financing revenue |
| 3 | Fast delivery | Low delivery cost |

Types are sampled independently and uniformly:

\[
Z_B,Z_S\sim\operatorname{Uniform}\{0,1,2,3\}.
\]

Each agent begins with two bits of uncertainty about its opponent:

\[
H(Z_{\text{opponent}})=2.
\]

**Important:** These types are deliberately simple. The production version should give every type a complete utility function over price, warranty, financing, and delivery.

That prevents the benchmark from accidentally rewarding agents for identifying labels rather than negotiating well.

---

## 2. Multiple-choice actions

Each turn, an agent receives four candidate actions.

For example, the buyer might receive:

| Option | Action | Expected IG |
|---|---|---:|
| A | Ask whether the seller prioritizes price | 1.000 bits |
| B | Ask whether warranty or financing matters more | 0.811 bits |
| C | Ask about a weakly informative delivery preference | 0.544 bits |
| D | Make an offer without gathering information | 0.000 bits |

These are **illustrative values**, not yet validated against a particular oracle.

For the actual experiment, every question will have an explicitly defined response distribution conditional on the seller's type.

That allows exact computation of expected information gain.

The candidate options should be randomized in presentation order to test positional bias.

### A crucial experimental control

Initially, we should make the information-gathering questions truthful and their response probabilities known.

Otherwise, we introduce two simultaneous problems:

1. Selecting informative questions.
2. Inferring whether the opponent is answering strategically.

Both are interesting, but separating them makes the first experiment interpretable.

---

## 3. The information-theoretic measurement

Let \(b_t^B\) represent the buyer's belief about the seller's type.

For a candidate question \(q\), define:

\[
\operatorname{EIG}_B(q)
=
H(b_t^B)
-
\mathbb E_{o\mid q,b_t^B}
\left[H(b_{t+1}^B)\right].
\]

The seller has the corresponding quantity:

\[
\operatorname{EIG}_S(q).
\]

We can calculate both exactly because the hidden types and response models are finite.

For every decision, we record the expected information gain of all four options.

We can then define **EIG regret**:

\[
R_t^{IG}
=
\max_{q\in Q_t}\operatorname{EIG}(q)
-
\operatorname{EIG}(q_t).
\]

This tells us how much information the agent sacrificed by choosing its particular action.

However, minimizing EIG regret is not necessarily optimal negotiation behavior.

An agent might rationally choose a low-information offer because the expected utility of closing the deal exceeds the value of further investigation.

That distinction is central to the experiment.

---

## 4. Separate information quality from negotiation quality

I would score agents along three dimensions.

| Metric | Question answered |
|---|---|
| Information efficiency | Did the agent choose informative questions? |
| Decision quality | Did the agent choose actions with high expected utility? |
| Negotiation outcome | Did the resulting agreement benefit the agent? |

The key measurement is **decision regret**:

\[
R_t^U
=
\max_{a\in A_t}Q^*(b_t,a)
-
Q^*(b_t,a_t).
\]

Here \(Q^*(b_t,a)\) is the optimal expected continuation value of action \(a\), including subsequent negotiation opportunities.

For the small finite game, we can compute this using dynamic programming.

This creates an exact reference policy.

It also makes a distinction possible that our earlier discussion overlooked:

**An agent can maximize information gain while making a poor negotiation decision.**

That is not a benchmark failure. It is one of the phenomena we want to detect.

---

## 5. Bilateral information gain

Now we introduce the genuinely multi-agent component.

At each round, we track:

\[
H_t^B=H(Z_S\mid h_t^B)
\]

\[
H_t^S=H(Z_B\mid h_t^S).
\]

This produces two uncertainty trajectories.

We can examine four patterns:

| Buyer uncertainty | Seller uncertainty | Interpretation |
|---|---|---|
| Falling | Falling | Mutual information acquisition |
| Falling | Stable | Buyer has an information advantage |
| Stable | Falling | Seller has an information advantage |
| Stable | Stable | Little information acquisition |

We can also calculate the cumulative uncertainty exposure of each agent:

\[
C_B=\sum_{t=0}^{T}H_t^B
\]

\[
C_S=\sum_{t=0}^{T}H_t^S.
\]

The difference

\[
\Delta C=C_B-C_S
\]

provides a descriptive measure of information asymmetry over the episode.

It is not automatically a measure of strategic advantage: some private information may be much more valuable than other information.

That relationship is something we can test.

---

## 6. Model comparisons

This experiment can compare the three agent families you've been considering.

| Agent | Decision mechanism | What we test |
|---|---|---|
| Jev | Structured choice among A/B/C/D | Decision quality under minimal output |
| Conventional LLM | Prompted choice among A/B/C/D | Whether general reasoning produces good information policies |
| BED-LLM | Bayesian experimental-design-oriented selection | Whether explicit information-seeking improves decisions |
| Exact Bayesian policy | Dynamic programming | Optimal reference |
| Greedy EIG policy | Always chooses maximum EIG | Information-seeking baseline |
| Random policy | Uniform choice | Lower baseline |

All agents receive the same observations and candidate actions.

The distinction is how they choose.

One important implementation detail: a one-token output is sufficient to **select** an action, but measuring action-distribution entropy requires probabilities or repeated samples. We should not infer uncertainty from a single emitted letter.

---

## 7. Experimental conditions

I would run four conditions, progressively introducing strategic complexity.

### Condition A — Pure information acquisition

Agents ask questions about opponent types.

No offers are permitted until the end.

The objective is to identify the opponent's type.

This is essentially bilateral 20 Questions.

**Purpose:** Verify that EIG calculations and information-seeking policies behave as expected.

### Condition B — Information acquisition with opportunity cost

Every question consumes one turn.

Agents may ask questions or make offers.

A deal closes the game.

**Purpose:** Test the tradeoff between gathering information and acting.

This is the first condition where maximizing EIG can be suboptimal.

### Condition C — Strategic disclosure

Agents can answer truthfully, provide partial information, or withhold information, according to explicitly defined game rules.

**Purpose:** Determine whether information-seeking policies remain effective when observations are strategically generated.

This introduces a partially observable stochastic game rather than a single-agent POMDP.

### Condition D — Natural-language negotiation

Replace the structured action descriptions with natural-language utterances while preserving the underlying action semantics and payoff structure.

**Purpose:** Measure how much performance changes when semantic interpretation becomes part of the task.

This is where comparison with the original LMRL-Gym environment becomes most interesting.

---

## 8. What would count as a meaningful result?

I would pre-register three hypotheses.

**H1 — Information efficiency:** Models differ measurably in EIG regret even when restricted to the same four candidate actions.

**H2 — Strategic value of information:** Greedy EIG performs well in Condition A but can underperform an optimal utility-directed policy in Condition B.

**H3 — Uncertainty calibration:** A model's action-selection entropy predicts its decision regret only imperfectly, and this relationship changes with the gap between the best and second-best actions.

H3 is especially relevant to our earlier ESI discussion.

We should test whether a diffuse choice distribution reflects genuine uncertainty or simply the presence of several equally good actions.

---

## 9. One design decision I would make differently from our earlier discussion

I would **not introduce potential-based reward shaping in the first experiment**.

We already have exact beliefs, information gain, and utility.

Adding shaping immediately would make it harder to identify what caused any observed differences.

Instead, after establishing the baseline, introduce:

\[
F_t=\gamma\Phi(b_{t+1})-\Phi(b_t),
\qquad
\Phi(b)=-H(b),
\]

with appropriate terminal treatment.

Then compare shaped and unshaped training while holding the underlying task fixed.

That becomes a separate experiment about learning efficiency and policy invariance.

---

## 10. The experiment's central question

The original 20 Questions benchmark asks:

> Can an agent select questions that efficiently identify a hidden state?

Binned Car Dealer asks a more difficult question:

> **Can an agent determine when acquiring additional information is worth more than acting on what it already knows, while its opponent is making the same calculation?**

That gives us a natural progression from EIG to decision theory to multi-agent strategic information acquisition.

### My proposed first implementation

Start with **Condition B**, using four buyer types, four seller types, four actions per turn, and a six-round horizon.

Build the exact Bayesian reference policy before evaluating any LLM.

The immediate deliverable should be a deterministic simulator with configurable hidden types, question-response matrices, offer utilities, and exact EIG calculations.

The hardest part is not generating the dialogue. It is designing a small enough payoff and observation structure that we can independently verify the optimal decisions.

Once that is done, Jev, conventional LLMs, and BED-LLM can all be evaluated against the same ground truth.
