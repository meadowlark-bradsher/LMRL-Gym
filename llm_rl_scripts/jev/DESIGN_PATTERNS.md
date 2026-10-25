# Jev / LLM / BED-LLM design patterns across LMRL-Gym

_Last updated: 2026-10-25_

A role map for the Jev × LMRL-Gym study: for each benchmark environment, what
job each of the three approaches does, and the reusable patterns that fall out.

> **Status caveat.** The **Jev** cells for Text-Nav, Chess, and Chess Endgames
> are measured results from live runs (see the per-environment `jev/` dirs and
> their reports). The **BED-LLM** cells are mostly architectural *predictions* —
> only Exp D (Text-Nav goal-POMDP + EIG) is drafted, and no BED loop has yet been
> closed on any of these environments. Treat BED-LLM entries as hypotheses until a
> loop is actually run.

## The three roles

- **LLM** — *proposer / text policy.* Broad world knowledge, reasons and plans in
  text, but uncertainty is implicit and uncalibrated and outputs are unconstrained
  (can be illegal / hallucinated). This is the paper's zero-shot GPT-4 column.
- **Jev** — *answerer / scorer / legal selector.* One structured, guaranteed-valid
  `Choice` over ≤255 **visible** options with a calibrated option-probability
  (`Noul`). No memory, no lookahead, no latent inference. Its `Noul` =
  p(answer | hypothesis, query) is exactly the **likelihood** a BED loop consumes.
- **BED-LLM** — *controller.* Maintains an explicit posterior over a latent, picks
  the query/action maximizing **expected information gain (EIG)**, and filters
  hypotheses on each observation. Needs a well-defined latent + hypothesis space +
  observation likelihood. (arXiv 2508.21184, ICLR 2026.)

## Expanded role table

| Environment | Relevance | Jev role | LLM role | BED-LLM role |
|---|---|---|---|---|
| **Fully Observed Maze** | Redundant | **Primary, sweet spot.** Greedy `Choice` over ≤4 legal moves from visible state; no uncertainty to resolve. | Overkill text planner; adds nothing over known state, risks invalid moves. | **Degenerate.** No latent → EIG ≈ 0. Don't instantiate a belief loop where nothing is hidden. |
| **Partially Observed Maze** | Essential | **Mismatched.** Reactive one-step selector with no belief/memory; can't maintain the location posterior the task demands (same failure mode as Text-Nav). | Carries an *implicit* belief in context ("walls seen here"); uncalibrated. | **Canonical.** Explicit posterior over position/map; active localization by max-EIG moves. Jev's option-probs can serve as the observation likelihood. |
| **Text-Nav** | Strongly useful | **Empirical holdout.** 0/50 baseline; only ~0.23 with explicit goal + full history (A4). Evokes salience, not multi-step plan composition — can't discover the hidden quest. | Text-native: reads feedback, infers latent quest, composes the plan. Strongest standalone here. | **Wrap the selector.** Goal-POMDP + EIG supplies the missing exploration drive (Exp D). Jev's logged option-probs scored by EIG as a yardstick. |
| **Wordle** | Canonical | **Scorer over a shortlist.** 255 cap means it can't hold the full word list; scores a BED-pruned candidate set. | Proposes plausible guesses from knowledge; suboptimal without explicit entropy accounting. | **Textbook EIG solver.** Posterior over secret word; pick guess maximizing expected feedback entropy reduction. Pipeline: LLM shortlist → Jev/BED score → BED picks. |
| **Chess** | Diagnostic | **Legality win.** `Choice` over legal SAN = 0% illegal by construction; ≈700–900 Elo novice, no search/tactics. State observable → no belief. | Knows chess in prose but emits illegal moves (GPT-4 scores 0 from illegality); unreliable search. | **Mostly idle.** Fully observable → little latent. Residual use: belief over *opponent policy/type*. Diagnostic, not core. |
| **Chess Endgames** | Strong | **Legal but no technique.** Converts ~10% (hangs the queen, 0 losses); missing capability is lookahead, not belief. | Knows endgame principles; illegal-move risk, no exact calculation. | **Partial.** Latent is *continuation success / evaluation* — belief over "does this line mate within N," score continuations by info about winability. More scope than full chess because outcomes are uncertain-but-studiable. |
| **20 Questions** | Core objective | **The likelihood model.** Supplies yes/no answer-probability (`Noul`) = p(answer\|hypothesis, question) — exactly what BED's EIG + hypothesis filter consume. Not a standalone questioner. | Proposes the hypothesis space (candidate entities) and phrases NL questions; can also play answerer. | **The controller.** Posterior over hidden entity, max-EIG question, filter on each answer. Canonical fit. Pipeline: LLM proposes → Jev answers (likelihood) → BED selects + updates. |
| **Guess My City** | Core, but noisy | Same likelihood role, but the channel is **noisy/ambiguous** → calibrated `Noul` matters more and is harder to get right. | Proposes city hypotheses + questions; richer world knowledge helps disambiguate unreliable semantic evidence. | Same EIG loop as 20Q **under a noisy observation model** — must down-weight/robustify unreliable likelihoods. |
| **Car Dealer** | Mixed | **Weak fit.** Could pick a next offer from a discrete menu, but objective is price-maximization, not inference — no latent to *infer* for its own sake. | **Primary.** Text-native negotiator/persuader; models the buyer, generates dialogue, closes. | **Instrumental only.** A latent *does* exist (buyer's reservation price/preferences), so info-seeking helps — but it's **explore (learn) vs exploit (persuade/close)**: EIG must be traded against expected reward, not maximized alone. |

## Patterns that fall out

**1. The recurring pipeline is Propose → Answer → Select.** LLM proposes
(hypotheses, shortlists, NL queries, or a raw text policy); Jev answers/scores with
a calibrated, guaranteed-valid `Noul`; BED-LLM selects by EIG and updates the
posterior. 20Q, Guess-My-City, and Wordle are the clean instances; the rest are
degeneracies of it.

**2. Which component is *primary* is set by two switches — is there a latent, and
is inferring it the objective:**

- **Observable + discrete action, no latent → Jev alone.** FO-Maze, Chess. BED
  idles, LLM is overkill. This is where Jev's legality/validity guarantee is the
  whole story.
- **Latent, and inference *is* the objective → BED-LLM primary**, with Jev as
  likelihood and LLM as proposer. 20Q, Guess-My-City, Wordle, PO-Maze.
- **Latent exists but reward is sparse/multi-step → LLM policy, or BED-wrapped to
  supply exploration.** Text-Nav.
- **Reward-maximization where the latent is only instrumental → LLM policy primary,
  BED optional and regime-gated.** Car Dealer — the explore/exploit split, not pure
  EIG.

**3. Jev has two distinct jobs.** It shows up either as a *legal-action selector*
(Maze-FO, Chess, Endgames — the value is 0% invalid output) or as a
*likelihood/answer model* (20Q, Guess-My-City, Wordle — the value is the calibrated
`Noul`). The second composes into BED; the first stands alone. Where neither applies
cleanly (Car Dealer), Jev is the holdout.

**4. EIG value tracks observability inversely.** Fully observable (FO-Maze, Chess) →
EIG ≈ 0, BED degenerate. Hidden-but-static latent (20Q, Wordle) → EIG is the
objective. Hidden-and-dynamic or reward-coupled (Text-Nav, Car Dealer) → EIG is a
*term*, traded against reward or used to drive exploration, never maximized alone.

## Relation to RL formalisms

A natural question is whether the triad is an **actor-critic** split. It rhymes with
one but is not one, and the mismatch is the useful part.

**The surface rhyme.** Actor-critic = an *actor* (policy that proposes/selects) and a
*critic* (value estimate that evaluates and supplies the learning signal). The
Propose → Score → Select pipeline lines up as: LLM = actor, Jev = evaluator,
BED-LLM = selector.

**Why it is not actor-critic:**

1. **Nothing learns here.** The critic exists in AC to produce a TD-error / advantage
   that *trains the actor* over episodes. Jev, the LLM, and BED-LLM are all zero-shot,
   inference-time — no gradient loop, so the defining actor↔critic feedback is absent.
   (The paper's *trained* RL columns are where real actor-critic structure lives; this
   triad is the inference-time decomposition instead.)
2. **The "critic" scores the wrong quantity.** A critic estimates *expected return*
   (value). Jev's `Noul` is a **likelihood** p(answer | hypothesis); BED-LLM's EIG is
   **expected information gain**. Neither is a reward-value — so Jev is a critic *of the
   observation model*, and EIG is an *acquisition function*, not V(s).
3. **Selector ≠ evaluator.** In AC the actor both proposes and selects; the critic never
   picks. Here BED-LLM selects (argmax EIG) over Jev's scores — structurally closer to
   **value-based / greedy control** (soft-Q with argmax) than policy-gradient AC.

**Better-fitting vocabulary per component:**

- **LLM proposer** → actor / behavior policy (or a proposal distribution).
- **Jev scorer** → a **Q-like evaluator** (its option-probs act as a soft action-value
  that BED takes greedily) — but a *likelihood* critic, not a reward critic, except in
  Car Dealer.
- **BED-LLM** → **model-based POMDP planning** (posterior = belief state) plus an
  **intrinsic-motivation / exploration signal** — EIG is information-directed sampling /
  curiosity, an exploration bonus, not a value function.

**The one clean actor-critic instance: Car Dealer.** It is the only environment with
extrinsic reward (sale price), so it is the only one where "critic = value" is literal:
actor = LLM negotiator; critic = a value estimator of expected sale price (a role Jev
*could* fill if scored on expected return rather than likelihood); EIG = an exploration
bonus added to that value — i.e. the explore/exploit split already in the table.

So the triad is a **propose/evaluate/select** decomposition, not actor-critic. It
collapses *into* actor-critic only when the task carries extrinsic reward and Jev is
repurposed from a likelihood model into a value critic; everywhere else the "critic"
scores information or likelihood, placing it in Bayesian-OED / active-learning /
value-based-control territory rather than policy-gradient AC.
