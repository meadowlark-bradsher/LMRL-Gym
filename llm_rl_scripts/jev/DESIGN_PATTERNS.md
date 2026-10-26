# Jev / LLM / BED-LLM design patterns across LMRL-Gym

_Last updated: 2026-10-25_

A role map for the Jev × LMRL-Gym study: for each benchmark environment, what
job each of the three approaches does, and the reusable patterns that fall out.
The narrative synthesis and the full evidence live in [`FINDINGS.md`](FINDINGS.md);
this is the reference table.

> **Status caveat.** The **Jev** cells for Text-Nav, Chess, and Chess Endgames
> are measured from live runs (see each `jev/` dir's reports). The **BED-LLM**
> cells are architectural *predictions* — **except Wordle and Car-Dealer**, which
> are now measured: Wordle is a closed code loop (`wordle/jev/bed.py` maintains the
> posterior + exact-EIG optimum; Jev selects), Car-Dealer's "Jev + belief filter"
> condition is built and run (an external acceptance-probability filter; see
> `car_dealer_discrete/jev/`), and Text-Nav's Exp D ran the belief-space planner and
> the EIG-vacuity check (`text_nav/jev/EXP_C_D_RESULTS.md`). Treat the remaining
> BED-LLM entries (Maze, 20Q, Guess-My-City) as hypotheses until a loop is run.
>
> The chess figures below reflect a **correction**: the original strength ladder
> (~700–900 Elo, a draw column) was retracted after a measurement-artifact re-run
> (move-cap truncations had been miscounted as draws). See `chess/jev/VALIDATION.md`
> and `.../results/ladder_20261025/LADDER_RESULTS.md`.

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
| **Text-Nav** | Strongly useful | **Reactive selector; completion needs externalized planning (measured, complete).** A1 (implicit goal, reactive) = **0.000**; only **A4** (explicit goal + full history) completes, **0.190 [0.113,0.273]** (K=15 calib 0.227). History is the lever (main effect +1.63; goal×history interaction +0.87); goal *alone* (A2) is counterproductive. Failure decomposes: recognition moves with context (Exp B: select target over distractor 0→0.833, eating suppressed), preferences coherent (Exp C: 0 cycles), but sequential control is absent. | Text-native: reads feedback, infers latent quest, composes the plan. Strongest standalone here. | **Belief loop solves it — but not via EIG.** Exp D: EIG is *vacuous* in this game family (room observations are unique, so nothing latent to resolve), so it can't drive exploration here; but a zero-RL **belief-space planner solves 50/50 (100%)**, pinning Jev's ~0 to the absence of exploration/belief-tracking/planning, not task difficulty. |
| **Wordle** | Canonical | **Chance-level selector (measured, census).** On official-200 (≤255, so Jev picks every turn) its per-turn selection is ≈random *within the posterior*: match@1 to the exact-EIG word **0.275 [0.255,0.296]**, regret **0.586 bits**, both matching a random mock and flat across 6/4/3-guess budgets. It still solves (any consistent guess narrows the set) but **adds no information-theoretic value**: solve **0.741 [0.696,0.784]** vs optimal **0.936** at a 3-guess budget (gap **0.196 [0.146,0.247]**). | Proposes plausible guesses from knowledge; suboptimal without explicit entropy accounting. | **Textbook EIG solver — realized, and it does all the work.** `bed.py` maintains the posterior + exact-EIG optimum in pure code; Jev merely selects. The value is entirely in the externalized loop — Jev's selection is chance-level (see Jev cell). |
| **Chess** | Diagnostic | **Legality is the whole win — and only that.** `Choice` over legal SAN = 0% illegal, but Jev loses every *finished* game to every engine setting (incl. depth-1/Skill-0); the ~700–900 Elo / draw column was a move-cap artifact (only firm claim: ≤1 non-loss in 10 vs SF-1320). Move quality is **not Jev-specific** — indistinguishable from constrained gpt-4o, far below claude-opus-5-5 (45% vs 20% best-move), and beaten by a ~30-line heuristic. Confidence is calibrated (high-conf ≈3.4× more likely best → abstain/escalate lever). State observable → no belief. | Knows chess in prose but emits illegal moves (GPT-4 scores 0 from illegality); unreliable search. | **Mostly idle.** Fully observable → little latent. Residual use: belief over *opponent policy/type*. Diagnostic, not core. |
| **Chess Endgames** | Strong | **Legal but no technique.** Converts ~10% (hangs the queen, 0 losses); missing capability is lookahead, not belief. | Knows endgame principles; illegal-move risk, no exact calculation. | **Partial.** Latent is *continuation success / evaluation* — belief over "does this line mate within N," score continuations by info about winability. More scope than full chess because outcomes are uncertain-but-studiable. |
| **20 Questions** | Core objective | **The likelihood model.** Supplies yes/no answer-probability (`Noul`) = p(answer\|hypothesis, question) — exactly what BED's EIG + hypothesis filter consume. Not a standalone questioner. | Proposes the hypothesis space (candidate entities) and phrases NL questions; can also play answerer. | **The controller.** Posterior over hidden entity, max-EIG question, filter on each answer. Canonical fit. Pipeline: LLM proposes → Jev answers (likelihood) → BED selects + updates. |
| **Guess My City** | Core, but noisy | Same likelihood role, but the channel is **noisy/ambiguous** → calibrated `Noul` matters more and is harder to get right. | Proposes city hypotheses + questions; richer world knowledge helps disambiguate unreliable semantic evidence. | Same EIG loop as 20Q **under a noisy observation model** — must down-weight/robustify unreliable likelihoods. |
| **Car Dealer** (discretized) | Mixed | **Fails alone, works with a filter (measured).** J/J play walks away every time (**0/12**; **0/130** offers ever worth ≥r to the recipient — the gap is *offer generation*, not acceptance). With an external acceptance-probability filter annotating each candidate offer, Jev selects to **24/24** agreements. | **Primary.** Text-native negotiator/persuader; models the buyer, generates dialogue, closes. | **Keystone, measured — and it splits in two.** The external belief filter is what makes Jev competent. A **prior forward model** (which deals most *types* accept, public info; non-learning) is sufficient for the **outcome** (prior-only → 12/12). **Instance inference (EIG/learning)** buys only the **quality margin** (Δ Nash **+0.158 [0.054,0.263]**, replicated; +0.102 hard regime) at a proposal cost, and the gain lands on **whoever moves second** (Δ G_responder **+0.200**), not a role. |

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
(Maze-FO, Chess, Endgames — the value is 0% invalid output, and *only* that: the
chess P2 arm shows move quality is the underlying model's, GPT-4o-class, not
Jev-specific, so the wrapper guarantees legality, not better decisions) or as a
*likelihood/answer model* (20Q, Guess-My-City, Wordle — the value is the calibrated
`Noul`). The second composes into BED; the first stands alone. Where neither applies
cleanly (Car Dealer), Jev is the holdout.

**4. EIG value tracks observability inversely.** Fully observable (FO-Maze, Chess) →
EIG ≈ 0, BED degenerate. Hidden-but-static latent (20Q, Wordle) → EIG is the
objective. Hidden-and-dynamic or reward-coupled (Text-Nav, Car Dealer) → EIG is a
*term*, traded against reward or used to drive exploration, never maximized alone.

**5. Jev contributes validity, not quality.** The structured `Choice` guarantees a
*valid* decision, but the data show it does not supply a *better* one: in Chess its
move quality equals a constrained same-era LLM's and is beaten by a ~30-line
heuristic (legality is a wrapper property), and in Wordle its selection inside the
posterior is chance-level (match@1 0.275 ≈ random). So reach for Jev when you need
legality/validity + calibration; expect decision quality to come from the model class
or the external loop, not from Jev.

**6. Externalize the belief loop — and separate the prior from instance inference.**
Where a task has a latent or needs belief/planning, move that into an external
component (code posterior, belief-space planner, acceptance filter) and let Jev
select; this is validated in Wordle (code posterior → Jev solves), Car-Dealer (belief
filter → 0→24/24), and negatively in Text-Nav (a belief-space planner solves 100%
where reactive Jev scores ~0). When you externalize, distinguish two layers: a
**prior forward model** (public, non-learning) often unlocks the *outcome* cheaply
(Car-Dealer prior-only → 12/12), while **instance-specific inference (EIG/learning)**
buys only the *quality margin* and accrues to the party that receives informative
observations. That split is the practical rule for when BED-LLM's inference machinery
is worth building.

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
