# Jev × LMRL-Gym — Findings & design patterns for Jev / LLM / BED-LLM

_Last updated: 2026-10-27. Synthesis report across the merged benchmark branches._

This report synthesizes the completed experiment programs now on `main` (Text-Nav,
Chess, Wordle, Car-Dealer, 20Q) into a single account of **what role Jev, an LLM, and a
BED-style wrapper each play**, and the reusable design patterns that follow. The
companion reference is [`DESIGN_PATTERNS.md`](DESIGN_PATTERNS.md) (the role table);
this is the narrative and the evidence.

## Definitions, notation & sources

Terms are grouped; later sections use them without re-defining.

### Methods and systems

- **LMRL-Gym** — the benchmark suite these experiments run on: Abdulhai, White, Snell,
  Sun, Hong, Zhai, Xu, Levine, *"LMRL Gym: Benchmarks for Multi-Turn Reinforcement
  Learning with Language Models"* (arXiv:2311.18232, 2023; ICML 2025, PMLR 267). Eight
  multi-turn tasks (3 dialogue + 5 RL-capability). "**The paper**" refers to this; its
  baseline columns are a zero-shot **GPT-4** policy and RL-trained policies (BC =
  behaviour cloning, ILQL, MC-returns, PPO, Online Filtered BC) — the "**RL-trained
  columns**". Code: `github.com/abdulhaim/LMRL-Gym`.
- **Jev** — the structured-decision model under test (TypeSafe's "System One" model,
  API model `jev-latest`, per the project's own setup), which emits structured
  decisions, not text. It exposes structured primitives; this report uses **`Choice`**
  (pick one option from an enumerated set) and **`Noul`** (a calibrated
  probability/confidence over the options — the signal a Bayesian loop can consume as a
  likelihood). `Choice` caps at **255 options** (hence "≤255 visible options").
- **LLM (baseline)** — a generative language model used as a text policy (the paper's
  GPT-4, or `gpt-4o` / `claude-opus-5-5` here). Broad knowledge, plans in text, but
  outputs are unconstrained (can be illegal) and its uncertainty is implicit.
- **BED-LLM** — Bayesian Experimental Design with LLMs: Choudhury, Williamson,
  Goliński, Miao, Bickford Smith, Kirchhof, Zhang, Rainforth, *"BED-LLM: Intelligent
  Information Gathering with LLMs and Bayesian Experimental Design"* (arXiv:2508.21184;
  ICLR 2026). Maintains an explicit posterior over a latent and picks the query/action
  maximizing **expected information gain (EIG)**, with a sample-then-filter belief
  update. Used here as the template for the "external belief/forward model" role.
- **The thesis** — the study's central claim being tested: a structured `Choice` over
  the *legal* option set makes invalid/illegal outputs impossible by construction,
  removing the failure mode that zeroes a zero-shot LLM. "**The thesis win**" = a
  result confirming it (e.g. Chess 0% illegal).
- **The three roles** — LLM = proposer, Jev = validity-guaranteeing selector, BED-LLM =
  external belief + acquisition; defined fully in §2 (forward-referenced in §1).

### Environments

All are LMRL-Gym tasks. **FO-Maze** = the fully observed maze navigation task.
**Text-Nav** = a TextWorld kitchen task (put the perishable food in the
fridge and close it). **Wordle**; **Chess** (+ **Chess endgames**); **20Q** = 20
Questions (158 objects / 17 categories, §3.5); **Guess-My-City**; **Car-Dealer** = a
buyer/seller negotiation (here a discretized redesign, §3.4). This report presents live
results for Text-Nav, Chess, Wordle, Car-Dealer, and 20Q; Guess-My-City / FO-Maze appear
only as predicted role placements.

### Repo, branches, and where the numbers live

Work is on the `meadowlark-bradsher/LMRL-Gym` fork; "**main**" is that fork's main
branch, into which the per-environment branches ("**the merged branches**") are merged.
"**Committed analyzers**" are the per-experiment analysis scripts that regenerate every
number; "**the source docs**" are the result write-ups under `llm_rl_scripts/<env>/jev/`.
The companion role table this report defers to for the RL argument (§5) is
`llm_rl_scripts/jev/DESIGN_PATTERNS.md`, in the same folder as this report.
Named runs/docs used as shorthand below:

- **Chess:** `REPORT.md` (first mini-report, partly superseded); **P0 validation** =
  `VALIDATION.md` (pre-eval gates); **primary (P1)** =
  `results/primary_*/PRIMARY_RESULTS.md` (200 fixed positions); **ladder re-run** =
  `results/ladder_*/LADDER_RESULTS.md` (strength ladder under fixed termination
  accounting); **P2** = `results/p2_*/P2_RESULTS.md` (is move quality Jev-specific?).
  The P0–P3 run sequence is defined under Experiment design below.
- **Wordle:** `bed.py` (pure-code posterior + exact-EIG optimum), `eval_jev.py`; the
  **"Jev-vs-exact-EIG"** program compares Jev's selection to that optimum; the **census
  run** plays every word in `wordle_official_200.txt` as a target (the file holds 236
  five-letter words; the "200" is nominal), 5 independent Jev rollouts each —
  1,180 games (`results/live_200_t3_census_*`).
- **20Q:** adapter `llm_rl_scripts/twenty_questions/jev/` (`bed.py` = pure-code
  posterior + exact-EIG + question bank; `matrix.py` = the cached object×attribute
  **Noul matrix** `matrix_jev.json`; `policy.py` = live J/J `JevOracle` + `JevSelector`;
  `analyze_matrix.py` = Noul calibration; `analyze_selector.py` = cluster-bootstrap over
  objects). Docs `DESIGN.md` / `RESULTS.md`; numbers in `results/*/summary.json` and
  `results/selector_ci_20261027/selector_ci.json`. "**Noul matrix**" = p(yes | object,
  attribute) built once (the likelihood depends only on the pair, not on history).
- **Car-Dealer:** spec is `discretized_car_dealer_experiment.md` (its **§5** lists
  policy conditions incl. "**Jev + explicit belief filter**"; its **§5.1** is the
  Jev/LLM/BED 3×3 cross-play, specified but not run). Pipeline stages → docs:
  **engine/smoke** (`SMOKE_RESULTS.md`), **salience** (`SALIENCE_RESULTS.md`), **aid** +
  **controls** (`AID_RESULTS.md`), **quality** (`QUALITY_RESULTS.md`), **responder
  replication** (`RESPONDER_RESULTS.md`), **hard regime** (`HARD_RESULTS.md`),
  **forced-counter probe** (`COUNTER_RESULTS.md`). "Under aid" / "the aid's 12/12" refer
  to the belief-filter condition.

### Experiment design

- **K** — independent rollouts per (instance, condition); repeats are needed because
  `jev-latest` is stochastic at near-tie decisions. **seed** — a fixed game instance
  (TextWorld layout, or sampled negotiation types); it is the unit of clustering for
  CIs, so "**seed-dominated**" = variance driven by which instance, not by the
  condition. **pilot / audit / calibration / main run** — successive Text-Nav runs: a
  small pilot (K=1), an identical re-run that audited its reproducibility, a K=15
  calibration that estimated the real rate, and the pre-registered 50-seed × K=8
  confirmatory main run.
- **Text-Nav experiments** (design `EXPERIMENT_DESIGN.md`):
  - **Exp A** — 2×2 of goal disclosure (G0 implicit / G1 explicit objective) × history
    (H0 reactive / H1 full episode history): **A1** implicit+reactive (closest to the
    original benchmark), **A2** explicit+reactive, **A3** implicit+history, **A4**
    explicit+history. Primary endpoint = task success; secondary = **milestone** = the
    greatest correctly-ordered subgoal reached, **M0** nothing achieved → **M1**
    Acquire → **M2** Transport → **M3** Prepare (fridge open while carrying) → **M4**
    Store → **M5** Complete (so "milestone 0.31 vs 0.45" is a mean over M0–M5).
  - **Exp B** — 2×2 salience probe (narrative on/off × food label generic/"**milk**") of
    *recognition*: does context make Jev take the **target** food over a fresh-food
    **distractor**, and suppress the fail action (eating). "**Opportunity-conditioned
    rate**" = denominator is only the logged steps where the action was admissible (so
    "eating suppressed 0/363, 0/390" = 0 eats out of 363 / 390 steps where eating was a
    legal move).
  - **Exp C** — fixed-state preference consistency: freeze decision snapshots and
    re-query (**triads** = the three pairwise orderings A-B / B-C / C-A; "12 live
    triads"); **argmax stability** = how often the top choice repeats across repeats;
    **position bias** = choice driven by option order not content; **cycles** =
    intransitive preferences (A≻B≻C≻A).
  - **Exp D** — goal-POMDP framing: score Jev against **EIG** and against a belief-space
    planner. "EIG vacuous" = observations here already pin the state (unique rooms), so
    there is nothing latent for information-gain to resolve.
  - **Design Outcome E** — the design pre-registered five expected outcomes, A–E (one
    per experiment A–D, plus this cross-cutting fifth); only E is called out in this
    report. E is the prediction that recognition, preference coherence, and sequential
    control are *separable* capabilities (borne out).
- **Car-Dealer game** (full spec in §3.4): **IR** = *individually rational* — a deal
  both sides weakly prefer to their outside option **r** (an "IR deal" clears both
  reservation values). Regimes: **easy** r=0.40 (36/36 type pairs have an IR deal),
  **hard** r=0.55 (30/36 do). **J/J** = Jev-buyer vs Jev-seller self-play. The **belief
  filter** is a finite Bayesian filter over the 6 opponent rankings with softmax
  offer-likelihood inverse-temperature **β=8** and a non-acceptance noise floor
  **ε=0.05**; **placebo** = the same filter emitting a constant 0.50 (isolates the
  information from the extra sentence); **prior-only** = the filter that never updates
  (isolates the prior from learning). **C1** = the forced-counter probe's primary pre-registered contrast (null here).
  **Move structure:** one seat is designated to open each episode and the opener is
  counterbalanced across episodes (every type pair is played buyer-first and
  seller-first), while turns alternate strictly within an episode; on its turn a seat
  may accept the standing offer, reject it (clearing it), propose any of the 27 deals (a
  counter replaces the standing offer), or walk. Six proposals are allowed per episode
  across both sides — only proposals consume the budget — and at the deadline a standing
  offer may still be accepted but no new one made; an unaccepted standing offer does not
  count, and no-deal pays each side its outside option r. Each turn a seat sees its own
  ranking, r, the budget used, the standing offer with its utility to itself, the full
  public offer/response history, and its legal-action list; with the filter on, every
  *proposal* candidate is annotated with P(opponent accepts), but the accept option is
  never annotated — the estimate informs which deal to propose, not whether to accept
  (on a receiving turn both appear on one menu, so the accept-vs-counter choice sees the
  counters' P(accept) and no acceptance estimate for accepting). The filter's prior over
  the 6 opponent rankings is uniform.
- **Chess runs** — a P-numbered sequence: **P0** = pre-eval validation gates; **P1** =
  the primary fixed-position move-quality study (200 positions); **P2** = the
  constrained-LLM comparison (is move quality Jev-specific?); **P3** = a planned,
  not-yet-run prompt-ablation study crossing three prompt framings — **S0** minimal
  (FEN + the legal-move menu, no stated objective), **S1** = S0 + an explicit goal line,
  **S2** = S0 + a factual natural-language board description — with **option-order
  randomization** (shuffling the legal-move display order). P3's full specification is
  not committed to the repo; the current P1/P2 prompt carries a terse objective line, so
  it sits closest to S1, not a clean S0.

### Metrics

- **match@1** — fraction of decisions where the chosen option equals the
  reference-optimal one (top-1 agreement). **cp lost** (centipawns; 100 cp = one pawn) —
  the eval drop vs the engine's best move (Stockfish grader). **regret (bits)** — EIG of
  the optimal guess minus EIG of the chosen one (expected information left on the table).
  **bits** = base-2 information; "**−4.94 bits**" is the worst seat's log-loss of the
  true type, −log₂ q(θ_true), i.e. confidently wrong.
- Negotiation outcome/quality: **G_i** = normalized attainable gain
  (U_i − r_i)/(U_i\* − r_i) ∈ [0,1]; **min(G)** = the worse-off side's G; **G_responder**
  = G of whoever moves second. **Nash / Q_N** = normalized Nash surplus ratio,
  (U_B−r_B)(U_S−r_S) divided by its maximum over feasible deals. **Pareto** = fraction of
  agreements on the Pareto frontier. "**+0.58 proposals**" = extra offers exchanged
  before agreement.
- **cluster-bootstrap** — resampling whole clusters (seeds, or Wordle target words)
  rather than individual rollouts, because repeats within a cluster are correlated; CIs
  are 95%. "**~18.5% per-run false-fail rate under the null**" = the chance a placebo
  gate ("all four CIs include 0") spuriously trips in one run even when nothing is wrong.

### Models and dates (as of October 2026)

- **`gpt-4o`** (OpenAI) — treated as GPT-4-class, the paper's lineage;
  **`claude-opus-5-5`** (Anthropic) — the current **frontier** model. "**Same-era LLM**"
  and "**frontier**" are relative to late 2026. In §3.2, free-generation illegal-move
  rates are **gpt-4o 18%**, **claude-opus-5-5 3.5%** ("Claude 3.5%" = that illegal rate).
  The "**P2 token-budget bug**" was `claude-opus-5-5` run with thinking-always-on and a
  64-token cap, so reasoning consumed the budget and left empty text (spurious ~86%
  "illegal"); diagnosed, fixed, and re-run. **SF = Stockfish** (chess engine); **SF
  depth-1**, **Skill-0**, and **UCI_Elo 1320** ("SF-1320") are engine strength settings
  (the binary floors at Elo 1320).

## 0. Scope & provenance

- **Merged & complete:** Text-Nav (Exp A main run + B/C/D), Chess (REPORT + P0
  validation + primary + ladder re-run + P2 + confidence H2H), Wordle (Jev-vs-exact-EIG,
  incl. the rigorous census run), Car-Dealer (engine + smoke → salience → aid → controls
  → quality → responder replication → hard regime → forced-counter probe), 20Q (Noul
  matrix + likelihood/oracle census + selector arms with repeats + cluster-bootstrap CIs).
- **Discipline:** every number below is regenerated by committed analyzers; claims
  are tagged *exact-on-logs* vs *stochastic* in the source docs; confirmatory runs
  were pre-registered before the data were read. Where a headline was later
  corrected, the correction is reported, not hidden.
- **Natural baseline** throughout is the paper's zero-shot GPT-4 column, not the
  RL-trained columns. ("The paper" = the LMRL-Gym paper; full citation in the
  Definitions above.)

## 1. Executive summary

**One pattern explains all four environments.** Jev is a *calibrated single-step
selector*: it chooses one valid option from the visible state and reports a
calibrated confidence, but it has no belief state, no forward/opponent model, no
search, and no cross-episode learning. Its only intrinsic guarantee is **validity**
(legal/structured output). It succeeds exactly when belief-maintenance and planning
are **trivial or externalized**, and fails whenever success requires it to internally
track a belief, compose a multi-step plan, model the future/opponent, or discover a
latent goal. 20Q (§3.5) is the sharpest single-task demonstration: the *same* model on
the *same* task solves **158/158** when its calibrated `Noul` is the likelihood a code
BED loop plans with, and **0/158** when it must do the planning (pick the questions)
itself.

**The sharp sub-pattern** (the most useful result of the study) is that a BED-style
wrapper provides *two separable things*, and they unlock *different* outcomes:

1. a **prior / forward model** — "what is consistent or acceptable given public
   structure" — which unlocks the **outcome** (Car-Dealer agreement goes 0→24/24 with
   a *non-learning* prior-only filter; Wordle is solvable at all only because the
   posterior of still-consistent words is maintained); and
2. **instance-specific Bayesian inference / EIG** — learning *this* opponent or
   secret — which buys only the **quality margin**, not the outcome (Car-Dealer
   learning: ΔNash +0.158; it lands on whoever has information to act on).

This maps cleanly onto the three roles: **LLM = proposer/policy**, **Jev =
validity-guaranteeing selector (+ calibrated confidence)**, **BED-LLM = the external
prior/forward model + EIG acquisition**.

## 2. The three roles

- **LLM** — *proposer / text policy.* Broad knowledge, plans in text, but
  unconstrained (illegal/hallucinated outputs) and uncertainty is implicit.
- **Jev** — *validity-guaranteeing selector / answerer.* One structured `Choice` over
  ≤255 visible options with a calibrated option-probability (`Noul`). No memory, no
  lookahead, no latent inference. Contributes a *valid* decision and a *calibrated
  confidence* — **not** a better decision (see §4.2).
- **BED-LLM** — *external belief + acquisition.* Maintains a posterior/forward model
  over a latent and selects by expected information gain. Decomposes into a prior
  forward model and instance inference (§1, §4.4).

## 3. Evidence by environment

### 3.1 Text-Nav — Jev fails, and the failure *decomposes*

The condition nearest the original benchmark (A1: implicit goal, reactive) scores
**0**. Completion appears only when both an explicit goal **and** full history are
supplied (A4), and even then only sometimes.

- **Main run** (50 seeds × 4 conditions × K=8 = 1,600 episodes, pre-registered):
  A1 = A2 = A3 = **0.000**; **A4 = 0.190 [0.113, 0.273]** (cluster-bootstrap over
  seeds). Calibration (K=15) independently pinned A4 ≈ **0.227 [0.147, 0.333]**.
- **History is the lever, not the goal.** History main effect **+1.633
  [1.163, 2.130]**; goal × history interaction **+0.868 [0.520, 1.238]**; an explicit
  goal *alone* (A2) is if anything counter-productive (milestone 0.31 < A1's 0.45).
  A4 completes on 22/50 seeds; success is seed-dominated.
- **The capability splits three ways** (Design Outcome E): *recognition* moves with
  context (Exp B: selective target-taking 0 → **0.833** only under narrative+"milk";
  eating suppressed **0/363, 0/390**); *preferences are coherent* (Exp C: **0 cycles**
  in 12 live triads, argmax stability 0.963, position bias only on genuinely
  indifferent options); *sequential control* is the missing piece.
- **The decisive control:** in Exp D a zero-RL **belief-space planner solves 50/50
  (100%)** within the 20-step budget. So Jev's 0 → 0.19 "is not about task difficulty
  or an unfair benchmark — it is precisely the absence of exploration, belief/map
  tracking, and planning." EIG itself is *vacuous* here (room descriptions are unique,
  so observations are already fully informative).

**Role lesson:** the planning/belief component is exactly what Jev lacks; supply it
(history, or an external planner) and the task is solvable.

### 3.2 Chess — legality is the whole win, and *only* that

- **0% illegal by construction** (the thesis win; the paper's GPT-4 scores 0 from
  illegal play — and free-generation LLMs reproduce it: gpt-4o 18% illegal, Claude
  3.5%).
- **But Jev loses every *finished* game.** The original ~700–900 Elo / draw column was
  a **measurement artifact** (move-cap truncations miscounted as draws); the re-run
  under fixed termination accounting shows zero genuine draws — vs random 3 wins (of 3
  finished, 3 more truncated), vs SF depth-1 0/0/2, Skill-0 0/0/5, SF-1320 **1W/9L**
  (triples are wins/draws/losses). Only firm claim: ≤1 non-loss in 10 vs SF-1320.
- **Move quality is not Jev-specific.** On 200 fixed positions Jev beats random
  (match@1 20.5% vs 3.0%) but is **beaten by a ~30-line heuristic** (26.5%, and Jev
  blunders more: 55.5% vs 40.5%), and is **statistically indistinguishable from
  gpt-4o constrained to the legal menu** (Δmatch@1 +0.035 [−0.053, +0.089]), far below
  frontier claude-opus-5-5 constrained (**45.2%**). So the structured `Choice` buys
  legality + GPT-4o-class moves — nothing beyond a same-era LLM on the same menu.
- **Confidence (`Noul`) is informative about move quality, with the signal concentrated
  in the top quintile:** splitting at the top vs bottom tercile of `Noul` (top third =
  `Noul` ≥ 0.21, bottom third = `Noul` ≤ 0.14, n=66 each), top-third moves are ~3.4×
  more likely to be best (40.9% vs 12.1% match@1; 22 vs 522 median cp lost). In finer
  quintile bins (figure) the effect is a *threshold*, not a gradient — the bottom four
  bins are flat (match@1 6.8–18.4%, ~500 cp lost) and only the top quintile (`Noul` ≥
  0.23) jumps (47.8% / 12 cp). So `Noul` **discriminates** move quality rather than
  tracking it smoothly → abstain / escalate is a genuine lever. (See
  `chess/jev/results/primary_20261025/calibration.png`.) Endgames (K+Q-vs-K) convert
  only ~10% — Jev **hangs the queen** to the bare king (no lookahead to see the
  recapture).
- **…but `Noul` is not a *better* confidence signal than a free LLM's (pre-registered
  head-to-head, H0):** on the same 200 positions and menu, `Noul`'s calibration error on
  the "acceptable move" event (ECE 0.174) is indistinguishable from a menu-constrained
  `gpt-4o`'s own logprob (0.177; paired ΔECE −0.003 [−0.011, +0.007]), and its
  discrimination edge is not significant (AUROC 0.660 vs 0.588; ΔAUROC +0.072 [−0.025,
  +0.167]). `Noul` *is* far better than a **verbalized** probability (`gpt-4o` verbal ECE
  0.657, wildly overconfident), but `claude-opus-5-5`'s verbalized probability is about as
  calibrated as `Noul` (ECE 0.162). So RLCD is **not shown to buy a calibration advantage**
  here; Jev's distinctive edge is **latency** (`Noul` ≈0.13 s/call vs `gpt-4o` logprob
  0.72 s). On the stricter exact-best event `Noul` is modestly but significantly
  better-calibrated (ΔECE −0.026 [−0.031, −0.017]) — below the pre-registered 0.05 margin.
  Details: [`confidence_h2h/`](../chess/jev/confidence_h2h/CONFIDENCE_HEAD_TO_HEAD.md).

**Role lesson:** Jev's contribution is the validity guarantee (a *wrapper* property
any model gets), plus confidence that is informative about move quality — not decision
quality.

### 3.3 Wordle — Jev's selection is chance-level; the BED loop does the work

Wordle is the clean case where the belief loop is **externalized into pure code**
(`bed.py` maintains the posterior of still-consistent words + the exact-EIG optimum;
no LLM). Jev only picks each guess from the ≤255-word posterior.

- **Rigorous census** (1,180 games, cluster-bootstrap over target words): Jev solve
  **0.741 [0.696, 0.784]** vs optimal **0.936 [0.912, 0.958]**, gap **0.196
  [0.146, 0.247]** (clears zero) at a 3-guess budget.
- **Per-turn selection is essentially random within the posterior:** match@1 to the
  exact-EIG word **0.275 [0.255, 0.296]**, regret **0.586 bits** — matching the
  *random* mock (0.246 / 0.712), and flat across 6/4/3-guess budgets. "The quantity
  that changes with the budget is the *cost* of chance-level selection, not its
  quality."

**Role lesson:** with the belief/planning loop externalized, Jev-as-selector still
*solves* (because any consistent guess narrows the set) but adds **no information-
theoretic value** over random selection. The value is entirely in the externalized
BED component.

### 3.4 Car-Dealer — the keystone: an external belief filter flips the outcome

A finite, exactly-scored bilateral negotiation (27 deals, 6 private rankings/side, 36
type pairs; easy r=0.40 → 36/36 have an IR deal). The program is the cleanest
demonstration of the central pattern because it *builds the missing component and
measures what it unlocks.*

- **Jev alone cannot negotiate.** J/J play walks away every time — 0/12 agreements (three sampled type pairs × both
  starts × K = 2), and the diagnosis is decisive: **0 of 130 offers were ever worth ≥ r to the
  recipient.** Each seat offers only its own ideal and never concedes. Salience
  (deadline/value framing) does **not** fix it (0/12) — the binding constraint is the
  **offer side**, not the accept side.
- **An external acceptance-probability filter flips it: 0/24 → 24/24 agreements** —
  baseline + salience (0/12 each) versus aid + aid+salience (12/12 each), on the same
  three sampled type pairs. The
  "Jev + explicit belief filter" condition annotates each candidate offer with
  P(opponent accepts) from a finite Bayesian filter over the 6 rankings (β=8, ε=0.05);
  it never ranks, recommends, prunes, or shows the posterior. Jev used the number
  (0/41 ideal offers under aid). The filter uses only the *belief* half of BED — a
  posterior over the opponent's ranking, updated from the public history — and no
  acquisition term: no offer is ever chosen for its expected information gain; the
  annotation reports P(accept) per candidate and Jev selects.
- **Mechanism controls separate outcome from inference:**
  - **Placebo** (constant P=0.50) reproduces the baseline *exactly* (0/12) → the effect
    is the *differential information*, not the extra sentence. Confound closed.
  - **Prior-only** (the filter never updates — learning = 0 bits by construction)
    reproduces the aid's **12/12** agreement, same efficient deals. ⇒ **a prior-
    averaged forward model — which deals most *types* find acceptable, all public
    info — is sufficient for agreement.** Identifying *this* opponent is not what
    unlocks the deal.
- **Learning buys the quality margin, not the outcome** (pre-registered, 576
  episodes/arm, replicated): Δ Nash Q_N **+0.158 [0.054, 0.263]** (replication +0.154;
  hard regime +0.102 [0.029, 0.183]); Δ min(G) +0.109; Δ Pareto +0.103; at a cost of
  +0.58 proposals. That is the easy regime, where agreement is already ≈1.0 under
  prior-only (Δ agreement +0.02); in the hard regime learning also lifts agreement,
  Δ +0.09 [+0.02, +0.18]. The gain **lands on whoever moves second** (Δ G_responder
  **+0.208 [0.146, 0.272]**; replication +0.200 [0.138, 0.266]) — not a buyer/seller role:
  Δ G_starter is flat (-0.037 [-0.091, +0.018]; replication -0.041), and the second
  mover is the one with an observation to learn from. An apparent "seller asymmetry" was shown by a
  pre-registered forced-counter probe to be **selection + small-n noise** (C1 null,
  well powered), not a role effect.
- **Honest failure mode:** the filter is probabilistic and sometimes confidently wrong
  (worst seat −4.94 bits); and the scaled 576-episode runs sit at a **precision ceiling** — their 36 type pairs
  are the *entire* type space, so more rollouts cannot narrow the intervals.

**Role lesson:** this is the pattern in its strongest form. Jev supplies the valid
selection; the **BED-style external belief filter supplies the forward model**, and
the forward model is what makes Jev competent. The prior does the heavy lifting
(outcome); EIG/learning is a quality refinement with a specific locus.

### 3.5 Twenty Questions — the Noul finally used as a BED likelihood

The one environment where the prediction of §4.1/§4.5 is actually built: Jev's `Noul`
is consumed as the per-(object, question) answer likelihood `p(yes | h, q)` that a
pure-code BED loop (`bed.py`: posterior over the 158 objects, exact EIG, Bayesian
filter) ingests for both acquisition and the hypothesis filter. Unlike Wordle — the
*closed-form* BED case where the likelihood is deterministic — here the useful questions
are natural-language attributes ("Is it alive?", "bigger than a microwave?") with **no
closed form**, so the likelihood genuinely needs a world model. The real T5 oracle is an
unavailable checkpoint, so the answerer is **J/J self-play** (Jev also answers).

- **Primitive choice:** the likelihood is read from Jev's `nouls`, which returns
  **graded** probabilities (in `matrix_jev.json`: Cat/alive 0.93, Cat/fly 0.02, Car/alive
  0.02, Car/has-motor 0.96) — whereas a de-risking probe found a yes/no `Choice` over
  {yes,no} **saturates to 0/1** at confidence 1.0. EIG and the filter both need the graded
  quantity, so `Choice` would throw the likelihood away. (The `Noul`, not `Choice`,
  carries the calibrated probability — the clean statement of the two primitives'
  division of labor.)
- **The Noul is a calibrated likelihood** (the claim §4.5 had flagged as untested): over
  the 14 attributes that name a category (ground truth known from the object
  dictionary), the `Noul` separates in- from out-of-category objects with mean **AUC
  0.977** (min 0.857, 13/14 ≥ 0.90; the one low value is a genuinely partial attribute,
  not an error) — `analyze_matrix.py`.
- **Likelihood role → BED solves every object in ~10 questions:** census of all 158
  objects, **158/158 = 1.000**, avg **10.21** questions with the deterministic
  (matrix-threshold) oracle, **10.19** with the live J/J oracle. Jev's live answers match
  its own Noul matrix **390/393 = 0.992** of the time (the 3 mismatches cost 0 losses) —
  its world-model is coherent across the two ways it is queried, so answer stochasticity
  costs ~nothing.
- **Selector role → collapse, rescued only by history** (158-object census × K=3,
  cluster-bootstrap over objects, `analyze_selector.py`): when Jev *picks the question*
  via `Choice` over the EIG-ranked shortlist instead of supplying the likelihood, it
  solves **0.000 [0, 0]**, matching the EIG argmax only **0.049 [0.047, 0.051]** of the
  time (≈chance). Mechanism: it fixates on four generic questions — "smaller than a
  microwave?", "man-made?", "indoors?", "alive?" — and **re-asks them**, with no belief
  state to track what it resolved. Mechanically forbidding repeats does *not* help
  (**0.000**, match@1 0.076); only putting the running transcript in the Choice state
  rescues it (**0.911 [0.871, 0.947]**, avg 14.0, match@1 0.202 [0.190, 0.215], regret
  0.315 bits). So the binding constraint is the **belief/history representation, not
  repetition**, and even with memory selection stays far from optimal (match@1 0.20, and
  the solve CI excludes the BED planner's 1.0).

**Role lesson:** the same model, on the same task, is **perfect as the calibrated local
judgment the planner consumes (158/158) and useless as the planner (0/158)**. 20Q folds
the Text-Nav history effect (§3.1) and the Wordle "posterior does the work" effect
(§3.3) into one experiment, and it is the first place the `Noul`-as-likelihood bridge
(§4.5) is demonstrated rather than predicted.

## 4. The design patterns

### 4.1 Propose → Answer → Select
The recurring decomposition: an **LLM proposes** (candidate set, queries, hypotheses,
or a text policy); **Jev answers/selects** with a valid `Choice` + calibrated `Noul`;
a **BED component selects/updates** by posterior + EIG. 20Q (§3.5) and Wordle (§3.3) are
the clean built instances — 20Q with Jev's `Noul` as the genuine (non-closed-form)
likelihood, Wordle as the closed-form degeneracy; Guess-My-City is the clean predicted
instance; the others are partial builds of it.

### 4.2 Jev contributes validity, not quality
The single most important correction from the data. Jev's structured `Choice`
guarantees a *valid* decision — but:
- in Chess, move quality equals a constrained same-era LLM's and is beaten by a cheap
  heuristic (the guarantee is a *wrapper* property);
- in Wordle, selection inside the posterior is *chance-level*;
- in 20Q, Jev-as-question-selector solves **0/158** (match@1 ≈ chance) and even with the
  transcript stays well below the code planner (§3.5).

So "use Jev" buys you legality/validity + calibration, and you should expect decision
*quality* to come from the model class or the external loop, not from Jev.

### 4.3 Externalize the belief loop
Where the task has a latent or needs belief/planning, **move that into an external
component** (code-maintained posterior, belief-space planner, acceptance filter) and
let Jev select from what it maintains. Validated twice: Wordle (code posterior → Jev
solves) and Car-Dealer (belief filter → 0→24/24). Text-Nav's Exp D shows the same in
the negative: the belief-space planner solves 100% where reactive Jev scores ~0.

### 4.4 Prior forward model vs instance inference — unlock different things
When you externalize, distinguish two layers, because they have different cost/benefit:
- a **prior forward model** (public, non-learning: "what's consistent/acceptable in
  general") is often enough for the **outcome** and is cheap — reach for it first;
- **instance-specific inference (EIG/learning)** buys only the **quality margin** and
  costs interaction rounds; spend it only when the margin matters, and expect its
  benefit to accrue to the party that actually receives informative observations.

This is the practical rule for *when BED-LLM's inference machinery is worth building*.

### 4.5 Confidence (`Noul`) is informative — but not better-calibrated than a free LLM's
`Noul` is informative about outcome quality (Chess: top-tercile moves ≈3.4× more likely
best; Car-Dealer: confidence trajectories), and it is the natural signal to consume — as
an abstain/escalate gate, or as the likelihood a BED loop ingests (`Noul` =
p(answer|hypothesis) for 20Q / Guess-My-City). **But a pre-registered head-to-head (§3.2,
`chess/jev/confidence_h2h/`) found no RLCD calibration advantage on Chess:** `Noul`'s
calibration error is indistinguishable from a menu-constrained `gpt-4o`'s own logprob
(ΔECE −0.003, CI includes 0) and its discrimination edge is not significant (ΔAUROC
+0.072, CI includes 0). So the case for `Noul` over a free LLM menu-probability rests on
**cost/latency** — now measured: `Noul` ≈0.13 s/call vs `gpt-4o` logprob 0.72 s (cost not
priceable without a public Jev rate). The "composable into BED" bridge therefore stands on
`Noul` being *available and cheap*, and on it being a calibrated *likelihood* — the
latter **no longer untested: 20Q (§3.5) shows the `Noul` separates category membership at
mean AUC 0.977 and, ingested as `p(answer | hypothesis)` by a code BED loop, drives it to
solve 158/158 in ~10 questions.** What remains untested is a likelihood-side head-to-head
against a free LLM's answer-probability (the 20Q oracle is J/J, not an LLM baseline), so
the *relative* calibration claim is still open the way the Chess move-choice one was.

### 4.6 When each component is primary (decision rule)
- **Observable + discrete action, no latent → Jev alone.** FO-Maze, legal-move Chess.
  Value = the validity guarantee; don't expect quality.
- **Latent, inference is the objective → BED primary**, Jev as likelihood/selector,
  LLM as proposer. 20Q (built, §3.5: Noul = likelihood solves 158/158), Wordle,
  Guess-My-City (predicted).
- **Latent/planning needed, reward sparse or multi-step → externalize the loop**
  (planner or belief filter), Jev selects. Text-Nav, Car-Dealer.
- **Reward-maximization with an instrumental latent → prior forward model for the
  outcome; add EIG only for the quality margin.** Car-Dealer.

## 5. Relation to RL / actor-critic

A natural question is whether the triad is an **actor-critic** split. It rhymes with
one but is not one, and the mismatch is the instructive part.

**The surface rhyme.** Actor-critic pairs an *actor* (a policy that proposes/selects
actions) with a *critic* (a value estimate that evaluates them and supplies the
learning signal). The Propose → Answer → Select pipeline (§4.1) lines up as: LLM =
actor, Jev = evaluator, BED component = selector.

**Why it is not actor-critic:**

1. **Nothing learns here.** The critic exists in actor-critic to produce a TD-error /
   advantage that *trains the actor* over episodes. Jev, the LLM, and the BED component
   are all zero-shot and inference-time — there is no gradient loop, so the defining
   actor↔critic feedback is absent. (The paper's *trained* RL columns are where real
   actor-critic structure lives; this triad is the inference-time decomposition instead.)
2. **The "critic" scores the wrong quantity.** A critic estimates *expected return*
   (value). Jev's `Noul` is a **likelihood** (a calibrated p(option), or
   p(answer | hypothesis)); EIG is **expected information gain**. Neither is a
   reward-value — so Jev is a critic *of the observation model*, and EIG is an
   *acquisition function*, not a value function V(s).
3. **Selector ≠ evaluator.** In actor-critic the actor both proposes and selects; the
   critic never picks. Here the BED component selects (argmax EIG, or the belief filter
   steering which offer to make) over Jev's scores — structurally closer to
   **value-based / greedy control** (a soft action-value taken greedily) than to
   policy-gradient actor-critic.

**Better-fitting vocabulary per component:** the **LLM proposer** is an actor /
behaviour policy; **Jev** is a *Q-like evaluator* whose calibrated option-probs are
taken greedily — but a likelihood critic, not a reward critic (except in Car-Dealer,
below); the **BED component** is model-based / POMDP planning (the posterior is the
belief state) plus an *intrinsic-motivation / exploration signal* — EIG is
information-directed sampling, an exploration bonus, not a value function.

**The one clean actor-critic instance: Car-Dealer.** It is the only environment here
with an extrinsic reward (the negotiated deal's surplus — see §3.4), so it is the only
one where "critic = value" is literal: the actor is the offer-proposing policy; the
critic would be an estimator of expected negotiated surplus (a role Jev *could* fill if
scored on expected return rather than likelihood); and the belief filter's information
term sits alongside as an exploration / forward-model signal — the
explore-(learn the opponent)-vs-exploit-(close the deal) split that §3.4 and §4.4 make
concrete.

So the triad is a **propose/evaluate/select** decomposition, not actor-critic. It
collapses *into* actor-critic only when the task carries extrinsic reward and Jev is
repurposed from a likelihood model into a value critic; everywhere else the "critic"
scores information or likelihood, which places it in Bayesian-experimental-design /
active-learning / value-based-control territory rather than policy-gradient
actor-critic. (`DESIGN_PATTERNS.md` § "Relation to RL formalisms" carries the same
argument in the companion role-table context.)

## 6. Methodology as a transferable pattern

The program is rigorously **self-correcting**, and that discipline is itself reusable
when evaluating structured-decision models:
- **Determinism audits reverse headline claims:** Text-Nav A4 pilot 3/10 → audit 0/10
  → calibration **0.227** (a real rate; the single runs were an up-draw and a
  down-draw of the same process). "One pass per seed is insufficient" — repeats
  mandatory, seed as the bootstrap cluster.
- **Measurement bugs caught in the open:** Chess draw-column artifact (truncation
  miscounted); P2 token-budget bug (Claude's thinking ate the 64-token budget →
  spurious 86% illegal); Car-Dealer placebo gate has an ~18.5% per-run false-fail rate
  under the null, diagnosed rather than reported as a leak.
- **Controls that isolate the mechanism:** placebo + prior-only in Car-Dealer;
  opportunity-conditioned rates in Exp B; EIG-vacuity check in Exp D.
- **Honest ceilings:** 36 type pairs = the whole type space → intervals can't be
  narrowed by more K; sharper estimates need a richer type space.

## 7. Open questions, limits, in-flight

- **Car-Dealer:** single runs for hard regime and the forced-counter probe;
  candidate-order randomization (shuffling the order the deal options are presented to
  Jev, to rule out position/order bias) not run; β/ε unswept; the IR-threshold response
  model is simple/optimistic; the −4.94-bit "confidently wrong" posterior is a live
  hazard. The full §5.1 Jev/LLM/BED 3×3 cross-play is specified but not run. The EIG
  extension — an acquisition term over offers — is also unrun, and it is non-trivial: an
  offer is both a query and a binding commitment, so an informative offer may simply be
  accepted, ending the game on a deal the proposer did not want; any EIG term therefore
  has to be weighted against expected surplus and P(accept) (the spec's §5 objective,
  score(a) = E[surplus | a] + λ·EIG(a) − c(a)).
- **Wordle:** only the ≤255-word vocab is tested (Jev picks every turn); larger vocab
  is perf-gated (EIG is O(n²)) and the model-free-opener path — the larger-vocab mode
  where Jev plays a fixed opening guess until the posterior shrinks under the 255-option
  cap — is untested.
- **Chess:** no Elo-calibrated weak opponents below SF-1320, so the Elo question is
  genuinely open (only an upper bound exists); the **P3** prompt ablations (S0/S1/S2,
  defined above) crossed with option-order randomization are not done.
- **Text-Nav:** Exp A is complete; the only un-run item is a targeted Exp C follow-up
  on hand-picked snapshots (milk-vs-distractor under A1 vs A2 framing).
- **20Q:** built (§3.5) and it confirmed the prediction — the `Noul`-as-likelihood BED
  loop solves 158/158. Open: the oracle is J/J, so there is no independent-LLM oracle arm
  (and hence no likelihood-side calibration head-to-head); the selector's history rescue
  (0.911) is one prompt; the within-category attribute bank leaves a few fuzzy
  dimensions (e.g. "used for writing/drawing" AUC 0.857).
- **Not yet built:** a real BED-LLM loop on Guess-My-City, the other task where Jev's
  `Noul` would be the answer likelihood — the 20Q result predicts the same division of
  labor (calibrated likelihood consumed by an external loop; Jev weak as the planner).

## 8. Provenance table

| Environment | Status | Jev's measured role | Headline number |
|---|---|---|---|
| Text-Nav | merged, complete | reactive selector; no planning | A1=0, A4=0.19 [0.11,0.27]; planner 100% |
| Chess | merged, complete | validity wrapper; GPT-4o-class moves | 0% illegal, loses every finished game |
| Chess confidence H2H | merged, complete | `Noul` vs LLM menu-probability | H0: ΔECE −0.003, ΔAUROC +0.07 vs gpt-4o logprob (both CIs incl 0); Jev edge is latency (0.13 vs 0.72 s/call) |
| Wordle | merged, complete | chance-level selector in a code BED loop | solve 0.74 vs opt 0.94; match@1 0.275 |
| Car-Dealer | merged, complete | selector + external belief filter | 0/12 alone → 24/24 with filter; ΔNash +0.158 |
| 20Q | merged, complete | `Noul` = BED likelihood (vs selector) | likelihood solves 158/158 ~10Q (AUC 0.977); selector 0/158, 0.911 with history |
| Guess-My-City | not built | (predicted) `Noul` = answer likelihood | — |
