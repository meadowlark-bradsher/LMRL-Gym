# Experiment Design: Jev Context Sensitivity and Sequential Decision-Making

**Project:** `Text-Nav × Jev`
**Status:** Proposed experiment (design frozen; execution gated on infra + API-cost sign-off)
**Model:** `jev-latest`
**Environment:** LMRL-Gym Text-Nav, TextWorld 1.5.4
**Primary mode:** Partially observed (PO), 20-step limit
**Training:** None — zero-shot structured choice
**Baseline already established:** 0/50 success (PO, goal mode); see [`REPORT.md`](REPORT.md)

> This document has two parts. **Sections 1–13** are the experiment design as
> specified. The **Appendix** grounds that design in the actual codebase: what
> already exists, what must be built, two facts about the environment that change
> how a factor must be implemented, and the phased execution plan with its
> API-cost gates. Read the Appendix before writing code — it is the contract
> between the design and this repository.

## 1. Motivation

Initial Text-Nav experiments demonstrated that Jev's action selection responds to
situational framing without successfully completing the environment's latent
objective.

In particular, a narrative concerning perishability, calorie restriction, and food
waste changed Jev's behavior:

- The model began taking food from the environment.
- Eating behavior was suppressed.
- The model did not complete the required sequence of transporting the perishable
  food, inserting it into the refrigerator, and closing the refrigerator.
- Selection of the perishable item over a distractor was not established.

All reported episodes ended without task success.

These observations motivate a distinction among three capabilities:

1. **Goal inference:** Inferring a useful objective from situational context.
2. **Sequential composition:** Choosing actions that advance a persistent
   objective across multiple states.
3. **Preference consistency:** Maintaining coherent preferences among available
   actions within a fixed decision context.

The experiment is designed to investigate these capabilities separately.

## 2. Research questions

### RQ1 — Goal inference

Does making the environment's true objective explicit improve Jev's progress
through the task?

The initial narrative may successfully activate food-related behavior without
conveying the exact reward condition. If explicit goal disclosure substantially
improves performance, the primary bottleneck may be goal identification rather
than multi-step control.

### RQ2 — Sequential composition

Does retaining an episode's action-observation history improve Jev's ability to
complete the task?

A purely reactive policy may repeatedly select locally reasonable actions without
maintaining a coherent trajectory. If history improves progress, some previously
observed failures may reflect missing state or goal continuity rather than an
inability to select useful actions.

### RQ3 — Preference consistency

When environmental state and context are held fixed, does Jev express preferences
among admissible actions that are consistent with a scalar ranking? This question
connects the experiment to pairwise ranking and harmonic residual analysis.

Importantly, context-dependent preference changes do not themselves demonstrate
inconsistency. Consistency must first be tested within a fixed context.

---

## 3. Experiment A: Goal inference versus sequential composition

### 3.1 Factorial design

A 2 × 2 design with two independently manipulated factors.

**Factor G — Goal disclosure**

- **G0: Implicit goal.** Provide the existing narrative concerning perishability
  and avoiding waste. Do not explicitly disclose the game's reward condition.
- **G1: Explicit goal.** Provide the identical narrative, plus the actual task
  objective: locate the perishable food, put it inside the refrigerator, and close
  the refrigerator.

**Factor H — Episode history**

- **H0: Reactive.** At every step, provide the persistent context, current
  observation, and admissible actions. Do not provide preceding actions or
  observations.
- **H1: History-aware.** Provide the same information plus the ordered history of
  previous observations and selected actions in the episode.

| Condition | Goal | History | Tests |
|---|---|---|---|
| A1 | Implicit | None | Context-driven reactive choice |
| A2 | Explicit | None | Goal-directed reactive choice |
| A3 | Implicit | Full episode | Goal inference with continuity |
| A4 | Explicit | Full episode | Goal-directed sequential choice |

**Critical control:** All four conditions use the same environment, object names,
opening observations, legal-action interface, model version, step limit, and
narrative. Only the two experimental factors change.

The environment's surfaced objective must be neutralized in the implicit
conditions (G0) so it does not act as a second, partial goal signal. Explicit
disclosure (G1) is a diagnostic intervention, not a valid score for the original
latent-goal benchmark. *(See Appendix A.1 — the environment surfaces its objective
through two independent channels, and both must be handled.)*

### 3.2 Environment configuration

Use the previously tested Text-Nav runner.

- Partially observed environments only.
- Maximum 20 actions per episode.
- No RL updates or fine-tuning.
- All chosen commands must come from the supplied admissible-action list.
- No access to hidden state, winning walkthrough, or reward specification in model
  inputs, except for the deliberate explicit-goal condition.
- Reset the episode between trials.
- **Use identical generated games across conditions** — the same seed yields the
  same map, object placement, *and* player start room for every condition.

Preserve the existing narrative and milk naming for the primary factorial
experiment. Changing food names at the same time as goal disclosure would
introduce an avoidable confound.

Episode history must be reconstructed explicitly in the prompt rather than
depending on implicit API conversation state (Jev is stateless per call — this is
already how the policy works). Keep the current observation and permitted actions
identical across paired conditions.

### 3.3 Experimental sample

**Pilot:** 10 matched game seeds per condition, 40 episodes total.

**Main experiment:** 50 matched game seeds per condition, 200 episodes total.

The primary comparison uses the same 50 games across all four conditions. If model
outputs are stochastic, add repeated runs per seed to estimate policy variability,
subject to API cost.

The episode, not the individual action, is the primary independent experimental
unit. Repeated choices within an episode must not be treated as independent
samples.

Record the model identifier and run timestamp to detect changes in the hosted
model.

## 4. Outcome measurements

### 4.1 Primary endpoint

**Task success:** Whether the environment reports `won=True` within 20 steps. This
remains the original benchmark success criterion.

### 4.2 Intermediate progress

Instrument the runner to identify observable milestones:

| Milestone | Operational definition |
|---|---|
| M1: Acquire | Agent takes the target perishable food |
| M2: Transport | Agent reaches the refrigerator while carrying the target |
| M3: Prepare | Refrigerator is open while the agent carries the target |
| M4: Store | Target food is inserted into the refrigerator |
| M5: Complete | Refrigerator is closed with target food inside; `won=True` |

Record both whether each milestone ever occurs and the greatest correctly ordered
milestone reached. A milestone reached out of sequence should not imply that
earlier milestones were completed. Some games may begin in different
configurations; the milestone evaluator must account for the initial world state.

Milestones are computed from **ground-truth world facts**, never from the policy
prompt. *(See Appendix A.2 — read `EnvInfos(facts=True)`.)*

### 4.3 Behavioral diagnostics

Record:

- Total actions before termination.
- Food pickup attempts and successful pickups.
- Food consumption and fail events.
- Refrigerator openings and closings.
- Successful food insertion.
- Drops and subsequent re-pickups.
- Revisited locations and repeated state-action loops.
- Actions that reverse earlier progress.
- Number of steps spent holding the target without advancing toward storage.

For object preference, record the choice probability **conditional on both target
and distractor actions being available**:

`P(select target | target and distractor are both admissible)`

Raw pickup counts alone are insufficient because some episodes may present more
opportunities or repeatedly cycle through the same state. **Store counts of
opportunities as well as selections.**

## 5. Planned analysis

### 5.1 Primary comparisons

**Goal effect** — compare A2 vs A1, and A4 vs A3 (effect of explicit goal at each
history setting).

**History effect** — compare A3 vs A1, and A4 vs A2 (effect of persistent history
at each goal setting).

**Interaction** — whether the value of history depends on explicit goal
disclosure. History might help only when there is a definite objective to maintain.

### 5.2 Statistical treatment

Report success proportions with confidence intervals. Use paired comparisons for
binary success because the conditions share game seeds (exact McNemar tests for
pairwise contrasts). For milestone progression and action frequencies, use paired
seed-level differences or cluster bootstrap confidence intervals.

Keep hypothesis tests exploratory unless sample size and primary contrasts have
been fixed before the main run. Fifty seeds may be insufficient to distinguish
small success differences.

### 5.3 Interpretation matrix

| Finding | Supported interpretation |
|---|---|
| Explicit goal improves success | Goal specification was an important bottleneck |
| History improves success | Episode continuity contributes to performance |
| Only goal + history succeeds | Both objective specification and continuity matter |
| Milestones improve but success stays zero | Partial sequential capability without full completion |
| All conditions fail at the same milestone | Candidate localized control bottleneck |
| All conditions fail without progress | Inconclusive regarding the precise cause |

No single unsuccessful condition demonstrates an architectural impossibility.

## 6. Experiment B: Isolate semantic salience

Tests whether the observed pickup behavior is driven by the narrative, the word
"milk," or their interaction. A 2 × 2 design:

**Factor N — Narrative:** N0 neutral description; N1 perishability/calorie/waste
narrative.

**Factor L — Object naming:** L0 generic or original food label; L1 milk label.

| Condition | Narrative | Food label |
|---|---|---|
| B1 | Neutral | Generic |
| B2 | Neutral | Milk |
| B3 | Perishability | Generic |
| B4 | Perishability | Milk |

Use identical games and the same reactive Choice mechanism, with no explicit
objective. The comparison should distinguish a mere increase in pickup activity
from selective attention to the target.

Measure:

1. Probability of taking the target when offered.
2. Preference for target versus distractor when both are legal.
3. Probability of eating when eating is legal.
4. Probability of initiating refrigerator-related actions.
5. Number of distinct state-action opportunities encountered.

If both milk and fresh food are collected equally often, the result may represent
generalized food-related activation rather than target recognition. Treat the
previous runs as exploratory observations, not as substitutes for the controlled
comparisons.

---

## 7. Experiment C: Fixed-state preference consistency

Logically independent of task completion. Its purpose is to determine whether
Jev's Choice responses admit a coherent scalar ranking within a fixed decision
context.

### 7.1 Construct decision snapshots

Capture several Text-Nav states from actual episodes. Each snapshot must include:
exact environment observation; exact admissible actions; narrative context;
current explicit goal, if any; relevant episode history, if any.

Select snapshots containing at least three meaningful admissible actions. Only
compare actions simultaneously legal in the same state. Do not fabricate
comparisons between commands that cannot coexist.

### 7.2 Pairwise choice protocol

For a snapshot with three actions A, B, and C, obtain repeated binary decisions: A
vs B, B vs C, C vs A. Randomize the presentation order within each pair, balancing
both orders. Repeat each comparison enough times to estimate the choice
probability and its variability. **Do not advance the environment during this
experiment.** The model receives the same fixed state and contextual instructions
on each query; only the candidate action pair and presentation order change.
Include multiple independent snapshots.

> **Jev gives full option probabilities in a single call.** The Choice primitive
> already returns a probability for *every* option plus a confidence (see
> `ChoiceAnswer.probabilities`). So a snapshot's full-menu distribution is one
> call, and a pairwise comparison is a two-option Choice. Repetition still matters
> for order-sensitivity and sampling variability, but the per-call richness lowers
> the cost of this experiment considerably.

### 7.3 Analysis

Estimate pairwise win probabilities and construct a directed preference graph. Fit
a Bradley–Terry model or an equivalent scalar-ranking baseline. For connected
comparison graphs with cycles, evaluate residual structure using the harmonic
ranking residual instrument, subject to its current assumptions and calibration
requirements.

Measure: pairwise selection probabilities; reversal sensitivity to presentation
order; scalar-ranking model fit; cyclic preference patterns; harmonic residual
magnitude and calibrated significance, where valid.

**Important controls:**

- Never combine different narratives when testing within-context consistency.
- Do not attribute position bias to harmonic preference structure.
- Do not equate random choice variability with systematic nontransitivity.
- Do not claim a harmonic certificate without a valid sampling model and
  calibrated test.

Context sensitivity can then be studied by comparing separately estimated
preference systems across contexts.

### 7.4 Secondary analysis: Full-menu effects

At the same fixed snapshot, compare choices made from all three actions with those
made from binary subsets. This tests whether adding a third admissible action
changes the relative preference between the original two. Menu-dependent choices
are distinct from cyclic pairwise preferences and should be reported separately.

## 8. Optional diagnostic: Supplied plan

If the explicit-goal/history-aware condition (A4) fails to complete any episodes,
add one diagnostic condition supplying the correct *abstract plan*:

1. Find the perishable. 2. Pick it up. 3. Travel to the refrigerator. 4. Open the
refrigerator. 5. Insert the perishable. 6. Close the refrigerator.

The plan does not supply the correct command at each step; Jev must still map the
plan to available actions and observations. Improved performance would suggest a
bottleneck in plan formulation or maintenance; continued failure would point
toward plan execution, state interpretation, or action selection. This is an
intervention, not zero-shot latent-goal benchmark performance.

## 9. Implementation requirements

Extend the existing `llm_rl_scripts/text_nav/jev/` evaluation code. Provide:

- Explicit control of goal-disclosure and history factors.
- Reproducible game-seed generation and matching.
- A uniform policy request builder.
- Structured per-step logging.
- Milestone detection from observations and environment state.
- Snapshot capture for the fixed-state experiment.
- Pairwise and full-menu Choice probing.
- Separate analysis scripts for success, milestone progression, semantic behavior,
  and preference consistency.

Recommended outputs:

- `episodes.jsonl` — full trial metadata and results.
- `steps.jsonl` — observations, actions, candidate commands, and diagnostics.
- `snapshots.jsonl` — reproducible decision contexts.
- `pairwise.jsonl` — repeated fixed-state comparisons.
- `summary.csv` — condition-level outcomes.
- `report.md` — results, confidence intervals, plots, and interpretation.

For internal instrumentation, environment state may be used to evaluate
milestones, but it must never leak into the policy prompt unless explicitly part
of an intervention. Validate the runner against the game's winning walkthrough
before collecting primary results.

## 10. Reproducibility and validity criteria

An experiment is valid only when:

- All policy actions come from the admissible command set.
- Matched conditions use the same games and initial states.
- The model is not trained or updated between episodes.
- No future observations or hidden reward information leak into policy inputs.
- All independent variables are explicitly logged.
- Success is determined by actual environment termination.
- The preference experiment freezes the decision state.
- API errors and incomplete episodes are tracked separately from legitimate
  failures.

Run an offline self-test of configuration, logging, and milestone detection before
spending API calls.

## 11. Expected research outcomes

- **Outcome A — Context-responsive reactive selection.** If narrative framing
  consistently changes action frequencies, Jev demonstrates measurable situational
  sensitivity.
- **Outcome B — Goal-directed selection.** If explicitly stating the objective
  improves progress, Jev demonstrates that goal specification influences local
  choices.
- **Outcome C — Sequential control.** If history enables correct action sequences
  and terminal success, Jev demonstrates at least some capacity for goal-directed
  behavior across observations without RL training.
- **Outcome D — Preference structure.** If repeated fixed-state comparisons are
  well explained by a scalar ranking, a simple utility interpretation may suffice.
  If reproducible nontransitivity remains after controlling for order and sampling
  uncertainty, the harmonic instrument may become useful for characterizing the
  deviation.
- **Outcome E — Separation of capabilities.** The most scientifically valuable
  result may be a clear separation between semantic activation, preference
  coherence, and sequential policy effectiveness. These are not interchangeable
  properties of a decision model.

## 12. Epistemic status

The existing Text-Nav results are observations from reported runs. The
explanations involving salience, goal inference, and planning remain hypotheses.
The proposed experiments discriminate among these hypotheses rather than
presuppose an architectural deficiency or advantage. In particular:

- Failure to complete the task is not proof that Jev cannot plan.
- Changed action frequency is not necessarily evidence of selective salience.
- Context-dependent rankings are not automatically inconsistent.
- A ranking inconsistency does not necessarily predict task failure.
- A successful trajectory does not establish general planning competence.

## 13. Research objective

**Determine whether Jev's structured Choice mechanism supports context-sensitive
preferences, whether those preferences are internally rankable, and under what
informational conditions they can be composed into successful sequential
behavior.**

The core contribution is not a single benchmark score. It is an experimental
decomposition of recognition, preference structure, and control in a
non-generative decision model.

---

# Appendix: Codebase grounding & execution plan

This appendix reconciles the design above with the repository as it stands on this
branch. It records what exists, two environment facts that constrain how a factor
must be built, the new components to write, and the phased execution order with
API-cost gates. The design (§1–13) is authoritative for *intent*; this appendix is
authoritative for *how it lands in code*.

## A. Environment facts that change the design's implementation

### A.1 The surfaced objective arrives through two channels, not one

`make_game.build_and_compile_game` ([`../env/make_game.py:136`](../env/make_game.py))
sets a **full, explicit** objective string:

> "Your objective is to put the {food} on the table in the living room into the
> fridge in the kitchen."

This is richer than the "close the fridge" framing described in the earlier
report. Separately, TextWorld's grammar blends a *task preamble* into the first
observation's `feedback` ("You are now playing a profound round of TextWorld!…
Your task for today is to…"), which the current runner strips via
`_strip_task_preamble` ([`eval_jev.py:53`](eval_jev.py)).

**Consequence for Factor G:**

- The **G0 (implicit)** conditions must withhold *both* channels: pass
  `use_objective=False` (so `state["objective"]` is not pinned) **and** keep the
  preamble strip on. The current `--narrative` path already does exactly this.
- The **G1 (explicit)** conditions must supply a *controlled* objective string we
  author (the design's wording: "locate the perishable food, put it inside the
  refrigerator, and close the refrigerator"), **not** the raw `game.objective`.
  Using the raw string would confound "explicit goal" with the environment's own
  phrasing and would omit the "close" step. So G1 injects a fixed, hand-written
  goal independent of `state["objective"]`.
- **Validation task:** before the pilot, dump `state["objective"]` and the raw
  first-observation feedback for several seeds to confirm exactly what each channel
  contains and that G0 leaks neither. (Offline where possible; one cheap live
  reset otherwise.)

### A.2 Matched seeds are NOT currently supported — this is the first thing to fix

The design's central statistical requirement is *the same games across all
conditions* (paired McNemar). The current code cannot do this:

- `build_game` ([`../env/make_game.py:82`](../env/make_game.py)) picks the player's
  start room with `random.choice(...)` using the **global unseeded** `random`
  module.
- `TextNavEnv._reset(seed=...)` ([`../env/env.py:29`](../env/env.py)) **accepts a
  `seed` argument but ignores it** — it neither seeds `random` nor sets
  `GameOptions` seeds.
- Object placement is otherwise deterministic (hard-coded `.add(...)`), and the
  grammar uses a fixed `RandomState(1234)`, so room *descriptions* are stable. The
  only live source of per-episode variation is the start room (and any
  `GameOptions` RNG).

**Fix:** thread a real seed through. Seed Python's `random` (and set
`options.seeds`) inside the game build so a given integer seed deterministically
fixes the start room and full game. This must land and be verified (same seed →
identical opening observation and start room across repeated builds) before any
matched-condition run. Implement without disturbing existing call sites
(default-seeded behavior stays the same when no seed is passed).

### A.3 Milestone detection needs ground-truth facts

Enable `EnvInfos(facts=True)` so the instrumentation can read world propositions
(`in(food, fridge)`, `closed(fridge)`, `eaten(food)`, player location, inventory
contents) directly, rather than parsing feedback text. These facts feed the
milestone evaluator and the diagnostics in §4.2–4.3 and **must never be placed in
the policy state**. The milestone evaluator snapshots the initial facts (per A.2's
varying start room) so "greatest correctly ordered milestone" is computed relative
to the actual starting configuration.

## B. What already exists (reuse, do not rebuild)

- **Jev client** ([`../../../LLM_RL/jev/client.py`](../../../LLM_RL/jev/client.py)) —
  `choice()` returns `ChoiceAnswer{choice, probabilities, confidence, raw}`. Full
  per-option probabilities come free in one call (important for Experiment C).
  `MockJevBackend` enables offline loop tests.
- **Policy** ([`policy.py`](policy.py)) — `JevTextNavPolicy.build_state` already
  composes `Context` + `Goal` + `Transcript` + `Current observation`. The four A
  conditions are combinations of *which of these parts are included*:
  - Context: always the narrative (both G0 and G1 in Experiment A).
  - Goal: included iff G1.
  - Transcript: included iff H1 (set `history_chars=0` for H0, full for H1).
- **Runner** ([`eval_jev.py`](eval_jev.py)) — `run_episode` drives
  `_reset/_step/state` directly (working around the two `env.py` wrapper bugs) and
  already supports `--narrative` (= G0 context, objective withheld, preamble
  stripped) and `--food-name`.
- **Offline self-test** (`--self-test`) and **fake env** for logic checks with no
  key/network/TextWorld.

## C. New components to build

1. **Seeding** (A.2) — deterministic game generation keyed by integer seed.
   *(No API cost; blocks everything else.)*
2. **Condition matrix** — a small config layer expressing A1–A4 (and B1–B4) as
   `{context, goal_text, history}` triples, so one runner produces any cell. G1's
   goal text is the fixed authored string (A.1), not `game.objective`.
3. **Milestone evaluator** (A.3) — from `facts`, compute M1–M5, ever-reached and
   greatest-ordered, initial-state aware.
4. **Structured logging** — `episodes.jsonl`, `steps.jsonl` (per step: state parts
   sent, admissible set, chosen command, full option probabilities, confidence,
   facts-derived diagnostics, opportunity flags for target/distractor
   availability), plus run metadata (model id, timestamp, seed, condition).
5. **Analysis scripts** — success (paired McNemar + CIs), milestone progression,
   semantic behavior (Experiment B), and preference consistency (Experiment C:
   Bradley–Terry fit, order-reversal, cycle/harmonic-residual, menu effects).
6. **Snapshot capture + pairwise/full-menu prober** — record frozen decision
   contexts to `snapshots.jsonl`; replay them for repeated two-option and
   full-menu Choice calls to `pairwise.jsonl`, with randomized/balanced order.
7. **Report generator** — `report.md` with tables, CIs, and the §5.3 / §11
   interpretation framing.

## D. Phased execution order (with cost gates)

| Phase | Work | API cost | Gate |
|---|---|---|---|
| 0 | Seeding (A.2), condition matrix, milestone evaluator, logging | none | — |
| 1 | Extend `--self-test` to cover all A/B conditions, milestone logic, logging schema | none | offline green |
| 2 | Walkthrough validation per §9 — step each seed's own walkthrough, confirm M1–M5 + `won` fire; confirm G0 leaks no objective | ~1 cheap reset/seed | **STOP — review before any Jev spend** |
| 3 | **Pilot** Experiment A: 10 matched seeds × 4 conditions = 40 episodes | ~40 × ≤20 calls | **STOP — review pilot; fix sample size & pre-register contrasts before main run** |
| 4 | **Main** Experiment A: 50 matched seeds × 4 = 200 episodes | ~200 × ≤20 calls | review |
| 5 | Experiment B: 2×2, reuse matched seeds, reactive, no goal | similar to a pilot | review |
| 6 | Experiment C: snapshots from Phase 3–4 episodes; pairwise + full-menu probes | low per snapshot (probabilities are one call) | review |
| 7 | Optional §8 supplied-plan diagnostic, *only if* A4 wins 0 episodes | small | — |
| 8 | Analysis + `report.md` | none | — |

**Cost control.** Every live move is one Jev API call; a 200-episode run is up to
~4,000 calls. Runs use `conda run --no-capture-output -n LLM_RL python -u …` (plain
`conda run` buffers until exit and looks hung). Start small; the Phase-2 and
Phase-3 gates exist specifically so we confirm correctness and see pilot signal
before committing to the full main run. API errors and truncated episodes are
logged and counted separately from genuine task failures (§10).

## E. Open choices — resolved (2026-10-25)

- **Stochasticity / repeats per seed (§3.3) — RESOLVED: repeats are mandatory.**
  A 12-call probe first suggested the argmax was deterministic, but it only
  sampled wide-margin states. The pilot determinism audit (the full 10×4 re-run)
  settled it: only 8/40 realized trajectories were identical across two runs, and
  all three of the pilot's A4 wins vanished on re-run (A4 3/10 → 0/10). So
  `jev-latest`'s per-episode behavior is substantially stochastic at near-tie
  decisions; single-pass binary success is a coin flip, not a measurement. The
  unit of analysis is the **per-(seed, condition) success rate over K independent
  rollouts** (runner `--repeats K`), reported with cluster-bootstrap CIs. K and
  scale to be set from a repeats-calibration run. See
  [`PILOT_A_RESULTS.md`](PILOT_A_RESULTS.md).
- **Pilot scale — RESOLVED.** Ran 1-seed smoke then the full 10 seeds × 4.
- **G1 goal wording — RESOLVED.** Using the authored default in
  [`conditions.py`](conditions.py): "Your goal is to locate the perishable food,
  put it inside the refrigerator, and close the refrigerator."
