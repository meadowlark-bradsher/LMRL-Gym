# Live Car-Dealer — design decisions (`live-car-dealer-v0.1`), Phases 0 and 1

**Source:** [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md)
(`CD-VOI-1` / `CD-MIS-1`, copied verbatim from the author's document, 2026-10-27).
**Scope of this note:** Phase 0 (implementation and oracle readiness) and the Phase 1 build
(menu rule, arms, runner, scripted baselines, costlessness check). No research claims, no
registration, no Jev or LLM call has been made. Numbers live in [`PHASE0_RESULTS.md`](PHASE0_RESULTS.md),
rendered from `results/live_phase0_sweep_20261027/sweep.json`.

## 1. Rules the document left to Phase 0 (section 3.2, "revise before preregistration")

| Question | Decision | Why |
|---|---|---|
| A question does not cancel a standing offer, so who moves after an ASK? | Initiative passes to the other side (as written). If that side's **own** offer is the one standing, it may **HOLD** (free; keep the offer, pass back), re-PROPOSE (concede), ASK, or WALK. HOLD cannot loop: only the proposer may hold, and after a hold the other side faces the offer again. | Keeps "the respondent observes which question was asked" and the tempo cost of asking (the opponent gets a move in between) without a dead state. |
| Deadline when the last opportunity is spent | If an offer stands, the **non-proposer** gets one final ACCEPT / WALK decision regardless of whose turn it was; otherwise no deal. | Legacy semantics ("a standing offer may still be accepted"); it also makes "ask, hear the answer, then accept or not" a legal final sequence. |
| Budget accounting | H opportunities shared by both sides; PROPOSE and ASK consume one each; ACCEPT, REJECT, WALK, HOLD are free. | Design document item 1. The clock is therefore **not** the binned game's `rounds + final offer`. |
| Objective | Net gain `(U − r)·1[deal] − c_q · (#own asks)`, no deal = 0. | Design document section 4. |
| Answers in T1 | Truthful and deterministic (`answer_noise = 0`), given by the opponent itself. | The "clean primary condition" of section 3.3. |

## 2. The T1 kernel (`opponent.py`), parameters exposed to the oracle

A history-responsive scripted negotiator of hidden ranking θ. **Aspiration**
a(remaining) = r + (α₀ − r)·remaining/H falls linearly to r at the deadline. Facing the other
side's offer it ACCEPTs iff U_θ ≥ a (minus `accept_noise`), else COUNTERs with prob `p_counter`
or REJECTs. Proposals are a softmax(β·U_θ) over its `shortlist` best deals with U_θ ≥ a, **never a
deal the other side has rejected, never its own standing deal** (the history dependence). With
its own offer standing after being asked, it HOLDs with prob `p_hold`, else concedes. At the
deadline it accepts anything IR; it never walks early and never asks. Every probability it uses is
returned by `T1Kernel.distribution`, so the oracle and the matched filter use identical numbers.

`shortlist = 3` is a tractability choice (it caps the opponent's branching at 5); the sweep
records it, and the design document's mismatch study (CD-MIS-1) is where this kernel gets perturbed.

## 3. The oracle (`oracle.py`)

Exact finite-horizon dynamic programming over (public state key, posterior), floats. Focal nodes
maximise over the **full** legal action set (27 proposals, ≤ 9 questions, accept/reject/hold/walk
as legal); opponent nodes take the kernel expectation marginalised over the belief; ASK branches
on the answer distribution and is charged `c_q`. `ask_allowed=False` removes ASK at every future
decision, giving V\*_noask; Δ\*_active = V\*_ask − V\*_noask ≥ 0 is the global value of inquiry. The
oracle also counts focal nodes where ASK is **strictly** optimal (Phase 0 gate 7).

Gates passed (`test_live.py`): kernel distributions sum to 1 and are legal at every reachable
node; the opponent never accepts below r; deadline and HOLD rules behave as specified; the focal
belief updates only from opponent evidence and stays normalised; the oracle value equals an
exhaustive no-max policy evaluation of its own policy to < 1e-9 and a Monte Carlo rollout within
4 SE; V\*_ask ≥ V\*_noask at every root and the forbidden oracle never asks.

Cost: ~0.0 s at H = 2, ~0.3 s at H = 3, ~3 s at H = 4 per (kernel, type, opener) pair on this
machine; H = 6 is left for approximate planning as the document anticipates.

## 4. Phase 1 decisions received 2026-10-29 (recorded before registration)

Taken by the study's reviewer after reading `PHASE0_RESULTS.md`; the registration will state that
the kernel was chosen after Phase 0 and before any model call.

| Item | Decision |
|---|---|
| Confirmatory cell | H = 4, c_q = 0, kernel α₀ = 0.8, β = 8, p_counter = 1, p_hold = 1, shortlist 3 (the aggregate with the largest Δ\*_active) |
| Named secondary regimes | H = 2, c_q = 0 (negative control: ASK never optimal); H = 3, c_q = 0 (positive control across all eight kernels); H = 4, c_q = 3/100 (does the policy respect question cost). The rest of the grid is reported in full. |
| Proposal menu | Shortlist primary (section 5: three offer candidates + planner completeness); the full 27-deal menu is a named secondary arm. EIG-shown / EIG-hidden treatment carried over from Condition B unchanged. |
| Costlessness | Must cover both restrictions. Δ\*_menu is the ceiling; value the restricted menu loses against the full-menu oracle is reported separately and not counted against the model. **Amended 2026-10-29 (reviewer):** three offer candidates, each by a written rule, recorded as a deviation from Condition B's two; ceilings computed both ways with the difference reported as the menu artefact; the C_plan loss reported for both restrictions; stopping rule: a shift of Δ\* above 0.005 stops the registration. Section 5 records how the rule fired at H = 3 and was resolved by planner completeness. |
| Belief | The registration states which belief SAFE / AMBITIOUS are built from (the matched filter, updated on answers and on the opponent's moves) and scopes the claim to the ask-versus-offer decision with belief maintained outside the model. |
| Primary estimand | Jev with ASK − Jev without ASK, paired by cell, role-balanced, **focal-opens cells** (the opponent-opens ceiling sits below δ = 0.02); all cells secondary. `C_plan − B_plan` = Δ\* reported as the exact ceiling with no CI, plus the fraction of it Jev captures. |
| Selectivity | Jev's ask rate split by whether ASK is strictly optimal in that state (best ASK Q\* > best non-ASK Q\*), reported in the primary cell, the H = 2 cell and the c_q = 3/100 cell. |
| Logging | Exact Q\* and EIG of every option at every decision (for the residual information-bias fit). |
| Denominators | Each Phase 0 aggregate has **24** cells (2 roles × 2 openers × 6 focal types; the opponent type is marginalised by the prior), not 72; the means were already over 24 and the note header was corrected. |

## 5. The Phase 1 menu rule (`menu_live.py`), as registered

At each focal decision the model sees, in this order (then lettered and shuffled for Jev):

1. the legal free moves: ACCEPT / REJECT when facing the other side's offer, HOLD when its own
   offer stands, WALK always; at the deadline the menu is {ACCEPT, WALK};
2. the **two unasked questions with the highest exact EIG** under the current belief (ties by bank
   order), plus **PLAN_Q**: the ASK planner's best question if it is not already among them;
3. **three offer candidates** built from the current belief over the own-IR pool minus the focal's
   own standing deal — SAFE = argmax P_b(accept); AMBITIOUS = argmax own utility among deals with
   P_b(accept) > 0; **PLAN** = the proposal with the highest exact no-ask value at this (state,
   belief); BALANCED = argmax P_b(accept)·(U − r) fills only if the three coincide — plus
   **PLAN_ASK**: the ASK planner's best proposal if it is not already present.

P_b(accept) is the kernel's exact probability of ACCEPT at the opponent node after the proposal,
marginalised over the belief (so it reflects the opponent's aspiration at that point in the budget
and the deadline rule). EIG of an offer is the information in the opponent's response; offers are
informative. The belief is the matched filter, updated on answers and on the opponent's moves, and
is maintained outside the model. Tags are analyst quantities; Jev never sees them.

**Why three candidates and planner completeness (2026-10-29).** The reviewer's costlessness
condition was tested with the Condition B rule (two offers, two questions). It is costless for the
ASK planner at H = 4 but costs the **no-ask** planner 0.0148 utility in every focal-opens cell
(0.0045 with BALANCED as a third): the planner's best opening offer is neither SAFE nor AMBITIOUS,
and no EIG / P(accept) / utility rule reproduces it — two deals tie on EIG and P(accept) and the
planner prefers the one whose accepting types leave the more useful residual belief (a VOI-H3
instance). The reviewer then set a stopping rule: if the registered menu shifts Δ\* by more than
0.005 (a quarter of δ), stop and report before registering. With PLAN the no-ask side became
lossless everywhere, but at H = 3 the **question** restriction cost the ASK planner 0.015–0.018 and
removed the whole value of asking in the focal-opens cells, for the same reason on the question side,
so the rule fired for the H = 3 secondary regime. The response is *planner completeness*: PLAN_ASK
and PLAN_Q guarantee that both planners' preferred moves are always on the menu, so
V\*_menu = V\*_full for both planners at every state **by construction** and the menu artefact on Δ\*
is exactly zero (`test_phase1.py` asserts it; `costlessness_check.py` verifies it per cell). The
heuristic candidates remain as the model's decoys. Numbers: `PHASE1_SCRIPTED_RESULTS.md` and
`results/live_phase1_scripted3_20261029/costlessness_*.json` (sizes `2x2`, `3x2`, `3x2c`).

Consequence for interpretation: the menu never favours asking over offering or the reverse; a
model that fails to ask when the ASK planner would is choosing a visible, lower-value option.

## 6. Phase 1 arms and runner (`arms_live.py`, `jev_policy_live.py`, `run_phase1.py`)

The oracle takes a `menu` hook, so `C_plan` = the exact oracle over the shortlist menu with ASK
and `B_plan` = the same oracle with ASK forbidden; at H ≤ 4 these **are** the practical planners
of design section 5 (no search budget to equalise). Scripted arms: `exact_ask` (C_plan),
`exact_noask` (B_plan), `greedy_eig` (argmax EIG, offers included; ties by immediate value),
`random` (uniform over the menu minus WALK). Jev arms: `jev_ask_plain`, `jev_ask_eig`,
`jev_noask_plain`, `jev_noask_eig` (ASK in the menu or not × EIG shown or hidden), lettered and
shuffled, mock backend offline. Cells = 2 roles × 2 openers × 6 focal types × 6 opponent types =
144, × K repeats (the opponent is stochastic), seeds derived from (seed, cell, repeat). Every
decision logs the menu with exact EIG / P(accept) / own utility, Q\* of every ask-menu option
under C_plan and of every non-ASK option under B_plan, whether ASK is strictly optimal, the choice,
its regrets, and the Jev probabilities and position. `ceilings.json` holds six exact root values
per (role, opener, type): full / shortlist × ASK / no-ASK, offers-restricted-only,
questions-restricted-only. Normalised net gain Y = net / (U_max − r) with U_max = 1 for every type.

Gates (`test_phase1.py`): the menu is a legal, duplicate-free subset with ≤ 2 offers (own-IR,
SAFE first) and ≤ 2 unasked questions, {ACCEPT, WALK} at the deadline; the restricted oracle is
bounded by the full one and Δ\*_menu ≥ 0; the `exact_ask` / `exact_noask` arms attain their
oracles' root values by Monte Carlo within 4 SE and `exact_noask` never asks; the runner's records
are complete, normalised, stub-free, and Jev positions are logged.

## 7. CD-MIS-1 (`opponents_mis.py`, `filters_mis.py`, `run_mis.py`), registered 2026-10-29

Seven true opponent processes (one perturbation axis each, parameters in `PREREGISTRATION_MIS.md`), the
registered T1 as the agent's assumed kernel, the exact oracle under the assumed kernel as the fixed
planner, filters A_prior / B_naive / B_tempered / B_mixture (background floor 10⁻³ for naive and
tempered; mixture over the opponent's legal actions), C arms with ASK, and O_true (true process, exact
filter). Diagnostics separate belief error (posterior on the true type, log-loss, entropy, predictive
log-loss, Brier) from decision loss (true-model decision regret at every focal choice). Confirmatory
contrast M1 = `B_naive − A_prior` under history-contingent countering; rules in `verdicts_mis.py`.
Gates in `test_mis.py`.

## 8. CD-MIS-2 (`opponents_mis.py` severity ids, `filters_mis.py` hedged answer model, `verdicts_mis2.py`), registered 2026-10-30

The severity sweep CD-MIS-1 listed as a limit, plus a confirmatory test of its one exploratory finding
(trusting evasive answers hurts). Process ids `family@value` set the one scalar parameter of a family
(evasive flip rate 0–0.5, history step 0.1–0.4, acceptance noise 0.15–0.5); new ASK arms `C_ans a`
assume a fixed answer flip rate a in both planner and filter, so on the evasive family the grid is the
true-rate × assumed-rate matrix with `O_true` as its diagonal. Confirmatory E1 = `C_ans0.2 − C_naive`
at true rate 0.3; rules in `verdicts_mis2.py`; registration in `PREREGISTRATION_MIS2.md`. Random streams
are seeded by family, so the no-ask arms are identical across evasive levels and each family's CD-MIS-1
level replicates CD-MIS-1 exactly (gated). Gates in `test_mis2.py`.

## 9. What is still not done

Phase 2 heuristic arms (`C_bonus`, `C_greedy` beyond `greedy_eig`) and the general-LLM adapter on the same
menus, Phase 3 (T2 adaptive bilateral play), an adaptive (estimated rather than fixed) answer-noise model,
and the reviewer's read of the planner-completeness fix. The Phase 2 **selector** arms (`A_prior`,
`B_passive`, `P_placebo`, the posterior as a named treatment) are CD-SEL-1 (§10).

## 10. CD-SEL-1 (`jev_policy_live.py` shown-information treatments, `run_phase1.py` arm specs, `verdicts_sel.py`), registered 2026-10-30

Jev's state text has carried the event history since Phase 1; CD-SEL-1 shows it what the history implies.
`JevLivePolicy` gains `show_belief` (the matched posterior over the opponent's six rankings, in words, with
probabilities and the count of rankings ruled out, appended to the state text) and `accept_annotation`
(every PROPOSE option carries "If you propose it now, the seller accepts with probability p": `posterior`
= B_passive, `prior` = A_prior — the same number under the never-updated uniform prior —, `placebo` = the
constant `PLACEBO_P_ACCEPT` = 0.50). The runner computes the numbers (`shown_p_accept`) from the menu's
own filter; the policy only formats what it is handed; every shown number is logged per decision under
`jev.shown`. `run_phase1.JEV_ARMS` is now a spec table (`ask`, `show_eig`, `show_belief`,
`accept_annotation`); new arms `jev_ask_belief`, `jev_ask_pacc`, `jev_ask_prior`, `jev_ask_placebo`,
`jev_ask_both`, `jev_noask_both`; the Phase 1 arms and prompts are unchanged (gated by `test_sel.py`).
`analyze_phase1.py` adds the selector pairs (every selector arm − `jev_ask_plain`; `pacc − prior`,
`both − pacc`, `both − belief`, `prior − placebo`), `both − noask_both` as an ask/no-ask pair with its
captured fraction, and walk-rate contrasts on the same pairing. Primary P1 = `jev_ask_both − jev_ask_plain`
over all cells; rules S1–S9 and the gates in `verdicts_sel.py`; registration in `PREREGISTRATION_SEL.md`;
note rendered by `render_sel_note.py` (reusing the Phase 1 section renderer).

Decisions made here: the primary cell set is all cells (the shown information bears on every decision,
not only the ask-versus-offer choice); the in-study `jev_ask_plain` arm is re-run rather than borrowed
from Phase 1 because `jev-latest` may have drifted (reported as a drift check, not a gate); the posterior
is shown as a ranked list of surviving rankings, never as the planner's values or any recommendation.

Result (2026-10-30, `SEL_RESULTS.md`, FINDINGS §3.10): P1 -0.058 [-0.117, -0.000] — the shown
acceptance probability ends every walk-out and makes Jev propose the SAFE offer almost always; the posterior in words produces the
first selective questions (0.064 per episode) and no net change.
