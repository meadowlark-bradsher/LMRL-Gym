# Live Car-Dealer, Phase 0 — design decisions (`live-car-dealer-v0.1`)

**Source:** [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md)
(`CD-VOI-1` / `CD-MIS-1`, copied verbatim from the author's document, 2026-10-27).
**Scope of this note:** Phase 0 only — implementation and oracle readiness, no research claims,
no registration, no Jev calls. Numbers live in [`PHASE0_RESULTS.md`](PHASE0_RESULTS.md),
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

## 4. What Phase 0 does *not* do

No registration, no Jev or LLM call, no `B_plan`/`C_plan` practical planners (at H ≤ 4 the exact
oracle *is* the planner, so `C_plan − B_plan` collapses to Δ\*_active), no T2, no mismatch grid.
Those are Phases 1–3 and CD-MIS-1.
