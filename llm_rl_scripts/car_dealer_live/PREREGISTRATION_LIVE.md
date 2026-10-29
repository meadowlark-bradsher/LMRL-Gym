# Pre-registration: Jev on the live Car-Dealer, CD-VOI-1 Phase 1

**Registered:** 2026-10-29, before any Jev decision was collected on this game. **Frozen at tag:** `prereg-live-20261029` (the commit that adds this file)
(this file, `verdicts_live.py`, `menu_live.py`, `run_phase1.py`, `analyze_phase1.py` as of that commit).
**Game:** `live-car-dealer-v0.1` · **Model:** `jev-latest` (HTTP backend, `TYPESAFE_API_KEY`) · **Regime:** easy (r = 0.40) ·
**Opponent:** T1 kernel α₀ = 0.8, β = 8, p_counter = 1, p_hold = 1, shortlist 3, truthful answers, no noise.
**Design:** [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md) §5–6, §8; decisions in
[`DESIGN_LIVE.md`](DESIGN_LIVE.md) §4–6. **Scripted baselines frozen:** [`PHASE1_SCRIPTED_RESULTS.md`](PHASE1_SCRIPTED_RESULTS.md)
(`results/live_phase1_scripted3_20261029/`).

## Provenance of the choices (recorded as the reviewer asked)

- The kernel and the confirmatory cell were chosen **after Phase 0** (`PHASE0_RESULTS.md`, the aggregate with the largest exact
  Δ\*_active) and **before any model call**.
- **Three offer candidates**, each by a written rule (SAFE, AMBITIOUS, PLAN; `DESIGN_LIVE.md` §5), recorded as a **deviation from
  Condition B's two**, made because the costlessness check showed two candidates cost the no-ask planner 0.015 utility in every
  focal-opens cell and no EIG / P(accept) / utility rule reproduces the planner's opening offer.
- **Planner completeness** (PLAN_ASK, PLAN_Q): added after the reviewer's stopping rule fired for the H = 3 secondary regime under
  the three-candidate menu (question restriction cost the ASK planner 0.015–0.018 utility; Δ\*_menu = 0 in its focal-opens
  cells). With completeness both planners' preferred moves are always on the menu, so V\*_menu = V\*_full for both by construction.
- **Stopping rule (reviewer):** a registered-menu shift of mean Δ\* above 0.005 in any cell set stops the registration.
  Evaluated by `costlessness_check.py` on `results/live_phase1_scripted3_20261029/costlessness_<regime>.json`
  (mean Δ\*_menu − Δ\*_full, utility units, all cells / focal-opens cells):

  | regime | three candidates, heuristic only (`3x2`) | registered menu (`3x2c`, planner-complete) | cells lossless (`3x2c`) | rule on the registered menu |
  |---|---|---|---|---|
  | H = 4, c_q = 0 (primary) | -0.0000 / -0.0000 | -0.0000 / -0.0000 | 24/24 | **PROCEED** |
  | H = 2, c_q = 0 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 24/24 | **PROCEED** |
  | H = 3, c_q = 0 | -0.0148 / -0.0184 | +0.0000 / +0.0000 | 24/24 | **PROCEED** |
  | H = 4, c_q = 3/100 | -0.0012 / -0.0023 | +0.0000 / +0.0000 | 24/24 | **PROCEED** |

  The heuristic-only three-candidate menu exceeded the threshold at H = 3 (the question-side loss), which is why
  planner completeness was added before registration; on the registered menu the artefact is zero in every regime.
- The belief the menu is built from is the matched finite filter over the opponent's six rankings, updated on the opponent's
  answers and moves only, maintained outside the model. **The claim is scoped to the ask-versus-offer decision with belief
  maintained outside the model.**

## Conditions

Shortlist menu (`DESIGN_LIVE.md` §5), lettered and shuffled; Jev sees no tags, no P(accept), no posterior; own utility of a deal
is shown. Four Jev arms differ only in whether questions are on the menu and whether each option carries its exact EIG in bits:

| arm | questions on the menu | EIG shown |
|---|---|---|
| `jev_ask_plain` | yes | no |
| `jev_ask_eig` | yes | yes |
| `jev_noask_plain` | no | no |
| `jev_noask_eig` | no | yes |

Scripted arms in every run (same seeds, same cells): `exact_ask` (C_plan), `exact_noask` (B_plan), `greedy_eig`, `random`.
Named secondary arm: the **full 27-deal menu** (`--menu full`), `jev_ask_plain` / `jev_noask_plain` only, primary regime.

Cells: 2 focal roles × 2 openers × 6 focal types × 6 opponent types = 144, K = 5 repeats → 720 episodes per arm per regime.
Seeds derive each episode's randomness from (seed, cell, repeat); `--seed` is provenance only for `jev-latest` (stochastic
server-side) and fixes the option shuffle.

| run | H | c_q | role | Jev arms | seed | out |
|---|---|---|---|---|---|---|
| **primary** | 4 | 0 | confirmatory | all four | 21 | `results/live_phase1_jev_h4_cq0_20261029` |
| secondary | 2 | 0 | negative control (ASK never optimal) | `jev_ask_plain`, `jev_noask_plain` | 22 | `results/live_phase1_jev_h2_cq0_20261029` |
| secondary | 3 | 0 | positive control (Δ\* > 0 for all eight kernels) | `jev_ask_plain`, `jev_noask_plain` | 23 | `results/live_phase1_jev_h3_cq0_20261029` |
| secondary | 4 | 3/100 | question cost | `jev_ask_plain`, `jev_noask_plain` | 24 | `results/live_phase1_jev_h4_cq3_20261029` |
| secondary | 4 | 0 | full 27-deal menu | `jev_ask_plain`, `jev_noask_plain` | 25 | `results/live_phase1_jev_h4_cq0_full_20261029` |

```bash
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 0 --n-offers 3 \
  --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_ask_eig,jev_noask_plain,jev_noask_eig \
  --repeats 5 --seed 21 --jev-backend http --out results/live_phase1_jev_h4_cq0_20261029
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 2 --ask-cost 0 --n-offers 3 \
  --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_noask_plain --repeats 5 --seed 22 --jev-backend http \
  --out results/live_phase1_jev_h2_cq0_20261029
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 3 --ask-cost 0 --n-offers 3 \
  --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_noask_plain --repeats 5 --seed 23 --jev-backend http \
  --out results/live_phase1_jev_h3_cq0_20261029
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 3/100 --n-offers 3 \
  --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_noask_plain --repeats 5 --seed 24 --jev-backend http \
  --out results/live_phase1_jev_h4_cq3_20261029
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 0 --menu full \
  --arms exact_ask,exact_noask,greedy_eig,random,jev_ask_plain,jev_noask_plain --repeats 5 --seed 25 --jev-backend http \
  --out results/live_phase1_jev_h4_cq0_full_20261029
```

## Estimands and decision rules (fixed in advance; applied by `verdicts_live.py`)

Outcome Y = focal normalized net gain = (U − r − c_q · own asks) / (U_max − r), U_max = 1 for every type; no deal → Y = 0.
Unit of inference: the **36 private type pairs** (design §8); every Δ is a paired difference of type-pair means with a percentile
cluster bootstrap, **B = 5000**, seed = the run's `--seed`. Cell sets: `focal_opens` (**primary**, because the opponent-opens
exact ceiling ≈ 0.015 utility lies below δ), `all`, `opponent_opens`. Materiality **δ = 0.02** normalized-gain units (design §6,
provisional by design).

Verdict vocabulary: *materially positive* = 95 % CI above 0 and lower bound > δ; *positive* = CI above 0; *negative* = CI below 0;
*unresolved* = CI includes 0. For "> 0" / "< 0" predictions: *confirmed* / *opposite* / *not supported*; for "not > 0":
*confirmed* if the CI includes 0 or lies below it.

| | quantity | prediction |
|---|---|---|
| **P1 (primary)** | Δ Y, `jev_ask_plain − jev_noask_plain`, focal-opens, H = 4, c_q = 0 | > 0; material if lower bound > δ. Reported with the **fraction of the exact ceiling captured**: Δ\*_menu = Δ\*_full (identical by construction) in Y units, no CI on the ceiling |
| S1 | the same, all cells | > 0 |
| S2 | Δ Y, `jev_ask_eig − jev_noask_eig`, focal-opens | > 0 |
| S3 (VOI-H4) | Δ questions / episode, `jev_ask_eig − jev_ask_plain`, all cells | > 0 (EIG shown → more asking) |
| S4 (VOI-H4) | Δ Y, `jev_ask_eig − jev_ask_plain`, focal-opens | **not** > 0 (EIG shown does not raise net gain) |
| S5 (selectivity) | P(ask \| ASK strictly optimal) − P(ask \| ASK legal, not optimal), `jev_ask_plain` and `jev_ask_eig`, primary | > 0 |
| S6 | Δ Y, `jev_ask_plain − exact_ask`, focal-opens | < 0 (the gap to the planner; its size is the headline) |
| S7 | Δ Y, `jev_ask_plain − jev_noask_plain`, all cells, **H = 2** | not > 0; Jev's ask rate reported (every question is strictly suboptimal there) |
| S8 | Δ Y, `jev_ask_plain − jev_noask_plain`, all cells, **H = 3** | > 0 |
| S9 | questions / episode, `jev_ask_plain`, **c_q = 3/100 vs c_q = 0** | lower, with disjoint CIs |
| S10 | Δ Y, `jev_ask_plain − jev_noask_plain`, focal-opens, c_q = 3/100 | > 0 |
| S11 | Δ Y, `jev_ask_plain − jev_noask_plain`, focal-opens, **full 27-deal menu** | > 0; captured fraction of Δ\*_full |
| descriptive | chosen-position distribution, well-formed share, confidence, deal distribution, candidate take-up by tag, Jev decision regret and EIG regret per decision, residual information-bias fit on the logged Q\* / EIG | reported, no rule |

One primary, no multiplicity correction on the secondaries; a secondary CI touching 0 is not a failure of the program.

## Sanity gates (fail → fix the pipeline, do not read results)

- `well_formed_share` = 1.0 for every Jev arm; no error stubs; every arm has exactly 720 episodes per regime.
- The option shuffle is active (letter → position map not degenerate).
- The in-run `random` arm's mean Y lies inside the frozen scripted `random` arm's 95 % CI for the same regime
  (`results/live_phase1_scripted3_20261029/<regime>/analysis.json`).
- `exact_ask − exact_noask` in the Jev run reproduces Δ\*_menu within its CI, as in the frozen baselines.

## Predictions, stated plainly

From Condition B: Jev reads the numbers it is given and asks more when EIG is shown (S3), without lowering decision regret (S4).
From Phase 0: asking has real value only at H = 3–4. Prediction: P1 **positive but below δ** (Jev asks sometimes, not selectively);
S5 small positive; S6 confirmed with a gap of the order of the Condition B optimality gap (~0.05 utility); S7 confirmed; S9
confirmed if Jev reads the stated cost. Honest alternative: Jev's choice is near-uniform over the lettered menu, P1 is unresolved
and S5 is null, in which case the record says so and Jev is reported as chance-level on the ask-versus-offer decision.

## Analysis (fixed in advance)

```bash
for d in results/live_phase1_jev_h4_cq0_20261029 results/live_phase1_jev_h2_cq0_20261029 results/live_phase1_jev_h3_cq0_20261029 \
         results/live_phase1_jev_h4_cq3_20261029 results/live_phase1_jev_h4_cq0_full_20261029; do
  PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir $d --B 5000 --seed 21; done
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_live \
  --primary results/live_phase1_jev_h4_cq0_20261029/analysis.json --h2 results/live_phase1_jev_h2_cq0_20261029/analysis.json \
  --h3 results/live_phase1_jev_h3_cq0_20261029/analysis.json --cq3 results/live_phase1_jev_h4_cq3_20261029/analysis.json \
  --full results/live_phase1_jev_h4_cq0_full_20261029/analysis.json --scripted results/live_phase1_scripted3_20261029/h4_cq0/analysis.json
```

Verdicts applied by code and frozen before reading. No rule changes after data. One held-out replicate (a fresh K = 1 pass of the
primary cell, seed 31) is run after the main analysis to check reproducibility of the P1 sign.

## Limits accepted in advance

One kernel (the firmest in the Phase 0 sweep); exact oracle as the planner (no search-budget fairness question at H ≤ 4);
one question bank; easy regime only; zero noise; `jev-latest` pinned by name only; the 36 type pairs are the whole type space,
so cluster intervals have a precision ceiling that more K cannot lower; the menu contains the planners' preferred moves by
construction, so the test is whether the model recognises them, not whether it could generate them.
