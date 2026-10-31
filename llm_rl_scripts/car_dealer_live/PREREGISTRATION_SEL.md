# Pre-registration: CD-SEL-1 — Jev shown what the history implies (live Car-Dealer, design Phase 2 selector arms)

**Registered:** 2026-10-30, before any Jev decision of this study was collected. **Frozen at tag:** `prereg-sel-20261030` (the commit
that adds this file; `jev_policy_live.py`, `run_phase1.py`, `analyze_phase1.py`, `verdicts_sel.py`, `render_sel_note.py`, `test_sel.py`
as of that commit). **Game:** `live-car-dealer-v0.1` · **Model:** `jev-latest` (HTTP backend, `TYPESAFE_API_KEY`) · **Regime:** the Phase 1
primary cell — easy (r = 0.40), H = 4, c_q = 0 · **Opponent:** T1 kernel α₀ = 0.8, β = 8, p_counter = 1, p_hold = 1, shortlist 3, truthful
answers, no noise · **Menu:** the registered planner-complete shortlist (`DESIGN_LIVE.md` §5; `PREREGISTRATION_LIVE.md`).
**Design:** [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md) §5 (the `A_prior` / `B_passive` /
`P_placebo` selector arms and the rule "do not present the posterior itself unless that is a separately named treatment") and §8 Phase 2.

## Why this study

Phase 1 (`JEV_RESULTS_LIVE.md`, registered at `prereg-live-20261029`) found that Jev never asks a question (0 of 2,555 eligible
decisions), walks out of 33 % of negotiations, and finishes 0.166 Y [0.139, 0.193] below the exact planner over all cells
(0.277 Y focal-opens), even though the planner's preferred move is on its menu at every decision by construction. Jev's state text
has carried the full event history since Phase 1 — every proposal, answer, rejection and hold, in words. What it has never been shown is
what that history **implies**: the matched posterior over the opponent's six priority rankings, and the posterior probability that each
offer on the menu is accepted. The Text-Nav program found that Jev composes multi-step behaviour only when handed its history
explicitly, and the 20-Questions adapter found that Jev selects well only when the belief is carried for it. This study asks whether the
same holds here: **does a one-step selector perform in the live game once the inference from history is placed in front of it?**

## Conditions

All Jev arms see the same lettered, shuffled shortlist menu as Phase 1 (own utility of a deal shown; no tags, no Q\*, no EIG). The
treatments add shown information, each a separately named arm (`run_phase1.JEV_ARMS`, `jev_policy_live.py`):

| arm | questions on the menu | shown in the state text | shown on every PROPOSE option | design name |
|---|---|---|---|---|
| `jev_ask_plain` | yes | event history (as in Phase 1) | — | Phase 1 control, re-run in-study |
| `jev_ask_belief` | yes | history + **the posterior over the opponent's rankings, in words** (support with probabilities; rankings ruled out counted) | — | posterior as a named treatment |
| `jev_ask_pacc` | yes | history | **posterior P(accept)** | `B_passive` selector |
| `jev_ask_prior` | yes | history | P(accept) under the **never-updated uniform prior** | `A_prior` selector |
| `jev_ask_placebo` | yes | history | the **constant 0.50** | `P_placebo` |
| `jev_ask_both` | yes | history + posterior in words | posterior P(accept) | full passive information |
| `jev_noask_both` | no | history + posterior in words | posterior P(accept) | ASK-value control for `both` |

Wording (frozen in `jev_policy_live.py`): the instructions gain one sentence per treatment ("You are also told what the other party's
answers and moves so far imply about its priority ranking." / "Each proposal also shows the estimated probability that the other party
accepts it."); the belief line reads "What the seller's answers and moves so far imply about its priority ranking (most to least
important), with probabilities: warranty > delivery > price (0.50); delivery > warranty > price (0.50). The other 4 rankings are ruled
out." (uniform: "Nothing the seller has said or done so far distinguishes its 6 possible priority rankings; each has probability
0.17."); each offer gains "If you propose it now, the seller accepts with probability 0.50." The `prior`, `pacc` and `placebo` arms differ
**only in the printed number**. The numbers are computed by the runner (`run_phase1.shown_p_accept`) from the same matched filter that
builds the menu; the policy formats what it is handed, and every shown number is logged per decision (`jev.shown`).

Scripted arms in the same run and cells: `exact_ask` (C_plan), `exact_noask` (B_plan), `random` (gates). Cells: 2 focal roles × 2 openers
× 6 focal types × 6 opponent types = 144, K = 5 repeats → 720 episodes per arm, 7,200 episodes, about 10,400 Jev calls (Phase 1 averaged
2.06 Jev decisions per episode). Seed 51 (provenance for the stochastic server-side model; fixes the option shuffle).

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 0 --n-offers 3 \
    --arms exact_ask,exact_noask,random,jev_ask_plain,jev_ask_belief,jev_ask_pacc,jev_ask_prior,jev_ask_placebo,jev_ask_both,jev_noask_both \
    --repeats 5 --seed 51 --jev-backend http --out results/live_sel_jev_h4_cq0_20261030
```

## Outcome, clusters, verdicts

Y = focal normalized net gain, as in Phase 1. Contrasts are paired over the 36 (buyer ranking, seller ranking) type-pair clusters with a
percentile cluster bootstrap, B = 5000, seed 51 (`analyze_phase1.py`). Walk-rate contrasts use the same pairing on the indicator "the
episode ended with a walk-out". Verdict vocabulary as in `verdicts_live.py`; materiality δ = 0.02 Y. **Primary cell set: all cells** —
the shown information bears on every decision (which offer, whether to accept, whether to walk), not only on the ask-versus-offer choice
that made focal-opens the Phase 1 primary; focal-opens is reported for every contrast.

Rules fixed in `verdicts_sel.py`:

| | quantity | prediction |
|---|---|---|
| **P1 (primary)** | Δ Y, `jev_ask_both − jev_ask_plain`, all cells | > 0; material if lower bound > δ. Reported with the share of the plain arm's gap to `exact_ask` that is closed, (both − plain) / (exact_ask − plain), as a point estimate |
| S1 | Δ Y, `jev_ask_belief − jev_ask_plain`, all cells | > 0 (the posterior in words alone) |
| S2 | Δ Y, `jev_ask_pacc − jev_ask_plain`, all cells | > 0 (posterior P(accept) alone; `B_passive`) |
| S3 | Δ Y, `jev_ask_pacc − jev_ask_prior`, all cells | > 0 (passive learning in the selector: `B_passive − A_prior`) |
| S4 | Δ Y, `jev_ask_placebo − jev_ask_plain`, all cells | = 0 (CI includes 0): a constant annotation changes nothing |
| S5 / S5b | Δ questions per episode, `jev_ask_belief − jev_ask_plain` and `jev_ask_both − jev_ask_plain`, all cells | > 0 (seeing the residual uncertainty prompts asking) |
| S6 | Δ Y, `jev_ask_both − jev_noask_both`, focal-opens | > 0 (the value of questions on the menu once the belief is shown); reported with the fraction of Δ\*_menu captured |
| S7 | Δ Y, `jev_ask_both − exact_ask`, all cells | < 0 (the remaining gap) |
| S8 / S8b | Δ walk rate, `jev_ask_pacc − jev_ask_plain` and `jev_ask_both − jev_ask_plain`, all cells | < 0 (acceptance numbers prevent the rejected-ambitious-offer walk-outs) |
| S9 | selectivity P(ask \| ASK optimal) − P(ask \| ASK legal, not optimal), `jev_ask_belief` and `jev_ask_both` | > 0 |
| descriptive | `prior − placebo`, `both − pacc`, `both − belief` (all cells), per-arm mean Y / walk rate / agreement / offer tags / questions, chosen position, confidence | reported, no rule |

One primary, no multiplicity correction on the secondaries.

## Sanity gates (fail → fix the pipeline, do not read results)

- No error stubs; every arm has the same number of episodes (720); `well_formed_share` = 1.0 for every Jev arm; all ten arms present.
- The option shuffle is active (every Jev arm uses at least two positions).
- The in-run `random` arm's mean Y lies inside the frozen scripted `random` arm's 95 % CI for this regime
  (`results/live_phase1_scripted3_20261029/h4_cq0/analysis.json`: [−0.098, +0.073]).
- `exact_ask − exact_noask` (all cells) reproduces the exact Δ\*_menu in Y units (0.064) within its CI.

**Drift check (reported, not a gate):** the in-study `jev_ask_plain` mean Y against the frozen Phase 1 run's 0.465 [0.373, 0.558], with
walk rate (Phase 1: 0.328) and questions per episode (Phase 1: 0.000). `jev-latest` is stochastic and pinned by name only; a shift here
is model drift, disclosed with the results, and is why the plain arm is re-run inside the study rather than borrowed from Phase 1.

## Predictions, stated plainly

- **P1 positive, probably material.** The Phase 1 failure mode is concrete: 55 % of Jev's offers were AMBITIOUS, the opponent rejected
  them, and a third of negotiations ended in a walk-out. A printed acceptance probability addresses exactly that, so S2 and S8 are the
  mechanism predictions and P1 should inherit them. The honest alternative: Jev treats the number as decoration and S4, S2 and P1 are all
  unresolved — then the record says a one-step selector does not use passive belief information either, and the Car-Dealer gap is not
  closable by presentation.
- **S1 uncertain.** The posterior in words tells Jev which rankings remain, but turning that into an offer still needs the step the
  planner does; registered as "> 0" with a null outcome as the plausible alternative.
- **S5 is the open question of the program.** Jev has never asked. If seeing "two rankings remain, equally likely" makes it ask, that is
  the first evidence of inquiry from the model; the Phase 1 EIG-shown arm argues it will not.
- **S6 likely unresolved**: even the exact planner's ASK value is 0.102 Y focal-opens, and the shown belief does not change it.
- **S7 confirmed**: no presentation makes a one-step selector a planner.

## Disclosures made at registration

- A pipeline smoke of **one** live episode (`jev_ask_both`, one cell, a handful of calls) was run against the HTTP backend before the
  freeze to confirm the longer prompt is accepted; its records are not part of the study and are not kept in `results/`.
- The in-study arms share random streams with each other but not with Phase 1 (seed 51 vs 21); Jev's choices are stochastic
  server-side in any case.
- `PLACEBO_P_ACCEPT` = 0.50 is a false statement about the world on most offers; it is a salience control, not a model of anything.

## Analysis (fixed in advance)

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/live_sel_jev_h4_cq0_20261030 --B 5000 --seed 51
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_sel --analysis results/live_sel_jev_h4_cq0_20261030/analysis.json \
    --scripted results/live_phase1_scripted3_20261029/h4_cq0/analysis.json --phase1 results/live_phase1_jev_h4_cq0_20261029/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_sel_note --dir results/live_sel_jev_h4_cq0_20261030 --out llm_rl_scripts/car_dealer_live/SEL_RESULTS.md
```

Verdicts applied by code and frozen before reading. No rule changes after data; any pipeline correction after the run is disclosed in a
post-run section of this file.

## Post-run status (2026-10-30)

The registered run completed (7200 episodes, 0 error stubs, 1020.45 s); all four gates pass; the drift check
places the in-study `jev_ask_plain` arm inside the Phase 1 plain arm's interval (0.458 vs 0.465
[0.373, 0.558]). **No pipeline correction was made after the run; no rule changed.** The primary
came out on the opposite side of its registered prediction (P1 -0.0579 [-0.1174, -0.0003], verdict
*negative*); the results note is `SEL_RESULTS.md` and the reading is in `llm_rl_scripts/jev/FINDINGS.md` §3.10.

## Limits accepted in advance

One regime and one kernel; the belief shown is the **matched** filter (the opponent model is correct by construction, so this is the
best case for shown information — CD-MIS-1/2 cover what happens when it is wrong); the posterior is shown as a list of rankings, not
as the planner's values, so the test is whether the model can act on a belief, not whether it can be told what to do; `jev-latest` pinned
by name only; 36 type pairs are the whole type space, so cluster intervals have a precision ceiling.
