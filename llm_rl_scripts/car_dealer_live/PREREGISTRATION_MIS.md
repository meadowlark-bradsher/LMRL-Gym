# Pre-registration: CD-MIS-1 — learning with the wrong opponent model

**Registered:** 2026-10-29, before the confirmatory grid was run. **Frozen at tag:** `prereg-mis-20261029` (the commit that adds this
file). **Design:** [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md) §7–8. **No model is involved:**
every arm is scripted; every number is exact-on-model or Monte Carlo over frozen seeds. **Game:** `live-car-dealer-v0.1`, easy regime
(r = 0.40), **H = 4, c_q = 0** (the Phase 1 confirmatory cell).

## What stays fixed (the agent)

- **Assumed kernel:** the registered T1 (α₀ = 0.8, β = 8, p_counter = 1, p_hold = 1, shortlist 3, truthful answers) — the model the
  filter's likelihood and the planner's dynamics both use.
- **Planner:** the exact oracle under the assumed kernel over the full legal action set; ASK forbidden for the A/B arms, allowed for
  the C arms. (At H = 4 the exact oracle is the planner; no search budget to equalise.)
- **Prior:** uniform over the six opponent rankings. **Question bank, costs, protocol:** as in Phase 1.

## What varies

**True opponent process** (`opponents_mis.py`; parameters frozen here; labels are mechanisms, not claims about people):

| id | true process | perturbation |
|---|---|---|
| `matched` | T1 reference | none — sanity check |
| `threshold` | deterministic threshold responder | accepts anything with U ≥ r at any time; always proposes its best remaining deal |
| `stochastic` | noisier acceptance and proposals | rejects an acceptable offer w.p. 0.15; softmax β = 2 over a shortlist of 5 |
| `anchoring` | high anchor, late concession | α₀ = 0.95; aspiration r + (α₀ − r)(remaining/H)³ |
| `withholding` | strategic stalling | never accepts while more than one opportunity remains (counters w.p. 1, else rejects); IR acceptance at the end |
| **`history`** | **history-contingent countering** | **aspiration + 0.1 for every question the focal has asked and every offer it has rejected (cap 1)** |
| `evasive` | noisy answers | each answer flipped w.p. 0.3 |

**Filter / policy arms** (`filters_mis.py`): `A_prior` (no update); `B_naive` (Bayes under the assumed kernel, background floor
η = 10⁻³ so an impossible observation cannot break the filter); `B_temp0.5`, `B_temp0.75` (b′ ∝ b·p_eff^τ); `B_mix0.05`, `B_mix0.2`
(p_eff = (1 − η)p_model + η·p_background, p_background uniform over the opponent's legal actions at that node and over {yes, no});
`C_naive`, `C_mix0.2` (the same beliefs, ASK allowed); `O_true` (plans with the true process and the exact filter under it — the upper
reference, never available to a deployed agent).

**Cells:** 2 focal roles × 2 openers × 6 focal types × 6 opponent types = 144, K = 5 → 720 episodes per (process, arm); 7 × 9 × 720 =
45,360 episodes. Seeds derive from (seed 41, process, cell, repeat), so arms share random streams where the play allows.

```bash
for p in matched threshold stochastic anchoring withholding history evasive; do
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process $p --horizon 4 --ask-cost 0 \
    --arms A_prior,B_naive,B_temp0.5,B_temp0.75,B_mix0.05,B_mix0.2,C_naive,C_mix0.2,O_true --repeats 5 --seed 41 \
    --out results/live_mis_20261029/$p
done
```
(the seven processes may run in parallel; the analyzer reads the merged dir)

## Estimands and decision rules (fixed in advance; applied by `verdicts_mis.py`)

Outcome Y = focal normalized net gain, as in Phase 1. Unit of inference: the **36 private type pairs**; paired differences of type-pair
means with a percentile cluster bootstrap, B = 5000, seed 41; **all cells** (both openers, both focal seats). Vocabulary: a "> 0" or
"< 0" prediction is *confirmed* when the 95 % CI excludes 0 on that side, *opposite* when it excludes 0 on the other side, otherwise
*not supported*; a "≤ 0" prediction (a cost that may be zero) is *confirmed* unless the CI lies entirely above 0.

| | process | quantity | prediction |
|---|---|---|---|
| **M1 (confirmatory)** | **history** | Δ Y, `B_naive − A_prior` | **< 0** (negative learning: updating the wrong model hurts) |
| M2 | withholding | Δ Y, `B_naive − A_prior` | < 0 |
| M3 | matched | Δ Y, `B_naive − A_prior` | > 0 (positive control) |
| M4, M5 | matched | Δ Y, `B_mix0.2 − B_naive`; `B_temp0.5 − B_naive` | ≤ 0 (matched-model cost of robustness; its size is reported) |
| M6, M7 | history | Δ Y, `B_mix0.2 − B_naive`; `B_temp0.5 − B_naive` | > 0 (robust filtering recovers value) |
| M8 | history | Δ Y, `O_true − B_naive` | > 0 (cost of misspecification) |
| M9 | history | Δ Y, `C_naive − B_naive` | > 0 (does asking help when the model is wrong) |
| M10 | history | Δ final posterior log-loss on the true type (bits), `B_naive − A_prior` | > 0 (belief error) |
| M11 | history | Δ true-model decision regret, `B_naive − A_prior` | > 0 (decision loss) |
| G_* | every mismatch process | Δ Y, `B_naive − A_prior` | the full grid, reported without selection |
| descriptive | per (process, arm): mean Y, agreement, walk rate, asks, final posterior on the true type and its log-loss, entropy, argmax-true share, mass on the best wrong type, the exact true filter's posterior, predictive log-loss of observed responses, Brier of predicted acceptance, true-model decision regret; cross-process table `process − matched` per arm | reported |

M10 and M11 together separate **belief error** from **decision loss** (design §7.2): a confidently wrong posterior matters only insofar
as it changes the decision. One confirmatory contrast; the rest secondary, no multiplicity correction.

## Sanity gates (fail → fix the pipeline, do not read results)

- No error stubs; every (process, arm) has 720 episodes.
- Under `matched`, `O_true − C_naive` is not positive (the naive filter is then correctly specified up to the 10⁻³ floor).
  **Post-run correction (2026-10-29):** this gate was registered as `O_true − B_naive`, which compares an ASK-allowed planner with
  an ASK-forbidden one and is positive by exactly the value of asking (it read +0.066 [+0.048, +0.085]); the intended check is the
  ASK-allowed pair. The gate is a pipeline check, not a hypothesis; no decision rule (M1–M11, G_*) was changed after the run.
- `test_mis.py`: every process is a valid kernel at every reachable node; filters stay normalised and survive impossible observations;
  under `matched` the naive filter tracks the exact filter and `O_true` equals the assumed ASK planner.

## Why `history` is the confirmatory condition, and one disclosure

The design asks for a preselected history-dependent counteroffer or withholding condition. `history` is the mechanism the filter omits
by construction (conditional dependence on the focal's own acts), so it is the cleaner test of "learning the wrong model"; `withholding`
is the named secondary. **Disclosure:** a 48-episode timing probe (buyer, one focal type, K = 1, **H = 3**, arms A_prior / B_naive /
C_naive / O_true) was run on the `history` process before this registration to size the grid; in it `B_naive` scored above `A_prior`.
It is not a pilot of the registered cell (H = 4, full census) and `history` is kept as the confirmatory condition regardless.

## Predictions, stated plainly

Under `matched`, learning helps (M3) and robustness costs a little (M4, M5 small negative or zero). Under `history`, the naive filter
reads the opponent's stiffening as type evidence and concentrates on the wrong type (M10 positive); whether that costs utility (M1)
is the open question — the timing probe suggests it may not at H = 3, and a null M1 is informative: it would mean a wrong model that
still ranks offers sensibly is harmless here. `evasive` is the condition most likely to show real harm (a flipped answer eliminates
the true type under a near-truthful likelihood), which is why it is in the grid and not the confirmatory cell.

## Analysis (fixed in advance)

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.merge_mis --dir results/live_mis_20261029   # per-process dirs -> one
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir results/live_mis_20261029 --B 5000 --seed 41
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_mis --analysis results/live_mis_20261029/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis_note --dir results/live_mis_20261029 \
    --verdicts results/live_mis_20261029/verdicts.json --out llm_rl_scripts/car_dealer_live/MIS_RESULTS.md
```

## Post-run disclosure (2026-10-30): the grid was regenerated after a determinism fix

CD-MIS-2's exact-replication gate (its CD-MIS-1 levels must reproduce this grid's mean Y to 10⁻¹²) failed on its first run and
exposed a pipeline defect that predates this study: `belief_live.action_likelihoods` built the opponent-action table from a Python
set, so the oracle's summation order — and therefore the break of *exact ties* between two questions at the uniform prior — depended
on the interpreter's hash seed. The no-ask arms were unaffected; the ASK arms chose a different (equally optimal) question in some
episodes, which changed the sampled answers and the later play. The table is now ordered by action key, `test_mis2.py` pins the
behaviour across hash seeds, `REGENERATE.sh` also exports `PYTHONHASHSEED=0`, and this grid was regenerated with the fixed code. The
first run's `analysis.json` is kept beside it as `analysis_first_run_before_determinism_fix.json`. Effect: 12 of the 63
(process, arm) mean-Y values moved (largest contrast move 0.007 in Y; e.g. `C_naive − B_naive` under `evasive` −0.094 → −0.087);
**no verdict changed**, no rule changed, and `MIS_RESULTS.md` and FINDINGS §3.8 were re-rendered from the regenerated JSON.

## Limits accepted in advance

One assumed kernel and one perturbation per axis with one parameter value each (no severity sweep); the exact oracle as planner; easy
regime; H = 4 only; the 36 type pairs are the whole type space; `A_prior` plans with the uniform prior at every decision, which is the
design's prior-only forward model, not a tuned heuristic.
