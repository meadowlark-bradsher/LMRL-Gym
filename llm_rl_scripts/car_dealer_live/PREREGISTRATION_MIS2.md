# Pre-registration: CD-MIS-2 — severity sweep, and the evasive-answer channel

**Registered:** 2026-10-30, before the confirmatory grid was run. **Frozen at tag:** `prereg-mis2-20261030` (the commit that adds this
file). **Follows:** CD-MIS-1 ([`PREREGISTRATION_MIS.md`](PREREGISTRATION_MIS.md), tag `prereg-mis-20261029`; results
[`MIS_RESULTS.md`](MIS_RESULTS.md)). **Design source:** [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md)
§7–8; this study is not in the design document — it is the severity sweep the CD-MIS-1 registration listed as a limit, plus a
confirmatory test of CD-MIS-1's one exploratory finding. **No model is involved.** **Game:** `live-car-dealer-v0.1`, easy regime
(r = 0.40), **H = 4, c_q = 0** (the Phase 1 / CD-MIS-1 cell).

## What CD-MIS-1 left open

CD-MIS-1 refuted negative learning: Bayesian updating of a wrong behavioural model helped under every mismatch but withholding, the
robust filters never changed a decision, and asking insured the planner against every wrong concession rule. The one harm was in the
**question channel**: with answers flipped w.p. 0.3, the ASK-allowed learner that trusted them (`C_naive`) fell below the no-ask
learner (`C_naive − B_naive` = −0.094 [−0.119, −0.067]) while the planner that knew the flip rate (`O_true`) nearly stopped asking and
matched the no-ask learner (`O_true − B_naive` = −0.002 [−0.009, +0.007]). That contrast was exploratory (the registered M9 was the
history-process version). Two questions follow, and both need a sweep:

1. **Is the evasive-answer harm a modelling error or an information loss?** If a planner that merely *hedges* its answer model with a
   fixed assumed flip rate — not knowing the true one — recovers the loss, the harm is a modelling error a deployed agent can insure
   against. If only the exact rate recovers it, it is not.
2. **Do the CD-MIS-1 nulls survive harsher perturbation?** "Learning the wrong model does not hurt", "robust filters are inert" and
   "asking insures against the behavioural model" were each shown at one parameter value per axis.

## What stays fixed (the agent)

As in CD-MIS-1: assumed behavioural kernel = the registered T1 (α₀ = 0.8, β = 8, p_counter = 1, p_hold = 1, shortlist 3); planner =
the exact oracle under the assumed kernel over the full legal action set (ASK forbidden for A/B arms, allowed for C arms); uniform
prior; Phase 1 question bank, costs, protocol. **New:** the `C_ans a` arms assume an answer flip rate a in *both* the planner (so the
expected information gain of a question is discounted) and the filter (`filters_mis.py` `B_ans a`: p_model = (1 − a)·p_truthful +
a·(1 − p_truthful), then the 10⁻³ floor). Their behavioural kernel is still the registered T1.

## What varies

**Severity levels** (`opponents_mis.py`, process id `family@value`; a bare family name is the CD-MIS-1 level):

| family | severity parameter | levels (CD-MIS-1 level in bold) |
|---|---|---|
| `evasive` | P(answer flipped) | 0, 0.1, 0.2, **0.3**, 0.4, 0.5 |
| `history` | aspiration step per focal question or rejected offer (cap 1) | **0.1**, 0.2, 0.4 |
| `stochastic` | P(reject an acceptable offer) (β = 2, shortlist 5 kept) | **0.15**, 0.3, 0.5 |

`anchoring` is left out: its α₀ is capped at 1 and the concession power has no monotone "more wrong" direction; `threshold`,
`withholding` and `matched` have no scalar parameter. `evasive@0` is the matched process (a sanity level of the sweep).

**Arms.** Evasive levels: `A_prior`, `B_naive`, `C_naive`, `C_mix0.2`, `C_ans0.1`, `C_ans0.2`, `C_ans0.3`, `O_true` (8). Behavioural
levels: `A_prior`, `B_naive`, `B_mix0.2`, `C_naive`, `O_true` (5). On the evasive family `O_true` is the planner that assumes the true
flip rate, so the grid is the **true rate × assumed rate** matrix with `O_true` as its diagonal; at 0.3, `C_ans0.3` *is* `O_true`
(same kernel; its 10⁻³ floor can only break exact ties differently).

**Cells:** 2 focal roles × 2 openers × 6 focal types × 6 opponent types = 144, K = 5 → 720 episodes per (process, arm).
6 × 8 × 720 + 6 × 5 × 720 = **56,160 episodes**. Random streams are seeded by (seed 41, process **family**, cell, repeat): the
no-ask arms are identical across evasive levels (they never see an answer), and the CD-MIS-1 level of each family replicates the
CD-MIS-1 episodes exactly for the arms both studies share.

```bash
OUT=results/live_mis2_20261030
for p in evasive@0 evasive@0.1 evasive@0.2 evasive@0.3 evasive@0.4 evasive@0.5; do
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process $p --horizon 4 --ask-cost 0 --design CD-MIS-2 \
    --arms A_prior,B_naive,C_naive,C_mix0.2,C_ans0.1,C_ans0.2,C_ans0.3,O_true --repeats 5 --seed 41 --out $OUT/$p &
done
for p in history@0.1 history@0.2 history@0.4 stochastic@0.15 stochastic@0.3 stochastic@0.5; do
  PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process $p --horizon 4 --ask-cost 0 --design CD-MIS-2 \
    --arms A_prior,B_naive,B_mix0.2,C_naive,O_true --repeats 5 --seed 41 --out $OUT/$p &
done
wait
```

## Estimands and decision rules (fixed in advance; applied by `verdicts_mis2.py`)

Outcome Y = focal normalized net gain. Unit of inference: the **36 private type pairs**; paired differences of type-pair means with
a percentile cluster bootstrap, B = 5000, seed 41; **all cells**. Vocabulary: "> 0" / "< 0" → *confirmed* when the 95 % CI excludes
0 on that side, *opposite* on the other side, else *not supported*; "≤ 0" / "≥ 0" (a cost or gain that may be zero) → *confirmed*
unless the CI lies entirely on the wrong side; "= 0" → *confirmed* when the CI includes 0, with the share of seed-paired episodes in
which the two arms took identical trajectories reported beside it (a decision-level reading of "same choices").

| | process | quantity (Δ Y, all cells) | prediction |
|---|---|---|---|
| **E1 (confirmatory)** | **evasive@0.3** | **`C_ans0.2 − C_naive`** | **> 0** — a *fixed* hedge (assumed 0.2, true 0.3) recovers value lost by trusting evasive answers |
| E2 | evasive@0 | `C_ans0.2 − C_naive` | ≤ 0 — the premium paid for hedging when answers are truthful; its size is reported |
| E3a, E3b | evasive@0; evasive@0.5 | `C_naive − B_naive` | > 0 at 0; < 0 at 0.5 — the dose-response changes sign; the level at which it does is reported |
| E4 | evasive@0.1 | `O_true − B_naive` | > 0 — questions keep value under mild noise when the answer model is right |
| E5 | evasive@0.5 | `O_true − B_naive` | = 0 — pure-noise answers: the calibrated planner stops asking |
| E6 | evasive@0.3 | `C_ans0.2 − B_naive` | ≥ 0 — the hedged ASK arm is not worse than never asking |
| S1, S2 | history@0.4; stochastic@0.5 | `B_naive − A_prior` | ≥ 0 — learning the wrong model does not hurt at the harshest level |
| S3a, S3b | history@0.4; stochastic@0.5 | `B_mix0.2 − B_naive` | = 0 — the robust filter stays inert |
| S4a | history@0.4 | `O_true − C_naive` | = 0 — asking insures against the behavioural model at any severity |
| S4b | stochastic@0.5 | `O_true − C_naive` | > 0 — acceptance noise is the axis asking does not fully insure |
| G_* | every process | `B_naive − A_prior` | the full grid in CD-MIS-1's negative-learning direction (< 0), reported without selection |
| D_* | every evasive level | `C_naive − B_naive` | the dose-response, reported without selection |
| descriptive | the true-rate × assumed-rate matrix of mean Y and of asks per episode; per family, mean Y and the four contrasts `B_naive − A_prior`, `B_mix0.2 − B_naive`, `C_naive − B_naive`, `O_true − C_naive` by level with identical-trajectory shares; the CD-MIS-1 per-arm diagnostics | reported |

One confirmatory contrast (E1); the rest secondary, no multiplicity correction. The assumed rate 0.2 for E1 was chosen before the
run as "a moderate hedge a deployed agent might adopt without knowing the truth"; `C_ans0.1` and `C_ans0.3` are reported beside it.

## Sanity gates (fail → fix the pipeline, do not read results)

- No error stubs; every (process, arm) has 720 episodes.
- **No-ask invariance:** `B_naive` has identical mean Y at every evasive level (it never sees an answer; family-seeded streams).
- **Diagonal:** at evasive@0.3 the `C_ans0.3` arm has zero true-model decision regret (mean < 10⁻³ utility units; it is `O_true` up to
  tie-breaks). Its identical-trajectory share with `O_true` is reported, not gated — a pipeline check found that an exact tie between
  two questions can be broken differently under the floor.
- **Exact replication of CD-MIS-1:** at evasive@0.3, history@0.1 and stochastic@0.15 the arms shared with CD-MIS-1 reproduce its mean
  Y to 10⁻¹² (read from `results/live_mis_20261029/analysis.json`).
- `test_mis2.py` (4 tests) and the CD-MIS-1 gates `test_mis.py` (4) pass.

## Disclosure

Before this registration two pipeline checks were run and are disclosed: (i) the `test_mis2.py` runner test — buyer, one focal type,
two opponent types, K = 2, arms `A_prior / B_naive / C_naive / C_ans0.3 / O_true` on evasive@0.3, evasive@0.1 and history@0.4
(120 episodes); (ii) a 16-episode inspection of `C_ans0.3` against `O_true` on evasive@0.3 (same cell) to understand a failing
identity assertion; it showed identical Y in all 8 pairs and one question-tie broken differently, which produced the "zero regret, not
identical trajectory" gate above. No contrast in the table was computed from either.

## Predictions, stated plainly

E1 is the open question. The mechanism for "yes": at a true flip rate of 0.3 the harm in CD-MIS-1 came from a near-truthful
likelihood that *eliminated* the true type on one flipped answer; any hedge keeps the true type alive, and the planner with a
discounted information gain asks less. The mechanism for "no": the hedge also discounts truthful answers, so the planner may stop
asking altogether and fall to the no-ask value — in which case `C_ans0.2 − C_naive` is still positive (it removes the harm) but
`O_true − C_ans0.2` is also positive (it removes the value). E2 is expected small and negative. E3 should change sign between 0.1
and 0.3. E5 should hold exactly (asks → 0). S1 is expected to hold; S2 is the genuinely uncertain secondary — at acceptance noise
0.5 the naive filter reads half the rejections of acceptable offers as type evidence. S3 is expected to hold (the planner acts on
which type leads, not how confident the filter is). S4a should hold exactly; S4b should be positive, as it was at 0.15 (+0.029).

## Analysis (fixed in advance)

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.merge_mis --dir results/live_mis2_20261030
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir results/live_mis2_20261030 --B 5000 --seed 41
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_mis2 --analysis results/live_mis2_20261030/analysis.json --mis1 results/live_mis_20261029/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis2_note --dir results/live_mis2_20261030 \
    --verdicts results/live_mis2_20261030/verdicts.json --out llm_rl_scripts/car_dealer_live/MIS2_RESULTS.md
```

`analyze_mis.py` gained generic arm pairs (every `C_*` arm against `C_naive` and under `O_true`), the identical-trajectory share, and
the severity ordering; its existing pairs and numbers are unchanged, so re-running it on the CD-MIS-1 directory reproduces that
analysis with the new fields appended.

## Post-run disclosure (2026-10-30): the replication gate failed once, for a pipeline reason, and the grid was re-run

The first run of the registered grid passed four gates and failed **exact replication of CD-MIS-1**: the no-ask arms matched to
10⁻¹², the ASK arms did not (`C_naive` at evasive@0.3 differed by +0.005 in mean Y; 60 of 720 trajectories differed, every one at an
exact tie between two questions at the uniform prior). Cause: `belief_live.action_likelihoods` built the opponent-action table from a
Python set, so the oracle's floating-point summation order depended on the interpreter's hash seed (reproduced: PYTHONHASHSEED 4 broke
the tie the other way; the eight seeds 0–7 agree after the fix). The table is now ordered by action key; a fifth test in
`test_mis2.py` runs the tie cell under two hash seeds; both `REGENERATE.sh` scripts export `PYTHONHASHSEED=0`. **Both CD-MIS-1 and
CD-MIS-2 were regenerated** with the fixed code, and all five gates then passed. Nothing in the rules or the analysis changed; the
first run's `analysis.json` is kept as `results/live_mis2_20261030/analysis_first_run_before_determinism_fix.json` (every rule had the
same verdict in it). One pipeline fix after the run, not a change to any decision rule: `merge_mis.py` took the per-process metadata
from the first sub-directory only, which hid the severity levels from the rendered tables; it now unions them.

The Phase 1 scripted result directories (`results/live_phase1_scripted*_20261029/`) were produced before this fix and may differ at
exact ties if regenerated; the Jev runs cannot be regenerated. Neither is touched here.

## Limits accepted in advance

Three families swept, two levels beyond CD-MIS-1 on the behavioural axes; the hedge is a single fixed rate per arm (no adaptive
estimate of the flip rate); the exact oracle as planner; easy regime; H = 4; the 36 type pairs are the whole type space; `A_prior`
is the design's prior-only forward model, not a tuned heuristic.
