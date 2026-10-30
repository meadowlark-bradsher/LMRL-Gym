# Live Car-Dealer (`live-car-dealer-v0.1`) — CD-VOI-1 / CD-MIS-1 / CD-MIS-2 / CD-SEL-1, Phases 0–1 and the Phase 2 selector arms

The bilateral discretized Car-Dealer game with a public **ASK** action, a history-responsive
**T1 opponent** whose kernel is known to the oracle, a matched finite filter, and an **exact
finite-horizon oracle** that values every option in net gain. Phase 0 (implementation and oracle
readiness) is complete; Phase 1's menu rule, arms, runner and scripted baselines are built and
frozen; the Jev run is pre-registered in `PREREGISTRATION_LIVE.md`.

- Design source (verbatim): [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md)
- Phase 0 decisions and gates: [`DESIGN_LIVE.md`](DESIGN_LIVE.md)
- Phase 0 result (where is ASK optimal?): [`PHASE0_RESULTS.md`](PHASE0_RESULTS.md), rendered from
  `results/live_phase0_sweep_20261027/sweep.json`
- Phase 1 decisions (cell, menu, estimand) and the menu rule: [`DESIGN_LIVE.md`](DESIGN_LIVE.md) §4–6
- Phase 1 scripted baselines and costlessness check: [`PHASE1_SCRIPTED_RESULTS.md`](PHASE1_SCRIPTED_RESULTS.md),
  rendered from `results/live_phase1_scripted3_20261029/` (registered menu; `results/live_phase1_scripted_20261029/` holds the
  two-candidate baselines that motivated the change)
- Pre-registration of the Jev run: [`PREREGISTRATION_LIVE.md`](PREREGISTRATION_LIVE.md); results: [`JEV_RESULTS_LIVE.md`](JEV_RESULTS_LIVE.md)
- CD-MIS-1 (learning with the wrong opponent model): [`PREREGISTRATION_MIS.md`](PREREGISTRATION_MIS.md); results: [`MIS_RESULTS.md`](MIS_RESULTS.md)
- CD-MIS-2 (severity sweep; the evasive-answer channel, hedged answer models): [`PREREGISTRATION_MIS2.md`](PREREGISTRATION_MIS2.md);
  results: [`MIS2_RESULTS.md`](MIS2_RESULTS.md), rendered from `results/live_mis2_20261030/`
- CD-SEL-1 (Jev shown what the history implies: the posterior in words, P(accept) on every offer; design Phase 2 selector arms): [`PREREGISTRATION_SEL.md`](PREREGISTRATION_SEL.md); results: [`SEL_RESULTS.md`](SEL_RESULTS.md), rendered from `results/live_sel_jev_h4_cq0_20261030/`

## Code

| File | Role |
|---|---|
| `protocol_live.py` | public state, legal actions (PROPOSE / ACCEPT / REJECT / ASK / WALK / HOLD), shared budget, deadline rule, gain |
| `opponent.py` | `T1Kernel`: aspiration-conceding, countering, history-aware scripted negotiator; `distribution()` exposes every probability |
| `belief_live.py` | matched posterior over the 6 opponent rankings; updates only from opponent evidence |
| `oracle.py` | exact DP over (public key, posterior); `ask_allowed` toggles V\*_ask / V\*_noask; counts ASK-optimal nodes |
| `simulate.py` | Monte Carlo episodes of any focal policy against the sampled kernel |
| `sweep_ask_value.py` | kernel × H × c_q sweep → `sweep.json` |
| `render_phase0_note.py` | renders `PHASE0_RESULTS.md` from the JSON |
| `test_live.py` | the Phase 0 gates (6 tests, ~15 s) |
| `menu_live.py` | Phase 1 shortlist menu (SAFE / AMBITIOUS offers, top-EIG questions, free moves) with exact annotations; restriction switches |
| `arms_live.py` | `exact_ask` (C_plan), `exact_noask` (B_plan), `greedy_eig`, `random` |
| `jev_policy_live.py` | Jev Choice over the lettered, shuffled menu; EIG shown / hidden; CD-SEL-1 treatments: the posterior in words (`show_belief`), P(accept) on every offer under the posterior / prior / a placebo constant (`accept_annotation`) |
| `run_phase1.py` | Phase 1 runner: 144 cells × K, per-decision exact logs, `ceilings.json` |
| `costlessness_check.py` | full vs shortlist oracle values over menu sizes; the reviewer's stopping rule |
| `verdicts_live.py` | the pre-registered decision rules, applied by code |
| `opponents_mis.py`, `filters_mis.py`, `run_mis.py` | CD-MIS-1/2: seven true opponent processes with `family@severity` levels, robust and hedged-answer filters, the mismatch runner with belief-vs-decision diagnostics |
| `analyze_mis.py`, `verdicts_mis.py`, `render_mis_note.py` | CD-MIS-1 analysis (also CD-MIS-2: generic arm pairs, identical-trajectory share, severity ordering), pre-registered rules, rendered note |
| `verdicts_mis2.py`, `render_mis2_note.py` | CD-MIS-2 pre-registered rules and rendered note |
| `test_mis.py`, `test_mis2.py` | the CD-MIS-1 gates (4 tests) and CD-MIS-2 gates (4 tests) |
| `analyze_phase1.py`, `render_phase1_note.py` | analysis JSON (Phase 1 and CD-SEL-1 pairs, walk-rate contrasts) and the rendered Phase 1 note |
| `verdicts_sel.py`, `render_sel_note.py`, `test_sel.py` | CD-SEL-1 pre-registered rules, rendered note (`SEL_RESULTS.md`), gates (4 tests) |
| `test_phase1.py` | the Phase 1 gates (4 tests; incl. neither planner handicapped by the menu) |

```bash
PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_live
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.sweep_ask_value --out results/live_phase0_sweep_20261027/sweep.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase0_note --sweep results/live_phase0_sweep_20261027/sweep.json --out llm_rl_scripts/car_dealer_live/PHASE0_RESULTS.md
# CD-SEL-1 (registered run; needs TYPESAFE_API_KEY): see llm_rl_scripts/car_dealer_live/PREREGISTRATION_SEL.md and results/live_sel_jev_h4_cq0_20261030/REGENERATE.sh
# Phase 1 scripted baselines (primary cell; see results/live_phase1_scripted_20261029/REGENERATE.sh for all regimes)
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_phase1 --horizon 4 --ask-cost 0 --arms exact_ask,exact_noask,greedy_eig,random --repeats 5 --seed 0 --out results/live_phase1_scripted_20261029/h4_cq0
```

## Reuses

`car_dealer_discrete.game` (27 deals, 6 rankings, exact utilities) and `.scoring` (reservations);
`car_dealer_binned.questions` (the nine TOP / ABOVE / BOTTOM questions with exact response matrices).

## Conventions

As in `car_dealer_discrete/README.md`: notes rendered from frozen JSON, exact-on-model vs stochastic
tags, pre-registration before any confirmatory run, one primary contrast, figures regenerable like
numbers.
