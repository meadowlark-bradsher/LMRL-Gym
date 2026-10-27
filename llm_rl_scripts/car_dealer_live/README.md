# Live Car-Dealer (`live-car-dealer-v0.1`) — CD-VOI-1 / CD-MIS-1, Phase 0

The bilateral discretized Car-Dealer game with a public **ASK** action, a history-responsive
**T1 opponent** whose kernel is known to the oracle, a matched finite filter, and an **exact
finite-horizon oracle** that values every option in net gain. Phase 0 (implementation and oracle
readiness) is complete; **no registration, no Jev or LLM call**.

- Design source (verbatim): [`car_dealer_live_voi_experiment_design.md`](car_dealer_live_voi_experiment_design.md)
- Phase 0 decisions and gates: [`DESIGN_LIVE.md`](DESIGN_LIVE.md)
- Phase 0 result (where is ASK optimal?): [`PHASE0_RESULTS.md`](PHASE0_RESULTS.md), rendered from
  `results/live_phase0_sweep_20261027/sweep.json`

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

```bash
PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_live
PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.sweep_ask_value --out results/live_phase0_sweep_20261027/sweep.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase0_note --sweep results/live_phase0_sweep_20261027/sweep.json --out llm_rl_scripts/car_dealer_live/PHASE0_RESULTS.md
```

## Reuses

`car_dealer_discrete.game` (27 deals, 6 rankings, exact utilities) and `.scoring` (reservations);
`car_dealer_binned.questions` (the nine TOP / ABOVE / BOTTOM questions with exact response matrices).

## Conventions

As in `car_dealer_discrete/README.md`: notes rendered from frozen JSON, exact-on-model vs stochastic
tags, pre-registration before any confirmatory run, one primary contrast, figures regenerable like
numbers.
