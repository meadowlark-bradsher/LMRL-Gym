# Binned Car Dealer — Condition B (`binned-car-dealer-b-v0.1`)

An exactly solvable information-acquisition negotiation built on the
[`car_dealer_discrete`](../car_dealer_discrete/) engine: one learner (buyer or seller, 6 private
rankings) negotiates against a frozen scripted opponent, choosing each turn between asking a
question about the opponent's priorities and proposing a deal. Every option carries an exact
expected information gain and an exact Q* from a dynamic-programming reference, so each decision
yields an **EIG regret** (bits) and a **decision regret** (utility).

- Design source: [`binned_car_dealer_experiment.md`](binned_car_dealer_experiment.md) (verbatim)
- What was built and why it departs: [`DESIGN_B.md`](DESIGN_B.md)
- Scripted-arm results (exact-on-logs, no model): [`SCRIPTED_RESULTS.md`](SCRIPTED_RESULTS.md)
- Pre-registration for the Jev run: [`PREREGISTRATION_B.md`](PREREGISTRATION_B.md) — **no Jev
  decision has been collected on this game**; the Jev arm has only been exercised with the mock backend.

## Code

| File | Role |
|---|---|
| `questions.py` | 9-question bank with exact response matrices (TOP / ABOVE / BOTTOM) |
| `beliefs.py` | exact beliefs (Fractions), entropy, EIG |
| `env.py` | config, public state, menu rule, transitions, scripted opponent, per-option annotations |
| `reference.py` | exact DP reference (menu-constrained and unconstrained), regrets, Q gap |
| `arms.py` | `random`, `greedy_eig`, `myopic`, `exact` |
| `jev_policy.py` | Jev arm: lettered, shuffled Choice; `show_eig` toggles the EIG annotation |
| `run_condition_b.py` | enumerates all 72 cells × arms × repeats; streams exact per-decision JSONL |
| `analyze_condition_b.py` | per-arm metrics, paired cluster bootstrap vs `exact`, H3 quantities |
| `sweep_horizon.py` | horizon × regime sweep of the scripted arms (design evidence) |
| `render_scripted_note.py` | renders `SCRIPTED_RESULTS.md` from the frozen JSON |
| `verify_b.py` | re-derives every frozen `analysis.json` and the rendered note; `--sweep` recomputes the sweep too |
| `test_condition_b.py` | exactness identities (9 tests, ~7 s) |

```bash
PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_binned
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.verify_b
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.run_condition_b --arms exact,greedy_eig,myopic,random --rounds 2 --repeats 20 --seed 0 --out results/binned_b_scripted_r2_easy_20261026
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b results/binned_b_scripted_r2_easy_20261026 --seed 0 --json results/binned_b_scripted_r2_easy_20261026/analysis.json
```

## Results directories (repo root `results/`)

| Directory | Contents |
|---|---|
| `binned_b_scripted_r2_easy_20261026` | 2 rounds, easy; `exact`, `greedy_eig`, `myopic` once per cell, `random` × 20; `runs.jsonl.gz` (full per-decision menus and Q*), `summary.json`, `analysis.json`, `REGENERATE.sh` |
| `binned_b_scripted_r6_easy_20261026` | 6 rounds, easy (the document's horizon) |
| `binned_b_scripted_r2_hard_20261026` | 2 rounds, hard |
| `binned_b_sweep_20261026/sweep.json` | horizon 1–6 × regime sweep |

## Conventions

Same as `car_dealer_discrete/README.md`: notes are rendered from frozen JSON, claims are tagged
exact-on-logs or stochastic, confirmatory runs are pre-registered, bootstrap unit = cell with the
72-cell precision ceiling, one primary hypothesis per program, figures regenerable like numbers.
