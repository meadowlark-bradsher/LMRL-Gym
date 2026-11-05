# CD-DRL — standalone deep-RL focal policy for the live Car-Dealer game

A reward-trained focal negotiator that occupies the **planner/actor** seat of the
three-role frame (LLM = proposer, Jev = validity-guaranteeing selector, BED
component = belief + acquisition). It is the learned alternative to the exact
`car_dealer_live` oracle: it chooses from the *same shortlist menu the Jev arm
sees*, over the code-maintained matched belief, but by a trained net rather than a
single-step LLM selection. The question it answers is the one the Car-Dealer line
keeps circling — *given the belief, is the VOI policy learnable by a competent
non-LLM actor?* — so it is built as a curriculum that leans on the oracle hard at
first and weans off it.

## The rung ladder

| Rung | What it tests | Oracle role |
|---|---|---|
| **R0** (done) | Can the featurization + a tiny net **reproduce** the exact menu policy? | Teacher (behavior cloning of `exact_ask`) |
| **R1** (done) | Does RL **recover** `V*` with dense help? | Potential-based shaping, Φ = `V*` (potential only, no action labels) |
| R2 | Does model-free RL **discover ASK** from reward alone? (mirrors the Jev 0/2555 finding) | Scorer only |
| R3 | Best-response to the **CD-MIS opponent population**; transfer to held-out opponents | Per-draw ceiling |
| R4 | Weaker / absent oracles (mismatched kernel, H=6 approximate, none); the entropy potential of `deepink-revised` as the oracle-free Φ | Degraded → gone |

R0 is pure supervised imitation (no RL, no shaping): it establishes representation
sufficiency. R1 is the first rung with reward-driven learning — the net never sees
the oracle's chosen action, only the shaped reward `r' = R + Φ(s') − Φ(s)` with
Φ = V\*. Because the oracle's `q_value` is exactly `E[R + Φ(s′)]`, the expected
one-step shaped reward equals the optimal advantage `A*(s,a)`, so credit
assignment is dense while the optimal policy is left invariant (Ng–Harada–Russell).

## Code

| File | Role |
|---|---|
| `features.py` | `FocalContext` + state/row featurization. State features are a superset of the oracle's `PublicState.key()` plus belief and the 27-deal own-utility profile; row features identify the action fully (kind, exact EIG / P(accept) / own utility, tag, deal attributes, question one-hot) |
| `net.py` | `PolicyValueNet`: shared state encoder, per-row scorer (softmax over the variable menu), value head; masked padded rows |
| `dataset.py` | `build_cells` (registered regime), `collect_nodes` (roll out the scripted arms for coverage, dedup by `(role, type, state.key(), round_belief)`), `label_node` (teacher: exact Q* per row, V*, `exact_ask`'s pick), `to_tensors` |
| `train.py` | R0 behavior cloning: distillation from Q* (temperature-scaled) + auxiliary hard CE + value-head MSE to V*; early stop on validation optimal-match |
| `evaluate.py` | episode loop mirroring `run_phase1.play_episode`; scores a policy against the exact `cell.ask.root_value` ceiling; `oracle_choice` shim validates the harness |
| `run_r0.py` | R0 end-to-end: collect → train → gate; writes `results.json`, `policy.pt`, `R0_RESULTS.md` |
| `env.py` | R1 belief-MDP rollout with PBRS: emits the shaped reward `r' = R + Φ(s′) − Φ(s)`, Φ = `cell.ask.value` (terminal clamp Φ=0); `oracle_sampler` validates the dynamics |
| `train_r1.py` | R1 advantage actor-critic; one-step shaped-reward advantage (low variance because Φ=V\* absorbs the continuation), entropy-annealed, model-selected on greedy optimal-match |
| `run_r1.py` | R1 end-to-end: train shaped + the shaping-off ablation → gate; writes `results.json`, `policy.pt`, `R1_RESULTS.md` |
| `test_drl.py` | R0 gates (fast subset): feature dims, decision-sufficiency of the featurization, oracle-shim reproduces `exact_ask`, row masking, BC learns |
| `test_r1.py` | R1 gates (fast subset): PBRS telescoping identity, env dynamics reproduce the ceiling, shaping-off is sparse, actor-critic learns |

## Run

```bash
PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_drl
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r0 --out results/drl_r0_<date>
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r1 --out results/drl_r1_<date>
```

## Gate (R0)

- **Primary:** the trained policy sits within 4 SE of the exact `exact_ask`
  ceiling, pooled over all (role, opener, focal type) × opponent type × repeat
  episodes (the same tolerance `car_dealer_live` uses for its oracle MC gate).
- **Secondary:** validation optimal-match (the net's argmax row is in the
  oracle's within-EPS optimal set) ≥ 0.99.

## Gate (R1)

- **H1 (primary):** the shaped-RL greedy policy recovers V\* — mean net gain
  within 4 SE of the ceiling (same pooling as R0).
- **H2 (secondary):** per-decision optimal-match ≥ 0.90 (the level R0's clone
  reached in play).
- **H3 (contrast, pre-registered direction, not a pass/fail gate):** a shaping-off
  arm (Φ≡0, sparse terminal reward, identical budget/seed) is reported alongside.
  If it also clears H1/H2, the honest reading is that H=4 is too short for shaping
  to matter here — which motivates the longer-horizon rungs. The pre-registration
  is frozen in `run_r1.py`'s module docstring before the run.

## Reuses

`car_dealer_live` end to end: `Cell` (menu + the `ask`/`noask` oracles),
`belief_live`, `protocol_live`, `menu_live`, `arms_live`, and the registered
kernel (`KernelParams`). Notes are rendered from frozen JSON, exact-on-model, in
the `car_dealer_discrete` convention.
