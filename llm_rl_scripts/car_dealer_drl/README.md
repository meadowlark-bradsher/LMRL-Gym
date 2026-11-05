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
| **R2** (done) | Does model-free RL **discover ASK** from reward alone? (mirrors the Jev 0/2555 finding) | None in training; oracle labels eval nodes only |
| **R3** (done) | Best-response to the **CD-MIS opponent population**; transfer to held-out opponents | Assumed-kernel shaping only; O_true/O_assumed as eval references |
| **R4** (done) | Weaker / absent oracles: how much V\* recovery survives as Φ degrades (exact → wrong-kernel → inquiry-blind → entropy → none)? | Degraded → gone |

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
| `train_r1.py` | R1/R2 advantage actor-critic; one-step shaped-reward advantage (R1, low variance because Φ=V\* absorbs the continuation) or MC return-to-go (R2 sparse); entropy-annealed, model-selected on greedy optimal-match |
| `run_r1.py` | R1 end-to-end: train shaped + the shaping-off ablation → gate; writes `results.json`, `policy.pt`, `R1_RESULTS.md` |
| `ask_eval.py` | R2 ASK-aware evaluation: labels each node's ASK-optimality with the exact oracle, scores captured-VOI fraction, ASK-optimal recall/precision, ASK rate (oracle labels eval only) |
| `run_r2.py` | R2 end-to-end: train with no oracle (sparse reward, MC) → ASK-discovery verdict; writes `results.json`, `policy.pt`, `R2_RESULTS.md` |
| `env_mis.py` | R3 misspecified rollout: TRUE opponent (`opponents_mis`) acts/answers/pays while the agent plans with the ASSUMED kernel (robustified `filters_mis` belief, shortlist menu, Φ=V\*_assumed); `otrue_net_gain` is the clairvoyant O_true rollout |
| `evaluate_mis.py` | R3 per-process scoring vs O_assumed (misspecified planner) and O_true (clairvoyant ceiling), with the pre-registered train/held-out split and advantage SEs |
| `train_r3.py` | R3 best-response to the opponent mixture; one-step shaped-reward advantage; model-selected on mean true net over the training processes |
| `run_r3.py` | R3 end-to-end: train on the mixture → robustness + transfer verdicts (+ RL-competence gate on `matched`); writes `results.json`, `policy.pt`, `R3_RESULTS.md` |
| `potentials.py` | R4 shaping potentials: `phi_exact` (V\*), `make_phi_mismatch` (wrong-kernel V\*), `phi_noask` (inquiry-blind V\*), `phi_entropy` (oracle-free belief entropy, the deepink potential). `env.rollout` takes a `potential` hook |
| `run_r4.py` | R4 degradation ladder: train each potential (nstep1) → recovery vs the exact ceiling + ASK rate; writes `results.json`, `policies.pt`, `R4_RESULTS.md` |
| `test_drl.py` | R0 gates (fast subset): feature dims, decision-sufficiency of the featurization, oracle-shim reproduces `exact_ask`, row masking, BC learns |
| `test_r1.py` | R1 gates (fast subset): PBRS telescoping identity, env dynamics reproduce the ceiling, shaping-off is sparse, actor-critic learns |
| `test_r2.py` | R2 gates (fast subset): ASK-evaluator endpoints (oracle recall 1.0 / full VOI, noask-optimal recall 0 / zero VOI), verdict classifier |
| `test_r3.py` | R3 gates (fast subset): robustified filter survives every process, O_true ≥ O_assumed per process, `matched` is calibrated, `train_r3` learns |
| `test_r4.py` | R4 gates (fast subset): entropy potential well-formed + PBRS telescoping, mismatch is kernel-robust (and V\*_noask a stronger degradation), the potential hook trains |

## Run

```bash
PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_drl
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r0 --out results/drl_r0_<date>
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r1 --out results/drl_r1_<date>
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r2 --out results/drl_r2_<date>
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r3 --out results/drl_r3_<date>
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r4 --out results/drl_r4_<date>
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

## Verdict (R2)

R2 reports a **discovery verdict**, not a pass/fail — the scientific question has
no predetermined success. The training config is tuned for *value* (never for the
ASK outcome) and frozen in `run_r2.py`'s docstring before the run. In the exact
registered regime the value of inquiry is ~0.038 net (~10% of the ceiling) and ASK
is strictly optimal at ~2% of focal nodes, so the prize is real and rare.

- **captured-VOI fraction** = (net − noask-ceiling)/(ask-ceiling − noask-ceiling):
  0 = never-ask-optimal, 1 = ask-optimal.
- **ASK-optimal recall** = asks taken at the oracle's strictly-ASK-optimal nodes
  (Jev's 0/2555 is recall 0 — the number R2 asks whether RL beats).
- Verdict: **DISCOVERED** if captured-VOI ≥ 0.50 and recall ≥ 0.50; **PARTIAL** if
  recall ≥ 0.10 or captured-VOI ≥ 0.25; else **NOT_DISCOVERED** (Jev-like).

## Verdict (R3)

One policy is trained against a **mixture** of true opponents (the training
processes) while planning/observing only with the assumed kernel — it never sees
which opponent it faces. It is scored against two references per process:
**O_assumed** (the registered shortlist oracle deployed under misspecification —
same menu and belief as the learned actor) and **O_true** (the clairvoyant
per-draw ceiling: full action set + exact true belief). The pre-registration
(frozen in `run_r3.py`) fixes the transfer split — train
{matched, stochastic, anchoring, history}, hold out {threshold, withholding,
evasive} — and the verdicts:

- **RL-competence gate:** on `matched` (no misspecification) the learned actor must
  reach O_assumed within 2 SE; otherwise a shortfall elsewhere can't be blamed on
  misspecification and the robustness read is withheld.
- Per group, advantage = learned − O_assumed, z = advantage/SE: **BEATS_PLANNER**
  (z > +2), **MATCHES_PLANNER** (|z| ≤ 2), **BELOW_PLANNER** (z < −2).
- **Transfer:** does the held-out advantage track the training advantage (robust)
  or fall below it (overfit to the mixture)?
- Context only: misspecification-gap-recovered = advantage / (O_true − O_assumed),
  and regret vs O_true (no menu-restricted, model-blind actor can reach O_true).

## Result (R4)

Holding the nstep1 learner fixed and varying only Φ, recovery = (net − none)/(ceiling
− none), 1 = full V\* recovery:

| Φ | recovery | ASK/ep |
|---|---|---|
| exact V\* | +1.00 | 2.0 |
| wrong-kernel V\* | +1.00 | 2.0 |
| V\*_noask (inquiry-blind) | +1.00 | 2.0 |
| entropy @0.05 / @0.15 / @0.38 | −0.09 / −0.12 / −2.22 | 0 / 0 / 1.1 |
| none (Φ=0) | 0.00 | 0 |

Every **value** potential recovers V\* fully — V\* is nearly invariant to the opponent
kernel, and the small value of inquiry is recovered from the true one-step reward even
by an inquiry-blind potential. The **oracle-free entropy** potential fails at every
scale (categorical, not scale-sensitivity): rewarding information gain rather than
value misleads the bootstrapped learner where ASK is optimal at only ~2% of nodes — the
opposite of 20Q, where information gain *is* the objective. So the oracle's contribution
is specifically its **value** content, not merely dense shaping. (H=6 approximate-oracle
arm: documented follow-up — the exact H=6 ceiling is ~35 min to precompute.)

## Reuses

`car_dealer_live` end to end: `Cell` (menu + the `ask`/`noask` oracles),
`belief_live`, `protocol_live`, `menu_live`, `arms_live`, and the registered
kernel (`KernelParams`). Notes are rendered from frozen JSON, exact-on-model, in
the `car_dealer_discrete` convention.
