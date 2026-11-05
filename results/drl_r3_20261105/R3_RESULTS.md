# CD-DRL R3 — best-response to the CD-MIS opponent population, with transfer

**Robustness (train mix): MATCHES_PLANNER. Transfer (held-out): MATCHES_PLANNER.** Registered regime H=4, c_q=0; assumed kernel = registered T1, shortlist menu, B_naive belief filter.

One RL policy trained against a mixture of TRUE opponents (planning/observing only with the assumed kernel; Phi = V*_assumed shaping; one-step shaped-reward advantage). It never sees the process. Scored against O_assumed (the registered oracle deployed under misspecification -- same menu and belief) and O_true (the clairvoyant per-draw ceiling, full action set + true belief).

Train processes: matched, stochastic, anchoring, history. Held-out: threshold, withholding, evasive.

## RL-competence check (on 'matched', no misspecification)

- learned 0.3792 vs O_assumed 0.3792 (adv +0.0000); competent: True. 

## Results by group (mean true net)

| group | learned | O_assumed | O_true | adv over assumed | z | gap recovered | regret vs O_true |
|---|---|---|---|---|---|---|---|
| train | 0.3737 | 0.3746 | 0.3792 | -0.0008 | -0.39 | -0.18 | 0.0055 |
| held-out | 0.3467 | 0.3461 | 0.3662 | +0.0006 | +0.22 | +0.03 | 0.0195 |
| all | 0.3621 | 0.3624 | 0.3736 | -0.0002 | -0.13 | -0.02 | 0.0115 |

## Per process (learned vs O_assumed vs O_true)

| process | split | learned | O_assumed | O_true | adv | ask/ep |
|---|---|---|---|---|---|---|
| matched | train | 0.3792 | 0.3792 | 0.3791 | +0.0000 | 2.00 |
| stochastic | train | 0.3578 | 0.3612 | 0.3799 | -0.0034 | 2.00 |
| anchoring | train | 0.3786 | 0.3786 | 0.3785 | +0.0000 | 2.00 |
| history | train | 0.3792 | 0.3792 | 0.3791 | +0.0000 | 2.00 |
| threshold | held-out | 0.3756 | 0.3756 | 0.3788 | +0.0000 | 2.00 |
| withholding | held-out | 0.3792 | 0.3792 | 0.3791 | +0.0000 | 2.00 |
| evasive | held-out | 0.2852 | 0.2834 | 0.3405 | +0.0018 | 2.00 |

## Verdict detail

- Robustness (train): adv -0.0008 (-0.39 SE) -> **MATCHES_PLANNER**.
- Transfer (held-out): adv +0.0006 (+0.22 SE) -> **MATCHES_PLANNER**. held-out advantage tracks training (robust transfer)

Training: 600 iters x 256 episodes on the training mixture; best train-mean-net at iter 160.
Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r3 --out <dir>` (seed 0).
