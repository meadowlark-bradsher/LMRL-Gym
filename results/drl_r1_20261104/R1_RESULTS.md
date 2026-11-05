# CD-DRL R1 — RL recovers V* with dense help (PBRS, Phi = V*)

**Gate: PASS.** Registered regime H=4, c_q=0, T1 kernel (alpha0=4/5, beta=8, p_counter=1, p_hold=1, shortlist 3), shortlist menu.

Advantage actor-critic on the belief-MDP. The net learns its policy from reward only; the exact oracle enters **solely** as the potential `Phi = V*` in the Ng-Harada-Russell shaped reward `r' = R + Phi(s') - Phi(s)` (terminal clamp `Phi = 0`), never as an action label. Phi = V* makes the expected one-step shaped reward equal the optimal advantage `A*(s,a)`, so credit assignment is dense while the optimal policy is unchanged.

## Result (shaped arm)

- Training: 400 iters x 192 episodes; best greedy optimal-match at iter 100.
- Episodic (2880 episodes): mean net gain **0.3793** vs exact ceiling **0.3792**; residual +0.0000 (+0.01 SE, within 4 SE: True).
- Per-decision optimal-match **1.0000**; mean decision regret 0.00000.

## Contrast (H3): shaping off (Phi = 0, sparse terminal reward, same budget/seed)

Best-effort both ways: the shaped arm uses the one-step shaped-reward advantage (low variance because Phi=V* absorbs the continuation); the sparse arm uses the Monte-Carlo return-to-go (without a potential the one-step reward is 0 on every non-terminal step, so MC is the correct sparse-reward estimator).

- Mean net gain 0.3258; residual -0.0535 (-14.93 SE, within 4 SE: False).
- Per-decision optimal-match 0.3688; mean decision regret 0.03075.
- Shaping delta: optimal-match +0.6312, regret +0.03075 (positive = shaping helps). Read: shaping helps (H3 direction confirmed)

Harness sanity (oracle-shim policy): residual +0.0000 (+0.02 SE), optimal-match 1.0000.

## Gate detail

- H1 primary (shaped within 4 SE of ceiling): z = +0.01 -> True.
- H2 secondary (shaped optimal-match >= 0.9): 1.0000 -> True.
- H3 contrast (direction pre-registered, not a gate): shaping helps (H3 direction confirmed)

Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r1 --out <dir>` (seed 0).
