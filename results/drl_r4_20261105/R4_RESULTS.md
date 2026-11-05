# CD-DRL R4 — weaker / absent oracles: how much V* recovery survives a degraded potential

Registered regime H=4, c_q=0, matched (opponent = assumed = registered T1), shortlist menu. One learner (nstep1 actor-critic, R1 setup); only the shaping potential Phi changes. Exact ceiling (= R1): **0.3792**.

Under the one-step estimator Phi enters the bootstrap, so potential quality is the whole story: Phi = V* is unbiased (R1 recovers V*), weaker potentials trade bias for being cheap or oracle-free.

## Ladder (mean net vs the exact ceiling)

| potential | oracle | mean net | residual (SE) | recovery | optimal-match | ASK/ep | ASK recall |
|---|---|---|---|---|---|---|---|
| exact | exact V* | 0.3793 | +0.0000 (+0.0) | +1.00 | 1.000 | 2.00 | 1.000 |
| mismatch | wrong-kernel V* | 0.3793 | +0.0000 (+0.0) | +1.00 | 1.000 | 2.00 | 1.000 |
| noask | V*_noask (no inquiry) | 0.3793 | +0.0000 (+0.0) | +1.00 | 1.000 | 2.00 | 1.000 |
| entropy@0.05 | none (belief entropy) | 0.3034 | -0.0759 (-21.7) | -0.09 | 0.319 | 0.00 | 0.000 |
| entropy@0.15 | none (belief entropy) | 0.3019 | -0.0773 (-25.5) | -0.12 | 0.345 | 0.00 | 0.000 |
| entropy@0.38 | none (belief entropy) | 0.1562 | -0.2230 (-57.6) | -2.22 | 0.619 | 1.12 | 0.543 |
| none | none | 0.3099 | -0.0693 (-20.2) | +0.00 | 0.331 | 0.00 | 0.000 |

Recovery fraction = (net - none_net) / (ceiling - none_net): 1 = full V* recovery, 0 = the no-shaping floor. 'recovers V*' (within 4 SE of the ceiling): exact=True, mismatch=True, noask=True, entropy@0.05=False, entropy@0.15=False, entropy@0.38=False, none=False.

## Reading

Recovery: exact +1.00, mismatch +1.00, noask +1.00, entropy@0.05 -0.09, entropy@0.15 -0.12, entropy@0.38 -2.22, none 0. All three VALUE potentials (exact, wrong-kernel, inquiry-blind V*_noask) recover V* fully: V* is nearly invariant to the opponent model here, and the small value of inquiry is captured from the true one-step reward even when the potential ignores it. The oracle-free entropy potential fails at every scale tested (best entropy@0.05: -0.09): rewarding information gain rather than value misleads the bootstrapped learner in Car-Dealer, where ASK is optimal at only ~2% of nodes -- the opposite of 20Q, where information gain IS the objective. This is categorical, not mere scale-sensitivity.

Wrong kernel: alpha0=3/5, beta=4.0 (registered: 4/5, 8). Entropy scales (0.05, 0.15, 0.38) (primary 0.38). Training: 400 iters x 192 episodes, nstep1, per arm.
Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r4 --out <dir>` (seed 0).
