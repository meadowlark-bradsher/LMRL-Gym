# CD-DRL R4 (H=6 arm) — a cheap approximate oracle where exact planning is expensive

Regime H=6, c_q=0, matched, shortlist menu. Exact ceiling (mean root value): **0.3821** (computed once, ~35-40 min; cached). One nstep1 learner; only Phi varies.

Value potential = exact V* (reusing the ceiling's memo -- see the amortization note below). Entropy scale 0.1 (the gentle end of the H=4 sweep).

## Recovery (mean net vs the exact ceiling)

| potential | mean net | recovery | ASK/ep |
|---|---|---|---|
| exact | 0.3488 | +0.61 | 0.00 |
| entropy | 0.3117 | +0.17 | 0.00 |
| none | 0.2970 | +0.00 | 0.00 |

Recovery = (net - none_net) / (ceiling - none_net): 1 = full recovery of the exact H=6 value, 0 = the no-shaping floor.

## Reading

Exact-V* recovery +0.61; entropy +0.17; none 0 (ASK/ep: exact 0.00, entropy 0.00). The ordering of R4 holds at H=6: the exact value potential recovers far more (+0.61) than the oracle-free entropy potential (+0.17) -- information-shaping is not a substitute for value-shaping, consistent with H=4. But exact-V* recovers only +0.61 at the matched H=4 budget and asks 0.00/ep: the longer horizon is harder to learn, and the shortfall is largely the hard-to-reach ASK value (offer-play is learned but not the information-gathering refinements VOI grows with at H=6 -- echoing R2). Closing it needs more budget; this arm is a feasibility/ordering check, not a full H=6 solve.

## Amortization note (why exact-Phi, not an approximate oracle)

The approximate-oracle arm originally planned for H=6 (a cheap depth-capped V*, `TruncatedOracle`) turns out not to save compute here: the exact oracle is expensive once (the ~35-min ceiling solve) but O(1) per lookup thereafter, so exact-Phi training is cheap, whereas the truncated oracle is recomputed lazily at every node over changing beliefs and amortises poorly -- the full approximate arm measured at ~5+ hours for no benefit. The efficient value potential at H=6 is the exact one. (The `TruncatedOracle` is kept and tested as the approximate-potential implementation.)

Training: 400 iters x 192 episodes, nstep1, per arm (seed 0).
Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r4_h6 --out <dir>` (reuses the cached ceiling).
