# CD-DRL R2 — does model-free RL discover ASK from reward alone?

**Verdict: NOT_DISCOVERED.** Registered regime H=4, c_q=0, T1 kernel (alpha0=4/5, beta=8, p_counter=1, p_hold=1, shortlist 3), shortlist menu.

RL with **no oracle**: no behavior-cloning labels (R0), no PBRS potential (R1). Advantage actor-critic, Monte-Carlo return-to-go, from the raw terminal reward only. The net sees the same menu (EIG, P(accept)) the Jev arm sees; the exact oracle only labels eval nodes and supplies the ceilings. Mirrors the live finding that Jev asks 0/2555 even when shown EIG -- R2 asks whether a trained actor beats that zero.

## Calibration (exact)

- Value of inquiry: ask-ceiling **0.3792** vs noask-ceiling **0.3409** -> VOI **+0.0383** (~10% of the ceiling). ASK strictly optimal at ~2% of focal nodes.

## Learned policy (no oracle)

- Mean net gain **0.3258**; residual vs ask-ceiling -0.0535 (-14.93 SE). Above noask-ceiling: False.
- **Captured-VOI fraction -0.396** (0 = never-ask-optimal, 1 = ask-optimal).
- **ASK-optimal recall 0.000** (asks at the 2747 strictly-ASK-optimal nodes); ASK rate 0.000/episode (0.000/decision); ASK precision nan.
- Per-decision optimal-match 0.3688; mean decision regret 0.03075.

## Oracle reference (ground truth, same evaluator)

- Mean net 0.3793 (= ceiling); captured-VOI +1.001; ASK recall 1.000; ASK rate 1.683/episode.
- Jev comparison (live): asked 0/2555, recall 0. This actor's recall: 0.000.

## Verdict detail

- captured-VOI -0.396 (DISCOVERED >= 0.50, PARTIAL >= 0.25).
- ASK-optimal recall 0.000 (DISCOVERED >= 0.50, PARTIAL >= 0.10).
- Mastered offers (net >= noask-ceiling): False.
- **NOT_DISCOVERED.**

## Robustness: sustained exploration (constant entropy, no annealing)

A second arm holds the entropy bonus constant at 0.08 so the stochastic policy keeps sampling ASK throughout training (the primary arm's entropy collapsed to ~0.04, leaving ASK unexplored). This rules out premature-exploration-collapse as the cause of the null.

- Verdict NOT_DISCOVERED: captured-VOI -0.396, ASK-optimal recall 0.000, ASK rate 0.000/episode, net 0.3258.
- Reading: sustained exploration still does not discover ASK -- the null is robust.

Training: 800 iters x 384 episodes, MC return-to-go, lr 0.0015, entropy 0.03->annealed; best greedy optimal-match at iter 100.
Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r2 --out <dir>` (seed 0).
