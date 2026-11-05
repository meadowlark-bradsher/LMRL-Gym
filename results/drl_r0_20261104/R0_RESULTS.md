# CD-DRL R0 — behavior cloning of the exact menu policy

**Gate: PASS.** Registered regime H=4, c_q=0, T1 kernel (alpha0=4/5, beta=8, p_counter=1, p_hold=1, shortlist 3), shortlist menu.

A standalone PyTorch per-row scorer + value head, trained purely by behavior cloning against `car_dealer_live.Cell.ask` (the `exact_ask` / C_plan oracle) over the same shortlist menu the Jev arm is shown. No RL, no reward shaping — R0 only establishes that the featurization and a tiny net can reproduce the exact menu policy.

## Result

- Dataset: 1821 unique focal nodes (1548 train / 273 val), max menu rows 9.
- Validation optimal-match (argmax row in the oracle's within-EPS optimal set): **0.9927** (exact same-row match 0.9890).
- Episodic (2880 episodes, 20 repeats x 6 opponent types x 24 (role,opener,type) cells): mean net gain **0.3783** vs exact ceiling **0.3792**; residual -0.0010 (-0.37 SE, within 4 SE: True).
- Per-decision optimal-match in play: 0.9220; mean decision regret 0.00030.

Harness sanity (oracle-shim policy): residual +0.0000 (+0.02 SE), optimal-match 1.0000 — confirms the eval loop reproduces `exact_ask`.

## Gate detail

- Primary (within 4 SE of ceiling): residual z = -0.37 -> True.
- Secondary (val optimal-match >= 0.99): 0.9927 -> True.

Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r0 --out <dir>` (seed 0).
