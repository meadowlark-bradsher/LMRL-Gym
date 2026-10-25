# Experiment A — Repeats Calibration (success as a rate)

**Status:** Calibration run to settle the per-episode stochasticity exposed by the
determinism audit and to size the main run. Conditions **A1, A3, A4** × seeds
**0, 1, 3, 4, 5** × **K = 15** independent rollouts = 225 episodes, 0 API errors.
**Model:** `jev-latest` · real PO env · 20-step limit · 2026-10-25.

(A2 was not in this calibration — the run targeted the A4-vs-{A1,A3} contrast the
audit called into question. A2 returns in the main run to complete the 2×2.)

## Provenance

```bash
python -m llm_rl_scripts.text_nav.jev.analyze_calibration \
    llm_rl_scripts/text_nav/jev/results/calib_A_20261025   # -> calibration.{csv,json}
```

Raw logs + analysis: [`results/calib_A_20261025/`](results/calib_A_20261025/).
Same seeding (games identical across rollouts; only Jev's choices vary).

## Result: the condition effect is real; the single-run point value was not

| Condition | Pooled success | Wilson 95% | Per-seed rate range |
|---|---|---|---|
| A1 (implicit, reactive) | 0/75 | [0.000, 0.049] | 0.0 – 0.0 |
| A3 (implicit, history) | 0/75 | [0.000, 0.049] | 0.0 – 0.0 |
| **A4 (explicit + history)** | **17/75 = 0.227** | **[0.147, 0.333]** | **0.0 – 0.80** |

A4's interval does not overlap A1/A3, so **A4 genuinely completes the task where
A1 and A3 never do**. The determinism audit's "A4 0/10" was a down-draw of a real
~0.23 process, and the pilot's "A4 3/10" was an up-draw; neither single run was
wrong, both were noise around the same rate.

## Success is dominated by the seed

| A4 by seed | success [Wilson 95%] | mean greatest-ordered ± SE | reading |
|---|---|---|---|
| seed 4 | 0.80 [0.55, 0.93] | 4.47 ± 0.31 | reliably completes |
| seed 3 | 0.33 [0.15, 0.58] | 2.80 ± 0.46 | sometimes completes |
| seed 1 | 0.00 [0.00, 0.20] | 1.27 ± 0.18 | reaches M1–M2, never finishes |
| seed 0 | 0.00 [0.00, 0.20] | 0.00 ± 0.00 | trapped, never acquires |
| seed 5 | 0.00 [0.00, 0.20] | 0.00 ± 0.00 | trapped, never acquires |

The start room (distance to the living-room milk and the kitchen fridge) swamps
everything else. The pilot's apparent win on **seed 1 was luck** — it is 0/15
here; seeds 3 and 4 are the genuine winners.

## Mechanism, now robust

On the "easy" seeds (1, 3, 4) A1 and A3 reach **exactly M1 every rollout**
(mean ordered = 1.00 ± 0.00) and never go further. So the M2 (transport) wall is
not a failure to *acquire* the food — on reachable seeds they always pick it up —
but a failure to *carry it onward* without both an explicit goal and history.
Only A4 crosses M2, and only sometimes.

## Implications for the main run

1. **Unit of analysis = seed (cluster).** Per-seed success ranges 0.0–0.8, so the
   seed, not the rollout, carries the variance. Report condition rates with a
   **cluster bootstrap over seeds**, not pooled binomial CIs.
2. **Many seeds × modest K beats few seeds × huge K.** Pinning a *per-seed* rate
   is expensive (p = 0.8 needs K ≈ 28 for ±0.15, ≈ 62 for ±0.10), but the
   *condition-level* estimate is cheap when many seeds each contribute a rough
   per-seed rate. Recommended: **50 seeds × 4 conditions × K ≈ 8–10**, seed as the
   bootstrap cluster. (K = 5 halves cost with modest condition-level precision
   loss.)
3. **Include A2** to complete the 2×2 and test goal-alone.
4. **Pre-register** the contrasts (A4 vs A1, vs A2, vs A3; the interaction) and the
   milestone endpoint before running.

Rough cost: 50 × 4 × 10 ≈ 2000 episodes (~30k calls); K = 5 ≈ 1000 episodes
(~15k calls). This calibration was 225 episodes.
