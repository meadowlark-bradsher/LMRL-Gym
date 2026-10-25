# Pre-registration — Experiment A main run

Fixed **before** the 50-seed data were seen (committed prior to launch). Written
after the pilot, audit, and K=15 calibration, which motivated the design but used
seeds {0,1,3,4,5} only; the main run uses seeds 0–49 and is confirmatory.

## Design

- Conditions: **A1, A2, A3, A4** (goal G0/G1 × history H0/H1), narrative context in
  all, target = milk. Defined in [`conditions.py`](conditions.py).
- Seeds: **0–49** (50 matched games; same game across all four conditions).
- Rollouts: **K = 8** independent Jev rollouts per (seed, condition) — the game is
  fixed by seed, only Jev's stochastic choices vary.
- Env: real TextWorld Text-Nav, PO, 20-step limit, `jev-latest`, no RL training,
  actions from the admissible set only.

## Unit of analysis

The **seed** is the cluster. Per (seed, condition) we compute a success rate over
the 8 rollouts; condition-level estimates and all contrasts aggregate across the
50 seeds with a **cluster bootstrap resampling seeds** (10,000 resamples). We do
not treat the 400 rollouts of a condition as independent.

## Endpoints

- **Primary:** task success (`won` within 20 steps), as a per-condition mean of
  per-seed rates.
- **Secondary:** greatest correctly-ordered milestone reached (M0–M5), per-seed
  mean, same cluster bootstrap.

## Pre-registered contrasts (seed-paired)

1. **A4 vs A1** — full recipe vs reactive implicit.
2. **A4 vs A2** — value of history given an explicit goal.
3. **A4 vs A3** — value of an explicit goal given history.
4. **Goal main effect:** (A2+A4) − (A1+A3).
5. **History main effect:** (A3+A4) − (A1+A2).
6. **Interaction:** (A4 − A3) − (A2 − A1). Primary interest: history helps only
   when the goal is explicit.

Each contrast is reported as a paired per-seed difference with a 95% cluster-
bootstrap CI. Because binary success is expected to be 0 in ≥2 cells, the
milestone endpoint carries the interaction test where success is degenerate.

## Decision rules

- An effect is reported as supported if its 95% cluster-bootstrap CI excludes 0.
- Prediction (from calibration, not guaranteed): A4 success > 0 with CI excluding
  0; A1 and A3 at/near 0. A2 is the genuinely new cell — no main-run-independent
  estimate exists yet.
- Seed heterogeneity (per-seed A4 rate ranged 0.0–0.8 in calibration) is reported
  explicitly; the pooled rate alone is not the headline.

## Analysis artifact

`analyze_main_a.py` (seed-clustered) is the sole generator of the reported
numbers; the results write-up cites it and nothing is hand-typed.
