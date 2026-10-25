# Experiment A — Main Run Results (goal × history)

**Status:** Confirmatory main run, analysed per [`PREREGISTRATION_A.md`](PREREGISTRATION_A.md)
(fixed before these data, commit `d04bd25`).
**Design:** 50 matched seeds × 4 conditions × K = 8 rollouts = **1,600 episodes**,
0 API errors. `jev-latest`, real TextWorld Text-Nav, PO, 20-step limit,
zero-shot (no RL), actions from the admissible set only. 2026-10-25.

## Provenance

```bash
python -m llm_rl_scripts.text_nav.jev.analyze_main_a \
    llm_rl_scripts/text_nav/jev/results/main_A_20261025 --boot 10000
# -> main_analysis.json, main_summary.csv (the source of every figure below)
```

Logs: [`results/main_A_20261025/`](results/main_A_20261025/) — `episodes.jsonl`
(1,600), `steps.jsonl.gz` (full per-step option probabilities, for Experiments
B/C/D), `run.json`. Seed is the cluster; CIs are 10,000-resample cluster
bootstraps over seeds (shared resample across conditions, so paired contrasts stay
paired).

## Primary endpoint — task success

| Condition | Goal | History | Success rate [95% cluster-bootstrap CI] |
|---|---|---|---|
| A1 | implicit | none | 0.000 [0.000, 0.000] |
| A2 | explicit | none | 0.000 [0.000, 0.000] |
| A3 | implicit | full | 0.000 [0.000, 0.000] |
| **A4** | **explicit** | **full** | **0.190 [0.113, 0.273]** |

**Only A4 ever completes the task**, at ~0.19, CI excluding 0 and non-overlapping
with the other three (which are *exactly* 0 across all 400 episodes each). This
replicates the K=15 calibration (A4 ≈ 0.23). Because A1–A3 are degenerate at 0,
every success contrast equals A4's rate and the interaction test on *success* is
uninformative — as the pre-registration anticipated, the interaction is read from
the milestone endpoint below.

**Seed heterogeneity (as predicted):** A4 completed at least once on **22 of 50
seeds**, with per-seed rates spanning 0.125–1.0 (two seeds ~1.0; 28 seeds never).
The start room / goal distance dominates whether a game is solvable at all.

## Secondary endpoint — greatest-ordered milestone (better powered)

| Condition | Mean greatest-ordered [95% CI] |
|---|---|
| A1 | 0.445 [0.325, 0.570] |
| A2 | 0.307 [0.185, 0.438] |
| A3 | 0.828 [0.735, 0.910] |
| A4 | 1.558 [1.155, 1.985] |

Pre-registered contrasts (all six exclude 0):

| Contrast | Estimate [95% CI] |
|---|---|
| A4 vs A1 | +1.113 [0.688, 1.567] |
| A4 vs A2 | +1.250 [0.860, 1.677] |
| A4 vs A3 | +0.730 [0.362, 1.123] |
| Goal main effect (A2+A4)−(A1+A3) | +0.593 [0.177, 1.028] |
| History main effect (A3+A4)−(A1+A2) | +1.633 [1.163, 2.130] |
| **Interaction (A4−A3)−(A2−A1)** | **+0.868 [0.520, 1.238]** |

## Findings

1. **Completion requires both goal and history.** A4 (explicit goal + full
   history) is the only condition that ever wins (0.19), and it dominates every
   other condition on milestone progress. This is the core result.

2. **History is the dominant driver** (history main effect +1.63, the largest
   contrast): carrying the episode's observation-action history forward is what
   moves Jev along the milestone chain.

3. **An explicit goal alone does not help — if anything it hurts.** A2 (goal,
   reactive) has the *lowest* milestone mean (0.31 < A1's 0.45). Told to
   refrigerate the food but with no memory that it is already carrying it, a
   reactive agent flails. The goal becomes useful only once there is history to
   maintain it against.

4. **The goal × history interaction is real and positive** (+0.87 [0.52, 1.24]):
   adding history helps far more when the goal is explicit — (A4−A3)=+0.73 versus
   (A2−A1)=−0.14. This is the pre-registered hypothesis confirmed on the
   well-powered endpoint: *history is valuable, and an explicit goal is what lets
   that history be composed into progress.*

5. **The M2 (transport) wall, revisited.** A3 reliably acquires (high M1) but
   stalls; only A4 crosses M2 and beyond. The deficit for everyone else is not
   picking the food up — it is keeping it moving toward the fridge without both an
   objective and a memory of having acted on it.

## Scope and caveats

- **This is an intervention, not the latent-goal benchmark score.** A2–A4 disclose
  the objective and/or supply history; only **A1** resembles the original
  zero-shot latent-goal setup, and A1 = 0, consistent with the earlier 0/50
  baseline. A4's 0.19 says "*with* the goal stated and history retained, Jev can
  sometimes compose the sequence," which is a statement about controllability, not
  about discovering the latent goal from sparse reward.
- **Seed-dominated.** Report the per-seed heterogeneity, not just the pooled rate;
  "solvable" games (near the goal) vs "trapped" starts drive most of the variance.
- **Zero-shot, narrative context throughout, `jev-latest`, no RL.** Per-episode
  choice is stochastic (see the determinism audit); success is a rate, and the K=8
  rollouts with 50 seeds are what make the CIs tight.
- **Success contrasts are degenerate** (A1–A3 exactly 0); the interaction and main
  effects are interpreted from the milestone endpoint, as pre-registered.

## Relationship to the thesis

Text-Nav's point is discovering a latent multi-step goal from sparse reward. Jev,
a per-step selector with no reward/value/exploration, cannot do that (A1 = 0). But
Experiment A shows the failure **decomposes**: give Jev the goal *and* let it carry
history, and it recovers partial sequential control (A4 ≈ 0.19, strong milestone
gains) — with history the larger lever and the goal acting chiefly by making the
history worth composing. Recognition, objective specification, and sequential
control are separable, exactly the decomposition the program set out to find.
