# Open-world Twenty Questions — results

An open-world extension of the closed-world BED loop (`bed.py`, `RESULTS.md`): the
secret need not be in the hypothesis set. The known set `H_known` starts empty and
grows only by expert admission; a Pitman–Yor catch-all hypothesis models the
unseen tail and doubles as the novelty detector. The design, the construction, and
the pre-registered predictions are in `OPENWORLD_PREREG.md`.

All numbers regenerate offline from the committed run directories and the cached
Noul matrix (`matrix_jev.json`), no API:

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_openworld \
    --games 2000 --zipf 1.1 --seed 7
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_openworld \
    --dir llm_rl_scripts/twenty_questions/jev/results/openworld_20261108_161009
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_openworld \
    --noise-grid --levels 0.0,0.05,0.10 --noise-seeds 3
```

The true space is the 158 benchmark objects; the stream draws from a Zipf law
(exponent 1.1, random object→rank, seed 7) over them, so the head stands in for
common species and the long tail for the unbounded remainder. 2000 games.

## Headline

The externalised belief + EIG planner that solved the closed world **also carries
the open world**, and the Pitman–Yor catch-all is a near-perfect novelty detector
when the answerer is self-consistent — degrading gracefully, not catastrophically,
as the answerer gets noisy. The common head is served autonomously; the expert is
spent almost only on genuine novelty, at a rate that falls as the pool fills.

| quantity | value (clean, deterministic oracle) |
|---|---:|
| distinct species discovered | **146 / 158** in 2000 games |
| Pitman–Yor fit | **σ = 0.200**, θ = 20.0 (heavy-tailed; σ=0 would be Dirichlet) |
| detector AUC (out-of-pool vs in-pool) | **1.000** |
| TPR at the confidence gate (0.90) | **0.986** (144/146) |
| FPR at the confidence gate | **0.000** (0/1854) |
| novelty misses | 2 |
| median fire turn (earliness) | **12** of 20 |
| expert calls | **144**, of which **0** false-positive |
| autonomous wins | **1854 / 2000** |
| autonomous share, first half → second half | **0.882 → 0.974** |

## 1. Discovery is heavy-tailed (P1 ✓)

The discovery curve `K_n` is sublinear and does not plateau: 146 of 158 species
appear in 2000 games, 12 never drawn. The Pitman–Yor sequential-partition MLE
lands at **σ = 0.200** (θ = 20.0) — a genuine power-law tail, not the Dirichlet
σ = 0 that a log-growth stream would give. The fitted `p_new(t)` tracks the
Good–Turing missing mass (singletons/t) over the stream (`fig1_discovery.png`).
"Exhaust the space" is therefore *not* the operating regime at this horizon —
novelty keeps arriving, so the detector is load-bearing, not a convenience.

## 2. The catch-all is a near-perfect detector on a clean oracle (P2 ✓)

With the self-consistent matrix oracle, `max_t p_unknown` separates out-of-pool
from in-pool secrets with **AUC 1.000**. At the operational confidence gate (0.90)
the detector fires on **144/146 = 0.986** of true-novelty games with **zero** false
positives across 1854 in-pool games, at a **median of 12 questions** — well inside
the 20-question budget, so the expert is called early, not after a wasted game.
The 2 misses are novel objects whose answer pattern happened to match a known
object closely enough to be (wrongly) guessed. Mechanism: a novel secret matches
no known Noul row and every by-name guess returns "no", and because guesses push
mass onto the ε-floored `unknown`, the catch-all accrues the posterior the known
objects shed (`fig2_detector_roc.png`).

## 3. Coverage is size-biased; expert load decays (P3 ✓)

Autonomous-resolution rate falls monotonically from the common head to the rare
tail — the system is self-sufficient on what it sees often and leans on the expert
for the rare (`fig4_coverage_by_rank.png`):

| frequency-rank bin (1 = most common) | autonomous-resolution rate | n |
|---|---:|---:|
| 1–20 | 0.986 | 1457 |
| 21–40 | 0.902 | 205 |
| 41–59 | 0.822 | 107 |
| 60–79 | 0.753 | 77 |
| 80–99 | 0.655 | 58 |
| 100–118 | 0.579 | 38 |
| 119–138 | 0.500 | 34 |
| 139–158 | 0.375 | 24 |

Expert load falls over the stream as the missing mass falls: the autonomous share
rises from 0.882 in the first 1000 games to 0.974 in the second
(`fig3_expert_load.png`). Necessary (true-novelty) calls dominate; false-positive
calls are ~0 on the clean oracle.

## 4. Robustness: a noisy answerer degrades the detector gracefully (P2/P4 stress)

The clean AUC of 1.000 is the *easy* case — a deterministic, self-consistent
oracle. The honest stress is a noisy answerer: Jev matched its own matrix on 99.2%
of attribute questions live (`RESULTS.md` §3), so we flip each attribute answer at
rates that bracket and over-stress that (5% and 10%, i.e. 6×–12× the observed
rate), reporting the spread over 3 noise seeds per rate (stream fixed; only the
flips vary). The ε = 0.05 background floor is the load-bearing calibration choice,
exactly as the ε-clamp was in the closed world.

Mean over 3 noise seeds (stream fixed, seed 7); AUC range in brackets. Jev's
observed live inconsistency was ~0.8%, so even the 0.10 row is ~12× over-stress.

| attribute-flip rate | detector AUC | TPR @0.90 | FPR @0.90 | expert calls | false-pos calls | wrong (missed novelty) |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 (deterministic) | 1.000 (0.9997) | 0.986 | 0.000 | 144 | 0.0 | 2.0 |
| 0.05 (~6× Jev) | 0.999 [0.999, 0.999] | 0.980 | 0.004 | 151.7 | 8.0 | 31.7 |
| 0.10 (~12× Jev) | 0.997 [0.996, 0.999] | 0.962 | 0.008 | 162.7 | 19.3 | 60.3 |

The detector's **ranking is essentially noise-proof**: AUC moves only 1.000 → 0.997
as the flip rate goes to 10%. What degrades is the *operating point*, not the
separation — a flipped answer can make a novel object resemble a known one closely
enough to be guessed, so missed novelties climb (2 → 60) and false-positive expert
calls rise modestly (0 → 19 of ~1800 in-pool games, ~1%). The pre-registered
falsification clause (AUC → 0.5, or novelty undetected) does **not** trigger: the
ε = 0.05 attribute-marginal background is not too permissive even at 12× the
observed noise. Noise is paid for in a few extra wrong guesses and wasted expert
calls, not in blindness to novelty.

## Caveats

* The **true space is finite (158)** standing in for an unbounded one; the
  perpetual-novelty regime here comes from the heavy-tailed *draw order* within
  the horizon, not from an infinite support. A genuinely unbounded oracle would
  need attribute rows for unseen species — which is precisely the expert's second
  job (supply/correct the Noul row), noted but not exercised here.
* The live `p_new` prior uses the PY parameters fit from the realised stream (a
  design study, not a deployment); the discovery curve and the detector ranking
  (AUC) do not depend on that choice, only the absolute `p_unknown` level the gate
  sees does.
* The detector inherits the oracle's calibration. §4 is the real test; the clean
  §2 is the ceiling.

## Reproduce

Run directory for the clean arm: `results/openworld_20261108_161009/` (figures +
`records.json` + `summary.json`). Noise grid:
`results/openworld_noise_grid_20261108_162415/summary.json`.
