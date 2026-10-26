# First live J/J smoke — results

**Date:** 2026-10-25 · **Game:** `car-dealer-discrete-v0.1` · **Model:** `jev-latest` · **Condition:** pure Jev-only, no aid, no salience

The first live run of the wired policy: 3 episodes, 3 sampled type pairs (`--pairs 3 --seed 7`),
easy regime (r = 0.40), buyer-start, K = 1. It predates the `condition` field in the logs. Its purpose was
to prove the pipeline live; it also produced the study's first observation. Rendered from
`results/jj_smoke_20261025/analysis_smoke.json`; regenerate:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_salience results/jj_smoke_20261025
```

| | value | tag |
|---|---|---|
| Outcomes | {'walk': 3} | exact-on-logs |
| Offers made | 18 | exact-on-logs |
| Offers that were the offerer's ideal | 18 / 18 | exact-on-logs |
| Offers worth ≥ r to the recipient | 0 / 18 | exact-on-logs |
| Buyer offered | {'P0-W2-D2': 9} | exact-on-logs |
| Seller offered | {'P2-W0-D0': 9} | exact-on-logs |
| Who walked | {'start=buyer walker=buyer': 3} at decision [7] | exact-on-logs |
| Confidence at first offer | 0.723 [0.62, 0.87] (n=6) | stochastic |
| Confidence at last offer | 0.352 [0.32, 0.40] (n=6) | stochastic |
| Confidence at walk | 0.747 [0.73, 0.77] (n=3) | stochastic |

## What it showed

Every episode walked despite the easy regime guaranteeing an IR deal. Each seat offered **only its own
ideal** — the buyer `P0-W2-D2`, the seller `P2-W0-D0` — six proposals burned, and the buyer walked at
decision 7. Confidence decayed as the stuck state repeated, then flipped to WALK. This was the first
observation of the no-concession behaviour that the whole subsequent program dissects.

## Status

Superseded as a baseline by `SALIENCE_RESULTS.md`, whose baseline arm reproduces the same behaviour at
12 episodes with both starting sides. Kept because it is the chronological origin of the finding and the
first live validation of the pipeline (0 API errors). n = 3; nothing here is a rate.
