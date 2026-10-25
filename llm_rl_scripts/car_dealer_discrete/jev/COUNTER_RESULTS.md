# Forced-counter probe: are uninformed seller counters less acceptable? — results

**Date:** 2026-10-25 · **Pre-registration:** `PREREGISTRATION_COUNTER.md` (written 23:29:51Z, before any decision) · **Regime:** hard (r = 0.55) · **Aid:** prior-only

Every number is rendered from `results/jj_counter_20261025/analysis_counter.json` (verdict in
`analysis_counter_verdicts.json`, applied by code before reading). Regenerate:

```bash
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_counter_probe \
    results/jj_counter_20261025 --seed 4
```

## Design recap

Single-decision probe: the responder is shown the starter's actual logged opening (the 480 feasible
openings of the hard prior-only run) and a menu of **offers only**, so it must counter. Prior-only
aid → both roles see mirror-identical P(accept). K = 3 → 1440 decisions, 720 / 720 per cell,
0 failures. Paired type-pair-clustered bootstrap over 30 feasible pairs (B = 5000, seed 4).
Sanity: information mirror-symmetric = True; every decision an offer = True.

## Verdict

| | seller responds | buyer responds | Δ (seller − buyer) | 95% CI | |
|---|---|---|---|---|---|
| **C1 primary** — counter acceptable to starter | 0.708 | 0.771 | **-0.062** | [-0.175, +0.032] | includes 0 |
| C2 — counter's utility to the starter | 0.642 | 0.656 | **-0.014** | [-0.093, +0.057] | includes 0 |
| C3 — counter's own utility (self-interest) | 0.767 | 0.777 | **-0.010** | [-0.061, +0.042] | includes 0 |
| C4 — shown P(accept) of chosen counter | 0.545 | 0.513 | **+0.032** | [-0.039, +0.101] | includes 0 |

**C1: NOT SUPPORTED (CI includes 0) — adequately powered: a null is informative.**

The hypothesis fails in its strong form. Made to counter under identical information, uninformed
sellers and buyers produce counters that are equally acceptable, equally good for the starter,
equally self-interested, and carry equal displayed acceptance. **There is no role effect in
uninformed counter quality.** With 720 decisions per cell over 30 clusters this null is adequately
powered — it is a result, not an absence of one.

## C5 — the selection check is the real finding

| Cell × opening acceptable to the responder? | acceptable rate | n |
|---|---|---|
| seller_responds | opening acceptable to responder=True | **0.870** | 507 |
| seller_responds | opening acceptable to responder=False | **0.324** | 213 |
| buyer_responds | opening acceptable to responder=True | **0.965** | 513 |
| buyer_responds | opening acceptable to responder=False | **0.290** | 207 |

Whether a forced counter lands depends overwhelmingly on whether the **opening** was acceptable to
the responder: roughly **87–96 %** when it was, **29–32 %** when it was not — and that pattern is
**role-symmetric**. An opening the responder cannot accept signals a tough pair in which the
responder's own-good deals are bad for the starter, so any counter rarely clears r.

This explains the natural data. Responders counter mainly when the opening is bad for them — i.e.
in exactly the stratum where counters succeed ~30 % for either role — and the natural 0 % (seller)
vs 47 % (buyer) rested on roughly 25 and 38 such counters. Under the forced not-OK rates, 0 of 25
has probability ≈ 0.000 and ≥ 18 of 38 has probability ≈ 0.0125: the natural counts are tails of
a common selected process (natural countering being more selective still than the not-OK stratum),
not evidence of a role difference. **Selection plus small n, not role.**

## Descriptive: which deals, and a caveat

Top seller counters: P0-W0-D2×213, P2-W2-D2×176, P2-W2-D0×152, P2-W0-D2×85. Top buyer counters:
P0-W0-D0×188, P1-W0-D2×161, P2-W2-D2×142, P2-W2-D0×64. The chosen-deal distributions are **not** mirror images
(total-variation distance after mirroring the seller cell = 0.5097): under mirror-identical information
Jev-as-seller and Jev-as-buyer pick *different specific deals* of *equal quality*. Caveat: the
openings differ between cells and are not exactly mirrored, so this TVD is not a clean role test.
It is recorded as a curiosity about role framing, not a finding.

## What this closes

The hard-regime "seller-as-responder" asymmetry is now fully accounted for, in three steps:
1. It was not in learning — under the Bayes aid the two responder cells are indistinguishable
   (`HARD_RESULTS.md` §Exploratory).
2. It was in the prior-only baseline — the uninformed seller-as-responder did worse.
3. That baseline difference was **selection into tough openings plus small-n noise**, not a
   property of what uninformed sellers counter with (this probe).

No role effect survives at any level. The pre-registered responder effect (learning helps whoever
moves second) stands; every apparent role asymmetry along the way was sampling noise on symmetric
behaviour.

## Limits

Forced menu (no opt-out) — measures the counter *given* countering; contexts reused from one logged
run; prior-only aid; hard regime; one prompt; `jev-latest` pinned by name. The within-stratum gap at
"opening acceptable" (0.870 vs 0.965) was not a pre-registered contrast and carries no interval here.
