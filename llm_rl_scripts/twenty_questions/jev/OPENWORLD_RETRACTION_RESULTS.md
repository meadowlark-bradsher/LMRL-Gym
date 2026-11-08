# Open-world Twenty Questions — retraction study, results

Three belief backends through the §3.11 open-world loop and its noise grid, so the
only thing that changes between arms is the belief *update* (`openworld_backends.py`,
pre-registration in `OPENWORLD_RETRACTION_PREREG.md`):

* **`soft`** — the §3.11 ε-floored Bayesian posterior (never hard-eliminates).
* **`hard`** — monotone hard elimination (strike a contradicted known, never return).
* **`retract`** — the reference reversible backend: `hard`'s exact strike and
  detector, plus a logged *revive* (un-eliminate) when a struck known's accumulated
  ε-floored log-likelihood overtakes the best live hypothesis by within `margin`
  bits. One knob, `margin`, swept over {1, 2, 4}.

Offline on the cached Noul matrix, deterministic stream (Zipf s=1.1, seed 7, 2000
games); noisy arms flip each attribute answer at 5% and 10%, 2 seed replicates.

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_openworld \
    --retraction-grid --backends soft,hard,retract --margins 1,2,4 \
    --levels 0.0,0.05,0.10 --noise-seeds 2
```

## Headline: reversibility recovers the loss, but a scalar margin can't decouple recovery from novelty detection

`hard` breaks under noise exactly as predicted — a flipped answer strikes the true
object and monotonicity forbids recovery. Retraction *does* recover the in-pool
loss: at a wide margin `retract` drives the in-pool failure rate back to `soft`'s
level or below. **But the same margin that revives a flip-struck truth also revives
genuine wrongs** — a scalar log-likelihood margin cannot tell a truth struck by one
outlier answer from a wrong that is briefly out-scored — so wider margins degrade the
novelty detector (more mis-guesses, fewer escalations), *even on a clean oracle*.
There is no free-lunch margin; recovery and novelty detection trade off along it.
That tension is the result, and it is what motivates structured retraction.

## 1. In-pool failure rate — retraction recovers what hard loses (P2 ✓, P3 ✓-with-cost)

The metric: of the in-pool games (the secret *was* a known object, so it was
resolvable), the fraction the backend failed to resolve (a wrong guess or a
false-positive expert call). Mean over 2 seed replicates; run
`openworld_retraction_20261108_182642`.

| backend | clean | 5% flip | 10% flip |
|---|---:|---:|---:|
| soft | 0.000 | 0.023 | 0.039 |
| hard | 0.006 | 0.100 | **0.175** |
| retract @margin 1 | 0.007 | 0.064 | 0.111 |
| retract @margin 2 | 0.007 | 0.065 | 0.110 |
| retract @margin 4 | 0.007 | 0.020 | **0.027** |

`hard` climbs steeply — a flip strikes the truth and monotonicity forbids recovery.
`soft` degrades gently. `retract` sits between and moves toward `soft` as the revival
margin widens; at margin 4 it recovers essentially all of `hard`'s loss (0.027, below
`soft`'s 0.039). See the left panel of `fig_retraction_tradeoff.png`. Taken alone this
says retraction is a free win — which §2 refutes.

## 2. The cost: novelty detection degrades with the revival margin

Reviving a struck known keeps survivor sets from collapsing (→ mis-guesses) and from
emptying on a novel secret (→ its mass never flows to the catch-all, so the expert is
not called and the species is not admitted). The cleanest measure is the **detector
AUC** — how well `max_t p_unknown` separates out-of-pool from in-pool secrets:

| backend | detector AUC, clean | detector AUC, 10% | clean wrong-guesses | un-eliminates/game, 10% |
|---|---:|---:|---:|---:|
| soft | 1.000 | 0.996 | 2 | 0.00 |
| hard | 1.000 | 0.913 | 0 | 0.00 |
| retract @margin 1 | 0.674 | 0.700 | 198 | 1.79 |
| retract @margin 2 | 0.650 | 0.681 | 238 | 2.25 |
| retract @margin 4 | **0.525** | 0.563 | 537 | 6.69 |

`soft` and `hard` both detect novelty near-perfectly on a clean oracle (AUC 1.000).
`retract`'s detector is wrecked **at zero noise** — AUC 0.52–0.67, near the 0.50 chance
line — and it is wrecked *more* exactly where in-pool recovery is *best* (margin 4,
AUC 0.525, in-pool failure 0.027). Recovery and detection move in opposite directions
along the one margin (right panel of `fig_retraction_tradeoff.png`); there is no
setting that keeps both good.

The mechanism is specific. The true object often sits on many soft 0.6–0.7
likelihoods, so its cumulative log-likelihood is modest; a wrong object contradicted
once but otherwise agreeing can transiently out-score it. A scalar margin cannot
separate "struck by one outlier" (a flip-struck truth, worth reviving) from
"transiently ahead but contradicted on its merits" (a wrong, not worth reviving), so
any margin wide enough for the first admits the second — and the revived wrongs are
exactly what suppress the novelty signal, collapsing the AUC.

## 3. What this motivates — structured retraction (the follow-on)

The reference backend is deliberately the simplest reversible policy, and it reveals
the limit of a scalar trigger rather than hiding it. The distinction it cannot make
is exactly what **asserted-dead killers / per-killer tombstones** make: a retraction
keyed to *which* evidence struck a hypothesis (one outlier answer vs a pattern of
contradiction), not to a cumulative score. That is the model CMBS v0.2.0 implements
(`retract` / `tombstone` ops). The natural next step is a fourth backend wrapping the
CMBS store, measured against this scalar reference on the same grid: the question is
whether structure buys the in-pool recovery *without* the novelty-detection cost.

## Caveats

* The reference revival trigger is a scalar margin by design; its clean-oracle
  deviation from `hard` is a property of that trigger, not a bug, and is the point of
  §2. The pre-registration's P3 falsification clause ("the reversal trigger is wrong,
  not that reversibility is worthless") is what this study finds.
* Same stream, PY fit, and noise grid as §3.11; arms differ only in the backend.
* `retract`'s clean behavior would equal `hard`'s under a trigger that never fires on
  a trusted oracle — which a scalar cumulative-log-likelihood margin is not.

## Reproduce

Grid: `results/openworld_retraction_20261108_182642/summary.json` (every cell's
per-seed rows and aggregates). Table and figure:

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_retraction \
    --dir llm_rl_scripts/twenty_questions/jev/results/openworld_retraction_20261108_182642
```

The explainer page `retraction_tradeoff.html` renders the same trade-off for a first-
time reader.
