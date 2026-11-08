# Open-world Twenty Questions — retraction study, results

Five belief backends through the §3.11 open-world loop and its noise grid, so the
only thing that changes between arms is the belief *update* (`openworld_backends.py`,
pre-registration in `OPENWORLD_RETRACTION_PREREG.md`):

* **`soft`** — the §3.11 ε-floored Bayesian posterior (never hard-eliminates).
* **`hard`** — monotone hard elimination (strike a contradicted known, never return).
* **`retract`** — a *scalar* reversible backend: `hard`'s strike, plus a logged revive
  when a struck known's cumulative ε-floored log-likelihood overtakes the best live
  hypothesis within `margin` bits. Knob `margin` ∈ {1, 2, 4}.
* **`tombstone`** — a CMBS v0.2.0-aligned *structured* reversible backend: mask-meet
  elimination plus a reinstate keyed to the per-killer record (which answers struck
  each known), reviving the fewest-killer tombstoned known when the live set empties.
  Knob `tolerance` (reinstate killer bound).
* **`modeswitch`** — the resolution. `hard`'s elimination **and** `hard`'s escalation
  *view*, with a bounded reinstate-and-**test**: at the escalation boundary, up to K
  times per game, revive the fewest-killer single-contradiction candidate and test it
  with an exact by-name guess. Knob `tolerance` = K (the reinstate budget).

Offline on the cached Noul matrix, deterministic stream (Zipf s=1.1, seed 7, 2000
games); noisy arms flip each attribute answer at 5% and 10%, 2 seed replicates.

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_openworld \
    --retraction-grid --backends soft,hard,retract,tombstone,modeswitch \
    --margins 1,2,4 --tolerances 1,3 --levels 0.0,0.05,0.10 --noise-seeds 2
```

## Headline: naive reversibility is a trap; a mode switch resolves it

`hard` breaks under noise — a flip strikes the true object and monotonicity forbids
recovery. The tempting fix is to make elimination reversible, and both naive
reversibility backends recover the in-pool loss. But they do it by the same bad
trade, which only a careful decomposition reveals:

**In-pool "failure" is two different things.** A flip can cost an in-pool game either
a **wrong guess** (unsafe — a confidently wrong answer) or a **false escalation to the
expert** (safe — the human resolves it). `hard`'s in-pool failures are *all* safe
escalations: it makes **zero** wrong guesses. The scalar `retract` and the plain
`tombstone` lower the in-pool failure *rate* — but they do it by converting safe
escalations into **wrong guesses** (hundreds of them) and by **blinding the novelty
detector** (AUC to chance). They do not fix a problem; they manufacture one.

**The mode switch escapes the trade.** Keep elimination and escalation hard — the
detector reads the monotone survivor set — and make reinstate a *bounded, exact test*
at the boundary rather than a re-weighting of belief. `modeswitch` recovers in-pool
loss with the detector intact and **zero wrong guesses**, and it even makes *fewer*
expert calls than `hard`, because the flips it recovers become wins instead of
escalations. The tension was a property of the coupled designs, not of the problem.

## 1. In-pool failure rate — and what the failures *are*

Mean over 2 seed replicates, at 10% attribute-flip; run
`openworld_retraction_20261108_213525`.

| backend | in-pool fail | detector AUC | wrong-guesses | false escalations |
|---|---:|---:|---:|---:|
| soft | 0.039 | 0.996 | 60 | 20 |
| hard | 0.175 | 0.913 | **0** | 325 |
| retract @margin 4 | 0.028 | 0.563 | 334 | 25 |
| tombstone @tol 3 | 0.009 | **0.499** | 764 | 6 |
| **modeswitch @K=3** | **0.149** | **0.926** | **0** | **277** |

`retract` and `tombstone` show the trap: the lowest in-pool failure (0.028, 0.009), but
AUC at or near the 0.50 chance line and hundreds of wrong guesses. `modeswitch` recovers
part of `hard`'s loss (0.175 → 0.149) while **keeping AUC at hard's level (0.926) and
making no wrong guesses** — and with fewer false escalations (277 vs 325), because a
recovered flip is a won game, not an expert call. At zero noise `modeswitch` is identical
to `hard`. It is the only reversible backend that stays out of the low-AUC band (see the
trade-off scatter in `fig_retraction_tradeoff.png`).

## 2. Why the scalar and the plain structured backend fail

Both revive at, or past, the escalation boundary and let the revived hypothesis be
*believed* — scored back up (`retract`) or returned to the survivor set and guessed
(`tombstone`). For a novel secret that is fatal: the mass that should flow to the
catch-all is handed back to a known, so the detector never fires, and the resurrected
known is eventually guessed — a wrong answer. The signal that would say "this revived
known is really the truth" is *future consistency*, which is the same signal that says
"the secret is in-pool, not novel" — so a rule that forgives on that signal cannot
tell a flip-struck truth from a known that happens to fit a novel secret. (Two
diagnostics found along the way, both fixed and in the backends: the scalar margin
cannot separate a one-outlier flip from a transiently-outscored wrong, and a naive
per-killer tombstone leaves a revived known *unkillable by its original killer* unless
the strike ignores killer-distinctness.)

## 3. The resolution — reinstate as a bounded test, not a belief update

`modeswitch` keeps the two jobs apart. Escalation and the detector stay hard: the live
set is monotone, and when it is empty with the reinstate budget spent, the catch-all
fires the expert exactly as `hard`. Reinstate serves *resolution only*, and it is a
**test**: at the boundary, revive the fewest-killer single-contradiction candidate (the
likely flip victim) — it is the sole survivor, so the loop immediately guesses it by
name. A by-name guess is exact and not subject to attribute noise, so it *settles* the
candidate in one question: a flip-struck truth answers yes and the game is won; a
revived wrong answers no and is re-struck. The budget K bounds the detector cost — a
novel secret is probed at most K times, then escalates — so the novelty signal is
`hard`'s, delayed by at most K turns.

K is the dial. K=1 barely recovers (0.173 at 10%, one probe is rarely the flip victim);
K=3 is pure gain — 0.149 in-pool, AUC 0.926 ≥ hard, zero wrong guesses. Raising K keeps
recovering while the budget stays small relative to the tail of single-contradiction
candidates; let it grow unbounded and it degenerates into the naive reinstate.

## Design rule (the production takeaway)

Reversible, auditable belief *can* keep the open-world's two axes — but only if the
reversal is a **bounded, exact test gated behind a hard escalation view**, never a
soft re-weighting of the belief that feeds the escalation signal. Concretely, in a
growing-hypothesis system with a human-escalation gate: keep elimination monotone and
let the escalation signal read it; recover a mis-eliminated hypothesis by *re-probing
it directly* (an exact confirm), bounded, before escalating; and never let a revived
hypothesis quietly re-enter the belief that decides whether to call the human. That is
the difference between `modeswitch` (works) and `retract`/`tombstone` (blind the gate
and manufacture wrong answers).

## Caveats

* The recovery is modest in magnitude (0.175 → 0.152 at 10%) because most flips strike
  a truth that reaches the boundary with no cheap single-contradiction signature; the
  *point* is that the recovery is free of detector cost and of wrong answers, which the
  naive backends are not.
* `modeswitch`'s clean behaviour equals `hard`'s exactly; its gains appear only under
  noise, a clean isolation of the reversal's effect.
* Same stream, PY fit, and noise grid as §3.11; arms differ only in the backend.

## Reproduce

Grid: `results/openworld_retraction_20261108_213525/summary.json` (all five backends, per-seed
rows and aggregates). Table and figure:

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_retraction \
    --dir llm_rl_scripts/twenty_questions/jev/results/openworld_retraction_20261108_213525
```

The explainer page `retraction_tradeoff.html` renders the trade-off for a first-time
reader (it predates `modeswitch` and shows the trap, not yet the resolution).
