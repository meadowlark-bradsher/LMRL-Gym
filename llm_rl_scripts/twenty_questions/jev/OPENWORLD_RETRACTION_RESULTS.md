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
games); noisy arms flip each attribute answer at 5% and 10%, 2 seed replicates. Run
under Python 3.9 (the `LLM_RL` env; 3.12's float `sum` moves `soft` by an ulp). The
survivor-set backends break ties among equally weighted survivors in the fixed
`known_hyps` order, so the grid does not depend on `PYTHONHASHSEED` (see *Redraw*,
below).

```
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.eval_openworld \
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

| backend | in-pool fail | detector AUC | wrong-guesses | false escalations | questions/game |
|---|---:|---:|---:|---:|---:|
| soft | 0.039 | 0.996 | 60 | 20 | 9.77 |
| hard | 0.175 | 0.913 | **0** | 325 | **8.77** |
| retract @margin 4 | 0.027 | 0.563 | 333 | 25 | 10.04 |
| tombstone @tol 3 | 0.009 | **0.499** | 764 | 6 | 11.14 |
| **modeswitch @K=3** | **0.149** | **0.926** | **0** | **277** | 9.44 |

`retract` and `tombstone` show the trap: the lowest in-pool failure (0.027, 0.009), but
AUC at or near the 0.50 chance line and hundreds of wrong guesses. `modeswitch` recovers
part of `hard`'s loss (0.175 → 0.149) while **keeping AUC at hard's level (0.926) and
making no wrong guesses** — and with fewer false escalations (277 vs 325), because a
recovered flip is a won game, not an expert call. At zero noise `modeswitch` is identical
to `hard`. It is the only reversible backend that stays out of the low-AUC band (see the
trade-off scatter in `fig_retraction_tradeoff.png`).

The **questions/game** column carries the price a failure-rate-only table hides. `soft`'s
low in-pool failure is bought with questions: it never eliminates, so it keeps asking to
drive the posterior down, and at 10% it costs ~1 extra question per game over `hard` (9.77
vs 8.77) — the same ε-floor mechanism that earns its accuracy. `modeswitch` recovers most
of `hard`'s failure gap at a *smaller* surcharge (9.44, below `soft`), because its reinstate
is a single by-name probe, not a sustained reweighting. The naive backends are the worst of
both: higher question cost (10.04, 11.14) *and* hundreds of wrong guesses. A scorecard for
this task should therefore require questions-used per game alongside failure rate and AUC,
as this one now does.

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

## What the scorecard leaves out

Of the axes this study measures — in-pool failure, detector AUC, wrong-guesses,
escalations, questions per game — `soft` leads on three: in-pool failure (0.039), AUC
(0.996), and escalations (20). It does **not** sweep them. It makes 60 wrong guesses where
`hard` and `modeswitch` make **zero**, and it asks more questions than `hard` (9.77 vs
8.77). Still, on the headline axes a graded posterior looks strong, and read naively the
table says "just use `soft`." Even setting the wrong-guess and question-cost gaps aside,
that reading omits the axis the discrete backends exist for.

`hard`, `retract`, `tombstone`, and `modeswitch` are not chosen for their error rates;
they are chosen for a **discrete, append-only, replayable audit trail** — the ability
to reconstruct and verify *why* each hypothesis was eliminated or revived, op by op
(the CMBS premise). `soft`'s ε-floored posterior is a cloud of weights; it cannot
produce that log, and the study never scores the one thing it cannot do. So `soft`'s
dominance here is dominance *on the measured axes only*.

The honest claim is therefore narrower than "soft holds both axes": **among discrete,
auditable backends, `modeswitch` is the resolution** — it recovers in-pool loss without
blinding the escalation gate or manufacturing wrong answers, where the naive reversible
backends do both. `soft` is a different design point that buys the measured axes by
giving up auditability, and this study does not weigh that trade — so it should not be
read as recommending `soft` over the auditable backends wherever a replayable belief log
is the requirement. Scoring auditability directly (replay cost, op-log verifiability,
answer-the-why queries) is the measurement this study did not take.

## Caveats

* The recovery is modest in magnitude (0.175 → 0.149 at 10%) because most flips strike
  a truth that reaches the boundary with no cheap single-contradiction signature; the
  *point* is that the recovery is free of detector cost and of wrong answers, which the
  naive backends are not.
* `modeswitch`'s clean behaviour equals `hard`'s exactly; its gains appear only under
  noise, a clean isolation of the reversal's effect.
* Same stream, PY fit, and noise grid as §3.11; arms differ only in the backend.
* The discrete backends' questions/game depends on how ties among equally weighted
  survivors are broken, by about ±0.3 (see *Redraw*). The other columns do not.

## Redraw (2026-11-11): hash-independent survivor order

The first version of this grid kept each discrete backend's survivors in a Python
`set` and read them in set order. That order follows Python's string hash, which is
random per process unless `PYTHONHASHSEED` is pinned, so the committed values for
`hard`, `retract`, `tombstone` and `modeswitch` were one draw that could not be
reproduced. The backends now read survivors in the fixed `known_hyps` order (the
order objects were admitted), and the whole grid was rerun under Python 3.9. It is
identical under `PYTHONHASHSEED=0` and `=1`. `soft` keeps no survivor set and is
bit-identical to the first run.

Cited values, 10% flips, mean over noise seeds 1–2:

| cell | metric | first draw | redraw |
|---|---|---:|---:|
| `hard` | questions/game | 8.82 | 8.77 |
| `modeswitch@3` | questions/game | 9.47 | 9.44 |
| `retract@4` | in-pool fail | 0.028 | 0.027 |
| `retract@4` | wrong guesses | 334 | 333 |
| `retract@4` | false escalations | 25.0 | 24.5 |
| `retract@4` | AUC | 0.563 | 0.563 |
| `retract@4` | questions/game | 10.04 | 10.04 |

Every other cited value is unchanged, including all of `hard`'s and `modeswitch`'s
in-pool failure, AUC, wrong-guess and escalation figures, `modeswitch@1`'s 0.173, and
all of `tombstone@3`'s. `retract` moves on more than question count because its revival
reads the survivor set, and on `retract@1` at 10% that is visible: in-pool fail 0.111 →
0.109, AUC 0.700 → 0.687, wrong guesses 138 → 142.5. No conclusion changes. The cost
ordering `hard` < `modeswitch` < `soft` holds, and so do the trap (`retract`,
`tombstone`) and the resolution (`modeswitch`).

**The tie-break order is not neutral.** Admission order is roughly discovery order,
which on a Zipf stream is roughly popularity order. So when survivors tie, the
planner's top-k guess candidates favour the likely secrets. `tie_order_probe.py`
replays the 10% cell under three orders
(`results/openworld_tie_order_20261111/probe.json`):

| backend | admission (committed) | alphabetical | reversed admission |
|---|---:|---:|---:|
| `hard` questions/game | 8.77 | 8.82 | 9.09 |
| `modeswitch@3` questions/game | 9.44 | 9.49 | 9.75 |

In-pool failure, AUC, wrong guesses and false escalations are identical under all
three. The question-cost ordering survives even the adversarial reversed order
(`modeswitch` 9.75 < `soft` 9.77), but only just. So read the discrete backends'
questions/game as ±0.3, set by the tie-break, and not as a property of the belief
update alone.

## Reproduce

Grid: `results/openworld_retraction_20261108_213525/summary.json` (all five backends, per-seed
rows and aggregates). Table and figure:

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_retraction \
    --dir llm_rl_scripts/twenty_questions/jev/results/openworld_retraction_20261108_213525
```

Tie-break probe (about 2 min on 12 cores):

```
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.tie_order_probe \
    --out llm_rl_scripts/twenty_questions/jev/results/openworld_tie_order_20261111/probe.json
```

Regression gate: the committed grid must reproduce cell by cell. Run it under two hash
seeds to confirm hash independence:

```
PYTHONHASHSEED=1 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_regression --committed-only
```

The explainer page `retraction_tradeoff.html` renders the whole arc for a first-time
reader — the trap and the mode-switch resolution, with the good-corner scatter from
this run.
