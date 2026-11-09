# Open-world Twenty Questions — expert-onboarding study, results

**ENV-704 (Phase A).** The open-world loop admits a novel object for free — the expert is
called and the object's full Noul row appears in `H_known`. This study made admission
cost something and the elicited row *partial* (effort `E` of 44 attributes, by a
`diagnostic` or `random` policy; the rest left inert at 0.5; the world still answers from
the true matrix, via the `like_matrix` seam). The pre-registration
(`OPENWORLD_ONBOARDING_PREREG.md`) predicted diminishing returns with a modest `E` and an
interior cost optimum. **The run refutes both, and the honest finding is more useful:
onboarding attribute rows barely pays, because the loop's cheap fallbacks already resolve
recurrences.**

Run `openworld_onboarding_20261109_002035`, 2000 games, backend `soft`, 1854 recurrence
games, 145 species admitted. Clean-oracle numbers are deterministic (diagnostic
elicitation has no randomness).

## 1. Resolution is effort-invariant — name-guessing floors it (P1 refuted)

| policy | E | resolution | wrong | q/recurrence | onboard attrs | experts |
|---|---:|---:|---:|---:|---:|---:|
| diagnostic | 0 | 0.998 | 0.002 | 11.65 | 0 | 145 |
| diagnostic | 2 | 0.998 | 0.002 | 11.65 | 290 | 145 |
| diagnostic | 4 | **0.930** | **0.070** | 11.89 | 564 | 141 |
| diagnostic | 8 | **0.901** | **0.099** | 11.78 | 1136 | 142 |
| diagnostic | 16 | 0.998 | 0.000 | 9.52 | 2336 | 149 |
| diagnostic | 44 | 1.000 | 0.000 | 8.56 | 6336 | 144 |
| random | 0–16 | 0.998 | 0.002 | 11.65 | 0–2320 | 145 |
| random | 44 | 1.000 | 0.000 | 8.56 | 6336 | 144 |

Recurrences resolve at ~0.998 **whatever the onboarding effort**, including `E = 0` (no
attributes at all). The reason is that two cheap channels are untouched by onboarding: the
**category questions** are free and exact (asked, not elicited), and a **by-name guess**
is exact. So the loop pins the category and then name-guesses within it — ~11.6 questions,
within the 20-question budget — without needing a single attribute. Onboarding does not
buy resolution; it was never the binding constraint.

## 2. A partial diagnostic row is *harmful* — the collision danger zone

The striking result: `diagnostic` at `E = 4` and `E = 8` makes resolution **worse** than
admitting blind — 0.930 and 0.901, with wrong-guess rates of 0.070 and 0.099, against
0.002 at `E = 0`. A few elicited attributes give an object a confident signature on those
questions and 0.5 (uninformative) everywhere else; when a within-category neighbour shares
that short signature, the planner's posterior can cross the commit threshold on the
**wrong** object — a confident wrong guess. At `E = 0` no object has an attribute
signature, so the planner never commits on attributes and name-guesses safely; at
`E ≥ 16` the row is rich enough to actually separate neighbours. In between is a danger
zone where partial knowledge is worse than none. (Deterministic at 0% noise — a property
of the matrix and the loop, not a seed artefact; at 10% noise the dip persists and
`E = 8` falls to 0.893.)

## 3. The only real benefit is question-efficiency, and it is back-loaded (P1, P2)

What onboarding *does* buy is a modest efficiency gain: `diagnostic` q/recurrence falls
11.65 → 8.56 (~26%, about 3 questions). But the gain is **back-loaded**, not
diminishing-returns: nothing below `E = 16` (11.65 at E≤8, 9.52 at 16, 8.56 at 44). And it
is diagnostic-only — `random` elicitation is **flat at 11.65 until the row is complete**
(`E = 44`), because a random handful of attributes rarely includes the ones that separate
the species from its category neighbours. So "order matters" (P2) holds, but in the
deflating sense that random onboarding is wasted and diagnostic onboarding only helps once
it is nearly complete.

## 4. Onboarding doesn't pay — the cost optimum is E\*=0 (P3 refuted)

`total(E) = c_elicit · (E · admitted) + c_expert · expert_calls`. Expert calls are
~145 — the distinct species admitted — and **flat in `E`**: onboarding never lowers them,
because re-escalation is already ~0 (name-guessing resolves recurrences without a second
expert call). So the cost is a constant plus a term that only grows with `E`, and the
optimum is `E* = 0` for **every** cost ratio tested (c_elicit/c_expert ∈ {0.02, 0.1,
0.5}; normalised cost rises monotonically in each). There is no interior optimum: if you
are counting expert effort, the cost-minimising onboarding is none.

## What this says about the open-world idealization

ENV-702 and ENV-703 assumed free, full admission (`E = 44` here — which reproduces their
baseline bit-for-bit). This study asked whether that idealization flattered the loop. The
answer is **no, but for an unflattering reason**: the free-full-row assumption is nearly
harmless *because the loop barely needs the rows*. Category structure and exact
name-guessing carry recurrence resolution on their own; the elicited attributes add only a
~3-question efficiency gain, and only when near-complete. The onboarding cost the
idealization hid is real, but paying it is not worth it in this regime.

**Scope condition.** This holds because the question budget (20) comfortably exceeds the
within-category pool (~9 objects), so name-guessing fits. Tighter budgets, larger or
flatter categories, or a loop without the free category oracle would raise onboarding's
value — that boundary is where Phase B (two-oracle nested VOI, a cheap Noul oracle vs an
expensive human during the game) becomes the interesting question, and it is the named
follow-on.

## Caveats

* The inert value is 0.5 (maximally uninformative); an alternative (background marginal)
  would change the danger-zone magnitude but not its existence — a partial row that
  confidently matches a neighbour on a short signature is the mechanism, independent of
  the fill value.
* `soft` backend only (the BED resolver); the hard-elimination backends would strike on
  the elicited attributes and could turn a partial-row collision into a false escalation
  rather than a wrong guess — a variant worth a follow-up but orthogonal to the cost
  finding.
* Same stream, matrix, and PY fit as §3.11 / ENV-702; `E = 44` is the ENV-702 baseline.

## Reproduce

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.onboarding_openworld --self-test
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.onboarding_openworld --run \
    --games 2000 --efforts 0,2,4,8,16,44 --policies diagnostic,random --levels 0.0,0.10
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_onboarding \
    --dir llm_rl_scripts/twenty_questions/jev/results/openworld_onboarding_20261109_002035
```

`--self-test` proves the partial-row semantics and that full elicitation (`E = 44`)
reproduces the free-admission baseline bit-for-bit. The figure `fig_onboarding.png` shows
the efficiency-only benefit (with the danger-zone resolution dip) and the monotone cost.
