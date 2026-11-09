# Two-oracle nested VOI (ENV-704-B) — results

The follow-on to ENV-704-A, which found onboarding doesn't pay when the budget is loose
relative to the pool, and pointed at the regime where fidelity *does* matter: tighter
budgets, larger pools, and a **second, expensive answer source**. Here the questioner runs
the ordinary soft-posterior BED loop in a closed pool, answered by a **cheap oracle** that
flips each attribute answer with probability `q`; at a cost of one *human query* it can
instead get the truth. The nested-VOI question: on which questions is the truthful answer
worth paying for?

Run `twooracle_20261109_011300`, 1500 games/cell, soft posterior.

## 1. The human is decisive only in the hard regime (P1 confirmed)

| regime | cheap-only (floor) | human-all (ceiling) | human q/game | VOI room |
|---|---:|---:|---:|---:|
| easy  N=20, T=20, q=0.15 | 1.000 | 1.000 | 1.00 | **0.000** |
| hard  N=40, T=10, q=0.15 | 0.711 | 0.818 | 1.64 | **+0.107** |
| hard  N=40, T=10, q=0.25 | 0.629 | 0.818 | 1.64 | **+0.189** |

With a loose budget (easy) the cheap oracle already reaches 1.000 — extra questions absorb
the noise, so the human buys nothing, exactly ENV-704-A's lesson. Tighten the budget and
enlarge the pool and a real gap opens (0.107, rising to 0.189 as the cheap source gets
noisier). That gap is the room a query policy has to work in.

## 2. Targeting beats blind — but only a little (P2 weakly supported)

The accuracy-vs-human-spend frontier (`fig_twooracle.png`). At matched human spend
(~1.0 query/game), front-loading the human on the earliest, highest-EIG attribute question
(`first@1`) beats spending the budget at random:

| policy | human q/game | accuracy (q=0.15) | accuracy (q=0.25) |
|---|---:|---:|---:|
| random (p=0.60) | ~1.00 | 0.780 | 0.753 |
| **first@1** | 1.00 | **0.789** | **0.761** |

The edge is real but small (+0.009, +0.008). The reason is visible in the threshold
policies (`entropy`, `margin`, `voi_eig`): they are **bang-bang** — every threshold either
pays on *all* attribute questions (1.64/game → ceiling) or none (→ floor), with no
intermediate operating point. That is because the per-question score barely varies at the
moments attribute questions are actually asked: the planner reaches an attribute question
only after category-narrowing, in a similar belief state each time, so every attribute
question sits at nearly the same EIG / entropy / margin. There is little to target. The
dominant lever on the accuracy-cost tradeoff is **how many** human queries you buy, which
`random` already traces; **which** ones matters only at the margin.

## 3. A little well-placed truth goes far (P3 confirmed)

The strong result is how cheaply the gap closes. `first@1` — a single human query, spent
on the first in-pool attribute question — captures most of the VOI:

* q=0.15: 0.711 → **0.789** of a 0.818 ceiling — **73%** of the gap, with 1 query vs the
  ceiling's 1.64.
* q=0.25: 0.629 → **0.761** of 0.818 — **70%** of the gap.
* `first@2` reaches the ceiling (0.818) exactly — two well-placed human queries are as
  good as paying on every turn.

So the nested-VOI answer is concrete: **pay for the human early and sparingly** — one or
two truthful answers on the first within-pool attribute questions (the highest-EIG, most
pivotal ones) get you nearly the full benefit of a fully truthful game. Paying on every
turn is wasteful; paying at random leaves ~1 query of value on the table.

This is a mild positive for the EIG heuristic in an allocation role: front-loading by EIG
(=`first`, since EIG falls over a game) is the best policy found, so EIG does track where a
truthful answer is most valuable — modestly, because the asked questions' VOI is clustered.

## Where this connects

ENV-704-A and ENV-704-B bracket the same boundary. When the budget is loose relative to
the pool, neither onboarding (A) nor an expensive oracle (B) pays — the cheap loop already
resolves. When the budget is tight and the pool large, fidelity is decisive (B's 0.11–0.19
gap), and the right way to spend a scarce truthful-answer budget is **early and small**,
front-loaded by EIG.

## Caveats

* The targeting edge is small because the hard regime asks only ~1.64 attribute questions
  per game — few decision points to mis-target. A regime with flatter categories or larger
  within-category pools (more attribute questions per game) would widen `first`'s advantage
  over `random`; this is the natural next sweep.
* The threshold policies' bang-bang behaviour is a methodological finding, not a bug: a
  per-turn score with little spread at the decision points cannot be thresholded into an
  intermediate policy, so a **budget** policy (`first@B`) is the right tool for a matched
  comparison.
* Closed pool, soft posterior, cheap noise on attribute answers only (category/guess
  exact). Pool subsampled from the 158 objects, seeded.

## Reproduce

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.twooracle_20q --self-test
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.twooracle_20q --run --games 1500
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_twooracle \
    --dir llm_rl_scripts/twenty_questions/jev/results/twooracle_20261109_011300
```

`--self-test` asserts `human_all` equals a truthful run, `cheap_only ≤ human_all`, accuracy
is monotone in human spend within a policy, and flips hit attribute answers only. The
figure shows the accuracy-vs-human-spend frontier for the two hard regimes.
