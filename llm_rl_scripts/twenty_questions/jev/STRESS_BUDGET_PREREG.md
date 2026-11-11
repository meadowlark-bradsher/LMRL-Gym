# ST-5 Tight budget against a large pool — pre-registration

**Status: DRAFT, not registered.** Frozen when tagged `prereg-stress-st5-<date>`.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.

## Question

Two production recommendations hold only because the budget was loose relative to the
pool. The production note says so in its preconditions:

* **P1 "spend the expensive source early and sparingly"** (ENV-704-B,
  `TWOORACLE_RESULTS.md`): `first@1` closes 70–73% of the VOI gap and `first@2`
  reaches the ceiling; targeting beats random spend by only ~+0.01.
* **"Skip onboarding"** (ENV-704-A, `OPENWORLD_ONBOARDING_RESULTS.md`): E* = 0 because
  20 questions comfortably cover by-name guessing through a within-category pool of
  ~9.

This study sweeps budget against pool size and asks where each rule flips.

## The coordinate

Results are stated against one coordinate, **slack**:

  slack = T − Q̄(N)

where T is the budget and Q̄(N) is the mean questions per game that the truthful,
unlimited-budget `soft` loop needs on that pool. Q̄ is measured per pool, so it already
absorbs the pool's crowding. The study's main claim is that slack, not T or N alone,
predicts when each rule flips.

## Setup

Pools from ST-1's generator at ρ = ρ_real: N ∈ {40, 158 (real), 500, 2000}, plus
budgets T ∈ {8, 10, 12, 15, 20}. For N = 40, use the existing `twooracle_20q.py` pool
sampler for comparability. 3 pool seeds.

**(a) Two-oracle.** `twooracle_20q.py`, closed pool, `soft`, cheap flip q = 0.15 and
0.25. Policies: `cheap_only`, `human_all`, `first@1`, `first@2`, `first@3`, and
`random@p` with p chosen to match each `first@k`'s mean spend. 1500 games per cell, as
ENV-704-B.

**(b) Onboarding.** `onboarding_openworld.py` on the same pools, run in accumulation
mode: effort E ∈ {0, 2, 4, 8, 16, 44} (ENV-704-A's grid), `diagnostic` and `random`
policies, flip rate 0 and 0.05, ENV-704-A's cost model and cost ratios
c_elicit/c_expert ∈ {0.02, 0.1, 0.5}.

## Metrics

* (a) VOI room (`human_all` − `cheap_only`); fraction of the gap captured by
  `first@1`/`first@2`; the number of human queries needed to reach 95% of the ceiling;
  targeting edge (`first@k` − matched `random`).
* (b) Total cost against E; E* per cell; re-escalation rate of admitted species.
* Q̄(N) and its spread per pool; slack per cell.

## Predictions

* **P1 — VOI room is a function of slack.** VOI room is below 0.01 whenever
  slack ≥ 6, and rises as slack falls toward 0. Plotted against slack, the
  (N, T) cells collapse onto one curve per flip rate: no cell is more than 0.03 from a
  monotone fit of VOI room against slack.
* **P2 — "two queries suffice" has a slack floor.** `first@2` reaches ≥ 95% of the
  ceiling in every cell with slack ≥ 2. Below slack 2, the human queries needed to
  reach 95% of the ceiling exceed 2 and keep rising as slack falls.
* **P3 — targeting stays a small lever** (lower confidence). The targeting edge grows
  as slack shrinks but stays below 0.05 in every cell. If this holds, the production
  advice not to build a per-question targeting scorer survives tight budgets.
* **P4 — onboarding starts to pay below a slack threshold.** E* > 0 in cells with
  slack < 3 at the cheapest elicitation ratio (c_elicit/c_expert = 0.02), and E* = 0
  in every cell with slack ≥ 6 at every ratio, matching ENV-704-A. The mechanism:
  ENV-704-A's expert calls were flat in E because name-guessing resolved recurrences
  without re-escalating. Once the budget cannot cover the name-guessing, recurrences
  re-escalate, and a row that shortens the game saves expert calls. The partial-row harm ENV-704-A found at E = 4/8 shrinks as slack shrinks,
  because a thin row is then the only thing that resolves recurrences within budget.

## What would falsify, and what it would mean

* P1 failing (cells at the same slack disagree by more than 0.03): slack is the wrong
  coordinate. The production note's "budget-to-pool ratio" precondition then needs a
  different statement, likely one involving the pool's crowding directly.
* P3 failing (the targeting edge exceeds 0.05): in tight regimes a targeting heuristic
  earns its build cost, and the P1 production rule gets a regime split.

## Note

This study does not change any code path beyond pool construction. Both harnesses run
as committed, so the ENV-704-A and ENV-704-B cells inside this grid (N = 40 and
T = 10 for (a), the real pool at T = 20 for (b)) must reproduce their committed numbers
to within their quoted ±0.02 sampling noise. That is the regression check.
