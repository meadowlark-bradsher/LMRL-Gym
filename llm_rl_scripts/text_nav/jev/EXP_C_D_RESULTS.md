# Experiments C & D (Mode B) — zero-API reanalysis of the main-run logs

**Status:** Both computed entirely from the committed main-run step log
([`results/main_A_20261025/steps.jsonl.gz`](results/main_A_20261025/steps.jsonl.gz),
31,720 steps, 1,600 episodes) — **no new Jev calls**. Experiment C's pairwise /
transitivity core needs a bounded live probe (built, not yet run — see end).
2026-10-25.

## Provenance

```bash
python -m llm_rl_scripts.text_nav.jev.analyze_exp_c       results/main_A_20261025/steps.jsonl.gz
python -m llm_rl_scripts.text_nav.jev.analyze_exp_d_modeB results/main_A_20261025/steps.jsonl.gz
# -> exp_c_states.csv, exp_c_summary.json, exp_d_modeB.json (sources of the figures)
```

---

## Experiment C (free layer) — fixed-state preference *stability*

The reactive (H0) conditions have no history, so the policy input at a state is
exactly (context, observation, admissible set). Every time the agent re-encounters
an identical such state across the K = 8 rollouts and across episodes, we get an
independent draw of Jev's full-menu Choice distribution at a **frozen** context.
Over **262 fixed states** (reactive, ≥ 8 samples, ≥ 3 options):

| Metric | Mean | Range |
|---|---|---|
| Argmax stability (modal choice's share) | **0.963** | 0.45 – 1.00 |
| Fraction of states with a **unanimous** argmax | **0.805** | — |
| Top-2 set stability | 0.941 | 0.42 – 1.00 |
| Mean per-option probability SD across repeats | **0.011** | 0.00 – 0.027 |
| Mean confidence | 0.555 | 0.065 – 1.00 |

**Reading.** At a frozen full-menu state Jev is nearly deterministic — the
distribution barely moves (SD ≈ 0.01) and the top choice is unanimous 80% of the
time. This *reconciles the determinism audit*: per-episode trajectories diverged
not from gross randomness but from the ~4% of near-tie states where a small wobble
flips the argmax, and one flip changes all later states. A stable full-menu ranking
therefore exists — which is the precondition that makes the pairwise-consistency
question worth asking.

**Not answered by this layer** (needs live probing): pairwise-in-isolation
preferences, transitivity / cycles, presentation-order effects, and
menu-dependence. A single full-menu Luce vector is self-rankable by construction;
only pairwise data can reveal *non-transitivity*.

---

## Experiment D, Mode B — EIG yardstick, and why goal-directedness replaces it

**EIG is vacuous in this game family.** Every room has a unique description
(**0 of 1,209 distinct observations map to >1 room**), so the agent is localized
the instant it reads a room, and with deterministic observations the expected
information gain of every action collapses to ~0 after the opening move. Scoring
Jev against EIG would be scoring against a near-constant — the design's `b0`
regime (D.1) made exactly this prediction. So we apply the yardstick D.5 names for
the *exploit* regime instead: **goal-directedness** — does Jev's choice match a
shortest-path-optimal action toward the goal under the known map?

Coverage of the optimality model is 0.97–1.00 (the computed optimal action is
in the admissible set almost always), so the yardstick is faithful. Chance = the
rate a uniform-random admissible pick is optimal (|optimal|/|admissible|).

| Condition | argmax-optimal | chance | **lift over chance** | prob-mass on optimal |
|---|---|---|---|---|
| A1 implicit, reactive | 0.255 | 0.301 | **−0.046** | 0.294 |
| A2 explicit, reactive | 0.206 | 0.285 | **−0.080** | 0.298 |
| A3 implicit, history | 0.247 | 0.216 | **+0.031** | 0.219 |
| **A4 explicit, history** | 0.343 | 0.211 | **+0.132** | 0.330 |

**Reading.** Only **A4** is meaningfully goal-directed (+0.13 above its own
chance). A1 and A2 are *at or below chance* — their per-step choices are no better
than random with respect to the goal, and **A2 (goal-alone) is the worst**,
actively below chance. A3 (history-alone) is barely positive. This independently
corroborates the main run through a per-step measure rather than terminal success:
the goal × history combination is the only setting that produces genuinely
goal-aligned choices, and an explicit goal without memory is, if anything,
counterproductive — the same pattern the milestone interaction showed.

(Chance differs by condition because the conditions visit different state
distributions; the **lift over each condition's own chance** is the fair
comparison, and it is unambiguous.)

---

## Convergent picture (three independent measures)

| Measure | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| Task success (main run) | 0.00 | 0.00 | 0.00 | **0.19** |
| Mean greatest-ordered milestone | 0.45 | 0.31 | 0.83 | **1.56** |
| Goal-directedness lift over chance | −0.05 | −0.08 | +0.03 | **+0.13** |

All three agree: **A4 alone shows goal-aligned sequential behavior; explicit goal
without history (A2) is the weakest on every measure.** Recognition, objective
specification, and sequential control are separable, and only their conjunction
yields control.

---

## Experiment D, Mode A — belief-space explorer baseline (no API)

The principled baseline (design D.3): a scripted agent
([`explorer_policy.py`](explorer_policy.py)) that does what Jev structurally
cannot — build a map, explore unexplored exits (information-greedy reduces to
frontier exploration when observations are deterministic and room-unique), then
plan to the goal (fetch → fridge → open → insert → close) along shortest known
paths. It uses only observation-derived information (room identity + admissible
set) plus the stated goal/target; facts score milestones only. Run on the **same
50 matched seeds**, real PO env, 20-step limit.

| Policy | Success | Mean greatest-ordered | Mean steps (wins) |
|---|---|---|---|
| Jev A4 (explicit goal + history) | 0.19 | 1.56 | — |
| Jev A1/A2/A3 | 0.00 | ≤ 0.83 | — |
| **Belief-space explorer (Mode A)** | **1.000 (50/50)** | **5.00** | 16.9 (15–19) |

**The task is fully solvable zero-RL by a belief-space planner** — 50/50, every
seed to M5, all within the 20-step budget. So Jev's 0→0.19 is **not** about task
difficulty or an unfair benchmark: it is precisely the absence of exploration,
belief/map-tracking, and planning. Supplying those takes success from 0.19 to 1.0.

**But the budget bites.** Wins take 15–19 steps (mean 16.9) of the 20 allowed —
exploration overhead consumes most of the budget. The task is solvable, not
trivially so; a less efficient explorer, a larger map, or a worse start would
begin to time out. This quantifies how much of Jev's failure the 20-step cap could
mask: even an optimal-planning agent has little slack. Logs:
[`results/dmodeA_20261025/`](results/dmodeA_20261025/).

## Experiment C (live layer) — pairwise transitivity & order effects

[`probe_exp_c.py`](probe_exp_c.py) froze 6 three-option triads each from the A1
(bare reactive) and A2 (goal-disclosed reactive) framings — 12 triads, ~1,008 Jev
calls, the environment never advanced. (A4 is not probeable as a fixed state: its
history makes states unique and unrepeatable, and A4-minus-history *is* A2, so A2
carries the explicit-goal framing.) Each triad: full-menu ×12, and each of its 3
pairs ×12 with presentation order balanced. Logs:
[`results/expc_pairwise_20261025/`](results/expc_pairwise_20261025/).

| Finding | Value |
|---|---|
| 3-cycles (A≻B≻C≻A) detected | **0 / 12 triads** |
| Pairs with strong position bias (\|order effect\| ≥ 0.5) | **5 / 36** |
| …of which the symmetric-movement pair `go south`/`go west` | 4 / 5 (order effect 1.0) |

**Transitive within a fixed context.** No cycles in any triad: Jev's
pairwise preferences admit a scalar ranking. Where an option is clearly better or
worse (take/drop an object, open a chest vs. move), the preference is strong and
order-independent (win-probs pinned at 0 or 1, order effect ≈ 0). *The harmonic
residual instrument has nothing to certify here — there is no cyclic structure.*

**Ties are broken by position, not preference.** The only large order effects fall
on pairs Jev is indifferent between — above all two interchangeable movement
directions (`go south` vs `go west`), where order effect = 1.0: whichever is shown
first is chosen 100% of the time. This is presentation/position bias on indifferent
options, **not** non-transitivity — exactly the confound the design (§7.3) warned
not to mistake for harmonic preference structure. It also explains the ~4% of
near-tie argmax flips the free layer saw: those are indifferent pairs resolved by
position.

**Context (A1 vs A2) comparison is inconclusive here.** The most-resampled
3-option states are low-stakes navigation/distractor triads (bowl, chest, movement)
far from the goal object, and A1 and A2 look alike on them. The decisive
milk-vs-distractor state was not among them. A targeted follow-up probing
hand-picked snapshots (e.g. the living-room `take milk` vs `take fresh food` vs
`go` triad) under A1 vs A2 framing would test goal-driven context sensitivity
directly — `probe_exp_c.py` supports it via explicit snapshot selection.
