# Two-oracle nested VOI (ENV-704-B) — pre-registration

**The follow-on to ENV-704-A.** Onboarding attribute rows did not pay in the open-world
loop because the question budget (20) comfortably exceeded the within-category pool (~9),
so exact name-guessing floored resolution. ENV-704-A's scope note pointed at the regime
where fidelity *does* matter — tighter budgets, larger pools — and at the mechanism that
makes it matter: a **second, expensive answer source**. This study builds that.

During a game the questioner has two sources for each answer:

* a **cheap oracle** — answers attribute questions but **flips** each with probability
  `q` (a hurried or unsure user); category/guess answers are exact (logical);
* an **expensive human** — answers truthfully, at a cost of one *human query*.

The planner runs the ordinary soft-posterior BED loop (EIG-greedy questions) in a
**closed pool** of `N` objects (no catch-all, no onboarding — those confounds are out), on
a **tight budget** `T` questions. Each turn, after choosing the EIG-best question, a
**query policy** decides whether to spend a human query on it or take the cheap answer.
The nested-VOI question: *on which questions is the truthful answer worth paying for?*

## Query policies

Each is an **online threshold rule**; sweeping the threshold traces a curve in
(mean human queries per game, accuracy) space. The two anchors:

* `cheap_only` — never pay (the floor).
* `human_all` — always pay (the ceiling; equals a truthful game).

The contenders, each paying for the human iff a per-turn score passes threshold θ:

* `random@p` — pay with probability `p` each turn (the null policy: spend blind).
* `entropy@θ` — pay when the posterior entropy ≥ θ (pay when generally uncertain).
* `margin@θ` — pay when the commit race is tight, `top1 − top2 ≤ θ` (pay at the
  knife-edge, where a flipped answer most likely tips the final guess).
* `voi_eig@θ` — pay when the chosen question's **EIG ≥ θ** (pay on the most informative
  questions, where a corrupted answer does the most downstream damage). This is the
  nested VOI: value-of-truth ≈ P(cheap answer wrong) · (information at stake), and with a
  fixed flip rate the second factor is what varies.

## Metrics

Per (policy, threshold, regime), over the stream: **accuracy** (fraction of games that
correctly identify the secret within budget), **mean human queries / game**, **mean
questions / game**. The deliverable is the **accuracy-vs-human-queries Pareto frontier**:
at each level of human spend, which policy buys the most accuracy.

## Pre-registered predictions

* **P1 — the human is decisive in the hard regime.** `cheap_only` accuracy falls well
  below `human_all` at tight `T` / large `N` / meaningful `q`; the gap is the room a
  query policy has to work in. (In an easy regime — loose `T`, small `N` — the gap is
  small and the policies converge, echoing ENV-704-A.)
* **P2 — targeted beats blind.** `voi_eig` and `margin` dominate `random` and `entropy`
  on the frontier: at matched human-query spend they reach higher accuracy. Spending the
  expensive human on the high-information / knife-edge questions beats spending it
  uniformly or merely when the posterior is broad.
* **P3 — a little targeted truth goes far.** `voi_eig` reaches near-`human_all` accuracy
  at a **small** mean human-query spend (≪ T) — most of the fidelity benefit comes from
  correcting a few high-stakes answers, not from paying for every turn.
* **Falsification.** If `random` lies on or above the other curves, targeting buys
  nothing (P2 fails); if accuracy rises only linearly in human spend with no early knee,
  P3 fails; if `cheap_only ≈ human_all` in every regime, there is no VOI to study (P1
  fails) — which would itself extend ENV-704-A's "fidelity rarely matters here."

## Protocol

Closed pool subsampled from the 158 objects (seeded), secret uniform, soft posterior,
`make_likelihood` + the shared BED planner. Regimes: an **easy** control (`N=20`, `T=20`,
`q=0.15`) and a **hard** target (`N=40`, `T=10`, `q=0.15`), with a `q=0.25` check.
Thresholds swept per policy to trace each frontier; 1500 games per cell, seeded.
`--self-test` asserts: `human_all` equals a truthful run, `cheap_only ≤ human_all`
accuracy, accuracy is monotone in human spend within a policy, and the flip rate is
applied only to attribute answers.

Reproduce: `twooracle_20q --self-test`, then `twooracle_20q --run --out <dir>` and
`analyze_twooracle --dir <dir>`.
