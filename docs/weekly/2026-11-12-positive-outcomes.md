# Weekly recap, 2026-11-05 to 2026-11-12: the positive outcomes

Fork `main` went from #35 to #73 this week: 39 merged PRs across Car Dealer, open-world
Twenty Questions (20Q), Split Deal and the 20Q stress tests. This note collects the
results that came out in our favour, with enough detail to act on them. It does not
cover the negative results, which were most of the negotiation work (see *Not covered*
at the end).

**Where the numbers come from.** Every number below is quoted from a committed result
note, which names its run directory. Each section lists its source. Two checks were made
for this note:

- The retraction and audit numbers were checked against the 2026-11-11 redraw (#72),
  which changed how ties among survivors are broken. The cited values here are the
  redrawn ones.
- The missing-mass certificate table was regenerated from its committed records
  (`mm_certificate --analyze`) and matches #73 exactly.

## The headline: one design is safe, accurate and auditable

For a question-asking system that can hand off to a human, the 20Q work this week
reduced the main architectural choice to one decision. Use a **discrete backend that
eliminates hypotheses hard, gates the human on that hard state, and keeps an append-only
hash-chained log of every elimination**. Sections 1 and 2 are the evidence: the
`modeswitch` backend gets the accuracy back without hurting safety (§1), and the same
discrete design is fully auditable at no extra cost (§2). Section 3 is the guardrail
document that turns this into a build order.

## 1. The retraction trade-off went away (`modeswitch`, #41–#43)

**The problem.** Under answer noise, hard elimination sometimes strikes the true object
and can never get it back. The obvious fix is to make elimination reversible. Both
reversible designs tried (`retract`, a scalar likelihood race, and `tombstone`, a
CMBS-style reinstate) lower the in-pool failure rate, but they do it by turning safe
expert escalations into confident wrong guesses, and they push the novelty detector to
chance.

**The fix.** `modeswitch` keeps elimination and the escalation view hard. When the live
set empties, it revives the hypothesis with the fewest contradictions and tests it with
an exact by-name guess, at most K times per game, before calling the expert. A by-name
guess isn't subject to attribute noise, so one question settles the candidate.

Ten percent attribute flips, mean over 2 noise seeds, 2,000 games per run:

| backend | in-pool fail | detector AUC | wrong guesses | false escalations | questions/game |
|---|---:|---:|---:|---:|---:|
| soft | 0.039 | 0.996 | 60 | 20 | 9.77 |
| hard | 0.175 | 0.913 | 0 | 325 | 8.77 |
| retract @ margin 4 | 0.027 | 0.563 | 333 | 25 | 10.04 |
| tombstone @ tol 3 | 0.009 | 0.499 | 764 | 6 | 11.14 |
| **modeswitch @ K=3** | **0.149** | **0.926** | **0** | **277** | 9.44 |

What this buys:

- In-pool failure falls from 0.175 to 0.149 with **zero wrong guesses** and AUC at or
  above `hard`'s (0.926 vs 0.913).
- It makes fewer expert calls than `hard` (277 vs 325), because a recovered flip becomes
  a won game instead of an escalation.
- With no noise it behaves exactly like `hard`, so the gain is purely the reversal's.

**Caveats.** The recovery is modest (0.026), because most flips leave no cheap
single-contradiction signature. Questions/game for the discrete backends depends on the
tie-break order by about ±0.3, so treat that column as approximate. `soft` leads on
in-pool failure and AUC, but it makes 60 wrong guesses and has no audit trail (§2).

Source: `llm_rl_scripts/twenty_questions/jev/OPENWORLD_RETRACTION_RESULTS.md`, run
`results/openworld_retraction_20261108_213525`.

## 2. Auditability comes free with the discrete design (ENV-703, #50)

An external auditor rides the loop through an `observer` hook (no backend changed; with
the hook off the earlier runs are byte-for-byte unchanged). It rebuilds an op-log from
each backend's survivor-set changes and scores it. Ten percent flips, 2 noise seeds:

| backend | ops/game | replay exact | verifiable | answers "why" | reinstate reason is local |
|---|---:|:--:|:--:|:--:|:--:|
| soft | 0.0 (dense trace ≈1,104 cells/game) | — | — | — | — |
| hard | 107.2 | 1.000 | 1.000 | 1.000 | — |
| retract @4 | 93.9 | 1.000 | 1.000 | 1.000 | 0.000 |
| tombstone @3 | 28.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| **modeswitch @3** | **108.5** | **1.000** | **1.000** | **1.000** | **1.000** |

- **Replay:** the sparse log rebuilds the final survivor set exactly, with no
  likelihood model.
- **Verifiable:** every recorded elimination re-checks against the committed matrix,
  and a SHA-256 hash chain catches a single edited record.
- **Answers "why":** every elimination names the question and answer that struck it.
- `modeswitch` also gives a local reason for every reinstatement. `retract` can't: its
  revivals come from a global likelihood race.

So the backend chosen for safety in §1 is also the one that scores 1.000 on every audit
property. `soft`, the apparent winner on §1's accuracy columns, scores nothing here.

Source: `llm_rl_scripts/twenty_questions/jev/OPENWORLD_AUDIT_RESULTS.md`, run
`results/openworld_audit_20261108_233741`.

## 3. A ranked build order exists (#56)

`PRODUCTION_RECOMMENDATIONS.md` turns the 20Q line (ENV-701 to ENV-704-B) into one
ranked guardrail document:

1. **P0a.** Gate the human on hard elimination, and recover a wrong elimination with a
   bounded exact re-test (`modeswitch`).
2. **P0b.** Keep an append-only, hash-chained op-log. It costs almost nothing once P0a
   is in place.
3. **P1.** Spend the expensive answer source early and sparingly (§5).
4. **P2.** Add the Pitman-Yor catch-all novelty detector when the catalog is
   incomplete. Its AUC is 1.000 with no noise and 0.997 at 10% flips (3 noise seeds,
   range 0.996 to 0.999). It works because P0a protects it.
5. **Don't build** up-front attribute onboarding (ENV-704-A, #51). The best effort level
   is zero, and partly filled rows cause 7–10% wrong guesses.

The P0 items don't depend on budget, pool size or noise. The cost items do: check the
real ratio of question budget to pool size before importing them.

Sources: `PRODUCTION_RECOMMENDATIONS.md`, `OPENWORLD_RESULTS.md` (runs
`results/openworld_20261108_161009` for the clean AUC and
`results/openworld_noise_grid_20261108_162415` for the noise grid).

## 4. A guaranteed rule for "we've seen enough of the world" (#73)

**The question.** In the open-world loop, θ is the true probability that the next
secret is outside the known pool. Can the loop declare θ ≤ ε from its own estimates?

**The comparison.** Fifty Zipf seeds (s = 1.1, 2,000 games each), with θ recorded every
game:

- **Point estimates:** stop the first game the live Pitman-Yor (PY) or Good–Turing (GT)
  estimate falls to ε.
- **Betting certificate:** test-by-betting on the expert-corrected novelty indicator,
  α = 0.05. Because θ can only fall, Ville's inequality bounds the chance of a false stop
  by α.

A false stop is a stop at a game where the true θ is still above ε. Clean `soft` cell,
false stops out of 50 seeds (seeds that stopped in brackets):

| ε | point PY | point GT | certificate, PY stake | certificate, GT stake |
|---|---:|---:|---:|---:|
| 0.20 | 49/50 | 32/50 | **0/50** (50) | **0/50** (50) |
| 0.10 | 48/50 | 30/50 | **0/50** (50) | **0/50** (50) |
| 0.05 | 26/50 | 34/50 | **0/50** (50) | **0/50** (50) |
| 0.03 | 1/50 | 34/50 | 0/50 (0) | 0/50 (25) |

- The point estimates stop too early far too often, up to 49 of 50 seeds.
- The certificate never stops too early in any cell, with or without noise (`hard` and
  `modeswitch` at 5% and 10% flips give the same counts up to one or two seeds at
  ε ≤ 0.03).
- The cost is time. The median certificate stop comes about 230 to 645 games after the
  true crossing (ε = 0.2 to 0.05), and most seeds can't certify ε ≤ 0.03 within 2,000
  games.

**Caveats.** The certificate is a reconstruction of the standard betting method and
hasn't been checked against the C3 spec. The PY stake uses (σ, θ) fitted with
hindsight. The GT and rate stakes are fully predictable and also show zero false stops.

Source: `llm_rl_scripts/twenty_questions/jev/mm_certificate.py`, run
`results/mm_certificate_20261112` (regenerated for this note).

## 5. One or two truthful answers are enough (ENV-704-B, #54–#55)

A cheap source flips each attribute answer with probability q; one "human query" buys
the truth. 1,500 games per cell:

| regime | cheap only | truth every turn | one early human query (`first@1`) | share of gap closed |
|---|---:|---:|---:|---:|
| N=40, T=10, q=0.15 | 0.711 | 0.818 | 0.789 | 73% |
| N=40, T=10, q=0.25 | 0.629 | 0.818 | 0.761 | 70% |

- `first@2`, two human queries on the first in-pool attribute questions, reaches the
  ceiling of 0.818 exactly.
- When the budget is loose (N=20, T=20) the cheap source already reaches 1.000, so the
  human buys nothing.
- *How many* human queries you buy matters much more than *which*. Front-loading beats
  random spending by +0.008 at N=40, widening to +0.029 at N=120.

Sources: `TWOORACLE_RESULTS.md`, runs `results/twooracle_20261109_011300` and
`results/twooracle_20261109_014246`.

## 6. A current model holds a planted persona (CD-PERSONA-0/1, #37–#38)

The Car Dealer task plants a buyer persona and assumes the counterparty acts in line
with it. If that holds, an LLM counterparty can serve as a measured oracle with a known
noise level. The constraint tested: a buyer who only buys at a discount should never buy
when no discount is on offer.

Audit-corrected violation rate (`eps_buyer`), 2,250 conversations per current-model arm:

| | 2023 narrator (GPT-3.5) | Sonnet 5.5, 2023 narrator prompt | Sonnet 5.5 as buyer agent |
|---|---|---|---|
| `eps_buyer` | 0.198 [0.174, 0.223] | 0.000 | 0.000 |

- The leak came from the model, not the protocol: the model effect is +0.198, the
  protocol effect +0.000.
- The preregistration predicted leaks in both arms (P1–P3). Those were refuted in the
  favourable direction.

**Caveats.** The 0.000 is the registered estimator's output when every audited sale
turns out to be a no-sale. It is not a certainty claim. The scorer's raw rates were
4/497 (buyer agent) and 3/482 (narrator), and the blind audit found all seven were
no-sales, with no misses among 30 audited no-sales per arm. Two other preregistered
claims failed: 11.8% of buyer-agent replies hit the decision-token timeout (P6), and
acceptance of the 15% offer was 0.384, below the predicted 0.5 (P7).

Sources: `llm_rl_scripts/car_dealer_persona/RESULTS_BASELINE2023.md` and
`RESULTS_CURRENT.md`, runs `results/persona_baseline2023_20261106` and
`results/persona_current_20261107`.

## 7. The exact planner beats LLM choosers on the same menus (CD-NOUL-1, #45)

In free-text live Car Dealer, the exact planner using Noul's answer reads was compared
with two LLM agents choosing from the same menus (so the contrast isolates the chooser).
Gain in Y for the planner:

| opponent style | vs Jev agent | vs Sonnet agent |
|---|---:|---:|
| plain | +0.159 [+0.130, +0.192] | +0.160 [+0.130, +0.191] |
| oblique | +0.114 [+0.077, +0.154] | +0.120 [+0.074, +0.166] |

Two more confirmed results: on oblique answers Noul's reads beat regex parsing
(+0.045 Y, P2) and cut answer log-loss by 1.47 bits (P8).

**Caveat.** The primary claim (P1) came out positive (+0.051) but captured 0.4996 of the
available gain, just below the registered 0.5 minimum. Noul's soft reads hurt on plain
answers (P3 not supported).

Source: `llm_rl_scripts/car_dealer_noul/RESULTS_NOUL.md`.

## 8. The deep-RL ladder closed cleanly (CD-DRL R3–R4, #34–#35)

- **R3, misspecified opponents.** A policy trained against a mixture of true opponents
  ties the exact planner on training and held-out opponents (held-out advantage +0.0006,
  +0.22 SE). The planner was already robust. The remaining gap needs the opponent's
  identity, which no model-blind actor can see.
- **R4, weaker oracles.** Every *value* potential recovers V* fully, including a
  wrong-kernel one and one that ignores asking. The oracle-free entropy potential fails
  at every scale tried (recovery −0.09 to −2.22). At H=6 the order holds (exact +0.61,
  entropy +0.17).
- **Amortization.** The exact oracle costs one ~35-minute solve and is O(1) after that.
  The approximate oracle would cost 5+ hours for nothing. Use the exact one.

Sources: `llm_rl_scripts/car_dealer_drl/README.md`, `results/drl_r3_20261105/R3_RESULTS.md`.

## 9. Reproducibility got stronger (#71–#72)

- The ST-0 stress harness passes its regression gate, 52 of 52 checks.
- ST-0 found that the discrete backends' results depended on Python's string hash. Fixed
  in #72: survivors are now read in admission order, and the retraction and audit grids
  are identical under `PYTHONHASHSEED=0` and `=1`. No conclusion changed (see the
  *Redraw* section of `OPENWORLD_RETRACTION_RESULTS.md`).

## Not covered here

The negotiation design work was mostly negative this week, and those results matter
because they rule designs out:

- Split Deal v0.2 and v0.3 haven't produced a negotiator's dilemma yet (#62–#68).
- In one-sided Car Dealer, words are worth at most 0.043 Y (#60).
- The Sonnet opponent's targeting is no better than splitting the difference (#61).

**Still in progress:** the ST-1 crowding grid (running) and the Split Deal CFR engine
(built on `claude/split-deal-cfr-build`, not yet run).
