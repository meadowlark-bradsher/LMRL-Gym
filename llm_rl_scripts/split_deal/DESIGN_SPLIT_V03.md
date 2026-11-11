# Split Deal v0.3 — private outside options (2026-11-10; decisions in §6 settled, §5 frozen; no decision value computed)

Jira NEG-39, under the redesign epic NEG-25. Follows gate SD-D of v0.2 (`SWEEP_SPLIT_L2.md`, PR #64). v0.2's
spec, `DESIGN_SPLIT.md`, stays as registered; this note changes one thing about the game and states, before any
value is computed, what the sweeps on the changed game are expected to show.

## 1. Why

Gate SD-D found no all-or-nothing negotiator's dilemma in v0.2. Being known *raised* the responder's value: the
information rent came out negative in both regimes, and neither side's withholding was dominant. The reason is in
the game. The outside option was public, so the only private information was item values, and that information is
integrative. An uninformed proposer often makes an offer the responder rejects, which pays both nothing. An
informed one finds a bundle that the responder accepts and that is cheap for the proposer. (In easy, agreement when
B proposes rose from 68 % to 89 % once B knew A's values; `SWEEP_SPLIT_L2.md`.)

The registered consequence of that failure (`DESIGN_SPLIT.md` §7) names the mechanism to add: a private outside
option. Knowing a responder's reservation is distributive information. A proposer who knows it can offer exactly
that much. One who does not must either leave room for responders with a strong outside option, or accept that
they will reject. Whatever the responder gains from the proposer's uncertainty is an information rent, and
disclosing the outside option gives it up. v0.2 had no information of this kind.

One qualification from v0.2 carries over. Even there, A1 declined *specific* questions when it could choose (D4
held), so some value information was locally distributive. v0.3 adds a channel that is distributive by
construction, and asks whether that is enough for the all-or-nothing dilemma.

## 2. The change

Everything in `DESIGN_SPLIT.md` stands (instances, values, allocations, one binding offer, protocols 4.1 and 4.2,
structures T0–T3, B0 and A1) except:

- **Type.** A side's type is (values, level). The level ∈ {lo, hi} is drawn uniformly, independently of the values
  and of the other side's type. The outside options are d(lo) = 4 and d(hi) = 5.5, the d of v0.2's easy and hard
  regimes, so v0.2's two regimes become the two corners of v0.3 where both sides happen to share a level.
- **Acceptance and units.** A side accepts an allocation iff its utility is at least *its own* d. Its Y is net gain
  over its own d, as a share of 10 − d. In the both-lo and both-hi corners this is v0.2's Y exactly. Because a
  type's scale is a positive constant, it changes no policy, only how pooled means weight types; pooled means on a
  common scale, (U − own d)/10, are reported beside the primary ones (decision 2).
- **Bank.** Thirteen questions: v0.2's twelve, plus HIGH ("is your outside option the high one?"). The same sweeps
  without HIGH run as an ungated control (decision 3). With one binding offer the level is otherwise never
  disclosed, so the control separates the reservation channel from what a private d does to value answers.
- **Regimes.** There are no longer two regimes; the levels replace them.

One consequence for B0: HIGH splits every prior exactly in half, so it ties for the largest expected information
gain at the start. B0 breaks ties by bank order, and HIGH is last, so a value question that also splits the prior
exactly in half is asked before it. B0's opening question is HIGH in some instances and not others. On B0's T2 path
(A answers everything), though, it asks HIGH within Q = 3 for every A type in every instance: round 1, 2 or 3 by
instance (last two columns of §3). The tie-break is kept as built, so B0 stays v0.2's rule with no special case,
and D4′ is reported split by that round (decision 8).

Built: `private_d.py` gives the v0.3 tables to the unchanged L1 and L2 programs. `test_private_d.py` checks three
things. With one level and no HIGH the tables reproduce v0.2 exactly. The L1 and L2 programs agree with naive
literal simulations that use per-type outside options. Rollouts realize the computed values.

## 3. Structural numbers

<!-- spec-numbers-v03:begin -->
_Rendered by `spec_numbers_v03.py render` from `results/split_v03_spec_20261110/spec_numbers.json` (`split-deal-spec-v0.3`). Counts by enumeration; no decision value is computed._

| counts | types | answer patterns (13 questions) | largest unresolved class | pairs with a deal, both lo | one lo, one hi | both hi | B0's first question | round B0 asks HIGH (T2 path) |
|---|---|---|---|---|---|---|---|---|
| 1-1-3 | 52 | 40 | 4 | 0.93 | 0.85 | 0.72 | TOP(book) | 2 |
| 1-2-2 | 42 | 34 | 2 | 0.98 | 0.95 | 0.78 | HIGH() | 1 |
| 1-1-4 | 42 | 30 | 4 | 0.93 | 0.85 | 0.66 | HIGH() | 1 |
| 1-2-3 | 28 | 26 | 2 | 0.95 | 0.91 | 0.79 | PREF(hat,ball) | 2 |
| 2-2-2 | 42 | 30 | 2 | 1.00 | 0.98 | 0.80 | HIGH() | 1 |
| 1-1-5 | 36 | 24 | 4 | 0.90 | 0.82 | 0.63 | TOP(book) | 2 |
| 1-2-4 | 24 | 22 | 2 | 0.97 | 0.93 | 0.78 | ZERO(ball) | 3 |
| 1-3-3 | 20 | 20 | 1 | 0.91 | 0.91 | 0.84 | HIGH() | 1 |
| 2-2-3 | 18 | 14 | 2 | 1.00 | 0.95 | 0.74 | HIGH() | 1 |

Pooled over instances and level pairs, a mutually acceptable allocation exists for 0.87 of type pairs (lo/lo 0.95, hi/lo 0.89, hi/hi 0.73; pair-weighted).
<!-- spec-numbers-v03:end -->

The both-lo and both-hi columns equal v0.2's easy and hard columns, as they must. A mixed pair is in between.

## 4. Protocol details carried over unchanged

- Protocol 4.1 (one-sided, gate SD3-W): A asks up to Q questions, B answers truthfully, then A makes one binding
  offer or walks; B accepts iff U_B ≥ d(B's level).
- Protocol 4.2 (two-sided, gate SD3-D): Q rounds (A then B), proposer drawn after the talk (A with probability π),
  one binding offer. B0 draws no inference from a decline or from which question was asked.
- The operational reading of SD-D fixed in `twosided.py` (Q = 3, independent prior, pooled means, strict PD for
  both players, D4 counted on on-path decisions with ties going to decline) is reused.

## 5. Predictions and bars, stated before any decision value is computed

Frozen 2026-11-10 with the decisions of §6. Y below is the primary scale (own d); the common-scale means are
reported but decide nothing.

**Gate SD3-W (one-sided words value).** W = D_words(3) = V_words(3) − V_none, protocol 4.1, ask cost 0. The budget
is three questions from the 13-question bank, HIGH included. W is pooled over A's level and B's level (uniform)
and over the nine instances with equal weight. Same tiers as SD-W: go at W ≥ 0.10 Y, conditional from 0.08 Y with
a power calculation, no-go below. The re-offer scarcity check is repeated: H = 4, h = 0, words-minus-offers below
0.04 Y means the words value depends on the single-offer rule. W per level pair (lo/lo, hi/lo, lo/hi, hi/hi) is
reported as a diagnostic.
*No-go consequence:* there is no regime fallback, since levels replace regimes. The one-sided words study stays on
v0.2, which passed SD-W, and v0.3 is used only for SD3-D.
*Prediction:* go. v0.2 gave 0.166 Y and 0.106 Y in its two regimes (`SWEEP_SPLIT_L1.md`), and HIGH adds a question
that is directly useful to a proposer.

**Gate SD3-D (the dilemma), π = 1/2, Q = 3, read against B0:**

- **D1′.** Joint efficiency rises with disclosure: E(T2) ≥ E(T1) ≥ E(T0).
- **D2′.** Being known costs the responder: Y_A(T2) < Y_A(T1), so the information rent is positive.
- **D2′-lo.** The same holds for a lo-level A taken alone. A hi-level A may gain from being known, since B then
  stops making offers it would reject; its sign is reported, not predicted.
- **D3′.** The 2 × 2 game (each side answers everything, HIGH included, or nothing) is a strict prisoner's dilemma
  for both players, with B0 asking by expected information gain. The second reading (B asks by the exact program)
  is reported beside it.
- **D4′.** In T3, at on-path decisions where B0 asks HIGH, A1 answers more often when its level is hi than when it
  is lo, and declines in most such decisions when its level is lo. Reported pooled and split by the round in which
  B0 asks HIGH (§3).
- **D5′.** In T3, A1 answers value questions at a higher rate than it answers HIGH.

The no-HIGH control (§2) runs the same sweeps with no prediction.

*Registered consequences:*

- **D2′-lo fails.** If the rent is not positive even for a lo-level A, a private d does not make disclosure costly
  against B0, and the next rung is B2 before any change to the game.
- **D3′ fails again.** Making the reservation private is not enough against B0, which reads nothing into a
  decline, and the next rung is B2 before any other change to the game.
- **D2′ holds but D3′ fails.** Disclosure costs, but mutual disclosure does not beat mutual withholding, or
  withholding is not dominant. The 2 × 2 cell values say which.
- **D4′ holds.** Against B0, a low-level A hides by declining at no cost. That finding is specific to B0. A
  counterpart that reads declines (B2) would remove it, as in the unravelling argument. That B2 prediction is
  stated when B2 is designed, not here.

**Out of scope for the B0 sweeps:** B2, the CFR exploitability check (both carried over from NEG-31), and any live
call.

## 6. Decisions (settled by Meadowlark, 2026-11-10)

| # | Question | Settled | Not taken |
|---|---|---|---|
| 1 | Outside-option levels and prior | d ∈ {4, 5.5}, 1/2 each, independent of values and across sides | {2, 5}; three levels; correlated with values |
| 2 | Y scale | net gain over own d, as a share of 10 − own d; pooled means on a common scale of 10 reported beside it | common scale only |
| 3 | Is the level askable? | yes, the HIGH question; the same sweeps without HIGH run as an ungated control | no HIGH (the level would never be disclosed under one binding offer) |
| 4 | Gate SD3-W | Q = 3 from the 13-question bank, cost 0, pooled uniformly, SD-W tiers and scarcity check; W per level pair as a diagnostic; on a no-go the one-sided study stays on v0.2 | no named no-go rule; diagnostic only |
| 5 | Gate SD3-D reading | B0 primary, as in v0.2; B2 is a later rung reported separately; a D4′ hold is B0-specific | B2 primary; B0 and B2 co-primary |
| 6 | Predictions to register | D1′–D5′ with D2′-lo added, a consequence for D2′-lo failing, and the D3′ consequence reworded | as drafted; drop D5′ |
| 7 | B2 | designed after the B0 sweeps, its predictions registered before it runs | designed before; built and run before |
| 8 | B0's tie-break between equally informative questions | bank order, HIGH last (as built); HIGH's round on B0's T2 path in §3, and D4′ split by it | HIGH first |

## 7. Work order

1. ~~Settle §6.~~ Done 2026-11-10.
2. ~~Freeze §5 with the settled wording.~~ Done 2026-11-10.
3. ~~Run SD3-W (L1) and SD3-D (L2) against B0, with HIGH and without it (the control).~~ Done 2026-11-10; see §8.
4. Then design B2 and register its predictions; then B2 and the CFR check.
5. Gate SD-R (corpus reading, NEG-32) is unchanged and still needs an explicit go.

## 8. Outcome (2026-11-10)

Verdicts only; every number is in the rendered notes, which regenerate from their JSON.

- **SD3-W: go** (`SWEEP_SPLIT_V03_L1.md`, from `results/split_v03_l1_20261110/sweep.json`). The scarcity check
  fired again: with free re-offers the words add almost nothing, so the words value is conditional on the single
  binding offer, as in v0.2.
- **SD3-D** (`SWEEP_SPLIT_V03_L2.md`, from `results/split_v03_l2_20261110/sweep.json`): D1′ holds, D2′ fails,
  D2′-lo fails, D3′ fails, D4′ holds, D5′ holds. Being known still raises a lo-level A's value against B0. The
  2 × 2 game has no dilemma: mutual disclosure beats mutual withholding for both. But question by question, a
  lo-level A declines HIGH more often than not, and a hi-level A mostly answers it.
- **Registered consequences that apply:** those of D2′-lo failing, D3′ failing and D4′ holding. All three point to
  the same next rung: B2, a counterpart that reads declines, before any other change to the game. B2 is designed
  next, with its predictions registered before it runs (decision 7).
