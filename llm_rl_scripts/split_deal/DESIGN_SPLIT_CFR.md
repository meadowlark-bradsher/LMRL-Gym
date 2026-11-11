# Split Deal v0.3 — the CFR check on 2-2-3 (2026-11-11; decisions in §8 settled, §5 frozen; nothing computed but tree counts)

Jira NEG-39, under the redesign epic NEG-25. Follows gate SD3-K (`SWEEP_SPLIT_K.md`, PR #68). The game is v0.3 as
registered in `DESIGN_SPLIT_V03.md`. This note designs the equilibrium check that `DESIGN_SPLIT.md` §5 registered
before any oracle existed: "on the smallest instance, counterfactual regret minimisation (or the sequence-form LP)
as a check, reported as the exploitability of A1 and B2". It states what the check must show and its predictions
before any equilibrium, best response or ladder value is computed on the game it solves. The only numbers computed
for it are the tree counts of §3.

## 1. Why

The ladder is built so that every rung is an exact best response and no fixed point is needed (`DESIGN_SPLIT.md`
§5). It stops at A3 (`DESIGN_SPLIT_B2.md`, decision 2). Every SD3-K result is therefore a level-k result, and three of
them cannot be read without an equilibrium to compare against. All three are published in `SWEEP_SPLIT_K.md`, pooled
over the nine instances at Q = 3:

- **K3 failed.** A lo-level A3 declines B2's HIGH in 45.9 % of 582 on-path decisions, and its registered consequence
  says disclosure "already unravels at level 3". At level 3 the unravelling is partial: lo still declines almost half
  the time, and hiding is still worth +0.0148 Y to it (K4). Whether it completes is an equilibrium question.
- **Reading backfired once A adapted (off-gate).** B's value at A's best response is 0.3524 Y against B2 and
  0.4077 Y against B0. A3 does this through its asks: B2 reads them through A1's policy, and A3's true type leaves
  B2's belief in thousands of episodes. That is either a cost of reading A or a cost of reading A through a model A
  has outgrown. Only an opponent that models A3 correctly, or an equilibrium, separates the two.
- **K5 held.** The all-or-nothing dilemma is absent against every counterpart tried. Whether it is absent at
  equilibrium is still open.

## 2. What the check must show

**About B2.**

- How exploitable it is: A's best-response value against B2, and B's value at that best response, both set beside
  B's value at equilibrium.
- Whether SD3-K's "reading backfires" is a property of reading or of the level-k ladder.

**About A3.**

- How far (A3, B2) is from equilibrium. This is NashConv, the sum over players of what a best response would gain.
  A3 is already A's best response to B2, so NashConv is essentially B's gain from best-responding to A3.
- Whether A3's hiding (lo declines HIGH, hi answers) is equilibrium behaviour or an exploitation of B2's stale model.

**As registered in v0.2.** The exploitability of A1 and B2, reported for every ladder profile.

## 3. The game the check solves

The table counts the public game tree of 2-2-3, the smallest canonical instance (`spec_numbers_cfr.py`). The tree is
the one the ladder already lives in:

- A asks only when B will answer, and only a question that splits A's answer-consistent belief about B.
- B asks only a question it has not had declined, and only one that splits its answer-consistent belief about A.
- Either may pass.
- The responder accepts iff its utility is at least its own d, which is optimal at the last move and so fixed, not
  a choice.

<!-- spec-numbers-cfr:begin -->
_Rendered by `spec_numbers_cfr.py render` from `results/split_cfr_spec_20261111/spec_numbers.json` (`split-deal-cfr-spec-v0.1`). Counts by enumeration of the public tree; no strategy is played and no value is computed._

Instance 2-2-3: 18 types per side, 36 allocations, so 37 offers per seat with walking. Merged bank (10 of 13 questions): `ZERO(book)`, `ZERO(hat)`, `ZERO(ball)`, `TOP(book)`, `TOP(hat)`, `PREF(book,ball)`, `PREF(hat,ball)`, `HALF(book)`, `HALF(hat)`, `HIGH()`.

| bank | Q | structure | talk paths | A decision points | B decision points | offer points | leaf pairs | leaf evaluations per CFR pass |
|---|---|---|---|---|---|---|---|---|
| full | 1 | T2 | 729 | 18 | 252 | 13,608 | 63,504 | 4,699,296 |
| full | 1 | T3 | 1,080 | 6,336 | 252 | 23,202 | 122,472 | 9,062,928 |
| full | 2 | T2 | 259,081 | 6,822 | 72,774 | 2,734,348 | 7,214,596 | 533,880,104 |
| full | 2 | T3 | 394,737 | 1,334,180 | 76,050 | 4,899,742 | 15,226,520 | 1,126,762,480 |
| full | 2 | game:all/all | 259,081 | 6,822 | 72,774 | 2,734,348 | 7,214,596 | 533,880,104 |
| full | 2 | game:all/none | 509 | 270 | 504 | 11,848 | 48,348 | 3,577,752 |
| full | 2 | game:none/all | 93,147 | 6,822 | 37,856 | 2,168,184 | 8,847,684 | 654,728,616 |
| full | 2 | game:none/none | 183 | 270 | 270 | 6,588 | 59,292 | 4,387,608 |
| full | 3 | T2 | 31,125,241 | 1,373,996 | 9,965,698 | 216,866,888 | 377,758,096 | 27,954,099,104 |
| full | 3 | T3 | 52,763,472 | 122,330,388 | 11,496,920 | 452,090,820 | 987,108,764 | 73,046,048,536 |
| merged | 1 | T2 | 441 | 18 | 198 | 8,316 | 39,204 | 2,901,096 |
| merged | 1 | T3 | 651 | 3,798 | 198 | 14,076 | 74,844 | 5,538,456 |
| merged | 2 | T2 | 100,489 | 4,176 | 36,276 | 1,089,212 | 2,951,524 | 218,412,776 |
| merged | 2 | T3 | 153,285 | 527,596 | 38,256 | 1,952,576 | 6,240,044 | 461,763,256 |
| merged | 2 | game:all/all | 100,489 | 4,176 | 36,276 | 1,089,212 | 2,951,524 | 218,412,776 |
| merged | 2 | game:all/none | 317 | 216 | 396 | 7,424 | 30,924 | 2,288,376 |
| merged | 2 | game:none/all | 35,187 | 4,176 | 19,096 | 824,064 | 3,432,564 | 254,009,736 |
| merged | 2 | game:none/none | 111 | 216 | 216 | 3,996 | 35,964 | 2,661,336 |
| merged | 3 | T2 | 9,296,401 | 548,782 | 3,492,210 | 66,480,396 | 118,853,604 | 8,795,166,696 |
| merged | 3 | T3 | 15,609,835 | 36,777,454 | 4,077,494 | 136,563,796 | 304,678,540 | 22,546,211,960 |

T2 and T3 at every Q, and the four 2 × 2 cells at Q = 2; T1 and the other cells are in the JSON.
<!-- spec-numbers-cfr:end -->

What the counts say:

- **Q = 3 is out of reach for tabular CFR.** One pass over T3 evaluates about 7.3 × 10¹⁰ leaves with the full bank,
  and 2.3 × 10¹⁰ with the merged one, even on the smallest instance. A pass is one iteration, and CFR needs
  thousands.
- **Q = 1 breaks T3.** With one round, B always answers A's only question before B asks its own. A decline then
  never costs A an answer, and T3's reciprocity, which is what made declining costly, is gone.
- **Q = 2 is the smallest talk length that keeps T3's mechanism.** B's round-1 question comes before A's round-2
  ask, so a round-1 decline costs A that ask's answer.
- **The merged bank cuts a pass by a factor of about 2.4 at Q = 2**, from 1.1 × 10⁹ to 4.6 × 10⁸ leaf evaluations in
  T3. On 2-2-3, 13 questions induce only 10 distinct partitions of the 18 types. `ZERO(ball)`, `TOP(ball)` and
  `HALF(ball)` are one partition, and so are `TOP(book)` and `PREF(book,hat)`.

**Game G (decision 1):** v0.3 on 2-2-3, protocol 4.2, Q = 2, π = 1/2, independent uniform prior, the merged
bank (lowest bank index per partition, HIGH kept). Each side's payoff is its own-scale Y, so the game is general-sum.

**Why G's restrictions do not lose equilibria.** Moves G leaves out never change payoffs:

- asking a question that splits nothing;
- A asking when B will decline;
- asking a second question from a partition already in the bank.

Take any equilibrium of G and let each side read an omitted move as it reads the move it is payoff-equivalent to (a
pass, or the merged representative). The profile is then an equilibrium of the larger game too. So an equilibrium
found in G is one of the unrestricted game.

**One ladder channel G removes.** B2 ignores an ask that no type in K would make under A1. A1 always asks the lowest
index of equivalent questions, so in the full bank A3 can ask a higher-index duplicate and not be read. G has no
duplicates. A3's value on 2-2-3 at Q = 2 under the full and the merged bank is reported (§6), so the size of that
channel is known.

## 4. Method

**What is computed in G.**

- **Equilibria:** one equilibrium σ*_s for each structure s in decision 2: T2, T3 and the four 2 × 2 cells.
- **The ladder:** (A1, B0), (A1, B2) and (A3, B2), exactly as in SD3-K but on G.
- **Best responses:** each player's best response to every ladder strategy and to σ*.

**Solver (decision 3).** CFR+: regret matching+, alternating updates, linearly weighted averaging, written natively
with numpy. At each public node the information sets are the mover's own types consistent with the node, so regrets
and strategies are arrays over own types. Chance (the type pair, then the proposer) enters through per-type reach
vectors. The final stage needs no tree: the proposer's counterfactual value for each offer is a product of a fixed
(own type × offer × opponent type) payoff table with the opponent's reach vector.

**The certificate is not the solver's.** CFR's convergence guarantee is for two-player zero-sum games. Here the game
is general-sum, and the average strategy is only guaranteed to approach a coarse correlated equilibrium. So:

- every profile CFR returns is checked by exact best responses for both players;
- ε_A and ε_B are each player's gain from best-responding, and NashConv = ε_A + ε_B;
- a profile is a certified equilibrium iff NashConv ≤ ε̄ (decision 4).

The sequence-form LP of the v0.2 text applies only to zero-sum games. Its general-sum analogue, the sequence-form
LCP, is exact but practical only for the smallest cells, so it is not planned.

**Best response.** A generic exact best response to any behaviour strategy in G: a belief program over the
opponent's reach-weighted type distribution, in floating point. It must reproduce the ladder before it certifies
anything:

- the best response to B0 must have A1's value;
- the best response to A1 must have B2's value;
- the best response to B2 must have A3's value.

All three are in G, to within 1e-9. They are checks, not predictions.

**If a profile is not certified (decision 4).** In order: run four times the iterations; then switch to discounted
CFR; then record the structure as **not certified**. Predictions that use it are then not testable. Nothing is read
off an uncertified profile.

**Equilibrium selection (decision 5).** The game can have several equilibria. The primary run starts from uniform
strategies; a second run is warm-started at the ladder profile (A3, B2). A prediction is read on every certified
equilibrium found. If its verdict differs between them, it is recorded as equilibrium-dependent.

## 5. Predictions and bars, stated before anything is computed on G

Frozen 2026-11-11 with the decisions of §8. Gate name: **SD3-E** (equilibrium). Shared reading:

- G as in §3;
- means over the 324 type pairs, uniform;
- own-scale Y; "lo-level" and "hi-level" refer to A's level;
- σ*_s is a certified equilibrium of structure s; the ladder profiles are computed on G.

- **E1 (the ladder has not converged).** NashConv(A3, B2) in T3 exceeds 0.01 Y. Reasoning: A3 moves B2's belief off
  A's true type, and a B that models A3 correctly recovers what that costs.
- **E2 (B2's loss to A3 is a level-k artifact).** In T3, Y_B(σ*_T3) > Y_B(A3, B2): at equilibrium B does better than
  B2 does once A adapts to it.
- **E3 (the level unravels at equilibrium).** In σ*_T3, at on-path decisions where B asks HIGH:
  - a hi-level A answers with probability at least 0.95;
  - B's posterior that A is lo-level after a decline is at least 0.9.

  Reasoning: a hi-level A gains from being known, since B as proposer must clear 5.5. Declining would also cost it
  B's next answer. Once hi answers, a decline marks lo.

  *Support rule:* if the prior reach probability of B asking HIGH is below 0.05 for either A level, E3 is recorded as
  not testable.

  *Not predicted:* the lo-level decline rate. Where no ask of A's remains, lo is indifferent between answering and
  declining, and equilibrium does not pin that rate down. It is reported by round.
- **E4 (hiding is worth even less at equilibrium).** A lo-level A's gain from T3 over T2,
  Y_A^lo(σ*_T3) − Y_A^lo(σ*_T2), is smaller than the same difference in (A3, B2) on G. This continues K4's chain:
  hiding is worth most against B0, less against B2, least at equilibrium.
- **E5 (no all-or-nothing dilemma at equilibrium).** The 2 × 2 game read on the four cells' equilibrium values is not
  a strict prisoner's dilemma for both players. It is read as in D3′ and K5.

*Registered consequences:*

- **E1 fails.** On 2-2-3 at Q = 2 the ladder is already within 0.01 Y of equilibrium at level 3. SD3-K's values can
  stand as equilibrium approximations there.
- **E1 and E2 hold.** "Reading backfires" is cited only as reading through a stale model. A live counterpart that
  reads silence needs a correct or adaptive model of the focal side, and NEG-33 inherits that constraint.
- **E2 fails.** B does worse at equilibrium than against A3. SD3-K's off-gate finding is then not a level-k
  artifact, and no write-up presumes that a reading counterpart helps B.
- **E3 holds.** K3's consequence (no hiding treatment for the level) is confirmed at equilibrium on 2-2-3.
- **E3 fails, and is testable.** The level does not unravel at equilibrium, so K3's consequence is specific to level
  3. The hiding-treatment decision goes back to Meadowlark before NEG-33 uses it.
- **E4 fails.** Hiding is worth more at equilibrium than at level 3: the ladder understates the strategic content of
  selective disclosure. The write-up says so wherever K4 is cited.
- **E5 fails.** A strict dilemma appears at equilibrium. K5's consequence ("absent against every counterpart tried")
  is qualified to the ladder's counterparts, and NEG-33 inherits the equilibrium framing.
- **Not certified, or equilibrium-dependent.** The prediction is not settled. Nothing is inferred from it, and the
  write-up reports the profiles found.

## 6. Reported, not predicted

- **Exploitability (decision 7):**
  - NashConv of (A1, B0), (A1, B2), (A3, B2) and σ* in every structure;
  - B's value at A's best response, for B0, B2 and σ*_B;
  - A's value at B's best response, for A1, A3 and σ*_A.
- **Bridges:**
  - the ladder on G against the stored SD3-K ladder on 2-2-3 at Q = 3 with the full bank (what the reduction
    changes);
  - A3's value on 2-2-3 at Q = 2 under the full and the merged bank (the duplicate channel of §3).
- **Behaviour:** E, agreement rates and Y by level for every profile; lo- and hi-level answer rates at HIGH by round;
  B's posterior after a decline.
- **Solver:** iterations, seconds, NashConv per structure, and the second run's values beside the first's.

## 7. What this check cannot say

- **It is one instance at Q = 2.** Its verdicts speak for 2-2-3 under G. Carrying them to the nine instances at
  Q = 3 needs an argument, not a citation.
- **An equilibrium is a reference point, not a forecast of how an LLM negotiates.** No LLM is involved.
- **It is computed in floating point.** The ladder is exact Fractions; G's ladder values and best responses are
  compared with them to within 1e-9.

## 8. Decisions (settled by Meadowlark, 2026-11-11)

| # | Question | Settled | Not taken |
|---|---|---|---|
| 1 | The game CFR solves | G: 2-2-3, Q = 2, merged bank, the ladder's restrictions (§3) | Q = 2 with the full bank (2.4 × the cost; keeps the duplicate channel); Q = 1 (cheap, but no T3 reciprocity); Q = 3 (out of reach) |
| 2 | Structures | T2, T3 and the four 2 × 2 cells (all that E1–E5 read) | T3 only (loses E4 and E5); add T1 (nothing predicted on it) |
| 3 | Solver | native CFR+ with numpy, certified by exact best responses | OpenSpiel (Apache-2.0; needs an install and a custom game wrapper); sequence-form LCP (exact, but only for the smallest cells) |
| 4 | Certificate and fallback | NashConv ≤ 1e-3 Y; iteration budget from the pilot; fallback 4 × iterations, then discounted CFR, then "not certified" | ε̄ = 1e-4 (costlier); ε̄ = 1e-2 (as large as E1's bar) |
| 5 | Equilibrium selection | uniform start plus a warm start at (A3, B2); a verdict must agree across certified equilibria, else "equilibrium-dependent" | uniform start only; several random starts |
| 6 | Predictions | E1–E5 with bars 0.01 Y, 0.95, 0.9 and support 0.05, plus the consequences of §5 | drop E2; value-only (drop E3's behaviour reading) |
| 7 | Exploitability report | NashConv of every ladder profile and σ*, each strategy's value at the opponent's best response, with the three ladder reproductions as checks | NashConv only |
| 8 | Runtime pilot | seconds-only on G's T3, values discarded unread; fixes the iteration budget and run plan; no structure dropped for cost | no pilot |

## 9. Work order

1. ~~Settle §8 and freeze §5.~~ Done 2026-11-11.
2. Build G on `PrivateTables` with a chosen question list. Test that the ladder on G with the full bank reproduces
   the existing ladder on 2-2-3 at Q = 2, and that the merged bank leaves (A1, B0) and (A1, B2) unchanged, as §3
   argues.
3. Build the generic best response and CFR+. Test them on:
   - Kuhn poker (game value −1/18 for the first player), a known answer;
   - the three ladder reproductions of §4;
   - a smallest Split Deal case (Q = 1 on the 10-type test instance), certified.
4. Run the seconds-only pilot (decision 8) and fix the run plan.
5. Commit the code with the operational reading in its docstring; then run SD3-E; then commit the results.
