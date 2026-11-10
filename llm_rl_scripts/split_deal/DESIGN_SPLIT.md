# Split Deal — a two-sided negotiation game with exact oracles (game spec v0.2, 2026-11-10 — design decisions of §10 settled; nothing registered, no oracle built)

Jira NEG-28, under the redesign epic NEG-25. This note fixes the game, its protocols, the three oracle
layers that score it, and the offline gates that come before any live call. It computes no decision value:
the words-value bar in §7 is stated here so that the L1 sweep (NEG-30) is read against a number chosen
before it ran. Study prefix SD; the gates of §7 are SD-W, SD-D and SD-R.

## 1. Why a new game

Two offline results on the 6-ranking Car Dealer game changed the plan for CD-NOUL Stage 2:

- **Words are worth little when offers are free tests.** In the one-sided reshape, a rejected offer
  eliminates opponent rankings exactly, so offers recover most of the information value and the
  opponent's words add at most 0.0427 Y (easy, H = 4; `ONESIDED_SWEEP.md`, from
  `results/onesided_sweep_20261109/sweep.json`), against a two-sided menu ceiling of 0.1019 Y. A
  likelihood seat cannot show much in a game where its input is nearly redundant.
- **The live opponent is reactive, not inferential.** Sonnet's proposals track the focal side's last
  offer about as well as a meet-in-the-middle rule (`TARGETING_NOUL2.md`). The "two-way" property that
  made Car Dealer attractive is not something the current game exercises.

The thesis under test is unchanged: an LLM is used as a likelihood over a counterpart's words, and an
exact planner chooses the moves. The game has to give that thesis room. Three requirements follow.

1. **The private model is not trivial.** Well over six types per instance, from a prior both sides know.
2. **Words are the main information channel.** Binding offers are scarce, so a rejection is not a free test.
3. **Both sides hold private values and both can act on what they learn**, so that disclosure helps find
   a better deal and also costs the discloser. That is the negotiator's dilemma (creating versus claiming
   value), and it is the two-way structure the 6-ranking game lacked.

## 2. The game

Two sides, A (the focal side) and B, split a public pool of items of three kinds (book, hat, ball).

- **Instance.** The count of each kind, every kind present, 5 to 7 items in all. Up to relabelling of
  kinds there are nine instances (table below).
- **Type.** A side's private per-unit value for each kind: non-negative integers with
  Σ count × value = 10. The whole pool is worth 10 to every type, so the type set of an instance is
  finite and common knowledge. Types are drawn independently and uniformly for the two sides. A
  *sensitivity prior* restricts the draw to pairs where no kind is worthless to both sides and some kind
  is valued by both (the value-pair constraints recalled from Deal or No Deal, §8). It is reported beside
  the primary prior and never gates anything: under it a side's own type is evidence about the other's,
  which is information no question paid for.
- **Allocation.** A gets a[k] units of kind k, B gets the rest. U_A = v_A · a, U_B = v_B · (counts − a).
- **Outside option.** Disagreement pays each side d = 10 r, public and equal for both, with the Car
  Dealer regimes kept for comparability: easy r = 2/5 (d = 4), hard r = 11/20 (d = 5.5, so a side needs
  utility 6). A side accepts an allocation iff its utility is at least d. This is the rational response
  to a final offer, not a behavioural assumption, and it replaces the T1 aspiration schedule.
- **Units.** Y = (U − d) / (10 − d): net gain over the outside option as a share of the most a side
  could gain. Disagreement is Y = 0 for both.

Definitions are in `game.py`; the numbers below are pinned by `test_split_spec.py`.

<!-- spec-numbers:begin -->
_Rendered by `spec_numbers.py render` from `results/split_spec_20261110/spec_numbers.json` (`split-deal-spec-v0.2`). Counts by enumeration; no decision value is computed._

| counts | types | allocations | answer patterns (12 questions) | largest unresolved class | type pairs | integrative pairs | mean best joint value | pairs with a deal, easy | pairs with a deal, hard | pairs meeting the value-pair constraints | conflict-free pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1-3 | 26 | 16 | 20 | 4 | 676 | 0.96 | 14.78 | 0.93 | 0.72 | 0.76 | 0.02 |
| 1-2-2 | 21 | 18 | 17 | 2 | 441 | 0.95 | 14.95 | 0.98 | 0.78 | 0.71 | 0.07 |
| 1-1-4 | 21 | 20 | 15 | 4 | 441 | 0.95 | 14.83 | 0.93 | 0.66 | 0.67 | 0.02 |
| 1-2-3 | 14 | 24 | 13 | 2 | 196 | 0.93 | 14.92 | 0.95 | 0.79 | 0.68 | 0.05 |
| 2-2-2 | 21 | 27 | 15 | 2 | 441 | 0.95 | 14.95 | 1.00 | 0.80 | 0.71 | 0.07 |
| 1-1-5 | 18 | 24 | 12 | 4 | 324 | 0.94 | 14.91 | 0.90 | 0.63 | 0.51 | 0.09 |
| 1-2-4 | 12 | 30 | 11 | 2 | 144 | 0.92 | 15.06 | 0.97 | 0.78 | 0.58 | 0.07 |
| 1-3-3 | 10 | 32 | 10 | 1 | 100 | 0.90 | 14.86 | 0.91 | 0.84 | 0.69 | 0.00 |
| 2-2-3 | 9 | 36 | 7 | 2 | 81 | 0.89 | 14.99 | 1.00 | 0.74 | 0.43 | 0.07 |

Across the 9 instances: 9–26 types and 16–36 allocations per instance; 2844 type pairs in all, of which 0.95 are integrative (best joint value above 10), mean best joint value 14.89; a mutually acceptable allocation exists for 0.95 of pairs in easy and 0.73 in hard; 0.68 of pairs meet the value-pair constraints of the sensitivity prior, and 0.05 are conflict-free (both sides can have their full 10). All shares pair-weighted.
<!-- spec-numbers:end -->

What the table says about the three requirements. The type space is larger than the old six in every
instance (the types column), though only modestly so in the two smallest, which are kept so that the
family spans a range of type counts (§7). Almost every pair of types has integrative potential, so there is value to create, and in the
hard regime a deal exists *only* through an integrative allocation (both sides need 6 of a pool worth 10
to each). A deal is not guaranteed to exist: where it does not, the right move is to walk, and
recognising that is part of the task.

## 3. The question bank

Twelve yes/no questions about the answerer's values, the analogue of the nine TOP / ABOVE / BOTTOM
questions of `car_dealer_binned/questions.py`:

| Kind | Count | Asks |
|---|---|---|
| ZERO(k) | 3 | Is a unit of k worth nothing to you? |
| TOP(k) | 3 | Is no kind worth more to you per unit than k? |
| PREF(j, k) | 3 | Is a unit of j worth strictly more to you than a unit of k? |
| HALF(k) | 3 | Are all the units of k together worth at least half the pool to you? |

The bank does not identify the type: in most instances some types share all twelve answers (the
"largest unresolved class" column). That is deliberate. With binary answers and tens of types, a few
questions cannot pin the type down, so the planner has to ask what matters for the allocation, not what
shrinks the posterior fastest. Numeric questions ("what is a hat worth to you?") would identify the type
in two answers and are excluded from v0.

## 4. Protocols

Offers are binding and there is exactly one. This is the property that keeps a rejection from being a
free test, and it is the Deal or No Deal rule in structured form (talk, then a single final selection;
no agreement, no deal).

### 4.1 One-proposer protocol (scored by L1)

1. **Talk.** A has Q talk moves. Each is ASK q, for a bank question not yet asked, or STOP. B answers
   truthfully. An ask costs c in Y (default 0).
2. **Final stage.** A makes one binding offer or walks. B accepts iff U_B ≥ d. On acceptance the
   allocation is implemented; otherwise both get d.

B never asks and never proposes, so nothing B learns about A matters. This is the hardened-kernel shape:
rules decide B's moves and its only words are answers.

### 4.2 Random-proposer protocol (scored by L2)

1. **Talk.** Alternating talk turns, A first, Q each. The mover ASKs or PASSes. The asked side ANSWERs
   truthfully or DECLINEs, as the information structure (§4.3) allows. Lies are out of scope for v0.
2. **Final stage.** The proposer is drawn after the talk: A with public probability π. Primary π = 1/2;
   π = 1/4 and 3/4 are reported as sensitivity, and π = 1 is protocol 4.1. The proposer makes one binding
   offer or walks; the responder accepts iff its utility is at least d.

The draw comes after the talk on purpose. If the proposer were announced first, the proposer would have
nothing to hide and the responder nothing to learn, reciprocal exchange would have nothing to trade, and
the game would collapse to 4.1 with a responder who may decline.

Each side now wants to know the other's values, to make a good offer if it proposes, and wants its own
values unknown, because an uninformed proposer must leave room in its offer and an informed one need not.
What the responder gains from the proposer's ignorance is an information rent; disclosure spends it.

### 4.3 Information structures and policies

The survey's six treatments are two axes, not one list. Four information structures:

| | Structure | Who asks | Who must answer | Protocol |
|---|---|---|---|---|
| T0 | No queries | nobody | — | 4.1 with Q = 0 |
| T1 | Private queries | A only | B always | 4.1; in 4.2, B does not ask |
| T2 | Public queries | both | both, always | 4.2 |
| T3 | Reciprocal exchange | both | each may decline; B answers iff A answered B's latest question (or B has not asked yet) | 4.2 |

Reading T2 as "both must answer" is this note's interpretation of the survey's "mutually observed"
row: in a two-party game the cost of learning in public is that the other side learns too. In T2 A has no
disclosure choice, so T1 versus T2 isolates the cost of being known. T3 is where A decides.

And the focal policies, each scored by the same oracle:

| | Policy | Question choice | Offer |
|---|---|---|---|
| T4 | EIG-seeking | greedy expected information gain about the counterpart's type | optimal for the belief |
| — | Value-greedy | the question with the largest one-step gain in offer value | optimal for the belief |
| T5 | Utility-aware | exact dynamic program over the remaining talk | optimal for the belief |

T5 is the oracle itself. The gap between T5 and the two greedy policies measures how far this is a
"global problem solved by local steps" in the sense of 20 Questions: small where greedy is near-optimal,
large where it is not. `elicit_negotiation` showed the existence of an EIG ≠ value gap on a toy; this
game should say how large it is on a game with a real allocation decision.

## 5. Oracles

**L0 — frontier (NEG-29).** Per pair of types, by enumeration and in exact arithmetic: the allocations
acceptable to both sides, the maximum joint surplus S* over them, the Pareto set, and the Nash point(s)
maximising (U_A − d)(U_B − d). For any outcome o:

- joint surplus S(o) = (U_A − d) + (U_B − d) on agreement, 0 otherwise;
- joint efficiency E = S(o) / S*, on pairs with S* > 0;
- surplus share C_A = (U_A − d) / S(o), on agreements with S(o) > 0.

L0 needs no model of either player, so it scores every arm, including an LLM against an LLM. On pairs
with no acceptable allocation the only correct outcome is no deal; they are reported separately.

**L1 — one-sided belief program (NEG-30).** Protocol 4.1. Answers are deterministic given the type, so
A's posterior is uniform over the types consistent with the answers so far and every value is an exact
Fraction, as in `sweep_onesided.py`. The state is (answers so far, talk moves left); an instance has at
most 3¹² answer states. Values, averaged over A's type and the nine instances with equal weight:

- V_none: Q = 0, the best single offer at the prior.
- V_words(Q): Q questions, exactly read.
- V_reveal: B's type known.
- D_words(Q) = V_words(Q) − V_none, the ceiling a likelihood seat can recover; D_rest = V_reveal − V_words(Q),
  what the bank cannot carry.

**L2 — two-sided program (NEG-31).** Protocol 4.2, built as a ladder so that every rung is an exact
best response and no fixed point is needed:

- **B0**, a rule counterpart. Asks by greedy information gain with a fixed tie-break; answers as the
  structure dictates; draws no inference from a DECLINE; as proposer makes the offer that maximises its
  expected net gain under its belief about A; as responder accepts iff U_B ≥ d.
- **A1**, the exact best response to B0: what to ask, whether to answer, what to offer.
- **B2**, the exact best response to A1, as a stretch rung. This is where a counterpart starts to read
  A's declines and question choices.
- On the smallest instance, counterfactual regret minimisation (or the sequence-form LP) as a check,
  reported as the exploitability of A1 and B2.

## 6. Where the LLM sits (live stage, not designed here)

The live arrangement this spec is meant to support, to be designed only after §7 passes:

- **B's moves are rules; an LLM only speaks for it.** A speaker model phrases B's truthful answers under
  a brief (plain, or guarded: truthful but oblique). B's accept, ask and offer decisions stay with B0.
- **A is the Noul seat.** The likelihood reader turns each reply into P(text | yes) and P(text | no); the
  belief update and the choice of ask, offer or walk are the planner's.
- **The ceiling is L1's V_words, exact.** This is what Stage 2 lost when the opponent's moves came from a
  model and the ceiling had to be taken under a fitted kernel.
- A later arm can hand B's decisions to a model as well. That arm is scored by L0 only, and no ceiling
  is claimed for it.

## 7. Offline gates before any live spend

**Gate SD-W (NEG-30), the words-value bar.** Primary configuration: protocol 4.1, easy regime, Q = 3, ask
cost 0, truthful answers, independent uniform types, the nine instances weighted equally. Three questions
and one offer is four focal moves, the H = 4 of the live Car Dealer protocol. With W = D_words(3):

> **W ≥ 0.10 Y: go.** A live study is worth designing.
>
> **0.08 ≤ W < 0.10 Y: conditional.** Offline work may continue; live spend needs a power calculation
> showing that the planned primary contrast has an expected 95 % half-width below W / 2, so that capturing
> half the ceiling is separable from zero.
>
> **W < 0.08 Y: no-go** in this configuration.

Why these numbers. Stage 1's primary contrast was +0.0509 [+0.0113, +0.0909] against a ceiling of
0.1019 Y (`RESULTS_NOUL.md`, P1): positive, but with a half-width of about 0.04 Y the captured share came
out at 50 % [11 %, 89 %] and could not show that half the ceiling was captured. So a ceiling near 0.10 Y
is where a design of Stage 1's size can show a positive effect and no more; below it even that needs a
larger sample, which is why the lower tier asks for the calculation up front. 0.08 Y is the floor: there
half-capture is 0.04 Y, about Stage 1's half-width. The old game's one-sided ceiling of 0.043 Y fails
clearly. "Go" means a live study is worth designing; its registration still needs its own power
calculation (NEG-33). The tiers are a judgment about detectability, not a derived quantity.

**The one fallback.** The hard regime is read with the same configuration and the same tiers. If easy is
a no-go and hard is not, hard becomes the primary regime as a recorded revision of this note, and any
write-up says the regime was the fallback. If easy is conditional, easy stays primary whatever hard shows.
No other configuration may be promoted after the sweep.

Reported with the gate, and read as stated here:

- **Sweep.** Q = 0…6, both regimes, ask cost 0, 1/100, 3/100 Y, per instance and pooled; D_rest; questions
  asked per episode; the share of episodes where the optimal move is to walk.
- **Greedy gap.** V_words under T5 minus V_words under T4 and under value-greedy, at each Q.
- **Re-offer variant.** The same game with H = 4 moves that may be asks or binding offers, a rejected
  offer continuing with probability 1 − h for h = 0, 1/10, 1/4, and D_words there defined as in the
  one-sided sweep (with asks minus offers only). If the gate passes but this number at h = 0 is below
  0.04 Y, the words matter *because* offers are scarce: the live protocol must then keep the single-offer
  rule, and any write-up must say the result is conditional on it.
- **Type count.** W per instance against the instance's number of types. The complaint about Car Dealer
  was that its model was too simple; nine instances from 9 to 26 types read on that directly. If W does
  not rise with type count, type count was not the problem.
- **Sensitivity prior.** Every value above under the value-pair constraints of §2.
- **No-go in both regimes.** The game is revised (candidates: a fourth item kind, a coarser bank) as a
  new spec version with its own bar stated before its sweep.

**Gate SD-D (NEG-31), the dilemma map.** Not a go/no-go, a set of predictions written before the
computation, at π = 1/2 under optimal play by A1 (π = 1/4 and 3/4 reported beside it):

- D1. Joint efficiency rises with disclosure: E(T2) ≥ E(T1) ≥ E(T0).
- D2. Being known costs the focal side: Y_A(T2) < Y_A(T1), the difference being (1 − π) times A's lost
  information rent.
- D3. With both sides restricted to "answer everything" or "decline everything", the 2 × 2 game is a
  prisoner's dilemma in Y for some regime: withholding against a discloser beats mutual disclosure, which
  beats mutual withholding, which beats disclosing to a withholder.
- D4. In T3, A1 answers some questions and declines others, and which ones depends on A's type.

Any of these may fail. D3 failing in both regimes would mean this game has no negotiator's dilemma under
its rules, and the two-way motivation would need a different mechanism (for example a private outside
option) before a live study is worth designing.

**Gate SD-R (NEG-32), the reader on human language.** Needs an explicit go, since it is the first API
spend. Before any live counterpart: calibration and posterior mass on the true type when the likelihood
reader is run over Deal or No Deal and CaSiNo dialogues, whose true values and priorities are recorded.
Corpus dialogues are scored under the corpus's own prior, whatever §2 uses.

## 8. Relation to existing games

Everything this section says about outside systems is from memory of their papers and documentation.
NEG-27 checks each statement against the source, and none of the design above depends on them being exact.

- **Deal or No Deal** (Lewis et al., 2017) is the source of the shape: three item kinds, five to seven
  items, integer values worth 10 on the whole pool for each side, a dialogue and then a single final
  selection with nothing for a mismatch. Split Deal departs from it in four ways: a positive outside option instead of zero, bank
  questions instead of free chat, one binding offer with accept or reject instead of simultaneous entry,
  and independent uniform types as the primary prior, with the paper's constraints on value pairs kept
  only as the sensitivity prior of §2.
- **CaSiNo** has three kinds, three units each and three priority levels: six orderings per side, the
  same size as the old game. It is a language corpus for gate SD-R, not a game shape to adopt.
- **NegMAS** computes Pareto and Nash statistics for explicit utility functions. It is used as an
  independent check on L0 and later as a source of standard opponents (NEG-34); the belief programs are
  ours, because NegMAS scenarios are single profile pairs with no type prior and its protocol has no
  query move.
- **`elicit_negotiation`** is the nearest thing already in the repo: one-sided, 45 types, a one-shot
  offer, token-as-cost. Split Deal keeps its one-shot offer and adds the second side, the instance
  family, and the frontier metrics.

## 9. Not in v0

Lying answers (NEG-22). Free-form proposals during the talk, which a reacting counterpart would turn
back into tests. Private or unequal outside options. More than three item kinds. Counterpart types
outside the modelled set (NEG-21). More than two parties.

## 10. Decisions (settled by Meadowlark, 2026-11-10, before any decision value was computed)

| # | Question | Decision | Where |
|---|---|---|---|
| 1 | Name and package | "Split Deal", `llm_rl_scripts/split_deal`, study prefix SD | title |
| 2 | Primary configuration and bar of gate SD-W | Q = 3; two tiers: go at 0.10 Y, conditional from 0.08 Y with a power calculation, no-go below | §7 |
| 3 | Type prior | Independent uniform is primary; the value-pair constraints are a reported sensitivity prior | §2, §7 |
| 4 | When the proposer is known | Drawn after the talk; π = 1/2 primary, 1/4 and 3/4 reported | §4.2 |
| 5 | Primary regime | Easy; hard is the one pre-named fallback | §7 |
| 6 | The two smallest instances (1-3-3, 2-2-3) | Kept; W against type count is a reported diagnostic | §7 |

## 11. Work order

NEG-27 (verify the outside sources) can run alongside. Then NEG-29 (L0) and NEG-30 (L1 and gate SD-W),
both offline. NEG-31 (L2 and gate SD-D) only if SD-W is go or conditional, in easy or in the fallback.
NEG-32 (gate SD-R) needs a go. NEG-33 decides whether to register a live study, and owns the power
calculation the conditional tier asks for.
