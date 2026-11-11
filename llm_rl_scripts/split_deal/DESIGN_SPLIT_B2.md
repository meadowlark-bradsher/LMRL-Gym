# Split Deal v0.3 — B2, the counterpart that reads A (2026-11-10; decisions in §7 settled, §6 frozen; no B2 value computed)

Jira NEG-39, under the redesign epic NEG-25. Follows gate SD3-D (`SWEEP_SPLIT_V03_L2.md`, PR #66). The game is
v0.3 as registered in `DESIGN_SPLIT_V03.md`; nothing about it changes here. This note adds a counterpart and
states, before any value is computed against it, what it is expected to show.

## 1. Why

Against B0, which reads nothing into a decline, three registered conditions of SD3-D were met, and all three name
B2 as the next rung (`DESIGN_SPLIT_V03.md` §5, §8):

- **D2′-lo failed.** Being known still raised a lo-level A's value.
- **D3′ failed.** The 2 × 2 game had no dilemma: answering everything was dominant for both sides.
- **D4′ held.** Question by question, a lo-level A hid its level by declining HIGH more often than not, and a
  hi-level A mostly answered it. The note says this finding "is specific to B0".

The open question is whether that hiding survives a counterpart that reads it. In the unravelling argument, it does
not: once a decline is read as "lo", declining stops protecting a lo type.

## 2. What v0.2 already fixed about B2

`DESIGN_SPLIT.md` §5 registered the rung before any of this was computed: **B2 is the exact best response to A1**,
and it is "where a counterpart starts to read A's declines and question choices". The ladder is built so that every
rung is an exact best response and no fixed point is needed; the equilibrium question goes to a CFR check on the
smallest instance. This note keeps that definition (decision 1) and fills in what it leaves open.

## 3. B2

B2 sits in B's seat, knows its own type, and knows A1's policy for every A type (A1 being the exact best response to
B0 in the same structure, `twosided.py`). Against that model:

- **Belief.** B2's belief about A is the set K of A types whose A1 policy is consistent with everything A has done:
  the questions A asked, the answers A gave, and the questions A declined. The prior is uniform and A1 is
  deterministic, so the posterior is uniform over K and stays a bitmask, like every other belief here.
- **Questions.** B2 picks the question (or a pass) that maximises its own expected Y, given K and A1's predicted
  answers. B0's rule was the most expected information; B2's is the value of the information to B.
- **Offer.** As proposer, B2 makes the offer that maximises its expected gain over the A types in K: B0's offer
  rule, applied to B2's belief.
- **Answering.** Fixed by the structure, as for B0 (decision 5). In T3 B2 answers iff A answered its latest
  question.
- **Accepting.** As responder, B2 accepts iff U_B ≥ its own d. That is optimal at the last move, so it is not a
  choice.

**A1 off its own path.** B2 can ask questions B0 would not have asked, so it needs A1's answer to any question at
any state. A1 has one: it answers question j iff answering is strictly better than declining when B0 plays the
rest of the game (`Program.value` at the two continuations, ties to declining). This is A1 as built, applied to
states B0 would not reach; A1 still believes B is B0.

**A3.** The next rung is the exact best response to B2. B2's question choices depend on B's type, so A3 reads them,
just as B2 reads A1's. B2 still believes A is A1, so A3 can take an action that no type in K would take under A1.
B2's off-path rule (decision 4) treats the two kinds of evidence differently:

- **An ask or a decline** that no type in K would make is ignored: K is unchanged.
- **A truthful answer** is hard evidence. If no type in K fits it, K resets to the set of types consistent with all
  of A's answers so far, the belief B0 would hold.

The rule neither punishes a deviation, as a sceptical reading would, nor lets A erase what B2 has inferred, as a
reset on every deviation would. The share of (A3, B2) episodes in which A3's true type leaves K is reported.

**What gets scored.** Two pairings, both against the same B2:

- **(A1, B2): exploitability.** A1 is unchanged; B2 reads it. This measures what B0's blindness was worth to A.
- **(A3, B2): the response.** A adapts to being read. This measures whether hiding survives a counterpart that reads
  it, at level 3 of the ladder. Whether it survives in equilibrium is the CFR check's question, not this one.

**Tie convention (decision 3).** The primary A1 is A1 exactly as registered, ties to declining. A sensitivity
rebuilds B2 and A3 on A1′, the same program with ties broken to answering. A1′ has the same value as A1 against
B0; it differs only in which of equally good moves it takes.

**Decomposition (decision 1).** The variant B2-d keeps B0's question rule and offer rule but replaces B0's belief by
K built from A's declines only. It separates two effects: what reading declines is worth, and what B2's better
choice of questions is worth.

## 4. Two facts from SD3-D that bear on the design

Both are counts already published in `SWEEP_SPLIT_V03_L2.md` (from `results/split_v03_l2_20261110/sweep.json`);
nothing here is a B2 value.

- **How informative a decline is to a reader of A1.** On A1's on-path T3 decisions at B0's HIGH, π = 1/2, pooled over
  instances: lo-level types declined 3097 of 5644, and hi-level types 864 of 5654. Pooled over histories, a decline
  therefore comes from a lo-level type 3097/3961 ≈ 0.78 of the time. B2's posterior is per history, so its readings
  will differ from this aggregate, but the aggregate says how much a decline can tell it.
- **Most hi-level declines are ties.** 649 of the 864 hi-level declines are ties: answering and declining are worth
  the same to A1 against B0, and the tie-break sends them to declining. Counting only strict declines, a decline
  would come from a lo-level type 2488/2703 ≈ 0.92 of the time. A1's tie-break does not change A1's value against
  B0, but it changes what a reader of A1 learns. That is decision 3.

## 5. What B2 changes and what it cannot

- **In T0** nobody asks, so B2 has nothing to read: it makes B0's offer on the prior, and T0 is unchanged.
- **In the 2 × 2 cells where A answers nothing,** declines carry no information because every type declines. A's
  question choices still do.
- **In T1, T2 and the 2 × 2,** A1's questions are type-dependent, so B2 reads them. Asking therefore reveals
  something about A, which it never did against B0. This is a new channel and it can move D2′ and D3′.
- **In T3,** both channels operate. Selective declining is where D4′ lived.
- **By construction:** against A1, B2 does at least as well for B as B0 (B0's policy is one B2 could play), and A3
  does at least as well for A against B2 as A1. These are checks, not predictions.

## 6. Predictions and bars, stated before any B2 value is computed

Frozen 2026-11-10 with the decisions of §7. Gate name: **SD3-K** (the level-k rung). Shared reading unless a
prediction says otherwise: v0.3 game, 13-question bank, Q = 3, π = 1/2, independent uniform prior, own-scale Y,
pooled means as in SD3-D (instances equal, types uniform within an instance), A1 as built (ties to declining). The
comparison points against B0 are the stored SD3-D values in `results/split_v03_l2_20261110/sweep.json`.

- **K1 (B2 reads A1 profitably).** In T3, Y_B(A1, B2) > Y_B(A1, B0). The weak inequality holds by construction (§5);
  the prediction is that it is strict.
- **K2 (hiding is exploitable).** Against B2, a lo-level A1 does worse choosing what to answer than answering
  everything: Y_A^lo(T3; A1, B2) < Y_A^lo(T2; A1, B2), both terms in the same pairing against the same B2 (which
  best-responds to A1's policy in each structure). Against B0 the order was the reverse.
- **K3 (hiding survives level 3).** In T3, at on-path decisions where B2 asks HIGH, a lo-level A3 still declines more
  than half the time, and a hi-level A3 answers more often than a lo-level one. The reasoning: B2 reads a decline
  through A1's policy, in which some hi-level types decline (on ties, decision 3), so a decline still pools a
  lo-level A with some hi-level types.
  *Support rule:* B2 picks questions by their value to B and may seldom ask HIGH. If it asks HIGH in fewer than
  100 on-path T3 decisions for either A level (pooled over instances), K3 is recorded as **not testable**, not as
  failing. The counts are reported either way.
- **K4 (hiding is worth less against a reader).** A lo-level A's gain from choosing over answering everything,
  Y_A^lo(T3) − Y_A^lo(T2), is smaller in (A3, B2), both terms against B2, than it was in (A1, B0), both terms the
  stored SD3-D values.
- **K5 (still no all-or-nothing dilemma).** The 2 × 2 game in which B2 best-responds to A1 in each cell is not a
  strict prisoner's dilemma for both players, read on pooled means as D3′.

Reported, not predicted: K1–K5 on the common scale; the sensitivities (A1′ with ties to answering, the no-HIGH
control, π = 1/4 and 3/4, the decomposition B2-d); how often B2 asks HIGH; the share of (A3, B2) episodes in which
A3's true type leaves K; and the D1′–D3′ readings with B2.

*Registered consequences:*

- **K2 and K3 hold.** Hiding is exploitable by a reader, yet it persists against B2, because B2 reads A through a
  model A3 has outgrown. Only an equilibrium can say whether it lasts, so the CFR check on the smallest instance
  comes next, before any live design.
- **K3 fails.** Disclosure already unravels at level 3. Split Deal's selective hiding was an artifact of a
  counterpart that does not read declines, and the two-sided live design carries no hiding treatment.
- **K5 fails.** A strict dilemma appears only against a counterpart that reads A. Any two-sided live study must then
  give its counterpart a reading rule of B2's kind, and NEG-33 inherits that constraint.
- **K3's verdict differs under A1′** (ties to answering). Whether hiding survives a reader then depends on how
  the read player breaks ties it is indifferent about, and the write-up says so wherever K3 is cited.
- **K5 holds.** The all-or-nothing dilemma is absent in Split Deal against every counterpart tried (v0.2 with B0,
  v0.3 with B0, v0.3 with B2). The two-way motivation then rests on selective disclosure alone, and the write-up
  says so.

## 7. Decisions (settled by Meadowlark, 2026-11-10)

| # | Question | Settled | Not taken |
|---|---|---|---|
| 1 | What B2 is | the exact best response to A1, reading A's questions, answers and declines (v0.2 §5); B2-d (B0's questions and offer, belief from declines only) reported as a decomposition | exact BR only; B2-d only |
| 2 | Pairings scored | (A1, B2) for exploitability (K1, K2) and (A3, B2) for the response (K3, K4), both gated; K5 on the 2 × 2 against B2; the ladder stops at A3 | (A1, B2) only; continue to B4 |
| 3 | How B2 models A1's ties | A1 as registered (ties to declining) primary; A1′ (ties to answering) as a sensitivity; a registered consequence if K3's verdict differs | as built only; ties-to-answer primary |
| 4 | B2's off-path rule | an ask or decline no type in K would make is ignored; an answer no type in K fits resets K to the answers-consistent set; misread share reported | sceptical (worst type for A); reset on any deviation |
| 5 | B2's answering | fixed by the structure, as for B0 | B2 chooses |
| 6 | Structures and scope | primary (13-question bank, A1 as built, π = 1/2): T0–T3 and the 2 × 2, all nine instances, both pairings; each sensitivity runs T2, T3 and the 2 × 2 only | every structure everywhere; T2, T3 and the 2 × 2 only |
| 7 | Predictions to register | K1–K5 with K3's support rule, the T2 comparison named in K2 and K4, and decision 3's consequence | as drafted; drop K5 |
| 8 | Runtime pilot | allowed on 2-2-3 before the run, printing seconds only, values discarded unread; the run plan is fixed from it; no instance is ever dropped for cost | no pilot; pilot with a dropping order |

## 8. Work order

1. ~~Settle §7 and freeze §6.~~ Done 2026-11-10.
2. Build B2, B2-d, A1′ and A3 as programs beside `twosided.Program`. Test them against naive literal simulations on
   2-2-3, and check the two by-construction inequalities of §5 and that rollouts realise the computed values.
3. Run the seconds-only pilot on 2-2-3 and fix the run plan (decision 8).
4. Commit the code with the operational reading in its docstring, then run gate SD3-K, then commit the results.
5. Then the CFR check on the smallest instance.
6. Gate SD-R (corpus reading, NEG-32) is unchanged and still needs an explicit go.
