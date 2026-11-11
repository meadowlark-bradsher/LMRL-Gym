"""L2, the two-sided program of Split Deal: rule counterpart B0 and A's exact best response A1.

Protocol 4.2 of `DESIGN_SPLIT.md`: Q rounds of talk, each an A turn then a B turn; then the proposer is
drawn (A with probability pi), makes one binding offer or walks, and the responder accepts iff its
utility is >= d. Answers are truthful or declined; nothing is inferred from a decline or from which
question was asked. Values are exact Fractions in Y units, for each side.

B0 (rule counterpart):
* asks, at each of its turns, the question with the most expected information about A's type (least
  sum k log k over the answer parts of its uniform posterior S_A) among those it has not asked and that
  split S_A, lowest bank index on ties; passes if none;
* answers as the information structure says (always, never, or in T3 iff A answered B's latest
  question, or B has not asked yet);
* as proposer, makes the offer that maximises its own expected net gain over S_A: the same rule as
  A's final offer in L1 seen from B's seat (``belief.Cell.final`` with B's type), ties to the higher
  acceptance count, then the lower index of B's share, walking at 0.

A1 (exact best response to B0), knowing its own type and B0's rule:
* asks only when B will answer, a question that splits its posterior S_B over B's type;
* in T3 decides at each of B's questions whether to answer or decline;
* as proposer makes the L1 final offer for S_B; as responder accepts iff U_A >= d.
A1 maximises pi * (its value as proposer) + (1 - pi) * (its value as responder). Ties: not asking
before asking, then the lower bank index; declining before answering.

Information structures (§4.3), as parameters (who asks; how each side answers):

    T0  nobody asks
    T1  A asks; B always answers; B does not ask
    T2  both ask; both always answer
    T3  both ask; A chooses; B answers iff A answered B's latest question (or B has not asked yet)

and the 2 x 2 disclosure game of prediction D3: both ask; each side answers everything or nothing.

Operational choices for gate SD-D, fixed here before the sweep ran (the spec leaves them open):
* Q = 3 rounds, the Q of gate SD-W; pi = 1/2 primary, 1/4 and 3/4 reported; both regimes reported,
  easy primary for D1, D2 and D4 (decision 5 of §10); D3 holds if it holds in either regime.
* Independent uniform prior only. Under the value-pair prior B0's questions would depend on B's type
  and so carry information about it, which needs the inference-reading counterpart B2 (not built).
* D1, D2, D3 are read on pooled means (instances weighted equally, then A type and B type uniform).
* D3 holds in a regime iff the strict prisoner's-dilemma ordering holds for both players:
  for each, withhold-vs-disclose > both disclose > both withhold > disclose-vs-withhold. A second
  reading, with B asking by the exact program instead of B0's EIG rule, is reported beside it.
* D4 holds iff, over all on-path decisions of A1 in T3 (easy, pi = 1/2), A1 answers at least one
  question strictly (answering is strictly better than declining; ties go to declining), declines at
  least one, and some question is answered by some A types and declined by others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .belief import Cell, Tables

ZERO = Fraction(0)


@dataclass(frozen=True)
class Structure:
    a_asks: bool
    b_asks: bool
    a_answers: str          # "all" | "none" | "choose"
    b_answers: str          # "all" | "none" | "reciprocal"


STRUCTURES = {
    "T0": Structure(False, False, "none", "none"),
    "T1": Structure(True, False, "none", "all"),
    "T2": Structure(True, True, "all", "all"),
    "T3": Structure(True, True, "choose", "reciprocal"),
}
# the 2 x 2 disclosure game: (A's answering, B's answering)
GAME = {(a, b): Structure(True, True, a, b) for a in ("all", "none") for b in ("all", "none")}


class Shared:
    """What every A type of one (instance, regime) shares: tables, B's seat cells, the mirror of allocations."""

    def __init__(self, tables: Tables):
        self.t = tables
        index = {a: i for i, a in enumerate(tables.allocs)}
        self.mirror = [index[tables.inst.rest(a)] for a in tables.allocs]     # proposer share -> responder share
        self.seat = [Cell(tables, v, tables.full) for v in tables.types]       # each type in the proposer seat
        self._b0q: Dict[Tuple[int, frozenset], Optional[int]] = {}

    def b0_question(self, SA: int, declined: frozenset) -> Optional[int]:
        """B0's question: most expected information about A over S_A, among unasked questions that split it.

        Questions B asked and A answered no longer split S_A, so only declined ones need excluding."""
        key = (SA, declined)
        if key in self._b0q:
            return self._b0q[key]
        best, choice = None, None
        for j, y in enumerate(self.t.yes):
            if j in declined:
                continue
            k1, k0 = (SA & y).bit_count(), (SA & ~y).bit_count()
            if k1 == 0 or k0 == 0:
                continue
            e = k1 * math.log(k1) + k0 * math.log(k0)
            if best is None or e < best:
                best, choice = e, j
        self._b0q[key] = choice
        return choice

    def offer(self, b: int, SA: int) -> Optional[int]:
        """B0's offer as proposer (type b, belief S_A), as an index of the share it gives A; None to walk."""
        k = self.seat[b].final(SA)[1]
        return None if k is None else self.mirror[k]


class Program:
    """A1 for one A type (index ``ia``) against B0, under one structure, Q rounds and pi."""

    def __init__(self, shared: Shared, ia: int, structure: Structure, q: int, pi: Fraction):
        self.s, self.ia, self.st, self.q, self.pi = shared, ia, structure, q, pi
        self.t = shared.t
        self.me = shared.seat[ia]                       # A in the proposer seat
        self._v: Dict[tuple, Fraction] = {}
        self._resp: Dict[Tuple[int, int], Fraction] = {}

    # -- outcomes --------------------------------------------------------------------

    def a_takes(self, k: Optional[int]) -> Optional[int]:
        """The share B0 offers A, if A accepts it (U_A >= d), else None."""
        if k is None:
            return None
        return k if self.me.gain[k] >= 0 else None

    def responder(self, SB: int, SA: int) -> Fraction:
        """A's expected value as responder: B0's offer averaged over B's types in S_B."""
        key = (SB, SA)
        hit = self._resp.get(key)
        if hit is not None:
            return hit
        total, n = ZERO, 0
        for b in range(len(self.t.types)):
            if (SB >> b) & 1:
                n += 1
                k = self.a_takes(self.s.offer(b, SA))
                if k is not None:
                    total += self.me.gain[k]
        v = total / n
        self._resp[key] = v
        return v

    def terminal(self, SB: int, SA: int) -> Fraction:
        return self.pi * self.me.final(SB)[0] + (1 - self.pi) * self.responder(SB, SA)

    # -- the program -------------------------------------------------------------------

    def b_will_answer(self, owed: bool) -> bool:
        return self.st.b_answers == "all" or (self.st.b_answers == "reciprocal" and owed)

    def value(self, r: int, phase: str, SB: int, SA: int, declined: frozenset, owed: bool) -> Fraction:
        return self.step(r, phase, SB, SA, declined, owed)[0]

    def step(self, r, phase, SB, SA, declined, owed):
        """(value, choice): at an A turn the question to ask or None; at a B turn True to answer."""
        if r == 0:
            return self.terminal(SB, SA), None
        key = (r, phase, SB, SA, declined, owed)
        hit = self._v.get(key)
        if hit is not None:
            return hit
        if phase == "A":
            best, choice = self.value(r, "B", SB, SA, declined, owed), None
            if self.st.a_asks and self.b_will_answer(owed):
                n = SB.bit_count()
                for j, s1, s0 in self.me.splits(SB):
                    v = (s1.bit_count() * self.value(r, "B", s1, SA, declined, owed)
                         + s0.bit_count() * self.value(r, "B", s0, SA, declined, owed)) / n
                    if v > best:
                        best, choice = v, j
            out = (best, choice)
        else:
            j = self.s.b0_question(SA, declined) if self.st.b_asks else None
            if j is None:
                out = (self.value(r - 1, "A", SB, SA, declined, owed), None)
            else:
                told = self._told(SA, j)
                v_ans = self.value(r - 1, "A", SB, told, declined, True)
                v_dec = self.value(r - 1, "A", SB, SA, declined | {j}, False)
                if self.st.a_answers == "all":
                    out = (v_ans, True)
                elif self.st.a_answers == "none":
                    out = (v_dec, False)
                else:
                    out = (v_ans, True) if v_ans > v_dec else (v_dec, False)
        self._v[key] = out
        return out

    def _told(self, SA: int, j: int) -> int:
        y = self.t.yes[j]
        return SA & y if (y >> self.ia) & 1 else SA & ~y

    def root(self) -> Fraction:
        return self.value(self.q, "A", self.t.full, self.t.full, frozenset(), True)

    # -- one episode against B type b ---------------------------------------------------

    def rollout(self, b: int) -> dict:
        SB, SA, declined, owed = self.t.full, self.t.full, frozenset(), True
        decisions = []                                   # (question, answered, strict)
        rounds = []                                      # the round (1..Q) of each decision
        r, phase = self.q, "A"
        while r > 0:
            _, choice = self.step(r, phase, SB, SA, declined, owed)
            if phase == "A":
                if choice is not None:
                    y = self.t.yes[choice]
                    SB = SB & y if (y >> b) & 1 else SB & ~y
                phase = "B"
            else:
                j = self.s.b0_question(SA, declined) if self.st.b_asks else None
                if j is not None:
                    if self.st.a_answers == "choose":
                        v_ans = self.value(r - 1, "A", SB, self._told(SA, j), declined, True)
                        v_dec = self.value(r - 1, "A", SB, SA, declined | {j}, False)
                        decisions.append((j, bool(choice), v_ans != v_dec))
                        rounds.append(self.q - r + 1)
                    if choice:
                        SA, owed = self._told(SA, j), True
                    else:
                        declined, owed = declined | {j}, False
                r, phase = r - 1, "A"
        k_a = self.me.final(SB)[1]
        a_deal = k_a if (k_a is not None and (self.t.acc[k_a] >> b) & 1) else None
        b_deal = self.a_takes(self.s.offer(b, SA))
        return {"SB": SB, "SA": SA, "a_deal": a_deal, "b_deal": b_deal, "decisions": decisions,
                "decision_rounds": rounds}
