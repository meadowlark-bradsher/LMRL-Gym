"""Split Deal v0.3: a private outside option as part of each side's type (`DESIGN_SPLIT_V03.md`).

A type is now (values, level): the per-unit values of v0.2 and an outside-option level drawn
independently of them, with d(level) from ``LEVELS``. A side accepts an allocation iff its utility is
at least its own d, and its Y is net gain over its own d as a share of 10 - d. The bank gains one
question, HIGH ("is your outside option the high one?").

``PrivateTables`` exposes the attributes the L1 program (``belief.Cell``) and the L2 program
(``twosided.Program``) read from ``belief.Tables``: ``inst``, ``types``, ``allocs``, ``questions``,
``full``, ``acc``, ``yes`` and ``gains``. Both programs therefore run unchanged on the v0.3 game.
With a single level and no HIGH question it reproduces ``Tables`` exactly (tested).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from .game import ITEMS, TOTAL, Instance, Question, Values, answer, bank

LEVELS: Dict[str, Fraction] = {"lo": Fraction(4), "hi": Fraction(11, 2)}     # the easy and hard d of v0.2
HIGH = Question("HIGH", ())

PType = Tuple[Values, str]


def answer_private(inst: Instance, t: PType, q: Question, levels: Dict[str, Fraction] = LEVELS) -> bool:
    if q.kind == "HIGH":
        return levels[t[1]] == max(levels.values())
    return answer(inst, t[0], q)


class PrivateTables:
    def __init__(self, inst: Instance, levels: Dict[str, Fraction] = LEVELS, high_question: bool = True):
        self.inst, self.levels = inst, dict(levels)
        self.types: List[PType] = [(v, lvl) for v in inst.types() for lvl in self.levels]
        self.allocs = inst.allocations()
        self.questions = bank(len(ITEMS)) + ([HIGH] if high_question else [])
        self.full = (1 << len(self.types)) - 1
        self.acc = [self._mask(lambda t, a=a: inst.utility(t[0], inst.rest(a)) >= self.d(t)) for a in self.allocs]
        self.yes = [self._mask(lambda t, q=q: answer_private(inst, t, q, self.levels)) for q in self.questions]

    def d(self, t: PType) -> Fraction:
        return self.levels[t[1]]

    def scale(self, t: PType) -> Fraction:
        return TOTAL - self.d(t)

    def _mask(self, pred) -> int:
        return sum(1 << i for i, t in enumerate(self.types) if pred(t))

    def gains(self, t: PType) -> List[Fraction]:
        d, s = self.d(t), self.scale(t)
        return [(Fraction(self.inst.utility(t[0], a)) - d) / s for a in self.allocs]

    def s_star(self, ta: PType, tb: PType):
        """Maximum joint surplus over allocations acceptable to both, in utility units; None if there is none."""
        best = None
        for a in self.allocs:
            ua, ub = self.inst.utility(ta[0], a), self.inst.utility(tb[0], self.inst.rest(a))
            if ua >= self.d(ta) and ub >= self.d(tb):
                s = ua + ub - self.d(ta) - self.d(tb)
                best = s if best is None or s > best else best
        return best
