"""Split Deal: instances, types, allocations and the question bank (`DESIGN_SPLIT.md` §2-§3).

Two sides split a pool of items. An instance is the public count of each item kind. A side's
private type is its per-unit value for each kind: non-negative integers with
sum(count * value) = TOTAL, so the whole pool is worth TOTAL to every type and the type set of
an instance is finite and known to both sides. An allocation gives the focal side ``a[k]`` units
of kind k and the counterpart the rest. Disagreement pays each side its outside option
d = TOTAL * r; a side accepts an allocation iff its utility is >= d.

Nothing here plays the game; the oracles (L0 frontier, L1 belief DP, L2 two-sided) build on it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple

TOTAL = 10
ITEMS = ("book", "hat", "ball")
REGIMES = {"easy": Fraction(2, 5), "hard": Fraction(11, 20)}     # r, as in the Car Dealer studies

Values = Tuple[int, ...]
Alloc = Tuple[int, ...]


@dataclass(frozen=True)
class Instance:
    counts: Tuple[int, ...]

    @property
    def key(self) -> str:
        return "-".join(str(c) for c in self.counts)

    def types(self) -> List[Values]:
        """Every value vector worth TOTAL on the whole pool, in lexicographic order."""
        ranges = [range(TOTAL // c + 1) for c in self.counts]
        return [v for v in itertools.product(*ranges) if self.utility(v, self.counts) == TOTAL]

    def allocations(self) -> List[Alloc]:
        return list(itertools.product(*[range(c + 1) for c in self.counts]))

    def rest(self, a: Alloc) -> Alloc:
        return tuple(c - x for c, x in zip(self.counts, a))

    @staticmethod
    def utility(v: Values, share: Alloc) -> int:
        return sum(x * y for x, y in zip(v, share))


def canonical_instances(min_items: int = 5, max_items: int = 7, kinds: int = len(ITEMS)) -> List[Instance]:
    """Count vectors with every kind present, up to relabelling of kinds (sorted ascending)."""
    out = []
    for n in range(min_items, max_items + 1):
        for c in itertools.combinations_with_replacement(range(1, n + 1), kinds):
            if sum(c) == n:
                out.append(Instance(c))
    return out


def outside_option(regime: str) -> Fraction:
    return TOTAL * REGIMES[regime]


def y_units(regime: str, utility) -> Fraction:
    """Net gain over the outside option as a fraction of the most a side could gain."""
    d = outside_option(regime)
    return (Fraction(utility) - d) / (TOTAL - d)


# -- question bank -----------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    kind: str                    # ZERO | TOP | PREF | HALF
    args: Tuple[int, ...]

    @property
    def id(self) -> str:
        return f"{self.kind}({','.join(ITEMS[k] for k in self.args)})"


def bank(kinds: int = len(ITEMS)) -> List[Question]:
    """Twelve yes/no questions about the answerer's values (for three kinds)."""
    qs = [Question("ZERO", (k,)) for k in range(kinds)]
    qs += [Question("TOP", (k,)) for k in range(kinds)]
    qs += [Question("PREF", jk) for jk in itertools.combinations(range(kinds), 2)]
    qs += [Question("HALF", (k,)) for k in range(kinds)]
    return qs


def answer(inst: Instance, v: Values, q: Question) -> bool:
    """The truthful answer of type ``v``.

    ZERO(k): a unit of k is worth nothing.   TOP(k): no kind is worth more per unit than k.
    PREF(j,k): a unit of j is worth strictly more than a unit of k.
    HALF(k): all the units of k together are worth at least half the pool.
    """
    if q.kind == "ZERO":
        return v[q.args[0]] == 0
    if q.kind == "TOP":
        return v[q.args[0]] == max(v)
    if q.kind == "PREF":
        return v[q.args[0]] > v[q.args[1]]
    if q.kind == "HALF":
        return 2 * inst.counts[q.args[0]] * v[q.args[0]] >= TOTAL
    raise ValueError(q.kind)


def signature(inst: Instance, v: Values, qs: List[Question]) -> Tuple[bool, ...]:
    return tuple(answer(inst, v, q) for q in qs)
