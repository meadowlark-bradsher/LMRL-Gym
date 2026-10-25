"""Exact game primitives: deal catalog, private types, and utilities.

All utilities are computed with :class:`fractions.Fraction` so that
feasibility, Pareto, and Nash comparisons are exact (no float ties).

Canonical, versioned index (see spec section 10):

    price:    P0 = low,   P1 = medium, P2 = high
    warranty: W0 = short,  W1 = medium, W2 = long
    delivery: D0 = slow,   D1 = medium, D2 = fast

The level index stored on a :class:`Deal` *is* this canonical index, so a
deal id is simply ``f"P{price}-W{warranty}-D{delivery}"``.

Preference *directions* are public and fixed by role: the buyer always
prefers lower price / longer warranty / faster delivery; the seller
always prefers the opposite. What is *private* is each agent's strict
ranking of which attribute matters most.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple

ATTRIBUTES: Tuple[str, str, str] = ("price", "warranty", "delivery")
LEVELS: Tuple[int, int, int] = (0, 1, 2)

# Priority weights for (highest, middle, lowest) ranked attribute, /13.
# On three-level attributes these make the ordering unambiguous:
#   9 * 0.5 > 3 + 1   (moving the top attribute one level beats the rest)
#   3 * 0.5 > 1       (moving the middle attribute one level beats the lowest)
RANK_WEIGHTS: Tuple[Fraction, Fraction, Fraction] = (
    Fraction(9),
    Fraction(3),
    Fraction(1),
)
WEIGHT_DENOM: Fraction = Fraction(13)

# Human-readable level labels, indexed by canonical level.
LEVEL_LABELS: Dict[str, Tuple[str, str, str]] = {
    "price": ("low", "medium", "high"),
    "warranty": ("short", "medium", "long"),
    "delivery": ("slow", "medium", "fast"),
}

Ranking = Tuple[str, str, str]  # attributes ordered highest -> lowest priority


@dataclass(frozen=True, order=True)
class Deal:
    """One of the 27 legal deals, identified by canonical level indices."""

    price: int
    warranty: int
    delivery: int

    @property
    def id(self) -> str:
        return f"P{self.price}-W{self.warranty}-D{self.delivery}"

    def level(self, attribute: str) -> int:
        return getattr(self, attribute)

    def label(self) -> str:
        return ", ".join(
            f"{attr}={LEVEL_LABELS[attr][self.level(attr)]}" for attr in ATTRIBUTES
        )

    @classmethod
    def from_id(cls, deal_id: str) -> "Deal":
        try:
            p, w, d = deal_id.split("-")
            return cls(int(p[1:]), int(w[1:]), int(d[1:]))
        except (ValueError, IndexError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"malformed deal id: {deal_id!r}") from exc


# The complete, stable deal catalog (3 * 3 * 3 = 27).
ALL_DEALS: Tuple[Deal, ...] = tuple(
    Deal(p, w, d) for p, w, d in itertools.product(LEVELS, repeat=3)
)
DEALS_BY_ID: Dict[str, Deal] = {d.id: d for d in ALL_DEALS}

# The 6 possible strict rankings per agent (3! permutations).
ALL_RANKINGS: Tuple[Ranking, ...] = tuple(itertools.permutations(ATTRIBUTES))


def buyer_satisfaction(attribute: str, level: int) -> Fraction:
    """Buyer satisfaction on one attribute at a canonical level, in {0, 1/2, 1}.

    The buyer prefers low price (P0), long warranty (W2), fast delivery (D2).
    """
    if attribute == "price":
        return Fraction(2 - level, 2)  # low price (level 0) is best for the buyer
    # warranty / delivery: higher canonical index is better for the buyer
    return Fraction(level, 2)


def satisfaction(role: str, attribute: str, level: int) -> Fraction:
    """Per-attribute satisfaction for ``role`` ("buyer" or "seller").

    Seller satisfaction is the exact complement of the buyer's on every
    attribute/level (opposite directions on a symmetric three-level scale).
    """
    s = buyer_satisfaction(attribute, level)
    if role == "buyer":
        return s
    if role == "seller":
        return 1 - s
    raise ValueError(f"role must be 'buyer' or 'seller', got {role!r}")


def utility(role: str, ranking: Ranking, deal: Deal) -> Fraction:
    r"""Exact private utility ``U_i(d; theta_i)`` in ``[0, 1]``.

    ``U = (9*s(a1) + 3*s(a2) + 1*s(a3)) / 13`` where ``(a1, a2, a3)`` is the
    agent's ranking from highest to lowest priority.
    """
    if len(ranking) != 3 or set(ranking) != set(ATTRIBUTES):
        raise ValueError(f"ranking must be a permutation of {ATTRIBUTES}, got {ranking!r}")
    total = Fraction(0)
    for position, attribute in enumerate(ranking):
        total += RANK_WEIGHTS[position] * satisfaction(role, attribute, deal.level(attribute))
    return total / WEIGHT_DENOM


def all_type_pairs() -> List[Tuple[Ranking, Ranking]]:
    """All 6 * 6 = 36 ordered (buyer_rank, seller_rank) pairs."""
    return [(b, s) for b in ALL_RANKINGS for s in ALL_RANKINGS]
