"""Exhaustive truth-table scoring (spec section 4).

Everything here is computed by exact enumeration over the 27 deals, using
both hidden rankings. These are *evaluator-only* quantities: agents never
see them. Four score families are kept strictly separate:

  * individual outcome      -> G_B, G_S (section 4.1)
  * joint deal quality      -> Pareto frontier, Nash ratio Q_N (section 4.2)
  * epistemic quality       -> NOT computed here (section 4.3; out of scope
                               for pure Jev-only self-play)
  * operational success     -> deal-success / correct-no-deal (section 4.4)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .game import ALL_DEALS, Deal, Ranking, all_type_pairs, utility

# Reservation (outside-option) thresholds for the two v0 regimes.
EASY_RESERVATION = Fraction(2, 5)    # 0.40
HARD_RESERVATION = Fraction(11, 20)  # 0.55
RESERVATIONS: Dict[str, Fraction] = {
    "easy": EASY_RESERVATION,
    "hard": HARD_RESERVATION,
}


@dataclass(frozen=True)
class TypePairAnalysis:
    """Exhaustive ground truth for one (buyer_rank, seller_rank, r) cell."""

    buyer_rank: Ranking
    seller_rank: Ranking
    reservation: Fraction
    feasible: Tuple[Deal, ...]           # F(theta_B, theta_S)
    pareto_frontier: Tuple[Deal, ...]    # subset of feasible, not dominated in F
    buyer_u_star: Optional[Fraction]     # max U_B over F (None if F empty)
    seller_u_star: Optional[Fraction]    # max U_S over F (None if F empty)
    nash_max: Optional[Fraction]         # max (U_B-r)(U_S-r) over F (None if F empty)

    @property
    def is_feasible(self) -> bool:
        return len(self.feasible) > 0


def feasible_deals(buyer_rank: Ranking, seller_rank: Ranking, r: Fraction) -> List[Deal]:
    """F = deals individually rational for *both* sides (U_i >= r)."""
    return [
        d
        for d in ALL_DEALS
        if utility("buyer", buyer_rank, d) >= r and utility("seller", seller_rank, d) >= r
    ]


def pareto_frontier(
    deals: List[Deal], buyer_rank: Ranking, seller_rank: Ranking
) -> List[Deal]:
    """Deals in ``deals`` not Pareto-dominated by another deal in ``deals``.

    ``d`` is dominated by ``d'`` iff both agents are at least as well off
    under ``d'`` and at least one is strictly better off.
    """
    util = {
        d: (utility("buyer", buyer_rank, d), utility("seller", seller_rank, d)) for d in deals
    }
    front: List[Deal] = []
    for d in deals:
        ub, us = util[d]
        dominated = any(
            (ub2 >= ub and us2 >= us and (ub2 > ub or us2 > us))
            for d2 in deals
            if d2 != d
            for (ub2, us2) in (util[d2],)
        )
        if not dominated:
            front.append(d)
    return front


def nash_product(d: Deal, buyer_rank: Ranking, seller_rank: Ranking, r: Fraction) -> Fraction:
    """N(d) = (U_B(d) - r)(U_S(d) - r); the normalized Nash surplus numerator."""
    return (utility("buyer", buyer_rank, d) - r) * (utility("seller", seller_rank, d) - r)


def analyze_type_pair(
    buyer_rank: Ranking, seller_rank: Ranking, r: Fraction
) -> TypePairAnalysis:
    """Exhaustively analyze one type pair under reservation ``r``."""
    feasible = feasible_deals(buyer_rank, seller_rank, r)
    if feasible:
        buyer_u_star = max(utility("buyer", buyer_rank, d) for d in feasible)
        seller_u_star = max(utility("seller", seller_rank, d) for d in feasible)
        nash_max = max(nash_product(d, buyer_rank, seller_rank, r) for d in feasible)
        front = tuple(pareto_frontier(feasible, buyer_rank, seller_rank))
    else:
        buyer_u_star = seller_u_star = nash_max = None
        front = ()
    return TypePairAnalysis(
        buyer_rank=buyer_rank,
        seller_rank=seller_rank,
        reservation=r,
        feasible=tuple(feasible),
        pareto_frontier=front,
        buyer_u_star=buyer_u_star,
        seller_u_star=seller_u_star,
        nash_max=nash_max,
    )


def attainable_gain(
    realized: Fraction, u_star: Fraction, r: Fraction
) -> Optional[Fraction]:
    r"""Normalized attainable-gain score ``G_i = (U_i - r) / (U_i^* - r)``.

    Returns ``None`` (undefined) if the denominator is zero, per the spec's
    "report rather than divide by zero" rule.
    """
    denom = u_star - r
    if denom == 0:
        return None
    return (realized - r) / denom


@dataclass(frozen=True)
class EpisodeScore:
    """Scored outcome of one episode against its ground truth (section 4)."""

    feasible: bool                      # was F non-empty for this type pair?
    agreement: bool
    valid_agreement: bool               # agreement on an IR deal
    correct_no_deal: bool               # F empty and ended without an invalid accept
    buyer_utility: Fraction
    seller_utility: Fraction
    buyer_gain: Optional[Fraction]      # G_B (None if infeasible or denom zero)
    seller_gain: Optional[Fraction]     # G_S
    min_gain: Optional[Fraction]        # balanced individual attainment min(G_B, G_S)
    on_pareto_frontier: Optional[bool]  # None if no accepted deal / infeasible
    nash_ratio: Optional[Fraction]      # Q_N (None if infeasible; 0 for no-deal)


def score_episode(result, analysis: TypePairAnalysis) -> EpisodeScore:
    """Score one :class:`protocol.EpisodeResult` against its truth table.

    ``result`` and ``analysis`` must share the same type pair and reservation.
    Kept here (not in ``protocol``) so the engine stays agent- and
    evaluator-agnostic.
    """
    assert result.buyer_rank == analysis.buyer_rank
    assert result.seller_rank == analysis.seller_rank
    assert result.reservation == analysis.reservation
    r = analysis.reservation
    feasible = analysis.is_feasible

    valid_agreement = bool(result.agreement and result.individually_rational)
    correct_no_deal = (not feasible) and result.outcome_type != "invalid_acceptance" and not result.agreement

    buyer_gain = seller_gain = min_gain = None
    nash_ratio: Optional[Fraction] = None
    on_frontier: Optional[bool] = None

    if feasible:
        assert analysis.buyer_u_star is not None and analysis.seller_u_star is not None
        buyer_gain = attainable_gain(result.buyer_utility, analysis.buyer_u_star, r)
        seller_gain = attainable_gain(result.seller_utility, analysis.seller_u_star, r)
        if buyer_gain is not None and seller_gain is not None:
            min_gain = min(buyer_gain, seller_gain)
        # Nash ratio: 0 for no deal when F is non-empty; else N(d)/N_max.
        if analysis.nash_max is not None and analysis.nash_max != 0:
            if valid_agreement and result.accepted_deal_id is not None:
                from .game import DEALS_BY_ID  # local import avoids a module cycle

                d = DEALS_BY_ID[result.accepted_deal_id]
                nash_ratio = nash_product(d, analysis.buyer_rank, analysis.seller_rank, r) / analysis.nash_max
            else:
                nash_ratio = Fraction(0)
        if valid_agreement and result.accepted_deal_id is not None:
            from .game import DEALS_BY_ID

            on_frontier = DEALS_BY_ID[result.accepted_deal_id] in set(analysis.pareto_frontier)

    return EpisodeScore(
        feasible=feasible,
        agreement=result.agreement,
        valid_agreement=valid_agreement,
        correct_no_deal=correct_no_deal,
        buyer_utility=result.buyer_utility,
        seller_utility=result.seller_utility,
        buyer_gain=buyer_gain,
        seller_gain=seller_gain,
        min_gain=min_gain,
        on_pareto_frontier=on_frontier,
        nash_ratio=nash_ratio,
    )


def feasibility_counts(r: Fraction) -> Tuple[int, int]:
    """(# type pairs with a feasible deal, total pairs) under reservation ``r``.

    The spec asserts 36/36 (easy) and 30/36 (hard) for the exact v0 utility
    definition; this function is what the sanity tests check against.
    """
    pairs = all_type_pairs()
    feasible = sum(1 for b, s in pairs if feasible_deals(b, s, r))
    return feasible, len(pairs)
