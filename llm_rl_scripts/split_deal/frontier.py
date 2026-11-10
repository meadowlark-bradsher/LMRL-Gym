"""L0, the frontier oracle of Split Deal (`DESIGN_SPLIT.md` §5).

Per pair of types, by enumeration and in exact arithmetic: the allocations acceptable to both sides
(utility >= d for each), the maximum joint surplus S* over them, the Pareto set among them, and the
Nash point(s) maximising (U_A - d)(U_B - d). An allocation that is acceptable and Pareto-dominated is
dominated by another acceptable one, so the Pareto set here is the Pareto frontier restricted to the
acceptable allocations.

Outcome metrics, for an outcome that is either an allocation (agreement) or None (no deal):

    S(o)  = (U_A - d) + (U_B - d) on agreement, 0 otherwise
    E     = S(o) / S*                  on pairs with S* > 0
    C_A   = (U_A - d) / S(o)           on agreements with S(o) > 0

L0 needs no model of either player, so it can score any arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

from .game import Alloc, Instance, Values


@dataclass(frozen=True)
class PairFrontier:
    acceptable: Tuple[Alloc, ...]          # allocations both sides accept, in allocation order
    s_star: Optional[Fraction]             # None when no allocation is acceptable to both
    pareto: Tuple[Alloc, ...]              # Pareto set among the acceptable allocations
    nash: Tuple[Alloc, ...]                # maximisers of (U_A - d)(U_B - d) among the acceptable


def utilities(inst: Instance, va: Values, vb: Values, a: Alloc) -> Tuple[int, int]:
    return inst.utility(va, a), inst.utility(vb, inst.rest(a))


def pair_frontier(inst: Instance, va: Values, vb: Values, d: Fraction) -> PairFrontier:
    acc = [a for a in inst.allocations() if all(u >= d for u in utilities(inst, va, vb, a))]
    if not acc:
        return PairFrontier((), None, (), ())
    u = {a: utilities(inst, va, vb, a) for a in acc}
    s_star = max(ua + ub - 2 * d for ua, ub in u.values())
    pareto = tuple(a for a in acc
                   if not any(u[b][0] >= u[a][0] and u[b][1] >= u[a][1] and u[b] != u[a] for b in acc))
    best = max((ua - d) * (ub - d) for ua, ub in u.values())
    nash = tuple(a for a in acc if (u[a][0] - d) * (u[a][1] - d) == best)
    return PairFrontier(tuple(acc), s_star, pareto, nash)


def surplus(inst: Instance, va: Values, vb: Values, d: Fraction, outcome: Optional[Alloc]) -> Fraction:
    if outcome is None:
        return Fraction(0)
    ua, ub = utilities(inst, va, vb, outcome)
    return ua + ub - 2 * d


def efficiency(inst: Instance, va: Values, vb: Values, d: Fraction, outcome: Optional[Alloc],
               s_star: Optional[Fraction]) -> Optional[Fraction]:
    """E = S(o) / S*, or None where it is undefined (S* missing or zero)."""
    if not s_star:
        return None
    return surplus(inst, va, vb, d, outcome) / s_star


def share_a(inst: Instance, va: Values, vb: Values, d: Fraction, outcome: Optional[Alloc]) -> Optional[Fraction]:
    """C_A = (U_A - d) / S(o), or None where it is undefined (no deal, or S(o) <= 0)."""
    s = surplus(inst, va, vb, d, outcome)
    if outcome is None or s <= 0:
        return None
    return (inst.utility(va, outcome) - d) / s


def instance_summary(inst: Instance, d: Fraction, scale: Fraction) -> dict:
    """Pair-averaged frontier statistics for one instance and outside option; S* in units of ``scale``."""
    types = inst.types()
    n = with_deal = positive = 0
    s_sum, pareto_sum, nash_sum, nash_share_sum = Fraction(0), 0, 0, Fraction(0)
    for va in types:
        for vb in types:
            n += 1
            f = pair_frontier(inst, va, vb, d)
            if f.s_star is None:
                continue
            with_deal += 1
            s_sum += f.s_star / scale
            pareto_sum += len(f.pareto)
            nash_sum += len(f.nash)
            if f.s_star > 0:
                positive += 1
                nash_share_sum += min(share_a(inst, va, vb, d, a) for a in f.nash)
    return {"instance": inst.key, "n_pairs": n, "pairs_with_deal": with_deal, "pairs_positive": positive,
            "mean_s_star_given_deal": float(s_sum / with_deal) if with_deal else None,
            "mean_pareto_given_deal": pareto_sum / with_deal if with_deal else None,
            "mean_nash_points_given_deal": nash_sum / with_deal if with_deal else None,
            "mean_nash_share_a_given_positive": float(nash_share_sum / positive) if positive else None}


def frontier_table(instances: List[Instance], d: Fraction, scale: Fraction) -> List[dict]:
    return [instance_summary(inst, d, scale) for inst in instances]
