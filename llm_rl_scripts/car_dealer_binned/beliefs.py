"""Exact beliefs over the opponent's six rankings, entropy, and expected
information gain. Probabilities are Fractions; entropies are floats in bits.

A belief is a tuple of six Fractions aligned with ``game.ALL_RANKINGS``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, Mapping, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, Ranking

Belief = Tuple[Fraction, ...]
Likelihood = Tuple[Fraction, ...]  # P(outcome | theta) aligned with ALL_RANKINGS

N_TYPES = len(ALL_RANKINGS)


def uniform() -> Belief:
    return tuple(Fraction(1, N_TYPES) for _ in ALL_RANKINGS)


def point(ranking: Ranking) -> Belief:
    return tuple(Fraction(1) if r == ranking else Fraction(0) for r in ALL_RANKINGS)


def entropy_bits(b: Belief) -> float:
    return float(-sum((float(p) * math.log2(float(p))) for p in b if p > 0))


def support(b: Belief) -> Tuple[Ranking, ...]:
    return tuple(r for r, p in zip(ALL_RANKINGS, b) if p > 0)


def marginal(b: Belief, lik: Likelihood) -> Fraction:
    """P(outcome) = sum_theta b(theta) P(outcome | theta)."""
    return sum((p * l for p, l in zip(b, lik)), Fraction(0))


def update(b: Belief, lik: Likelihood) -> Belief:
    """Exact Bayes update; raises if the outcome has zero probability."""
    z = marginal(b, lik)
    if z == 0:
        raise ValueError("outcome has probability zero under the current belief")
    return tuple((p * l) / z for p, l in zip(b, lik))


def outcome_branches(
    b: Belief, liks: Mapping[str, Likelihood]
) -> Dict[str, Tuple[Fraction, Belief]]:
    """{outcome: (P(outcome), posterior)} over outcomes with positive probability."""
    out: Dict[str, Tuple[Fraction, Belief]] = {}
    for o, lik in liks.items():
        z = marginal(b, lik)
        if z > 0:
            out[o] = (z, tuple((p * l) / z for p, l in zip(b, lik)))
    return out


def expected_information_gain(b: Belief, liks: Mapping[str, Likelihood]) -> float:
    """EIG = H(b) - E_o[ H(b | o) ] in bits, exact up to float log2."""
    h0 = entropy_bits(b)
    exp_h = 0.0
    for z, post in outcome_branches(b, liks).values():
        exp_h += float(z) * entropy_bits(post)
    gain = h0 - exp_h
    # guard tiny negative float noise
    return 0.0 if abs(gain) < 1e-12 else gain
