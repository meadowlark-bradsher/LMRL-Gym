"""Matched finite filter for the focal negotiator: a posterior over the opponent's
six rankings, updated ONLY from the opponent's observed actions and answers,
using the kernel's own probabilities (Phase 0 = correctly specified model).

The focal party's own emitted actions never update the belief (design document
section 4): information arrives through the opponent's response.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, Ranking

from .opponent import T1Kernel
from .protocol_live import Action, PublicState

Belief = Tuple[float, ...]
N = len(ALL_RANKINGS)


def uniform() -> Belief:
    return tuple(1.0 / N for _ in ALL_RANKINGS)


def entropy_bits(b: Belief) -> float:
    return -sum(p * math.log2(p) for p in b if p > 1e-15)


def support(b: Belief, eps: float = 1e-12) -> Tuple[Ranking, ...]:
    return tuple(r for r, p in zip(ALL_RANKINGS, b) if p > eps)


def round_belief(b: Belief, nd: int = 10) -> Belief:
    return tuple(round(p, nd) for p in b)


def action_likelihoods(kernel: T1Kernel, state: PublicState) -> Dict[Action, Tuple[float, ...]]:
    """{opponent action: (P(action | theta) for theta in ALL_RANKINGS)} at an opponent node."""
    per_theta = [kernel.distribution(state, th) for th in ALL_RANKINGS]
    actions = set()
    for d in per_theta:
        actions.update(d)
    return {a: tuple(d.get(a, 0.0) for d in per_theta) for a in actions}


def answer_likelihoods(kernel: T1Kernel, question_id: str) -> Dict[str, Tuple[float, ...]]:
    per_theta = [kernel.answer_distribution(question_id, th) for th in ALL_RANKINGS]
    return {ans: tuple(d.get(ans, 0.0) for d in per_theta) for ans in ("yes", "no")}


def marginal(b: Belief, lik: Tuple[float, ...]) -> float:
    return sum(p * l for p, l in zip(b, lik))


def posterior(b: Belief, lik: Tuple[float, ...]) -> Belief:
    z = marginal(b, lik)
    if z <= 0:
        raise ValueError("zero-probability observation")
    return tuple(p * l / z for p, l in zip(b, lik))


def branches(b: Belief, liks: Dict) -> Dict:
    """{outcome: (P(outcome), posterior)} for outcomes with positive probability."""
    out = {}
    for o, lik in liks.items():
        z = marginal(b, lik)
        if z > 1e-15:
            out[o] = (z, tuple(p * l / z for p, l in zip(b, lik)))
    return out


def eig_bits(b: Belief, liks: Dict) -> float:
    h0 = entropy_bits(b)
    e = sum(z * entropy_bits(post) for z, post in branches(b, liks).values())
    g = h0 - e
    return 0.0 if abs(g) < 1e-12 else g
