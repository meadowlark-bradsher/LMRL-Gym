"""Scripted arms for Condition B. Every arm sees the same annotated menu.

An arm is a callable ``(state, rows, qvals, rng) -> option_key`` where
``rows`` is ``env.annotate(state)`` (public EIG / P(accept) / own utility per
option, in menu order) and ``qvals`` the exact reference's Q values (only the
``exact`` arm is allowed to look at them; the runner passes them to all arms
for uniformity and the tests assert the others ignore them).

  random      uniform over the menu
  greedy_eig  argmax EIG (ties -> menu order). Note offers carry EIG too
              (a rejection eliminates every type that would have accepted),
              so pure greedy-EIG is defined on offers-only menus as well.
  myopic      never asks: the offer with the highest myopic value
              m(d) = P(accept) U_own + (1 - P(accept)) r  (ties -> menu order)
  exact       argmax Q* (ties -> OFFER, then menu order)
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Callable, Dict, List

from .env import OFFER, State

Arm = Callable[[State, List[dict], Dict[str, Fraction], random.Random], str]


def random_arm(state, rows, qvals, rng) -> str:
    return rng.choice(rows)["key"]


def greedy_eig_arm(state, rows, qvals, rng) -> str:
    best = max(rows, key=lambda r: (r["eig_bits"], -rows.index(r)))
    return best["key"]


def myopic_arm(state, rows, qvals, rng) -> str:
    offers = [r for r in rows if r["kind"] == OFFER]
    best = max(offers, key=lambda r: (r["myopic_value"], -rows.index(r)))
    return best["key"]


def exact_arm(state, rows, qvals, rng) -> str:
    """argmax Q*; ties broken toward OFFER (act when information has no value),
    then menu order. Ties carry zero regret either way; the convention only
    affects descriptive counts such as questions asked."""
    best_q = max(qvals[r["key"]] for r in rows)
    tied = [r for r in rows if qvals[r["key"]] == best_q]
    offers = [r for r in tied if r["kind"] == OFFER]
    return (offers or tied)[0]["key"]


ARMS: Dict[str, Arm] = {
    "random": random_arm,
    "greedy_eig": greedy_eig_arm,
    "myopic": myopic_arm,
    "exact": exact_arm,
}
