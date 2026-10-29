"""Scripted focal arms for Phase 1. Each arm is ``arm(rows, q_ask, q_noask, rng) -> Action``
where ``rows`` is the menu the arm sees (``menu_live.Row``), ``q_ask`` the exact Q* of
every ask-menu option under the menu-restricted oracle with ASK allowed (C_plan),
and ``q_noask`` the exact Q of every non-ASK option under the same oracle with ASK
forbidden (B_plan).

* ``exact_ask``   C_plan: argmax q_ask; ties -> a non-ASK option, then menu order.
* ``exact_noask`` B_plan: argmax q_noask over the non-ASK options (never asks).
* ``greedy_eig``  argmax EIG (offers included: an offer is informative too); ties by
                  immediate value (ACCEPT: U - r; PROPOSE: P(accept) * (U - r); else 0),
                  then menu order. WALK is never chosen by a heuristic arm.
* ``random``      uniform over the menu minus WALK (WALK only if nothing else is legal).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List

from .menu_live import Row
from .protocol_live import ACCEPT, ASK, PROPOSE, WALK, Action

EPS = 1e-9
Arm = Callable[[List[Row], Dict[str, float], Dict[str, float], random.Random], Action]


def _argmax(rows: List[Row], score) -> Action:
    best = None
    for i, r in enumerate(rows):
        k = score(i, r)
        if best is None or k > best[0]:
            best = (k, r.action)
    return best[1]


def exact_ask(rows, q_ask, q_noask, rng) -> Action:
    top = max(q_ask[r.key] for r in rows)
    return _argmax(rows, lambda i, r: (q_ask[r.key] >= top - EPS, r.action.kind != ASK, -i))


def exact_noask(rows, q_ask, q_noask, rng) -> Action:
    cand = [r for r in rows if r.action.kind != ASK]
    top = max(q_noask[r.key] for r in cand)
    return _argmax(cand, lambda i, r: (q_noask[r.key] >= top - EPS, -i))


def _immediate(r: Row, reservation: float) -> float:
    if r.action.kind == ACCEPT:
        return float(r.own_utility) - reservation
    if r.action.kind == PROPOSE:
        return (r.p_accept or 0.0) * (float(r.own_utility) - reservation)
    return 0.0


def make_greedy_eig(reservation: float) -> Arm:
    def greedy_eig(rows, q_ask, q_noask, rng) -> Action:
        cand = [r for r in rows if r.action.kind != WALK] or rows
        return _argmax(cand, lambda i, r: (round(r.eig_bits, 9), _immediate(r, reservation), -i))
    return greedy_eig


def random_arm(rows, q_ask, q_noask, rng) -> Action:
    cand = [r for r in rows if r.action.kind != WALK] or rows
    return rng.choice(cand).action


def arms_for(reservation: float) -> Dict[str, Arm]:
    return {"exact_ask": exact_ask, "exact_noask": exact_noask, "greedy_eig": make_greedy_eig(reservation), "random": random_arm}
