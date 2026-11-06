"""Exact value-of-information oracle for the one-sided elicited-negotiation game.

The state is a belief ``b`` over the 45 buyer types plus the set of questions already
asked. At each node the seller may OFFER now (terminal) or ASK an un-asked question
(paying its token cost, then branching on the buyer's answer under the kernel):

    offer_value(b) = max_d  U_focal(d) * P_b(buyer accepts d)        (>= 0, the no-info floor at the prior)
    value(b, asked) = max( offer_value(b),
                           max over q not in asked:
                               -token_price*cost(q) + sum_a P_b(a) * value(posterior(b,q,a), asked+{q}) )

    V*       = value(uniform prior, {})          -- optimal elicit-then-offer value
    V*_noask = offer_value(uniform prior)        -- never ask
    VOI      = V* - V*_noask >= 0                 -- asking is optional, so this is non-negative

The DP is finite (the asked-set only grows; the bank is small) and memoised on the
rounded belief and the asked-set. ``token_price = 0`` gives the gross value of
inquiry; a positive ``token_price`` the net value once questions are charged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np

from . import game as g

_ANSWERS = (True, False)
_ID2Q = {q.id: i for i, q in enumerate(g.BANK)}


@dataclass
class Solver:
    kernel: g.Kernel = g.TRUTHFUL
    token_price: float = 0.0
    round_dp: int = 9

    def __post_init__(self):
        self._pyes = self.kernel.p_yes()                 # [Q, N_TYPES]
        self._memo: Dict[Tuple[bytes, FrozenSet[int]], float] = {}
        self._best: Dict[Tuple[bytes, FrozenSet[int]], str] = {}

    # -- offering ------------------------------------------------------------------
    def offer_value(self, b: np.ndarray) -> float:
        """Best single-offer expected payoff under belief b."""
        return float((g.U_FOCAL * g.accept_prob(b)).max())

    def best_offer(self, b: np.ndarray) -> int:
        return int(np.argmax(g.U_FOCAL * g.accept_prob(b)))

    # -- value of information ------------------------------------------------------
    def _key(self, b: np.ndarray, asked: FrozenSet[int]):
        return (np.round(b, self.round_dp).tobytes(), asked)

    def value(self, b: np.ndarray, asked: FrozenSet[int] = frozenset()) -> float:
        key = self._key(b, asked)
        if key in self._memo:
            return self._memo[key]
        best_v = self.offer_value(b)
        best_a = "OFFER"
        for q in range(g.N_QUESTIONS):
            if q in asked:
                continue
            qv = -self.token_price * g.BANK[q].cost
            for yes in _ANSWERS:
                lik = g.answer_likelihood(self._pyes, q, yes)
                p, post = g.posterior(b, lik)
                if p > 0:
                    qv += p * self.value(post, asked | {q})
            if qv > best_v + 1e-12:
                best_v, best_a = qv, g.BANK[q].id
        self._memo[key] = best_v
        self._best[key] = best_a
        return best_v

    def best_action(self, b: np.ndarray, asked: FrozenSet[int] = frozenset()) -> str:
        self.value(b, asked)
        return self._best[self._key(b, asked)]

    def action(self, b: np.ndarray, asked: FrozenSet[int] = frozenset()):
        """Structured optimal action: ``("OFFER", deal_idx)`` or ``("ASK", q_idx)``."""
        a = self.best_action(b, asked)
        if a == "OFFER":
            return ("OFFER", self.best_offer(b))
        return ("ASK", _ID2Q[a])

    # -- expected information gain (for the later EIG-vs-value contrast) -----------
    def eig_bits(self, b: np.ndarray, q: int) -> float:
        """Expected reduction in belief entropy from asking q under the kernel."""
        h0 = g.entropy_bits(b)
        exp_h = 0.0
        for yes in _ANSWERS:
            p, post = g.posterior(b, g.answer_likelihood(self._pyes, q, yes))
            if p > 0:
                exp_h += p * g.entropy_bits(post)
        return h0 - exp_h

    def ask_gain(self, b: np.ndarray, q: int, asked: FrozenSet[int] = frozenset()) -> float:
        """Net value of asking q (then playing on optimally) minus offering now --
        the decision-relevant value of the question, to contrast with its EIG."""
        qv = -self.token_price * g.BANK[q].cost
        for yes in _ANSWERS:
            p, post = g.posterior(b, g.answer_likelihood(self._pyes, q, yes))
            if p > 0:
                qv += p * self.value(post, asked | {q})
        return qv - self.offer_value(b)


def summarize(kernel: g.Kernel = g.TRUTHFUL, token_price: float = 0.0) -> dict:
    s = Solver(kernel=kernel, token_price=token_price)
    b0 = g.uniform()
    v_star = s.value(b0)
    v_noask = s.offer_value(b0)
    return {"token_price": token_price, "V_star": v_star, "V_noask": v_noask,
            "VOI": v_star - v_noask, "best_first_action": s.best_action(b0),
            "memo_states": len(s._memo)}
