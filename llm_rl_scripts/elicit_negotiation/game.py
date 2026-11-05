"""The one-sided elicited-negotiation game: types, utilities, acceptance, the
question bank, the answer kernels, and belief operations.

Deals reuse ``car_dealer_discrete`` (27 deals, 3 attributes x levels {0,1,2},
normalised to [0,1] by the max level). The buyer's hidden type is theta = (w, rho):
a weight vector ``w`` over the 3 attributes on a discretised simplex (compositions
of 4 into 3 parts -> 15 points) and a reservation ``rho`` in {0.3, 0.5, 0.7}, so the
type space is 15 x 3 = 45 enumerable types.

* Buyer utility  U_cp(theta, d) = w_p (1 - price) + w_w warranty + w_d delivery   in [0,1]
  (the buyer likes LOW price, HIGH warranty/delivery). The buyer ACCEPTS d iff
  U_cp(theta, d) >= rho.
* Seller (focal) utility  U_focal(d) = price (it wants a high price). Payoff of
  offering d = U_focal(d) if the buyer accepts, else 0. High-price deals are worth
  more to the seller but less likely accepted unless sweetened with warranty/delivery,
  and acceptance turns on the buyer's private (w, rho) -- so information lets the seller
  target the offer.

Questions are binary predicates over theta with a per-question TOKEN COST. The answer
kernel gives P(answer | question, theta); ``truthful`` is deterministic, ``noisy``
flips with probability eps, and the interface leaves room for a later strategic-lean
kernel (self-servingly biased answers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, List, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS

# -- deals / features -------------------------------------------------------------
_MAXLVL = float(max(max(d.price, d.warranty, d.delivery) for d in ALL_DEALS)) or 1.0  # = 2


def _features(d) -> Tuple[float, float, float]:
    """(price, warranty, delivery) normalised to [0, 1]."""
    return (d.price / _MAXLVL, d.warranty / _MAXLVL, d.delivery / _MAXLVL)


FEATURES = np.array([_features(d) for d in ALL_DEALS], dtype=np.float64)   # [27, 3]
U_FOCAL = FEATURES[:, 0].copy()                                            # seller wants high price
N_DEALS = len(ALL_DEALS)

# -- buyer type space: (w, rho) ---------------------------------------------------
RESERVATIONS = (0.3, 0.5, 0.7)


def _simplex_weights(total: int = 4) -> List[Tuple[float, float, float]]:
    """Compositions of ``total`` into 3 non-negative parts, divided by ``total``."""
    pts = []
    for a in range(total + 1):
        for b in range(total + 1 - a):
            c = total - a - b
            pts.append((a / total, b / total, c / total))
    return pts


WEIGHTS = _simplex_weights(4)                    # 15 points
TYPES: List[Tuple[Tuple[float, float, float], float]] = [
    (w, rho) for w in WEIGHTS for rho in RESERVATIONS]   # 45 types
N_TYPES = len(TYPES)

_W = np.array([w for (w, _) in TYPES], dtype=np.float64)       # [45, 3]
_RHO = np.array([rho for (_, rho) in TYPES], dtype=np.float64)  # [45]

# buyer utility of every (type, deal): the buyer likes LOW price, so flip the price feature
_BUYER_FEAT = FEATURES.copy()
_BUYER_FEAT[:, 0] = 1.0 - FEATURES[:, 0]
U_CP = _W @ _BUYER_FEAT.T                                       # [45, 27]
ACCEPT = (U_CP >= _RHO[:, None] - 1e-12)                        # [45, 27] bool


# -- questions --------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    id: str
    cost: int                              # token cost, subtracted (x token_price) from payoff
    pred: Callable[[int], bool] = field(compare=False)   # truthful answer for a type index (True = "yes")


def _top(axis: int):
    return lambda t: _W[t, axis] >= _W[t].max() - 1e-12


BANK: List[Question] = [
    Question("top_price", 1, _top(0)),
    Question("top_warranty", 1, _top(1)),
    Question("top_delivery", 1, _top(2)),
    Question("price_ge_warranty", 1, lambda t: _W[t, 0] >= _W[t, 1] - 1e-12),
    Question("warranty_ge_delivery", 1, lambda t: _W[t, 1] >= _W[t, 2] - 1e-12),
    Question("price_sensitive", 2, lambda t: _W[t, 0] >= 0.5 - 1e-12),
    Question("high_reservation", 2, lambda t: _RHO[t] >= 0.7 - 1e-12),
    Question("low_reservation", 2, lambda t: _RHO[t] <= 0.3 + 1e-12),
]
N_QUESTIONS = len(BANK)

# truthful "yes" indicator per question: TRUTH[q, type] in {0, 1}
TRUTH = np.array([[1.0 if q.pred(t) else 0.0 for t in range(N_TYPES)] for q in BANK])  # [Q, 45]


# -- answer kernels ---------------------------------------------------------------
@dataclass(frozen=True)
class Kernel:
    """P(answer | question, theta). ``eps`` is a symmetric flip probability on the
    truthful answer (eps = 0 is truthful). ``lean`` optionally adds a self-serving
    bias toward "yes" on named questions (reserved for a later strategic kernel;
    default none)."""
    eps: float = 0.0
    lean: Tuple[Tuple[str, float], ...] = ()      # (question_id, extra P(yes)) pairs

    def p_yes(self) -> np.ndarray:
        """[Q, N_TYPES]: P(answer = yes | q, theta)."""
        p = (1.0 - self.eps) * TRUTH + self.eps * (1.0 - TRUTH)
        if self.lean:
            bias = dict(self.lean)
            for qi, q in enumerate(BANK):
                if q.id in bias:
                    p[qi] = np.clip(p[qi] + bias[q.id], 0.0, 1.0)
        return p


TRUTHFUL = Kernel(eps=0.0)


def noisy(eps: float) -> Kernel:
    return Kernel(eps=eps)


# -- beliefs ----------------------------------------------------------------------
def uniform() -> np.ndarray:
    return np.full(N_TYPES, 1.0 / N_TYPES)


def entropy_bits(b: np.ndarray) -> float:
    p = b[b > 0]
    return float(-(p * np.log2(p)).sum())


def accept_prob(b: np.ndarray) -> np.ndarray:
    """P_b(buyer accepts d) for every deal d -> [27]."""
    return b @ ACCEPT


def answer_likelihood(kernel_pyes: np.ndarray, q: int, yes: bool) -> np.ndarray:
    """P(answer | q, theta) over types for a fixed answer -> [N_TYPES]."""
    return kernel_pyes[q] if yes else (1.0 - kernel_pyes[q])


def posterior(b: np.ndarray, lik: np.ndarray) -> Tuple[float, np.ndarray]:
    """Bayes update; returns (marginal P(answer), posterior). Posterior is b if the
    answer has zero probability (caller skips those branches)."""
    w = b * lik
    z = float(w.sum())
    if z <= 0:
        return 0.0, b
    return z, w / z
