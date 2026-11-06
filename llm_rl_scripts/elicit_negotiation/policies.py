"""Acquisition policies for the elicited-negotiation game.

A policy maps the seller's current belief (and the set of questions already asked)
to a structured action -- ``("ASK", q_idx)`` or ``("OFFER", deal_idx)``. Every policy
also exposes the answer kernel it *assumes* when updating its belief; the evaluator
may run it against a *different* kernel, which is how answer-kernel mismatch is
represented (see ``evaluate.py``).

Two policies:

* ``OraclePolicy`` -- the value-optimal elicit-then-offer DP (``oracle.Solver``). It
  reasons about the token price, so it asks only when a question's decision value
  exceeds its cost. Under a matched kernel it is optimal over *all* policies.
* ``EIGGreedy`` -- the information-greedy baseline. At each step it asks the un-asked
  question of maximal expected information gain (bits), stopping (and offering the
  belief-optimal deal) once the best available EIG falls below a threshold ``tau``.
  It chooses *what* to ask by EIG alone -- it never looks at decision value or token
  cost -- so it is exactly the "information gain as the acquisition signal" strawman
  the calibration predicted would ask high-EIG, decision-irrelevant questions.
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

import numpy as np

from . import game as g
from . import oracle as o


class OraclePolicy:
    def __init__(self, assumed_kernel: g.Kernel = g.TRUTHFUL, token_price: float = 0.0):
        self.assumed_kernel = assumed_kernel
        self.token_price = token_price
        self.solver = o.Solver(kernel=assumed_kernel, token_price=token_price)

    def assumed_pyes(self) -> np.ndarray:
        return self.solver._pyes

    def action(self, b: np.ndarray, asked: FrozenSet[int] = frozenset()) -> Tuple[str, int]:
        return self.solver.action(b, asked)


class EIGGreedy:
    """Ask the max-EIG un-asked question while that EIG >= ``tau`` bits, else offer.

    ``tau`` is a bits threshold standing in for the agent's willingness to keep
    asking; sweeping it traces the baseline's whole frontier. The policy is
    cost-blind by construction -- it ranks questions by EIG, not EIG-per-token and
    not decision value."""

    def __init__(self, tau_bits: float, assumed_kernel: g.Kernel = g.TRUTHFUL):
        self.assumed_kernel = assumed_kernel
        self.tau = float(tau_bits)
        self._pyes = assumed_kernel.p_yes()

    def assumed_pyes(self) -> np.ndarray:
        return self._pyes

    def _eig(self, b: np.ndarray, q: int) -> float:
        h0 = g.entropy_bits(b)
        exp_h = 0.0
        for yes in (True, False):
            p, post = g.posterior(b, g.answer_likelihood(self._pyes, q, yes))
            if p > 0:
                exp_h += p * g.entropy_bits(post)
        return h0 - exp_h

    def action(self, b: np.ndarray, asked: FrozenSet[int] = frozenset()) -> Tuple[str, int]:
        best_q, best_eig = None, -1.0
        for q in range(g.N_QUESTIONS):
            if q in asked:
                continue
            e = self._eig(b, q)
            if e > best_eig + 1e-12:
                best_eig, best_q = e, q
        if best_q is None or best_eig < self.tau:
            return ("OFFER", int(np.argmax(g.U_FOCAL * g.accept_prob(b))))
        return ("ASK", best_q)
