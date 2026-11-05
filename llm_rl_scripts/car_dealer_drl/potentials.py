"""R4 shaping potentials: a ladder from the exact oracle value down to an
oracle-free heuristic, all plugged into ``env.rollout``'s ``potential`` hook.

Under the one-step (nstep1) estimator the potential enters the bootstrap, so its
quality shows up directly: Phi = V*_exact gives an unbiased advantage (R1 recovers
V* exactly), while a weaker potential trades that for being cheap or oracle-free.
R4 holds the estimator fixed and varies only Phi to measure how much of R1's
recovery survives as the oracle degrades.

* ``phi_exact``     -- Phi = V*_assumed (the R1 reference; the exact menu oracle value).
* ``phi_mismatch``  -- Phi = V* under a WRONG kernel (right functional form, wrong
                       parameters): a biased but still value-shaped potential.
* ``phi_entropy``   -- Phi = normalized information gained = 1 - H(belief)/log2(N):
                       NO oracle, just the posterior. The live analogue of the
                       deepink log-survivor-ratio potential (increasing form, so the
                       standard PBRS increment gamma*Phi(s')-Phi(s) rewards info gain).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Dict, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.oracle import Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import other
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

_LOG2_N = math.log2(len(ALL_RANKINGS))   # log2(6)


def phi_exact(cell: Cell, state, belief) -> float:
    """Phi = V*_assumed: the exact menu-oracle value (the R1 reference potential)."""
    return float(cell.ask.value(state, belief))


def phi_noask(cell: Cell, state, belief) -> float:
    """Phi = V*_noask: the exact value of the ASK-FORBIDDEN oracle -- a genuinely
    weaker value potential that does not credit information (it differs from V*_ask
    exactly by the value of inquiry). Under nstep1 it bootstraps as if asking has no
    future value, biasing the actor AGAINST asking."""
    return float(cell.noask.value(state, belief))


def make_phi_mismatch(cells: Dict[Tuple[str, int], Cell], wrong: KernelParams) -> Callable:
    """Phi = V* under a mismatched kernel (same T1 form, ``wrong`` parameters), over
    each cell's own menu. The environment opponent is still the registered kernel;
    only the potential is computed from the wrong model."""
    oracles: Dict[Tuple[str, int], Oracle] = {}
    for (role, i), cell in cells.items():
        k = T1Kernel(other(role), cell.cfg, wrong)
        oracles[(role, i)] = Oracle(cell.cfg, role, ALL_RANKINGS[i], k, ask_allowed=True,
                                    menu=cell.menu.actions(True))

    def phi(cell: Cell, state, belief) -> float:
        return float(oracles[(cell.focal, ALL_RANKINGS.index(cell.ranking))].value(state, belief))
    phi.oracles = oracles   # keep refs alive / inspectable
    return phi


class TruncatedOracle(Oracle):
    """A cheap, depth-capped approximation of V*: exact DP, but only ``max_depth``
    focal decisions of lookahead; beyond the frontier the game is assumed to end
    with no deal (leaf value ``leaf``, default 0 = the reservation baseline). Each
    opponent->focal transition consumes one unit of depth. This is the R4 'weaker
    oracle' for the H=6 regime, where the full DP is expensive (~98s/cell) but a
    few-ply truncation costs a fraction of that."""

    def __init__(self, cfg, focal, ranking, kernel, menu, max_depth: int, leaf: float = 0.0):
        super().__init__(cfg, focal, ranking, kernel, ask_allowed=True, menu=menu)
        self.max_depth = max_depth
        self.leaf = leaf
        self._vt: Dict[tuple, float] = {}

    def value_trunc(self, state, belief, depth: int) -> float:
        if state.terminal is not None:
            return self.terminal_gain(state)
        if depth <= 0:
            return self.leaf
        key = (state.key(), bl.round_belief(belief), depth)
        if key in self._vt:
            return self._vt[key]
        if state.turn == self.focal:
            v = max(self._q(state, belief, a, depth) for a in self.focal_actions(state, belief))
        else:
            from llm_rl_scripts.car_dealer_live.protocol_live import apply
            liks = bl.action_likelihoods(self.kernel, state)
            v = sum(p * self.value_trunc(apply(state, act), post, depth - 1)
                    for act, (p, post) in bl.branches(belief, liks).items())
        self._vt[key] = v
        return v

    def _q(self, state, belief, action, depth: int) -> float:
        from llm_rl_scripts.car_dealer_live.protocol_live import ASK, apply
        if action.kind == ASK:
            liks = bl.answer_likelihoods(self.kernel, action.question_id)
            return sum(p * self.value_trunc(apply(state, action, ans), post, depth)
                       for ans, (p, post) in bl.branches(belief, liks).items()) - float(self.cfg.ask_cost)
        return self.value_trunc(apply(state, action), belief, depth)


def make_phi_approx(cells: Dict[Tuple[str, int], Cell], max_depth: int, leaf: float = 0.0) -> Callable:
    """Phi = depth-capped approximate V* (``TruncatedOracle``), over each cell's menu.
    A cheap stand-in for the exact oracle where the full DP is too expensive."""
    oracles: Dict[Tuple[str, int], TruncatedOracle] = {}
    for (role, i), cell in cells.items():
        oracles[(role, i)] = TruncatedOracle(cell.cfg, role, ALL_RANKINGS[i], cell.kernel,
                                             cell.menu.actions(True), max_depth=max_depth, leaf=leaf)

    def phi(cell: Cell, state, belief) -> float:
        o = oracles[(cell.focal, ALL_RANKINGS.index(cell.ranking))]
        return float(o.value_trunc(state, belief, o.max_depth))
    phi.oracles = oracles
    return phi


def phi_entropy(scale: float) -> Callable:
    """Phi = scale * (1 - H(belief)/log2(N)): information gained so far, in [0, scale].
    Oracle-free -- the deepink entropy potential. ``scale`` sets the shaping strength
    (PBRS leaves the optimal policy invariant for any scale); pick it near the value
    range so the increment is comparable to the reward."""
    def phi(cell: Cell, state, belief) -> float:
        return scale * (1.0 - bl.entropy_bits(belief) / _LOG2_N)
    return phi
