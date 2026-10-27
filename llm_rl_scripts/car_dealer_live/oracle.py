"""Exact finite-horizon oracle for the focal negotiator against the T1 kernel.

Objective (design document section 4): expected NET gain
    (U_focal(d) - r) * 1[deal] - ask_cost * (#focal asks)
with the belief as the state's sufficient statistic (matched filter). Focal
decision nodes maximise; opponent nodes take the expectation under the kernel
marginalised over the current belief; ASK branches on the answer distribution.

``ask_allowed=False`` removes ASK from the focal's action set at EVERY future
decision (the "ASK forbidden" value V*_noask). The difference
    Delta*_active = V*_ask - V*_noask >= 0
is the global value of inquiry; it is a structural fact that it is >= 0, the
empirical question is its size and where it is realised.

Also records, over all focal decision nodes solved, how many have a strictly
optimal ASK (max_Q over ASK > max_Q over non-ASK) — Phase 0 gate 7.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID, Ranking, utility

from . import belief_live as bl
from .opponent import T1Kernel
from .protocol_live import ACCEPT, ASK, LiveConfig, PublicState, Action, apply, legal_actions, new_game, other

EPS = 1e-9


class Oracle:
    def __init__(self, cfg: LiveConfig, focal: str, focal_ranking: Ranking, kernel: T1Kernel, ask_allowed: bool = True):
        assert kernel.role == other(focal)
        self.cfg = cfg
        self.focal = focal
        self.focal_ranking = focal_ranking
        self.kernel = kernel
        self.ask_allowed = ask_allowed
        self._v: Dict[tuple, float] = {}
        self.focal_nodes = 0
        self.ask_legal_nodes = 0
        self.ask_optimal_nodes = 0
        self.ask_tied_nodes = 0

    # -- rewards ------------------------------------------------------------------

    def terminal_gain(self, state: PublicState) -> float:
        t = state.terminal
        if t.deal_id is None:
            return 0.0
        return float(utility(self.focal, self.focal_ranking, DEALS_BY_ID[t.deal_id]) - self.cfg.reservation)

    # -- focal action set -----------------------------------------------------------

    def focal_actions(self, state: PublicState) -> List[Action]:
        acts = legal_actions(state, self.focal)
        if not self.ask_allowed:
            acts = [a for a in acts if a.kind != ASK]
        return acts

    # -- values -----------------------------------------------------------------------

    def q_value(self, state: PublicState, belief: bl.Belief, action: Action) -> float:
        if action.kind == ASK:
            liks = bl.answer_likelihoods(self.kernel, action.question_id)
            total = 0.0
            for ans, (p, post) in bl.branches(belief, liks).items():
                total += p * self.value(apply(state, action, ans), post)
            return total - float(self.cfg.ask_cost)
        return self.value(apply(state, action), belief)

    def q_values(self, state: PublicState, belief: bl.Belief) -> Dict[Action, float]:
        assert state.turn == self.focal and state.terminal is None
        return {a: self.q_value(state, belief, a) for a in self.focal_actions(state)}

    def value(self, state: PublicState, belief: bl.Belief) -> float:
        if state.terminal is not None:
            return self.terminal_gain(state)
        key = (state.key(), bl.round_belief(belief))
        if key in self._v:
            return self._v[key]
        if state.turn == self.focal:
            qs = self.q_values(state, belief)
            v = max(qs.values())
            self.focal_nodes += 1
            ask_qs = [q for a, q in qs.items() if a.kind == ASK]
            if ask_qs:
                self.ask_legal_nodes += 1
                best_non_ask = max(q for a, q in qs.items() if a.kind != ASK)
                if max(ask_qs) > best_non_ask + EPS:
                    self.ask_optimal_nodes += 1
                elif abs(max(ask_qs) - best_non_ask) <= EPS:
                    self.ask_tied_nodes += 1
        else:
            liks = bl.action_likelihoods(self.kernel, state)
            v = 0.0
            for act, (p, post) in bl.branches(belief, liks).items():
                v += p * self.value(apply(state, act), post)
        self._v[key] = v
        return v

    def root_value(self, opener: str) -> float:
        return self.value(new_game(self.cfg, opener), bl.uniform())

    def best(self, state: PublicState, belief: bl.Belief) -> Tuple[Action, float]:
        qs = self.q_values(state, belief)
        acts = self.focal_actions(state)
        return max(((qs[a], -i, a) for i, a in enumerate(acts)))[2], max(qs.values())

    def stats(self) -> dict:
        return {"focal_nodes": self.focal_nodes, "ask_legal_nodes": self.ask_legal_nodes,
                "ask_optimal_nodes": self.ask_optimal_nodes, "ask_tied_nodes": self.ask_tied_nodes,
                "memo_states": len(self._v)}
