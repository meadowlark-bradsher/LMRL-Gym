"""Exact dynamic-programming reference for Condition B.

Solves the learner's finite POMDP against the frozen scripted opponent. The
belief is a sufficient statistic (public response model, Bayes updates), the
horizon is ``rounds + 1`` decisions, and the menu is a deterministic function
of (t, belief, asked), so

    V(t, b, asked)   = max_{a in menu} Q(t, b, asked, a)
    Q(.., ASK q)     = sum_o P_b(o) V(t+1, b|o, asked + q)
    Q(.., OFFER d)   = P_b(accept) U_own(d) + P_b(reject) * [ V(t+1, b|reject, asked)  if t < final
                                                             r                          if t = final ]

Everything is a Fraction, so the reference value is exact and the identity
"mean realised utility of the exact arm over a uniform opponent = V(1, uniform)"
holds to equality (checked in ``test_condition_b.py``).

``full_menu=True`` solves the unconstrained game instead (every unasked
question and every one of the 27 deals available at every decision). It is not
a reference for any arm; it bounds what the menu rule costs: V_full >= V_menu.

Regret definitions used by the runner (design doc sections 3 and 4):
    EIG regret       R^IG_t = max_a EIG(a) - EIG(a_t)        (bits)
    decision regret  R^U_t  = max_a Q*(b_t, a) - Q*(b_t, a_t) (utility)
and ``gap`` = best Q minus second-best Q on the menu (for H3).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, Ranking

from . import beliefs
from .env import (
    ACCEPT,
    ASK,
    OFFER,
    REJECT,
    Config,
    Option,
    State,
    initial_state,
    menu,
    option_likelihoods,
    own_utility,
)
from .env import DEALS_BY_ID
from .questions import BANK


class ExactReference:
    def __init__(self, cfg: Config, role: str, own_ranking: Ranking, full_menu: bool = False):
        self.cfg = cfg
        self.role = role
        self.own_ranking = own_ranking
        self.full_menu = full_menu
        self._v: Dict[tuple, Fraction] = {}
        self.states_solved = 0

    # -- menus ---------------------------------------------------------------

    def options(self, state: State) -> List[Option]:
        if not self.full_menu:
            return menu(state, self.cfg)
        if state.t > self.cfg.final_t:
            return []
        opts: List[Option] = []
        if state.t <= self.cfg.rounds:
            opts += [Option(ASK, q.id) for q in BANK if q.id not in state.asked]
        opts += [Option(OFFER, d.id) for d in ALL_DEALS]
        return opts

    # -- values --------------------------------------------------------------

    def q_value(self, state: State, opt: Option) -> Fraction:
        liks = option_likelihoods(opt, state, self.cfg)
        total = Fraction(0)
        for outcome, (p, post) in beliefs.outcome_branches(state.belief, liks).items():
            if opt.kind == OFFER and outcome == ACCEPT:
                total += p * own_utility(state, DEALS_BY_ID[opt.target])
                continue
            if opt.kind == OFFER and state.t >= self.cfg.final_t:
                total += p * self.cfg.reservation
                continue
            asked = state.asked + ((opt.target,) if opt.kind == ASK else ())
            nxt = State(state.role, state.own_ranking, state.t + 1, post, asked)
            total += p * self.value(nxt)
        return total

    def q_values(self, state: State) -> Dict[str, Fraction]:
        return {opt.key: self.q_value(state, opt) for opt in self.options(state)}

    def value(self, state: State) -> Fraction:
        if state.t > self.cfg.final_t:
            return self.cfg.reservation
        key = state.key(self.cfg)
        if key in self._v:
            return self._v[key]
        opts = self.options(state)
        if not opts:  # cannot happen with the bank/menu as defined; defensive
            v = self.cfg.reservation
        else:
            v = max(self.q_value(state, o) for o in opts)
        self._v[key] = v
        self.states_solved += 1
        return v

    def prior_value(self) -> Fraction:
        return self.value(initial_state(self.role, self.own_ranking))

    def best(self, state: State) -> Tuple[Option, Fraction]:
        opts = self.options(state)
        qs = [(self.q_value(state, o), -i, o) for i, o in enumerate(opts)]
        q, _, o = max(qs)
        return o, q


def regrets(qvals: Dict[str, Fraction], chosen_key: str) -> Tuple[Fraction, Fraction]:
    """(decision regret, gap best-minus-second) from a menu's Q values."""
    vals = sorted(qvals.values(), reverse=True)
    best = vals[0]
    second = vals[1] if len(vals) > 1 else vals[0]
    return best - qvals[chosen_key], best - second
