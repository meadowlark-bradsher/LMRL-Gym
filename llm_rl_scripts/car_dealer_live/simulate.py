"""Monte Carlo episodes of a focal policy against the sampled T1 kernel.

A focal policy is ``policy(state, belief) -> Action``. The focal belief is the
matched filter (``belief_live``) updated only from opponent actions/answers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, List, Optional

from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID, Ranking, utility

from . import belief_live as bl
from .opponent import T1Kernel
from .protocol_live import ASK, LiveConfig, PublicState, Action, apply, asks_by, new_game, other

Policy = Callable[[PublicState, bl.Belief], Action]


@dataclass
class Episode:
    state: PublicState
    focal_gross_gain: float
    focal_net_gain: float
    focal_asks: int
    belief_final: bl.Belief


def run_episode(cfg: LiveConfig, focal: str, focal_ranking: Ranking, opp_ranking: Ranking,
                kernel: T1Kernel, policy: Policy, opener: str, rng: random.Random) -> Episode:
    state = new_game(cfg, opener)
    belief = bl.uniform()
    while state.terminal is None:
        if state.turn == focal:
            act = policy(state, belief)
            if act.kind == ASK:
                ans = kernel.sample_answer(act.question_id, opp_ranking, rng)
                lik = bl.answer_likelihoods(kernel, act.question_id)[ans]
                state = apply(state, act, ans)
                belief = bl.posterior(belief, lik)
            else:
                state = apply(state, act)   # own action: no belief update
        else:
            act = kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(kernel, state)[act]
            state = apply(state, act)
            belief = bl.posterior(belief, lik)
    t = state.terminal
    gross = 0.0 if t.deal_id is None else float(utility(focal, focal_ranking, DEALS_BY_ID[t.deal_id]) - cfg.reservation)
    n_ask = asks_by(state, focal)
    return Episode(state, gross, gross - float(cfg.ask_cost) * n_ask, n_ask, belief)
