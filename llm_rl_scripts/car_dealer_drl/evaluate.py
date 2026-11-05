"""Evaluate a focal policy against the registered T1 kernel and score it against
the exact ``exact_ask`` ceiling.

The episode loop mirrors ``run_phase1.play_episode`` (same kernel sampling, same
belief filter, same seed formula) so the numbers sit beside the scripted arms.
For every (role, opener, focal type) the exact ceiling is ``cell.ask.root_value``
(= ``ceilings.json``'s ``menu_ask``); averaging episodes over opponent types
(uniform prior) and repeats estimates it. The primary gate is the residual
``net - ceiling`` pooled over all episodes: an optimal policy sits within 4 SE of
zero. Per-decision optimal-match and regret (against the same oracle) are reported
as diagnostics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.oracle import EPS
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, apply, asks_by, new_game
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from .dataset import ROLES, TYPE_IDX, context_for
from .features import rows_matrix, state_features
from .net import PolicyValueNet

# A focal policy here is choose(cell, state, belief, rows) -> chosen Row index.
ChoiceFn = Callable[[Cell, object, bl.Belief, list], int]


def oracle_choice(cell: Cell, state, belief, rows) -> int:
    """Shim policy: pick exactly what ``exact_ask`` picks (validates the harness)."""
    best_action = cell.ask.best(state, belief)[0]
    return next(i for i, r in enumerate(rows) if r.key == best_action.key)


def net_choice_fn(net: PolicyValueNet) -> ChoiceFn:
    def choose(cell: Cell, state, belief, rows) -> int:
        ctx = context_for(cell)
        sf = state_features(ctx, state, belief)
        rf = rows_matrix(rows)
        mask = np.ones(len(rows), dtype=bool)
        return net.choose_index(sf, rf, mask)
    return choose


@dataclass
class EpisodeResult:
    role: str
    type_idx: int
    opener: str
    net_gain: float
    regret_sum: float
    n_decisions: int
    optimal_decisions: int


def run_episode(cell: Cell, choose: ChoiceFn, opener: str, opp_ranking, rng: random.Random) -> EpisodeResult:
    cfg, focal = cell.cfg, cell.focal
    state = new_game(cfg, opener)
    belief = bl.uniform()
    regret_sum = 0.0
    n_dec = optimal = 0
    while state.terminal is None:
        if state.turn != focal:
            act = cell.kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(cell.kernel, state)[act]
            state = apply(state, act)
            belief = bl.posterior(belief, lik)
            continue
        rows = cell.menu.rows(state, belief, include_questions=True)
        idx = choose(cell, state, belief, rows)
        chosen = rows[idx]
        q = {r.key: cell.ask.q_value(state, belief, r.action) for r in rows}
        best_q = max(q.values())
        regret = best_q - q[chosen.key]
        regret_sum += regret
        n_dec += 1
        optimal += int(regret <= EPS)
        act = chosen.action
        if act.kind == ASK:
            ans = cell.kernel.sample_answer(act.question_id, opp_ranking, rng)
            lik = bl.answer_likelihoods(cell.kernel, act.question_id)[ans]
            state = apply(state, act, ans)
            belief = bl.posterior(belief, lik)
        else:
            state = apply(state, act)
    t = state.terminal
    r = cfg.reservation
    deal = DEALS_BY_ID[t.deal_id] if t.deal_id else None
    u = utility(focal, cell.ranking, deal) if deal else r
    net = float(u - r) - float(cfg.ask_cost) * asks_by(state, focal)
    return EpisodeResult(focal, ALL_RANKINGS.index(cell.ranking), opener, net, regret_sum, n_dec, optimal)


def evaluate(cells: Dict[Tuple[str, int], Cell], choose: ChoiceFn, openers=ROLES, opp_types=TYPE_IDX,
             repeats: int = 20, seed: int = 0, sigma: float = 4.0) -> Dict:
    """Run every (role, opener, focal type) x opponent type x repeat, compare to the
    exact ceiling. ``sigma`` is the SE tolerance of the gate (repo convention: 4)."""
    episodes: List[EpisodeResult] = []
    ceilings: Dict[str, float] = {}
    for (role, i), cell in cells.items():
        for opener in openers:
            ceilings[f"{role}|{opener}|{i}"] = cell.ask.root_value(opener)
            for j in opp_types:
                for k in range(repeats):
                    rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|{k}")
                    episodes.append(run_episode(cell, choose, opener, ALL_RANKINGS[j], rng))

    residuals = np.array([e.net_gain - ceilings[f"{e.role}|{e.opener}|{e.type_idx}"] for e in episodes], dtype=np.float64)
    n = len(residuals)
    mean_res = float(residuals.mean())
    se = float(residuals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = mean_res / se if se > 0 else 0.0
    n_dec = sum(e.n_decisions for e in episodes)
    opt_dec = sum(e.optimal_decisions for e in episodes)

    # per (role, opener, type) means, for the note
    cells_report: Dict[str, Dict] = {}
    for key, c in ceilings.items():
        grp = [e for e in episodes if f"{e.role}|{e.opener}|{e.type_idx}" == key]
        m = float(np.mean([e.net_gain for e in grp]))
        cells_report[key] = {"ceiling": c, "mean_net": m, "gap": m - c, "n": len(grp)}

    return {
        "n_episodes": n, "repeats": repeats, "opp_types": list(opp_types), "sigma": sigma,
        "mean_net": float(np.mean([e.net_gain for e in episodes])),
        "mean_ceiling": float(np.mean(list(ceilings.values()))),
        "mean_residual": mean_res, "residual_se": se, "residual_z": z,
        "within_sigma_se": abs(z) <= sigma,
        "optimal_match": opt_dec / n_dec if n_dec else 0.0,
        "mean_decision_regret": sum(e.regret_sum for e in episodes) / n_dec if n_dec else 0.0,
        "n_decisions": n_dec, "cells": cells_report,
    }
