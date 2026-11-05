"""ASK-aware evaluation for R2: does a policy *discover* the value-of-information
(ASK) action from reward alone?

The episode loop is ``evaluate.run_episode`` with extra per-decision bookkeeping.
The exact oracle is used ONLY to label evaluation nodes (it never touches R2
training): at each focal node it classifies whether ASK is *legal* (an ASK row is
on the menu), *strictly optimal* (max Q over ASK rows exceeds the best non-ASK row
by > EPS) or *weakly optimal* (within EPS of the best row). Against those labels we
score the policy's ASK behavior.

Headline quantities (all against the exact ``cell.ask`` / ``cell.noask`` oracles):
  * captured-VOI fraction = (mean_net - noask_ceiling) / (ask_ceiling - noask_ceiling):
    0 if the policy captures none of the value of inquiry (never-ask-optimal), 1 if
    it captures all of it (ask-optimal). Negative means it plays even the offers
    below the no-ask ceiling.
  * ASK-optimal recall = asks taken / nodes where ASK is strictly optimal. Jev's
    0/2555 is recall 0; this is the number R2 asks whether reward-trained RL beats.
  * ASK precision = of the policy's asks, the fraction where asking was (weakly)
    optimal; ASK rate = asks / focal decisions and asks / episode.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.oracle import EPS
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, apply, asks_by, new_game
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from .dataset import ROLES, TYPE_IDX
from .evaluate import ChoiceFn


@dataclass
class _EpAsk:
    role: str
    type_idx: int
    opener: str
    net_gain: float
    n_dec: int
    regret_sum: float
    optimal: int           # decisions within EPS of best Q
    asks: int              # ASK decisions taken
    ask_legal: int         # nodes where ASK was on the menu
    ask_strict: int        # nodes where ASK was strictly optimal
    ask_strict_hit: int    # strict-ASK-optimal nodes where the policy asked
    ask_ok: int            # ASK decisions that were (weakly) optimal


def _run_episode_ask(cell: Cell, choose: ChoiceFn, opener: str, opp_ranking, rng: random.Random) -> _EpAsk:
    from .features import rows_matrix, state_features
    from .dataset import context_for
    cfg, focal = cell.cfg, cell.focal
    ctx = context_for(cell)
    state = new_game(cfg, opener)
    belief = bl.uniform()
    n_dec = optimal = asks = ask_legal = ask_strict = ask_strict_hit = ask_ok = 0
    regret_sum = 0.0
    while state.terminal is None:
        if state.turn != focal:
            act = cell.kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(cell.kernel, state)[act]
            state = apply(state, act); belief = bl.posterior(belief, lik)
            continue
        rows = cell.menu.rows(state, belief, include_questions=True)
        sf = state_features(ctx, state, belief)
        rf = rows_matrix(rows)
        idx = choose(cell, state, belief, rows)
        chosen = rows[idx]
        q = {r.key: cell.ask.q_value(state, belief, r.action) for r in rows}
        best_q = max(q.values())
        ask_q = [q[r.key] for r in rows if r.action.kind == ASK]
        nonask_q = [q[r.key] for r in rows if r.action.kind != ASK]
        regret = best_q - q[chosen.key]
        regret_sum += regret
        n_dec += 1
        optimal += int(regret <= EPS)
        if ask_q:
            ask_legal += 1
            strict = max(ask_q) > (max(nonask_q) if nonask_q else -1e18) + EPS
            ask_strict += int(strict)
            if chosen.action.kind == ASK:
                ask_strict_hit += int(strict)
        if chosen.action.kind == ASK:
            asks += 1
            ask_ok += int(regret <= EPS)
        act = chosen.action
        if act.kind == ASK:
            ans = cell.kernel.sample_answer(act.question_id, opp_ranking, rng)
            lik = bl.answer_likelihoods(cell.kernel, act.question_id)[ans]
            state = apply(state, act, ans); belief = bl.posterior(belief, lik)
        else:
            state = apply(state, act)
    t = state.terminal
    deal = DEALS_BY_ID[t.deal_id] if t.deal_id else None
    u = utility(focal, cell.ranking, deal) if deal else cfg.reservation
    net = float(u - cfg.reservation) - float(cfg.ask_cost) * asks_by(state, focal)
    return _EpAsk(focal, ALL_RANKINGS.index(cell.ranking), opener, net, n_dec, regret_sum,
                  optimal, asks, ask_legal, ask_strict, ask_strict_hit, ask_ok)


def evaluate_ask(cells: Dict[Tuple[str, int], Cell], choose: ChoiceFn, openers=ROLES,
                 opp_types=TYPE_IDX, repeats: int = 20, seed: int = 0, sigma: float = 4.0) -> Dict:
    """Score ``choose`` against the ask- and noask-ceilings with ASK diagnostics.
    Episode seeding matches ``evaluate`` so values sit beside R0/R1 and the arms."""
    eps: List[_EpAsk] = []
    ask_ceil: Dict[str, float] = {}
    noask_ceil: Dict[str, float] = {}
    for (role, i), cell in cells.items():
        for opener in openers:
            key = f"{role}|{opener}|{i}"
            ask_ceil[key] = cell.ask.root_value(opener)
            noask_ceil[key] = cell.noask.root_value(opener)
            for j in opp_types:
                for k in range(repeats):
                    rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|{k}")
                    eps.append(_run_episode_ask(cell, choose, opener, ALL_RANKINGS[j], rng))

    def key(e): return f"{e.role}|{e.opener}|{e.type_idx}"
    residuals = np.array([e.net_gain - ask_ceil[key(e)] for e in eps], dtype=np.float64)
    n = len(residuals)
    mean_res = float(residuals.mean())
    se = float(residuals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = mean_res / se if se > 0 else 0.0
    mean_net = float(np.mean([e.net_gain for e in eps]))
    mean_ask_ceil = float(np.mean(list(ask_ceil.values())))
    mean_noask_ceil = float(np.mean(list(noask_ceil.values())))
    voi = mean_ask_ceil - mean_noask_ceil
    captured = (mean_net - mean_noask_ceil) / voi if abs(voi) > 1e-12 else float("nan")

    n_dec = sum(e.n_dec for e in eps)
    asks = sum(e.asks for e in eps)
    ask_strict = sum(e.ask_strict for e in eps)
    ask_strict_hit = sum(e.ask_strict_hit for e in eps)
    ask_ok = sum(e.ask_ok for e in eps)
    return {
        "n_episodes": n, "repeats": repeats, "sigma": sigma,
        "mean_net": mean_net, "ask_ceiling": mean_ask_ceil, "noask_ceiling": mean_noask_ceil,
        "voi": voi, "captured_voi": captured,
        "mean_residual": mean_res, "residual_se": se, "residual_z": z,
        "within_sigma_se": abs(z) <= sigma,
        "above_noask_ceiling": bool(mean_net > mean_noask_ceil),
        "optimal_match": sum(e.optimal for e in eps) / n_dec if n_dec else 0.0,
        "mean_decision_regret": sum(e.regret_sum for e in eps) / n_dec if n_dec else 0.0,
        "ask_rate_per_decision": asks / n_dec if n_dec else 0.0,
        "ask_rate_per_episode": asks / n if n else 0.0,
        "ask_optimal_recall": ask_strict_hit / ask_strict if ask_strict else float("nan"),
        "ask_precision": ask_ok / asks if asks else float("nan"),
        "n_decisions": n_dec, "n_asks": asks, "n_ask_strict_optimal": ask_strict,
    }
