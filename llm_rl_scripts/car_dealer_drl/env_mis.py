"""R3 belief-MDP rollout under opponent misspecification (CD-MIS population).

The one structural change from ``env.py``: the opponent that *acts* (and answers,
and determines reward) is a **true** kernel drawn from the CD-MIS population
(``opponents_mis.true_kernel``), while everything the agent plans and observes with
-- the shortlist menu, the features, the shaping potential, the belief filter -- is
built on the agent's **assumed** kernel (the registered T1). The agent never sees
the process; it must generalise through the (misspecified) belief alone.

Because a true opponent takes actions the assumed kernel can give ~0 probability
(an early WALK, a threshold accept), the belief update is the robustified
``filters_mis`` filter (``B_naive``: assumed-kernel Bayes with an eta=1e-3
background floor), not the raw ``belief_live.posterior``. The same filter and menu
feed the O_assumed baseline, so the learned actor and the misspecified planner are
compared under identical observation machinery.

Reward is the TRUE net gain. The shaping potential is ``Phi = V*_assumed`` (the
assumed-kernel oracle value at the assumed belief): opponent-independent, so it
leaks nothing about the truth and -- being a function of the agent's state -- keeps
the optimal policy of the true-reward MDP invariant (PBRS). With the MC
return-to-go estimator the learned value baseline cancels the state-dependent
potential, so the policy-gradient advantage is the *true* advantage (unbiased):
the shaping only reduces variance, it does not bias the learner toward the assumed
optimum. That is what keeps the robustness test honest.
"""

from __future__ import annotations

import random
from typing import Optional

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live import filters_mis as fm
from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, apply, asks_by, new_game
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from llm_rl_scripts.car_dealer_live.oracle import Oracle

from .dataset import context_for
from .env import ChooseFn, Episode, Step
from .features import rows_matrix, state_features

DEFAULT_FILTER = fm.FILTERS["B_naive"]   # assumed-kernel Bayes with an eta=1e-3 background floor


def rollout_mis(assumed_cell: Cell, true_kernel: T1Kernel, opener: str, opp_ranking,
                rng: random.Random, choose: ChooseFn, shaping: bool = True,
                spec: fm.FilterSpec = DEFAULT_FILTER) -> Episode:
    """One episode: the agent plans/observes with ``assumed_cell`` (registered T1,
    shortlist menu, robustified belief ``spec``) while ``true_kernel`` acts, answers
    and sets the reward. Phi = V*_assumed at the assumed belief (opponent-independent)."""
    cfg, focal = assumed_cell.cfg, assumed_cell.focal
    assumed = assumed_cell.kernel
    ctx = context_for(assumed_cell)
    state = new_game(cfg, opener)
    belief = bl.uniform()
    steps = []

    while state.terminal is None:
        if state.turn != focal:                                    # TRUE opponent acts
            act = true_kernel.sample(state, opp_ranking, rng)
            lik = fm.opponent_action_likelihood(assumed, state, act, spec)   # assumed-model likelihood
            state = apply(state, act)
            belief = fm.update(belief, lik, spec)
            continue
        sf = state_features(ctx, state, belief)
        rows = assumed_cell.menu.rows(state, belief, include_questions=True)
        rf = rows_matrix(rows)
        phi = float(assumed_cell.ask.value(state, belief)) if shaping else 0.0
        idx = choose(assumed_cell, state, belief, rows, sf, rf)
        chosen = rows[idx]
        env_r = 0.0
        act = chosen.action
        if act.kind == ASK:
            env_r -= float(cfg.ask_cost)
            ans = true_kernel.sample_answer(act.question_id, opp_ranking, rng)   # TRUE (maybe noisy) answer
            lik = fm.answer_likelihood(assumed, act.question_id, ans, spec)
            state = apply(state, act, ans)
            belief = fm.update(belief, lik, spec)
        else:
            state = apply(state, act)
        while state.terminal is None and state.turn != focal:
            oact = true_kernel.sample(state, opp_ranking, rng)
            lik = fm.opponent_action_likelihood(assumed, state, oact, spec)
            state = apply(state, oact)
            belief = fm.update(belief, lik, spec)
        steps.append(Step(sf, rf, idx, phi, env_r))

    t = state.terminal
    deal = DEALS_BY_ID[t.deal_id] if t.deal_id else None
    u = utility(focal, assumed_cell.ranking, deal) if deal else cfg.reservation
    terminal_gain = float(u - cfg.reservation)
    if steps:
        steps[-1].env_reward += terminal_gain
    net_gain = terminal_gain - float(cfg.ask_cost) * asks_by(state, focal)
    for i, s in enumerate(steps):
        phi_next = steps[i + 1].phi if i + 1 < len(steps) else 0.0
        s.shaped_reward = s.env_reward + phi_next - s.phi
    acc = 0.0
    for s in reversed(steps):
        acc += s.shaped_reward
        s.ret_to_go = acc
    phi0 = steps[0].phi if steps else 0.0
    return Episode(steps, net_gain, phi0, focal, ALL_RANKINGS.index(assumed_cell.ranking), opener,
                   asks=asks_by(state, focal))


def net_greedy(net) -> ChooseFn:
    """Deterministic (argmax) rollout policy for a trained net (eval)."""
    def choose(cell, state, belief, rows, sf, rf) -> int:
        import numpy as np
        return net.choose_index(sf, rf, np.ones(rf.shape[0], dtype=bool))
    return choose


def assumed_oracle_choose(cell, state, belief, rows, sf, rf) -> int:
    """O_assumed: the registered shortlist oracle's pick at the ASSUMED belief
    (the exact R0/R1 planner, deployed under misspecification -- same menu and
    belief machinery as the learned actor)."""
    best = cell.ask.best(state, belief)[0]
    return next(i for i, r in enumerate(rows) if r.key == best.key)


def otrue_net_gain(cfg, focal: str, ranking, true_kernel: T1Kernel, o_true: Oracle,
                   opener: str, opp_ranking, rng: random.Random) -> float:
    """O_true (clairvoyant per-draw ceiling): plan with the exact true-kernel oracle
    over the FULL action set at the EXACT true-kernel posterior. Returns the true net
    gain. This is the unrestricted best response to the known opponent -- the upper
    reference the misspecified, menu-restricted learned actor is measured against."""
    state = new_game(cfg, opener)
    true_b = bl.uniform()
    while state.terminal is None:
        if state.turn != focal:
            act = true_kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(true_kernel, state).get(act)
            state = apply(state, act)
            true_b = bl.posterior(true_b, lik) if lik is not None else true_b
            continue
        act = o_true.best(state, true_b)[0]
        if act.kind == ASK:
            ans = true_kernel.sample_answer(act.question_id, opp_ranking, rng)
            state = apply(state, act, ans)
            true_b = bl.posterior(true_b, bl.answer_likelihoods(true_kernel, act.question_id)[ans])
        else:
            state = apply(state, act)
    t = state.terminal
    deal = DEALS_BY_ID[t.deal_id] if t.deal_id else None
    u = utility(focal, ranking, deal) if deal else cfg.reservation
    return float(u - cfg.reservation) - float(cfg.ask_cost) * asks_by(state, focal)
