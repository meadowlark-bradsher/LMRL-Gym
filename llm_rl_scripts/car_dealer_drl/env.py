"""The belief-MDP rollout for R1 (reinforcement learning), with potential-based
reward shaping.

A *focal decision node* ``(public_state, belief)`` is one state of the belief-MDP;
the environment transition out of it folds in the focal's chosen action *and* the
opponent's replies (which advance the matched belief filter) up to the next focal
node or terminal. Environmental reward is realized only at terminal -- the net
gain ``u - r`` on an accepted deal, 0 on no deal -- minus ``ask_cost`` per focal
ASK. This is exactly the dynamics ``evaluate.run_episode`` scores, written as a
reward-emitting rollout so an RL learner can consume it.

**Potential-based shaping (Ng-Harada-Russell).** The shaped reward of the
transition from focal node ``s_t`` to the next focal node ``s_{t+1}`` is

    r'_t = R_t + gamma * Phi(s_{t+1}) - Phi(s_t),   gamma = 1 (finite horizon),

with the potential ``Phi(s) = cell.ask.value(state, belief)`` = the exact V* at a
focal node, and the terminal clamp ``Phi(terminal) = 0``. Because the oracle's
``q_value`` is exactly ``E[R_t + Phi(s_{t+1}) | s_t, a]``, the expected one-step
shaped reward is ``Q*(s_t, a) - V*(s_t) = A*(s_t, a)`` -- the optimal advantage.
PBRS leaves the optimal policy invariant; Phi = V* makes the per-step signal the
advantage itself, which is the "dense help" R1 tests. Summed over an episode the
shaping telescopes to ``-Phi(s_0)`` (``test_r1`` asserts this), so the shaped
return and the true return differ only by the constant ``Phi(s_0)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, apply, asks_by, new_game
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from .dataset import context_for
from .features import rows_matrix, state_features

# A rollout policy: choose a menu row index for a focal node. It is handed both the
# featurized node (for a net) and the raw node (for the oracle / scripted choosers).
ChooseFn = Callable[[Cell, object, bl.Belief, list, np.ndarray, np.ndarray], int]


@dataclass
class Step:
    """One focal decision in a rollout, with the labels an RL update needs."""

    state_feat: np.ndarray    # [STATE_DIM]
    row_feat: np.ndarray      # [n_rows, ROW_DIM]
    action_idx: int           # the chosen row
    phi: float                # Phi(s_t) = V*(s_t), the potential at this node
    env_reward: float         # R_t realized on the transition OUT of this node
    shaped_reward: float = 0.0  # r'_t = R_t + Phi(s_{t+1}) - Phi(s_t); filled post hoc
    ret_to_go: float = 0.0    # G'_t = sum_{k>=t} r'_k within the episode (gamma=1)

    @property
    def n_rows(self) -> int:
        return self.row_feat.shape[0]


@dataclass
class Episode:
    steps: List[Step]
    net_gain: float           # the true (unshaped) episodic net gain u - r - c_q*asks
    phi0: float               # Phi(s_0): shaped return = net_gain - phi0 (telescoping)
    role: str
    type_idx: int
    opener: str
    asks: int = 0             # focal ASK actions taken this episode


def _terminal_gain(cell: Cell, state) -> float:
    t = state.terminal
    if t.deal_id is None:
        return 0.0
    return float(utility(cell.focal, cell.ranking, DEALS_BY_ID[t.deal_id]) - cell.cfg.reservation)


def rollout(cell: Cell, opener: str, opp_ranking, rng: random.Random, choose: ChooseFn,
            shaping: bool = True, potential: Optional[Callable] = None) -> Episode:
    """Play one episode under ``choose`` against the matched kernel and emit shaped
    rewards. The potential is ``potential(cell, state, belief)`` when given (R4's
    weaker/oracle-free potentials); otherwise Phi = V*_assumed when ``shaping`` else
    Phi = 0 (the R1/R2 default). The optimal policy is invariant to the choice."""
    cfg, focal = cell.cfg, cell.focal
    ctx = context_for(cell)
    state = new_game(cfg, opener)
    belief = bl.uniform()
    steps: List[Step] = []

    while state.terminal is None:
        if state.turn != focal:                                   # opponent node: part of the transition
            act = cell.kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(cell.kernel, state)[act]
            state = apply(state, act)
            belief = bl.posterior(belief, lik)
            continue
        # focal decision node
        sf = state_features(ctx, state, belief)
        rows = cell.menu.rows(state, belief, include_questions=True)
        rf = rows_matrix(rows)
        if potential is not None:
            phi = float(potential(cell, state, belief))
        else:
            phi = float(cell.ask.value(state, belief)) if shaping else 0.0
        idx = choose(cell, state, belief, rows, sf, rf)
        chosen = rows[idx]
        env_r = 0.0
        act = chosen.action
        if act.kind == ASK:
            env_r -= float(cfg.ask_cost)
            ans = cell.kernel.sample_answer(act.question_id, opp_ranking, rng)
            lik = bl.answer_likelihoods(cell.kernel, act.question_id)[ans]
            state = apply(state, act, ans)
            belief = bl.posterior(belief, lik)
        else:
            state = apply(state, act)
        # advance opponent replies until the next focal node or terminal
        while state.terminal is None and state.turn != focal:
            oact = cell.kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(cell.kernel, state)[oact]
            state = apply(state, oact)
            belief = bl.posterior(belief, lik)
        steps.append(Step(sf, rf, idx, phi, env_r))

    terminal_gain = _terminal_gain(cell, state)
    if steps:
        steps[-1].env_reward += terminal_gain                     # reward lands on the last transition
    net_gain = terminal_gain - float(cfg.ask_cost) * asks_by(state, focal)

    # shaped reward r'_t = R_t + Phi(s_{t+1}) - Phi(s_t), Phi(terminal) = 0
    for i, s in enumerate(steps):
        phi_next = steps[i + 1].phi if i + 1 < len(steps) else 0.0
        s.shaped_reward = s.env_reward + phi_next - s.phi
    acc = 0.0
    for s in reversed(steps):                                     # G'_t, gamma = 1
        acc += s.shaped_reward
        s.ret_to_go = acc

    phi0 = steps[0].phi if steps else 0.0
    return Episode(steps, net_gain, phi0, focal, ALL_RANKINGS.index(cell.ranking), opener,
                   asks=asks_by(state, focal))


def net_sampler(net, temperature: float = 1.0, generator: Optional["object"] = None) -> ChooseFn:
    """Stochastic rollout policy: sample a row from the net's softmax (exploration)."""
    import torch

    @torch.no_grad()
    def choose(cell, state, belief, rows, sf, rf) -> int:
        S = torch.as_tensor(sf, dtype=torch.float32).unsqueeze(0)
        Rf = torch.as_tensor(rf, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, rf.shape[0], dtype=torch.bool)
        logits, _ = net(S, Rf, mask)
        probs = torch.softmax(logits[0] / temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return choose


def oracle_sampler() -> ChooseFn:
    """Greedy exact-oracle rollout policy (validates the env dynamics against the
    ceiling, independently of ``evaluate``)."""
    def choose(cell, state, belief, rows, sf, rf) -> int:
        best_action = cell.ask.best(state, belief)[0]
        return next(i for i, r in enumerate(rows) if r.key == best_action.key)
    return choose
