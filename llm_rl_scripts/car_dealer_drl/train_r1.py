"""R1 reinforcement learning: advantage actor-critic on the PBRS-shaped belief-MDP.

The net learns its policy from reward alone -- the exact oracle enters *only* as
the shaping potential ``Phi = V*`` inside ``env.rollout`` (never as an action
label, unlike R0). Each iteration samples a batch of episodes under the current
stochastic policy, computes shaped returns-to-go ``G'_t`` (gamma = 1), and takes
one actor-critic step:

    advantage  A_t = G'_t - V_pred(s_t)          (critic = the net's own value head)
    policy     L_pi = - mean( logpi(a_t|s_t) * stop_grad(A_t) )
    value      L_v  = mean( (V_pred(s_t) - G'_t)^2 )
    entropy    H    = mean entropy of pi over the valid menu (exploration bonus)
    loss = L_pi + value_coef * L_v - entropy_coef * H

Because Phi = V*, ``E[G'_t] = V^pi(s_t) - V*(s_t) <= 0`` (the regret-to-go), so the
value head learns the gap and the advantage is an unbiased, dense regret signal.
Model selection is on the greedy policy's optimal-match over a held-out evaluation
(not on the training return), keeping the best net.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from .env import Episode, net_sampler, rollout
from .evaluate import evaluate, net_choice_fn
from .net import NEG_INF, PolicyValueNet


@dataclass
class R1Config:
    hidden: int = 256
    iters: int = 400
    episodes_per_iter: int = 192
    lr: float = 3e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_final: float = 0.001      # linearly annealed to this by the last iter
    grad_clip: float = 1.0
    sample_temp: float = 1.0          # softmax temperature during rollout (exploration)
    normalize_adv: bool = True
    advantage: str = "nstep1"         # "nstep1" = one-step shaped reward r'_t (low variance
                                      #   because Phi=V* absorbs the continuation); "mc" = the
                                      #   full shaped return-to-go G'_t (higher variance)
    seed: int = 0
    eval_every: int = 10
    eval_repeats: int = 6             # cheap greedy eval for model selection
    patience: int = 12                # eval checkpoints without improvement
    shaping: bool = True              # False = sparse terminal reward (ablation)


def _batch(steps, target) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of Steps into (state, rows, mask, action_idx, targets). ``target``
    maps a Step to its scalar learning target (the one-step shaped reward or the MC
    shaped return-to-go)."""
    R = max(s.n_rows for s in steps)
    n = len(steps)
    S = np.zeros((n, steps[0].state_feat.shape[0]), dtype=np.float32)
    Rf = np.zeros((n, R, steps[0].row_feat.shape[1]), dtype=np.float32)
    M = np.zeros((n, R), dtype=bool)
    a = np.zeros(n, dtype=np.int64)
    g = np.zeros(n, dtype=np.float32)       # target: either r'_t (nstep1) or G'_t (mc)
    for t, s in enumerate(steps):
        r = s.n_rows
        S[t] = s.state_feat
        Rf[t, :r] = s.row_feat
        M[t, :r] = True
        a[t] = s.action_idx
        g[t] = target(s)
    return (torch.as_tensor(S), torch.as_tensor(Rf), torch.as_tensor(M),
            torch.as_tensor(a), torch.as_tensor(g))


def _entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    ent = -(p * logp).masked_fill(~mask, 0.0).sum(dim=1)
    return ent.mean()


def _sample_episodes(net: PolicyValueNet, cells: Dict[Tuple[str, int], Cell], n: int,
                     cfg: R1Config, rng: random.Random, gen: torch.Generator, potential=None) -> List[Episode]:
    choose = net_sampler(net, temperature=cfg.sample_temp, generator=gen)
    keys = list(cells.keys())
    openers = ("buyer", "seller")
    eps: List[Episode] = []
    for _ in range(n):
        role, i = keys[rng.randrange(len(keys))]
        opener = openers[rng.randrange(2)]
        j = rng.randrange(len(ALL_RANKINGS))
        ep = rollout(cells[(role, i)], opener, ALL_RANKINGS[j], rng, choose,
                     shaping=cfg.shaping, potential=potential)
        if ep.steps:
            eps.append(ep)
    return eps


def train_r1(cells: Dict[Tuple[str, int], Cell], cfg: R1Config = R1Config(), potential=None) -> Tuple[PolicyValueNet, Dict]:
    """``potential``: an optional R4 shaping potential ``potential(cell, state, belief)``
    passed through to ``env.rollout``; None uses the R1/R2 default (Phi=V* when
    ``cfg.shaping`` else 0)."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(f"r1|{cfg.seed}|{'shaped' if cfg.shaping else 'sparse'}")
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    net = PolicyValueNet(hidden=cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    target = (lambda s: s.shaped_reward) if cfg.advantage == "nstep1" else (lambda s: s.ret_to_go)
    best_match, best_state, best_iter, hist = -1.0, None, -1, []
    bad = 0
    for it in range(cfg.iters):
        eps = _sample_episodes(net, cells, cfg.episodes_per_iter, cfg, rng, gen, potential=potential)
        steps = [s for ep in eps for s in ep.steps]
        S, Rf, M, a, g = _batch(steps, target)

        net.train()
        logits, value = net(S, Rf, M)
        logp = F.log_softmax(logits, dim=1)[torch.arange(len(a)), a]
        adv = g - value.detach()
        if cfg.normalize_adv and adv.numel() > 1 and float(adv.std()) > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(value, g)
        frac = it / max(cfg.iters - 1, 1)
        ent_coef = cfg.entropy_coef + (cfg.entropy_final - cfg.entropy_coef) * frac
        entropy = _entropy(logits, M)
        loss = policy_loss + cfg.value_coef * value_loss - ent_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        if it % cfg.eval_every == 0 or it == cfg.iters - 1:
            ev = evaluate(cells, net_choice_fn(net), repeats=cfg.eval_repeats, seed=cfg.seed, sigma=4.0)
            rec = {"iter": it, "n_steps": len(steps), "loss": float(loss.detach()),
                   "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                   "entropy": float(entropy.detach()), "optimal_match": ev["optimal_match"],
                   "residual_z": ev["residual_z"], "mean_residual": ev["mean_residual"]}
            hist.append(rec)
            if ev["optimal_match"] > best_match:
                best_match, best_iter = ev["optimal_match"], it
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad > cfg.patience:
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"best_optimal_match": best_match, "best_iter": best_iter,
                 "history": hist, "config": cfg.__dict__}
