"""R3 training: best-response to the CD-MIS opponent population.

Advantage actor-critic (reusing R1's batching/entropy/net), but each episode draws
a TRUE opponent uniformly from the training processes and rolls it out with
``env_mis.rollout_mis`` (assumed menu + belief, true opponent + reward).

Estimator: the one-step shaped-reward advantage r'_t = R_true + Phi_assumed(s') -
Phi_assumed(s) (Phi = V*_assumed). This is the honest learner for an agent whose
only model is the assumed one: the *immediate* reward R_true is the true outcome,
and only the continuation is bootstrapped on the assumed value. On the 'matched'
process it reduces to the exact R1 learner (recovers the menu optimum); on a
misspecified process it is biased only through the assumed-vs-true *continuation*
value, while the immediate true reward gives real signal to adapt. (The MC
return-to-go estimator would be unbiased but, with the potential telescoping to a
deterministic state constant, it carries no variance reduction from the shaping and
inherits R2's high-variance failure -- it would confound 'worse under
misspecification' with 'RL did not learn', so it is not used.) Model selection is on
mean true net over the training processes (never on the held-out ones).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .env import net_sampler
from .env_mis import net_greedy, rollout_mis
from .evaluate_mis import TRAIN_PROCESSES, R3Bundle, mean_net
from .net import PolicyValueNet
from .train_r1 import _batch, _entropy


@dataclass
class R3Config:
    hidden: int = 256
    iters: int = 600
    episodes_per_iter: int = 256
    lr: float = 1.5e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    entropy_final: float = 0.004
    grad_clip: float = 1.0
    sample_temp: float = 1.0
    normalize_adv: bool = True
    seed: int = 0
    eval_every: int = 20
    eval_repeats: int = 4            # cheap greedy MIS eval (training processes) for selection
    patience: int = 20


def _sample_episodes(net: PolicyValueNet, b: R3Bundle, processes: Sequence[str], n: int,
                     cfg: R3Config, rng: random.Random, gen: torch.Generator):
    choose = net_sampler(net, temperature=cfg.sample_temp, generator=gen)
    keys = list(b.assumed_cells.keys())
    openers = ("buyer", "seller")
    eps = []
    for _ in range(n):
        role, i = keys[rng.randrange(len(keys))]
        opener = openers[rng.randrange(2)]
        p = processes[rng.randrange(len(processes))]
        j = rng.randrange(len(ALL_RANKINGS))
        ep = rollout_mis(b.assumed_cells[(role, i)], b.true_kernels[(role, i, p)], opener,
                         ALL_RANKINGS[j], rng, choose, shaping=True)
        if ep.steps:
            eps.append(ep)
    return eps


def train_r3(b: R3Bundle, processes: Sequence[str] = TRAIN_PROCESSES,
             cfg: R3Config = R3Config()) -> Tuple[PolicyValueNet, Dict]:
    torch.manual_seed(cfg.seed)
    rng = random.Random(f"r3|{cfg.seed}")
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    roles = sorted({k[0] for k in b.assumed_cells})
    types = sorted({k[1] for k in b.assumed_cells})
    net = PolicyValueNet(hidden=cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    target = lambda s: s.shaped_reward   # one-step shaped-reward advantage (see module docstring)

    best_net, best_state, best_iter, hist, bad = -1e18, None, -1, [], 0
    for it in range(cfg.iters):
        eps = _sample_episodes(net, b, processes, cfg.episodes_per_iter, cfg, rng, gen)
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
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        if it % cfg.eval_every == 0 or it == cfg.iters - 1:
            mn = mean_net(net_greedy(net), b, processes, roles=roles, types=types,
                          repeats=cfg.eval_repeats, seed=cfg.seed)
            hist.append({"iter": it, "n_steps": len(steps), "train_mean_net": mn,
                         "entropy": float(entropy.detach()), "policy_loss": float(policy_loss.detach()),
                         "value_loss": float(value_loss.detach())})
            if mn > best_net:
                best_net, best_iter = mn, it
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad > cfg.patience:
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"best_train_mean_net": best_net, "best_iter": best_iter,
                 "train_processes": list(processes), "history": hist, "config": cfg.__dict__}
