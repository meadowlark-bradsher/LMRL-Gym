"""R1 gates (fast subset: buyer, two focal types).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_drl/test_r1.py

1. PBRS telescoping identity: over any episode, sum of shaped rewards ==
   net_gain - Phi(s_0). This validates the env's reward/potential bookkeeping and
   is the formal reason the shaping leaves the optimal policy invariant.
2. the env dynamics reproduce the exact ceiling when driven by the oracle sampler
   (independent of ``evaluate``), confirming the rollout's transitions and reward.
3. shaping off is the null: Phi == 0 everywhere, so the shaped reward is just the
   sparse terminal reward and the telescoping constant Phi(s_0) is 0.
4. actor-critic with PBRS learns on a small cell set (greedy optimal-match climbs
   well above chance within a tiny budget).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .dataset import build_cells
from .env import oracle_sampler, rollout
from .train_r1 import R1Config, train_r1


@pytest.fixture(scope="module")
def small_cells():
    return build_cells(roles=("buyer",), types=(0, 1))


def test_pbrs_telescoping(small_cells):
    """sum_t r'_t == net_gain - Phi(s_0) for every rollout (gamma = 1, Phi_terminal = 0)."""
    choose = oracle_sampler()
    checked = 0
    for (role, i), cell in small_cells.items():
        for opener in ("buyer", "seller"):
            for j in range(len(ALL_RANKINGS)):
                for k in range(3):
                    rng = random.Random(f"tele|{role}|{opener}|{i}|{j}|{k}")
                    ep = rollout(cell, opener, ALL_RANKINGS[j], rng, choose, shaping=True)
                    if not ep.steps:
                        continue
                    shaped_sum = sum(s.shaped_reward for s in ep.steps)
                    assert shaped_sum == pytest.approx(ep.net_gain - ep.phi0, abs=1e-6)
                    # G'_0 is the full shaped return
                    assert ep.steps[0].ret_to_go == pytest.approx(shaped_sum, abs=1e-6)
                    checked += 1
    assert checked > 50


def test_env_oracle_reproduces_ceiling(small_cells):
    """Oracle-driven rollouts hit the exact ceiling within 4 SE (validates dynamics)."""
    choose = oracle_sampler()
    residuals = []
    for (role, i), cell in small_cells.items():
        for opener in ("buyer", "seller"):
            ceiling = cell.ask.root_value(opener)
            for j in range(len(ALL_RANKINGS)):
                for k in range(12):
                    rng = random.Random(f"ceil|{role}|{opener}|{i}|{j}|{k}")
                    ep = rollout(cell, opener, ALL_RANKINGS[j], rng, choose, shaping=True)
                    residuals.append(ep.net_gain - ceiling)
    r = np.array(residuals)
    se = r.std(ddof=1) / math.sqrt(len(r))
    z = r.mean() / se if se > 0 else 0.0
    assert abs(z) <= 4.0, f"oracle rollouts off ceiling: residual {r.mean():+.4f} ({z:+.2f} SE)"


def test_shaping_off_is_sparse(small_cells):
    """Phi == 0 everywhere, so shaped reward == env reward and the constant is 0."""
    choose = oracle_sampler()
    cell = small_cells[("buyer", 0)]
    rng = random.Random("sparse|0")
    ep = rollout(cell, "buyer", ALL_RANKINGS[0], rng, choose, shaping=False)
    assert all(s.phi == 0.0 for s in ep.steps)
    assert ep.phi0 == 0.0
    assert sum(s.shaped_reward for s in ep.steps) == pytest.approx(ep.net_gain, abs=1e-6)


def test_pbrs_rl_learns(small_cells):
    """Actor-critic with Phi=V* and the one-step shaped-reward advantage climbs to
    high optimal-match on a small budget (the MC return-to-go estimator does not --
    see train_r1: Phi=V* makes the one-step reward a low-variance advantage)."""
    cfg = R1Config(hidden=64, iters=150, episodes_per_iter=96, eval_every=10,
                   eval_repeats=6, patience=100, seed=0)
    _, info = train_r1(small_cells, cfg)
    assert info["best_optimal_match"] >= 0.9, info["history"][-1]
