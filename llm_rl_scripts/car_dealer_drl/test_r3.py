"""R3 gates (fast subset: buyer, two focal types) for the misspecification machinery.

1. the robustified belief filter survives every true process (including threshold
   and withholding, which take actions the assumed kernel gives ~0 probability) --
   no zero-division, belief stays a distribution;
2. O_true (clairvoyant) >= O_assumed per process, up to MC noise -- the ceiling
   dominates the misspecified planner, so the learned actor is bracketed correctly;
3. on the 'matched' process the assumed belief is calibrated and O_assumed is near
   the clairvoyant ceiling (misspecification gap ~ 0) -- the sanity endpoint;
4. train_r3 learns on a tiny budget (train-mean-net climbs well above its start).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .env_mis import DEFAULT_FILTER, assumed_oracle_choose, rollout_mis
from .evaluate_mis import HELDOUT_PROCESSES, TRAIN_PROCESSES, build_r3, _otrue_by_process, _policy_by_process
from .train_r3 import R3Config, train_r3

R, T, OPP = ("buyer",), (0, 1), (0, 1, 2, 3, 4, 5)


@pytest.fixture(scope="module")
def bundle():
    return build_r3(roles=R, types=T)


def test_filter_survives_all_processes(bundle):
    """Belief stays a valid distribution against every true process (no filter break)."""
    for p in TRAIN_PROCESSES + HELDOUT_PROCESSES:
        cell = bundle.assumed_cells[("buyer", 0)]
        tk = bundle.true_kernels[("buyer", 0, p)]
        for k in range(6):
            rng = random.Random(f"{p}|{k}")
            ep = rollout_mis(cell, tk, "buyer", ALL_RANKINGS[k % 6], rng, assumed_oracle_choose)
            assert ep.steps, f"no focal steps for {p}"


def test_otrue_dominates_oassumed(bundle):
    oa = _policy_by_process(assumed_oracle_choose, bundle, TRAIN_PROCESSES + HELDOUT_PROCESSES, R, T, OPP, 8, 0)
    ot = _otrue_by_process(bundle, TRAIN_PROCESSES + HELDOUT_PROCESSES, R, T, OPP, 8, 0)
    for p in oa:
        assert ot[p]["mean_net"] >= oa[p]["mean_net"] - 3 * oa[p]["se"], \
            f"O_true below O_assumed for {p}: {ot[p]['mean_net']:.4f} < {oa[p]['mean_net']:.4f}"


def test_matched_is_calibrated(bundle):
    """On 'matched' the assumed model is correct, so O_assumed ~ O_true (gap ~ 0)."""
    oa = _policy_by_process(assumed_oracle_choose, bundle, ("matched",), R, T, OPP, 10, 0)["matched"]
    ot = _otrue_by_process(bundle, ("matched",), R, T, OPP, 10, 0)["matched"]
    assert abs(ot["mean_net"] - oa["mean_net"]) <= 4 * oa["se"] + 1e-6, \
        f"matched not calibrated: O_true {ot['mean_net']:.4f} vs O_assumed {oa['mean_net']:.4f}"


def test_train_r3_learns(bundle):
    cfg = R3Config(hidden=64, iters=120, episodes_per_iter=96, eval_every=20, eval_repeats=4,
                   patience=100, seed=0)
    _, info = train_r3(bundle, TRAIN_PROCESSES, cfg)
    h = info["history"]
    assert h[-1]["train_mean_net"] > h[0]["train_mean_net"] + 0.05, \
        f"train_r3 did not learn: {h[0]['train_mean_net']:.3f} -> {h[-1]['train_mean_net']:.3f}"
