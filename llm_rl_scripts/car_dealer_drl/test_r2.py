"""R2 gates (fast subset: buyer, two focal types) for the ASK-discovery evaluator.

The RL learning itself is exercised by R1's tests; R2's novel code is the
ASK-aware evaluator and the verdict classifier, so the gates pin their endpoints:

1. the oracle policy asks at every strictly-ASK-optimal node (recall 1.0), takes
   asks (rate > 0) and captures essentially all of the value of inquiry;
2. a never-ask-but-noask-optimal policy has recall 0, ask rate 0, and captures
   ~none of the VOI (sits at the noask ceiling) -- the Jev-like endpoint;
3. the verdict classifier maps (captured_voi, recall) to the pre-registered labels,
   NaN-safe.
"""

from __future__ import annotations

import math

import pytest

from .ask_eval import evaluate_ask
from .dataset import build_cells
from .evaluate import oracle_choice
from .run_r2 import _verdict


@pytest.fixture(scope="module")
def small_cells():
    return build_cells(roles=("buyer",), types=(0, 1))


def _noask_oracle_choice(cell, state, belief, rows) -> int:
    """Best action under the ASK-forbidden oracle (never an ASK)."""
    best = cell.noask.best(state, belief)[0]
    return next(i for i, r in enumerate(rows) if r.key == best.key)


def test_oracle_endpoint(small_cells):
    ev = evaluate_ask(small_cells, oracle_choice, openers=("buyer", "seller"),
                      opp_types=(0, 1, 2, 3, 4, 5), repeats=10, seed=0)
    assert ev["n_ask_strict_optimal"] > 0, "subset has no strictly-ASK-optimal nodes to test recall"
    assert ev["ask_optimal_recall"] == pytest.approx(1.0), ev["ask_optimal_recall"]
    assert ev["ask_rate_per_episode"] > 0.0
    assert ev["within_sigma_se"], f"oracle off ceiling, z={ev['residual_z']:.2f}"
    assert ev["captured_voi"] > 0.5, ev["captured_voi"]   # captures most of the (noisy) VOI


def test_noask_endpoint(small_cells):
    ev = evaluate_ask(small_cells, _noask_oracle_choice, openers=("buyer", "seller"),
                      opp_types=(0, 1, 2, 3, 4, 5), repeats=10, seed=0)
    assert ev["n_asks"] == 0
    assert ev["ask_rate_per_episode"] == 0.0
    assert ev["ask_optimal_recall"] == pytest.approx(0.0)
    assert ev["captured_voi"] < 0.5, ev["captured_voi"]   # sits ~at the noask ceiling
    assert ev["mean_net"] == pytest.approx(ev["noask_ceiling"], abs=4 * ev["residual_se"] + 1e-9)


def test_verdict_classifier():
    assert _verdict(0.8, 0.8) == "DISCOVERED"
    assert _verdict(0.9, 0.49) == "PARTIAL"      # captured high, recall just under
    assert _verdict(0.30, 0.05) == "PARTIAL"     # captured >= 0.25
    assert _verdict(0.10, 0.20) == "PARTIAL"     # recall >= 0.10
    assert _verdict(0.10, 0.00) == "NOT_DISCOVERED"
    assert _verdict(0.10, float("nan")) == "NOT_DISCOVERED"   # no strict-ASK nodes -> recall 0
