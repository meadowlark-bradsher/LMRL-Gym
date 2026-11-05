"""R0 gates (fast subset: buyer, two focal types).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_drl

1. feature dims match the featurizer;
2. the featurization is sufficient for the decision — no two collected nodes with
   identical features have different optimal sets or clone targets (so the net is
   never asked to map one input to two answers);
3. the eval harness reproduces ``exact_ask`` when driven by the oracle shim
   (optimal-match 1.0, within 4 SE of the ceiling);
4. the net forward respects the row mask (padded rows get ~zero probability);
5. BC learns (a small net reaches high optimal-match on a small dataset).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .dataset import build_cells, collect_nodes, context_for, to_tensors
from .evaluate import evaluate, net_choice_fn, oracle_choice
from .features import ROW_DIM, STATE_DIM, rows_matrix, state_features
from .net import PolicyValueNet
from .train import TrainConfig, train


@pytest.fixture(scope="module")
def small_cells():
    return build_cells(roles=("buyer",), types=(0, 1))


@pytest.fixture(scope="module")
def small_nodes(small_cells):
    return collect_nodes(small_cells, openers=("buyer", "seller"), opp_types=(0, 1, 2, 3, 4, 5), repeats=2, seed=0)


def test_feature_dims(small_cells):
    cell = small_cells[("buyer", 0)]
    from llm_rl_scripts.car_dealer_live import belief_live as bl
    from llm_rl_scripts.car_dealer_live.protocol_live import new_game
    st = new_game(cell.cfg, "buyer")
    b = bl.uniform()
    rows = cell.menu.rows(st, b, True)
    assert state_features(context_for(cell), st, b).shape == (STATE_DIM,)
    assert rows_matrix(rows).shape == (len(rows), ROW_DIM)


def test_feature_sufficiency(small_nodes):
    """Injective w.r.t. the decision: equal features => equal optimal set and target."""
    by_feat = {}
    for nd in small_nodes:
        key = (nd.state_feat.tobytes(), nd.row_feat.tobytes())
        if key in by_feat:
            prev = by_feat[key]
            assert prev.best_idx == nd.best_idx, "feature collision with different clone target"
            assert np.array_equal(prev.optimal_mask, nd.optimal_mask), "feature collision with different optimal set"
        else:
            by_feat[key] = nd
    assert len(small_nodes) > 50  # the subset actually produced coverage


def test_oracle_shim_reproduces_exact_ask(small_cells):
    res = evaluate(small_cells, oracle_choice, openers=("buyer", "seller"), opp_types=(0, 1, 2, 3, 4, 5),
                   repeats=6, seed=0, sigma=4.0)
    assert res["optimal_match"] == pytest.approx(1.0)
    assert res["mean_decision_regret"] <= 1e-9
    assert res["within_sigma_se"], f"oracle shim should sit within SE of ceiling, z={res['residual_z']:.2f}"


def test_net_masks_padded_rows():
    torch.manual_seed(0)
    net = PolicyValueNet(hidden=32)
    S = torch.randn(2, STATE_DIM)
    R = torch.randn(2, 5, ROW_DIM)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    logits, value = net(S, R, mask)
    probs = torch.softmax(logits, dim=1)
    assert torch.all(probs[~mask] < 1e-6)
    assert value.shape == (2,)


def test_bc_learns(small_nodes):
    ds = to_tensors(small_nodes)
    net, info = train(ds, TrainConfig(hidden=64, epochs=200, seed=0, patience=200))
    assert info["full"]["optimal_match"] >= 0.9, info["full"]
