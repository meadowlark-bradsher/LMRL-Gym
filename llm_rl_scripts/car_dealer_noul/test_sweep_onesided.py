"""Exactness checks for the one-sided Car-Dealer sweep (``sweep_onesided.py``).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_noul/test_sweep_onesided.py
"""

import json
import os
from fractions import Fraction

import pytest

from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.protocol_live import ROLES

from .sweep_onesided import ALL_TYPES, N_TYPES, YES_SETS, Cell, OneSidedConfig, render, solve_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, "results", "onesided_sweep_20261109", "sweep.json")
ANCHOR = os.path.join(ROOT, "results", "noul_stage1_20261108", "analysis.json")
NOTE = os.path.join(ROOT, "llm_rl_scripts", "car_dealer_noul", "ONESIDED_SWEEP.md")

CELLS = [(role, ALL_RANKINGS[i]) for role in ROLES for i in (0, 3)]


def _p_acc(cell, d, asp, S):
    return Fraction(sum(1 for t in S if cell.u_opp[d][t] >= asp), len(S))


def _brute_offers_h2(cell):
    """Offers only, H = 2, no hazard, by enumerating every open-loop pair (the only feedback is accept/reject)."""
    cfg = cell.cfg
    a1, a0 = cfg.aspiration(1), cfg.aspiration(0)
    best = Fraction(0)
    for d1 in range(len(ALL_DEALS)):
        acc1 = frozenset(t for t in ALL_TYPES if cell.u_opp[d1][t] >= a1)
        v1 = Fraction(len(acc1), N_TYPES) * cell.gain[d1]
        rej = ALL_TYPES - acc1
        second = Fraction(0)
        if rej:
            second = max(max(_p_acc(cell, d2, a0, rej) * cell.gain[d2] for d2 in range(len(ALL_DEALS))), Fraction(0))
        best = max(best, v1 + Fraction(len(rej), N_TYPES) * second)
    return best


def _brute_words_h2(cell):
    """Offers + questions, H = 2: either two offers, or one question then the best last offer per answer."""
    cfg = cell.cfg
    a0 = cfg.aspiration(0)
    best = _brute_offers_h2(cell)
    for q in range(len(BANK)):
        v = -cfg.ask_cost
        for part in (YES_SETS[q], ALL_TYPES - YES_SETS[q]):
            if part:
                last = max(max(_p_acc(cell, d, a0, part) * cell.gain[d] for d in range(len(ALL_DEALS))), Fraction(0))
                v += Fraction(len(part), N_TYPES) * last
        best = max(best, v)
    return best


@pytest.mark.parametrize("regime,alpha0,cost", [("easy", Fraction(4, 5), Fraction(0)), ("hard", None, Fraction(1, 100)),
                                                ("easy", Fraction(3, 5), Fraction(3, 100))])
def test_h2_matches_brute_force(regime, alpha0, cost):
    cfg = OneSidedConfig(regime=regime, horizon=2, alpha0=alpha0, ask_cost=cost)
    for role, ranking in CELLS:
        cell = Cell(cfg, role, ranking)
        assert cell.best(2, ALL_TYPES, frozenset(), ask=False)[0] == _brute_offers_h2(cell)
        assert cell.best(2, ALL_TYPES, frozenset(), ask=True)[0] == _brute_words_h2(cell)


@pytest.mark.parametrize("H", [1, 2, 3])
@pytest.mark.parametrize("alpha0,hazard", [(Fraction(4, 5), Fraction(0)), (None, Fraction(1, 10))])
def test_value_ordering(H, alpha0, hazard):
    cfg = OneSidedConfig(horizon=H, alpha0=alpha0, hazard=hazard)
    for role, ranking in CELLS:
        s = Cell(cfg, role, ranking).solve()
        assert 0 <= s["V_one"] <= s["V_offers"] <= s["V_words"] <= s["V_reveal"]
        if H == 1:
            assert s["V_one"] == s["V_offers"] == s["V_words"]


def test_flat_no_hazard_reveal_and_info_do_not_depend_on_horizon():
    outs = [solve_config(OneSidedConfig(horizon=H, alpha0=None)) for H in (1, 2, 3)]
    assert len({o["summary"]["exact_mean_Y"]["V_reveal"] for o in outs}) == 1
    assert len({o["summary"]["exact_mean_Y"]["V_one"] for o in outs}) == 1


def test_hazard_never_helps():
    for role, ranking in CELLS:
        v0 = Cell(OneSidedConfig(horizon=3), role, ranking).solve()
        v1 = Cell(OneSidedConfig(horizon=3, hazard=Fraction(1, 10)), role, ranking).solve()
        for k in ("V_one", "V_offers", "V_words", "V_reveal"):
            assert v1[k] <= v0[k]


def test_ideal_offer_is_a_sure_rejection():
    """Ignoring answers = not asking relies on an offer every ranking rejects at every aspiration."""
    for role, ranking in CELLS:
        cell = Cell(OneSidedConfig(), role, ranking)
        ideal = max(range(len(ALL_DEALS)), key=lambda d: cell.gain[d])
        assert all(u == 0 for u in cell.u_opp[ideal])


@pytest.mark.skipif(not os.path.exists(SWEEP), reason="sweep not run")
def test_regeneration():
    with open(SWEEP) as f:
        doc = json.load(f)
    small = [c for c in doc["configs"] if c["config"]["horizon"] <= 3]
    assert small
    for c in small:
        k = c["config"]
        cfg = OneSidedConfig(regime=k["regime"], horizon=k["horizon"],
                             alpha0=None if k["alpha0"] == "flat" else Fraction(k["alpha0"]),
                             hazard=Fraction(k["hazard"]), ask_cost=Fraction(k["ask_cost"]))
        assert solve_config(cfg)["summary"]["exact_mean_Y"] == c["summary"]["exact_mean_Y"]
    if os.path.exists(NOTE):
        with open(ANCHOR) as f:
            anchor = json.load(f)
        with open(NOTE) as f:
            assert f.read() == render(doc, anchor, "results/onesided_sweep_20261109/sweep.json",
                                      "results/noul_stage1_20261108/analysis.json")
