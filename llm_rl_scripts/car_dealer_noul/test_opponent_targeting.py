"""Checks for the opponent-targeting re-analysis (``opponent_targeting.py``).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_noul/test_opponent_targeting.py
"""

import json
import os
from fractions import Fraction

import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_live.protocol_live import ROLES, LiveConfig, other

from .opponent_targeting import (analyze, best_at_concession, ideal, midpoint, render, score_record, spec,
                                 summarize)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(ROOT, "results", "noul2_pilot_20261108")
OUT = os.path.join(ROOT, "results", "noul2_targeting_20261109", "targeting.json")
NOTE = os.path.join(ROOT, "llm_rl_scripts", "car_dealer_noul", "TARGETING_NOUL2.md")
CFG = LiveConfig()


def test_spec_sums_to_zero_over_rankings():
    for role in ROLES:
        for d in ALL_DEALS:
            assert sum((spec(role, t, d) for t in ALL_RANKINGS), Fraction(0)) == 0


def _opening(focal, fi, oi, deal_id):
    return {"episode": f"plan_noask|candid|{focal}|{other(focal)}|{fi}|{oi}|0", "brief": "candid",
            "opener": other(focal), "role": other(focal), "ranking": list(ALL_RANKINGS[oi]), "events": [],
            "action": f"PROPOSE {deal_id}"}


def _balanced(rule):
    kernels, rows = {}, []
    for focal in ROLES:
        for fi in range(6):
            for oi in range(6):
                rows.append(score_record(_opening(focal, fi, oi, rule(focal, fi, oi).id), CFG, kernels))
    return rows


def test_focal_blind_rules_average_zero_and_targeting_does_not():
    blind = _balanced(lambda f, fi, oi: ideal(other(f), ALL_RANKINGS[oi]))
    assert abs(sum(r["spec"] for r in blind) / len(blind)) < 1e-12
    assert abs(sum(r["t1_spec"] for r in blind) / len(blind)) < 1e-12        # the T1 placebo is exact
    floor = Fraction(1, 2)
    aimed = _balanced(lambda f, fi, oi: best_at_concession(f, ALL_RANKINGS[fi], other(f), ALL_RANKINGS[oi], floor))
    assert sum(r["spec"] for r in aimed) / len(aimed) > 0.05
    s = summarize(aimed)
    assert s["spec_llm_minus_t1"]["ci95"][0] > 0


def test_midpoint_and_best_at_concession():
    for a in ALL_DEALS:
        assert midpoint(a, a) == a
    a, b = DEALS_BY_ID["P0-W0-D0"], DEALS_BY_ID["P1-W2-D1"]
    m = midpoint(a, b)
    assert (m.level("price"), m.level("warranty"), m.level("delivery")) == (1, 1, 1)   # halves round toward b
    for role in ROLES:
        for oi in (0, 4):
            d = best_at_concession(other(role), ALL_RANKINGS[1], role, ALL_RANKINGS[oi], Fraction(0))
            assert utility(other(role), ALL_RANKINGS[1], d) == 1          # no floor: the focal side's ideal


@pytest.mark.skipif(not (os.path.exists(OUT) and os.path.exists(PILOT)), reason="pilot or analysis missing")
def test_regeneration():
    with open(OUT) as f:
        stored = json.load(f)
    fresh = analyze(PILOT)
    fresh["pilot"], fresh["commit"] = stored["pilot"], stored["commit"]
    assert json.loads(json.dumps(fresh)) == stored
    if os.path.exists(NOTE):
        with open(NOTE) as f:
            assert f.read() == render(stored, "results/noul2_targeting_20261109/targeting.json")
