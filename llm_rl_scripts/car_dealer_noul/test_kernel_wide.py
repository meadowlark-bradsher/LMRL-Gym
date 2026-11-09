"""The widened kernel family of DESIGN_NOUL2 §3.3a (eps_propose, p_ask): defaults reproduce T1 bit for bit and
leave every pre-widening JSON unchanged; widened distributions are proper, give every legal proposal positive
mass under every type, and carry no type evidence in an opponent question; the oracle plans through opponent
questions; the fit recovers planted support parameters; the split count is zero for proposals once eps > 0."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.oracle import Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, PROPOSE, LiveConfig, legal_actions

from . import gates_noul2 as g
from . import kernel_fit as kf
from .test_kernel_fit import CFG, simulate

WIDE = replace(REFERENCE, eps_propose=0.3, p_ask=0.05)
PLANTED_WIDE = replace(REFERENCE, alpha0=Fraction(7, 10), beta=3.0, p_counter=0.9, p_hold=0.5, accept_noise=0.1,
                       eps_propose=0.25, p_ask=0.06)


def _states(n: int = 300, seed: int = 4):
    recs = simulate(PLANTED_WIDE, 1, seed=seed)
    return kf.prepare(recs[:n], CFG)


def test_defaults_leave_json_and_distribution_unchanged():
    assert "eps_propose" not in REFERENCE.to_json() and "p_ask" not in REFERENCE.to_json()
    assert set(KernelParams().to_json()) == {"alpha0", "beta", "p_counter", "accept_noise", "p_hold", "shortlist"}
    zero = replace(REFERENCE, eps_propose=0.0, p_ask=0.0)
    for p in _states(150):
        for th in ALL_RANKINGS:
            assert T1Kernel(p.role, CFG, zero).distribution(p.state, th) == T1Kernel(p.role, CFG, REFERENCE).distribution(p.state, th)


def test_widened_distribution_is_proper_and_full_support():
    for p in _states():
        k = T1Kernel(p.role, CFG, WIDE)
        legal = legal_actions(p.state, p.role)
        props = [a for a in legal if a.kind == PROPOSE]
        asks = [a for a in legal if a.kind == ASK]
        ask_vecs = []
        for th in ALL_RANKINGS:
            d = k.distribution(p.state, th)
            assert abs(sum(d.values()) - 1.0) < 1e-9 and all(a in legal for a in d)
            assert all(d.get(a, 0.0) > 0 for a in props)
            ask_vecs.append(tuple(d.get(a, 0.0) for a in asks))
        assert len(set(ask_vecs)) == 1                    # an opponent question is type-independent
        if asks:
            assert math.isclose(sum(ask_vecs[0]), WIDE.p_ask)


def test_deadline_nodes_are_not_widened():
    for p in _states(600):
        if p.state.at_deadline:
            for th in ALL_RANKINGS:
                assert T1Kernel(p.role, CFG, WIDE).distribution(p.state, th) == T1Kernel(p.role, CFG, REFERENCE).distribution(p.state, th)


def test_oracle_plans_through_opponent_questions():
    cfg = LiveConfig(regime="easy", horizon=2, ask_cost=Fraction(0))
    for focal in ("buyer", "seller"):
        k = T1Kernel("seller" if focal == "buyer" else "buyer", cfg, WIDE)
        o = Oracle(cfg, focal, ALL_RANKINGS[0], k, ask_allowed=True)
        v = o.root_value("seller" if focal == "buyer" else "buyer")      # the opponent opens and may ask
        assert math.isfinite(v) and v >= 0


def test_fit_recovers_planted_support_parameters():
    recs = simulate(PLANTED_WIDE, 1, seed=13)
    prep = kf.prepare(recs, CFG)
    fitted, ll = kf.fit_best(prep, CFG, params_to_fit=kf.FIT_PARAMS_WIDE)
    assert ll >= kf.mean_ll(prep, PLANTED_WIDE, CFG) - 1e-6
    assert abs(fitted.eps_propose - 0.25) < 0.08, fitted
    assert abs(fitted.p_ask - 0.06) < 0.03, fitted
    _, zeros = kf.log_likelihood(prep, fitted, CFG)
    assert zeros == 0


def test_split_count_zero_for_proposals_once_widened():
    prep = kf.prepare(simulate(PLANTED_WIDE, 1, seed=17), CFG)
    t1 = g.split_breakdown(prep, REFERENCE, CFG)
    assert t1["split"] > 0 and t1["by_kind"]["ASK"].get("split", 0) == 0
    wide = g.split_breakdown(prep, replace(PLANTED_WIDE), CFG)
    assert wide["zero_at_true"] == 0 and wide["split"] == 0
    same = g.split_breakdown(kf.prepare(simulate(REFERENCE, 1, seed=19), CFG), REFERENCE, CFG)
    assert same["zero_at_true"] == 0
