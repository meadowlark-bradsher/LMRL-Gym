"""Phase 1 gates: menu rule, menu-restricted oracle, arms, runner."""

import gzip
import json
import math
import os
import random
import tempfile
from fractions import Fraction

import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from . import belief_live as bl
from .arms_live import arms_for
from .menu_live import Menu
from .opponent import KernelParams, T1Kernel
from .oracle import Oracle
from .protocol_live import ASK, PROPOSE, WALK, LiveConfig, apply, legal_actions, new_game, other
from .run_phase1 import Cell, main as run_main

CFG = LiveConfig(horizon=3)
PARAMS = KernelParams()


def _walk_states(cfg, focal, ranking, kernel, menu, n=300, seed=1):
    """Random play; yields (state, belief) at focal decision nodes."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        opp = rng.choice(ALL_RANKINGS)
        st, b = new_game(cfg, rng.choice(("buyer", "seller"))), bl.uniform()
        while st.terminal is None:
            if st.turn == focal:
                out.append((st, b))
                rows = menu.rows(st, b)
                a = rng.choice(rows).action
                if a.kind == ASK:
                    ans = kernel.sample_answer(a.question_id, opp, rng)
                    b = bl.posterior(b, bl.answer_likelihoods(kernel, a.question_id)[ans])
                    st = apply(st, a, ans)
                else:
                    st = apply(st, a)
            else:
                a = kernel.sample(st, opp, rng)
                b = bl.posterior(b, bl.action_likelihoods(kernel, st)[a])
                st = apply(st, a)
    return out


def test_menu_is_legal_small_and_well_formed():
    focal, rk = "buyer", ALL_RANKINGS[0]
    kernel = T1Kernel(other(focal), CFG, PARAMS)
    menu = Menu(CFG, focal, rk, kernel, plan_oracle=Oracle(CFG, focal, rk, kernel, False), plan_ask_oracle=Oracle(CFG, focal, rk, kernel, True))
    r = CFG.reservation
    seen_deadline = seen_two_offers = seen_plan = False
    for st, b in _walk_states(CFG, focal, rk, kernel, menu):
        rows = menu.rows(st, b)
        legal = set(legal_actions(st, focal))
        acts = [x.action for x in rows]
        assert len(set(acts)) == len(acts) and all(a in legal for a in acts)
        offers = [x for x in rows if x.action.kind == PROPOSE]
        qs = [x for x in rows if x.action.kind == ASK]
        assert len(offers) <= 4 and len(qs) <= 3
        assert all(x.own_utility >= r for x in offers)             # own-IR pool
        assert all(x.tag in ("SAFE", "AMBITIOUS", "PLAN", "PLAN_ASK", "BALANCED") for x in offers)
        assert all(x.tag in (None, "PLAN_Q") for x in qs)
        if any(x.tag == "PLAN" for x in offers):
            seen_plan = True
        assert all(0.0 <= x.p_accept <= 1.0 + 1e-12 for x in offers)
        assert all(x.eig_bits >= -1e-12 for x in rows)
        asked = st.asked_by(focal)
        assert all(x.action.question_id not in asked for x in qs)
        if st.at_deadline:
            seen_deadline = True
            assert [a.kind for a in acts] == ["ACCEPT", "WALK"]
        if len(offers) >= 2:
            seen_two_offers = True
            assert offers[0].tag == "SAFE"
        # no-question menu is the same minus the questions
        assert [x.key for x in menu.rows(st, b, include_questions=False)] == [x.key for x in rows if x.action.kind != ASK]
    assert seen_deadline and seen_two_offers and seen_plan


def test_restricted_oracle_is_bounded_by_full_and_plan_makes_noask_costless():
    for H in (3, 4):
        cfg = LiveConfig(horizon=H)
        for focal in ("buyer", "seller"):
            rk = ALL_RANKINGS[2]
            kernel = T1Kernel(other(focal), cfg, PARAMS)
            full_noask = Oracle(cfg, focal, rk, kernel, False)
            full_ask = Oracle(cfg, focal, rk, kernel, True)
            menu = Menu(cfg, focal, rk, kernel, plan_oracle=full_noask, plan_ask_oracle=full_ask)
            for opener in ("buyer", "seller"):
                fa = full_ask.root_value(opener)
                fn = full_noask.root_value(opener)
                ma = Oracle(cfg, focal, rk, kernel, True, menu=menu.actions(True)).root_value(opener)
                mn = Oracle(cfg, focal, rk, kernel, False, menu=menu.actions(False)).root_value(opener)
                assert ma <= fa + 1e-9 and mn <= fn + 1e-9
                assert abs(mn - fn) < 1e-9                  # PLAN: the no-ask planner is never handicapped
                assert abs(ma - fa) < 1e-9                  # PLAN_ASK / PLAN_Q: nor is the ASK planner
                assert ma >= mn - 1e-9                      # Delta_menu >= 0
                assert fa >= fn - 1e-9


def test_exact_ask_arm_attains_menu_ceiling_and_noask_never_asks():
    focal, rk, opener = "buyer", ALL_RANKINGS[0], "buyer"
    cell = Cell(CFG, PARAMS, focal, rk, "shortlist")
    arms = arms_for(float(CFG.reservation))
    v_ask, v_noask = cell.ask.root_value(opener), cell.noask.root_value(opener)
    rng = random.Random(7)
    nets = {"exact_ask": [], "exact_noask": []}
    for arm in nets:
        for ep in range(900):
            opp = ALL_RANKINGS[ep % 6]
            st, b = new_game(CFG, opener), bl.uniform()
            while st.terminal is None:
                if st.turn == focal:
                    rows = cell.menu.rows(st, b, include_questions=(arm == "exact_ask"))
                    rows_ask = cell.menu.rows(st, b, True)
                    qa = {r.key: cell.ask.q_value(st, b, r.action) for r in rows_ask}
                    qn = {r.key: cell.noask.q_value(st, b, r.action) for r in rows_ask if r.action.kind != ASK}
                    a = arms[arm](rows, qa, qn, rng)
                    if arm == "exact_noask":
                        assert a.kind != ASK
                    if a.kind == ASK:
                        ans = cell.kernel.sample_answer(a.question_id, opp, rng)
                        b = bl.posterior(b, bl.answer_likelihoods(cell.kernel, a.question_id)[ans])
                        st = apply(st, a, ans)
                    else:
                        st = apply(st, a)
                else:
                    a = cell.kernel.sample(st, opp, rng)
                    b = bl.posterior(b, bl.action_likelihoods(cell.kernel, st)[a])
                    st = apply(st, a)
            from .protocol_live import gain, asks_by
            g = float(gain(focal, rk, st.terminal, CFG.reservation)) - float(CFG.ask_cost) * asks_by(st, focal)
            nets[arm].append(g)
    for arm, v in (("exact_ask", v_ask), ("exact_noask", v_noask)):
        xs = nets[arm]
        m = sum(xs) / len(xs)
        se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))
        assert abs(m - v) <= 4 * se + 1e-9, (arm, m, v, se)


def test_runner_smoke_records_are_complete_and_normalised():
    with tempfile.TemporaryDirectory() as d:
        run_main(["--horizon", "3", "--types", "1", "--opp-types", "0,4", "--roles", "seller", "--openers", "buyer,seller",
                  "--repeats", "2", "--arms", "exact_ask,exact_noask,greedy_eig,random,jev_ask_eig,jev_noask_plain",
                  "--skip-ceilings", "--out", d])
        recs = [json.loads(l) for l in gzip.open(os.path.join(d, "runs.jsonl.gz"), "rt")]
        summ = json.load(open(os.path.join(d, "summary.json")))
        assert summ["error_stubs"] == 0 and len(recs) == 6 * 2 * 2 * 2
        for r in recs:
            assert "error" not in r
            assert r["u_max"]["float"] > float(CFG.reservation)
            assert abs(r["normalized_net_gain"]["float"] - r["net_gain"]["float"] / (r["u_max"]["float"] - float(CFG.reservation))) < 1e-12
            assert r["n_decisions"] == len(r["decisions"]) >= 1
            for dd in r["decisions"]:
                keys = {m["key"] for m in dd["menu"]}
                assert dd["chosen"] in keys
                assert set(dd["q_ask"]) >= keys and all(k in dd["q_ask"] for k in keys)
                if r["arm"] in ("exact_noask", "jev_noask_plain"):
                    assert dd["chosen_kind"] != ASK and not any(m["kind"] == ASK for m in dd["menu"])
                if r["arm"].startswith("jev"):
                    assert dd["jev"]["well_formed"] and dd["jev"]["chosen_position"] == 0
            if r["arm"] == "exact_ask":
                assert r["total_decision_regret"] <= 1e-9
