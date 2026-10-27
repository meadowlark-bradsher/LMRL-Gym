"""Exactness checks for Binned Car Dealer, Condition B.

Pure stdlib; run with ``PYTHONPATH=. python3 -m pytest llm_rl_scripts/car_dealer_binned``
or directly. The identities checked here are what make the reference policy a
reference: menu determinism, exact Bayes updates, the textbook EIG values of the
question bank, V_full >= V_menu, zero decision regret for the exact arm, zero
EIG regret for the greedy arm, and the equality

    mean over a uniform opponent of the exact arm's realised utility == V(1, uniform)

which holds as an identity of Fractions, not approximately.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS, utility

from llm_rl_scripts.car_dealer_binned import beliefs
from llm_rl_scripts.car_dealer_binned.arms import ARMS
from llm_rl_scripts.car_dealer_binned.env import (
    ACCEPT,
    ASK,
    OFFER,
    Config,
    ScriptedOpponent,
    annotate,
    initial_state,
    menu,
    option_likelihoods,
    other_role,
    transition,
)
from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_binned.reference import ExactReference, regrets

CFG = Config()


def _h(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def test_bank_shape_and_textbook_eig():
    assert len(BANK) == 9
    b = beliefs.uniform()
    assert abs(beliefs.entropy_bits(b) - math.log2(6)) < 1e-12
    for q in BANK:
        yes = sum(q.truthful_yes)
        assert 0 < yes < 6
        eig = beliefs.expected_information_gain(b, q.likelihoods(Fraction(0)))
        if q.id.startswith("above"):
            assert yes == 3 and abs(eig - 1.0) < 1e-12
        else:
            assert yes == 2 and abs(eig - _h(1 / 3)) < 1e-12
    # three ABOVE answers identify the type
    b2 = b
    target = ALL_RANKINGS[4]
    for q in BANK:
        if q.id.startswith("above"):
            b2 = beliefs.update(b2, q.likelihoods(Fraction(0))[q.truthful_answer(target)])
    assert beliefs.support(b2) == (target,)


def test_offer_eig_is_binary_entropy_of_acceptor_share():
    s = initial_state("buyer", ALL_RANKINGS[0])
    for d in ALL_DEALS:
        liks = option_likelihoods(type("O", (), {"kind": OFFER, "target": d.id})(), s, CFG)
        share = sum(1 for th in ALL_RANKINGS if utility("seller", th, d) >= CFG.reservation) / 6
        eig = beliefs.expected_information_gain(s.belief, liks)
        expect = 0.0 if share in (0, 1) else _h(share)
        assert abs(eig - expect) < 1e-12


def test_menu_shape_and_determinism():
    for role in ("buyer", "seller"):
        for own in ALL_RANKINGS:
            s = initial_state(role, own)
            m = menu(s, CFG)
            assert len(m) <= CFG.n_questions + CFG.n_offers
            assert [o.kind for o in m][: CFG.n_questions] == [ASK] * CFG.n_questions
            assert all(o.kind == OFFER for o in m[CFG.n_questions:])
            assert m == menu(s, CFG)
            assert len({o.key for o in m}) == len(m)
            final = type(s)(role, own, CFG.final_t, s.belief, ())
            assert all(o.kind == OFFER for o in menu(final, CFG))
            assert 1 <= len(menu(final, CFG)) <= CFG.n_offers


def test_offer_candidates_are_own_individually_rational():
    for regime in ("easy", "hard"):
        cfg = Config(regime=regime)
        for role in ("buyer", "seller"):
            for own in ALL_RANKINGS:
                s = initial_state(role, own)
                for o in menu(s, cfg):
                    if o.kind == OFFER:
                        d = ALL_DEALS[[x.id for x in ALL_DEALS].index(o.target)]
                        assert utility(role, own, d) >= cfg.reservation


def test_scripted_opponent_is_truthful_and_ir():
    for own in ALL_RANKINGS:
        for opp in ALL_RANKINGS:
            s = initial_state("buyer", own)
            o = ScriptedOpponent("seller", opp, CFG)
            for q in BANK:
                opt = type("O", (), {"kind": ASK, "target": q.id})()
                assert o.respond(opt, "buyer") == q.truthful_answer(opp)
            for d in ALL_DEALS:
                opt = type("O", (), {"kind": OFFER, "target": d.id})()
                ir = utility("seller", opp, d) >= CFG.reservation
                assert (o.respond(opt, "buyer") == ACCEPT) == ir


def test_transition_updates_exactly_and_terminates():
    s = initial_state("seller", ALL_RANKINGS[1])
    m = menu(s, CFG)
    ask = m[0]
    nxt, term = transition(s, ask, "yes", CFG)
    assert term is None and nxt.t == 2 and ask.target in nxt.asked
    assert sum(nxt.belief) == 1
    assert len(beliefs.support(nxt.belief)) < 6
    offer = [o for o in m if o.kind == OFFER][0]
    nxt2, term2 = transition(s, offer, ACCEPT, CFG)
    assert nxt2 is None and term2 == utility("seller", ALL_RANKINGS[1], ALL_DEALS[[d.id for d in ALL_DEALS].index(offer.target)])


def _play(arm, role, own, opp, ref, seed=0):
    rng = random.Random(seed)
    s = initial_state(role, own)
    o = ScriptedOpponent(other_role(role), opp, CFG, random.Random(seed))
    d_reg = Fraction(0)
    ig_reg = 0.0
    asked = 0
    while True:
        rows = annotate(s, CFG)
        q = ref.q_values(s)
        key = ARMS[arm](s, rows, q, rng)
        opt = next(x for x in menu(s, CFG) if x.key == key)
        dr, _ = regrets(q, key)
        d_reg += dr
        ig_reg += max(r["eig_bits"] for r in rows) - next(r for r in rows if r["key"] == key)["eig_bits"]
        asked += opt.kind == ASK
        out = o.respond(opt, role)
        nxt, term = transition(s, opt, out, CFG)
        if nxt is None:
            return term, d_reg, ig_reg, asked
        s = nxt


def test_reference_identities_all_cells():
    for role in ("buyer", "seller"):
        for own in ALL_RANKINGS:
            ref = ExactReference(CFG, role, own)
            full = ExactReference(CFG, role, own, full_menu=True)
            v = ref.prior_value()
            assert full.prior_value() >= v >= CFG.reservation
            # exact arm: zero decision regret, and realised mean == V exactly
            total = Fraction(0)
            for opp in ALL_RANKINGS:
                u, dreg, _, _ = _play("exact", role, own, opp, ref)
                assert dreg == 0
                total += u
            assert total / 6 == v
            # greedy: zero EIG regret; myopic: never asks; random: regrets non-negative
            for opp in ALL_RANKINGS:
                _, _, igr, _ = _play("greedy_eig", role, own, opp, ref)
                assert igr < 1e-12
                _, _, _, asked = _play("myopic", role, own, opp, ref)
                assert asked == 0
                _, dreg, igr, _ = _play("random", role, own, opp, ref, seed=3)
                assert dreg >= 0 and igr >= -1e-12


def test_noisy_answers_keep_full_support_and_dp_runs():
    cfg = Config(answer_noise=Fraction(1, 10))
    s = initial_state("buyer", ALL_RANKINGS[0])
    q = menu(s, cfg)[0]
    nxt, _ = transition(s, q, "yes", cfg)
    assert len(beliefs.support(nxt.belief)) == 6
    ref = ExactReference(cfg, "buyer", ALL_RANKINGS[0])
    v = ref.prior_value()
    assert cfg.reservation <= v <= 1


def test_hard_regime_infeasible_cells_are_solved():
    cfg = Config(regime="hard")
    ref = ExactReference(cfg, "buyer", ALL_RANKINGS[0])
    assert ref.prior_value() >= cfg.reservation


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                failures += 1
                print("FAIL", name, e)
    sys.exit(1 if failures else 0)
