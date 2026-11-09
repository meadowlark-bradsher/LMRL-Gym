"""Offline gates for CD-NOUL-2 Phase 0: the calibration constants, the per-question hedge plumbing
(parity with the global hedge when uniform), the offer likelihood, the LLM opponent's protocol against
scripted and kernel-driven clients, and the realizer's round-trip check. No network."""

from __future__ import annotations

import random
from fractions import Fraction
from pathlib import Path

import pytest

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, ATTRIBUTES
from llm_rl_scripts.car_dealer_live import filters_mis as fm
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.oracle import Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, PROPOSE, Action, LiveConfig, apply, legal_actions, new_game

from . import calibrate_noul as cn
from . import decode as dc
from . import opponent_llm as op
from . import realizer as rz

BANK_DIR = Path("results/noul_bank_20261108")
needs_bank = pytest.mark.skipif(not (BANK_DIR / "matrix_sonnet.jsonl").exists(), reason="frozen bank + matrices absent")
CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))
QIDS = [q.id for q in BANK]


# ---- calibration -----------------------------------------------------------------------------------

def test_calibrated_read_identity_fixed_point_monotone():
    for p in (0.1, 0.3, 0.7, 0.9):
        assert abs(dc.calibrated_p_yes(p, 1.0) - p) < 1e-9
    for T in (0.3, 1.0, 5.0):
        assert abs(dc.calibrated_p_yes(0.5, T) - 0.5) < 1e-12
    assert dc.calibrated_p_yes(0.9, 2.0) > 0.9 > dc.calibrated_p_yes(0.9, 0.5)
    ps = [dc.calibrated_p_yes(x / 20, 3.0) for x in range(1, 20)]
    assert ps == sorted(ps)


def _rows(ps, truths, regime="plain", qid="top:price"):
    return [{"p_yes": p, "truthful": t, "regime": regime, "question_id": qid} for p, t in zip(ps, truths)]


def test_temperature_fit_sharpens_an_underconfident_always_right_decoder():
    rows = _rows([0.9] * 50 + [0.1] * 50, ["yes"] * 50 + ["no"] * 50)
    T = cn.fit_temperature(rows)
    assert T > 5.0 and cn.nll_bits(rows, T) < cn.nll_bits(rows, 1.0)
    assert cn.accuracy(rows) == 1.0 and cn.wrong_confident(rows, T) == 0


def test_temperature_fit_recovers_one_for_a_calibrated_decoder():
    rng = random.Random(3)
    ps = [rng.uniform(0.05, 0.95) for _ in range(6000)]
    truths = ["yes" if rng.random() < p else "no" for p in ps]
    T = cn.fit_temperature(_rows(ps, truths))
    assert 0.85 < T < 1.2, T


def test_safe_temperature_caps_before_the_first_confident_wrong_read():
    # one wrong-leaning read at 0.7 becomes confident-wrong (> 0.9) once T passes logit(0.9)/logit(0.7)
    rows = _rows([0.9] * 20 + [0.7], ["yes"] * 20 + ["no"], regime="oblique")
    t = cn.safe_temperature(rows)
    import math
    t_star = math.log(0.9 / 0.1) / math.log(0.7 / 0.3)
    assert abs(t - t_star) < 1e-6 and cn.wrong_confident(rows, t) == 0 and cn.wrong_confident(rows, t + 1e-3) == 1


def test_platt_fit_removes_a_pure_bias():
    # a decoder that is right but shifted toward NO: p = 0.35 when the truth is yes, 0.05 when no
    rows = _rows([0.35] * 60 + [0.05] * 60, ["yes"] * 60 + ["no"] * 60)
    T, b = cn.fit_platt(rows)
    assert b > 0.5 and cn.nll_bits(rows, T, b) < cn.nll_bits(rows, cn.fit_temperature(rows))
    assert all(cn.read(0.35, T, b) > 0.5 for _ in [0]) and cn.read(0.05, T, b) < 0.5


def test_a_q_table_and_noise_map():
    rng = random.Random(1)
    rows = []
    for rg in cn.REGIMES:
        for q in QIDS:
            for _ in range(10):
                p = rng.uniform(0.0, 1.0)
                rows.append({"p_yes": p, "truthful": "yes" if p > 0.5 else "no", "regime": rg, "question_id": q})
    tab = cn.a_q_table(rows, 1.0)
    assert set(tab) == set(cn.REGIMES) and all(set(tab[rg]) == set(QIDS) for rg in cn.REGIMES)
    assert all(0.0 <= v <= cn.MAX_NOISE for rg in tab for v in tab[rg].values())
    nm = cn.as_noise_map(tab["plain"])
    assert [q for q, _ in nm] == QIDS and all(isinstance(a, Fraction) and Fraction(0) <= a < Fraction(1, 2) for _, a in nm)
    # a sharper decoder (T > 1) has a smaller expected read error
    assert cn.expected_error(rows, 3.0) < cn.expected_error(rows, 1.0)


@needs_bank
def test_calibration_on_the_frozen_bank():
    cal = cn.calibration(BANK_DIR)
    assert set(cal["decoders"]) == set(cn.DECODERS) and cal["questions"] == QIDS
    for d, rec in cal["decoders"].items():
        assert rec["n"] == 4320
        ll = rec["logloss_bits"]
        assert ll["pooled"]["calibrated"] <= ll["pooled"]["raw"] + 1e-9
        assert rec["accuracy"]["plain"] == 1.0
        for rg in cn.REGIMES:
            assert set(rec["a_q"][rg]) == set(QIDS) and all(v is not None for v in rec["a_q"][rg].values())
        assert cn.T_MIN < rec["temperature_pooled"] < cn.T_MAX
        c = rec["candidates"]
        assert set(c) == {"raw", "pooled", "oblique_fit", "safe", "platt", "platt_oblique"} and c["raw"]["temperature"] == 1.0
        assert c["platt"]["logloss_bits"]["pooled"] <= c["pooled"]["logloss_bits"]["pooled"] + 1e-9
        assert c["safe"]["wrong_confident"]["oblique"] == 0 and c["safe"]["temperature"] >= 1.0
        assert len(rec["confident_wrong_ids_pooled_oblique"]) == c["pooled"]["wrong_confident"]["oblique"]


# ---- per-question hedge plumbing -------------------------------------------------------------------

def test_live_config_and_filter_spec_json_unchanged_by_default():
    assert set(LiveConfig().to_json()) == {"env", "regime", "reservation", "horizon", "ask_cost", "answer_noise"}
    assert set(fm.FilterSpec("x").to_json()) == {"name", "update", "eta", "tau", "answer_noise"}


def test_uniform_per_question_noise_is_the_global_hedge():
    a = Fraction(1, 5)
    cfg_g = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0), answer_noise=a)
    cfg_q = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0), answer_noise=Fraction(0),
                       answer_noise_by_q=tuple((q, a) for q in QIDS))
    kg, kq = T1Kernel("seller", cfg_g, KernelParams()), T1Kernel("seller", cfg_q, KernelParams())
    for q in QIDS:
        for r in ALL_RANKINGS:
            assert kg.answer_distribution(q, r) == kq.answer_distribution(q, r)
    assert cfg_q.to_json()["answer_noise_by_q"][QIDS[0]] == "1/5"
    og = Oracle(CFG, "buyer", ALL_RANKINGS[0], kg, ask_allowed=True)
    oq = Oracle(CFG, "buyer", ALL_RANKINGS[0], kq, ask_allowed=True)
    assert og.root_value("buyer") == oq.root_value("buyer")
    sg = fm.FilterSpec("g", eta=0.0, answer_noise=0.2)
    sq = fm.FilterSpec("q", eta=0.0, answer_noise=0.0, answer_noise_by_q=tuple((q, 0.2) for q in QIDS))
    truthful = T1Kernel("seller", CFG, KernelParams())
    for q in QIDS:
        for ans in ("yes", "no"):
            assert fm.answer_likelihood(truthful, q, ans, sg) == fm.answer_likelihood(truthful, q, ans, sq)


def test_per_question_noise_applies_only_where_set():
    cfg = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0), answer_noise_by_q=(("top:price", Fraction(3, 10)),))
    k = T1Kernel("seller", cfg, KernelParams())
    r = ALL_RANKINGS[0]
    dist = k.answer_distribution("top:price", r)
    assert sorted(dist.values()) == [0.3, 0.7]
    assert set(k.answer_distribution("top:warranty", r).values()) == {1.0}
    assert cfg.noise_for("bottom:delivery") == Fraction(0)


# ---- offer likelihood ------------------------------------------------------------------------------

def test_offer_item_type_vector_and_prompt_hashes():
    target = "price>warranty>delivery"
    backend = MockJevBackend(noul_fn=lambda state, qid, instr: 0.8 if qid.endswith(target) else 0.2)
    item = dc.offer_item(JevClient(backend), "buyer", "PROPOSE P0-W1-D0", "I need it cheap, the rest can wait",
                         deal_label="price=low, warranty=standard, delivery=normal")
    v = item["type_vector"]
    assert max(v) == 1.0 and v[ALL_RANKINGS.index(("price", "warranty", "delivery"))] == 1.0
    assert all(abs(x - 0.25) < 1e-12 for i, x in enumerate(v) if ALL_RANKINGS[i] != ("price", "warranty", "delivery"))
    assert dc.prompt_hash() == "aea24a833e5c"          # Stage 1's registered prompt hash is untouched
    assert len(dc.offer_prompt_hash()) == 12 and dc.offer_prompt_hash() != dc.prompt_hash()


# ---- the LLM opponent ------------------------------------------------------------------------------

def test_parse_reply_and_answer():
    assert op.parse_reply("I think...\nDECISION: B\nSAY: Let's talk price.", "ABC") == ("B", "Let's talk price.")
    assert op.parse_reply("`DECISION: (C)`\n`SAY: Fine.`", "ABC") == ("C", "Fine.")
    assert op.parse_reply("DECISION: Q\nSAY: hi", "ABC") == (None, "hi")
    assert op.parse_reply("DECISION: A\nDECISION: B\nSAY: two lines\nof speech", "ABC") == ("B", "two lines\nof speech")
    assert op.parse_reply("just words", "ABC") == (None, "")
    assert op.parse_reply("DECISION: AB\nSAY: two letters", ["A", "B", "AB"]) == ("AB", "two letters")
    assert op.parse_reply("DECISION: AB\nSAY: x", "AB") == (None, "x")          # a string of single letters has no 'AB'
    from llm_rl_scripts.car_dealer_live.protocol_live import legal_actions, new_game
    letters = op.lettered(legal_actions(new_game(CFG, "seller"), "seller"), None)
    assert len(letters) > 26 and "AA" in letters
    assert op.parse_answer('SAY: "Price matters to me."') == "Price matters to me."
    assert op.parse_answer("Honestly, I'd rather not say.") == "Honestly, I'd rather not say."


def test_briefs_render_and_hash_is_stable():
    for name in op.BRIEFS:
        text = op.brief(name, "buyer", ("price", "warranty", "delivery"), CFG)
        assert "price > warranty > delivery" in text and "2/5" in text and "4 opportunities" in text
    assert "never state anything false" in op.brief("guarded", "seller", ALL_RANKINGS[3], CFG)
    assert "mislead" in op.brief("adversarial", "seller", ALL_RANKINGS[3], CFG)
    assert len(op.brief_hash()) == 12 and op.brief_hash() == op.brief_hash()
    for role in ("buyer", "seller"):
        sc = op.level_scores(role)
        for a in ATTRIBUTES:
            assert set(sc[a].values()) == {Fraction(0), Fraction(1, 2), Fraction(1)}, (role, a, sc[a])


def test_llm_opponent_move_well_formed_and_fallback():
    state = new_game(CFG, "seller")
    fallback = T1Kernel("seller", CFG, KernelParams())
    opp = op.LLMOpponent(op.ScriptedClient(["DECISION: A\nSAY: Hello there."]), "guarded", "seller", ALL_RANKINGS[0], CFG,
                         fallback=fallback, rng=random.Random(0), shuffle=False)
    action, say = opp.move(state)
    info = opp.last_info
    assert info.well_formed and not info.retried and not info.defaulted and info.letter == "A"
    assert action.key == info.letter_to_key["A"] and say == "Hello there." and opp.said[0] == "Hello there."
    assert action in legal_actions(state, "seller")
    opp2 = op.LLMOpponent(op.ScriptedClient(["nonsense", "still nonsense"]), "candid", "seller", ALL_RANKINGS[0], CFG,
                          fallback=fallback, rng=random.Random(0))
    action2, say2 = opp2.move(state)
    assert opp2.last_info.defaulted and opp2.last_info.retried and not opp2.last_info.well_formed
    assert action2 in legal_actions(state, "seller") and say2 == "" and len(opp2.calls) == 2


def test_llm_opponent_answer_and_fallback():
    state = new_game(CFG, "buyer")
    fallback = T1Kernel("seller", CFG, KernelParams())
    opp = op.LLMOpponent(op.ScriptedClient(["SAY: I'd rather not get into that."]), "guarded", "seller", ALL_RANKINGS[0], CFG,
                         fallback=fallback, rng=random.Random(0))
    assert opp.answer(state, "top:price") == "I'd rather not get into that." and opp.last_info.kind == "answer"
    opp2 = op.LLMOpponent(op.ScriptedClient(["", ""]), "candid", "seller", ALL_RANKINGS[0], CFG, fallback=fallback, rng=random.Random(0))
    text = opp2.answer(state, "top:price")
    assert opp2.last_info.defaulted and text in ("yes", "no")


def test_kernel_driven_opponent_plays_a_full_game_offline():
    rng = random.Random(7)
    seller_ranking = ALL_RANKINGS[4]
    kernel = T1Kernel("seller", CFG, KernelParams())
    client = op.KernelClient(kernel, seller_ranking, rng, CFG)
    opp = op.KernelOpponent(client, "guarded", "seller", seller_ranking, CFG, fallback=kernel, rng=rng)
    focal = T1Kernel("buyer", CFG, KernelParams())
    state = new_game(CFG, "buyer")
    moves = 0
    while state.terminal is None:
        if state.turn == "buyer":
            act = focal.sample(state, ALL_RANKINGS[0], rng)
            state = apply(state, act)
        else:
            act, say = opp.move(state)
            assert opp.last_info.well_formed and say
            state = apply(state, act)
            moves += 1
    assert moves >= 1 and state.terminal is not None
    assert all(idx in opp.said for idx, e in enumerate(state.events) if e.actor == "seller")


def test_state_text_shows_both_sides_words():
    state = new_game(CFG, "seller")
    state = apply(state, Action(PROPOSE, deal_id="P2-W0-D0"))
    state = apply(state, Action(PROPOSE, deal_id="P0-W2-D2"))
    text = op.state_text(state, "seller", CFG, said={0: "Take it or leave it."}, heard={1: "No thanks, here is mine."})
    assert 'You said: "Take it or leave it."' in text and 'They said: "No thanks, here is mine."' in text
    assert "[P2-W0-D0]" in text and "[P0-W2-D2]" in text


# ---- the realizer -----------------------------------------------------------------------------------

def test_template_text_each_kind():
    assert op.BANK_BY_ID["top:price"].text == rz.template_text(Action(ASK, question_id="top:price"))
    assert rz.template_text(Action(PROPOSE, deal_id="P1-W1-D1")).startswith("I propose ")
    for kind, word in (("ACCEPT", "accept"), ("REJECT", "accept"), ("HOLD", "stands"), ("WALK", "walking")):
        assert word in rz.template_text(Action(kind))


def test_realizer_round_trip_good_and_failures():
    state = new_game(CFG, "buyer")
    legal = legal_actions(state, "buyer")
    ask = Action(ASK, question_id="top:price")
    good = rz.Realizer(rz.EchoClient("good"), "buyer", CFG).realize(state, ask, legal, {}, {})
    assert good.ok and not good.fallback and "ask" in good.text.lower()
    bad = rz.Realizer(rz.EchoClient("bad_letter"), "buyer", CFG).realize(state, ask, legal, {}, {})
    assert not bad.ok and bad.fallback and bad.retried and bad.text == rz.template_text(ask) and "letter" in bad.reasons
    leak = rz.Realizer(rz.EchoClient("leak"), "buyer", CFG).realize(state, ask, legal, {}, {})
    assert not leak.ok and any(r.startswith("forbidden:") for r in leak.reasons)
    prop = Action(PROPOSE, deal_id="P1-W1-D1")
    od = rz.Realizer(rz.EchoClient("other_deal"), "buyer", CFG).realize(state, prop, legal, {}, {})
    assert not od.ok and "other_deal:P0-W0-D0" in od.reasons
    ok_prop = rz.Realizer(rz.EchoClient("good"), "buyer", CFG).realize(state, prop, legal, {}, {})
    assert ok_prop.ok
    assert rz.TemplateRealizer().realize(state, prop, legal, {}, {}).text == rz.template_text(prop)
    assert len(rz.realizer_hash()) == 12
