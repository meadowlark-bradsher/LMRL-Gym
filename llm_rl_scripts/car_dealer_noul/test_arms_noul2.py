"""Gates for the Stage 2 arms and runner (NEG-8): arm grammar; the calibration map and noise maps; **parity** —
``plan_ask-oracle-0`` with the scripted T1 opponent, templated text and the T1 kernel reproduces Stage 1's
``plan_ask-oracle-0`` decision for decision, and ``plan_noask`` likewise; ``plan_reveal`` never asks and is
at least as good as no-ask in expectation on a cell; an offline smoke of the runner with the kernel-driven
opponent through the DECISION:/SAY: protocol (resumable, logs opponent moves, no stubs). No network."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.protocol_live import LiveConfig

from . import arms_noul as a1
from . import arms_noul2 as a2
from . import run_noul2 as rn
from .realizer import TemplateRealizer

BANK = Path("results/noul_bank_20261108")
CAL = BANK / "calibration_noul2.json"
needs_cal = pytest.mark.skipif(not CAL.exists(), reason="calibration_noul2.json absent")
CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))


def test_arm_grammar():
    a = a2.parse_arm("plan_ask-noulcal+offer-adaptive-T1-template")
    assert (a.kind, a.decoder, a.offer, a.hedge, a.kernel, a.realizer) == ("plan_ask", "noulcal", True, "adaptive", "T1", "template")
    assert a2.parse_arm("plan_ask-regex-0.2").kernel == "fitted" and a2.parse_arm("plan_reveal").kind == "plan_reveal"
    assert a2.parse_arm("sonnet_agent").decoder == a2.AGENT_MENU_DECODER
    nc = a2.parse_arm("plan_ask-noulcal-adaptive-noclamp")
    assert nc.clamp is False and a2.parse_arm("plan_ask-noulcal-adaptive").clamp is True
    assert a2.briefs_for("plan_ask-noulcal-adaptive-noclamp", ("candid", "guarded", "adversarial")) == ["guarded"]
    for bad in ("plan_ask-regex-adaptive", "plan_ask-noul-0", "plan_ask-noul_type-0.2", "plan_ask-oracle-0.2-T2", "plan_ask-noulcal-0.2-noclamp"):
        with pytest.raises(ValueError):
            a2.parse_arm(bad)
    for name in a2.design_arms():
        a2.parse_arm(name)
    assert a2.briefs_for("plan_ask-noulcal-adaptive-T1", ("candid", "guarded", "adversarial")) == ["guarded"]
    assert a2.briefs_for("plan_ask-noul-adaptive", ("candid", "guarded", "adversarial")) == ["candid", "guarded"]
    assert a2.briefs_for("plan_noask", ("candid", "guarded", "adversarial")) == ["candid", "guarded", "adversarial"]


@needs_cal
def test_calibration_map_and_noise_maps():
    cal = a2.Calibration(CAL, a2.MapSpec("platt", 0.95))
    T, b = cal.params("noulcal")
    assert T > 1.0 and b > 0.0
    assert cal.read("noulcal", 0.5) > 0.5                      # the Platt intercept moves a deflection toward YES
    assert cal.read("noulcal", 0.999) == pytest.approx(0.95) and cal.read("noulcal", 0.001) == pytest.approx(0.05)
    assert cal.read("sonnet", 0.3) == pytest.approx(0.3) and cal.read("noul", 0.3) == pytest.approx(0.3)
    nm = cal.noise_map("noulcal", "guarded")
    assert len(nm) == 9 and all(Fraction(0) <= a < Fraction(1, 2) for _, a in nm)
    assert sum(a for _, a in cal.noise_map("noulcal", "guarded")) > sum(a for _, a in cal.noise_map("noulcal", "candid"))
    assert cal.read("noulcal", 0.999, clamp=1.0) > 0.95                      # the -noclamp arm's read is unclipped
    raw = a2.Calibration(CAL, a2.MapSpec("raw", 1.0))
    assert raw.read("noulcal", 0.2) == pytest.approx(0.2) and raw.read("noulcal", 0.999) == pytest.approx(0.999)


def _stage2(seed: int, role: str, i: int, j: int, opener: str, arm_name: str, cal):
    cell = a2.Stage2Cell(CFG, role, ALL_RANKINGS[i], fitted=REFERENCE)
    arm = a2.parse_arm(arm_name)
    rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|0")
    opp = a2.ScriptedT1Opponent(cell.planner("0", "T1")["kernel"], ALL_RANKINGS[j], rng)
    dec = a2.Decoders(cal, JevClient(MockJevBackend()), None)
    return a2.play_episode(cell, arm, "candid", opener, ALL_RANKINGS[j], rng, opp, TemplateRealizer(), dec, cal, 0)


@needs_cal
@pytest.mark.parametrize("arm1,arm2", [("plan_ask-oracle-0", "plan_ask-oracle-0-T1"), ("plan_noask", "plan_noask")])
def test_parity_with_stage1_on_scripted_t1(arm1, arm2):
    cal = a2.Calibration(CAL)
    seed, role, i = 21, "buyer", 0
    n_asks = 0
    for opener in ("buyer", "seller"):
        for j in range(len(ALL_RANKINGS)):
            r1 = _stage1_oracle(seed, role, i, j, opener, arm1)
            r2 = _stage2(seed, role, i, j, opener, arm2, cal)
            assert [d["chosen"] for d in r1["decisions"]] == [d["chosen"] for d in r2["decisions"]], (opener, j)
            assert r1["terminal"] == r2["terminal"] and r1["net_gain"] == r2["net_gain"]
            assert [d["ask_optimal_true"] for d in r1["decisions"]] == [d["ask_optimal_true"] for d in r2["decisions"]]
            n_asks += r2["questions_asked"]
    if arm1.startswith("plan_ask"):
        assert n_asks > 0


def _stage1_oracle(seed, role, i, j, opener, arm_name):
    """Stage 1's episode for the oracle decoder, with the real frozen bank if present, else a minimal stub."""
    cell = a1.NoulCell(CFG, role, ALL_RANKINGS[i])
    arm = a1.parse_arm(arm_name)
    rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|0")
    bank = None
    if arm.uses_text:
        bank = a1.TextBank(BANK) if (BANK / "matrix_sonnet.jsonl").exists() else None
        if bank is None:
            pytest.skip("frozen bank absent")
    return a1.play_episode(cell, arm, "plain" if arm.uses_text else a1.NO_TEXT_REGIME, opener, ALL_RANKINGS[j], rng,
                           a1.variant_plan(seed, role, opener, i, j, 0, bank.k if bank else 1), bank, 0)


@needs_cal
def test_plan_reveal_never_asks_and_plans_at_the_true_type():
    cal = a2.Calibration(CAL)
    for arm in ("plan_reveal", "plan_noask"):
        for j in range(len(ALL_RANKINGS)):
            r = _stage2(5, "buyer", 0, j, "buyer", arm, cal)
            assert r["questions_asked"] == 0 and r["kind"] == arm
            assert all(d["chosen_kind"] != "ASK" for d in r["decisions"])
    # the revealed planner's first offer is a best response to the true type under the assumed kernel: it never
    # differs from what the no-ask planner would choose at the one-hot belief
    cell = a2.Stage2Cell(CFG, "buyer", ALL_RANKINGS[0])
    from llm_rl_scripts.car_dealer_live.protocol_live import new_game
    own = cell.planner("0", "fitted")
    for j in range(len(ALL_RANKINGS)):
        onehot = a2._onehot(j)
        rows = own["menu"].rows(new_game(CFG, "buyer"), onehot, include_questions=False)
        q = {r.key: own["noask"].q_value(new_game(CFG, "buyer"), onehot, r.action) for r in rows}
        rec = _stage2(5, "buyer", 0, j, "buyer", "plan_reveal", cal)
        first = rec["decisions"][0]["chosen"]
        assert q[first] == pytest.approx(max(q.values()))


@needs_cal
def test_adaptive_hedge_uses_a_per_question_planner_and_the_read_in_the_filter():
    cal = a2.Calibration(CAL)
    cell = a2.Stage2Cell(CFG, "buyer", ALL_RANKINGS[0])
    nm = cal.noise_map("noulcal", "guarded")
    p = cell.planner("adaptive", "fitted", nm)
    assert p["kernel"].cfg.answer_noise_by_q == nm and p["kernel"].cfg.answer_noise == Fraction(0)
    lik = a2.answer_likelihood("top:price", 0.9, "adaptive")
    lik02 = a2.answer_likelihood("top:price", 0.9, "0.2")
    assert max(lik) == pytest.approx(0.9) and min(lik) == pytest.approx(0.1)
    assert max(lik02) < 0.9 and min(lik02) > 0.1            # the fixed hedge blunts the read


@needs_cal
def test_offline_smoke_with_kernel_opponent_is_resumable_and_logs_moves(tmp_path):
    cal = a2.Calibration(CAL)
    jev = JevClient(MockJevBackend(noul_fn=lambda state, qid, instr: 0.6 if qid == "yes" else 0.4))
    arms = ["plan_ask-noulcal+offer-adaptive", "plan_ask-regex-0.2", "plan_noask", "plan_reveal", "sonnet_agent"]
    kw = dict(opponent_mode="kernel", realizer_mode="template", calibration=cal, jev_client=jev, sonnet_client=rn.FakeSonnet(),
              roles=("buyer",), openers=("buyer", "seller"), types=[0], opp_types=[0, 3], workers=2)
    s1 = rn.run(CFG, arms, ["guarded"], tmp_path, 1, 7, **kw)
    assert s1["error_stubs"] == 0 and s1["episodes_written"] == 5 * 2 * 2 and s1["opponent_moves_logged"] > 0
    recs = [json.loads(l) for l in (tmp_path / "runs.jsonl").open()]
    assert len({r["episode_key"] for r in recs}) == len(recs)
    for r in recs:
        assert r["brief"] == "guarded" and all(o["well_formed"] for o in r["opponent"])
        assert all(c["ok"] for c in r["realizer_calls"])
    offer_arm = [r for r in recs if r["arm"] == "plan_ask-noulcal+offer-adaptive"]
    assert any(r["offer_decodes"] for r in offer_arm)
    assert all(r["questions_asked"] == 0 for r in recs if r["arm"] in ("plan_noask", "plan_reveal"))
    moves = [json.loads(l) for l in (tmp_path / "opponent_moves.jsonl").open()]
    assert moves and all(m["brief"] == "guarded" and m["role"] == "seller" for m in moves)
    s2 = rn.run(CFG, arms, ["guarded"], tmp_path, 1, 7, **kw)
    assert s2["episodes_written"] == 0 and s2["episodes_resumed"] == 20
    assert len([l for l in (tmp_path / "runs.jsonl").open()]) == 20


@needs_cal
def test_cli_refuses_confirmatory_without_tag_and_live_backends(tmp_path):
    with pytest.raises(SystemExit):
        rn.main(["--phase", "confirmatory", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        rn.main(["--phase", "pilot", "--out", str(tmp_path), "--opponent", "llm"])   # llm opponent needs the live backend
