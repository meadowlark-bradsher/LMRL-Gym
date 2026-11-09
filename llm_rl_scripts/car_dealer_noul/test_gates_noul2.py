"""Gates from the pilot (``gates_noul2``): thresholds, the blind-judge parser, a full replay of an offline run into a
realizer packet whose sealed key a key-reading judge recovers exactly, the clairvoyant contrast, the rendered note, and —
when the pilot result dir is present — regeneration of the pinned gate files (protocol shares, Δ_clair, the sealed key)."""

from __future__ import annotations

import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_live.protocol_live import LiveConfig

from . import arms_noul2 as a2
from . import gates_noul2 as gn
from . import kernel_fit as kf
from . import run_noul2 as rn
from .opponent_llm import Completion, describe

CAL = Path("results/noul_bank_20261108/calibration_noul2.json")
PILOT = Path("results/noul2_pilot_20261108")
CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))
needs_cal = pytest.mark.skipif(not CAL.exists(), reason="frozen Stage 1 calibration file not present")
needs_pilot = pytest.mark.skipif(not (PILOT / "gates" / "clairvoyant.json").exists(), reason="pilot gate files not present")


def _row(brief, calls, well_formed, defaulted, asked=False):
    return {"brief": brief, "opponent_asked": asked, "realizer_calls": [{"ok": True, "fallback": False, "retried": False}],
            "opponent_raw": [], "opponent": [{"kind": "move", "well_formed": i < well_formed, "retried": False, "defaulted": i < defaulted}
                                             for i in range(calls)]}


def test_protocol_thresholds():
    rows = [_row("candid", 100, 100, 0), _row("guarded", 100, 99, 0), _row("adversarial", 100, 98, 2)]
    rep = gn.protocol(rows)
    assert rep["briefs"]["candid"]["passes"] and rep["briefs"]["guarded"]["passes"]
    assert not rep["briefs"]["adversarial"]["passes"] and not rep["passes"]
    assert rep["briefs"]["guarded"]["well_formed_share"] == pytest.approx(0.99)


def test_parse_judgement():
    assert gn.parse_judgement("DECISION: AB\nADDED: none", ["A", "AB"]) == ("AB", "")
    assert gn.parse_judgement("DECISION: A\nADDED: None.", ["A"]) == ("A", "")
    assert gn.parse_judgement("DECISION: A\nADDED: \"or a longer warranty if you prefer\"", ["A"]) == ("A", "\"or a longer warranty if you prefer\"")
    assert gn.parse_judgement("DECISION: Z\nADDED: none", ["A"]) == (None, "")
    assert gn.parse_judgement("I think it is A.", ["A"]) == (None, None)


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    if not CAL.exists():
        pytest.skip("frozen Stage 1 calibration file not present")
    out = tmp_path_factory.mktemp("noul2_gates")
    jev = JevClient(MockJevBackend(noul_fn=lambda state, qid, instr: 0.5))
    s = rn.run(CFG, ["plan_noask", "plan_reveal"], ["guarded"], out, 1, 7, opponent_mode="kernel", realizer_mode="template",
               calibration=a2.Calibration(CAL), jev_client=jev, sonnet_client=rn.FakeSonnet(), roles=("buyer", "seller"),
               openers=("buyer", "seller"), types=[0, 1], opp_types=[0, 3], workers=1)
    assert s["error_stubs"] == 0 and s["episodes_written"] == 2 * 2 * 2 * 2 * 2
    return out


def test_offline_run_replays_into_a_packet_the_key_recovers(offline_run):
    rows = gn.load_rows(offline_run)
    moves, skipped = gn.replay_focal_moves(rows, kf.load_records(str(offline_run / "opponent_moves.jsonl")), CFG)
    assert skipped == Counter() and len(moves) == sum(r["n_decisions"] for r in rows)
    assert all(m["action"] in m["legal"] for m in moves)
    built = gn.build_packet(moves, n_per_role=6, seed=3)
    assert len(built["packet"]) == 12 and {k["role"] for k in built["key"]} == {"buyer", "seller"}
    for row, k in zip(built["packet"], built["key"]):
        # the packet never carries the chosen move; the key's letter points at its description in the packet's list
        assert "action" not in row and k["letter"] in row["labels"]
        assert f"  {k['letter']}. {describe(kf.action_from_key(k['action']))}" in row["moves"]
    # a judge that always recovers the key's move passes; one that never does fails
    good = [{"item": k["item"], "letter": k["letter"], "added": ""} for k in built["key"]]
    rep = gn.realizer_report(built["key"], good)
    assert rep["passes"] and all(v["recovery"] == 1.0 and v["added_commitments"] == 0 for v in rep["roles"].values())
    bad = [{"item": k["item"], "letter": next(L for L in row["labels"] if L != k["letter"]), "added": ""}
           for row, k in zip(built["packet"], built["key"])]
    rep = gn.realizer_report(built["key"], bad)
    assert not rep["passes"] and all(v["recovery"] == 0.0 for v in rep["roles"].values())
    leak = [{**g, "added": "or a longer warranty"} if g["item"] == 0 else g for g in good]
    assert not gn.realizer_report(built["key"], leak)["passes"]
    # the offline fake judge runs end to end and logs every reply
    labels = gn.judge(built["packet"], gn.FakeJudge(), offline_run / "judge_raw.jsonl")
    assert len(labels) == 12 and all(l["letter"] in row["labels"] for l, row in zip(labels, built["packet"]))
    assert len((offline_run / "judge_raw.jsonl").read_text().splitlines()) == 12


def test_clairvoyant_and_render_on_offline_run(offline_run):
    rows = gn.load_rows(offline_run)
    ceil = json.load((offline_run / "ceilings.json").open())
    rep = gn.clairvoyant(rows, ceil, B=200, seed=0)
    assert rep["n_clusters"] >= 2 and isinstance(rep["point"], float) and len(rep["ci95"]) == 2
    assert rep["exact_delta_full_Y_under_assumed_kernel"] is not None and rep["y_scale"] > 0
    g = gn.gates_dir(offline_run)
    gn._dump(gn.protocol(rows), g / "protocol.json")
    gn._dump(rep, g / "clairvoyant.json")
    text = gn.render(offline_run)
    assert "PENDING" in text and "6 clairvoyant gap" in text and "4 kernel fit" not in text   # no kernel file written here
    assert f"{rep['point']:.4f}" in text


@needs_pilot
def test_pilot_gate_files_regenerate():
    rows = gn.load_rows(PILOT)
    cfg = gn.config_of(PILOT)
    P = json.load((PILOT / "gates" / "protocol.json").open())
    assert gn.protocol(rows) == P and P["passes"]
    C = json.load((PILOT / "gates" / "clairvoyant.json").open())
    rep = gn.clairvoyant(rows, json.load((PILOT / "ceilings.json").open()), C["B"], C["seed"])
    assert rep["point"] == C["point"] and rep["ci95"] == C["ci95"] and C["passes"] and C["point"] >= gn.DELTA
    moves, skipped = gn.replay_focal_moves(rows, kf.load_records(str(PILOT / "opponent_moves.jsonl")), cfg)
    assert skipped == Counter()
    key = json.load((PILOT / "gates" / "realizer_key.json").open())
    assert gn.build_packet(moves)["key"] == key                       # the sealed key regenerates from the run


@needs_pilot
def test_pilot_kernel_wide_counts_regenerate():
    """The widened gate 4 file (DESIGN_NOUL2 §3.3a): the split counts and the floor-free likelihoods regenerate
    exactly from the stored fitted parameters, the widened fit clears the gate in every brief, and it leaves no
    proposal outside its support."""
    W = json.load((PILOT / "gates" / "kernel_wide.json").open())
    cfg = gn.config_of(PILOT)
    records = kf.load_records(str(PILOT / "opponent_moves.jsonl"))
    for b in gn.BRIEFS:
        k = W["briefs"][b]
        prepared = kf.prepare([r for r in records if r["brief"] == b], cfg)
        wide = gn._params_from_json(k["fitted_all"])
        t1fit = gn._params_from_json(k["t1_family_fit"])
        assert gn.split_breakdown(prepared, kf.REFERENCE, cfg) == k["split"]["T1"]
        assert gn.split_breakdown(prepared, t1fit, cfg) == k["split"]["t1_family_fit"]
        assert gn.split_breakdown(prepared, wide, cfg) == k["split"]["wide_fit"]
        assert gn.ll_on_t1_support(prepared, wide, cfg) == k["on_t1_support"]["wide_fit"]
        assert k["adopt_fit"] and k["support_fitted"]["PROPOSE"]["zero"] == 0 and k["support_fitted"]["ASK"]["zero"] == 0
