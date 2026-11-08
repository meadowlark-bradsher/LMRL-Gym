"""Gates for the analyzer, verdict rules and renderer, on synthetic records with known structure
(no outcome of a real run is read here) plus a schema-only pass over an offline run if present."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from . import analyze_noul as az
from . import arms_noul as an
from . import render_noul_note as rn
from . import verdicts_noul as vn

SMOKE = Path("/private/tmp/claude-501/-Users-meadowlarkbradsher-workspace-repos-genai-LMRL-Gym--claude-worktrees-drl-policy-value-assessment-735eb4/"
             "0c76bc52-aafa-4325-8ad5-dca379dfee76/scratchpad/noul_smoke_run")


def _rec(arm: str, regime: str, focal: str, opener: str, i: int, j: int, k: int, y: float, asks: int, decode_p=None,
         ask_optimal=(True, False), agent=False) -> dict:
    """A minimal record with the fields the analyzer reads."""
    a = an.parse_arm(arm)
    decisions = []
    for n, opt in enumerate(ask_optimal):
        chosen = "ASK" if (asks > n) else "PROPOSE"
        dec = None
        if chosen == "ASK" and a.uses_text and decode_p is not None:
            dec = {"decoder": a.decoder, "bank_id": f"{regime}|seller|top:price|p>w>d|00", "variant": 0, "truthful": "yes",
                   "read": {"p_yes": decode_p}, "likelihood": [1.0] * 6, "p_true_before": 1 / 6, "p_true_after": decode_p, "p_true_exact_after": 0.5}
        decisions.append({"ask_legal": True, "ask_optimal_true": opt, "chosen_kind": chosen, "decision_regret_true": 0.0, "decode": dec,
                          "agent": {"chosen_position": n % 3, "well_formed": True, "retried": False,
                                    "usage": {"input_tokens": 10, "output_tokens": 2}} if agent else None})
    opp = ALL_RANKINGS[j]
    return {"arm": arm, "kind": a.kind, "decoder": a.decoder, "hedge": str(a.hedge), "regime": regime, "focal": focal, "opener": opener,
            "focal_ranking": list(ALL_RANKINGS[i]), "repeat": k, "private_truth_evaluator_only": {"opp_ranking": list(opp)},
            "decisions": decisions, "n_decisions": len(decisions), "questions_asked": asks, "proposals_made": 1,
            "terminal": {"deal_id": "d", "reason": "accept"}, "agreement": True,
            "net_gain": {"float": y * 0.6}, "gross_gain": {"float": y * 0.6}, "normalized_net_gain": {"exact": str(y), "float": y},
            "u_max": {"float": 1.0}, "total_decision_regret_true": 0.0, "zero_regret_decisions_true": len(decisions),
            "final_agent_belief": {"p_true": 0.9 if decode_p is None else decode_p, "logloss_true_bits": 0.15},
            "episode_key": f"{arm}|{regime}|{focal}|{opener}|{i}|{j}|{k}"}


def _synthetic_dir(tmp_path: Path, effects: dict) -> Path:
    """Every (arm, regime) in ``effects`` gets Y = base(cell) + effect, over 2 roles x 2 openers x 6 x 6 cells, 1 repeat."""
    out = tmp_path / "syn"
    out.mkdir(parents=True)
    rng = random.Random(0)
    base = {}
    with (out / "runs.jsonl").open("w") as f:
        for (arm, regime), eff in effects.items():
            a = an.parse_arm(arm)
            for focal in ("buyer", "seller"):
                for opener in ("buyer", "seller"):
                    for i in range(6):
                        for j in range(6):
                            b = base.setdefault((focal, opener, i, j), rng.uniform(0.2, 0.5))
                            asks = 0 if not a.asks else (2 if a.kind == "plan_ask" else 1)
                            p = None if not a.uses_text else (0.95 if a.decoder != "regex" else 0.5)
                            f.write(json.dumps(_rec(arm, regime, focal, opener, i, j, 0, b + eff, asks, p, agent=a.kind.endswith("agent"))) + "\n")
    json.dump({"env": "live-car-dealer-v0.1", "design": "CD-NOUL-1", "config": {"regime": "easy", "reservation": "2/5", "horizon": 4, "ask_cost": "0"},
               "kernel": {"alpha0": "4/5", "beta": 8.0, "p_counter": 1.0, "p_hold": 1.0, "shortlist": 3}, "bank": "bank", "agent_menu_decoder": "noul_answer",
               "registered_hedge": "1/5", "arms": sorted({k[0] for k in effects}), "regimes": ["plain", "oblique"], "repeats": 1, "seed": 1, "elapsed_s": 1.0},
              (out / "summary.json").open("w"))
    cells = {}
    for role in ("buyer", "seller"):
        for opener in ("buyer", "seller"):
            for i in range(6):
                cells[f"{role}|{opener}|{'/'.join(x[0] for x in ALL_RANKINGS[i])}"] = {"menu_ask": 0.36, "menu_noask": 0.30, "full_ask": 0.36, "full_noask": 0.30,
                                                                                         "delta_full": 0.06, "delta_menu": 0.06}
    json.dump({"cells": cells}, (out / "ceilings.json").open("w"))
    return out


EFFECTS = {
    ("plan_noask", "none"): 0.0,
    ("plan_ask-oracle-0", "plain"): 0.10, ("plan_ask-oracle-0", "oblique"): 0.10,
    ("plan_ask-noul_answer-0.2", "plain"): 0.08, ("plan_ask-noul_answer-0.2", "oblique"): 0.06,
    ("plan_ask-noul_answer-0", "plain"): 0.09, ("plan_ask-noul_answer-0", "oblique"): 0.03,
    ("plan_ask-regex-0.2", "plain"): 0.08, ("plan_ask-regex-0.2", "oblique"): 0.01,
    ("plan_ask-regex-0", "plain"): 0.08, ("plan_ask-regex-0", "oblique"): -0.02,
    ("plan_ask-noul_type-0.2", "plain"): 0.07, ("plan_ask-noul_type-0.2", "oblique"): 0.07,
    ("plan_ask-sonnet_answer-0.2", "plain"): 0.08, ("plan_ask-sonnet_answer-0.2", "oblique"): 0.065,
    ("sonnet_agent", "plain"): 0.0, ("sonnet_agent", "oblique"): -0.01,
    ("jev_agent", "plain"): -0.01, ("jev_agent", "oblique"): -0.01,
}


def test_analyzer_recovers_known_effects(tmp_path: Path):
    d = _synthetic_dir(tmp_path, EFFECTS)
    a = az.analyze(str(d), B=300, seed=1)
    assert a["episodes"] == len(EFFECTS) * 144 and a["error_stubs"] == 0
    assert a["y_scale"] == pytest.approx(0.6)
    c = a["contrasts"]["ask_value|plan_ask-noul_answer-0.2|plain"]["focal_opens"]
    assert c["point"] == pytest.approx(0.08, abs=1e-9) and c["n_clusters"] == 36
    assert c["captured_fraction"]["delta_menu"]["ceiling"] == pytest.approx(0.06 / 0.6)
    assert c["captured_fraction"]["delta_menu"]["point"] == pytest.approx(0.08 / 0.1)
    assert a["contrasts"]["decoder|plan_ask-noul_answer-0.2|oblique"]["all"]["point"] == pytest.approx(0.05, abs=1e-9)
    assert a["contrasts"]["decoder|plan_ask-noul_answer-0.2|plain"]["all"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert a["contrasts"]["hedge|noul_answer|oblique"]["all"]["point"] == pytest.approx(0.03, abs=1e-9)
    assert a["contrasts"]["type_vs_answer|1/5|oblique"]["all"]["point"] == pytest.approx(0.01, abs=1e-9)
    assert a["contrasts"]["chooser|sonnet_agent|1/5|oblique"]["all"]["point"] == pytest.approx(0.07, abs=1e-9)
    assert a["contrasts"]["text_invariance|plan_ask-oracle-0"]["all"]["point"] == pytest.approx(0.0, abs=1e-12)
    # selectivity: the synthetic planner asks at decision 0 (optimal) and 1 (not optimal)
    s = a["selectivity"]["plan_ask-noul_answer-0.2|plain"]
    assert s["p_ask_given_optimal"] == 1.0 and s["p_ask_given_not_optimal"] == 1.0 and s["difference"] == 0.0
    s2 = a["selectivity"]["sonnet_agent|plain"]
    assert s2["p_ask_given_optimal"] == 1.0 and s2["p_ask_given_not_optimal"] == 0.0 and s2["difference"] == 1.0
    # decoder diagnostics and the log-loss contrast (regex at 0.5 -> 1 bit; noul at 0.95 -> 0.074 bits)
    dcs = a["per_key"]["plan_ask-regex-0.2|oblique"]["all"]["decode"]
    assert dcs["n_decodes"] == 144 * 2 and dcs["mean_answer_logloss_bits"] == pytest.approx(1.0)
    ll = a["contrasts_answer_logloss"]["decoder|plan_ask-noul_answer-0.2|oblique"]["all"]["point"]
    assert ll == pytest.approx(-__import__("math").log2(0.95) - 1.0)
    assert a["per_key"]["sonnet_agent|plain"]["all"]["agent"]["well_formed_share"] == 1.0


def test_verdicts_rules_and_gates_on_synthetic(tmp_path: Path):
    d = _synthetic_dir(tmp_path, EFFECTS)
    a = az.analyze(str(d), B=300, seed=1)
    v = vn.verdicts(a)
    assert v["primary_hedge"] == "1/5" and v["gates"]["pipeline"]["pass"]
    assert v["gates"]["oracle_pair_reproduces_ceiling"]["pass"]           # 0.10 = 0.06/0.6 exactly
    assert v["gates"]["oracle_decoder_text_invariant"]["pass"] and v["gates_pass"]
    R = v["rules"]
    assert R["P1"]["verdict"] == "confirmed" and R["P1"]["captured_fraction_of_delta_menu"]["point"] == pytest.approx(0.8)
    assert R["P2"]["verdict"] == "confirmed"
    assert R["P3"]["verdict"] == "confirmed"                               # exactly 0 in plain
    assert R["P4"]["verdict"] == "confirmed"
    assert R["P5_noul_answer"]["verdict"] == "confirmed" and R["P5_regex"]["verdict"] == "confirmed"
    assert R["P6_plain"]["verdict"] == "not supported"                     # the synthetic agent is perfectly selective
    assert R["P7_sonnet_agent_oblique"]["verdict"] == "confirmed" and R["P7_jev_agent_plain"]["verdict"] == "confirmed"
    assert R["P8"]["verdict"] == "confirmed"
    # a P1 that is positive but captures too little
    weak = dict(EFFECTS); weak[("plan_ask-noul_answer-0.2", "plain")] = 0.02
    a2 = az.analyze(str(_synthetic_dir(tmp_path / "w", weak)), B=300, seed=1)
    assert vn.verdicts(a2)["rules"]["P1"]["verdict"] == "positive but below captured minimum"
    assert vn.within_delta({"point": 0.0, "ci95": [-0.01, 0.01]}) == "confirmed"
    assert vn.within_delta({"point": 0.0, "ci95": [-0.03, 0.01]}) == "not supported"
    assert vn.not_negative({"point": -0.01, "ci95": [-0.03, 0.005]}) == "confirmed"
    assert vn.not_negative({"point": -0.02, "ci95": [-0.03, -0.01]}) == "not supported"


def test_renderer_formats_every_section(tmp_path: Path):
    d = _synthetic_dir(tmp_path, EFFECTS)
    a = az.analyze(str(d), B=200, seed=1)
    v = vn.verdicts(a)
    text = rn.render(str(d), a, v)
    for heading in ("## Gates", "## Exact ceilings", "## Per (arm, regime)", "## Decoders in play", "## Selectivity", "## Agent arms",
                    "## Paired contrasts in Y", "## Pre-registered rules and verdicts", "## Read-out (computed)", "## Limits"):
        assert heading in text, heading
    assert "| P1 |" in text and "**confirmed**" in text and "captured 80%" in text
    assert "ask_value|plan_ask-noul_answer-0.2|plain" in text


@pytest.mark.skipif(not (SMOKE / "runs.jsonl").exists(), reason="offline smoke run absent")
def test_analyzer_schema_on_offline_run():
    """Schema only: the smoke run's contrasts are not read (design: a pilot is plumbing)."""
    a = az.analyze(str(SMOKE), B=50, seed=0)
    assert a["error_stubs"] == 0 and a["episodes"] > 0
    assert set(a["keys"]) >= {"plan_noask|none", "plan_ask-oracle-0|plain", "sonnet_agent|oblique"}
    assert "ask_value|plan_ask-noul_answer-0.2|plain" in a["contrasts"] and "text_invariance|plan_ask-oracle-0" in a["contrasts"]
    for key in a["keys"]:
        assert a["per_key"][key]["all"]["n"] == 144
    v = vn.verdicts(a)
    assert v["gates"]["oracle_decoder_text_invariant"]["pass"]
    assert set(v["rules"]) >= {"P1", "P2", "P3", "P4", "P5_noul_answer", "P5_regex", "P6_plain", "P6_oblique", "P7_sonnet_agent_plain", "P8"}
