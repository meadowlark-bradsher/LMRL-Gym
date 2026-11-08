"""Pre-registered decision rules for CD-NOUL-1 Stage 1 (design §4, P1–P8).

Rules are fixed HERE, in code, before the confirmatory seed is used; PREREGISTRATION_NOUL.md
quotes them. ``PRIMARY_HEDGE`` is the one registration-time choice the design left open and
must be set before the tag. Input: ``analysis.json`` (analyze_noul.py). Output: ``verdicts.json``
beside it.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.verdicts_noul --analysis results/<dir>/analysis.json

Verdict vocabulary (95 % percentile cluster-bootstrap CI over the 36 type pairs; δ = 0.02 Y):
  '> 0'            confirmed if the CI excludes 0 above; opposite if it excludes 0 below; else not supported
  '< 0'            the mirror image
  '≥ 0'            confirmed if the CI is not entirely below 0
  'within δ of 0'  confirmed if the CI lies inside [−δ, +δ]
  P1 also needs its captured fraction (point) ≥ CAPTURED_MIN; P6 needs the selectivity
  difference's upper CI bound below SELECTIVITY_MAX.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

from llm_rl_scripts.car_dealer_live.verdicts_live import DELTA, ci_verdict, est, load

from . import arms_noul as an

STUDY = "CD-NOUL-1"
PRIMARY_HEDGE = Fraction(1, 5)       # SET BY THE REGISTRATION (design §7c): 0.2 or 0
PRIMARY_CELLS = "focal_opens"
CAPTURED_MIN = 0.5
SELECTIVITY_MAX = 0.2
WELL_FORMED_MIN = 0.99


def _h(a: Fraction) -> str:
    return "0" if a == 0 else "0.2"


def _c(A, name, cs=PRIMARY_CELLS, table="contrasts"):
    return (A.get(table, {}).get(name) or {}).get(cs)


def within_delta(c):
    if not c or c.get("point") is None:
        return "no data"
    lo, hi = c["ci95"]
    return "confirmed" if (lo >= -DELTA and hi <= DELTA) else "not supported"


def not_negative(c):
    if not c or c.get("point") is None:
        return "no data"
    return "confirmed" if c["ci95"][1] >= 0 else "not supported"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True)
    args = ap.parse_args(argv)
    A = load(args.analysis)
    out = verdicts(A)
    dest = os.path.join(os.path.dirname(args.analysis), "verdicts.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}; gates pass = {out['gates_pass']}")
    for k, v in out["rules"].items():
        e = v.get("estimate")
        s = "" if not e or "point" not in e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        print(f"  {k:6s} {v['verdict']:14s} {s:34s} {v['quantity']}")
    return 0


def verdicts(A: dict) -> dict:
    a = PRIMARY_HEDGE
    noul = an.arm_name("noul_answer", a); regex = an.arm_name("regex", a); ntype = an.arm_name("noul_type", a)
    orc0 = an.arm_name("oracle", Fraction(0))
    out = {"study": STUDY, "delta": DELTA, "primary_hedge": str(a), "primary_cells": PRIMARY_CELLS,
           "captured_min": CAPTURED_MIN, "selectivity_max": SELECTIVITY_MAX, "gates": {}, "rules": {}}

    # ---- gates: fail -> fix the pipeline, do not read results ----
    pk = A["per_key"]
    ns = {k: v["all"].get("n") for k, v in pk.items()}
    wf = {k: v["all"]["agent"]["well_formed_share"] for k, v in pk.items() if v["all"].get("agent")}
    out["gates"]["pipeline"] = {"error_stubs": A["error_stubs"], "episodes_per_key": ns, "well_formed": wf,
                                "pass": A["error_stubs"] == 0 and len(set(ns.values())) == 1 and all(w >= WELL_FORMED_MIN for w in wf.values())}
    c = _c(A, f"ask_value|{orc0}|plain", "all")
    ceil = (A.get("ceilings", {}).get("all") or {}).get("mean_delta_menu_Y")
    if c and c.get("point") is not None and ceil is not None:
        out["gates"]["oracle_pair_reproduces_ceiling"] = {"contrast": est(c), "ceiling_delta_menu_Y": ceil,
                                                          "pass": c["ci95"][0] - 1e-12 <= ceil <= c["ci95"][1] + 1e-12}
    t = _c(A, f"text_invariance|{orc0}", "all")
    if t and t.get("point") is not None:
        out["gates"]["oracle_decoder_text_invariant"] = {"contrast": est(t), "pass": abs(t["point"]) < 1e-12}
    out["gates_pass"] = all(g.get("pass", False) for g in out["gates"].values())

    def rule(key, quantity, prediction, c, verdict, extra=None):
        r = {"quantity": quantity, "prediction": prediction, "estimate": est(c), "verdict": verdict}
        if extra:
            r.update(extra)
        out["rules"][key] = r

    # P1 primary: the architecture captures the value of inquiry in plain text
    c = _c(A, f"ask_value|{noul}|plain")
    cap = ((c or {}).get("captured_fraction") or {}).get("delta_menu") or {}
    v = ci_verdict(c, "pos")
    if v == "confirmed" and (cap.get("point") is None or cap["point"] < CAPTURED_MIN):
        v = "positive but below captured minimum"
    rule("P1", f"Δ Y, {noul} (plain) − plan_noask, {PRIMARY_CELLS} (PRIMARY)", f"> 0 and captured fraction of Δ*_menu ≥ {CAPTURED_MIN}", c, v,
         {"captured_fraction_of_delta_menu": cap, "ceiling_delta_menu_Y": (A["ceilings"].get(PRIMARY_CELLS) or {}).get("mean_delta_menu_Y")})
    # P2 / P3: the LLM likelihood earns its keep only where the text is indirect
    rule("P2", f"Δ Y, {noul} − {regex}, oblique, {PRIMARY_CELLS}", "> 0", _c(A, f"decoder|{noul}|oblique"), ci_verdict(_c(A, f"decoder|{noul}|oblique"), "pos"))
    rule("P3", f"Δ Y, {noul} − {regex}, plain, {PRIMARY_CELLS}", f"within δ = {DELTA} of 0", _c(A, f"decoder|{noul}|plain"), within_delta(_c(A, f"decoder|{noul}|plain")))
    # P4: the type reading carries more than the yes/no (exploratory, sign not required)
    rule("P4", f"Δ Y, {ntype} − {noul}, oblique, {PRIMARY_CELLS}", "≥ 0 (exploratory)", _c(A, f"type_vs_answer|{a}|oblique"), not_negative(_c(A, f"type_vs_answer|{a}|oblique")))
    # P5: the fixed hedge is not harmful under decoding error
    for d in ("noul_answer", "regex"):
        rule(f"P5_{d}", f"Δ Y, plan_ask-{d}-0.2 − plan_ask-{d}-0, oblique, {PRIMARY_CELLS}", "≥ 0", _c(A, f"hedge|{d}|oblique"), not_negative(_c(A, f"hedge|{d}|oblique")))
    # P6: the LLM agent asks non-selectively
    for rg in ("plain", "oblique"):
        s = A.get("selectivity", {}).get(f"{an.SONNET_AGENT}|{rg}")
        c_sel = None if not s or s.get("difference") is None or "difference_ci95" not in s else \
            {"point": s["difference"], "ci95": s["difference_ci95"], "n_clusters": s.get("n_clusters")}
        v = "no data" if not c_sel else ("confirmed" if c_sel["ci95"][1] < SELECTIVITY_MAX else "not supported")
        rule(f"P6_{rg}", f"P(ask | ASK optimal) − P(ask | legal, not optimal), {an.SONNET_AGENT}, {rg}", f"< {SELECTIVITY_MAX} (upper CI bound)", c_sel, v,
             {"p_ask_given_optimal": (s or {}).get("p_ask_given_optimal"), "p_ask_given_not_optimal": (s or {}).get("p_ask_given_not_optimal")})
    # P7: the architecture beats the LLM-as-agent on identical menus
    for agent in an.AGENT_ARMS:
        for rg in ("plain", "oblique"):
            c = _c(A, f"chooser|{agent}|{a}|{rg}")
            rule(f"P7_{agent}_{rg}", f"Δ Y, {noul} − {agent}, {rg}, {PRIMARY_CELLS}", "> 0", c, ci_verdict(c, "pos"))
    # P8: the decoder reads the text better than the parser (in-episode answer log-loss, bits)
    c = _c(A, f"decoder|{noul}|oblique", "all", "contrasts_answer_logloss")
    rule("P8", f"Δ answer log-loss (bits), {noul} − {regex}, oblique, all cells", "< 0", c, ci_verdict(c, "neg"))
    # descriptive companions (no verdict): the same contrasts at the other hedge, and the Sonnet judge as decoder
    other = Fraction(0) if a != 0 else Fraction(1, 5)
    out["descriptive"] = {
        "P1_other_hedge": est(_c(A, f"ask_value|{an.arm_name('noul_answer', other)}|plain")),
        "P2_other_hedge": est(_c(A, f"decoder|{an.arm_name('noul_answer', other)}|oblique")),
        "sonnet_judge_vs_regex_oblique": est(_c(A, f"decoder|{an.arm_name('sonnet_answer', a)}|oblique")),
        "sonnet_judge_vs_noul_oblique": None,
        "oracle_ask_value_plain": est(_c(A, f"ask_value|{orc0}|plain")),
        "mean_Y_focal_opens": {k: v[PRIMARY_CELLS].get("mean_normalized_net_gain") for k, v in pk.items()},
        "questions_per_episode": {k: v["all"].get("mean_questions_asked") for k, v in pk.items()},
    }
    sj, nj = _c(A, f"vs_oracle|{an.arm_name('sonnet_answer', a)}|oblique"), _c(A, f"vs_oracle|{noul}|oblique")
    if sj and nj and sj.get("point") is not None and nj.get("point") is not None:
        out["descriptive"]["sonnet_judge_vs_noul_oblique"] = {"point_difference_of_gaps_to_oracle": sj["point"] - nj["point"],
                                                              "note": "two separately bootstrapped contrasts; no interval"}
    return out


if __name__ == "__main__":
    sys.exit(main())
