"""Pre-registered decision rules for the live Car-Dealer Jev run (CD-VOI-1, Phase 1).

Rules are fixed HERE, in code, before any Jev call; the registration note quotes them.
Input: analysis.json files written by analyze_phase1.py for the primary regime
(H = 4, c_q = 0) and the named secondary regimes, plus the frozen scripted analysis
of the primary regime for the random-arm sanity gate. Output: verdicts.json next
to the primary analysis.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_live \
        --primary results/<jev h4_cq0>/analysis.json --h2 results/<jev h2_cq0>/analysis.json \
        --h3 results/<jev h3_cq0>/analysis.json --cq3 results/<jev h4_cq3>/analysis.json \
        --full results/<jev h4_cq0 full menu>/analysis.json \
        --scripted results/live_phase1_scripted3_20261029/h4_cq0/analysis.json

Verdict vocabulary (95% percentile cluster-bootstrap CI over the 36 type pairs):
  'positive'            CI entirely above 0
  'materially positive' CI entirely above 0 AND lower bound > DELTA
  'negative'            CI entirely below 0
  'unresolved'          CI includes 0
  For a 'not > 0' prediction: 'confirmed' if CI includes 0 or lies below 0, else 'not supported'.
  For a '> 0' / '< 0' prediction: 'confirmed' if the CI excludes 0 on that side, 'opposite' if it
  excludes 0 on the other side, else 'not supported'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DELTA = 0.02          # materiality threshold, normalized-gain units (design section 6, provisional by design)
PRIMARY_CELLS = "focal_opens"
P1_CONTRAST = "jev_ask_plain-jev_noask_plain"


def ci_verdict(c, predicted: str):
    if not c or c.get("point") is None:
        return "no data"
    lo, hi = c["ci95"]
    if predicted == "pos_material":
        if lo > 0:
            return "materially positive" if lo > DELTA else "positive"
        return "negative" if hi < 0 else "unresolved"
    if predicted == "pos":
        return "confirmed" if lo > 0 else ("opposite" if hi < 0 else "not supported")
    if predicted == "neg":
        return "confirmed" if hi < 0 else ("opposite" if lo > 0 else "not supported")
    if predicted == "not_pos":
        return "confirmed" if lo <= 0 else "not supported"
    if predicted == "zero":
        return "confirmed" if lo <= 0 <= hi else "not supported"
    raise ValueError(predicted)


def est(c):
    return None if not c or c.get("point") is None else {"point": c["point"], "ci95": c["ci95"], "n_clusters": c["n_clusters"]}


def load(path):
    return json.load(open(path)) if path else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--primary", required=True); ap.add_argument("--h2"); ap.add_argument("--h3"); ap.add_argument("--cq3")
    ap.add_argument("--full"); ap.add_argument("--scripted", help="frozen scripted analysis.json of the primary regime (random-arm gate)")
    args = ap.parse_args(argv)
    P = load(args.primary); H2 = load(args.h2); H3 = load(args.h3); CQ3 = load(args.cq3); FULL = load(args.full); SCR = load(args.scripted)
    out = {"delta": DELTA, "primary_cells": PRIMARY_CELLS, "gates": {}, "rules": {}}

    # ---- sanity gates: fail -> fix the pipeline, do not read results ----
    for name, a in (("primary", P), ("h2", H2), ("h3", H3), ("cq3", CQ3), ("full", FULL)):
        if a is None:
            continue
        g = {"error_stubs": a["error_stubs"], "episodes_per_arm": {k: v["all"].get("n") for k, v in a["per_arm"].items()},
             "well_formed": {k: v.get("well_formed_share") for k, v in a.get("position", {}).items()}}
        ns = {v for v in g["episodes_per_arm"].values()}
        g["pass"] = (a["error_stubs"] == 0 and len(ns) == 1 and all((w or 0) >= 1.0 - 1e-12 for w in g["well_formed"].values()))
        out["gates"][name] = g
    if SCR is not None and "random" in P["per_arm"] and "random" in SCR["per_arm"]:
        m = P["per_arm"]["random"]["all"]["mean_normalized_net_gain"]
        ci = SCR["per_arm"]["random"]["all"].get("ci95_normalized_net_gain")
        out["gates"]["random_matches_frozen"] = {"jev_run_random_mean_Y": m, "frozen_random_ci95": ci,
                                                 "pass": (ci is not None and ci[0] - 1e-12 <= m <= ci[1] + 1e-12)}
    out["gates_pass"] = all(g.get("pass", False) for g in out["gates"].values())

    def rule(key, quantity, prediction, c, kind, extra=None):
        r = {"quantity": quantity, "prediction": prediction, "estimate": est(c), "verdict": ci_verdict(c, kind)}
        if extra:
            r.update(extra)
        out["rules"][key] = r

    C = P["contrasts"]; CQ = P.get("contrasts_questions", {})
    c = C.get(P1_CONTRAST, {}).get(PRIMARY_CELLS)
    rule("P1", f"Δ Y, {P1_CONTRAST}, {PRIMARY_CELLS}, H=4 c_q=0 (PRIMARY)", f"> 0; material if lower CI bound > {DELTA}", c, "pos_material",
         {"captured_fraction_of_delta_menu": (c or {}).get("captured_fraction", {}).get("delta_menu"),
          "captured_fraction_of_delta_full": (c or {}).get("captured_fraction", {}).get("delta_full"),
          "ceiling_delta_menu_Y": P["ceilings"][PRIMARY_CELLS].get("mean_delta_menu_Y"),
          "ceiling_delta_full_Y": P["ceilings"][PRIMARY_CELLS].get("mean_delta_full_Y")})
    rule("S1", f"Δ Y, {P1_CONTRAST}, all cells, H=4 c_q=0", "> 0", C.get(P1_CONTRAST, {}).get("all"), "pos_material")
    rule("S2", f"Δ Y, jev_ask_eig-jev_noask_eig, {PRIMARY_CELLS}, H=4 c_q=0", "> 0", C.get("jev_ask_eig-jev_noask_eig", {}).get(PRIMARY_CELLS), "pos_material")
    rule("S3", "Δ questions asked / episode, jev_ask_eig-jev_ask_plain, all cells (EIG shown -> more asking)", "> 0",
         CQ.get("jev_ask_eig-jev_ask_plain", {}).get("all"), "pos")
    rule("S4", f"Δ Y, jev_ask_eig-jev_ask_plain, {PRIMARY_CELLS} (EIG shown does NOT raise net gain)", "not > 0",
         C.get("jev_ask_eig-jev_ask_plain", {}).get(PRIMARY_CELLS), "not_pos")
    for arm in ("jev_ask_plain", "jev_ask_eig"):
        s = P.get("selectivity", {}).get(arm)
        c_sel = None if not s or s.get("difference") is None else {"point": s["difference"], "ci95": s["difference_ci95"], "n_clusters": s.get("n_clusters")}
        rule(f"S5_{arm}", f"P(ask | ASK strictly optimal) − P(ask | ASK legal, not optimal), {arm}, H=4 c_q=0", "> 0", c_sel, "pos",
             {"p_ask_given_optimal": (s or {}).get("p_ask_given_optimal"), "p_ask_given_not_optimal": (s or {}).get("p_ask_given_not_optimal")})
    rule("S6", f"Δ Y, jev_ask_plain-exact_ask, {PRIMARY_CELLS}, H=4 c_q=0 (gap to the planner)", "< 0", C.get("jev_ask_plain-exact_ask", {}).get(PRIMARY_CELLS), "neg")
    if H2 is not None:
        rule("S7", f"Δ Y, {P1_CONTRAST}, all cells, H=2 c_q=0 (negative control: ASK never optimal)", "not > 0",
             H2["contrasts"].get(P1_CONTRAST, {}).get("all"), "not_pos",
             {"jev_ask_plain_questions_per_episode": H2["per_arm"].get("jev_ask_plain", {}).get("all", {}).get("mean_questions_asked"),
              "note": "every question at H=2 is strictly suboptimal; the ask rate is the information-bias read-out"})
    if H3 is not None:
        rule("S8", f"Δ Y, {P1_CONTRAST}, all cells, H=3 c_q=0 (positive control across kernels)", "> 0",
             H3["contrasts"].get(P1_CONTRAST, {}).get("all"), "pos_material")
    if CQ3 is not None:
        a0 = P["per_arm"].get("jev_ask_plain", {}).get("all", {}); a3 = CQ3["per_arm"].get("jev_ask_plain", {}).get("all", {})
        ci0, ci3 = a0.get("ci95_questions_asked"), a3.get("ci95_questions_asked")
        v = "no data"
        if ci0 and ci3:
            v = "confirmed" if (a3["mean_questions_asked"] < a0["mean_questions_asked"] and ci3[1] < ci0[0]) else "not supported"
        out["rules"]["S9"] = {"quantity": "questions / episode, jev_ask_plain, H=4 c_q=3/100 vs c_q=0 (does the policy respect question cost)",
                              "prediction": "< at c_q=0 (CIs disjoint)", "estimate": {"cq0": {"mean": a0.get("mean_questions_asked"), "ci95": ci0},
                                                                                     "cq3": {"mean": a3.get("mean_questions_asked"), "ci95": ci3}}, "verdict": v}
        rule("S10", f"Δ Y, {P1_CONTRAST}, {PRIMARY_CELLS}, H=4 c_q=3/100", "> 0", CQ3["contrasts"].get(P1_CONTRAST, {}).get(PRIMARY_CELLS), "pos_material")
    if FULL is not None:
        rule("S11", f"Δ Y, {P1_CONTRAST}, {PRIMARY_CELLS}, H=4 c_q=0, FULL 27-deal menu (named secondary arm)", "> 0",
             FULL["contrasts"].get(P1_CONTRAST, {}).get(PRIMARY_CELLS), "pos_material",
             {"captured_fraction_of_delta_full": (FULL["contrasts"].get(P1_CONTRAST, {}).get(PRIMARY_CELLS) or {}).get("captured_fraction", {}).get("delta_full")})
    out["descriptive"] = {"position": P.get("position"), "ceilings": P.get("ceilings")}

    dest = os.path.join(os.path.dirname(args.primary), "verdicts.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}; gates pass = {out['gates_pass']}")
    for k, v in out["rules"].items():
        e = v.get("estimate")
        s = "" if not e or "point" not in e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        print(f"  {k:16s} {v['verdict']:22s} {s:36s} {v['quantity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
