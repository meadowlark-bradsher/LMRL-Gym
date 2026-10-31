"""Pre-registered decision rules for CD-MIS-2 (severity sweep; the evasive-answer channel).

Fixed in code before the confirmatory run; PREREGISTRATION_MIS2.md quotes them. Input: the
analysis.json of the CD-MIS-2 grid (analyze_mis.py) and, for the replication gates, the CD-MIS-1
analysis.json. Output: verdicts.json beside the CD-MIS-2 analysis.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_mis2 \
        --analysis results/<mis2 dir>/analysis.json --mis1 results/live_mis_20261029/analysis.json

Vocabulary (95% percentile cluster-bootstrap CI over the 36 type pairs, outcome Y, all cells):
  '> 0' / '< 0'  -> 'confirmed' (CI excludes 0 on that side), 'opposite' (excludes 0 on the other side), else 'not supported'.
  '<= 0' / '>= 0' (a cost or a gain that may be zero) -> 'confirmed' unless the CI lies entirely on the wrong side ('opposite').
  '= 0' -> 'confirmed' if the CI includes 0, else 'not supported'; the identical-trajectory share is reported beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CONFIRMATORY = ("evasive@0.3", "C_ans0.2-C_naive")      # E1: a fixed hedged answer model recovers the evasive-answer loss
CELLS = "all"
EVASIVE_LEVELS = ["evasive@0", "evasive@0.1", "evasive@0.2", "evasive@0.3", "evasive@0.4", "evasive@0.5"]
SHARED_ARMS_WITH_MIS1 = {"evasive@0.3": ("evasive", ["A_prior", "B_naive", "C_naive", "C_mix0.2", "O_true"]),
                         "history@0.1": ("history", ["A_prior", "B_naive", "B_mix0.2", "C_naive", "O_true"]),
                         "stochastic@0.15": ("stochastic", ["A_prior", "B_naive", "B_mix0.2", "C_naive", "O_true"])}
DIAGONAL_REGRET_TOL = 1e-3     # utility units per episode; the 1e-3 floor can only move the hedged arm off O_true at exact ties
PRED = {"< 0": "neg", "> 0": "pos", "<= 0": "nonpos", ">= 0": "nonneg", "= 0": "zero"}


def verdict(c, predicted):
    if not c or c.get("point") is None or not c.get("ci95"):
        return "no data"
    lo, hi = c["ci95"]
    if predicted == "neg":
        return "confirmed" if hi < 0 else ("opposite" if lo > 0 else "not supported")
    if predicted == "pos":
        return "confirmed" if lo > 0 else ("opposite" if hi < 0 else "not supported")
    if predicted == "nonpos":
        return "confirmed" if lo <= 0 else "opposite"
    if predicted == "nonneg":
        return "confirmed" if hi >= 0 else "opposite"
    if predicted == "zero":
        return "confirmed" if lo <= 0 <= hi else "not supported"
    raise ValueError(predicted)


def est(c):
    return None if not c or c.get("point") is None else {"point": c["point"], "ci95": c["ci95"], "n_clusters": c["n_clusters"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--mis1", default=None, help="CD-MIS-1 analysis.json for the exact-replication gate")
    args = ap.parse_args(argv)
    A = json.load(open(args.analysis))
    C = A["contrasts"]; I = A.get("trajectory_identity", {}); P = A["per_arm"]
    out = {"confirmatory": {"process": CONFIRMATORY[0], "contrast": CONFIRMATORY[1]}, "cells": CELLS, "gates": {}, "rules": {}}

    # ---- gates ------------------------------------------------------------------------------------
    counts = {p: {a: s["all"]["n"] for a, s in arms.items()} for p, arms in P.items()}
    flat = {n for arms in counts.values() for n in arms.values()}
    out["gates"]["no_stubs"] = {"error_stubs": A["error_stubs"], "pass": A["error_stubs"] == 0}
    out["gates"]["balanced"] = {"episodes_per_process_arm": counts, "pass": len(flat) == 1}
    # the no-ask arms never see an answer, so along the evasive axis their episodes are identical (family-seeded streams)
    ev = [p for p in EVASIVE_LEVELS if p in P]
    b_means = {p: P[p]["B_naive"]["all"]["mean_normalized_net_gain"] for p in ev if "B_naive" in P[p]}
    out["gates"]["noask_invariant_to_answer_noise"] = {"B_naive_mean_Y_by_level": b_means,
                                                       "pass": len(b_means) >= 2 and max(b_means.values()) - min(b_means.values()) < 1e-12}
    # the hedged arm whose assumed rate equals the true rate IS the true-model planner (the evasive family keeps the T1
    # behaviour), so its true-model decision regret is zero up to exact ties broken differently by the 1e-3 floor
    dreg = P.get("evasive@0.3", {}).get("C_ans0.3", {}).get("all", {}).get("mean_total_decision_regret_true")
    diag = I.get("evasive@0.3", {}).get("O_true-C_ans0.3")
    out["gates"]["diagonal_zero_regret"] = {"C_ans0.3 mean true-model decision regret at evasive@0.3": dreg, "tolerance": DIAGONAL_REGRET_TOL,
                                            "O_true-C_ans0.3 identical trajectories (ties may differ)": diag,
                                            "pass": dreg is not None and dreg < DIAGONAL_REGRET_TOL}
    if args.mis1 and os.path.exists(args.mis1):
        M = json.load(open(args.mis1))["per_arm"]
        rep = {}
        ok = True
        for p2, (p1, arms) in SHARED_ARMS_WITH_MIS1.items():
            if p2 not in P or p1 not in M:
                continue
            for a in arms:
                if a in P[p2] and a in M[p1]:
                    d = P[p2][a]["all"]["mean_normalized_net_gain"] - M[p1][a]["all"]["mean_normalized_net_gain"]
                    rep[f"{p2}:{a}"] = d
                    ok = ok and abs(d) < 1e-12
        out["gates"]["replicates_mis1_exactly"] = {"mean_Y_difference_by_process_arm": rep, "pass": ok and bool(rep)}
    out["gates_pass"] = all(g["pass"] for g in out["gates"].values())

    # ---- rules ------------------------------------------------------------------------------------
    def rule(key, process, pair, pred, note):
        c = C.get(process, {}).get(pair, {}).get(CELLS)
        ident = I.get(process, {}).get(pair)
        out["rules"][key] = {"process": process, "quantity": f"Δ Y, {pair}, {CELLS} cells", "prediction": pred, "note": note,
                             "estimate": est(c), "identical_trajectory_share": None if not ident else ident.get("identical_share"),
                             "verdict": verdict(c, PRED[pred])}

    # the evasive channel
    rule("E1", "evasive@0.3", "C_ans0.2-C_naive", "> 0", "PRIMARY: a fixed hedged answer model (assumed flip 0.2) recovers value lost by trusting evasive answers (true flip 0.3)")
    rule("E2", "evasive@0", "C_ans0.2-C_naive", "<= 0", "cost of the hedge when answers are truthful (its size is the premium)")
    rule("E3a", "evasive@0", "C_naive-B_naive", "> 0", "dose-response start: asking helps when answers are truthful")
    rule("E3b", "evasive@0.5", "C_naive-B_naive", "< 0", "dose-response end: trusting pure-noise answers hurts")
    rule("E4", "evasive@0.1", "O_true-B_naive", "> 0", "questions keep value under mild noise when the answer model is right")
    rule("E5", "evasive@0.5", "O_true-B_naive", "= 0", "pure-noise answers: the calibrated planner stops asking and equals the no-ask learner")
    rule("E6", "evasive@0.3", "C_ans0.2-B_naive", ">= 0", "the hedged ASK arm is not worse than never asking at the CD-MIS-1 rate")
    # behavioural severity
    rule("S1", "history@0.4", "B_naive-A_prior", ">= 0", "learning the wrong model does not hurt at the harshest history step")
    rule("S2", "stochastic@0.5", "B_naive-A_prior", ">= 0", "learning the wrong model does not hurt at the harshest acceptance noise")
    rule("S3a", "history@0.4", "B_mix0.2-B_naive", "= 0", "robust filter inert at the harshest history step")
    rule("S3b", "stochastic@0.5", "B_mix0.2-B_naive", "= 0", "robust filter inert at the harshest acceptance noise")
    rule("S4a", "history@0.4", "O_true-C_naive", "= 0", "asking insures against the behavioural model at any severity")
    rule("S4b", "stochastic@0.5", "O_true-C_naive", "> 0", "acceptance noise is the behavioural axis asking does not fully insure")
    for proc in A["processes"]:
        rule(f"G_{proc}", proc, "B_naive-A_prior", "< 0", "full grid (the CD-MIS-1 negative-learning direction), reported without selection")
    for proc in [p for p in EVASIVE_LEVELS if p in C]:
        rule(f"D_{proc}", proc, "C_naive-B_naive", "> 0", "dose-response of the value of trusting answers, reported without selection")

    # descriptive: where the dose-response changes sign
    signs = []
    for proc in [p for p in EVASIVE_LEVELS if p in C]:
        c = C[proc].get("C_naive-B_naive", {}).get(CELLS)
        if c and c.get("ci95"):
            lo, hi = c["ci95"]
            signs.append({"process": proc, "point": c["point"], "ci95": c["ci95"], "side": "above 0" if lo > 0 else ("below 0" if hi < 0 else "unresolved")})
    out["descriptive"] = {"dose_response_C_naive_minus_B_naive": signs, "severity": A.get("severity"), "cross_process": A.get("cross_process")}

    dest = os.path.join(os.path.dirname(args.analysis), "verdicts.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}; gates pass = {out['gates_pass']}")
    for g, v in out["gates"].items():
        print(f"  gate {g:36s} {'PASS' if v['pass'] else 'FAIL'}")
    for k, v in out["rules"].items():
        e = v["estimate"]
        s = "" if not e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        print(f"  {k:18s} {v['verdict']:14s} {s:34s} {v['process']:16s} {v['quantity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
