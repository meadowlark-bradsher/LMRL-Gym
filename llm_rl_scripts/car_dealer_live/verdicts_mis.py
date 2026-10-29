"""Pre-registered decision rules for CD-MIS-1 (learning with the wrong opponent model).

Fixed in code before the confirmatory run; PREREGISTRATION_MIS.md quotes them. Input: the
analysis.json of the mismatch grid (analyze_mis.py). Output: verdicts.json beside it.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_mis --analysis results/<dir>/analysis.json

Vocabulary (95% percentile cluster-bootstrap CI over the 36 type pairs, outcome Y unless stated):
  '> 0' / '< 0' predictions -> 'confirmed' (CI excludes 0 on that side), 'opposite', 'not supported'.
  '<= 0' prediction (a cost that may be zero) -> 'confirmed' if the CI upper bound <= 0 or the CI includes 0, else 'opposite'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CONFIRMATORY_PROCESS = "history"       # preselected: history-contingent countering (design 7.1 "conditional dependence omitted from filter")
SECONDARY_PROCESS = "withholding"
CELLS = "all"


def verdict(c, predicted):
    if not c or c.get("point") is None:
        return "no data"
    lo, hi = c["ci95"]
    if predicted == "neg":
        return "confirmed" if hi < 0 else ("opposite" if lo > 0 else "not supported")
    if predicted == "pos":
        return "confirmed" if lo > 0 else ("opposite" if hi < 0 else "not supported")
    if predicted == "nonpos":
        return "confirmed" if lo <= 0 else "opposite"
    if predicted == "zero":
        return "confirmed" if lo <= 0 <= hi else "not supported"
    raise ValueError(predicted)


def est(c):
    return None if not c or c.get("point") is None else {"point": c["point"], "ci95": c["ci95"], "n_clusters": c["n_clusters"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True)
    args = ap.parse_args(argv)
    A = json.load(open(args.analysis))
    C = A["contrasts"]; D = A.get("contrasts_diag", {})
    out = {"confirmatory_process": CONFIRMATORY_PROCESS, "cells": CELLS, "gates": {}, "rules": {}}

    # gates: no stubs; every (process, arm) has the same episode count; matched process = sanity
    counts = {p: {a: s["all"]["n"] for a, s in arms.items()} for p, arms in A["per_arm"].items()}
    flat = {n for arms in counts.values() for n in arms.values()}
    out["gates"]["no_stubs"] = {"error_stubs": A["error_stubs"], "pass": A["error_stubs"] == 0}
    out["gates"]["balanced"] = {"episodes_per_process_arm": counts, "pass": len(flat) == 1}
    m = A["per_arm"].get("matched", {})
    if "C_naive" in m and "O_true" in m:
        # under the matched process the naive filter is correctly specified (up to the 1e-3 floor), so the
        # ASK-allowed planner on it must equal O_true: O_true - C_naive not positive. (Corrected after the
        # run from O_true - B_naive, which compares an ASK-allowed with an ASK-forbidden planner and is
        # positive by the value of asking; disclosed in PREREGISTRATION_MIS.md and MIS_RESULTS.md.)
        c = C.get("matched", {}).get("O_true-C_naive", {}).get(CELLS)
        out["gates"]["matched_sanity"] = {"O_true-C_naive": est(c), "pass": c is not None and c["ci95"][0] <= 1e-9,
                                          "note": "corrected post-run from O_true-B_naive (mis-specified: ASK-allowed vs ASK-forbidden)"}
    out["gates_pass"] = all(g["pass"] for g in out["gates"].values())

    def rule(key, process, pair, pred, note, outcome=None):
        src = C if outcome is None else D.get(outcome, {})
        c = src.get(process, {}).get(pair, {}).get(CELLS)
        out["rules"][key] = {"process": process, "quantity": f"Δ {outcome or 'Y'}, {pair}, {CELLS} cells", "prediction": pred, "note": note,
                             "estimate": est(c), "verdict": verdict(c, {"< 0": "neg", "> 0": "pos", "<= 0": "nonpos", "= 0": "zero"}[pred])}

    rule("M1", CONFIRMATORY_PROCESS, "B_naive-A_prior", "< 0", "PRIMARY: negative learning under history-contingent countering")
    rule("M2", SECONDARY_PROCESS, "B_naive-A_prior", "< 0", "negative learning under strategic withholding")
    rule("M3", "matched", "B_naive-A_prior", "> 0", "positive control: correctly specified learning helps")
    rule("M4", "matched", "B_mix0.2-B_naive", "<= 0", "matched-model cost of robustness (mixture)")
    rule("M5", "matched", "B_temp0.5-B_naive", "<= 0", "matched-model cost of robustness (tempering)")
    rule("M6", CONFIRMATORY_PROCESS, "B_mix0.2-B_naive", "> 0", "robust filtering recovers value under the confirmatory mismatch")
    rule("M7", CONFIRMATORY_PROCESS, "B_temp0.5-B_naive", "> 0", "tempering recovers value under the confirmatory mismatch")
    rule("M8", CONFIRMATORY_PROCESS, "O_true-B_naive", "> 0", "cost of misspecification (reference, never available to a deployed agent)")
    rule("M9", CONFIRMATORY_PROCESS, "C_naive-B_naive", "> 0", "does asking help under the confirmatory mismatch")
    rule("M10", CONFIRMATORY_PROCESS, "B_naive-A_prior", "> 0", "belief error: naive posterior log-loss on the true type vs the prior (bits)", outcome="final_logloss_true_bits")
    rule("M11", CONFIRMATORY_PROCESS, "B_naive-A_prior", "> 0", "decision loss: true-model decision regret, naive vs prior", outcome="total_decision_regret_true")
    for proc in A["processes"]:
        if proc in ("matched",):
            continue
        rule(f"G_{proc}", proc, "B_naive-A_prior", "< 0", "full grid, reported without selection")
    out["descriptive"] = {"cross_process": A.get("cross_process")}
    dest = os.path.join(os.path.dirname(args.analysis), "verdicts.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}; gates pass = {out['gates_pass']}")
    for k, v in out["rules"].items():
        e = v["estimate"]
        s = "" if not e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        print(f"  {k:14s} {v['verdict']:14s} {s:34s} {v['process']:11s} {v['quantity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
