"""Regenerate every frozen analysis from the committed logs and diff it.

This is the committed form of the cited-numbers discipline used throughout the
study: every results note renders from a frozen JSON; this script proves those
JSONs still regenerate from the logs with the analyzers as they now stand. It
re-runs each analyzer with its recorded seed and flags, re-applies every
pre-registered decision rule, re-derives the A5 gate diagnosis, and reports
UNCHANGED / DRIFTED per artifact. Exit status is non-zero on any drift.

Schema keys added to analyzers after a file was frozen are stripped from BOTH
sides before comparison (listed per entry); nothing numeric is ever excluded.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.verify_cited
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.verify_cited --only hard
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = [sys.executable, "-m"]
MOD = "llm_rl_scripts.car_dealer_discrete.jev."

Q, QR, QP, H, C, A, S, SM = ("results/jj_quality_20261025", "results/jj_quality_rep1_20261025", "results/jj_quality_pooled_20261025",
                              "results/jj_hard_20261025", "results/jj_counter_20261025", "results/jj_aid_20261025",
                              "results/jj_salience_20261025", "results/jj_smoke_20261025")
AID_STRIP = {"aid_modes_in_logs", "max_abs_dev_vs_logged"}
Q_STRIP = {"feasible_only"}

# (frozen path, analyzer module, argv, keys to strip on both sides)
REGISTRY = [
    (f"{A}/analysis_2x2.json", "analyze_aid", [f"{S}/baseline", f"{S}/salience", f"{A}/aid", f"{A}/aid_salience"], AID_STRIP),
    (f"{A}/analysis_controls.json", "analyze_aid", [f"{S}/baseline", f"{S}/salience", f"{A}/aid", f"{A}/aid_salience", f"{A}/placebo", f"{A}/prior_only"], AID_STRIP),
    (f"{S}/analysis_salience.json", "analyze_salience", [f"{S}/baseline", f"{S}/salience"], set()),
    (f"{SM}/analysis_smoke.json", "analyze_salience", [SM], set()),
    (f"{Q}/analysis_quality.json", "analyze_quality", [f"{Q}/bayes", f"{Q}/prior_only", "--seed", "0"], Q_STRIP),
    (f"{Q}/exploratory_seller_asymmetry.json", "explore_seller_asymmetry", [f"{Q}/bayes", f"{Q}/prior_only", "--seed", "0"], Q_STRIP),
    (f"{QR}/analysis_quality.json", "analyze_quality", [f"{QR}/bayes", f"{QR}/prior_only", "--seed", "1"], Q_STRIP),
    (f"{QR}/confirmatory_responder.json", "explore_seller_asymmetry", [f"{QR}/bayes", f"{QR}/prior_only", "--seed", "1"], Q_STRIP),
    (f"{QP}/analysis_quality_pooled.json", "analyze_quality", [f"{QP}/bayes", f"{QP}/prior_only", "--seed", "2"], Q_STRIP),
    (f"{QP}/responder_pooled.json", "explore_seller_asymmetry", [f"{QP}/bayes", f"{QP}/prior_only", "--seed", "2"], Q_STRIP),
    (f"{H}/analysis_quality_feasible.json", "analyze_quality", [f"{H}/bayes", f"{H}/prior_only", "--feasible-only", "--seed", "3"], set()),
    (f"{H}/responder_feasible.json", "explore_seller_asymmetry", [f"{H}/bayes", f"{H}/prior_only", "--feasible-only", "--seed", "3"], set()),
    (f"{H}/infeasible.json", "analyze_infeasible", [f"{H}/bayes", f"{H}/prior_only", "--seed", "3"], set()),
    (f"{H}/exploratory_role_asymmetry.json", "explore_hard_role_asymmetry", [f"{H}/bayes", f"{H}/prior_only", "--seed", "3"], set()),
    (f"{C}/analysis_counter.json", "analyze_counter_probe", [C, "--seed", "4"], set()),
]


def strip(d, keys):
    if isinstance(d, dict):
        return {k: strip(v, keys) for k, v in d.items() if k not in keys}
    if isinstance(d, list):
        return [strip(v, keys) for v in d]
    return d


def norm(d):  # tuples -> lists etc., exactly as json.dump would have written them
    return json.loads(json.dumps(d, default=str))


def regen(module, argv, out_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    with open(out_path, "w") as f:
        r = subprocess.run(PY + [MOD + module] + argv + ["--json"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{module} failed: {r.stderr[-400:]}")
    return json.load(open(out_path))


# --- pre-registered decision rules, exactly as applied when the verdict files were frozen ---
def incl0(ci): return ci[0] <= 0 <= ci[1]
def neg(v): lo, hi = v["ci95_delta"]; return "CONFIRMED" if hi < 0 else ("OPPOSITE SIGN" if lo > 0 else "UNDERPOWERED/NULL (CI includes 0)")
def pos(v): lo, hi = v["ci95_delta"]; return "CONFIRMED" if lo > 0 else ("OPPOSITE SIGN" if hi < 0 else "UNDERPOWERED/NULL (CI includes 0)")

def rules_responder(X):
    L1, L2, L2b = X["lens1_starting_side"], X["lens2_who_concedes"], X["lens2b_placebo_starter_vs_responder_first_offer"]
    p4 = all(incl0(L2b[k]["ci95_delta"]) for k in L2b if k.startswith("STARTER"))
    r, s, a = L1["G_responder (either role)"], L1["G_starter (either role)"], L2["starter's proposal was the accepted one (rate)"]
    return p4, r["ci95_delta"][0] > 0, (incl0(s["ci95_delta"]) and abs(s["delta"]) <= 0.10), a["ci95_delta"][1] < 0

VERDICTS = {
    f"{QR}/confirmatory_responder_verdicts.json": (f"{QR}/confirmatory_responder.json",
        lambda X: dict(zip(["P4_validity_gate", "P1_primary", "P2", "P3"], rules_responder(X)))),
    f"{H}/analysis_quality_feasible_verdicts.json": (f"{H}/analysis_quality_feasible.json",
        lambda Qj: {"A1_primary_nash_gt_0": Qj["metrics"]["nash_ratio"]["ci95_delta"][0] > 0}),
    f"{H}/responder_feasible_verdicts.json": (f"{H}/responder_feasible.json",
        lambda X: dict(zip(["A5_validity_gate", "A4_P1_responder", "A4_P2_starter_flat", "A4_P3_starter_accepted_falls"], rules_responder(X)))),
    f"{H}/infeasible_verdicts.json": (f"{H}/infeasible.json",
        lambda I: {"B1_proposals_fewer": neg(I["metrics"]["proposals_used"]), "B2_walk_more": pos(I["metrics"]["walk"]),
                   "B3_invalid_zero": (I["metrics"]["invalid_acceptance"]["bayes"] or 0) == 0 and (I["metrics"]["invalid_acceptance"]["prior_only"] or 0) == 0,
                   "B4_verdict_falls": neg(I["metrics"]["final_max_pacc_own_ir"]),
                   "B5_correct_no_deal": [I["metrics"]["correct_no_deal"]["bayes"], I["metrics"]["correct_no_deal"]["prior_only"]]}),
    f"{C}/analysis_counter_verdicts.json": (f"{C}/analysis_counter.json",
        lambda Aj: {"C1_verdict": Aj["C1_verdict"], "C1_delta": Aj["C1_primary_delta_acceptable_seller_minus_buyer"]["delta"],
                    "C1_ci95": Aj["C1_primary_delta_acceptable_seller_minus_buyer"]["ci95_delta"]}),
}


def a5_diagnosis():
    """Re-derive results/jj_hard_20261025/a5_gate_diagnosis.json from the logs."""
    arms = {}
    for a in ("bayes", "prior_only"):
        recs = [json.loads(l) for l in open(os.path.join(ROOT, H, a, "runs.jsonl")) if l.strip()]
        arms[a] = [r for r in recs if "result" in r and r["score"]["feasible"]]
    shown = defaultdict(set); dist = {}; mean_p = {}
    for arm, recs in arms.items():
        c = Counter(); p = []
        for r in recs:
            st = r["starting_side"]; s0 = r["jev_steps"][st][0]
            shown[(arm, st)].add(tuple(sorted((k, round(v, 12)) for k, v in s0["acceptance"].items())))
            first = next(t["deal_id"] for t in r["turns"] if t["action"] == "OFFER"); c[first] += 1; p.append(s0["acceptance"][first])
        dist[arm] = c.most_common(8); mean_p[arm] = sum(p) / len(p)
    leak = any(len(shown[(a, ro)]) != 1 for a in arms for ro in ("buyer", "seller")) or any(shown[("bayes", ro)] != shown[("prior_only", ro)] for ro in ("buyer", "seller"))
    return {"pipeline_leak": leak, "distinct_dicts_per_role_per_arm": {f"{a}/{ro}": len(shown[(a, ro)]) for a in arms for ro in ("buyer", "seller")},
            "identical_across_arms": {ro: shown[("bayes", ro)] == shown[("prior_only", ro)] for ro in ("buyer", "seller")},
            "starter_first_deal_distribution": {a: dist[a] for a in arms}, "mean_shown_pacc_of_chosen": {a: round(mean_p[a], 4) for a in arms},
            "placebo_tests_so_far": 12, "exclusions_so_far": 1, "false_fail_rate_per_run_under_null_4_tests": round(1 - 0.95 ** 4, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="Substring filter on frozen path.")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    # pooled logs are derived; rebuild if absent
    if not os.path.exists(f"{QP}/bayes/runs.jsonl"):
        subprocess.run(["sh", f"{QP}/REGENERATE.sh"], check=True, capture_output=True)
    ok = True
    tmp = tempfile.mkdtemp(prefix="verify_cited_")
    regenerated = {}
    for frozen, module, argv_, keys in REGISTRY:
        if args.only and args.only not in frozen: continue
        try:
            new = regen(module, argv_, os.path.join(tmp, os.path.basename(frozen)))
            old = json.load(open(frozen))
            same = norm(strip(old, keys)) == norm(strip(new, keys))
            regenerated[frozen] = new
        except Exception as e:
            same = False; print(f"  ERROR {frozen}: {e}")
        ok &= same
        print(f"{'UNCHANGED' if same else 'DRIFTED  '}  {frozen}")
    for vpath, (src, rule) in VERDICTS.items():
        if args.only and args.only not in vpath: continue
        try:
            base = regenerated.get(src) or json.load(open(src))
            same = norm(rule(base)) == norm(json.load(open(vpath)))
        except Exception as e:
            same = False; print(f"  ERROR {vpath}: {e}")
        ok &= same
        print(f"{'UNCHANGED' if same else 'DRIFTED  '}  {vpath}  (rules re-applied)")
    if not args.only or args.only in f"{H}/a5_gate_diagnosis.json":
        same = norm(a5_diagnosis()) == norm(json.load(open(f"{H}/a5_gate_diagnosis.json")))
        ok &= same
        print(f"{'UNCHANGED' if same else 'DRIFTED  '}  {H}/a5_gate_diagnosis.json  (re-derived from logs)")
    print("\nVERIFY:", "PASS — every frozen analysis, verdict and diagnosis regenerates identically" if ok else "FAIL — see DRIFTED lines")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
