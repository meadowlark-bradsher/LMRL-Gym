"""Apply the pre-registered decision rules of PREREGISTRATION_B.md to a frozen
analysis.json and write the verdicts next to it. Rules are fixed here in code;
nothing is read from the data to choose a rule.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.verdicts_b \
        results/binned_b_jev_r2_easy_<date>/analysis.json --horizon primary
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def sign_rule(d, predicted: str):
    """predicted in {'neg', 'pos', 'not_neg'} -> verdict string."""
    if d is None:
        return "undefined"
    lo, hi = d["ci95"]
    if predicted == "neg":
        if hi < 0:
            return "confirmed"
        if lo > 0:
            return "opposite sign"
        return "not supported"
    if predicted == "pos":
        if lo > 0:
            return "confirmed"
        if hi < 0:
            return "opposite sign"
        return "not supported"
    if predicted == "not_neg":  # S2: prediction is that the CI does NOT lie below 0
        return "confirmed" if not (hi < 0) else "not supported (EIG shown lowered decision regret)"
    raise ValueError(predicted)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("analysis")
    p.add_argument("--horizon", choices=("primary", "secondary"), required=True)
    args = p.parse_args(argv)
    a = json.load(open(args.analysis))
    pairs = a["pairs"]
    out = {"source": os.path.relpath(args.analysis), "horizon": args.horizon, "rules": {}}

    def pair(x, y, field):
        return pairs.get(f"{x} - {y}", {}).get(field)

    if args.horizon == "primary":
        d = pair("jev_plain", "random", "total_decision_regret")
        out["rules"]["P1"] = {"quantity": "Δ total decision regret, jev_plain − random", "prediction": "< 0", "estimate": d, "verdict": sign_rule(d, "neg")}
        d = pair("jev_eig", "jev_plain", "eig_regret_per_decision_bits")
        out["rules"]["S1"] = {"quantity": "Δ per-decision EIG regret, jev_eig − jev_plain", "prediction": "< 0", "estimate": d, "verdict": sign_rule(d, "neg")}
        d = pair("jev_eig", "jev_plain", "total_decision_regret")
        out["rules"]["S2"] = {"quantity": "Δ total decision regret, jev_eig − jev_plain", "prediction": "not < 0", "estimate": d, "verdict": sign_rule(d, "not_neg")}
        d = pair("jev_eig", "jev_plain", "questions_asked")
        out["rules"]["S3"] = {"quantity": "Δ questions asked, jev_eig − jev_plain", "prediction": "> 0", "estimate": d, "verdict": sign_rule(d, "pos")}
        # S4: pooled Jev decisions -> use the per-arm tercile correlations; pool rule: both arms reported, verdict on the pooled-by-arm pattern
        s4 = {}
        for arm in ("jev_plain", "jev_eig"):
            j = a["by_arm"].get(arm, {}).get("jev")
            if not j:
                continue
            large = j["by_gap_tercile"].get("large_gap", {}).get("spearman_entropy_vs_regret")
            small = j["by_gap_tercile"].get("small_gap", {}).get("spearman_entropy_vs_regret")
            lg_ok = large is not None and large.get("ci95") and large["ci95"][0] > 0
            sm_ok = small is not None and small.get("ci95") and small["ci95"][0] <= 0 <= small["ci95"][1]
            s4[arm] = {"large_gap": large, "small_gap": small, "overall": j["spearman_entropy_vs_decision_regret"],
                       "pattern": "supported" if (lg_ok and sm_ok) else "not supported"}
        out["rules"]["S4"] = {"quantity": "Spearman(choice entropy, decision regret) by Q-gap tercile", "prediction": "large-gap CI > 0 and small-gap CI includes 0", "per_arm": s4}
        for arm in ("jev_plain", "jev_eig"):
            d = pair(arm, "exact", "learner_utility")
            out["rules"][f"S5_{arm}"] = {"quantity": f"Δ learner utility, {arm} − exact", "prediction": "< 0", "estimate": d, "verdict": sign_rule(d, "neg")}
    else:
        d = pair("jev_plain", "random", "total_decision_regret")
        out["rules"]["S6"] = {"quantity": "Δ total decision regret, jev_plain − random (6 rounds)", "prediction": "< 0", "estimate": d, "verdict": sign_rule(d, "neg")}
        for arm in ("jev_plain", "jev_eig"):
            d = pair(arm, "exact", "learner_utility")
            out["rules"][f"S5_{arm}_6r"] = {"quantity": f"Δ learner utility, {arm} − exact (6 rounds)", "prediction": "< 0", "estimate": d, "verdict": sign_rule(d, "neg")}

    # sanity gates
    gates = {}
    for arm in ("jev_plain", "jev_eig"):
        j = a["by_arm"].get(arm, {}).get("jev")
        gates[f"well_formed_{arm}"] = {"value": None if not j else j["well_formed_share"], "pass": bool(j and j["well_formed_share"] == 1.0)}
        gates[f"positions_{arm}"] = None if not j else j["chosen_position_counts"]
    out["gates"] = gates
    dest = os.path.join(os.path.dirname(args.analysis), "verdicts.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out["rules"].items():
        if "verdict" in v:
            e = v["estimate"]
            est = "—" if e is None else f"{e['delta']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
            print(f"{k:14s} {v['verdict']:22s} {est}   {v['quantity']}")
        else:
            for arm, r in v["per_arm"].items():
                print(f"{k:14s} {r['pattern']:22s} {arm}")
    print("gates:", {k: v for k, v in gates.items() if k.startswith("well")})
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
