"""Pre-registered decision rules for CD-SEL-1: Jev shown what the history implies
(design Phase 2 selector arms) on the live Car-Dealer, primary cell H = 4, c_q = 0.

Rules are fixed HERE, in code, before any Jev call of the study; the registration
note (PREREGISTRATION_SEL.md) quotes them. Input: the study's analysis.json
(analyze_phase1.py), the frozen scripted analysis of the same regime (random-arm
gate) and the frozen Phase 1 Jev analysis (drift check, reported, not a gate).
Output: verdicts.json next to the analysis.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.verdicts_sel \
        --analysis results/<sel dir>/analysis.json \
        --scripted results/live_phase1_scripted3_20261029/h4_cq0/analysis.json \
        --phase1 results/live_phase1_jev_h4_cq0_20261029/analysis.json

Verdict vocabulary: verdicts_live.ci_verdict (95 % percentile cluster-bootstrap CI over
the 36 type pairs; 'materially positive' needs the lower bound above DELTA = 0.02 Y).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .verdicts_live import DELTA, ci_verdict, est, load

PRIMARY_CELLS = "all"            # the shown information bears on every decision, not only the ask-versus-offer one
P1_CONTRAST = "jev_ask_both-jev_ask_plain"
REQUIRED_ARMS = ("exact_ask", "exact_noask", "random", "jev_ask_plain", "jev_ask_belief", "jev_ask_pacc",
                 "jev_ask_prior", "jev_ask_placebo", "jev_ask_both", "jev_noask_both")
SELECTIVITY_ARMS = ("jev_ask_belief", "jev_ask_both")


def _contrast(a, name, cs):
    return (a.get("contrasts", {}).get(name) or {}).get(cs)


def gap_closed(a, cs=PRIMARY_CELLS):
    """Point estimate of the share of the plain arm's gap to the exact planner that the
    both arm closes: (both − plain) / (exact_ask − plain). Point only: the two contrasts
    are bootstrapped separately, so no interval is formed for the ratio."""
    num = _contrast(a, P1_CONTRAST, cs)
    den = _contrast(a, "jev_ask_plain-exact_ask", cs)
    if not num or not den or num.get("point") is None or den.get("point") is None or abs(den["point"]) < 1e-12:
        return None
    return num["point"] / (-den["point"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--scripted", help="frozen scripted analysis.json of the same regime (random-arm gate)")
    ap.add_argument("--phase1", help="frozen Phase 1 Jev analysis.json of the same regime (drift check, reported)")
    args = ap.parse_args(argv)
    A = load(args.analysis); SCR = load(args.scripted); P1 = load(args.phase1)
    out = {"study": "CD-SEL-1", "delta": DELTA, "primary_cells": PRIMARY_CELLS, "primary_contrast": P1_CONTRAST,
           "gates": {}, "rules": {}, "checks": {}}

    # ---- sanity gates: fail -> fix the pipeline, do not read results ----
    arms = A["arms"]
    g = {"error_stubs": A["error_stubs"], "missing_arms": [x for x in REQUIRED_ARMS if x not in arms],
         "episodes_per_arm": {k: v["all"].get("n") for k, v in A["per_arm"].items()},
         "well_formed": {k: v.get("well_formed_share") for k, v in A.get("position", {}).items()}}
    g["pass"] = (A["error_stubs"] == 0 and not g["missing_arms"] and len(set(g["episodes_per_arm"].values())) == 1
                 and all((w or 0) >= 1.0 - 1e-12 for w in g["well_formed"].values()))
    out["gates"]["pipeline"] = g
    shuffle = {k: len([c for c in v.get("counts", {}).values() if c > 0]) for k, v in A.get("position", {}).items() if isinstance(v, dict)}
    out["gates"]["shuffle_active"] = {"positions_used_per_arm": shuffle, "pass": all(n >= 2 for n in shuffle.values()) if shuffle else False}
    if SCR is not None and "random" in A["per_arm"] and "random" in SCR["per_arm"]:
        m = A["per_arm"]["random"]["all"]["mean_normalized_net_gain"]
        ci = SCR["per_arm"]["random"]["all"].get("ci95_normalized_net_gain")
        out["gates"]["random_matches_frozen"] = {"study_random_mean_Y": m, "frozen_random_ci95": ci,
                                                 "pass": (ci is not None and ci[0] - 1e-12 <= m <= ci[1] + 1e-12)}
    c = _contrast(A, "exact_ask-exact_noask", "all")
    ceil = (A.get("ceilings", {}).get("all") or {}).get("mean_delta_menu_Y")
    if c and c.get("point") is not None and ceil is not None:
        out["gates"]["exact_pair_reproduces_ceiling"] = {"contrast": est(c), "ceiling_delta_menu_Y": ceil,
                                                         "pass": c["ci95"][0] - 1e-12 <= ceil <= c["ci95"][1] + 1e-12}
    out["gates_pass"] = all(v.get("pass", False) for v in out["gates"].values())

    # ---- drift check (reported, not a gate): the in-study plain arm against the frozen Phase 1 plain arm ----
    if P1 is not None and "jev_ask_plain" in A["per_arm"] and "jev_ask_plain" in P1["per_arm"]:
        m = A["per_arm"]["jev_ask_plain"]["all"]["mean_normalized_net_gain"]
        ci = P1["per_arm"]["jev_ask_plain"]["all"].get("ci95_normalized_net_gain")
        out["checks"]["plain_arm_drift"] = {
            "study_mean_Y": m, "phase1_mean_Y": P1["per_arm"]["jev_ask_plain"]["all"]["mean_normalized_net_gain"], "phase1_ci95": ci,
            "inside_phase1_ci": (ci is not None and ci[0] - 1e-12 <= m <= ci[1] + 1e-12),
            "study_walk_rate": (A.get("choices", {}).get("jev_ask_plain") or {}).get("walk_rate"),
            "phase1_walk_rate": (P1.get("choices", {}).get("jev_ask_plain") or {}).get("walk_rate"),
            "study_questions_per_episode": A["per_arm"]["jev_ask_plain"]["all"].get("mean_questions_asked"),
            "phase1_questions_per_episode": P1["per_arm"]["jev_ask_plain"]["all"].get("mean_questions_asked"),
            "note": "jev-latest is stochastic server-side and pinned by name only; a shift here is model drift, not a pipeline fault, and is disclosed with the results"}

    def rule(key, quantity, prediction, c, kind, extra=None):
        r = {"quantity": quantity, "prediction": prediction, "estimate": est(c), "verdict": ci_verdict(c, kind)}
        if extra:
            r.update(extra)
        out["rules"][key] = r

    CQ = A.get("contrasts_questions", {}); CW = A.get("contrasts_walk", {})
    rule("P1", f"Δ Y, {P1_CONTRAST}, {PRIMARY_CELLS} cells (PRIMARY): belief in words + posterior P(accept) shown",
         f"> 0; material if lower CI bound > {DELTA}", _contrast(A, P1_CONTRAST, PRIMARY_CELLS), "pos_material",
         {"gap_closed_point": gap_closed(A), "both_minus_exact_ask": est(_contrast(A, "jev_ask_both-exact_ask", PRIMARY_CELLS)),
          "plain_minus_exact_ask": est(_contrast(A, "jev_ask_plain-exact_ask", PRIMARY_CELLS))})
    rule("S1", "Δ Y, jev_ask_belief-jev_ask_plain, all cells (the posterior in words alone)", "> 0",
         _contrast(A, "jev_ask_belief-jev_ask_plain", "all"), "pos_material")
    rule("S2", "Δ Y, jev_ask_pacc-jev_ask_plain, all cells (posterior P(accept) alone; B_passive selector)", "> 0",
         _contrast(A, "jev_ask_pacc-jev_ask_plain", "all"), "pos_material")
    rule("S3", "Δ Y, jev_ask_pacc-jev_ask_prior, all cells (passive learning in the selector: B_passive − A_prior)", "> 0",
         _contrast(A, "jev_ask_pacc-jev_ask_prior", "all"), "pos")
    rule("S4", "Δ Y, jev_ask_placebo-jev_ask_plain, all cells (a constant annotation; salience check)", "= 0 (CI includes 0)",
         _contrast(A, "jev_ask_placebo-jev_ask_plain", "all"), "zero")
    rule("S5", "Δ questions asked / episode, jev_ask_belief-jev_ask_plain, all cells (seeing the uncertainty -> asking)", "> 0",
         (CQ.get("jev_ask_belief-jev_ask_plain") or {}).get("all"), "pos",
         {"questions_per_episode": {a: A["per_arm"][a]["all"].get("mean_questions_asked") for a in arms if a.startswith("jev_ask")}})
    rule("S5b", "Δ questions asked / episode, jev_ask_both-jev_ask_plain, all cells", "> 0",
         (CQ.get("jev_ask_both-jev_ask_plain") or {}).get("all"), "pos")
    c6 = _contrast(A, "jev_ask_both-jev_noask_both", "focal_opens")
    rule("S6", "Δ Y, jev_ask_both-jev_noask_both, focal-opens cells (the value of having questions on the menu once the belief is shown)", "> 0",
         c6, "pos_material", {"captured_fraction_of_delta_menu": (c6 or {}).get("captured_fraction", {}).get("delta_menu"),
                             "ceiling_delta_menu_Y": (A.get("ceilings", {}).get("focal_opens") or {}).get("mean_delta_menu_Y")})
    rule("S7", "Δ Y, jev_ask_both-exact_ask, all cells (the remaining gap to the planner)", "< 0",
         _contrast(A, "jev_ask_both-exact_ask", "all"), "neg")
    rule("S8", "Δ walk rate, jev_ask_pacc-jev_ask_plain, all cells (acceptance numbers -> fewer walk-outs)", "< 0",
         (CW.get("jev_ask_pacc-jev_ask_plain") or {}).get("all"), "neg",
         {"walk_rate": {a: (A.get("choices", {}).get(a) or {}).get("walk_rate") for a in arms}})
    rule("S8b", "Δ walk rate, jev_ask_both-jev_ask_plain, all cells", "< 0", (CW.get("jev_ask_both-jev_ask_plain") or {}).get("all"), "neg")
    for arm in SELECTIVITY_ARMS:
        s = A.get("selectivity", {}).get(arm)
        c_sel = None if not s or s.get("difference") is None else {"point": s["difference"], "ci95": s.get("difference_ci95"), "n_clusters": s.get("n_clusters")}
        if c_sel and c_sel["ci95"] is None:
            c_sel = None
        rule(f"S9_{arm}", f"P(ask | ASK strictly optimal) − P(ask | ASK legal, not optimal), {arm}", "> 0", c_sel, "pos",
             {"p_ask_given_optimal": (s or {}).get("p_ask_given_optimal"), "p_ask_given_not_optimal": (s or {}).get("p_ask_given_not_optimal"),
              "asked_when_optimal": (s or {}).get("asked_when_optimal"), "decisions_ask_optimal": (s or {}).get("decisions_ask_optimal")})
    out["descriptive"] = {
        "mean_Y_all": {a: A["per_arm"][a]["all"].get("mean_normalized_net_gain") for a in arms},
        "mean_Y_focal_opens": {a: A["per_arm"][a]["focal_opens"].get("mean_normalized_net_gain") for a in arms},
        "agreement_rate": {a: A["per_arm"][a]["all"].get("agreement_rate") for a in arms},
        "offer_tags": {a: (A.get("choices", {}).get(a) or {}).get("offer_tag") for a in arms if a.startswith("jev_")},
        "prior_vs_placebo": est(_contrast(A, "jev_ask_prior-jev_ask_placebo", "all")),
        "both_vs_pacc": est(_contrast(A, "jev_ask_both-jev_ask_pacc", "all")),
        "both_vs_belief": est(_contrast(A, "jev_ask_both-jev_ask_belief", "all")),
        "position": A.get("position"), "ceilings": A.get("ceilings"),
    }
    dest = os.path.join(os.path.dirname(args.analysis), "verdicts.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}; gates pass = {out['gates_pass']}")
    for k, v in out["rules"].items():
        e = v.get("estimate")
        s = "" if not e or "point" not in e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        print(f"  {k:20s} {v['verdict']:22s} {s:36s} {v['quantity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
