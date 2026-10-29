"""Analyze one Phase 1 result dir (``run_phase1.py`` output) into ``analysis.json``.

Reads ``runs.jsonl.gz``, ``summary.json`` and ``ceilings.json``; skips error stubs.
Clusters for every paired contrast are (focal, opener, focal_ranking, opp_ranking);
the paired difference is over cluster means with a percentile cluster bootstrap.
Ceilings are exact (no uncertainty); the captured fraction divides the bootstrap
draws of a contrast by the exact mean ceiling of the same cell set.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir results/<dir>
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple

ASK_NOASK_PAIRS = [("exact_ask", "exact_noask"), ("jev_ask_plain", "jev_noask_plain"), ("jev_ask_eig", "jev_noask_eig")]
NOASK_ARMS = {"exact_noask", "jev_noask_plain", "jev_noask_eig"}
CELL_SETS = ("all", "focal_opens", "opponent_opens")


def _clean(x):
    if isinstance(x, float):
        if x != x:
            return None
        return 0.0 if x == 0 else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def load_runs(path: str) -> Tuple[List[dict], int]:
    recs, stubs = [], 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                stubs += 1
                continue
            recs.append(r)
    return recs, stubs


def cluster_of(r: dict) -> tuple:
    return (r["focal"], r["opener"], "/".join(r["focal_ranking"]), "/".join(r["private_truth_evaluator_only"]["opp_ranking"]))


def in_set(r_or_key, cell_set: str) -> bool:
    focal, opener = (r_or_key["focal"], r_or_key["opener"]) if isinstance(r_or_key, dict) else (r_or_key[0], r_or_key[1])
    if cell_set == "all":
        return True
    return (focal == opener) if cell_set == "focal_opens" else (focal != opener)


def y(r: dict) -> float:
    return r["normalized_net_gain"]["float"]


def per_arm_stats(recs: List[dict]) -> dict:
    if not recs:
        return {"n": 0}
    n_dec = sum(r["n_decisions"] for r in recs)
    return {
        "n": len(recs),
        "mean_normalized_net_gain": mean(y(r) for r in recs),
        "mean_net_gain": mean(r["net_gain"]["float"] for r in recs),
        "mean_gross_gain": mean(r["gross_gain"]["float"] for r in recs),
        "agreement_rate": mean(1.0 if r["agreement"] else 0.0 for r in recs),
        "mean_questions_asked": mean(r["questions_asked"] for r in recs),
        "mean_proposals_made": mean(r["proposals_made"] for r in recs),
        "mean_total_decision_regret": mean(r["total_decision_regret"] for r in recs),
        "zero_regret_decision_share": (sum(r["zero_regret_decisions"] for r in recs) / n_dec) if n_dec else None,
        "mean_total_eig_regret_bits": mean(r["total_eig_regret_bits"] for r in recs),
        "mean_final_entropy_bits": mean(r["final_entropy_bits"] for r in recs),
        "n_decisions": n_dec,
    }


def cluster_means(recs: List[dict], cell_set: str) -> Dict[tuple, float]:
    by = defaultdict(list)
    for r in recs:
        if in_set(r, cell_set):
            by[cluster_of(r)].append(y(r))
    return {k: mean(v) for k, v in by.items()}


def paired_bootstrap(a: Dict[tuple, float], b: Dict[tuple, float], B: int, seed: int,
                     ceilings: Optional[Dict[str, float]] = None) -> dict:
    keys = sorted(set(a) & set(b))
    skipped = len(set(a) ^ set(b))
    if not keys:
        return {"n_clusters": 0, "skipped_clusters": skipped, "point": None, "ci95": None}
    diffs = [a[k] - b[k] for k in keys]
    point = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    draws = []
    for _ in range(B):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo, hi = draws[int(0.025 * B)], draws[min(B - 1, int(0.975 * B))]
    out = {"n_clusters": n, "skipped_clusters": skipped, "point": point, "ci95": [lo, hi], "B": B}
    if ceilings:
        out["captured_fraction"] = {}
        for name, c in ceilings.items():
            if c is None or abs(c) < 1e-12:
                out["captured_fraction"][name] = {"ceiling": c, "point": None, "ci95": None}
            else:
                out["captured_fraction"][name] = {"ceiling": c, "point": point / c, "ci95": [lo / c, hi / c]}
    return out


def ceiling_stats(ceil: dict, y_scale: Optional[float] = None) -> dict:
    """Means of the exact root values per cell set. ``y_scale`` = U_max - r converts the
    utility-unit ceilings to Y units (normalized net gain), the unit of every contrast; the
    ``*_Y`` fields are the ones a contrast is compared with."""
    cells = ceil.get("cells", {})
    out = {}
    for cs in CELL_SETS:
        sub = {k: v for k, v in cells.items() if in_set(tuple(k.split("|")), cs)}
        if not sub:
            out[cs] = {"n_cells": 0}
            continue
        vals = list(sub.values())

        def m(key):
            xs = [v[key] for v in vals if key in v]
            return mean(xs) if xs else None

        def mdiff(k1, k2):
            xs = [v[k1] - v[k2] for v in vals if k1 in v and k2 in v]
            return mean(xs) if xs else None

        out[cs] = {
            "n_cells": len(vals),
            "mean_delta_full": m("delta_full"),
            "mean_delta_menu": m("delta_menu"),
            "mean_loss_ask": m("loss_ask"),
            "mean_loss_noask": m("loss_noask"),
            "mean_loss_offers_restriction_ask": None if mdiff("full_ask", "offers_restricted_ask") is None else mdiff("full_ask", "offers_restricted_ask"),
            "mean_loss_questions_restriction_ask": None if mdiff("full_ask", "questions_restricted_ask") is None else mdiff("full_ask", "questions_restricted_ask"),
            "cells_with_any_loss": sum(1 for v in vals if max(v.get("loss_ask", 0.0), v.get("loss_noask", 0.0),
                                                            v["full_ask"] - v.get("offers_restricted_ask", v["full_ask"]),
                                                            v["full_ask"] - v.get("questions_restricted_ask", v["full_ask"])) > 1e-9),
            "cells_with_delta_menu_positive": sum(1 for v in vals if v["delta_menu"] > 1e-9),
            "mean_full_ask": m("full_ask"), "mean_full_noask": m("full_noask"),
            "mean_menu_ask": m("menu_ask"), "mean_menu_noask": m("menu_noask"),
        }
        if y_scale:
            out[cs]["y_scale"] = y_scale
            out[cs]["mean_delta_full_Y"] = None if m("delta_full") is None else m("delta_full") / y_scale
            out[cs]["mean_delta_menu_Y"] = None if m("delta_menu") is None else m("delta_menu") / y_scale
    return out


def selectivity(recs: List[dict], B: int, seed: int) -> dict:
    by = defaultdict(lambda: [0, 0, 0, 0])  # cluster -> [asked|opt, n|opt, asked|notopt, n|notopt]
    for r in recs:
        c = by[cluster_of(r)]
        for d in r["decisions"]:
            if not d["ask_legal"]:
                continue
            if d["ask_optimal"]:
                c[1] += 1
                c[0] += d["chosen_kind"] == "ASK"
            else:
                c[3] += 1
                c[2] += d["chosen_kind"] == "ASK"
    tot = [sum(c[i] for c in by.values()) for i in range(4)]
    p_opt = tot[0] / tot[1] if tot[1] else None
    p_not = tot[2] / tot[3] if tot[3] else None
    out = {"asked_when_optimal": tot[0], "decisions_ask_optimal": tot[1], "p_ask_given_optimal": p_opt,
           "asked_when_not_optimal": tot[2], "decisions_ask_legal_not_optimal": tot[3], "p_ask_given_not_optimal": p_not,
           "difference": (p_opt - p_not) if (p_opt is not None and p_not is not None) else None, "n_clusters": len(by)}
    if p_opt is not None and p_not is not None and by:
        rng = random.Random(seed)
        cl = list(by.values())
        n = len(cl)
        draws = []
        for _ in range(B):
            s = [0, 0, 0, 0]
            for _ in range(n):
                c = cl[rng.randrange(n)]
                for i in range(4):
                    s[i] += c[i]
            if s[1] and s[3]:
                draws.append(s[0] / s[1] - s[2] / s[3])
        if draws:
            draws.sort()
            out["difference_ci95"] = [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]]
            out["B_effective"] = len(draws)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    summary = json.load(open(os.path.join(args.dir, "summary.json")))
    ceil_path = os.path.join(args.dir, "ceilings.json")
    ceil = json.load(open(ceil_path)) if os.path.exists(ceil_path) else {"cells": {}}
    recs, stubs = load_runs(os.path.join(args.dir, "runs.jsonl.gz"))
    arms = [a for a in summary["arms"] if any(r["arm"] == a for r in recs)]
    by_arm = {a: [r for r in recs if r["arm"] == a] for a in arms}

    per_arm = {}
    for a in arms:
        per_arm[a] = {cs: per_arm_stats([r for r in by_arm[a] if in_set(r, cs)]) for cs in CELL_SETS}

    # every contrast is in Y units; the ceilings are in utility units -> convert with U_max - r
    r_res = float(Fraction(summary["config"]["reservation"]))
    umax = sorted({round(r["u_max"]["float"], 12) for r in recs})
    assert len(umax) == 1, f"u_max differs across records: {umax}"
    y_scale = umax[0] - r_res
    assert y_scale > 0
    ceilings = ceiling_stats(ceil, y_scale)

    pairs: List[Tuple[str, str]] = [(x, z) for x, z in ASK_NOASK_PAIRS if x in arms and z in arms]
    pairs += [(a, "exact_ask") for a in arms if a != "exact_ask" and "exact_ask" in arms]
    if "jev_ask_eig" in arms and "jev_ask_plain" in arms:
        pairs.append(("jev_ask_eig", "jev_ask_plain"))
    contrasts = {}
    for x, z in pairs:
        name = f"{x}-{z}"
        contrasts[name] = {}
        is_ask_pair = (x, z) in ASK_NOASK_PAIRS
        for cs in CELL_SETS:
            cm = ceilings.get(cs, {})
            ceil_for = {"delta_menu": cm.get("mean_delta_menu_Y"), "delta_full": cm.get("mean_delta_full_Y")} if is_ask_pair and cm.get("n_cells") else None
            contrasts[name][cs] = paired_bootstrap(cluster_means(by_arm[x], cs), cluster_means(by_arm[z], cs), args.B, args.seed, ceil_for)

    sel = {a: selectivity(by_arm[a], args.B, args.seed) for a in arms if a not in NOASK_ARMS}

    position = {}
    for a in arms:
        if a.startswith("jev_"):
            cnt = defaultdict(int)
            for r in by_arm[a]:
                for d in r["decisions"]:
                    if d.get("jev"):
                        cnt[str(d["jev"]["chosen_position"])] += 1
            position[a] = dict(sorted(cnt.items(), key=lambda kv: int(kv[0])))

    analysis = {
        "env": summary.get("env"), "config": summary["config"], "kernel": summary["kernel"], "menu": summary.get("menu"),
        "arms": arms, "episodes": len(recs), "error_stubs": stubs, "elapsed_s": summary.get("elapsed_s"),
        "repeats": summary.get("repeats"), "seed": summary.get("seed"), "bootstrap": {"B": args.B, "seed": args.seed},
        "cluster_definition": "(focal, opener, focal_ranking, opp_ranking)",
        "per_arm": per_arm, "contrasts": contrasts, "ceilings": ceilings, "selectivity": sel, "position": position,
    }
    out = os.path.join(args.dir, "analysis.json")
    with open(out, "w") as f:
        json.dump(_clean(analysis), f, indent=1)
    print(f"wrote {out}: {len(recs)} episodes, {stubs} stubs, arms {arms}")
    for name, c in contrasts.items():
        a = c["all"]
        if a["point"] is not None:
            print(f"  {name:32s} all: {a['point']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] n={a['n_clusters']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
