"""FLYW-1 reported tables (not scored): the numbers RESULTS_FLYW1 quotes beyond the analyzer's verdicts.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_report \
        --analysis llm_rl_scripts/twenty_questions/jev/results/flyw1_analysis_20261119.json \
        --part2 llm_rl_scripts/twenty_questions/jev/results/flyw1_part2_20261118 \
        --out llm_rl_scripts/twenty_questions/jev/results/flyw1_report_20261119.json

* **by_construction**: Part 1's cells (``analysis.json`` ``part1.cells``) pooled over each
  construction's maps, weighted by each cell's game count. That is the analyzer's own pooling
  (``pooled``: the mean over every game of every map in the construction). Measures: probes,
  wrong actions per game, expert share, B/C/D fallbacks, and ``cost_r0.2`` (1 per probe, 10 per
  expert call, 50 per wrong action). It also gives the class-map basis ``p0a_pure_cost`` per
  construction (mean over its maps) beside B's observed flip-0 probes.
* **trigger_points**: Part 2, arms B and D, noisy flips, trigger ``mean_r0.2_pu1.0``. A game
  reaches the trigger point when it does not end in a right action before it. Per arm, map group
  (``align``+``cat17``, ``rand``) and game kind (``pool`` = in-pool and same-class, ``new`` =
  new-class novels): the weighted number of trigger points, the share where T-loss acted
  instead of calling the expert (where the two triggers disagree), the share of those actions
  that were right, and quantiles of P(X*) at the first decision. An empty P(X*) log means
  every known class had already been struck by earlier wrong actions.
* **triggers_by_ratio**: Part 2, B and D, T-mass against T-loss (``mean``) and its ``max``
  shadow, each costed at the same ratio (0.5, 0.2, 0.05), per map group and flip 0 / noisy.
  The analyzer's reported block costs T-mass at 0.2 only.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from .analyze_flyw1 import C_EXPERT, load_jobs, load_maps, p2_outcome

ARMS = ("A", "B", "C", "Cbin", "D")
FIELDS = ("probes", "wrong_per_game", "expert_share", "fallbacks", "cost_r0.2")
ORDER = ("align", "mix05", "mix15", "rand", "category")
TRIG = "mean_r0.2_pu1.0"
NOISY = (0.05, 0.1)


def by_construction(cells: dict, maps: dict) -> dict:
    agg = defaultdict(lambda: defaultdict(float))
    for c in cells.values():
        a = agg[(maps[c["map"]]["alignment_construction"], c["flip"], c["arm"])]
        a["n"] += c["n"]
        for f in FIELDS:
            if c.get(f) is not None:
                a[f] += c["n"] * c[f]
                a["n_" + f] += c["n"]
    out = {}
    for (con, flip, arm), a in sorted(agg.items()):
        out["{}|{}|{}".format(con, flip, arm)] = {
            f: round(a[f] / a["n_" + f], 4) for f in FIELDS if a["n_" + f]}
    basis = defaultdict(list)
    for m in maps.values():
        pc = m.get("measures", {}).get("p0a_pure_cost")
        if pc is not None:
            basis[m["alignment_construction"]].append(pc)
    out_basis = {c: {"p0a_pure_cost_basis": round(sum(v) / len(v), 3),
                     "B_probes_flip0": out["{}|0.0|B".format(c)]["probes"],
                     "B_fallbacks_flip0": out["{}|0.0|B".format(c)]["fallbacks"]}
                 for c, v in basis.items() if "{}|0.0|B".format(c) in out}
    return {"cells": out, "basis_vs_observed": out_basis}


def trigger_points(part2: str, maps: dict) -> dict:
    st = defaultdict(lambda: defaultdict(float))
    pstar = defaultdict(list)
    for j in load_jobs(part2):
        jb = j["job"]
        if jb["arm"] not in ("B", "D") or jb["flip"] not in NOISY:
            continue
        grp = "rand" if maps[jb["map_id"]]["alignment_construction"] == "rand" else "align+cat17"
        for g in j["games"]:
            if g["outcome"] == "right":
                continue
            t = g["tloss"][TRIG]
            kind = "pool" if g["kind"] in ("in_pool", "same") else "new"
            k = "{}|{}|{}".format(jb["arm"], grp, kind)
            w = g["weight"]
            acted = t["extra_actions"] >= 1
            st[k]["trigger_points"] += w
            st[k]["acted"] += w * acted
            st[k]["acted_right"] += w * (acted and t["outcome"] != "expert")
            if t["p_star"]:
                pstar[k].append(t["p_star"][0])
    out = {}
    for k, s in sorted(st.items()):
        v = sorted(pstar[k])
        q = (lambda p: round(v[min(len(v) - 1, int(p * len(v)))], 3)) if v else (lambda p: None)
        out[k] = {"trigger_points_weighted": s["trigger_points"],
                  "tloss_acted_share": round(s["acted"] / s["trigger_points"], 4),
                  "acted_right_share": (round(s["acted_right"] / s["acted"], 3)
                                        if s["acted"] else None),
                  "p_star_n": len(v), "p_star_median": q(0.5), "p_star_p90": q(0.9),
                  "p_star_share_above_0.8": (round(sum(x > 0.8 for x in v) / len(v), 4)
                                             if v else None)}
    return out


def triggers_by_ratio(part2: str, maps: dict) -> dict:
    """T-mass against T-loss (mean, p_U as run) with both costed at the same ratio.
    ``analysis.json``'s reported block costs T-mass at 0.2 only, so its T-loss costs at
    0.5 and 0.05 are not comparable with the T-mass cost beside them."""
    acc = defaultdict(lambda: defaultdict(float))
    for j in load_jobs(part2):
        jb = j["job"]
        if jb["arm"] not in ("B", "D"):
            continue
        cls = maps[jb["map_id"]]["classes"]
        grp = "rand" if maps[jb["map_id"]]["alignment_construction"] == "rand" else "align+cat17"
        fl = "flip0" if jb["flip"] == 0.0 else "noisy"
        for ratio in (0.5, 0.2, 0.05):
            for trig in ("mass", "mean_r{}_pu1.0".format(ratio), "max_r{}_pu1.0".format(ratio)):
                k = "{}|{}|{}|r{}|{}".format(jb["arm"], grp, fl, ratio,
                                             trig.split("_")[0])
                a = acc[k]
                for g in j["games"]:
                    o = p2_outcome(g, trig, cls)
                    w = g["weight"]
                    a["w"] += w
                    a["cost"] += w * (o["probes"] + o["expert"] * C_EXPERT
                                      + o["wrong"] * C_EXPERT / ratio)
                    if g["kind"] in ("in_pool", "same"):
                        a["w_sp"] += w
                        a["sp"] += w * o["expert"]
                    else:
                        a["w_new"] += w
                        a["miss"] += w * o["acted_at_trigger"]
    return {k: {"cost": round(a["cost"] / a["w"], 3),
                "spurious": round(a["sp"] / a["w_sp"], 4),
                "missed": round(a["miss"] / a["w_new"], 4)} for k, a in sorted(acc.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--part2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    maps = load_maps()
    with open(a.analysis) as f:
        an = json.load(f)
    res = {"analysis": os.path.basename(a.analysis), "part2": os.path.basename(a.part2.rstrip("/")),
           "trigger": TRIG, "by_construction": by_construction(an["part1"]["cells"], maps),
           "trigger_points": trigger_points(a.part2, maps),
           "triggers_by_ratio": triggers_by_ratio(a.part2, maps)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    bc = res["by_construction"]["cells"]
    for fld in ("cost_r0.2", "probes", "wrong_per_game", "expert_share"):
        print("==", fld)
        for con in ORDER:
            for flip in (0.0, 0.05, 0.1):
                print("{:9s} {:4} ".format(con, flip) + " ".join(
                    "{:>7}".format(bc.get("{}|{}|{}".format(con, flip, arm), {}).get(fld, "-"))
                    for arm in ARMS))
    print(json.dumps(res["by_construction"]["basis_vs_observed"], indent=1))
    print(json.dumps(res["trigger_points"], indent=1))
    for k, v in res["triggers_by_ratio"].items():
        print(k, v)


if __name__ == "__main__":
    main()
