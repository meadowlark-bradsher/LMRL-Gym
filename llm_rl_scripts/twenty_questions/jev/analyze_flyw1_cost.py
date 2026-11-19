"""FLYW-1 total-cost follow-up: score P1-P4 and S1 (``FLYW1_COST_PREREG.md``).

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw1_cost \
        --run llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_<date> \
        --maps llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_maps_<date> \
        --flyw1-analysis llm_rl_scripts/twenty_questions/jev/results/flyw1_analysis_20261119.json \
        --out llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_analysis_<date>.json

**Cost.** Per game: 1 per probe + ``C_EXPERT`` per expert call + c_wrong per wrong action.
A is scored on each map at its first right-class guess (``score_a_record``), as in FLYW-1.

**Cells.** One construction x flip, pooled over its maps with FLYW-1's game-weighted
pooling (the mean over every paired game of every map in the cell). The scored cells are
the fresh maps' constructions (``mix05``, ``mix15``, ``rand``, generator seeds
``FRESH_SEEDS``) at every flip, plus ``align`` and ``category`` (``cat17``) at the noisy
flips. ``align`` and ``cat17`` at flip 0 are deterministic and already seen: reported, and
checked against FLYW-1's cells.

**SE.** Within-secret, with the secret as the unit (``SE_UNIT = "secret"``). For each
secret, average its paired difference over the cell's maps at each noise seed, take the
sample variance of those averages across seeds, and SE of the cell mean =
sqrt(sum var) / (N sqrt S) over the N secrets. Flip 0 is deterministic (SE 0). S1 pools the
two noisy flips, so its units are (flip, secret).

Why not FLYW-1's (map, secret) unit: A plays each (secret, seed) game once and is scored on
every map, and the keyed noise is shared by every arm's game of that secret and seed, so a
cell's maps are not independent. On FLYW-1 Part 1 (``align``, flip 0.05, A - B at
c_wrong 50) the (map, secret) SE at 20 seeds is 0.53, the secret SE 1.42, and the SD of
per-seed cell means / sqrt S 1.60. The (map, secret) SE is reported beside it as
``se_map_secret``.

**Verdicts.** A test passes when estimate - 2 SE clears its bar, fails when
estimate + 2 SE falls short of it, and is otherwise INCONCLUSIVE, which counts as not
passed. P3 at flip 0 (fresh maps) is wrong_B <= wrong_A on the point value.

The margins and costs the user may still change are the named constants below, and
nowhere else.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze_flyw1 import load_jobs, load_maps
from .flyw1_a_calibration import spearman
from .flyw1_decision import score_a_record

# --------------------------------------------------------------------------- #
# Registered constants (FLYW1_COST_PREREG.md). Change them here only.          #
# --------------------------------------------------------------------------- #
C_EXPERT = 10.0                       # per expert call, throughout
C_WRONG_PRIMARY = 50.0                # P1, P3's companion cost, P4, S1
P1_MARGIN = 2.0                       # P1: A - B >= this, in probe-equivalents per game
P2_C_WRONG = 5.0                      # P2: the cheap wrong action
P2_BAR = 0.0                          # P2: A - B > this at P2_C_WRONG
C_WRONG_REPORTED = (5.0, 20.0, 50.0, 200.0)
FRESH_SEEDS = (3, 4)                  # generator seeds of the fresh maps
FRESH_CONSTRUCTIONS = ("mix05", "mix15", "rand")
FIXED_CONSTRUCTIONS = ("align", "category")
NOISY = (0.05, 0.1)
FLIPS = (0.0, 0.05, 0.1)
ARMS = ("A", "B", "D")


def cost(r: dict, c_wrong: float) -> float:
    return r["probes"] + r["expert"] * C_EXPERT + r["wrong"] * c_wrong


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def load_rows(run_dir: str, maps: Dict[str, dict], max_seed: Optional[int] = None
              ) -> Dict[Tuple[str, str, float], Dict[Tuple[int, str], dict]]:
    """(map, arm, flip) -> (seed, secret) -> {probes, wrong, expert}. A's games are scored
    on every map. Only arms A, B and D are read; other arms in the directory are ignored."""
    a_by: Dict[Tuple[float, int], List[dict]] = {}
    rows: Dict[Tuple[str, str, float], Dict[Tuple[int, str], dict]] = defaultdict(dict)
    for j in load_jobs(run_dir, slim=True):
        jb = j["job"]
        if jb["arm"] not in ARMS or (max_seed is not None and jb["seed"] > max_seed):
            continue
        if jb["arm"] == "A":
            a_by[(jb["flip"], jb["seed"])] = j["games"]
        elif jb["map_id"] in maps:
            for g in j["games"]:
                rows[(jb["map_id"], jb["arm"], jb["flip"])][(jb["seed"], g["secret"])] = {
                    "probes": g["probes"], "wrong": g["wrong"],
                    "expert": g["outcome"] != "right"}
    for m, mp in maps.items():
        cls = mp["classes"]
        for (flip, seed), games in a_by.items():
            for g in games:
                r = score_a_record(g, cls)
                rows[(m, "A", flip)][(seed, g["secret"])] = {
                    "probes": r["probes"], "wrong": r["wrong"],
                    "expert": r["outcome"] != "right"}
    return rows


def construction(mp: dict) -> str:
    return mp["alignment_construction"]


def cell_maps(maps: Dict[str, dict], c: str, fresh_seeds: Sequence[int]) -> List[str]:
    if c in FRESH_CONSTRUCTIONS:
        return sorted(m for m, mp in maps.items()
                      if construction(mp) == c and mp["seed"] in fresh_seeds)
    return sorted(m for m, mp in maps.items() if construction(mp) == c)


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #

SE_UNIT = "secret"                    # see the module docstring; "map_secret" is FLYW-1's


def paired(rows, ms: Sequence[str], x: str, y: str, flips: Sequence[float],
           value, unit: str = SE_UNIT) -> Tuple[Optional[float], float, int, int]:
    """Mean over every paired game of value(x) - value(y), and its within-secret SE with
    ``unit`` "secret" (maps averaged per (secret, seed)) or "map_secret" (FLYW-1's).
    Returns (mean, se, games, seeds)."""
    diffs: List[float] = []
    acc: Dict[tuple, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    seeds = set()
    for m in ms:
        for f in flips:
            rx, ry = rows.get((m, x, f), {}), rows.get((m, y, f), {})
            for k, a in rx.items():
                b = ry.get(k)
                if b is None:
                    continue
                d = value(a) - value(b)
                diffs.append(d)
                u = (f, k[1]) if unit == "secret" else (m, f, k[1])
                acc[u][k[0]].append(d)
                if f != 0.0:
                    seeds.add(k[0])
    if not diffs:
        return None, float("nan"), 0, 0
    if all(f == 0.0 for f in flips):
        return statistics.mean(diffs), 0.0, len(diffs), 1
    tot = 0.0
    for per_seed in acc.values():
        v = [statistics.mean(ds) for ds in per_seed.values()]
        if len(v) > 1:
            tot += statistics.variance(v)
    s = len(seeds)
    se = math.sqrt(tot) / (len(acc) * math.sqrt(s)) if s else float("nan")
    return statistics.mean(diffs), se, len(diffs), s


def verdict(est: Optional[float], se: float, bar: float) -> str:
    """Pass when est - 2 SE > bar; fail when est + 2 SE < bar; else INCONCLUSIVE."""
    if est is None or se != se:
        return "MISSING"
    if est - 2 * se > bar:
        return "PASS"
    if est + 2 * se < bar:
        return "FAIL"
    return "INCONCLUSIVE"


def arm_means(rows, ms: Sequence[str], arm: str, flip: float) -> Optional[dict]:
    v = [r for m in ms for r in rows.get((m, arm, flip), {}).values()]
    if not v:
        return None
    n = len(v)
    out = {"n": n,
           "probes": round(sum(r["probes"] for r in v) / n, 4),
           "wrong_per_game": round(sum(r["wrong"] for r in v) / n, 4),
           "expert_share": round(sum(r["expert"] for r in v) / n, 4)}
    for c in C_WRONG_REPORTED:
        out["cost_c{:g}".format(c)] = round(sum(cost(r, c) for r in v) / n, 4)
    return out


def c_star(x: dict, y: dict) -> dict:
    """Break-even c_wrong between x and y (cost = base + c_wrong x wrong, base = probes +
    C_EXPERT x expert share). When x makes fewer wrong actions, x is cheaper than y whenever
    c_wrong > c* (``dominates`` when c* <= 0). When x makes more, x is cheaper only below
    c*. With equal wrong actions the base decides."""
    bx = x["probes"] + C_EXPERT * x["expert_share"]
    by = y["probes"] + C_EXPERT * y["expert_share"]
    dw = y["wrong_per_game"] - x["wrong_per_game"]
    if dw == 0:
        return {"c_star": None, "note": "dominates" if bx <= by else "x never cheaper"}
    cs = round((bx - by) / dw, 3)
    if dw > 0:
        return {"c_star": cs, "note": "dominates" if cs <= 0 else "x cheaper above c*"}
    return {"c_star": cs, "note": "x never cheaper" if cs <= 0 else "x cheaper only below c*"}


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #

def score(rows, maps: Dict[str, dict], fresh_seeds: Sequence[int] = FRESH_SEEDS,
          flyw1_cells: Optional[dict] = None) -> dict:
    scored_cells = [(c, f) for c in FRESH_CONSTRUCTIONS for f in FLIPS] + \
                   [(c, f) for c in FIXED_CONSTRUCTIONS for f in NOISY]
    res: Dict[str, object] = {
        "constants": {"C_EXPERT": C_EXPERT, "C_WRONG_PRIMARY": C_WRONG_PRIMARY,
                      "P1_MARGIN": P1_MARGIN, "P2_C_WRONG": P2_C_WRONG, "P2_BAR": P2_BAR,
                      "FRESH_SEEDS": list(fresh_seeds), "SE_UNIT": SE_UNIT},
        "scored_cells": ["{}|{}".format(c, f) for c, f in scored_cells]}

    p1, p2, p3 = {}, {}, {}
    for c, f in scored_cells:
        ms = cell_maps(maps, c, fresh_seeds)
        key = "{}|{}".format(c, f)
        c50 = lambda r: cost(r, C_WRONG_PRIMARY)          # noqa: E731
        c5 = lambda r: cost(r, P2_C_WRONG)                # noqa: E731
        est, se, n, s = paired(rows, ms, "A", "B", (f,), c50)
        p1[key] = {"A_minus_B": _r(est), "se": _r(se, 5),
                   "se_map_secret": _r(paired(rows, ms, "A", "B", (f,), c50, "map_secret")[1], 5),
                   "bar": P1_MARGIN, "games": n, "seeds": s, "maps": len(ms),
                   "verdict": verdict(est, se, P1_MARGIN)}
        est, se, n, s = paired(rows, ms, "A", "B", (f,), c5)
        p2[key] = {"A_minus_B": _r(est), "se": _r(se, 5),
                   "se_map_secret": _r(paired(rows, ms, "A", "B", (f,), c5, "map_secret")[1], 5),
                   "bar": P2_BAR, "verdict": verdict(est, se, P2_BAR)}
        est, se, n, s = paired(rows, ms, "A", "B", (f,), lambda r: r["wrong"])
        if f == 0.0:
            v = "MISSING" if est is None else ("PASS" if est >= 0 else "FAIL")
            p3[key] = {"wrongA_minus_wrongB": _r(est), "se": 0.0, "rule": "wrong_B <= wrong_A",
                       "verdict": v}
        else:
            p3[key] = {"wrongA_minus_wrongB": _r(est), "se": _r(se, 5), "bar": 0.0,
                       "rule": "wrong_B < wrong_A (2 SE)", "verdict": verdict(est, se, 0.0)}

    # P4: every fresh map x flip, point estimate at the primary cost.
    p4_rows, p4_bad = {}, []
    for c in FRESH_CONSTRUCTIONS:
        for m in cell_maps(maps, c, fresh_seeds):
            for f in FLIPS:
                est, _, n, _ = paired(rows, [m], "A", "B", (f,),
                                      lambda r: cost(r, C_WRONG_PRIMARY))
                k = "{}|{}".format(m, f)
                p4_rows[k] = _r(est)
                if est is None or est <= 0:
                    p4_bad.append(k)
    p4 = {"per_map": p4_rows, "n": len(p4_rows), "not_cheaper": p4_bad,
          "verdict": "MISSING" if not p4_rows else ("PASS" if not p4_bad else "FAIL")}

    # S1: D - B > 0 at the primary cost, per construction, pooled over the noisy flips.
    s1 = {}
    for c in FRESH_CONSTRUCTIONS + FIXED_CONSTRUCTIONS:
        ms = cell_maps(maps, c, fresh_seeds)
        est, se, n, s = paired(rows, ms, "D", "B", NOISY, lambda r: cost(r, C_WRONG_PRIMARY))
        s1[c] = {"D_minus_B": _r(est), "se": _r(se, 5), "bar": 0.0,
                 "verdict": verdict(est, se, 0.0),
                 "D_cheaper_beyond_2se": est is not None and est + 2 * se < 0}

    def all_v(d, v="PASS"):
        return all(x["verdict"] == v for x in d.values())

    def any_v(d, v):
        return [k for k, x in d.items() if x["verdict"] == v]

    res["P1"] = dict(cells=p1, PASS=all_v(p1), FAIL=any_v(p1, "FAIL"),
                     INCONCLUSIVE=any_v(p1, "INCONCLUSIVE"))
    res["P2"] = dict(cells=p2, PASS=all_v(p2), FAIL=any_v(p2, "FAIL"),
                     INCONCLUSIVE=any_v(p2, "INCONCLUSIVE"))
    res["P3"] = dict(cells=p3, PASS=all_v(p3), FAIL=any_v(p3, "FAIL"),
                     INCONCLUSIVE=any_v(p3, "INCONCLUSIVE"))
    res["P4"] = p4
    res["S1"] = dict(constructions=s1, PASS=all_v(s1),
                     D_noted=[c for c, x in s1.items() if x["D_cheaper_beyond_2se"]])

    # Decision rule and kill criterion.
    missing = [t for t, d in (("P1", p1), ("P2", p2), ("P3", p3)) if any_v(d, "MISSING")]
    if missing or p4["verdict"] == "MISSING":
        decision = "INCOMPLETE: missing cells in {}".format(missing or ["P4"])
    elif res["P1"]["FAIL"] or res["P3"]["FAIL"]:
        decision = ("KILL: class targeting does not pay on P0a even in total cost; FLYW-6 "
                    "closes with no remediation label (P1 or P3 failed in {})".format(
                        res["P1"]["FAIL"] + res["P3"]["FAIL"]))
    elif res["P1"]["PASS"] and res["P3"]["PASS"] and p4["verdict"] == "PASS":
        if res["P2"]["PASS"]:
            decision = ("ADOPT: FLYW-6 adds a remediation label; B is the default for the "
                        "noisy regime; holds whenever a wrong remediation costs more than "
                        "{:g} probes (P2 passes)".format(P2_C_WRONG))
        else:
            decision = ("ADOPT: FLYW-6 adds a remediation label; B is the default for the "
                        "noisy regime; P2 not passed in {}: record their c* (domains with "
                        "remediations cheaper than about c* probes may not benefit)".format(
                            res["P2"]["FAIL"] + res["P2"]["INCONCLUSIVE"]))
    else:
        decision = ("NOT DECIDED: label not adopted on this run (inconclusive: P1 {}, P3 {}; "
                    "P4 maps not cheaper: {}); a follow-up needs its own prereg".format(
                        res["P1"]["INCONCLUSIVE"], res["P3"]["INCONCLUSIVE"], p4["not_cheaper"]))
    res["decision"] = decision
    res["S1_choice"] = ("B for the noisy regime" if res["S1"]["PASS"] else
                        "B, with D noted in {}".format(res["S1"]["D_noted"])
                        if res["S1"]["D_noted"] else "B (S1 not passed everywhere; see SEs)")

    # ---------------- reported, not scored ----------------
    rep_cells = {}
    for c in FRESH_CONSTRUCTIONS + FIXED_CONSTRUCTIONS:
        ms = cell_maps(maps, c, fresh_seeds)
        for f in FLIPS:
            arms = {a: arm_means(rows, ms, a, f) for a in ARMS}
            if not all(arms.values()):
                continue
            rep_cells["{}|{}".format(c, f)] = dict(
                arms, cstar_B_vs_A=c_star(arms["B"], arms["A"]),
                cstar_B_vs_D=c_star(arms["B"], arms["D"]),
                scored="{}|{}".format(c, f) in res["scored_cells"])
    res["reported_cells"] = rep_cells

    per_map = {}
    align_x, gap_y = defaultdict(list), defaultdict(list)
    for c in FRESH_CONSTRUCTIONS:
        for m in cell_maps(maps, c, fresh_seeds):
            for f in FLIPS:
                arms = {a: arm_means(rows, [m], a, f) for a in ARMS}
                if not all(arms.values()):
                    continue
                gap = arms["A"]["cost_c50"] - arms["B"]["cost_c50"]
                per_map["{}|{}".format(m, f)] = {
                    "alignment": maps[m]["measures"]["alignment"],
                    "A_minus_B_c50": round(gap, 4),
                    "cstar_B_vs_A": c_star(arms["B"], arms["A"]),
                    "cstar_B_vs_D": c_star(arms["B"], arms["D"])}
                align_x[f].append(maps[m]["measures"]["alignment"])
                gap_y[f].append(gap)
    res["reported_per_fresh_map"] = per_map
    res["reported_spearman_alignment_vs_gap"] = {
        str(f): round(spearman(align_x[f], gap_y[f]), 4) for f in align_x if len(align_x[f]) > 2}

    if flyw1_cells is not None:
        # The two unscored flip-0 cells reproduce FLYW-1's cells (reproducibility check).
        chk, ok = {}, True
        for c in FIXED_CONSTRUCTIONS:
            for m in cell_maps(maps, c, fresh_seeds):
                for a in ARMS:
                    mine = arm_means(rows, [m], a, 0.0)
                    ref = flyw1_cells.get("{}|{}|0.0".format(m, a))
                    if mine is None or ref is None:
                        continue
                    same = (abs(mine["probes"] - ref["probes"]) < 1e-3
                            and abs(mine["wrong_per_game"] - ref["wrong_per_game"]) < 1e-3
                            and abs(mine["expert_share"] - ref["expert_share"]) < 1e-3)
                    ok &= same
                    chk["{}|{}".format(m, a)] = same
        res["reported_flip0_matches_flyw1"] = {"all": ok and bool(chk), "cells": chk}
    return res


def _r(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None or x != x else round(x, d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the run's output directory")
    ap.add_argument("--maps", required=True, help="the map set the run used")
    ap.add_argument("--flyw1-analysis", help="FLYW-1's analysis JSON, for the flip-0 check")
    ap.add_argument("--fresh-seeds", type=int, nargs="+", default=list(FRESH_SEEDS),
                    help="generator seeds of the fresh maps (a basis check on FLYW-1 "
                         "Part 1 uses 1 2)")
    ap.add_argument("--max-seed", type=int, help="read noise seeds up to this only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    maps = load_maps(a.maps)
    rows = load_rows(a.run, maps, a.max_seed)
    ref = None
    if a.flyw1_analysis:
        with open(a.flyw1_analysis) as f:
            ref = json.load(f)["part1"]["cells"]
    res = score(rows, maps, a.fresh_seeds, ref)
    res["inputs"] = {"run": os.path.basename(a.run.rstrip("/")),
                     "maps": os.path.basename(a.maps.rstrip("/")), "max_seed": a.max_seed}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    for t in ("P1", "P2", "P3"):
        print(t, "PASS" if res[t]["PASS"] else "not passed",
              {k: v["verdict"] for k, v in res[t]["cells"].items()})
    print("P4", res["P4"]["verdict"], "not cheaper:", res["P4"]["not_cheaper"])
    print("S1", {k: v["verdict"] for k, v in res["S1"]["constructions"].items()})
    print("decision:", res["decision"])


if __name__ == "__main__":
    main()
