"""FLYW-1 analyzer: scores DE0-DE5 (Part 1) and ET1-ET3 (Part 2) exactly as
``FLYW1_DECISION_PREREG.md`` registers them. Where the prereg's wording is ambiguous,
this code is the reference (prereg, "Scoring and decision rule").

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw1 \
        --part1 .../flyw1_part1_<date> --part2 .../flyw1_part2_<date> \
        --gate .../flyw1_gate_<date>/gate.json --out .../flyw1_analysis_<date>.json

Readings fixed here (see ``FLYW1_BUILD.md``):

* A cell's games are pooled over noise seeds; a construction x flip pools its maps'
  games (each map has the same games, so this equals the mean of map means).
* "Unresolvable" (DE0(c)) uses P0a's kill rule (``flyw1_decision.kill_rule_twins``).
* ET2 and ET3 pool flips 0.05 and 0.10, as ET1 states; flip 0 is reported.
* Part 2 rates weight each game by its ``weight`` (in-pool games at flip 0 are played
  once with the weight of their repeats).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .bed import load_hypotheses
from .flyw1_a_calibration import spearman
from .flyw1_decision import kill_rule_twins, score_a_record

HERE = os.path.dirname(__file__)
MAPS = os.path.join(HERE, "results", "flyw1_class_maps_20261118")
CONSTRUCTIONS = ("align", "mix05", "mix15", "rand", "category")
C_EXPERT = 10.0


def load_jobs(d: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "jobs", "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            out.append(json.load(f))
    return out


def load_maps(maps_dir: str = MAPS) -> Dict[str, dict]:
    with open(os.path.join(maps_dir, "index.json")) as f:
        index = json.load(f)
    maps = {}
    for row in index["maps"]:
        with open(os.path.join(maps_dir, "maps", row["map_id"] + ".json")) as f:
            maps[row["map_id"]] = json.load(f)
    return maps


def mean(v: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(v), 4) if v else None


# --------------------------------------------------------------------------- #
# Part 1                                                                       #
# --------------------------------------------------------------------------- #

def part1(d: str, maps: Dict[str, dict], gate: Optional[dict]) -> dict:
    jobs = load_jobs(d)
    a_by = {}                                   # (flip, seed) -> games
    arm_by = defaultdict(dict)                  # (map, arm, flip) -> seed -> games
    for j in jobs:
        jb = j["job"]
        if jb["arm"] == "A":
            a_by[(jb["flip"], jb["seed"])] = j["games"]
        else:
            arm_by[(jb["map_id"], jb["arm"], jb["flip"])][jb["seed"]] = j["games"]
    flips = sorted({k[0] for k in a_by})
    map_ids = sorted({k[0] for k in arm_by if k[0] in maps})
    arms_present = sorted({k[1] for k in arm_by})

    # Per-game rows: (map, flip) -> arm -> list of dicts aligned on (seed, secret).
    def a_rows(m, flip):
        cls = maps[m]["classes"]
        rows = []
        for (f, s), games in sorted(a_by.items()):
            if f != flip:
                continue
            for g in games:
                r = score_a_record(g, cls)
                r["key"] = (s, g["secret"])
                r["strict"] = g["n_probes"]
                rows.append(r)
        return rows

    def arm_rows(m, arm, flip):
        rows = []
        for s, games in sorted(arm_by.get((m, arm, flip), {}).items()):
            for g in games:
                rows.append(dict(g, key=(s, g["secret"])))
        return rows

    cells = {}
    per = {}
    for m in map_ids:
        for flip in flips:
            ra = a_rows(m, flip)
            per[(m, "A", flip)] = ra
            for arm in arms_present:
                rr = arm_rows(m, arm, flip)
                if rr:
                    per[(m, arm, flip)] = rr
    for (m, arm, flip), rows in per.items():
        n = len(rows)
        acts = sum(r["actions"] for r in rows)
        right = sum(r["outcome"] == "right" for r in rows)
        cells["{}|{}|{}".format(m, arm, flip)] = {
            "map": m, "arm": arm, "flip": flip, "n": n,
            "probes": mean([r["probes"] for r in rows]),
            "accuracy": round(right / acts, 4) if acts else None,
            "wrong_per_game": round(sum(r["wrong"] for r in rows) / n, 4),
            "expert_share": round(sum(r["outcome"] != "right" for r in rows) / n, 4),
            "decided": round(right / n, 4),
            "cost_r0.2": mean([r["probes"] + (r["outcome"] != "right") * C_EXPERT
                               + r["wrong"] * C_EXPERT / 0.2 for r in rows]),
            "a_strict": mean([r["strict"] for r in rows]) if arm == "A" else None,
            "fallbacks": (round(sum(r.get("fallbacks", 0) for r in rows) / n, 4)
                          if arm != "A" else None)}

    def pooled(arm, constr, flip, field="probes"):
        v = []
        for m in map_ids:
            if maps[m]["alignment_construction"] != constr:
                continue
            v.extend(r[field] for r in per.get((m, arm, flip), []))
        return statistics.mean(v) if v else None

    res: Dict[str, object] = {"maps_run": len(map_ids), "flips": flips,
                              "arms": ["A"] + arms_present}

    # DE0(c) over every flip-0 job.
    hyps = load_hypotheses()
    twins = None
    viol = []
    if 0.0 in flips:
        from .flyw1_run import load_inputs
        matrix, _, _ = load_inputs()
        twins = kill_rule_twins(matrix, hyps)
        for m in map_ids:
            cls = maps[m]["classes"]
            unres = {o for o in cls if any(cls[t] != cls[o] for t in twins[o])}
            for arm in ("B", "C", "D"):
                for r in per.get((m, arm, 0.0), []):
                    if r["wrong"] and r["secret"] not in unres:
                        viol.append([m, arm, r["secret"]])
    de0 = {"gate_pass": bool(gate and gate.get("PASS")), "c_violations": viol[:20],
           "c_full": not viol}
    de0["PASS"] = de0["gate_pass"] and de0["c_full"]
    res["DE0"] = de0

    # DE1.
    d1a, d1b, d1c = {}, {}, {}
    for c in CONSTRUCTIONS:
        for flip in flips:
            pa, pb = pooled("A", c, flip), pooled("B", c, flip)
            if pa is None or pb is None:
                continue
            d1a["{}|{}".format(c, flip)] = {"A": round(pa, 4), "B": round(pb, 4),
                                            "pass": pb <= pa + 0.1}
            if flip > 0:
                wa, wb = pooled("A", c, flip, "wrong"), pooled("B", c, flip, "wrong")
                d1c["{}|{}".format(c, flip)] = {"A": round(wa, 4), "B": round(wb, 4),
                                                "pass": wb <= wa + 0.02}
    for c in ("align", "mix05"):
        for flip in (0.0, 0.05):
            pa, pb = pooled("A", c, flip), pooled("B", c, flip)
            if pa is not None and pb is not None:
                d1b["{}|{}".format(c, flip)] = {"saving": round(pa - pb, 4),
                                                "pass": pa - pb >= 1.0}
    res["DE1"] = {"a": d1a, "b": d1b, "c": d1c,
                  "PASS": bool(d1a and d1b and d1c
                               and all(x["pass"] for x in list(d1a.values())
                                       + list(d1b.values()) + list(d1c.values())))}

    def map_mean(m, arm, flip, field="probes"):
        rows = per.get((m, arm, flip), [])
        return statistics.mean(r[field] for r in rows) if rows else None

    # DE2.
    syn = [m for m in map_ids if maps[m]["family"] == "synthetic"]
    if 0.0 in flips and syn:
        x = [maps[m]["measures"]["alignment"] for m in syn]
        y = [map_mean(m, "A", 0.0) - map_mean(m, "B", 0.0) for m in syn]
        rho = spearman(x, y)
        res["DE2"] = {"spearman": round(rho, 4), "n_maps": len(syn), "PASS": rho >= 0.8}

    # DE3.
    rand = [m for m in map_ids if maps[m]["alignment_construction"] == "rand"]
    if rand and 0.0 in flips:
        cb = pooled("C", "rand", 0.0) - pooled("B", "rand", 0.0)
        out = {"a_c_minus_b": round(cb, 4), "a_pass": cb >= 1.0}
        if 0.1 in flips:
            wc = pooled("C", "rand", 0.1, "wrong") - pooled("B", "rand", 0.1, "wrong")
            out.update({"b_c_minus_b_wrong": round(wc, 4), "b_pass": wc >= 0.02})
        out["PASS_a"] = out["a_pass"]
        out["PASS_b"] = out.get("b_pass")
        res["DE3"] = out

    # DE4.
    if 0.0 in flips:
        cross = [m for m in map_ids if maps[m]["alignment_construction"] in ("mix15", "rand")]
        al = [m for m in map_ids if maps[m]["alignment_construction"] == "align"]
        bd = lambda ms: statistics.mean(map_mean(m, "B", 0.0) - map_mean(m, "D", 0.0)
                                        for m in ms)
        if cross and al:
            a = bd(cross) - bd(al)
            dc = [m for m in rand if map_mean(m, "D", 0.0) <= map_mean(m, "C", 0.0)]
            res["DE4"] = {"b_minus_d_cross": round(bd(cross), 4),
                          "b_minus_d_align": round(bd(al), 4), "a_diff": round(a, 4),
                          "a_pass": a >= 0.2, "b_rand_d_le_c": len(dc),
                          "b_rand_maps": len(rand), "b_pass": len(dc) == len(rand)}
            res["DE4"]["PASS"] = res["DE4"]["a_pass"] and res["DE4"]["b_pass"]

    # DE5.
    if 0.05 in flips:
        allp = lambda arm: statistics.mean(r["probes"] for m in map_ids
                                           for r in per.get((m, arm, 0.05), []))
        b = allp("B")
        sc, sd = b - allp("C"), b - allp("D")
        res["DE5"] = {"c_saves_over_b": round(sc, 4), "d_saves_over_b": round(sd, 4),
                      "PASS": max(sc, sd) < 0.5}
        # Reported ratio.
        gb = gbest = 0.0
        for m in map_ids:
            pa = map_mean(m, "A", 0.05)
            vals = [map_mean(m, x, 0.05) for x in ("B", "C", "D")]
            gb += pa - vals[0]
            gbest += pa - min(vals)
        res["DE5"]["ratio_reported"] = round(gb / gbest, 4) if gbest else None

    # Decision rule.
    d1 = res["DE1"]["PASS"]
    d5 = res.get("DE5", {}).get("PASS")
    if d1 and d5:
        rule = "B: only the stopping rule changes; FLYW-6 specifies a remediation label"
    elif d1 and d5 is False:
        sc, sd = res["DE5"]["c_saves_over_b"], res["DE5"]["d_saves_over_b"]
        rule = "question score matters: recommend {} if within DE1(c)'s bound".format(
            "C" if sc >= sd else "D")
    else:
        rule = "class targeting does not pay on P0a; FLYW-6 records why no label"
    res["decision"] = rule
    res["cells"] = cells
    return res


# --------------------------------------------------------------------------- #
# Part 2                                                                       #
# --------------------------------------------------------------------------- #

def p2_outcome(g: dict, trigger: str, cls: Dict[str, int]) -> dict:
    """One game's outcome under a trigger config ("mass" or a tloss key)."""
    if g["arm"] == "A":
        r = score_a_record(g, cls)
        return {"expert": r["outcome"] != "right", "wrong": r["wrong"],
                "probes": r["probes"]}
    if g["outcome"] == "right":
        return {"expert": False, "wrong": g["wrong"], "probes": g["probes"]}
    if trigger == "mass":
        return {"expert": True, "wrong": g["wrong"], "probes": g["probes"]}
    t = g["tloss"][trigger]
    return {"expert": t["outcome"] == "expert", "wrong": g["wrong"] + t["extra_wrong"],
            "probes": g["probes"] + t["extra_actions"]}


def part2(d: str, maps: Dict[str, dict]) -> dict:
    jobs = load_jobs(d)
    rows = defaultdict(list)     # (map, arm, flip, trigger) -> [(g, outcome)]
    triggers_b = ["mass"] + ["mean_r{}_pu{}".format(r, s) for r in (0.5, 0.2, 0.05)
                             for s in (1.0, 0.5, 2.0)] + \
                 ["max_r{}_pu1.0".format(r) for r in (0.5, 0.2, 0.05)]
    for j in jobs:
        jb = j["job"]
        cls = maps[jb["map_id"]]["classes"]
        trigs = ["mass"] if jb["arm"] == "A" else triggers_b
        for g in j["games"]:
            for t in trigs:
                rows[(jb["map_id"], jb["arm"], jb["flip"], t, jb["rotation"])].append(
                    (g, p2_outcome(g, t, cls)))

    def rates(ms, arm, flips, trig, ratio):
        w_sp = n_sp = w_new = n_miss = cost = w_all = 0.0
        for (m, a, f, t, rot), lst in rows.items():
            if m not in ms or a != arm or f not in flips or t != trig:
                continue
            for g, o in lst:
                w = g["weight"]
                w_all += w
                cost += w * (o["probes"] + o["expert"] * C_EXPERT
                             + o["wrong"] * C_EXPERT / ratio)
                if g["kind"] in ("in_pool", "same"):
                    w_sp += w
                    n_sp += w * o["expert"]
                else:
                    w_new += w
                    n_miss += w * (o["wrong"] >= 1 or not o["expert"])
        return {"spurious": round(n_sp / w_sp, 4) if w_sp else None,
                "missed": round(n_miss / w_new, 4) if w_new else None,
                "cost": round(cost / w_all, 4) if w_all else None}

    all_maps = sorted({k[0] for k in rows})
    noisy = (0.05, 0.1)
    align_maps = [m for m in all_maps if maps[m]["alignment_construction"] in ("align",
                                                                               "category")]
    rand_maps = [m for m in all_maps if maps[m]["alignment_construction"] == "rand"]
    res: Dict[str, object] = {"maps": all_maps}
    et1, et2, et3, adopt = {}, {}, {}, {}
    for arm in ("B", "D"):
        tm = rates(align_maps, arm, noisy, "mass", 0.2)
        tl = rates(align_maps, arm, noisy, "mean_r0.2_pu1.0", 0.2)
        et1[arm] = {"mass": tm["spurious"], "loss": tl["spurious"],
                    "pass": (tm["spurious"] is not None and tl["spurious"] is not None
                             and tl["spurious"] <= 0.7 * tm["spurious"])}
        per_map = {}
        for m in all_maps:
            r = rates([m], arm, noisy, "mean_r0.2_pu1.0", 0.2)
            per_map[m] = {"missed": r["missed"], "pass": r["missed"] is not None
                          and r["missed"] <= 0.10}
        et2[arm] = {"per_map": per_map, "pass": all(v["pass"] for v in per_map.values())}
        rm = rates(rand_maps, arm, noisy, "mass", 0.2)
        rl = rates(rand_maps, arm, noisy, "mean_r0.2_pu1.0", 0.2)
        et3[arm] = {"mass": rm["spurious"], "loss": rl["spurious"],
                    "pass": (rm["spurious"] is not None and rl["spurious"] is not None
                             and abs(rl["spurious"] - rm["spurious"]) <= 0.02)}
        cm = rates(all_maps, arm, noisy, "mass", 0.2)["cost"]
        cl = rates(all_maps, arm, noisy, "mean_r0.2_pu1.0", 0.2)["cost"]
        adopt[arm] = {"cost_mass": cm, "cost_loss": cl}
    res["ET1"] = dict(et1, PASS=all(v["pass"] for v in et1.values()))
    res["ET2"] = dict(et2, PASS=all(v["pass"] for v in et2.values()))
    res["ET3"] = dict(et3, PASS=all(v["pass"] for v in et3.values()))
    for arm in ("B", "D"):
        a = adopt[arm]
        a["adopt_tloss"] = bool(et1[arm]["pass"] and et2[arm]["pass"]
                                and a["cost_loss"] is not None and a["cost_mass"] is not None
                                and a["cost_loss"] < a["cost_mass"])
        a["adopt_with_alignment_check"] = bool(et1[arm]["pass"] and not et2[arm]["pass"])
    res["decision"] = adopt
    # Reported: every trigger config by construction and flip, and A's reference.
    rep = {}
    for c, ms in (("align+cat17", align_maps), ("rand", rand_maps)):
        for flip in (0.0, 0.05, 0.1):
            for arm in ("A", "B", "D"):
                for t in (["mass"] if arm == "A" else triggers_b):
                    ratio = float(t.split("_r")[1].split("_")[0]) if t != "mass" else 0.2
                    r = rates(ms, arm, (flip,), t, ratio)
                    if r["spurious"] is not None:
                        rep["{}|{}|{}|{}".format(c, flip, arm, t)] = r
    res["reported"] = rep
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part1")
    ap.add_argument("--part2")
    ap.add_argument("--gate")
    ap.add_argument("--maps", default=MAPS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    maps = load_maps(a.maps)
    gate = None
    if a.gate:
        with open(a.gate) as f:
            gate = json.load(f)
    out = {}
    if a.part1:
        out["part1"] = part1(a.part1, maps, gate)
    if a.part2:
        out["part2"] = part2(a.part2, maps)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for part, r in out.items():
        print("==", part)
        for k, v in r.items():
            if k in ("cells", "reported"):
                continue
            print(" ", k, json.dumps(v)[:400])


if __name__ == "__main__":
    main()
