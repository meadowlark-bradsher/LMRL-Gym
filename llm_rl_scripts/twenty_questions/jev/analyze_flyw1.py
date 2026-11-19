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
* ET2's missed call is T-loss acting at the trigger point on a new-class novel (user
  decision 2026-11-18); wrong actions before the trigger point are reported per arm.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
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


# Part 1 fields the scoring reads; the rest (live sets, flips, step lists of B/C/D)
# is dropped on load so 33k jobs fit well under the 8 GB per-job memory rule.
P1_KEEP_A = ("secret", "outcome", "n_probes", "steps")
P1_KEEP_ARM = ("secret", "outcome", "probes", "actions", "wrong", "fallbacks")


def load_jobs(d: str, slim: bool = False) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "jobs", "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            j = json.load(f)
        if slim:
            keep = P1_KEEP_A if j["job"]["arm"] == "A" else P1_KEEP_ARM
            j["games"] = [{k: g[k] for k in keep if k in g} for g in j["games"]]
        out.append(j)
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

def part1(d: str, maps: Dict[str, dict], gate: Optional[dict],
          cap_binds: bool = False, ext: Optional[str] = None,
          ext_cells: Optional[List[dict]] = None) -> dict:
    """``ext``/``ext_cells``: Addendum 1's DE1 extension (A and B on seeds 67..S in the
    cells whose blind S > 66). DE1(a)/(c) in those cells are scored on seeds 1..S, and
    the registered 66-seed result is kept beside them as the primary-protocol result."""
    jobs = load_jobs(d, slim=True)
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
    def a_rows(m, flip, src=None, s_max=None):
        cls = maps[m]["classes"]
        rows = []
        for (f, s), games in sorted((src if src is not None else a_by).items()):
            if f != flip or (s_max is not None and s > s_max):
                continue
            for g in games:
                r = score_a_record(g, cls)
                r["key"] = (s, g["secret"])
                r["strict"] = g["n_probes"]
                rows.append(r)
        return rows

    def arm_rows(m, arm, flip, src=None, s_max=None):
        rows = []
        for s, games in sorted((src if src is not None else arm_by).get((m, arm, flip),
                                                                       {}).items()):
            if s_max is not None and s > s_max:
                continue
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

    def pooled(arm, constr, flip, field="probes", src=None):
        src = per if src is None else src
        v = []
        for m in map_ids:
            if maps[m]["alignment_construction"] != constr:
                continue
            v.extend(r[field] for r in src.get((m, arm, flip), []))
        return statistics.mean(v) if v else None

    def se_diff(ms, x, y, flip, field="probes", src=None) -> float:
        """Within-secret SE of the pooled mean of (x - y) (seed rule, 2026-11-18): the
        secrets are the whole pool, so only seed-to-seed noise counts.
        SE = sqrt(sum_i var_i) / (N sqrt(S)), N = (map, secret) pairs."""
        if flip == 0.0:
            return 0.0
        src = per if src is None else src
        tot, n, s_seen = 0.0, 0, set()
        for m in ms:
            rx = {r["key"]: r[field] for r in src.get((m, x, flip), [])}
            ry = {r["key"]: r[field] for r in src.get((m, y, flip), [])}
            by_secret = defaultdict(list)
            for k in rx:
                if k in ry:
                    by_secret[k[1]].append(rx[k] - ry[k])
                    s_seen.add(k[0])
            for v in by_secret.values():
                if len(v) > 1:
                    tot += statistics.variance(v)
                n += 1
        S = len(s_seen)
        return math.sqrt(tot) / (n * math.sqrt(S)) if n and S else float("nan")

    def ms_of(c):
        return [m for m in map_ids if maps[m]["alignment_construction"] == c]

    def verdict(stat, bar, se, passes):
        """PASS/FAIL, or INCONCLUSIVE within 2 SE of the bar when the seed cap bound."""
        if cap_binds and se == se and abs(stat - bar) < 2 * se:
            return "INCONCLUSIVE"
        return "PASS" if passes else "FAIL"

    res: Dict[str, object] = {"maps_run": len(map_ids), "flips": flips,
                              "seed_cap_binds": cap_binds,
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
            se = se_diff(ms_of(c), "B", "A", flip)
            v = verdict(pb - pa, 0.1, se, pb <= pa + 0.1)
            d1a["{}|{}".format(c, flip)] = {"A": round(pa, 4), "B": round(pb, 4),
                                            "se": round(se, 5), "verdict": v,
                                            "pass": v == "PASS"}
            if flip > 0:
                wa, wb = pooled("A", c, flip, "wrong"), pooled("B", c, flip, "wrong")
                se = se_diff(ms_of(c), "B", "A", flip, "wrong")
                v = verdict(wb - wa, 0.02, se, wb <= wa + 0.02)
                d1c["{}|{}".format(c, flip)] = {"A": round(wa, 4), "B": round(wb, 4),
                                                "se": round(se, 5), "verdict": v,
                                                "pass": v == "PASS"}
    # Bars rescaled 2026-11-18 (user) to the P0a-tree basis: align >= 0.5, mix05 >= 0.3.
    for c, bar in (("align", 0.5), ("mix05", 0.3)):
        for flip in (0.0, 0.05):
            pa, pb = pooled("A", c, flip), pooled("B", c, flip)
            if pa is not None and pb is not None:
                se = se_diff(ms_of(c), "A", "B", flip)
                v = verdict(pa - pb, bar, se, pa - pb >= bar)
                d1b["{}|{}".format(c, flip)] = {"saving": round(pa - pb, 4), "bar": bar,
                                                "se": round(se, 5), "verdict": v,
                                                "pass": v == "PASS"}
    # Addendum 1, item 2: the DE1 extension. In each extended cell, A and B are
    # re-pooled over seeds 1..S (registered run + extension); the 66-seed entry is kept
    # as "primary_protocol". The cells come from the blind seed-rule file.
    ext_info = None
    if ext:
        ea_by, earm_by = dict(a_by), defaultdict(dict)
        for k, v in arm_by.items():
            if k[1] == "B":
                earm_by[k] = dict(v)
        for j in load_jobs(ext, slim=True):
            jb = j["job"]
            assert jb["arm"] in ("A", "B"), jb["job_id"]
            if jb["arm"] == "A":
                assert (jb["flip"], jb["seed"]) not in a_by, jb["job_id"]
                ea_by[(jb["flip"], jb["seed"])] = j["games"]
            else:
                key = (jb["map_id"], "B", jb["flip"])
                assert jb["seed"] not in arm_by.get(key, {}), jb["job_id"]
                earm_by[key][jb["seed"]] = j["games"]
        ext_info = {"dir": ext, "cells": []}
        for cell in ext_cells or []:
            c, flip, S = cell["construction"], cell["flip"], cell["seeds"]
            ms = ms_of(c)
            eper = {}
            complete = True
            for m in ms:
                eper[(m, "A", flip)] = a_rows(m, flip, ea_by, S)
                eper[(m, "B", flip)] = arm_rows(m, "B", flip, earm_by, S)
                seeds_b = set(earm_by.get((m, "B", flip), {}))
                seeds_a = {s for (f, s) in ea_by if f == flip}
                need = set(range(1, S + 1))
                complete = complete and need <= seeds_a and need <= seeds_b
            ck = "{}|{}".format(c, flip)
            ext_info["cells"].append(dict(cell, complete=complete))
            if not complete:
                for dd in (d1a, d1c):
                    if ck in dd:
                        dd[ck]["extension"] = "INCOMPLETE: scored on the 66-seed run only"
                continue
            for dd, field, bar in ((d1a, "probes", 0.1), (d1c, "wrong", 0.02)):
                if ck not in dd:
                    continue
                pa = pooled("A", c, flip, field, eper)
                pb = pooled("B", c, flip, field, eper)
                se = se_diff(ms, "B", "A", flip, field, eper)
                v = verdict(pb - pa, bar, se, pb <= pa + bar)
                prim = {k2: dd[ck][k2] for k2 in ("A", "B", "se", "verdict", "pass")}
                dd[ck] = {"A": round(pa, 4), "B": round(pb, 4), "se": round(se, 5),
                          "verdict": v, "pass": v == "PASS", "seeds": S,
                          "primary_protocol": dict(prim, seeds=66)}
    res["DE1_extension"] = ext_info
    d1_all = list(d1a.values()) + list(d1b.values()) + list(d1c.values())
    res["DE1"] = {"a": d1a, "b": d1b, "c": d1c,
                  "PASS": bool(d1a and d1b and d1c and all(x["pass"] for x in d1_all)),
                  # Addendum 1, item 1: an inconclusive DE1 counts as not passed.
                  "INCONCLUSIVE": bool(d1_all) and not any(x["verdict"] == "FAIL"
                                                           for x in d1_all)
                  and any(x["verdict"] == "INCONCLUSIVE" for x in d1_all)}

    def map_mean(m, arm, flip, field="probes"):
        rows = per.get((m, arm, flip), [])
        return statistics.mean(r[field] for r in rows) if rows else None

    # DE2.
    syn = [m for m in map_ids if maps[m]["family"] == "synthetic"]
    if 0.0 in flips and syn:
        x = [maps[m]["measures"]["alignment"] for m in syn]
        y = [map_mean(m, "A", 0.0) - map_mean(m, "B", 0.0) for m in syn]
        rho = spearman(x, y)
        res["DE2"] = {"spearman": round(rho, 4), "n_maps": len(syn), "bar": 0.5,
                      "PASS": rho >= 0.5}

    # DE3.
    rand = [m for m in map_ids if maps[m]["alignment_construction"] == "rand"]
    if rand and 0.0 in flips:
        cb = pooled("C", "rand", 0.0) - pooled("B", "rand", 0.0)
        out = {"a_c_minus_b": round(cb, 4), "a_pass": cb >= 1.0}
        if 0.1 in flips:
            wc = pooled("C", "rand", 0.1, "wrong") - pooled("B", "rand", 0.1, "wrong")
            se = se_diff(rand, "C", "B", 0.1, "wrong")
            v = verdict(wc, 0.02, se, wc >= 0.02)
            out.update({"b_c_minus_b_wrong": round(wc, 4), "b_se": round(se, 5),
                        "b_verdict": v, "b_pass": v == "PASS"})
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
            # DE4(a) is reported only (2026-11-18: its P0a-tree basis, 0.04, is inside
            # the noise); DE4 is scored on (b).
            res["DE4"] = {"b_minus_d_cross": round(bd(cross), 4),
                          "b_minus_d_align": round(bd(al), 4),
                          "a_diff_reported": round(a, 4), "b_rand_d_le_c": len(dc),
                          "b_rand_maps": len(rand), "b_pass": len(dc) == len(rand)}
            res["DE4"]["PASS"] = res["DE4"]["b_pass"]

    # DE5.
    if 0.05 in flips:
        allp = lambda arm: statistics.mean(r["probes"] for m in map_ids
                                           for r in per.get((m, arm, 0.05), []))
        b = allp("B")
        sc, sd = b - allp("C"), b - allp("D")
        se_c = se_diff(map_ids, "B", "C", 0.05)
        se_d = se_diff(map_ids, "B", "D", 0.05)
        best, se_b = (sc, se_c) if sc >= sd else (sd, se_d)
        v = verdict(best, 0.5, se_b, best < 0.5)
        res["DE5"] = {"c_saves_over_b": round(sc, 4), "d_saves_over_b": round(sd, 4),
                      "se_c": round(se_c, 5), "se_d": round(se_d, 5), "verdict": v,
                      "PASS": v == "PASS"}
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
    elif res["DE1"]["INCONCLUSIVE"]:
        # Addendum 1, item 1.
        rule = ("DE1 inconclusive (counts as not passed): B is not recommended; the SEs "
                "are reported and more seeds are needed")
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
                "probes": r["probes"], "pre_wrong": r["wrong"], "acted_at_trigger": False}
    if g["outcome"] == "right":
        return {"expert": False, "wrong": g["wrong"], "probes": g["probes"],
                "pre_wrong": g["wrong"], "acted_at_trigger": False}
    if trigger == "mass":
        return {"expert": True, "wrong": g["wrong"], "probes": g["probes"],
                "pre_wrong": g["wrong"], "acted_at_trigger": False}
    t = g["tloss"][trigger]
    return {"expert": t["outcome"] == "expert", "wrong": g["wrong"] + t["extra_wrong"],
            "probes": g["probes"] + t["extra_actions"], "pre_wrong": g["wrong"],
            "acted_at_trigger": t["extra_actions"] >= 1}


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
        w_sp = n_sp = w_new = n_miss = cost = w_all = pre = n_orig = 0.0
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
                    # ET2 (amended 2026-11-18): a missed call is the trigger *acting*
                    # on a new-class novel. Wrong actions before the trigger point are
                    # the stopping rule's and are the same under every trigger.
                    n_miss += w * o["acted_at_trigger"]
                    pre += w * o["pre_wrong"]
                    # The original measure, reported (user, 2026-11-18): any wrong action
                    # before the expert, or no expert at all. It shows that a novel
                    # incident costs wrong actions before escalation is ever consulted.
                    n_orig += w * (o["wrong"] >= 1 or not o["expert"])
        return {"spurious": round(n_sp / w_sp, 4) if w_sp else None,
                "missed": round(n_miss / w_new, 4) if w_new else None,
                "pre_trigger_wrong_per_new_game": round(pre / w_new, 4) if w_new else None,
                "missed_original_reported": round(n_orig / w_new, 4) if w_new else None,
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
    # Reported: alignment of each rotation's reduced known pool on P0a's own question
    # tree, read off A's flip-0 in-pool games of that rotation (control arm only).
    from .class_maps import p0a_measures
    align = {}
    for j in jobs:
        jb = j["job"]
        if jb["arm"] != "A" or jb["flip"] != 0.0:
            continue
        cls = maps[jb["map_id"]]["classes"]
        games = [g for g in j["games"] if g["kind"] == "in_pool"]
        seqs = {g["secret"]: tuple((st[0] + ":" + st[1], st[2]) for st in g["steps"]
                                   if st[0] != "g")
                for g in games}
        objs = [g["secret"] for g in games]
        align[jb["rotation"]] = p0a_measures(objs, seqs, cls)["alignment"]
    res["reduced_pool_alignment_p0a"] = align
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part1")
    ap.add_argument("--part2")
    ap.add_argument("--gate")
    ap.add_argument("--maps", default=MAPS)
    ap.add_argument("--seed-rule", help="seed_rule.json (sets whether the seed cap bound)")
    ap.add_argument("--part1-ext", help="Addendum 1 DE1 extension dir (needs --seed-rule)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    maps = load_maps(a.maps)
    gate = None
    if a.gate:
        with open(a.gate) as f:
            gate = json.load(f)
    out = {}
    cap = False
    if a.seed_rule:
        with open(a.seed_rule) as f:
            cap = bool(json.load(f)["cap_binds"])
    ext_cells = None
    if a.part1_ext:
        if not a.seed_rule:
            ap.error("--part1-ext needs --seed-rule")
        from .flyw1_run import extension_cells
        ext_cells = extension_cells(a.seed_rule)
    if a.part1:
        out["part1"] = part1(a.part1, maps, gate, cap, a.part1_ext, ext_cells)
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
