"""Score the ST-3 end-game follow-up (``STRESS_ST3_ENDGAME_PREREG.md``, *Scoring*).

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_st3_endgame \
        --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_endgame_<stamp> \
        --base-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_20261113

Every end-game job ``<st3 id>_end<rule>`` pairs with the ST-3 job ``<st3 id>`` in
the base run: the same stream, objects, oracle seed and backend, under the `true`
prior. Only the budget-end rule differs, and ``top`` (ST-3's rule) is the base
cell itself. Writes ``analysis.json`` and one figure into the run directory.
Populations and rates are ``analyze_st3``'s; the code is the reference when the
prereg's wording is ambiguous.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, List

from .analyze_st3 import EXPERT, WON, WRONG, Pool_, rates

BUDGET = 20
RULES = ["top", "half", "conf", "always"]          # top = the ST-3 base cell
ARMS = [("primary", "soft"), ("primary", "modeswitch_guard"), ("norepeat", "modeswitch_guard")]
SCORED = [("primary", "soft"), ("norepeat", "modeswitch_guard")]
SIGMAS = [0.0, 0.25, 0.5, 0.75]
N_JOBS = 324
AUTONOMOUS_FPR = 0.20     # autonomous: FPR <= this
SAFE_WRONG = 0.05         # safe: wrong rate <= this (SK2's floor)
EG3_GAIN = 0.05           # soft's won gain under `always`: pays >= +this, costs <= -this
EG4_MAX = 0.02            # correct unconfirmed answers given up under `always`
EG5_TOL = 0.01            # per-step slack in the rule ordering
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _load(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _base_id(job_id: str, rule: str) -> str:
    suffix = "_end" + rule
    assert job_id.endswith(suffix), job_id
    return job_id[:-len(suffix)]


def job_checks(rec: dict, base: dict) -> List[str]:
    """EG0 (a)-(c) for one job against its ST-3 pair; returns the failures."""
    j, g, b = rec["job"], rec["games"], base["games"]
    n = len(b["obj"])
    bad = []
    diff = next((i for i in range(n) if any(g[k][i] != b[k][i] for k in b)), None)
    if diff is not None:
        cols = [k for k in b if g[k][diff] != b[k][diff]]
        if not (cols == ["outcome"] and b["outcome"][diff] == WRONG
                and b["n_questions"][diff] == BUDGET and g["outcome"][diff] == EXPERT):
            bad.append("first difference at game {} is not an escalated budget-end "
                       "wrong (columns {})".format(diff, cols))
    for i in range(n):
        if g["end_budget"][i]:
            if g["end_u_leads"][i] and g["outcome"][i] != EXPERT:
                bad.append("game {}: UNKNOWN led at the budget end but no escalation".format(i))
                break
        elif g["outcome"][i] == WRONG:
            bad.append("game {}: wrong without reaching the budget end".format(i))
            break
    if j["end_rule"] == "always":
        if WRONG in g["outcome"]:
            bad.append("`always` left a wrong game")
        if any(g["outcome"][i] != EXPERT for i in range(n) if not g["in_pool"][i]):
            bad.append("`always` left a novel game without the expert")
    return bad


def end_stats(cols: dict, idx: List[int]) -> dict:
    """Budget-end bookkeeping a rule job carries (ST-3's base jobs do not)."""
    n = len(idx)
    end = [i for i in idx if cols["end_budget"][i]]
    gave_up = sum(1 for i in end if not cols["end_u_leads"][i]
                  and cols["end_top_is_secret"][i] == 1 and cols["outcome"][i] == EXPERT)
    answered_right = sum(1 for i in end if cols["outcome"][i] == WRONG
                         and cols["end_top_is_secret"][i] == 1)
    out = {"budget_end_share": len(end) / n if n else float("nan"),
           "gave_up_correct": gave_up / n if n else float("nan"),
           "answered_correct_unconfirmed": answered_right / n if n else float("nan"),
           "end_p_top": {}}
    for label, pick in (("novel", lambda i: not cols["in_pool"][i]),
                        ("in_pool", lambda i: cols["in_pool"][i])):
        v = sorted(cols["end_p_top"][i] for i in end if pick(i))
        if v:
            out["end_p_top"][label] = {
                "n": len(v),
                "quantiles": {str(q): v[min(len(v) - 1, int(q * len(v)))] for q in QUANTILES},
                "share_ge_0.5": sum(1 for x in v if x >= 0.5) / len(v),
                "share_ge_0.9": sum(1 for x in v if x >= 0.9) / len(v)}
    return out


def metrics(p: Pool_, with_end: bool, finals: List[int]) -> dict:
    allg = list(range(len(p.g)))
    out = {"all": rates(p.cols, allg),
           "first_half": rates(p.cols, p.idx(lambda g, i: g < 2500)),
           "second_half": rates(p.cols, p.idx(lambda g, i: g >= 2500))}
    out["all"]["won"] = out["all"]["autonomous"]
    last = p.idx(lambda g, i: g >= 4750)
    out["mean_prior_last_window"] = (sum(p.cols["prior"][i] for i in last) / len(last)
                                     if last else float("nan"))
    out["admitted"] = sum(1 for i in allg if not p.cols["in_pool"][i]
                          and p.cols["outcome"][i] == EXPERT)
    out["mean_final_known"] = sum(finals) / len(finals) if finals else float("nan")
    if with_end:
        out["end"] = end_stats(p.cols, allg)
    return out


def _final_known(cols: dict) -> int:
    i = len(cols["obj"]) - 1
    return cols["k_before"][i] + int(not cols["in_pool"][i] and cols["outcome"][i] == EXPERT)


def _job_rates(cols: dict) -> dict:
    r = rates(cols, list(range(len(cols["obj"]))))
    return {"wrong": r["wrong_rate"], "fpr": r["fpr"], "won": r["autonomous"]}


def analyze(run_dir: str, base_dir: str, make_figures: bool = True) -> dict:
    pools: Dict[tuple, Pool_] = defaultdict(Pool_)
    finals: Dict[tuple, List[int]] = defaultdict(list)
    deltas: Dict[tuple, List[dict]] = defaultdict(list)
    base_seen = set()
    checks: Dict[str, List[str]] = {}
    identical_d: Dict[str, bool] = {}
    n_jobs, missing = 0, []
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        rec = _load(path)
        j = rec["job"]
        if "end_rule" not in j:          # a smoke run keeps its base jobs here too
            continue
        n_jobs += 1
        base_id = _base_id(j["job_id"], j["end_rule"])
        bpath = os.path.join(base_dir, "jobs", base_id + ".json.gz")
        if not os.path.exists(bpath):
            missing.append(base_id)
            continue
        base = _load(bpath)
        assert base["job"]["oracle_seed"] == j["oracle_seed"], base_id
        assert base["job"]["prior"] == "true" and j["prior"] == "true", base_id
        assert base["games"]["obj"] == rec["games"]["obj"], base_id
        key = (j["sigma"], j["flip"], j["arm"], j["backend"])
        pools[key + (j["end_rule"],)].add(rec)
        finals[key + (j["end_rule"],)].append(_final_known(rec["games"]))
        if base_id not in base_seen:
            base_seen.add(base_id)
            pools[key + ("top",)].add(base)
            finals[key + ("top",)].append(_final_known(base["games"]))
        jr, br = _job_rates(rec["games"]), _job_rates(base["games"])
        deltas[key + (j["end_rule"],)].append({m: jr[m] - br[m] for m in jr})
        checks[j["job_id"]] = job_checks(rec, base)
        if j["sigma"] == 0.75 and (j["arm"], j["backend"]) == ("primary", "modeswitch_guard"):
            b = base["games"]
            identical_d[j["job_id"]] = {k: rec["games"][k] for k in b} == b

    cells, C = [], {}
    for key in sorted(pools):
        m = metrics(pools[key], key[4] != "top", finals[key])
        row = {"sigma": key[0], "flip": key[1], "arm": key[2], "backend": key[3],
               "rule": key[4], **m}
        if key[4] != "top":
            d = deltas[key]
            row["job_delta_range"] = {k: [min(x[k] for x in d), max(x[k] for x in d)]
                                      for k in ("wrong", "fpr", "won")}
            row["n_jobs"] = len(d)
        cells.append(row)
        C[key] = row

    complete = n_jobs == N_JOBS and not missing
    v = score(C, checks, identical_d)
    out = {"schema": "twenty_questions_stress_st3_endgame_analysis_v1",
           "run_dir": run_dir, "base_dir": base_dir, "n_jobs": n_jobs,
           "missing_base_jobs": missing, "complete_grid": complete,
           "job_check_failures": {k: x for k, x in checks.items() if x},
           "cells": cells,
           "verdicts": v if complete else {},
           "verdicts_partial_grid": {} if complete else v}
    if make_figures:
        out["figures"] = [os.path.basename(figure(C, run_dir, base_dir))]
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def score(C: Dict[tuple, dict], checks: Dict[str, List[str]],
          identical_d: Dict[str, bool]) -> dict:
    def a(sg, arm, bk, rule, flip=0.0):
        m = C.get((sg, flip, arm, bk, rule))
        return m["all"] if m else None

    v: Dict[str, dict] = {}
    # EG0: mechanical gate.
    n_d = 3 * 9                    # published P0a, sigma 0.75: 3 rules x 9 seed combos
    parts = {"pairing_and_budget_end": bool(checks) and not any(checks.values()),
             "published_p0a_s0.75_identical": len(identical_d) == n_d
             and all(identical_d.values())}
    v["EG0"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts,
                "jobs_checked": len(checks),
                "jobs_failing": sorted(k for k, x in checks.items() if x)}

    # EG1: under `always`, soft and norepeat P0a stay autonomous at sigma 0.5 / 0.75.
    parts, fpr = {}, {}
    for arm, bk in SCORED:
        for sg in (0.5, 0.75):
            m = a(sg, arm, bk, "always")
            k = "{}_{}_s{:g}".format(arm, bk, sg)
            fpr[k] = m["fpr"] if m else float("nan")
            parts[k] = bool(m) and m["fpr"] <= AUTONOMOUS_FPR
    v["EG1"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts,
                "fpr": fpr}

    # EG2: norepeat P0a at sigma 0.75: `half` unsafe, `conf` safe and autonomous.
    h = a(0.75, "norepeat", "modeswitch_guard", "half")
    c = a(0.75, "norepeat", "modeswitch_guard", "conf")
    parts = {"half_unsafe": bool(h) and h["wrong_rate"] > SAFE_WRONG,
             "conf_safe": bool(c) and c["wrong_rate"] <= SAFE_WRONG,
             "conf_autonomous": bool(c) and c["fpr"] <= AUTONOMOUS_FPR}
    v["EG2"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts,
                "half": {k: h[k] for k in ("wrong_rate", "fpr")} if h else None,
                "conf": {k: c[k] for k in ("wrong_rate", "fpr")} if c else None}

    # EG3: soft's won gain from admission under `always`.
    t, w = a(0.75, "primary", "soft", "top"), a(0.75, "primary", "soft", "always")
    if t and w:
        gain = w["won"] - t["won"]
        reading = ("pays" if gain >= EG3_GAIN else
                   "crowding costs" if gain <= -EG3_GAIN else "neutral")
    else:
        gain, reading = float("nan"), "incomplete"
    v["EG3"] = {"verdict": "PASS" if reading == "pays" else "FAIL", "reading": reading,
                "won_gain": gain}

    # EG4: correct unconfirmed answers given up under `always`.
    parts, given = {}, {}
    for arm, bk in SCORED:
        for sg in (0.5, 0.75):
            m = C.get((sg, 0.0, arm, bk, "always"))
            k = "{}_{}_s{:g}".format(arm, bk, sg)
            given[k] = m["end"]["gave_up_correct"] if m else float("nan")
            parts[k] = bool(m) and m["end"]["gave_up_correct"] <= EG4_MAX
    v["EG4"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts,
                "gave_up_correct": given}

    # EG5: wrong falls and FPR rises along top -> half -> conf -> always.
    parts = {}
    for arm, bk in SCORED:
        for sg in (0.5, 0.75):
            seq = [a(sg, arm, bk, r) for r in RULES]
            k = "{}_{}_s{:g}".format(arm, bk, sg)
            if not all(seq):
                parts[k] = False
                continue
            parts[k] = all(seq[i]["wrong_rate"] >= seq[i + 1]["wrong_rate"] - EG5_TOL
                           and seq[i]["fpr"] <= seq[i + 1]["fpr"] + EG5_TOL
                           for i in range(len(seq) - 1))
    v["EG5"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts}
    return v


def figure(C: Dict[tuple, dict], run_dir: str, base_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ARMS), figsize=(4 * len(ARMS), 3.2), sharey=True)
    xs = list(range(len(RULES)))
    for ax, (arm, bk) in zip(axes, ARMS):
        for sg, ls in ((0.75, "-"), (0.5, "--")):
            for metric, col, lab in (("wrong_rate", "C3", "wrong"), ("fpr", "C2", "FPR"),
                                     ("won", "C0", "won")):
                pts = [(x, C[(sg, 0.0, arm, bk, r)]["all"][metric])
                       for x, r in zip(xs, RULES) if (sg, 0.0, arm, bk, r) in C]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col,
                            marker="o", ms=3, label="{} (σ={:g})".format(lab, sg))
        ax.set_title("{} ({})".format(bk, arm), fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(RULES)
        ax.set_xlabel("budget-end rule")
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=6)
    fig.text(0.01, 0.005, "ST-3 end-game, flip 0, true prior, pooled over stream seeds; "
             "top = ST-3's committed cell; wrong of all games, FPR of in-pool, won of all games.\n"
             "analyze_st3_endgame, {} vs {}".format(run_dir, base_dir), fontsize=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = os.path.join(run_dir, "fig_endgame_rules.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    a = ap.parse_args()
    out = analyze(a.run_dir, a.base_dir)
    print("{} jobs, complete grid: {}, jobs failing EG0 checks: {}".format(
        out["n_jobs"], out["complete_grid"], len(out["job_check_failures"])))
    for k, x in (out["verdicts"] or out["verdicts_partial_grid"]).items():
        print("  {:4s} {}{}".format(k, x["verdict"],
                                    "  ({})".format(x["reading"]) if "reading" in x else ""))
    print("wrote {}".format(os.path.join(a.run_dir, "analysis.json")))


if __name__ == "__main__":
    main()
