"""Score ST-1 (crowding x scale) against its pre-registration.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_st1 \
        --dir llm_rl_scripts/twenty_questions/jev/results/stress_st1_<stamp>

Reads ``summary.json`` and the per-job game columns (``jobs/*.json.gz``) that
``stress_st1`` wrote. Pools a *cell* -- (N, crowding level, flip rate, arm,
backend) -- over its replicate pools and noise seeds, and scores P1-P7 exactly
as the *Scoring* section of ``STRESS_CROWDING_PREREG.md`` defines them. Writes
``analysis.json`` and four figures into the run directory; every caption names
this analyzer and the run directory, so each figure regenerates like a number.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .analyze_openworld import auc

WON, EXPERT, WRONG, FALSE_CONFIRM = 0, 1, 2, 3
DISCRETE = ("hard", "modeswitch")
ARMS = ("primary", "norepeat")
LEVEL_ORDER = {"twins": 0, "crowded": 1, "realmatrix": 2, "real": 2, "floor": 3}


# --------------------------------------------------------------------------- #
# Loading and pooling                                                          #
# --------------------------------------------------------------------------- #

def load_cells(run_dir: str) -> Tuple[dict, Dict[tuple, dict]]:
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    census = summary["census"]
    games: Dict[tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pools: Dict[tuple, set] = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        j = rec["job"]
        key = (j["n"], j["level"], j["flip"], j["arm"], j["backend"])
        for col, vals in rec["games"].items():
            games[key][col].extend(vals)
        pools[key].add(j["pool_id"])
    cells = {key: cell_metrics(cols, [census[p] for p in sorted(pools[key])])
             for key, cols in games.items()}
    return summary, cells


def _frac(num: int, den: int) -> float:
    return num / den if den else float("nan")


def cell_metrics(g: Dict[str, list], censuses: List[dict]) -> dict:
    n = len(g["outcome"])
    inp = [i for i in range(n) if g["in_pool"][i]]
    nov = [i for i in range(n) if not g["in_pool"][i]]
    oc = g["outcome"]
    wrong = [i for i in range(n) if oc[i] == WRONG and g["n_live_final"][i] >= 0]
    struck = [i for i in inp if g["struck_by_flip"][i] == 1]
    qs = sorted(g["n_questions"])
    n_q = sum(qs)
    return {
        "games": n, "in_pool_games": len(inp), "novel_games": len(nov),
        "won": _frac(sum(oc[i] == WON for i in inp), len(inp)),
        "safe": _frac(sum(oc[i] == EXPERT for i in inp), len(inp)),
        "wrong": _frac(sum(oc[i] == WRONG for i in inp), len(inp)),
        "false_confirm": _frac(sum(oc[i] == FALSE_CONFIRM for i in inp), len(inp)),
        "fail": _frac(sum(oc[i] != WON for i in inp), len(inp)),
        "q95": qs[min(n - 1, int(math.ceil(0.95 * n)) - 1)] if n else float("nan"),
        "mean_questions": n_q / n if n else float("nan"),
        "guess_share": _frac(sum(g["n_guesses"]), n_q),
        "budget_exhaustion_share": _frac(sum(g["n_live_final"][i] >= 2 for i in wrong),
                                         len(wrong)),
        "repeat_rate": _frac(sum(r > 0 for r in g["n_repeat_questions"]), n),
        "flip_struck": len(struck),
        "rescue_trigger_rate": _frac(sum(g["emptied"][i] == 1 for i in struck), len(struck)),
        "auc": (auc(g["p_unknown_max"], [0 if x else 1 for x in g["in_pool"]])
                if inp and nov else float("nan")),
        # Crowding coordinates: means over the cell's replicate pools.
        "crowding": sum(c["nn_within_1"] / c["n_objects"] for c in censuses) / len(censuses),
        "identical_frac": sum(c["objects_sharing_a_row"] / c["n_objects"]
                              for c in censuses) / len(censuses),
        "n_pools": len(censuses),
    }


# --------------------------------------------------------------------------- #
# Scoring helpers                                                              #
# --------------------------------------------------------------------------- #

def spearman(xs: List[float], ys: List[float]) -> float:
    pts = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    if len(pts) < 3:
        return float("nan")
    import warnings
    from scipy.stats import spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant input -> nan, scored as not holding
        rho = spearmanr([p[0] for p in pts], [p[1] for p in pts]).correlation
    return float(rho) if rho is not None else float("nan")


def select(cells, **kw) -> List[Tuple[tuple, dict]]:
    names = ("n", "level", "flip", "arm", "backend")
    out = []
    for key, m in cells.items():
        d = dict(zip(names, key))
        if all((d[k] in v) if isinstance(v, (tuple, list, set)) else d[k] == v
               for k, v in kw.items()):
            out.append((key, m))
    return sorted(out)


def verdict(ok: Optional[bool]) -> str:
    return {True: "PASS", False: "FAIL", None: "NOT TRIGGERED"}[ok]


# --------------------------------------------------------------------------- #
# Predictions (definitions mirror STRESS_CROWDING_PREREG.md, Scoring)          #
# --------------------------------------------------------------------------- #

def score(cells: Dict[tuple, dict]) -> dict:
    res: Dict[str, dict] = {}

    # P1 -- crowding costs questions before it costs errors.
    p1 = {}
    for arm in ARMS:
        for bk in DISCRETE:
            cs = select(cells, flip=0.0, arm=arm, backend=bk)
            a = all(m["wrong"] == 0 for _, m in cs if m["q95"] < 20)
            rho = spearman([m["guess_share"] for _, m in cs],
                           [m["identical_frac"] for _, m in cs])
            p1["{}/{}".format(arm, bk)] = {"a_no_wrong_below_budget": a,
                                           "b_spearman_guess_share_vs_identical": rho,
                                           "pass": a and rho > 0}
    res["P1"] = p1

    # P2 -- when the discrete backends fail, it is the budget.
    p2 = {}
    for arm in ARMS:
        for bk in DISCRETE:
            cs = select(cells, flip=0.0, arm=arm, backend=bk)
            failing = [(k, m) for k, m in cs if m["wrong"] > 0]
            if not failing:
                p2["{}/{}".format(arm, bk)] = {"pass": None, "failing_cells": 0}
                continue
            a = all(m["budget_exhaustion_share"] >= 0.95 for _, m in failing)
            twins_big = [m["crowding"] for k, m in cs if k[1] == "twins" and k[0] >= 2000]
            b = (min(m["crowding"] for _, m in failing) >= min(twins_big)
                 if twins_big else False)
            p2["{}/{}".format(arm, bk)] = {
                "failing_cells": len(failing), "a_budget_share_ge_095": a,
                "b_first_at_twins_N_ge_2000": b, "pass": a and b,
                "min_budget_share": min(m["budget_exhaustion_share"] for _, m in failing)}
    res["P2"] = p2

    # P3 -- crowding disarms the rescue (noisy cells).
    p3 = {}
    for arm in ARMS:
        rec, crowd, trig, crowd_t = [], [], [], []
        for (n, lvl, flip, a, bk), m_ms in select(cells, flip=(0.05, 0.10), arm=arm,
                                                  backend="modeswitch"):
            m_h = cells.get((n, lvl, flip, arm, "hard"))
            if m_h is None:
                continue
            if m_h["fail"] > 0:
                rec.append((m_h["fail"] - m_ms["fail"]) / m_h["fail"])
                crowd.append(m_ms["crowding"])
            trig.append(m_ms["rescue_trigger_rate"])
            crowd_t.append(m_ms["crowding"])
        r1, r2 = spearman(rec, crowd), spearman(trig, crowd_t)
        p3[arm] = {"spearman_recovery_vs_crowding": r1,
                   "spearman_trigger_vs_crowding": r2,
                   "pass": (r1 <= -0.5) and (r2 <= -0.5)}
    res["P3"] = p3

    # P4 -- noise moves the failure boundary in.
    p4 = {}
    for arm in ARMS:
        for bk in DISCRETE:
            def c_star(flip):
                cs = [m["crowding"] for _, m in select(cells, flip=flip, arm=arm, backend=bk)
                      if m["wrong"] > 0.01]
                return min(cs) if cs else math.inf
            c0, c10 = c_star(0.0), c_star(0.10)
            ok = None if (c0 == math.inf and c10 == math.inf) else (c10 < c0)
            p4["{}/{}".format(arm, bk)] = {"c_star_flip0": c0, "c_star_flip10": c10,
                                           "pass": ok}
    res["P4"] = p4

    # P5 -- the detector degrades with crowding, discrete backends first.
    p5 = {}
    for flip in (0.0, 0.05, 0.10):
        per_bk = {}
        for bk in ("soft",) + DISCRETE:
            cs = select(cells, flip=flip, arm="primary", backend=bk)
            if not cs:
                continue
            rho = spearman([m["auc"] for _, m in cs], [m["crowding"] for _, m in cs])
            worst = min(cs, key=lambda kv: (kv[1]["auc"], kv[0]))
            per_bk[bk] = {"spearman_auc_vs_crowding": rho, "min_auc_level": worst[0][1],
                          "min_auc": worst[1]["auc"]}
        split = any(cells.get((n, lvl, flip, "primary", "soft"), {}).get("auc", 0) >= 0.95
                    and m["auc"] < 0.95
                    for (n, lvl, _, _, _), m in select(cells, flip=flip, arm="primary",
                                                       backend="hard"))
        ok = (all(v["spearman_auc_vs_crowding"] < 0 for v in per_bk.values())
              and split and all(v["min_auc_level"] == "twins" for v in per_bk.values()))
        p5["flip{:g}".format(flip)] = {"backends": per_bk, "b_soft_holds_where_hard_falls": split,
                                       "pass": ok}
    res["P5"] = {"by_flip": p5, "pass": all(v["pass"] for v in p5.values())}

    # P6 -- the real-matrix anchor.
    h = cells.get((158, "realmatrix", 0.10, "primary", "hard"))
    ms = cells.get((158, "realmatrix", 0.10, "primary", "modeswitch"))
    res["P6"] = ({"pass": None, "missing": True} if h is None or ms is None else
                 {"hard_wrong": h["wrong"], "ms_wrong": ms["wrong"], "hard_fail": h["fail"],
                  "ms_fail": ms["fail"],
                  "pass": h["wrong"] == 0 and ms["wrong"] == 0 and ms["fail"] < h["fail"]})

    # P7 -- crowding pulls the loop in.
    anchor = [cells.get((158, "realmatrix", 0.0, "primary", bk)) for bk in DISCRETE]
    a = all(m is not None and m["repeat_rate"] <= 0.02 for m in anchor)
    prim = select(cells, arm="primary", backend=DISCRETE)
    rho = spearman([m["repeat_rate"] for _, m in prim], [m["crowding"] for _, m in prim])
    diffs = []
    for (n, lvl, flip, arm, bk), m in select(cells, level="twins", n=(2000, 5000),
                                             arm="primary", backend=DISCRETE):
        nr = cells.get((n, lvl, flip, "norepeat", bk))
        if nr is not None:
            diffs.append(nr["won"] - m["won"])
    c = (sum(diffs) / len(diffs)) if diffs else float("nan")
    res["P7"] = {"a_anchor_repeat_le_002": a, "b_spearman_repeat_vs_crowding": rho,
                 "c_mean_won_gap_twins_N_ge_2000": c,
                 "pass": a and rho >= 0.5 and c >= 0.05}

    # Attribution: a P1-P5 part that holds without repeats but fails with them.
    attributed = []
    for p in ("P1", "P2", "P4"):
        for bk in DISCRETE:
            pr, nr = res[p].get("primary/" + bk, {}), res[p].get("norepeat/" + bk, {})
            if nr.get("pass") is True and pr.get("pass") is False:
                attributed.append("{} {}".format(p, bk))
    if res["P3"]["norepeat"]["pass"] and not res["P3"]["primary"]["pass"]:
        attributed.append("P3")
    res["attributed_to_loop"] = attributed
    return res


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def figures(cells: Dict[tuple, dict], run_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = os.path.relpath(run_dir)
    src = ("Source: analyze_st1 on {}\n"
           "Each point is one cell (N, crowding level, flip rate, arm, backend) pooled over "
           "its replicate pools and noise seeds.").format(
        rel if not rel.startswith("..") else os.path.abspath(run_dir))
    out = []
    xlab = "crowding: share of objects within 1 attribute of a neighbour"

    def save(fig, name):
        fig.text(0.01, 0.005, src, fontsize=7, color="0.35", va="bottom")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        path = os.path.join(run_dir, name)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        out.append(path)

    flips = (0.0, 0.05, 0.10)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for r, bk in enumerate(DISCRETE):
        for c, flip in enumerate(flips):
            ax = axes[r][c]
            cs = select(cells, flip=flip, arm="primary", backend=bk)
            xs = [m["crowding"] for _, m in cs]
            for metric, color in (("won", "C2"), ("safe", "C0"), ("wrong", "C3")):
                ax.scatter(xs, [m[metric] for _, m in cs], s=18, color=color, label=metric)
            ax.set_title("{} (primary), flips {:.0%}".format(bk, flip), fontsize=9)
            if c == 0:
                ax.set_ylabel("share of in-pool games")
    axes[0][0].legend(fontsize=8)
    fig.supxlabel(xlab, fontsize=9, y=0.07)
    save(fig, "fig_st1_outcomes.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, marker in (("primary", "o"), ("norepeat", "^")):
        xs, rec, trig = [], [], []
        for (n, lvl, flip, a, bk), m in select(cells, flip=(0.05, 0.10), arm=arm,
                                               backend="modeswitch"):
            h = cells.get((n, lvl, flip, arm, "hard"))
            if h and h["fail"] > 0:
                xs.append(m["crowding"])
                rec.append((h["fail"] - m["fail"]) / h["fail"])
                trig.append(m["rescue_trigger_rate"])
        axes[0].scatter(xs, rec, marker=marker, s=22, label=arm)
        axes[1].scatter(xs, trig, marker=marker, s=22, label=arm)
    axes[0].set_ylabel("modeswitch recovery over hard")
    axes[1].set_ylabel("rescue trigger rate (live set emptied)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.supxlabel(xlab, fontsize=9, y=0.08)
    save(fig, "fig_st1_rescue.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, flip in zip(axes, flips):
        for bk in ("soft",) + DISCRETE:
            cs = select(cells, flip=flip, arm="primary", backend=bk)
            ax.scatter([m["crowding"] for _, m in cs], [m["auc"] for _, m in cs], s=18, label=bk)
        ax.axhline(0.95, color="0.6", lw=0.8, ls="--")
        ax.set_title("novelty AUC, flips {:.0%}".format(flip), fontsize=9)
    axes[0].set_ylabel("pooled AUC")
    axes[0].legend(fontsize=8)
    fig.supxlabel(xlab, fontsize=9, y=0.08)
    save(fig, "fig_st1_auc.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for arm, marker in (("primary", "o"), ("norepeat", "^")):
        for bk, color in (("hard", "C1"), ("modeswitch", "C4")):
            cs = select(cells, arm=arm, backend=bk)
            ax.scatter([m["crowding"] for _, m in cs], [m["repeat_rate"] for _, m in cs],
                       marker=marker, color=color, s=18, label="{} {}".format(arm, bk))
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel("share of games that re-ask a question")
    ax.legend(fontsize=8)
    save(fig, "fig_st1_repeat.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    summary, cells = load_cells(args.dir)
    res = score(cells)
    print("{} cells from {}".format(len(cells), args.dir))
    print("{:>5} {:10s} {:>5} {:9s} {:10s} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
        "N", "level", "flip", "arm", "backend", "crowd", "won", "safe", "wrong", "AUC",
        "q/g", "rep"))
    for key, m in sorted(cells.items(), key=lambda kv: (kv[0][0], LEVEL_ORDER.get(kv[0][1], 9),
                                                         kv[0][2], kv[0][3], kv[0][4])):
        print("{:>5} {:10s} {:>5.2f} {:9s} {:10s} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} "
              "{:6.2f} {:6.3f}".format(*key, m["crowding"], m["won"], m["safe"], m["wrong"],
                                       m["auc"], m["mean_questions"], m["repeat_rate"]))
    for p in ("P1", "P2", "P3", "P4"):
        for k, v in res[p].items():
            print("{} {:22s} {}".format(p, k, verdict(v["pass"])))
    print("P5 {:22s} {}".format("all flips", verdict(res["P5"]["pass"])))
    print("P6 {:22s} {}".format("real-matrix anchor", verdict(res["P6"]["pass"])))
    print("P7 {:22s} {}".format("loop vs crowding", verdict(res["P7"]["pass"])))
    print("attributed to the loop: {}".format(res["attributed_to_loop"] or "none"))
    paths = [] if args.no_figures else figures(cells, args.dir)
    with open(os.path.join(args.dir, "analysis.json"), "w") as f:
        json.dump({"cells": {"|".join(map(str, k)): v for k, v in cells.items()},
                   "predictions": res, "figures": [os.path.basename(p) for p in paths]},
                  f, indent=1, sort_keys=True, default=str)
    print("wrote {}".format(os.path.join(args.dir, "analysis.json")))


if __name__ == "__main__":
    main()
