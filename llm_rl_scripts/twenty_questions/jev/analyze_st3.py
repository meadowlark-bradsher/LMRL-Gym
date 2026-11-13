"""Score ST-3 (``STRESS_TAIL_PREREG.md``, *Scoring*) from a run directory.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_st3 \
        --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_<stamp>

Reads ``jobs/*.json.gz`` (per-game columns) and ``stream_census.json``, writes
``analysis.json`` and the figures into the run directory. Every rule below is
the prereg's; the code is the reference when wording is ambiguous.

Game populations (per job, then pooled over the stream and noise seeds of a cell):

* *novel*: the secret was not in ``H_known`` when the game started.
* *necessary call*: a novel game that ended with an expert call.
* *false escalation*: an in-pool game that ended with an expert call.
* TPR = necessary calls / novel games; FPR = false escalations / in-pool games;
  AUC = ``p_unknown_max`` of novel games against in-pool games, ties at 0.5.
* *autonomous*: the game ended ``won``.
* halves: games 0-2499 and 2500-4999.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze_openworld import auc

WON, EXPERT, WRONG, FCONF = 0, 1, 2, 3
HALF = 2500
WINDOW = 250
LOOP_FREE = [("primary", "soft"), ("norepeat", "hard"), ("norepeat", "modeswitch_guard")]
NOREPEAT = [("norepeat", "hard"), ("norepeat", "modeswitch_guard")]
PUBLISHED_DISCRETE = [("primary", "hard"), ("primary", "modeswitch_guard")]
P1_SIGMAS = [0.0, 0.25, 0.5]
P1_TPR_DRIFT = 0.05
P1_RATIO_TOL = 0.25
P2_TOL = 0.02
P2B_MIN = 0.10
P3_TPR_DROP = 0.05
P4_SIGMA_TOL = 0.05
P4_TPR_TOL = 0.02
P4_FIT_GAME = 2000
P5_ADJ_RISE = 0.01
P5_BIN_SPREAD = 0.02
P5_BINS = [(100, 200), (200, 400), (400, 800), (800, 1600)]
P5_MIN_NOVEL, P5_MIN_INPOOL = 30, 200


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def rates(cols: dict, idx: Sequence[int]) -> dict:
    """TPR, FPR, AUC, necessary-call rate, autonomous share over games ``idx``."""
    nov = [i for i in idx if not cols["in_pool"][i]]
    inp = [i for i in idx if cols["in_pool"][i]]
    nec = sum(1 for i in nov if cols["outcome"][i] == EXPERT)
    fp = sum(1 for i in inp if cols["outcome"][i] == EXPERT)
    scores = [cols["p_unknown_max"][i] for i in nov + inp]
    labels = [1] * len(nov) + [0] * len(inp)
    return {"games": len(idx), "novel": len(nov), "in_pool": len(inp),
            "tpr": _rate(nec, len(nov)), "fpr": _rate(fp, len(inp)),
            "auc": auc(scores, labels) if nov and inp else float("nan"),
            "necessary_rate": _rate(nec, len(idx)),
            "autonomous": _rate(sum(1 for i in idx if cols["outcome"][i] == WON), len(idx)),
            "wrong_rate": _rate(sum(1 for i in idx if cols["outcome"][i] == WRONG), len(idx)),
            "repeat_rate": _rate(sum(1 for i in idx if cols["n_repeat_questions"][i] > 0),
                                 len(idx))}


class Pool_:
    """Per-game columns of every job of a cell, concatenated (with a job offset)."""

    def __init__(self):
        self.cols: Dict[str, list] = defaultdict(list)
        self.g: List[int] = []           # game index within its own stream
        self.fits_at: List[Optional[float]] = []

    def add(self, rec: dict) -> None:
        cols = rec["games"]
        n = len(cols["obj"])
        for k, v in cols.items():
            self.cols[k].extend(v)
        self.g.extend(range(n))
        fit = [f for f in rec["meta"].get("online_fits", []) if f[0] == P4_FIT_GAME]
        self.fits_at.append(fit[0][1] if fit else None)

    def idx(self, pred) -> List[int]:
        return [i for i, g in enumerate(self.g) if pred(g, i)]


def cell_metrics(p: Pool_) -> dict:
    allg = list(range(len(p.g)))
    out = {"all": rates(p.cols, allg),
           "first_half": rates(p.cols, p.idx(lambda g, i: g < HALF)),
           "second_half": rates(p.cols, p.idx(lambda g, i: g >= HALF)),
           "from_2000": rates(p.cols, p.idx(lambda g, i: g >= P4_FIT_GAME))}
    n_win = (max(p.g) + 1) // WINDOW if p.g else 0
    by_win: List[List[int]] = [[] for _ in range(n_win)]
    by_bin: Dict[str, List[int]] = {"{}-{}".format(lo, hi): [] for lo, hi in P5_BINS}
    for i, g in enumerate(p.g):
        if g // WINDOW < n_win:
            by_win[g // WINDOW].append(i)
        k = p.cols["k_before"][i]
        for lo, hi in P5_BINS:
            if lo <= k < hi:
                by_bin["{}-{}".format(lo, hi)].append(i)
    out["windows"] = [rates(p.cols, ix) for ix in by_win]
    out["k_bins"] = {key: rates(p.cols, ix) for key, ix in by_bin.items()}
    out["mean_prior_windows"] = [
        sum(p.cols["prior"][i] for i in ix) / len(ix) if ix else float("nan")
        for ix in by_win]
    fits = [f for f in p.fits_at if f is not None]
    out["online_sigma_at_2000"] = sum(fits) / len(fits) if fits else None
    return out


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #

def _ok(x) -> bool:
    return x is not None and x == x


def _v(parts: Dict[str, bool]) -> dict:
    return {"verdict": "PASS" if parts and all(parts.values()) else "FAIL", "parts": parts}


def stream_halves(streams: dict, sigma: float) -> Tuple[float, float]:
    vs = [v for v in streams.values() if abs(v["sigma"] - sigma) < 1e-9]
    return (sum(v["true_pnew_first_half"] for v in vs) / len(vs),
            sum(v["true_pnew_second_half"] for v in vs) / len(vs))


def score(cells: Dict[tuple, dict], streams: dict, sigmas, flips) -> dict:
    def c(sg, pr, fl, arm, bk):
        return cells.get((sg, pr, fl, arm, bk))

    v: Dict[str, dict] = {}

    # P1: expert load tracks the missing mass (loop-free arms, true prior, sigma <= 0.5).
    parts = {}
    for arm, bk in LOOP_FREE:
        for fl in flips:
            for sg in P1_SIGMAS:
                m = c(sg, "true", fl, arm, bk)
                if m is None:
                    parts["{}_{}_f{}_s{}".format(arm, bk, fl, sg)] = False
                    continue
                t1, t2 = m["first_half"]["tpr"], m["second_half"]["tpr"]
                s1, s2 = stream_halves(streams, sg)
                r1, r2 = m["first_half"]["necessary_rate"], m["second_half"]["necessary_rate"]
                key = "{}_{}_f{:g}_s{:g}".format(arm, bk, fl, sg)
                parts[key + "_tpr_drift"] = _ok(t1) and _ok(t2) and abs(t2 - t1) <= P1_TPR_DRIFT
                ratio = (r2 / r1) / (s2 / s1) if r1 and s1 else float("nan")
                parts[key + "_tracks"] = _ok(ratio) and abs(ratio - 1.0) <= P1_RATIO_TOL
    v["P1"] = _v(parts)

    # P2: with the loop removed, the discrete gate ignores the prior (flip 0).
    parts = {}
    for arm, bk in NOREPEAT:
        for sg in sigmas:
            ms = [c(sg, pr, 0.0, arm, bk) for pr in ("true", "online", "fixed")]
            key = "{}_s{:g}".format(bk, sg)
            if any(m is None for m in ms):
                parts[key] = False
                continue
            for metric in ("tpr", "fpr"):
                xs = [m["all"][metric] for m in ms]
                parts[key + "_" + metric] = all(_ok(x) for x in xs) and max(xs) - min(xs) <= P2_TOL
    v["P2"] = _v(parts)

    # P2b: in the published arm the prior reaches the discrete gate through the loop.
    parts = {}
    for arm, bk in PUBLISHED_DISCRETE:
        t, f = c(0.75, "true", 0.0, arm, bk), c(0.75, "fixed", 0.0, arm, bk)
        parts[bk] = (t is not None and f is not None and _ok(t["all"]["fpr"])
                     and _ok(f["all"]["fpr"])
                     and t["all"]["fpr"] - f["all"]["fpr"] >= P2B_MIN)
    v["P2b"] = _v(parts)

    # P3: the soft gate depends on the prior.
    parts = {}
    for fl in flips:
        t, f = c(0.75, "true", fl, "primary", "soft"), c(0.75, "fixed", fl, "primary", "soft")
        parts["s0.75_f{:g}_tpr_drop".format(fl)] = (
            t is not None and f is not None
            and t["all"]["tpr"] - f["all"]["tpr"] > P3_TPR_DROP)
        t, f = c(0.0, "true", fl, "primary", "soft"), c(0.0, "fixed", fl, "primary", "soft")
        parts["s0_f{:g}_fpr_rise".format(fl)] = (
            t is not None and f is not None and f["all"]["fpr"] > t["all"]["fpr"])
    v["P3"] = _v(parts)

    # P4: the online estimate is usable (sigma >= 0.25).
    parts = {}
    for sg in [s for s in sigmas if s >= 0.25]:
        for fl in flips:
            for arm, bk in (("primary", "soft"), ("norepeat", "modeswitch_guard")):
                m = c(sg, "online", fl, arm, bk)
                s_hat = m["online_sigma_at_2000"] if m else None
                parts["{}_s{:g}_f{:g}_sigma".format(bk, sg, fl)] = (
                    s_hat is not None and abs(s_hat - sg) <= P4_SIGMA_TOL)
            o, t = c(sg, "online", fl, "primary", "soft"), c(sg, "true", fl, "primary", "soft")
            parts["soft_s{:g}_f{:g}_tpr".format(sg, fl)] = (
                o is not None and t is not None
                and abs(o["from_2000"]["tpr"] - t["from_2000"]["tpr"]) <= P4_TPR_TOL)
    v["P4"] = _v(parts)

    # P5: the ranking falls with sigma, but not at matched pool size (true prior, flip 0).
    parts = {}
    for arm, bk in LOOP_FREE:
        ms = [c(sg, "true", 0.0, arm, bk) for sg in sigmas]
        if any(m is None for m in ms):
            parts[bk + "_monotone"] = False
            continue
        a = [m["all"]["auc"] for m in ms]
        parts[bk + "_monotone"] = all(_ok(x) for x in a) and all(
            a[i + 1] <= a[i] + P5_ADJ_RISE for i in range(len(a) - 1))
        qualifying = 0
        for lo, hi in P5_BINS:
            key = "{}-{}".format(lo, hi)
            vals = [m["k_bins"][key]["auc"] for m in ms
                    if m["k_bins"][key]["novel"] >= P5_MIN_NOVEL
                    and m["k_bins"][key]["in_pool"] >= P5_MIN_INPOOL]
            if len(vals) >= 2:
                qualifying += 1
                parts["{}_bin{}".format(bk, key)] = (all(_ok(x) for x in vals)
                                                     and max(vals) - min(vals) <= P5_BIN_SPREAD)
        parts[bk + "_has_bin"] = qualifying > 0
    v["P5"] = _v(parts)
    return v


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def figures(cells, streams, sigmas, run_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prov = "analyze_st3, {}".format(os.path.normpath(run_dir))
    paths = []
    fig, axes = plt.subplots(1, len(LOOP_FREE), figsize=(4 * len(LOOP_FREE), 3.4), sharey=True)
    for ax, (arm, bk) in zip(axes, LOOP_FREE):
        for sg in sigmas:
            m = cells.get((sg, "true", 0.0, arm, bk))
            if m is None:
                continue
            xs = [(w + 0.5) * WINDOW for w in range(len(m["windows"]))]
            ys = [w["necessary_rate"] for w in m["windows"]]
            line = ax.plot(xs, [y if y and y > 0 else float("nan") for y in ys],
                           label="σ={:g} calls".format(sg))[0]
            vs = [s for s in streams.values() if abs(s["sigma"] - sg) < 1e-9]
            tp = [sum(s["window_true_pnew"][w] for s in vs) / len(vs) for w in range(len(xs))]
            ax.plot(xs, tp, ls=":", color=line.get_color())
        ax.set_yscale("log")
        ax.set_title("{} ({})".format(bk, arm), fontsize=8)
        ax.set_xlabel("game")
    axes[0].set_ylabel("necessary expert calls per game\n(dotted: stream's true p_new)")
    axes[0].legend(fontsize=6)
    fig.text(0.01, 0.005, "ST-3 P1, true prior, flip 0, pooled over stream seeds. " + prov,
             fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(run_dir, "fig_expert_load_vs_missing_mass.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, len(LOOP_FREE), figsize=(4 * len(LOOP_FREE), 3.2), sharey=True)
    for ax, (arm, bk) in zip(axes, LOOP_FREE):
        for sg in sigmas:
            m = cells.get((sg, "true", 0.0, arm, bk))
            if m is None:
                continue
            xs, ys = [], []
            for lo, hi in P5_BINS:
                r = m["k_bins"]["{}-{}".format(lo, hi)]
                if r["novel"] >= P5_MIN_NOVEL and r["in_pool"] >= P5_MIN_INPOOL:
                    xs.append((lo * hi) ** 0.5)
                    ys.append(r["auc"])
            ax.plot(xs, ys, marker="o", label="σ={:g}".format(sg))
        ax.set_xscale("log")
        ax.set_title("{} ({})".format(bk, arm), fontsize=8)
        ax.set_xlabel("|H_known| entering the game (bin)")
    axes[0].set_ylabel("novelty AUC")
    axes[0].legend(fontsize=6)
    fig.text(0.01, 0.005, "ST-3 P5, true prior, flip 0; bins with >= {} novel and >= {} "
             "in-pool games. ".format(P5_MIN_NOVEL, P5_MIN_INPOOL) + prov, fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(run_dir, "fig_auc_by_pool_size.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def analyze(run_dir: str, make_figures: bool = True) -> dict:
    pools: Dict[tuple, Pool_] = defaultdict(Pool_)
    jobs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        j = rec["job"]
        pools[(j["sigma"], j["prior"], j["flip"], j["arm"], j["backend"])].add(rec)
        jobs.append(j)
    with open(os.path.join(run_dir, "stream_census.json")) as f:
        streams = json.load(f)
    cells = {k: cell_metrics(p) for k, p in pools.items()}
    sigmas = sorted({k[0] for k in cells})
    flips = sorted({k[2] for k in cells})
    config = json.load(open(os.path.join(run_dir, "config.json")))
    complete = config.get("mode") == "run" and len(jobs) == 648
    verdicts = score(cells, streams, sigmas, flips)
    out = {"schema": "twenty_questions_stress_st3_analysis_v1", "run_dir": run_dir,
           "n_jobs": len(jobs), "complete_grid": complete,
           "cells": [{"sigma": k[0], "prior": k[1], "flip": k[2], "arm": k[3],
                      "backend": k[4], **v} for k, v in sorted(cells.items())],
           "verdicts": verdicts if complete else {},
           "verdicts_partial_grid": {} if complete else verdicts}
    if make_figures:
        out["figures"] = [os.path.basename(p) for p in figures(cells, streams, sigmas, run_dir)]
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = analyze(args.run_dir, make_figures=not args.no_figures)
    print("{} jobs, complete grid: {}".format(out["n_jobs"], out["complete_grid"]))
    for k, v in sorted((out["verdicts"] or out["verdicts_partial_grid"]).items()):
        print("  {:6s} {}".format(k, v["verdict"]))
    print("wrote {}".format(os.path.join(args.run_dir, "analysis.json")))


if __name__ == "__main__":
    main()
