"""Score ST-2 (``STRESS_LOOKALIKE_PREREG.md``, *Scoring*) from a run directory.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_st2 \
        --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st2_<stamp>

Reads ``jobs/*.json.gz`` (per-game columns), writes ``analysis.json`` and the
figures into the run directory. Every rule below is the prereg's; the code is the
reference when wording is ambiguous.

Game populations (per job, then pooled over the jobs of a cell):

* *look-alike* games: the secret is one of la000..la099 (object index >= 158).
* *novel* games: the secret was not in ``H_known`` at the start of the game.
* *parent known*: a look-alike game played after its parent was admitted.
* *first appearance*: the object's first game in the stream.
* *fire*: the game ended with an expert call (``outcome == expert``); the fire turn
  is the recorded turn ``p_unknown`` first crossed the gate.
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

from .analyze_openworld import auc

N_REAL = 158
WON, EXPERT, WRONG, FCONF = 0, 1, 2, 3
DISCRETE = ["hard", "modeswitch", "modeswitch_guard"]
MODESWITCH = ["modeswitch", "modeswitch_guard"]
TAU_AUC = 0.01          # P2: an adjacent AUC drop up to this is seed noise
TAU_FIRE = 1.0          # P2: an adjacent median-fire-turn rise up to this is seed noise
P1_EXTRA_Q = (0.5, 1.5)
P3_DELAY = (1.0, 3.0)
P4_RATIO = (2 / 3, 3 / 2)
P4_PG_MIN = 0.8
P5_TAX = (0.5, 1.5)


# --------------------------------------------------------------------------- #
# Per-job derived game table                                                   #
# --------------------------------------------------------------------------- #

def derive(rec: dict) -> List[dict]:
    """One dict per game with the populations above resolved."""
    cols = rec["games"]
    meta = rec["meta"]
    n = len(cols["obj"])
    parent_of = meta["parent_of"]
    names = sorted(parent_of)                       # la000.. in index order
    parent_idx = {N_REAL + i: None for i in range(len(names))}
    # parent indices: recover from the job's object order (real objects first).
    real_index = rec.get("_real_index")
    for i, la in enumerate(names):
        parent_idx[N_REAL + i] = real_index[parent_of[la]] if real_index else None
    admitted_at: Dict[int, int] = {}
    for g in range(n):
        o = cols["obj"][g]
        if cols["outcome"][g] == EXPERT and not cols["in_pool"][g] and o not in admitted_at:
            admitted_at[o] = g
    out = []
    for g in range(n):
        o = cols["obj"][g]
        is_la = o >= N_REAL
        p = parent_idx.get(o) if is_la else None
        out.append({
            "g": g, "obj": o, "is_la": is_la, "parent": p,
            "parent_known": (is_la and p is not None and p in admitted_at
                             and admitted_at[p] < g),
            "in_pool": bool(cols["in_pool"][g]), "is_new": bool(cols["is_new"][g]),
            "outcome": cols["outcome"][g], "nq": cols["n_questions"][g],
            "score": cols["p_unknown_max"][g], "fire_turn": cols["fire_turn"][g],
            "parent_guessed": cols["parent_guessed"][g],
            "n_repeat_guesses": cols["n_repeat_guesses"][g],
            "n_retractions": cols["n_retractions"][g],
            "n_repeat_questions": cols["n_repeat_questions"][g],
        })
    return out, admitted_at, parent_idx


P5_WINDOW = 100         # P5: control games within this many games of a parent game
# P5 counts only games from here on: before it, the stream's PY prior (k = distinct
# objects seen) is above the repeat-loop threshold of ~0.15 (STRESS_ST0.md, Finding
# 2) and in-pool games of the discrete backends run to 20 questions. Derived from the
# stream alone (stress_st2.stream_and_prior), so it is the same for every job.
P5_FROM_GAME = 274


def parent_tax_events(games: List[dict], admitted_at: Dict[int, int],
                      parent_idx: Dict[int, int]) -> List[dict]:
    """P5 events. For each parent known before its earliest-admitted look-alike's
    admission at game a: the parent's in-pool games before vs after a. Each parent
    game is adjusted by the mean questions of the *control* games (in-pool games of
    real objects that are no look-alike's parent) within ``P5_WINDOW`` games of it,
    so the stream-wide drift (a growing ``H_known``, the early high prior) cancels.
    ``did`` = mean adjusted after - mean adjusted before; ``raw`` is unadjusted."""
    parents = {p for p in parent_idx.values() if p is not None}
    first: Dict[int, int] = {}
    for la, a in admitted_at.items():
        if la >= N_REAL:
            p = parent_idx[la]
            if (a >= P5_FROM_GAME and p in admitted_at and admitted_at[p] < a
                    and (p not in first or a < first[p])):
                first[p] = a
    n = len(games)
    ctrl_nq = [0.0] * n
    ctrl_is = [False] * n
    games = [x for x in games if x["g"] >= P5_FROM_GAME]
    for x in games:
        if x["in_pool"] and not x["is_la"] and x["obj"] not in parents:
            ctrl_nq[x["g"]] = x["nq"]
            ctrl_is[x["g"]] = True
    # prefix sums for windowed control means
    cs, cn = [0.0], [0]
    for i in range(n):
        cs.append(cs[-1] + (ctrl_nq[i] if ctrl_is[i] else 0.0))
        cn.append(cn[-1] + (1 if ctrl_is[i] else 0))

    def local(g: int) -> Optional[float]:
        lo, hi = max(0, g - P5_WINDOW), min(n, g + P5_WINDOW + 1)
        k = cn[hi] - cn[lo]
        return (cs[hi] - cs[lo]) / k if k else None

    events = []
    for p, a in first.items():
        pg = [x for x in games if x["obj"] == p and x["in_pool"]]
        before = [x for x in pg if x["g"] < a]
        after = [x for x in pg if x["g"] > a]
        adj_b = [x["nq"] - local(x["g"]) for x in before if local(x["g"]) is not None]
        adj_a = [x["nq"] - local(x["g"]) for x in after if local(x["g"]) is not None]
        if not adj_b or not adj_a:
            continue
        events.append({"parent": p, "admitted_at": a,
                       "raw": (statistics.mean(x["nq"] for x in after)
                               - statistics.mean(x["nq"] for x in before)),
                       "did": statistics.mean(adj_a) - statistics.mean(adj_b),
                       "n_before": len(adj_b), "n_after": len(adj_a)})
    return events


# --------------------------------------------------------------------------- #
# Cell metrics                                                                 #
# --------------------------------------------------------------------------- #

def _frac(xs, pred) -> float:
    xs = list(xs)
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else float("nan")


def _mean(xs) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def _median(xs) -> float:
    xs = list(xs)
    return statistics.median(xs) if xs else float("nan")


def _auc(pos: List[dict], neg: List[dict]) -> float:
    if not pos or not neg:
        return float("nan")
    return auc([x["score"] for x in neg + pos], [0] * len(neg) + [1] * len(pos))


def _ci(xs: Sequence[float]):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    m, se = statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))
    return (m - 1.96 * se, m + 1.96 * se)


def cell_metrics(games: List[dict], events: List[dict], g_rate: float) -> dict:
    inpool = [x for x in games if x["in_pool"]]
    la_nov = [x for x in games if x["is_la"] and not x["in_pool"]]
    ord_nov = [x for x in games if not x["is_la"] and not x["in_pool"]]
    la_all = [x for x in games if x["is_la"]]
    la_first = [x for x in la_all if x["is_new"]]
    ord_first = [x for x in games if not x["is_la"] and x["is_new"]]
    la_nov_pk = [x for x in la_nov if x["parent_known"]]
    m = {
        "games": len(games), "in_pool_games": len(inpool),
        "la_novel_games": len(la_nov), "ord_novel_games": len(ord_nov),
        "la_novel_parent_known_games": len(la_nov_pk),
        "auc_la": _auc(la_nov, inpool), "auc_ord": _auc(ord_nov, inpool),
        "tpr_la": _frac(la_nov, lambda x: x["score"] >= 0.9),
        "tpr_ord": _frac(ord_nov, lambda x: x["score"] >= 0.9),
        "fpr": _frac(inpool, lambda x: x["score"] >= 0.9),
        "median_fire_la": _median(x["fire_turn"] for x in la_nov if x["fire_turn"] >= 0),
        "median_fire_ord": _median(x["fire_turn"] for x in ord_nov if x["fire_turn"] >= 0),
        "la_first_fire_rate": _frac(la_first, lambda x: x["outcome"] == EXPERT),
        "la_first_fire_rate_parent_known": _frac(
            [x for x in la_first if x["parent_known"]], lambda x: x["outcome"] == EXPERT),
        "la_wrong": sum(1 for x in la_all if x["outcome"] == WRONG),
        "la_false_confirm": sum(1 for x in la_all if x["outcome"] == FCONF),
        "wrong_all": sum(1 for x in games if x["outcome"] == WRONG),
        "false_confirm_all": sum(1 for x in games if x["outcome"] == FCONF),
        "la_outcomes": {k: _frac(la_all, lambda x, c=c: x["outcome"] == c)
                        for k, c in (("won", WON), ("expert", EXPERT), ("wrong", WRONG),
                                     ("false_confirm", FCONF))},
        "la_novel_outcomes": {k: _frac(la_nov, lambda x, c=c: x["outcome"] == c)
                              for k, c in (("won", WON), ("expert", EXPERT), ("wrong", WRONG),
                                           ("false_confirm", FCONF))},
        "p_parent_guessed": _frac(la_nov, lambda x: x["parent_guessed"]),
        "p_parent_guessed_parent_known": _frac(la_nov_pk, lambda x: x["parent_guessed"]),
        "fc_rate_parent_known": _frac(la_nov_pk, lambda x: x["outcome"] == FCONF),
        "fc_rate_la_novel": _frac(la_nov, lambda x: x["outcome"] == FCONF),
        "fc_rate_la_in_pool": _frac([x for x in la_all if x["in_pool"]],
                                    lambda x: x["outcome"] == FCONF),
        "extra_q_first_parent_known": (
            _mean(x["nq"] for x in la_first if x["parent_known"] and x["outcome"] == EXPERT)
            - _mean(x["nq"] for x in ord_first if x["outcome"] == EXPERT)),
        "extra_q_first_all": (_mean(x["nq"] for x in la_first if x["outcome"] == EXPERT)
                              - _mean(x["nq"] for x in ord_first if x["outcome"] == EXPERT)),
        "mean_repeat_guesses": _mean(x["n_repeat_guesses"] for x in games),
        "mean_repeat_guesses_la_novel": _mean(x["n_repeat_guesses"] for x in la_nov),
        "mean_retractions_la_novel": _mean(x["n_retractions"] for x in la_nov),
        "mean_questions": _mean(x["nq"] for x in games),
        "repeat_rate": _frac(games, lambda x: x["n_repeat_questions"] > 0),
        "p5_events": len(events),
        "p5_raw": _mean(e["raw"] for e in events),
        "p5_did": _mean(e["did"] for e in events),
        "p5_did_ci": _ci([e["did"] for e in events]),
    }
    pg = m["p_parent_guessed_parent_known"]
    m["fc_expected_parent_known"] = g_rate * pg if g_rate > 0 and pg == pg else float("nan")
    return m


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #

def _v(ok: Optional[bool]) -> str:
    return "NOT TRIGGERED" if ok is None else ("PASS" if ok else "FAIL")


def _in(x: float, lo_hi) -> bool:
    return x == x and lo_hi[0] <= x <= lo_hi[1]


def score(cells: Dict[tuple, dict], d_levels, flips, g_rates, backends) -> dict:
    """``cells`` keys are (arm, g, d, flip, backend)."""
    def c(arm, g, d, flip, bk):
        return cells.get((arm, g, d, flip, bk))

    verdicts = {}
    small_d = [d for d in d_levels if d <= 2]
    big_d = [d for d in d_levels if d >= 6]

    # P1 ------------------------------------------------------------------- #
    for bk in [b for b in DISCRETE if b in backends]:
        rows = [c("exact", 0.0, d, 0.0, bk) for d in d_levels]
        a = all(r["la_first_fire_rate"] >= 0.95 for r in rows)
        b = all(r["la_wrong"] == 0 for r in rows)
        cc = all(_in(c("exact", 0.0, d, 0.0, bk)["extra_q_first_parent_known"], P1_EXTRA_Q)
                 for d in small_d)
        verdicts["P1/" + bk] = {"verdict": _v(a and b and cc), "a_fire_ge_0.95": a,
                                "b_la_wrong_0": b, "c_extra_q_in_range": cc}

    # P2 ------------------------------------------------------------------- #
    for bk in backends:
        for flip in flips:
            aucs = [c("exact", 0.0, d, flip, bk)["auc_la"] for d in d_levels]
            fires = [c("exact", 0.0, d, flip, bk)["median_fire_la"] for d in d_levels]
            a = all(aucs[i + 1] >= aucs[i] - TAU_AUC for i in range(len(aucs) - 1))
            b = all(x == x for x in aucs) and aucs[0] <= min(aucs)
            cc = (all(fires[i + 1] <= fires[i] + TAU_FIRE for i in range(len(fires) - 1))
                  and fires[-1] < fires[0])
            verdicts["P2/{}/f{:g}".format(bk, flip)] = {
                "verdict": _v(a and b and cc), "a_auc_nondecreasing": a,
                "b_auc_lowest_at_d1": b, "c_fire_turn_falls": cc,
                "auc_by_d": aucs, "median_fire_by_d": fires}

    # P3 ------------------------------------------------------------------- #
    if "hard" in backends:
        for ms in [b for b in MODESWITCH if b in backends]:
            parts = {}
            ok = True
            for flip in [f for f in flips if f > 0]:
                for d in small_d:
                    h, m = c("exact", 0.0, d, flip, "hard"), c("exact", 0.0, d, flip, ms)
                    delay = m["median_fire_la"] - h["median_fire_la"]
                    good = _in(delay, P3_DELAY) and m["la_wrong"] == 0
                    parts["d{}_f{:g}".format(d, flip)] = {"delay": delay,
                                                          "la_wrong": m["la_wrong"],
                                                          "holds": good}
                    ok = ok and good
            verdicts["P3/" + ms] = {"verdict": _v(ok), "parts": parts}

    # P4 (headline) ------------------------------------------------------- #
    for bk in backends:
        parts, ok = {}, True
        for g in g_rates:
            for d in small_d:
                r = c("G", g, d, 0.0, bk)
                ratio = (r["fc_rate_parent_known"] / r["fc_expected_parent_known"]
                         if r["fc_expected_parent_known"] else float("nan"))
                good = _in(ratio, P4_RATIO)
                if bk in DISCRETE:
                    good = good and r["p_parent_guessed_parent_known"] >= P4_PG_MIN
                parts["g{:g}_d{}".format(g, d)] = {
                    "fc_rate": r["fc_rate_parent_known"], "expected": r["fc_expected_parent_known"],
                    "ratio": ratio, "p_parent_guessed": r["p_parent_guessed_parent_known"],
                    "holds": good}
                ok = ok and good
        verdicts["P4a/" + bk] = {"verdict": _v(ok), "parts": parts}
    if "soft" in backends:
        for bk in [b for b in DISCRETE if b in backends]:
            parts, ok = {}, True
            for g in g_rates:
                for d in small_d:
                    x = c("G", g, d, 0.0, bk)["fc_rate_parent_known"]
                    s = c("G", g, d, 0.0, "soft")["fc_rate_parent_known"]
                    parts["g{:g}_d{}".format(g, d)] = {"fc": x, "fc_soft": s, "holds": x >= s}
                    ok = ok and x >= s
            verdicts["P4b/" + bk] = {"verdict": _v(ok), "parts": parts}

    # P5 ------------------------------------------------------------------- #
    for bk in backends:
        parts, ok = {}, True
        for d in small_d + big_d:
            r = c("exact", 0.0, d, 0.0, bk)
            if d <= 2:
                good = _in(r["p5_did"], P5_TAX)
            else:
                lo, hi = r["p5_did_ci"]
                good = lo == lo and lo <= 0.0 <= hi
            parts["d{}".format(d)] = {"did": r["p5_did"], "ci": r["p5_did_ci"],
                                      "raw": r["p5_raw"], "events": r["p5_events"],
                                      "holds": good}
            ok = ok and good
        verdicts["P5/" + bk] = {"verdict": _v(ok), "parts": parts}
    return verdicts


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def figures(cells, d_levels, flips, g_rates, backends, run_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prov = "analyze_st2.py on {}".format(os.path.relpath(run_dir))
    paths = []
    fig, axes = plt.subplots(1, len(flips), figsize=(4 * len(flips), 3.4), sharey=True)
    for ax, flip in zip(axes, flips):
        for bk in backends:
            ax.plot(d_levels, [cells[("exact", 0.0, d, flip, bk)]["auc_la"] for d in d_levels],
                    marker="o", label=bk)
        ax.set_title("flip {:g}".format(flip))
        ax.set_xlabel("look-alike distance d")
        ax.set_xscale("log", base=2)
        ax.set_xticks(d_levels)
        ax.set_xticklabels([str(d) for d in d_levels])
    axes[0].set_ylabel("detector AUC, look-alike novelty")
    axes[-1].legend(fontsize=7)
    fig.text(0.01, 0.005, "ST-2 P2. " + prov, fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = os.path.join(run_dir, "fig_auc_by_d.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for bk in backends:
        for d, ls in zip([x for x in d_levels if x <= 2], ("-", "--")):
            ys = [cells[("G", g, d, 0.0, bk)]["fc_rate_parent_known"] for g in g_rates]
            ax.plot(g_rates, ys, marker="o", linestyle=ls, label="{} d={}".format(bk, d))
    ax.set_xlabel("near-name false-yes rate g")
    ax.set_ylabel("false_confirm rate\n(look-alike novelty, parent known)")
    ax.legend(fontsize=6)
    fig.text(0.01, 0.005, "ST-2 P4. " + prov, fontsize=6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(run_dir, "fig_false_confirm_by_g.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def analyze(run_dir: str, make_figures: bool = True) -> dict:
    from .bed import load_hypotheses
    real_index = {h.name: i for i, h in enumerate(load_hypotheses())}
    pooled_games = defaultdict(list)
    pooled_events = defaultdict(list)
    jobs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        rec["_real_index"] = real_index
        job = rec["job"]
        key = (job["arm"], job["g"], job["d"], job["flip"], job["backend"])
        games, admitted, pidx = derive(rec)
        pooled_games[key].extend(games)
        pooled_events[key].extend(parent_tax_events(games, admitted, pidx))
        jobs.append(job)
    cells = {k: cell_metrics(pooled_games[k], pooled_events[k], k[1]) for k in pooled_games}
    d_levels = sorted({k[2] for k in cells})
    flips = sorted({k[3] for k in cells if k[0] == "exact"})
    g_rates = sorted({k[1] for k in cells if k[0] == "G"})
    backends = [b for b in ("soft", "hard", "modeswitch", "modeswitch_guard")
                if any(k[4] == b for k in cells)]
    complete = all(("exact", 0.0, d, f, b) in cells for d in d_levels for f in flips
                   for b in backends) and all(
        ("G", g, d, 0.0, b) in cells for g in g_rates for d in d_levels if d <= 2
        for b in backends)
    verdicts = score(cells, d_levels, flips, g_rates, backends) if complete else {}
    out = {"schema": "twenty_questions_stress_st2_analysis_v1", "run_dir": run_dir,
           "n_jobs": len(jobs), "complete_grid": complete,
           "cells": [{"arm": k[0], "g": k[1], "d": k[2], "flip": k[3], "backend": k[4],
                      **v} for k, v in sorted(cells.items())],
           "verdicts": verdicts}
    if make_figures and complete:
        out["figures"] = [os.path.basename(p) for p in
                          figures(cells, d_levels, flips, g_rates, backends, run_dir)]
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
    for k, v in sorted(out["verdicts"].items()):
        print("  {:28s} {}".format(k, v["verdict"]))
    print("wrote {}".format(os.path.join(args.run_dir, "analysis.json")))


if __name__ == "__main__":
    main()
