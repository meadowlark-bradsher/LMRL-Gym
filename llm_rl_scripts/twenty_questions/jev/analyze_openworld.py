"""Analyse an open-world accumulation run: discovery curve, detector ROC, expert
load over the stream, and autonomous coverage by frequency.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_openworld \
        --dir llm_rl_scripts/twenty_questions/jev/results/openworld_<stamp>

Prints the cited numbers and writes four figures into the run directory. Every
figure caption names this analyzer and the run directory (provenance), so the
numbers and the plots regenerate together.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .openworld import py_p_new


def load_run(path: str) -> Tuple[dict, List[dict]]:
    with open(os.path.join(path, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(path, "records.json")) as f:
        records = json.load(f)
    return summary, records


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def discovery_curve(records: List[dict]) -> Tuple[List[int], List[int]]:
    """n (games played) vs K_n (distinct species seen)."""
    seen: set = set()
    ns, ks = [], []
    for i, r in enumerate(records, start=1):
        seen.add(r["true_name"])
        ns.append(i)
        ks.append(len(seen))
    return ns, ks


def good_turing_missing_mass(records: List[dict]) -> Tuple[List[int], List[float]]:
    """Good-Turing missing-mass estimate (singletons / n) at each step."""
    counts: Dict[str, int] = {}
    ns, mm = [], []
    for i, r in enumerate(records, start=1):
        counts[r["true_name"]] = counts.get(r["true_name"], 0) + 1
        singletons = sum(1 for c in counts.values() if c == 1)
        ns.append(i)
        mm.append(singletons / i)
    return ns, mm


def model_p_new(records: List[dict], sigma: float, theta: float
                ) -> Tuple[List[int], List[float]]:
    """PY p_new(t) using the live distinct-count K_t at each step."""
    seen: set = set()
    ns, pn = [], []
    for i, r in enumerate(records):
        n_before, k_before = i, len(seen)
        ns.append(i + 1)
        pn.append(py_p_new(k_before, n_before, sigma, theta) if n_before > 0 else 1.0)
        seen.add(r["true_name"])
    return ns, pn


def auc(scores: List[float], labels: List[int]) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg), ties at 0.5."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    # rank-based for O(n log n)
    paired = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank = 0.0
    i = 0
    rank_sum_pos = 0.0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average rank for ties
        for k in range(i, j):
            if paired[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_points(scores: List[float], labels: List[int]) -> Tuple[List[float], List[float]]:
    """ROC as (fpr, tpr) sweeping the threshold over observed scores."""
    thresholds = sorted(set(scores), reverse=True)
    thresholds = [thresholds[0] + 1e-9] + thresholds
    P = sum(labels)
    N = len(labels) - P
    fprs, tprs = [], []
    for t in thresholds:
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 0)
        tprs.append(tp / P if P else 0.0)
        fprs.append(fp / N if N else 0.0)
    return fprs, tprs


def windowed_rate(records: List[dict], key, window: int = 100) -> Tuple[List[float], List[float]]:
    """Centre of each window (game index) vs mean of key(r) over the window."""
    xs, ys = [], []
    for start in range(0, len(records), window):
        chunk = records[start:start + window]
        if not chunk:
            continue
        xs.append(start + len(chunk) / 2.0)
        ys.append(sum(1.0 if key(r) else 0.0 for r in chunk) / len(chunk))
    return xs, ys


def resolution_by_rank(records: List[dict], n_bins: int = 8
                       ) -> Tuple[List[str], List[float], List[int]]:
    """Autonomous-resolution rate (outcome == won) binned by frequency rank."""
    ranks = [r["rank"] for r in records if r["rank"] > 0]
    if not ranks:
        return [], [], []
    rmax = max(ranks)
    edges = [int(round(rmax * i / n_bins)) for i in range(n_bins + 1)]
    labels, rates, counts = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        chunk = [r for r in records if lo < r["rank"] <= hi]
        if not chunk:
            continue
        won = sum(1 for r in chunk if r["outcome"] == "won")
        labels.append("{}-{}".format(lo + 1, hi))
        rates.append(won / len(chunk))
        counts.append(len(chunk))
    return labels, rates, counts


# --------------------------------------------------------------------------- #
# Report + figures                                                             #
# --------------------------------------------------------------------------- #

def analyze(path: str) -> None:
    summary, records = load_run(path)
    sigma = summary["pitman_yor_fit"]["sigma"]
    theta = summary["pitman_yor_fit"]["theta"]

    out_labels = [0 if r["in_pool"] else 1 for r in records]
    scores = [r["p_unknown_max"] for r in records]
    a = auc(scores, out_labels)

    conf = summary["config"]["confidence"]
    tp = sum(1 for r in records if (not r["in_pool"]) and r["p_unknown_max"] >= conf)
    fn = sum(1 for r in records if (not r["in_pool"]) and r["p_unknown_max"] < conf)
    fp = sum(1 for r in records if r["in_pool"] and r["p_unknown_max"] >= conf)
    tn = sum(1 for r in records if r["in_pool"] and r["p_unknown_max"] < conf)
    n_out = tp + fn
    tpr = tp / n_out if n_out else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    fires = [r["fire_turn"] for r in records
             if (not r["in_pool"]) and r["fire_turn"] is not None]
    med_fire = sorted(fires)[len(fires) // 2] if fires else None

    ns, ks = discovery_curve(records)

    print("=" * 66)
    print("Open-world accumulation:", os.path.basename(path))
    print("  games {}  distinct seen {}/{}  PY sigma={:.3f} theta={:.3f}".format(
        summary["totals"]["games"], ks[-1],
        summary["config"]["n_true_objects"], sigma, theta))
    print("-" * 66)
    print("Detector (label = out-of-pool, score = max_t p_unknown):")
    print("  AUC = {:.3f}".format(a))
    print("  at confidence gate {:.2f}:  TPR={:.3f} ({}/{})  FPR={:.3f} ({}/{})".format(
        conf, tpr, tp, n_out, fpr, fp, fp + tn))
    print("  novelty misses (out-of-pool, never fired): {}".format(fn))
    print("  earliness: median fire turn = {}  (n={})".format(med_fire, len(fires)))
    print("-" * 66)
    print("Expert load:")
    print("  total expert calls      : {}".format(summary["totals"]["expert_calls"]))
    print("  false-positive calls    : {}".format(
        summary["totals"]["false_positive_expert_calls"]))
    print("  autonomous wins         : {}".format(summary["totals"]["won"]))
    print("  wrong (missed novelty)  : {}".format(summary["totals"]["wrong"]))
    # coverage: share of the stream served without an expert, late half vs early.
    half = len(records) // 2
    early = sum(1 for r in records[:half] if not r["expert_called"]) / half
    late = sum(1 for r in records[half:] if not r["expert_called"]) / (len(records) - half)
    print("  autonomous share  first half {:.3f}  ->  second half {:.3f}".format(
        early, late))
    print("-" * 66)
    labels, rates, counts = resolution_by_rank(records)
    print("Autonomous-resolution rate by frequency rank (common -> rare):")
    for lab, rate, c in zip(labels, rates, counts):
        print("  rank {:>10}:  won {:.3f}   (n={})".format(lab, rate, c))
    print("=" * 66)

    _figures(path, summary, records, sigma, theta, a)


def _figures(path, summary, records, sigma, theta, a) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("[figures] matplotlib unavailable ({}); skipping plots".format(e))
        return

    run = os.path.basename(path)
    cap = "analyze_openworld.py  |  {}".format(run)

    # Fig 1: discovery curve + missing mass.
    ns, ks = discovery_curve(records)
    gt_n, gt_mm = good_turing_missing_mass(records)
    mp_n, mp_pn = model_p_new(records, sigma, theta)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ns, ks, color="#1f77b4")
    ax1.set_xlabel("games played  n"); ax1.set_ylabel("distinct species  K_n")
    ax1.set_title("Discovery curve (PY sigma={:.2f})".format(sigma))
    ax2.plot(gt_n, gt_mm, label="Good-Turing missing mass (singletons/n)",
             color="#ff7f0e")
    ax2.plot(mp_n, mp_pn, label="PY p_new(t)", color="#2ca02c", ls="--")
    ax2.set_xlabel("games played  n"); ax2.set_ylabel("P(next is new)")
    ax2.set_title("Missing mass: Good-Turing vs PY"); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)
    fig.text(0.5, 0.005, cap, ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(path, "fig1_discovery.png"), dpi=130)
    plt.close(fig)

    # Fig 2: detector ROC + score distributions.
    out_labels = [0 if r["in_pool"] else 1 for r in records]
    scores = [r["p_unknown_max"] for r in records]
    fpr, tpr = roc_points(scores, out_labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(fpr, tpr, color="#1f77b4"); ax1.plot([0, 1], [0, 1], ls=":", color="#999")
    ax1.set_xlabel("false-positive rate"); ax1.set_ylabel("true-positive rate")
    ax1.set_title("Detector ROC (AUC={:.3f})".format(a))
    s_in = [s for s, y in zip(scores, out_labels) if y == 0]
    s_out = [s for s, y in zip(scores, out_labels) if y == 1]
    ax2.hist([s_in, s_out], bins=20, stacked=False,
             label=["in-pool", "out-of-pool"], color=["#2ca02c", "#d62728"])
    ax2.set_xlabel("max_t p_unknown"); ax2.set_ylabel("games")
    ax2.set_title("Catch-all mass, in- vs out-of-pool"); ax2.legend(fontsize=8)
    ax2.set_yscale("log")
    fig.text(0.5, 0.005, cap, ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(path, "fig2_detector_roc.png"), dpi=130)
    plt.close(fig)

    # Fig 3: expert load over the stream.
    xe, ye = windowed_rate(records, lambda r: r["expert_called"])
    xn, yn = windowed_rate(records, lambda r: r["expert_called"] and not r["in_pool"])
    xf, yf = windowed_rate(records, lambda r: r["false_positive"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xe, ye, label="all expert calls", color="#1f77b4")
    ax.plot(xn, yn, label="necessary (true novelty)", color="#2ca02c", ls="--")
    ax.plot(xf, yf, label="false-positive", color="#d62728", ls=":")
    ax.set_xlabel("game index"); ax.set_ylabel("rate (per 100-game window)")
    ax.set_title("Expert load decays as the pool fills"); ax.legend(fontsize=8)
    fig.text(0.5, 0.005, cap, ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(path, "fig3_expert_load.png"), dpi=130)
    plt.close(fig)

    # Fig 4: autonomous coverage by frequency rank.
    labels, rates, counts = resolution_by_rank(records)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(labels)), rates, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("frequency rank bin  (1 = most common  ->  rare)")
    ax.set_ylabel("autonomous-resolution rate (won)")
    ax.set_title("Coverage is size-biased: the common head is served without an expert")
    ax.set_ylim(0, 1)
    fig.text(0.5, 0.005, cap, ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(path, "fig4_coverage_by_rank.png"), dpi=130)
    plt.close(fig)

    print("[figures] wrote fig1_discovery, fig2_detector_roc, fig3_expert_load, "
          "fig4_coverage_by_rank -> {}".format(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    analyze(args.dir)


if __name__ == "__main__":
    main()
