"""Analyse the two-oracle grid (ENV-704-B): print the per-regime policy tables and draw
the accuracy-vs-human-spend Pareto frontier — which policy buys the most accuracy per
expensive human query.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_twooracle \
        --dir llm_rl_scripts/twenty_questions/jev/results/twooracle_<stamp>

Every number regenerates from the run's summary.json; the figure caption names this
analyzer and the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional


def load(path: str) -> dict:
    with open(os.path.join(path, "summary.json")) as f:
        return json.load(f)


def regime_tags(data: dict) -> List[str]:
    out = []
    for c in data["cells"]:
        if c["regime"] not in out:
            out.append(c["regime"])
    return out


def policy_curve(cells: List[dict], regime: str, policy: str):
    """Return the threshold sweep for one policy in one regime, sorted by human spend:
    list of (human_q_per_game, accuracy)."""
    pts = [(c["metrics"]["human_q_per_game"], c["metrics"]["accuracy"])
           for c in cells if c["regime"] == regime and c["policy"] == policy]
    return sorted(pts)


def _point(cells, regime, policy):
    pts = policy_curve(cells, regime, policy)
    return pts[0] if pts else (0.0, float("nan"))


def report(path: str) -> None:
    data = load(path)
    cells = data["cells"]
    policies = data["config"]["policies"]
    thresh = [p for p in policies if p not in ("cheap_only", "human_all")]
    print("=" * 74)
    print("Two-oracle grid:", os.path.basename(path))
    for regime in regime_tags(data):
        floor = _point(cells, regime, "cheap_only")
        ceil = _point(cells, regime, "human_all")
        print("-" * 74)
        print("regime {}:  cheap_only acc={:.3f} (0 human)   human_all acc={:.3f} "
              "({:.2f} human/g)".format(regime, floor[1], ceil[1], ceil[0]))
        gap = ceil[1] - floor[1]
        print("   VOI room (ceiling - floor) = {:+.3f}".format(gap))
        if gap < 0.02:
            print("   (negligible room — policies converge)")
        # for a matched half-ceiling spend, which policy is best?
        target = ceil[0] * 0.5
        print("   policy      best-acc @ <= {:.2f} human/g   (full sweep)".format(target))
        for policy in thresh:
            curve = policy_curve(cells, regime, policy)
            under = [(h, a) for h, a in curve if h <= target + 1e-9]
            best = max((a for _, a in under), default=float("nan"))
            sweep = " ".join("{:.2f}:{:.3f}".format(h, a) for h, a in curve)
            print("   {:9s}  {:.3f}                     {}".format(policy, best, sweep))
    print("=" * 74)
    _figure(path, data)


def _figure(path: str, data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("[figure] matplotlib unavailable ({}); skipping".format(e))
        return

    cells = data["cells"]
    policies = data["config"]["policies"]
    thresh = [p for p in policies if p not in ("cheap_only", "human_all")]
    regimes = regime_tags(data)
    # the two hard regimes are the interesting ones; show up to two panels.
    show = [r for r in regimes if "T10" in r] or regimes[:2]
    show = show[:2]
    pcol = {"random": "#999999", "entropy": "#b07a16", "margin": "#3a46a6",
            "voi_eig": "#1a9988"}
    mk = {"random": "x", "entropy": "^", "margin": "s", "voi_eig": "o"}

    fig, axes = plt.subplots(1, len(show), figsize=(5.8 * len(show), 4.6), squeeze=False)
    for ax, regime in zip(axes[0], show):
        floor = _point(cells, regime, "cheap_only")
        ceil = _point(cells, regime, "human_all")
        ax.axhline(floor[1], color="#c24a30", ls=":", lw=1.2)
        ax.text(0.02, floor[1], " cheap_only (floor)", fontsize=7, color="#c24a30",
                va="bottom")
        ax.axhline(ceil[1], color="#2d8a66", ls=":", lw=1.2)
        ax.text(0.02, ceil[1], " human_all (ceiling)", fontsize=7, color="#2d8a66",
                va="top")
        for policy in thresh:
            curve = policy_curve(cells, regime, policy)
            xs = [0.0] + [h for h, _ in curve]       # anchor every curve at (0, floor)
            ys = [floor[1]] + [a for _, a in curve]
            ax.plot(xs, ys, "-" + mk.get(policy, "o"), color=pcol.get(policy, "#555"),
                    label=policy, markersize=5, linewidth=1.3)
        ax.scatter([ceil[0]], [ceil[1]], c="#2d8a66", s=60, zorder=5, marker="*")
        ax.set_xlabel("human queries / game  (expensive)")
        ax.set_ylabel("accuracy")
        ax.set_title("Pareto frontier — regime {}".format(regime))
        ax.legend(fontsize=7, loc="lower right")

    fig.text(0.5, 0.005, "analyze_twooracle.py  |  {}".format(os.path.basename(path)),
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(path, "fig_twooracle.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[figure] wrote fig_twooracle.png -> {}".format(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    report(args.dir)


if __name__ == "__main__":
    main()
