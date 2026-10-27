"""Figure for the confidence head-to-head: reliability + quintile lift + selective accuracy.

Reads `raw/<condition>.jsonl` and the frozen grader (same build as analyze.py), scores the
primary event (acceptable = cp_lost<=50), and draws one figure
`figures/confidence_h2h.png`:
  * top row  — reliability diagram per condition (predicted vs observed, vs the diagonal),
  * bottom-left — quintile lift per condition (top-quintile minus bottom-four accept-rate),
  * bottom-right — selective-accuracy curves (accept-rate vs coverage).
Colours are Okabe-Ito (colourblind-safe). Caption carries source + regen command.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_rl_scripts.chess.jev.confidence_h2h import analyze, common

CONDS = ["j_noul", "l_logprob", "l_verbal_gpt", "l_verbal_claude"]
LABELS = {"j_noul": "Noul (jev)", "l_logprob": "gpt-4o logprob",
          "l_verbal_gpt": "gpt-4o verbal", "l_verbal_claude": "claude verbal"}
COLOR = {"j_noul": "#0072B2", "l_logprob": "#D55E00",
         "l_verbal_gpt": "#009E73", "l_verbal_claude": "#E69F00"}
MUTED = "#888888"


def reliability(triples, nbins=10):
    """Equal-width bins: (x=mean p, y=mean event, n) per non-empty bin."""
    buckets = defaultdict(list)
    for _, p, e in triples:
        b = min(nbins - 1, int(p * nbins))
        buckets[b].append((p, e))
    pts = []
    for b in range(nbins):
        grp = buckets.get(b)
        if grp:
            pts.append((statistics.fmean(p for p, _ in grp),
                        statistics.fmean(e for _, e in grp), len(grp)))
    return pts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default=os.path.join(common.HERE, "raw"))
    p.add_argument("--out", default=os.path.join(common.HERE, "figures", "confidence_h2h.png"))
    args = p.parse_args()

    grader = common.load_grader()
    data = {}
    for c in CONDS:
        raw = analyze.load_raw(args.raw_dir, c)
        tr, top = analyze.build(raw, grader, "acceptable")
        data[c] = (tr, top)

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.38, wspace=0.35)

    # Top row: reliability per condition.
    for i, c in enumerate(CONDS):
        ax = fig.add_subplot(gs[0, i])
        ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
        pts = reliability(data[c][0])
        if pts:
            xs = [x for x, _, _ in pts]; ys = [y for _, y, _ in pts]
            ns = [n for _, _, n in pts]
            smax = max(ns)
            ax.plot(xs, ys, color=COLOR[c], lw=1.5, zorder=2)
            ax.scatter(xs, ys, s=[18 + 60 * n / smax for n in ns], color=COLOR[c], zorder=3)
        ax.set_title(LABELS[c], fontsize=10, color="#222")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted p", fontsize=8)
        if i == 0:
            ax.set_ylabel("observed\nacceptable-rate", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Bottom-left: quintile lift bars.
    axl = fig.add_subplot(gs[1, :2])
    lifts = [analyze.quintile_lift([(s, y) for _, s, y in data[c][1]]) or 0.0 for c in CONDS]
    axl.bar(range(len(CONDS)), lifts, color=[COLOR[c] for c in CONDS], width=0.6)
    axl.set_xticks(range(len(CONDS)))
    axl.set_xticklabels([LABELS[c] for c in CONDS], fontsize=8, rotation=12)
    axl.set_ylabel("quintile lift\n(top − bottom-4 accept-rate)", fontsize=9)
    axl.axhline(0, color=MUTED, lw=0.8)
    for i, v in enumerate(lifts):
        axl.annotate("{:+.2f}".format(v), (i, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8, color="#222")
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)

    # Bottom-right: selective-accuracy curves.
    axr = fig.add_subplot(gs[1, 2:])
    taus = [50, 60, 70, 80, 90, 95]
    for c in CONDS:
        tops = [(s, y) for _, s, y in data[c][1]]
        base = statistics.fmean(y for _, y in tops) if tops else 0.0
        covs = [1.0] + [(100 - t) / 100.0 for t in taus]
        accs = [base] + [analyze.selective(tops, t) or 0.0 for t in taus]
        axr.plot(covs, accs, color=COLOR[c], lw=1.8, marker="o", ms=4, label=LABELS[c])
    axr.set_xlabel("coverage (fraction of positions kept, most-confident first)", fontsize=9)
    axr.set_ylabel("acceptable-rate", fontsize=9)
    axr.set_xlim(0, 1.02)
    axr.legend(frameon=False, fontsize=8, loc="lower left")
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)

    fig.suptitle("Chess confidence head-to-head: Noul vs LLM menu-probabilities "
                 "(primary event: cp_lost ≤ 50)", fontsize=12, y=0.99)
    fig.text(0.5, 0.005,
             "source: confidence_h2h/raw/*.jsonl + frozen eval_cache.json  |  "
             "regen: python -m llm_rl_scripts.chess.jev.confidence_h2h.plot",
             ha="center", fontsize=6.5, color=MUTED)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
