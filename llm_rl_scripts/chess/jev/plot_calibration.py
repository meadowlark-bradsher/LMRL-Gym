"""Jev confidence (Noul) calibration curve for the Chess fixed-position study.

Turns the two-number tertile claim (low-conf vs high-conf) into a curve: bin
Jev's per-move Choice confidence into quantile bins and plot, against the bin's
mean confidence,

  * top panel  -- best-move match@1 and match@3 rate (higher = better), and
  * bottom panel -- median centipawn loss vs the best legal move, non-mate
    positions only (lower = better).

Two measures of different scale => two stacked panels sharing the x-axis, never a
dual-axis plot. Rates carry Wilson 95% intervals; the median loss carries its
IQR. Colours are Okabe-Ito (colourblind-safe by construction).

    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.chess.jev.plot_calibration \
      --scores results/primary_20261025/position_scores.jsonl \
      --out-png results/primary_20261025/calibration.png \
      --out-csv results/primary_20261025/calibration_bins.csv
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colourblind-safe palette.
BLUE = "#0072B2"
GREEN = "#009E73"
VERMILLION = "#D55E00"
INK = "#222222"
MUTED = "#888888"


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def quantile_bins(values: List[float], nbins: int) -> List[float]:
    """Edges that split sorted values into nbins near-equal groups."""
    xs = sorted(values)
    edges = [xs[0]]
    for i in range(1, nbins):
        edges.append(xs[min(len(xs) - 1, round(i * len(xs) / nbins))])
    edges.append(xs[-1] + 1e-9)
    # De-duplicate collapsed edges (ties) while preserving order.
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--arm", default="jev")
    p.add_argument("--nbins", type=int, default=5)
    p.add_argument("--mark-split", type=float, default=0.20,
                   help="draw a reference line at this Noul and report the two-group "
                        "split there (the split the §4.5 text headline uses)")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.scores) if l.strip()]
    arm = [r for r in rows if r["arm"] == args.arm and r.get("confidence") is not None
           and r.get("is_legal")]
    confs = [r["confidence"] for r in arm]
    edges = quantile_bins(confs, args.nbins)

    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        grp = [r for r in arm if lo <= r["confidence"] < hi]
        if not grp:
            continue
        n = len(grp)
        k1 = sum(1 for r in grp if r.get("match_best"))
        k3 = sum(1 for r in grp if r.get("match_top3"))
        p1, lo1, hi1 = wilson(k1, n)
        p3, lo3, hi3 = wilson(k3, n)
        cps = [r["cp_loss"] for r in grp
               if r.get("cp_loss") is not None and not r.get("best_is_mate")]
        med = statistics.median(cps) if cps else float("nan")
        q1 = statistics.quantiles(cps, n=4)[0] if len(cps) >= 2 else med
        q3 = statistics.quantiles(cps, n=4)[2] if len(cps) >= 2 else med
        bins.append({
            "n": n, "conf_mean": statistics.fmean(r["confidence"] for r in grp),
            "conf_lo": lo, "conf_hi": hi,
            "match1": p1, "match1_lo": lo1, "match1_hi": hi1,
            "match3": p3, "match3_lo": lo3, "match3_hi": hi3,
            "med_cp": med, "cp_q1": q1, "cp_q3": q3, "n_cp": len(cps),
        })

    x = [b["conf_mean"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

    # --- Panel A: match rates (higher = better) ---
    ax1.errorbar(x, [b["match1"] for b in bins],
                 yerr=[[b["match1"] - b["match1_lo"] for b in bins],
                       [b["match1_hi"] - b["match1"] for b in bins]],
                 color=BLUE, lw=2, marker="o", ms=7, capsize=3, label="match@1 (best move)")
    ax1.errorbar(x, [b["match3"] for b in bins],
                 yerr=[[b["match3"] - b["match3_lo"] for b in bins],
                       [b["match3_hi"] - b["match3"] for b in bins]],
                 color=GREEN, lw=2, marker="s", ms=7, capsize=3, label="match@3 (top-3)")
    ax1.axhline(0.03, color=MUTED, lw=1, ls=":", label="random match@1 (3%)")
    ax1.set_ylabel("best-move match rate", color=INK)
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax1.grid(True, axis="y", color="#eeeeee", lw=0.8)
    ax1.set_title("Jev Choice confidence (Noul) is informative about move quality\n"
                  "(signal concentrated in the top quintile) — Chess, 200 fixed positions",
                  fontsize=10.5, color=INK)

    # --- Panel B: median centipawn loss (lower = better) ---
    ax2.errorbar(x, [b["med_cp"] for b in bins],
                 yerr=[[max(0, b["med_cp"] - b["cp_q1"]) for b in bins],
                       [max(0, b["cp_q3"] - b["med_cp"]) for b in bins]],
                 color=VERMILLION, lw=2, marker="D", ms=7, capsize=3)
    ax2.set_ylabel("median centipawn loss\n(non-mate; lower = better)", color=INK)
    ax2.set_xlabel("Jev Choice confidence (Noul), bin mean", color=INK)
    ax2.grid(True, axis="y", color="#eeeeee", lw=0.8)
    ax2.invert_yaxis()  # better (lower loss) reads upward

    # Mark the split the §4.5 text headline uses, so text and figure reconcile:
    # the top tertile is Noul >= ~0.21, i.e. the same moves as Noul > 0.2.
    if args.mark_split is not None:
        for ax in (ax1, ax2):
            ax.axvline(args.mark_split, color=MUTED, lw=1, ls="--")
        ax1.annotate("text split\nNoul>{:.1f} (top third)".format(args.mark_split),
                     xy=(args.mark_split, 0.55), xycoords=("data", "axes fraction"),
                     xytext=(4, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=8, color=MUTED)

    # Per-bin n annotations along the top strip (clear of the baseline line).
    for b in bins:
        ax1.annotate("n={}".format(b["n"]), xy=(b["conf_mean"], 0.97),
                     xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=8, color=MUTED)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Provenance baked into the figure: the source scores and the script that
    # drew it, so the plot is as regenerable as the cited numbers.
    import os
    prov = "source: {}  |  {} match@1/median-cpL by Noul quantile bin  |  regen: python -m llm_rl_scripts.chess.jev.plot_calibration".format(
        os.path.relpath(args.scores), args.arm)
    fig.text(0.5, 0.005, prov, ha="center", va="bottom", fontsize=6.5, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote {}".format(args.out_png))

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bins[0].keys()))
            w.writeheader()
            for b in bins:
                w.writerow(b)
        print("wrote {}".format(args.out_csv))

    # Echo the table so the numbers are visible/citable.
    print("\nbin  conf_range        n   match@1  match@3  median_cpL")
    for b in bins:
        print("     {:.2f}-{:.2f}      {:3d}   {:5.1f}%  {:5.1f}%  {:7.1f}".format(
            b["conf_lo"], b["conf_hi"], b["n"], 100 * b["match1"], 100 * b["match3"], b["med_cp"]))

    # The §4.5 headline split (text uses this; the figure marks it). Emitting it
    # here means the headline is regenerable from the same script as the figure.
    def grp_stats(grp):
        m1 = statistics.fmean(1.0 if r.get("match_best") else 0.0 for r in grp)
        cp = [r["cp_loss"] for r in grp
              if r.get("cp_loss") is not None and not r.get("best_is_mate")]
        return len(grp), 100 * m1, (statistics.median(cp) if cp else float("nan"))
    # The §4.5 text headline is a TERTILE split (bottom third vs top third,
    # middle excluded). Emit it verbatim so text and figure share one source,
    # and note that the top third coincides with Noul > ~0.2 (the marked line).
    srt = sorted(arm, key=lambda r: r["confidence"])
    t = len(srt) // 3
    low3, high3 = srt[:t], srt[-t:]
    nl, m1l, cpl = grp_stats(low3)
    nh, m1h, cph = grp_stats(high3)
    print("\n§4.5 headline = TERTILE split (bottom third vs top third):")
    print("  low  third: Noul in [{:.3f},{:.3f}]  n={:3d}  match@1={:.1f}%  medianCpL={:.0f}".format(
        low3[0]["confidence"], low3[-1]["confidence"], nl, m1l, cpl))
    print("  high third: Noul in [{:.3f},{:.3f}]  n={:3d}  match@1={:.1f}%  medianCpL={:.0f}  ({:.2f}x match@1)".format(
        high3[0]["confidence"], high3[-1]["confidence"], nh, m1h, cph, m1h / m1l if m1l else float("nan")))
    print("  (top third starts at Noul={:.2f}, ~= the marked Noul>{:.1f} line on the figure)".format(
        high3[0]["confidence"], args.mark_split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
