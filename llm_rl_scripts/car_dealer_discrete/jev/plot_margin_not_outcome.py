"""Figure: "margin, not outcome" — what the forward model's prior unlocks vs what learning adds.

Panel A (outcome): agreement rate by arm on the same three type pairs (easy regime)
across the ladder baseline -> salience -> placebo -> prior-only -> aid -> aid+salience.
Panel B (margin): Δ = bayes − prior-only with 95% paired type-pair-clustered bootstrap
CIs for the utility-scale metrics, across the three independent runs (run 1, replication,
hard regime). Every number is read from the frozen analysis JSON; nothing is typed in.
The plotted values are also written to a sidecar JSON next to the image.

Design follows the dataviz method: single hue for the one-series bars; three categorical
slots in fixed order for the three runs (validated adjacent + all-pairs, light surface);
distinct markers as secondary encoding; a value column as relief for the low-contrast
third slot; thin marks, solid hairline grid, text in ink tokens only.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.plot_margin_not_outcome
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "llm_rl_scripts", "car_dealer_discrete", "figures")
STEM = "car_dealer_margin_not_outcome"

# --- palette (reference instance; validated) ---------------------------------
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3, fixed order
MARKERS = ["o", "s", "D"]                           # secondary encoding
BAR_HUE = SERIES[0]                                 # one series -> slot 1

# --- data (frozen JSON only) ----------------------------------------------------
def J(p): return json.load(open(os.path.join(ROOT, p)))
CTRL = J("results/jj_aid_20261025/analysis_controls.json")
ARMS = [("baseline", "baseline"), ("salience", "salience"), ("placebo", "placebo (0.50)"),
        ("prior_only", "prior-only"), ("aid", "aid (bayes)"), ("aid_salience", "aid + salience")]
RUNS = [
    ("Run 1 (easy)", "results/jj_quality_20261025/analysis_quality.json", "results/jj_quality_20261025/exploratory_seller_asymmetry.json"),
    ("Replication (easy)", "results/jj_quality_rep1_20261025/analysis_quality.json", "results/jj_quality_rep1_20261025/confirmatory_responder.json"),
    ("Hard regime (feasible)", "results/jj_hard_20261025/analysis_quality_feasible.json", "results/jj_hard_20261025/responder_feasible.json"),
]
ROWS = [("Nash Q_N (primary)", "q", "nash_ratio"), ("min(G_B, G_S)", "q", "min_gain"),
        ("G_responder", "x", "G_responder (either role)"), ("G_starter", "x", "G_starter (either role)"),
        ("agreement", "q", "agreement")]

def collect():
    a = [{"arm": lab, "agreement": CTRL[k]["outcome_by_regime"]["easy"]["agreement_rate"],
          "n": CTRL[k]["outcome_by_regime"]["easy"]["episodes"]} for k, lab in ARMS]
    b = []
    for name, qpath, xpath in RUNS:
        Q, X = J(qpath), J(xpath)["lens1_starting_side"]
        for label, src, key in ROWS:
            v = Q["metrics"][key] if src == "q" else X[key]
            b.append({"run": name, "metric": label, "delta": v["delta"], "ci95": list(v["ci95_delta"]),
                      "pairs": Q["pairs_compared"]})
        b.append({"run": name, "metric": "proposals to agreement (caption only)", "delta": Q["metrics"]["proposals"]["delta"],
                  "ci95": list(Q["metrics"]["proposals"]["ci95_delta"]), "pairs": Q["pairs_compared"]})
    return a, b

def main() -> int:
    A, B = collect()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": BASELINE,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(14.5, 6.3), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.46, left=0.05, right=0.775, top=0.825, bottom=0.265)

    # ---------------- Panel A: outcome ----------------
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURFACE)
    xs = list(range(len(A))); w = 0.42; r = 0.035
    for i, d in enumerate(A):
        h = d["agreement"]
        if h > 0:
            ax.add_patch(FancyBboxPatch((i - w / 2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=0, fc=BAR_HUE, mutation_aspect=1))
            ax.add_patch(Rectangle((i - w / 2, 0), w, min(r, h), lw=0, fc=BAR_HUE))  # square at the baseline
        ax.text(i, h + 0.03, f"{h:.2f}", ha="center", va="bottom", color=INK2, fontsize=9.5)
    ax.set_xticks(xs); ax.set_xticklabels([d["arm"] for d in A], rotation=22, ha="right", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_xlim(-0.6, len(A) - 0.4)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("agreement rate", color=INK2)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE); ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    # brackets: what the arms share
    for (lo, hi, label) in [(0, 2, "no forward model"), (3, 5, "forward model present")]:
        y = 1.09; ax.plot([lo - 0.3, hi + 0.3], [y, y], color=BASELINE, lw=1)
        ax.text((lo + hi) / 2, y + 0.015, label, ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_title("A · Outcome: agreement rate by arm", loc="left", color=INK, fontsize=11.5, pad=22)
    ax.text(0, 1.0, f"same 3 type pairs · easy regime (r = 0.40) · n = {A[0]['n']} episodes per arm",
            transform=ax.transAxes, va="bottom", color=MUTED, fontsize=8.5)

    # ---------------- Panel B: margin ----------------
    bx = fig.add_subplot(gs[1]); bx.set_facecolor(SURFACE)
    plotted = [b for b in B if not b["metric"].startswith("proposals")]
    metrics = [m for m, _, _ in ROWS]; run_names = [n for n, _, _ in RUNS]
    offs = [0.24, 0.0, -0.24]
    ymap = {m: len(metrics) - 1 - i for i, m in enumerate(metrics)}
    bx.axvline(0, color=INK2, lw=1, zorder=1)
    for ri, run in enumerate(run_names):
        for b in plotted:
            if b["run"] != run: continue
            y = ymap[b["metric"]] + offs[ri]; lo, hi = b["ci95"]
            bx.plot([lo, hi], [y, y], color=SERIES[ri], lw=2, solid_capstyle="round", zorder=2)
            bx.plot([b["delta"]], [y], marker=MARKERS[ri], ms=8, mfc=SERIES[ri], mec=SURFACE, mew=2, ls="none", zorder=3)
    bx.set_yticks([ymap[m] for m in metrics]); bx.set_yticklabels(metrics, color=INK2, fontsize=9.5)
    for lbl in bx.get_yticklabels():
        if "primary" in lbl.get_text(): lbl.set_color(INK); lbl.set_fontweight("semibold")
    bx.set_ylim(-0.6, len(metrics) - 0.4)
    bx.set_xlim(-0.17, 0.34)
    bx.set_xlabel("Δ  (bayes − prior-only), with 95% paired type-pair-clustered bootstrap CI", color=INK2)
    bx.grid(axis="x", color=GRID, lw=1); bx.set_axisbelow(True)
    for s in ("top", "right", "left"): bx.spines[s].set_visible(False)
    bx.spines["bottom"].set_color(BASELINE); bx.tick_params(length=0)
    # value column (relief for the low-contrast third slot; the figure's own table view)
    tr = bx.get_yaxis_transform()  # x in axes fraction, y in data
    for b in plotted:
        ri = run_names.index(b["run"]); y = ymap[b["metric"]] + offs[ri]
        bx.text(1.03, y, f"{b['delta']:+.3f}  [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]", transform=tr, va="center", ha="left",
                color=INK2, fontsize=8.4, clip_on=False, family="monospace")
    bx.text(1.03, len(metrics) - 0.42, "Δ  [95% CI]", transform=tr, va="bottom", ha="left",
            color=MUTED, fontsize=8.3, clip_on=False)
    handles = [plt.Line2D([], [], marker=MARKERS[i], ms=8, mfc=SERIES[i], mec=SURFACE, mew=2, color=SERIES[i], lw=2, label=run_names[i])
               for i in range(3)]
    bx.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.6, labelcolor=INK2, handlelength=1.2, handletextpad=0.5, borderaxespad=0.2)
    bx.set_title("B · Margin: what learning adds over the prior", loc="left", color=INK, fontsize=11.5, pad=22)
    bx.text(0, 1.0, "36 type pairs × both starts × K = 8 per run (30 feasible pairs in the hard regime)",
            transform=bx.transAxes, va="bottom", color=MUTED, fontsize=8.5)

    fig.suptitle("Car Dealer × Jev — the forward model's prior unlocks the outcome; learning buys the margin",
                 x=0.055, ha="left", color=INK, fontsize=13, fontweight="semibold", y=0.952)
    props = [b for b in B if b["metric"].startswith("proposals")]
    # Provenance caption: every panel names its ANALYZER and RESULT DIR, same as the numbers do,
    # so the figure is regenerable end to end (logs -> analyzer -> frozen JSON -> this script).
    lines = [
        "A: analyze_aid -> results/jj_aid_20261025/analysis_controls.json   (arms from results/jj_salience_20261025 and results/jj_aid_20261025).",
        "B: analyze_quality (Nash, minG, agreement) + explore_seller_asymmetry (G_responder, G_starter) -> analysis_quality*.json / *responder*.json",
        "    in results/jj_quality_20261025, results/jj_quality_rep1_20261025, results/jj_hard_20261025 (--feasible-only); seeds 0 / 1 / 3, B = 5000.",
        "Figure: jev/plot_margin_not_outcome.py; plotted values in figures/" + STEM + ".values.json.",
        "Not plotted (different units): proposals-to-agreement \u0394, a cost of learning \u2014 "
        + "; ".join(f"{p['run'].split(' (')[0]} {p['delta']:+.2f}" for p in props) + ".",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.05, 0.118 - 0.022 * i, ln, color=MUTED, fontsize=7.8)

    png, svg = os.path.join(OUT_DIR, STEM + ".png"), os.path.join(OUT_DIR, STEM + ".svg")
    fig.savefig(png, dpi=200, facecolor=SURFACE, pad_inches=0.12); fig.savefig(svg, facecolor=SURFACE, pad_inches=0.12)
    json.dump({"panel_A_agreement_by_arm": A, "panel_B_deltas": B, "palette": SERIES, "sources": [
        "results/jj_aid_20261025/analysis_controls.json"] + [p for _, q, x in RUNS for p in (q, x)]},
        open(os.path.join(OUT_DIR, STEM + ".values.json"), "w"), indent=2)
    print(f"wrote {png}\nwrote {svg}\nwrote {os.path.join(OUT_DIR, STEM + '.values.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
