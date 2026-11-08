"""Analyse the retraction grid: print the backend x noise x margin tables and draw
the trade-off figure (in-pool recovery against novelty-detection cost).

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_retraction \
        --dir llm_rl_scripts/twenty_questions/jev/results/openworld_retraction_<stamp>

Every number regenerates from the run's summary.json; the figure caption names this
analyzer and the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional


def load(path: str) -> dict:
    with open(os.path.join(path, "summary.json")) as f:
        return json.load(f)


def cell(cells: List[dict], backend: str, margin, noise: float) -> Optional[dict]:
    for c in cells:
        if (c["backend"] == backend and c.get("margin") == margin
                and abs(c["oracle_noise"] - noise) < 1e-9):
            return c["agg"]
    return None


def _m(agg, key):
    return agg[key]["mean"]


def report(path: str) -> None:
    data = load(path)
    cells = data["cells"]
    levels = data["config"]["levels"]
    margins = data["config"]["margins"]
    labels = [("soft", None), ("hard", None)] + [("retract", m) for m in margins]

    print("=" * 70)
    print("Retraction grid:", os.path.basename(path),
          " PY sigma={:.3f}".format(data["pitman_yor_fit"]["sigma"]))
    print("-" * 70)
    header = "backend     " + "".join("  ipfail@{:.0f}%".format(l * 100) for l in levels)
    print(header + "    AUC@hi   wrong@0   retr@hi")
    hi = levels[-1]
    for bk, m in labels:
        name = bk if m is None else "{}@{:g}".format(bk, m)
        ipf = ["{:.3f}".format(_m(cell(cells, bk, m, l), "in_pool_fail_rate"))
               for l in levels]
        auc_hi = _m(cell(cells, bk, m, hi), "auc")
        wrong0 = _m(cell(cells, bk, m, 0.0), "wrong")
        retr_hi = _m(cell(cells, bk, m, hi), "mean_retractions")
        print("{:11s}".format(name) + "".join("   {:>8s}".format(v) for v in ipf)
              + "    {:.3f}   {:6.0f}    {:5.2f}".format(auc_hi, wrong0, retr_hi))
    print("=" * 70)
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
    levels = data["config"]["levels"]
    margins = data["config"]["margins"]
    hi = levels[-1]
    xs = [l * 100 for l in levels]
    run = os.path.basename(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: in-pool failure vs noise, one line per backend/margin.
    def series(bk, m):
        return [_m(cell(cells, bk, m, l), "in_pool_fail_rate") for l in levels]
    ax1.plot(xs, series("soft", None), "-o", color="#2d8a66", label="soft")
    ax1.plot(xs, series("hard", None), "-o", color="#c24a30", label="hard")
    styles = ["--", ":", "-"]
    for i, m in enumerate(margins):
        ax1.plot(xs, series("retract", m), styles[i % 3], marker="s", color="#3a46a6",
                 label="retract@{:g}".format(m), markersize=4)
    ax1.set_xlabel("answers flipped (%)")
    ax1.set_ylabel("in-pool failure rate")
    ax1.set_title("Hard breaks; retraction recovers in-pool loss")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    # Right: the trade-off at the high-noise level -- recovery vs detector AUC, over margin.
    ipf_hi = [_m(cell(cells, "retract", m, hi), "in_pool_fail_rate") for m in margins]
    auc_hi = [_m(cell(cells, "retract", m, hi), "auc") for m in margins]
    ax2.plot(margins, ipf_hi, "-s", color="#3a46a6", label="in-pool failure @{:.0f}%".format(hi * 100))
    ax2.set_xlabel("retract revival margin (bits)")
    ax2.set_ylabel("in-pool failure rate", color="#3a46a6")
    ax2.tick_params(axis="y", labelcolor="#3a46a6")
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(margins)
    ax2b = ax2.twinx()
    ax2b.plot(margins, auc_hi, "-o", color="#c24a30", label="detector AUC @{:.0f}%".format(hi * 100))
    ax2b.axhline(0.5, color="#999", ls=":", lw=1)
    ax2b.set_ylabel("detector AUC (0.5 = chance)", color="#c24a30")
    ax2b.tick_params(axis="y", labelcolor="#c24a30")
    ax2b.set_ylim(0.5, 1.02)
    ax2.set_title("The trade-off: recovery up, detection down")
    lines = ax2.get_lines() + ax2b.get_lines()[:1]
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

    fig.text(0.5, 0.005, "analyze_retraction.py  |  {}".format(run),
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(path, "fig_retraction_tradeoff.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[figure] wrote fig_retraction_tradeoff.png -> {}".format(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    report(args.dir)


if __name__ == "__main__":
    main()
