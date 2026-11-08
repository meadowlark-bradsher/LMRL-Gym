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


def cell(cells: List[dict], backend: str, knob, noise: float) -> Optional[dict]:
    """``knob`` is the backend's swept value — the margin for ``retract``, the
    tolerance for ``tombstone``, ``None`` for ``soft``/``hard``."""
    for c in cells:
        kv = c.get("margin") if c.get("margin") is not None else c.get("tolerance")
        if (c["backend"] == backend and kv == knob
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
    tolerances = data["config"].get("tolerances", [])
    labels = ([("soft", None), ("hard", None)]
              + [("retract", m) for m in margins]
              + [("tombstone", t) for t in tolerances]
              + [("modeswitch", t) for t in tolerances])

    print("=" * 70)
    print("Retraction grid:", os.path.basename(path),
          " PY sigma={:.3f}".format(data["pitman_yor_fit"]["sigma"]))
    print("-" * 70)
    header = "backend     " + "".join("  ipfail@{:.0f}%".format(l * 100) for l in levels)
    print(header + "    AUC@hi   wrong@0   retr@hi    q@hi")
    hi = levels[-1]
    for bk, m in labels:
        name = bk if m is None else "{}@{:g}".format(bk, m)
        ipf = ["{:.3f}".format(_m(cell(cells, bk, m, l), "in_pool_fail_rate"))
               for l in levels]
        auc_hi = _m(cell(cells, bk, m, hi), "auc")
        wrong0 = _m(cell(cells, bk, m, 0.0), "wrong")
        retr_hi = _m(cell(cells, bk, m, hi), "mean_retractions")
        q_hi = _m(cell(cells, bk, m, hi), "mean_questions")
        print("{:11s}".format(name) + "".join("   {:>8s}".format(v) for v in ipf)
              + "    {:.3f}   {:6.0f}    {:5.2f}   {:5.2f}".format(
                  auc_hi, wrong0, retr_hi, q_hi))
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
    tolerances = data["config"].get("tolerances", [])
    hi = levels[-1]
    xs = [l * 100 for l in levels]
    run = os.path.basename(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    def series(bk, knob):
        return [_m(cell(cells, bk, knob, l), "in_pool_fail_rate") for l in levels]

    # Left: in-pool failure vs noise, one line per backend config.
    ax1.plot(xs, series("soft", None), "-o", color="#2d8a66", label="soft")
    ax1.plot(xs, series("hard", None), "-o", color="#c24a30", label="hard")
    styles = ["--", ":", "-"]
    for i, m in enumerate(margins):
        ax1.plot(xs, series("retract", m), styles[i % 3], marker="s", color="#3a46a6",
                 label="retract@{:g}".format(m), markersize=4)
    for i, t in enumerate(tolerances):
        ax1.plot(xs, series("tombstone", t), styles[i % 3], marker="^", color="#b07a16",
                 label="tombstone@{:g}".format(t), markersize=5)
    for i, t in enumerate(tolerances):
        ax1.plot(xs, series("modeswitch", t), styles[i % 3], marker="D", color="#1a9988",
                 label="modeswitch@{:g}".format(t), markersize=5)
    ax1.set_xlabel("answers flipped (%)")
    ax1.set_ylabel("in-pool failure rate")
    ax1.set_title("Hard breaks; the reversible backends recover it")
    ax1.legend(fontsize=7)
    ax1.set_ylim(bottom=0)

    # Right: the trade-off plane at high noise -- in-pool failure vs detector AUC.
    # Only soft reaches the good corner (low failure, high AUC); every reversible
    # config lands in the low-AUC band whatever its in-pool recovery.
    def pt(bk, knob):
        a = cell(cells, bk, knob, hi)
        return _m(a, "in_pool_fail_rate"), _m(a, "auc")
    def scat(bk, knob, color, marker, lab):
        x, y = pt(bk, knob)
        ax2.scatter([x], [y], c=color, marker=marker, s=70, zorder=3, label=lab)
    scat("soft", None, "#2d8a66", "o", "soft")
    scat("hard", None, "#c24a30", "o", "hard")
    for m in margins:
        scat("retract", m, "#3a46a6", "s", None)
    for t in tolerances:
        scat("tombstone", t, "#b07a16", "^", None)
    for t in tolerances:
        scat("modeswitch", t, "#1a9988", "D", None)
    ax2.scatter([], [], c="#3a46a6", marker="s", label="retract (scalar)")
    ax2.scatter([], [], c="#b07a16", marker="^", label="tombstone (naive)")
    ax2.scatter([], [], c="#1a9988", marker="D", label="modeswitch (mode switch)")
    ax2.axhline(0.5, color="#999", ls=":", lw=1)
    ax2.text(ax2.get_xlim()[1], 0.51, "chance", fontsize=7, color="#999",
             ha="right", va="bottom")
    ax2.set_xlabel("in-pool failure @{:.0f}%  (lower better →)".format(hi * 100))
    ax2.set_ylabel("detector AUC @{:.0f}%  (↑ better)".format(hi * 100))
    ax2.set_ylim(0.45, 1.03)
    ax2.set_title("Only the mode switch recovers without dropping the detector")
    ax2.legend(fontsize=7, loc="center left")

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
