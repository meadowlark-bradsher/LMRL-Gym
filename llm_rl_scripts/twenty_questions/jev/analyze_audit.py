"""Analyse the auditability grid (ENV-703): print the backend x noise table and draw
the scorecard — the three auditability properties per backend, and the sparse-vs-dense
log-storage asymmetry.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_audit \
        --dir llm_rl_scripts/twenty_questions/jev/results/openworld_audit_<stamp>

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
    for c in cells:
        kv = c.get("margin") if c.get("margin") is not None else c.get("tolerance")
        if (c["backend"] == backend and kv == knob
                and abs(c["oracle_noise"] - noise) < 1e-9):
            return c["agg"]
    return None


def _labels(data: dict):
    margins = data["config"]["margins"]
    tolerances = data["config"].get("tolerances", [])
    return ([("soft", None), ("hard", None)]
            + [("retract", m) for m in margins]
            + [("tombstone", t) for t in tolerances]
            + [("modeswitch", t) for t in tolerances])


def _fmt(x, d="  -  "):
    return d if x is None else "{:.3f}".format(x)


def report(path: str) -> None:
    data = load(path)
    cells = data["cells"]
    hi = data["config"]["levels"][-1]
    print("=" * 78)
    print("Auditability grid:", os.path.basename(path),
          "  (metrics at {:.0f}% noise)".format(hi * 100))
    print("-" * 78)
    print("backend      disc  ops/g  logcells/g  replay  verify   why  reinst_local")
    for bk, m in _labels(data):
        a = cell(cells, bk, m, hi)
        if a is None:
            continue
        name = bk if m is None else "{}@{:g}".format(bk, m)
        print("{:11s}  {:>4s}  {:5.2f}  {:9.1f}   {:>5s}   {:>5s}  {:>4s}   {:>5s}".format(
            name, "yes" if a["discrete"] else "no",
            a["ops_per_game"], a["log_cells_per_game"],
            _fmt(a["replay_exact_frac"]), _fmt(a["verifiable_frac"]),
            _fmt(a["why_frac"]), _fmt(a["reinstate_local_frac"])))
    print("=" * 78)
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
    hi = data["config"]["levels"][-1]
    run = os.path.basename(path)
    labels = _labels(data)
    names = [bk if m is None else "{}@{:g}".format(bk, m) for bk, m in labels]
    aggs = [cell(cells, bk, m, hi) for bk, m in labels]
    colors = {"soft": "#2d8a66", "hard": "#c24a30", "retract": "#3a46a6",
              "tombstone": "#b07a16", "modeswitch": "#1a9988"}
    bar_c = [colors[bk] for bk, _ in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left: the three auditability properties, grouped per backend. soft sits at 0 on
    # all three; every discrete backend sits at 1.0 -- the axis ENV-702 never scored.
    import numpy as np
    props = ["replay_exact_frac", "verifiable_frac", "why_frac"]
    plabels = ["replay-exact", "verifiable", "answer-why"]
    x = np.arange(len(labels))
    w = 0.26
    hatches = [None, "//", ".."]
    for i, (pk, pl) in enumerate(zip(props, plabels)):
        vals = [(a.get(pk) or 0.0) if a else 0.0 for a in aggs]
        ax1.bar(x + (i - 1) * w, vals, w, label=pl, color=bar_c, alpha=0.85,
                edgecolor="#333", hatch=hatches[i], linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("score @{:.0f}% noise".format(hi * 100))
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Auditability: soft scores 0, the discrete backends 1.0")
    ax1.legend(fontsize=7, loc="center right")
    ax1.text(0.0, 0.04, "soft", fontsize=7, color="#2d8a66", ha="center")

    # Right: storage to replay the committed state -- sparse op-log vs soft's dense
    # posterior. Log scale; soft is orders of magnitude larger and still not replayable.
    cellsz = [a["log_cells_per_game"] if a else 0.0 for a in aggs]
    ax2.bar(x, cellsz, 0.6, color=bar_c, alpha=0.85, edgecolor="#333", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("log cells / game to replay  (log scale)")
    ax2.set_title("Replay storage: a sparse op-log vs soft's dense trace")
    for xi, (a, nm) in enumerate(zip(aggs, names)):
        if a:
            tag = "dense" if not a["discrete"] else "ops"
            ax2.text(xi, (a["log_cells_per_game"] or 0.5) * 1.15, tag,
                     ha="center", fontsize=6, color="#555")

    fig.text(0.5, 0.005, "analyze_audit.py  |  {}".format(run),
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(path, "fig_auditability.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[figure] wrote fig_auditability.png -> {}".format(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    report(args.dir)


if __name__ == "__main__":
    main()
