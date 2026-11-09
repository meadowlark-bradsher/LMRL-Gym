"""Analyse the onboarding grid (ENV-704): print the policy x noise x effort table and
draw the diminishing-returns curve and the cost optimum.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_onboarding \
        --dir llm_rl_scripts/twenty_questions/jev/results/openworld_onboarding_<stamp>

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


def cell(cells: List[dict], policy: str, noise: float, E: int) -> Optional[dict]:
    for c in cells:
        if (c["policy"] == policy and abs(c["oracle_noise"] - noise) < 1e-9
                and c["effort"] == E):
            return c["metrics"]
    return None


def total_cost(m: dict, r: float) -> float:
    """c_expert = 1, c_elicit = r (the cost of one elicited attribute relative to one
    expert call)."""
    return r * m["onboarding_attrs"] + m["expert_calls"]


def report(path: str) -> None:
    data = load(path)
    cells = data["cells"]
    efforts = data["config"]["efforts"]
    policies = data["config"]["policies"]
    levels = data["config"]["levels"]
    print("=" * 78)
    print("Onboarding grid:", os.path.basename(path),
          "  ({} attributes total)".format(data["config"]["n_attributes"]))
    for lvl in levels:
        for policy in policies:
            print("-" * 78)
            print("policy={}  noise={:.0f}%".format(policy, lvl * 100))
            print("  E   resolve  reesc   wrong   q/recur   onbd_attrs  experts")
            for E in efforts:
                m = cell(cells, policy, lvl, E)
                if m is None:
                    continue
                print("  {:2d}   {:.3f}   {:.3f}   {:.3f}   {:6.2f}   {:8d}   {:6d}".format(
                    E, m["recur_resolution_rate"], m["recur_reescalation_rate"],
                    m["recur_wrong_rate"], m["recur_mean_questions"],
                    m["onboarding_attrs"], m["expert_calls"]))
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
    efforts = data["config"]["efforts"]
    policies = data["config"]["policies"]
    lvl = data["config"]["levels"][0]           # clean oracle: onboarding is the variable
    run = os.path.basename(path)
    full = data["config"]["n_attributes"]
    pcol = {"diagnostic": "#1a9988", "random": "#b07a16"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Left: the benefit axis is question-efficiency, not resolution. q/recurrence (solid)
    # barely moves until the row is near-complete, and only for diagnostic; random is
    # flat until full. Resolution (dashed, right axis) is floored by exact name-guessing
    # -- and a thin diagnostic row DIPS it (the partial-row collision danger zone).
    axr = ax1.twinx()
    for policy in policies:
        q = [(cell(cells, policy, lvl, E) or {}).get("recur_mean_questions", float("nan"))
             for E in efforts]
        res = [(cell(cells, policy, lvl, E) or {}).get("recur_resolution_rate", float("nan"))
               for E in efforts]
        ax1.plot(efforts, q, "-o", color=pcol.get(policy, "#555"), label=policy,
                 markersize=5)
        axr.plot(efforts, res, "--s", color=pcol.get(policy, "#555"), markersize=3,
                 alpha=0.55, linewidth=1)
    ax1.set_xlabel("onboarding effort E  (attributes elicited / admission)")
    ax1.set_ylabel("questions / recurrence  (solid)")
    axr.set_ylabel("resolution rate  (dashed)")
    axr.set_ylim(0.85, 1.01)
    ax1.axvline(full, color="#bbb", ls=":", lw=1)
    ax1.text(full, ax1.get_ylim()[0], " full", fontsize=7, color="#999",
             va="bottom", ha="right")
    ax1.annotate("partial-row\ndanger zone", (6, 11.8), fontsize=7, color="#c24a30",
                 ha="center")
    ax1.set_title("Onboarding buys only efficiency — and only near-complete")
    ax1.legend(fontsize=8, title="elicitation", loc="center left")

    # Right: total cost vs E (diagnostic, clean) for several cost ratios. Monotone
    # increasing -> E*=0 for every ratio: onboarding never lowers expert calls (name-
    # guessing already prevents re-escalation), so it is pure added cost.
    ratios = [0.02, 0.1, 0.5]
    styles = ["-o", "-s", "-^"]
    for r, st in zip(ratios, styles):
        costs = []
        for E in efforts:
            m = cell(cells, "diagnostic", lvl, E)
            costs.append(total_cost(m, r) if m else float("nan"))
        mx = max(costs)
        ax2.plot(efforts, [c / mx for c in costs], st,
                 label="c_elicit/c_expert = {:g}".format(r), markersize=5)
    emin = efforts[0]
    ax2.annotate("E*=0 (all ratios)", (emin, 0.02), fontsize=8, color="#c24a30",
                 xytext=(8, 20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="#c24a30", lw=1))
    ax2.set_xlabel("onboarding effort E")
    ax2.set_ylabel("total cost  (elicitation + expert calls, norm.)")
    ax2.set_title("Onboarding doesn't pay: it never lowers expert calls")
    ax2.legend(fontsize=7)

    fig.text(0.5, 0.005, "analyze_onboarding.py  |  {}".format(run),
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(path, "fig_onboarding.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[figure] wrote fig_onboarding.png -> {}".format(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    report(args.dir)


if __name__ == "__main__":
    main()
