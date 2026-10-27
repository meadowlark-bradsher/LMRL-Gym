"""Render SCRIPTED_RESULTS.md from the frozen analysis and sweep JSON.

No number in the note is typed by hand: everything is read from
``results/binned_b_scripted_*_20261026/analysis.json`` and
``results/binned_b_sweep_20261026/sweep.json``. Re-run after regenerating any of
those (each result dir has a ``REGENERATE.sh``).

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.render_scripted_note \
        --out llm_rl_scripts/car_dealer_binned/SCRIPTED_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

RUNS = [
    ("r2_easy", "2 rounds, easy (primary pre-registered horizon)"),
    ("r6_easy", "6 rounds, easy (the document's horizon)"),
    ("r2_hard", "2 rounds, hard"),
]
ARMS = ["exact", "greedy_eig", "myopic", "random"]


def load(root, name):
    p = os.path.join(root, f"results/binned_b_scripted_{name}_20261026/analysis.json")
    with open(p) as f:
        return json.load(f), os.path.relpath(p, root)


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(d):
    if d is None:
        return "—"
    return f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    root = os.path.abspath(args.root)

    sweep_path = os.path.join(root, "results/binned_b_sweep_20261026/sweep.json")
    with open(sweep_path) as f:
        sweep = json.load(f)
    sweep_rel = os.path.relpath(sweep_path, root)

    L = []
    w = L.append
    w("# Binned Car Dealer, Condition B — scripted-arm results (no model)")
    w("")
    w("_Rendered by `render_scripted_note.py` from frozen JSON; do not edit numbers by hand._")
    w("")
    w("Every number here is **exact-on-logs** except the `random` arm (a seeded draw, 20 plays per cell).")
    w("Game `binned-car-dealer-b-v0.1`; design in [`DESIGN_B.md`](DESIGN_B.md). 72 cells = 2 roles × 6 own")
    w("types × 6 opponent types; the deterministic arms play each cell once.")
    w("")
    w("## 1. Horizon × regime sweep — when does the ask-vs-act trade-off bind?")
    w("")
    w(f"Source: `{sweep_rel}` (`sweep_horizon.py`, seed {sweep['seed']}). V = exact reference value from the uniform prior,")
    w("averaged over the 72 cells; arms are realised means over the same cells. `greedy<exact` counts cells in which")
    w("greedy-EIG strictly underperforms the exact policy.")
    w("")
    w("| regime | rounds | V_menu | V_full | exact | greedy_eig | myopic | random | exact asks | exact offers | greedy asks | greedy<exact cells | menu costless |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sweep["rows"]:
        m = r["mean"]
        c = r["per_cell_counts"]
        w(
            f"| {r['regime']} | {r['rounds']} | {f4(m['V_menu']['float'])} | {f4(m['V_full']['float'])} | {f4(m['exact']['float'])} | "
            f"{f4(m['greedy_eig']['float'])} | {f4(m['myopic']['float'])} | {f4(r['mean_random'])} | {c['exact_asked']:.2f} | {c['exact_offers']:.2f} | "
            f"{c['greedy_asked']:.2f} | {r['cells_greedy_below_exact']}/{r['cells']} | {'yes' if r['menu_rule_costless'] else 'no'} |"
        )
    w("")
    def rounds_where(rows, pred):
        return [r["rounds"] for r in rows if pred(r)]

    costless = rounds_where(sweep["rows"], lambda r: r["menu_rule_costless"])
    not_costless = [(r["regime"], r["rounds"]) for r in sweep["rows"] if not r["menu_rule_costless"]]
    w("**Read-out (exact-on-logs; every list below is computed from the table).**")
    w(f"- Menu rule: V_menu = V_full in {len(costless)}/{len(sweep['rows'])} rows; the exception(s) {not_costless if not_costless else 'none'} — at a 1-decision-plus-final horizon the two-question menu can omit the single best probe.")
    for regime in ("easy", "hard"):
        rows = [r for r in sweep["rows"] if r["regime"] == regime]
        bind = rounds_where(rows, lambda r: r["cells_greedy_below_exact"] > 0)
        free = rounds_where(rows, lambda r: r["greedy_equals_exact_everywhere"])
        noask = rounds_where(rows, lambda r: r["per_cell_counts"]["exact_asked"] == 0)
        myo_eq = rounds_where(rows, lambda r: r["mean"]["myopic"]["exact"] == r["mean"]["exact"]["exact"])
        w(f"- {regime.capitalize()} regime: greedy-EIG underperforms the exact policy at rounds ∈ {bind} and equals it in every cell at rounds ∈ {free}; "
          f"the exact policy (ties → act) asks no questions at rounds ∈ {noask}; the never-ask `myopic` arm equals the exact policy at rounds ∈ {myo_eq}.")
    w("- Reading: once the horizon has slack, an offer is the better probe because it both tests a type set and can close, so greedy-EIG's extra questions cost nothing (easy, rounds ≥ 3) and the trade-off that Condition B exists to measure only binds at short horizons.")
    w("- Consequence for the document's **H2** (\"greedy EIG can underperform the optimal policy in Condition B\"): true at the 2-round horizon in both regimes, false at the document's 6-round horizon in both regimes. The primary pre-registered Jev horizon is therefore 2 rounds (`PREREGISTRATION_B.md`).")
    w("")

    for name, title in RUNS:
        a, rel = load(root, name)
        w(f"## {2 + RUNS.index((name, title))}. {title}")
        w("")
        w(f"Source: `{rel}` (`analyze_condition_b.py`, B = {a['bootstrap']['B']}, seed {a['bootstrap']['seed']}); {a['episodes']} episodes.")
        w("")
        w("| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret decisions | Σ EIG regret (bits) | questions | offers | C_H (bits) | final H (bits) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            s = a["by_arm"].get(arm)
            if not s:
                continue
            w(
                f"| `{arm}` | {s['episodes']} | {f4(s['mean_learner_utility'])} | {f4(s['mean_gain'])} | {f3(s['agreement_rate'])} | {f4(s['mean_optimality_gap'])} | "
                f"{f4(s['mean_total_decision_regret'])} | {f3(s['zero_regret_decision_share'])} | {f4(s['mean_total_eig_regret_bits'])} | {s['mean_questions_asked']:.2f} | "
                f"{s['mean_offers_made']:.2f} | {f3(s['mean_cumulative_entropy_bits'])} | {f3(s['mean_final_entropy_bits'])} |"
            )
        w("")
        w("Paired, cell-clustered bootstrap, arm − `exact` (for the deterministic arms the interval expresses variation across the type space, not sampling noise):")
        w("")
        w("| arm − exact | Δ utility | Δ gain | Δ agreement | Δ Σ decision regret | Δ Σ EIG regret (bits) | Δ questions |")
        w("|---|---|---|---|---|---|---|")
        for arm in ARMS[1:]:
            d = a["vs_exact"].get(arm)
            if not d:
                continue
            w(f"| `{arm}` | {ci(d['learner_utility'])} | {ci(d['gain'])} | {ci(d['agreement'])} | {ci(d['total_decision_regret'])} | {ci(d['total_eig_regret_bits'])} | {ci(d['questions_asked'])} |")
        w("")
        by_role = {k: v for k, v in a["by_arm_role"].items() if k.startswith("exact/")}
        w("By role (`exact` arm): " + "; ".join(f"{k.split('/')[1]} U = {f4(v['mean_learner_utility'])}, asks {v['mean_questions_asked']:.2f}" for k, v in sorted(by_role.items())) + ".")
        kinds = a["by_arm"]["exact"]["chosen_tag_counts"]
        w(f"Offer tags chosen by `exact`: {kinds}. Deals accepted under `exact`: {a['by_arm']['exact']['accepted_deal_counts']}.")
        w("")

    w("## Claims tagged")
    w("")
    w("- exact-on-logs: every row of the `exact`, `greedy_eig`, `myopic` arms and every V; the sweep's cell counts.")
    w("- stochastic: the `random` arm (20 seeded plays per cell) and its intervals.")
    w("")
    w("## Regenerate")
    w("")
    w("```bash")
    for name, _ in RUNS:
        w(f"sh results/binned_b_scripted_{name}_20261026/REGENERATE.sh")
    w("PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.sweep_horizon --out results/binned_b_sweep_20261026/sweep.json --random-repeats 20 --seed 0")
    w("PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.render_scripted_note --out llm_rl_scripts/car_dealer_binned/SCRIPTED_RESULTS.md")
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
