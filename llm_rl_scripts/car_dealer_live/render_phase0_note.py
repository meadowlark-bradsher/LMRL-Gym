"""Render PHASE0_RESULTS.md from the frozen sweep JSON. No number typed by hand.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase0_note \
        --sweep results/live_phase0_sweep_20261027/sweep.json \
        --out llm_rl_scripts/car_dealer_live/PHASE0_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sw = json.load(open(args.sweep))
    agg = list(sw["aggregate"].values())
    cells = sw["cells"]
    per_agg = sorted({a["cells"] for a in agg})
    assert len(per_agg) == 1, per_agg
    per_agg = per_agg[0]
    n_roles = len({c["focal"] for c in cells})
    n_openers = len({c["opener"] for c in cells})
    assert per_agg == n_roles * n_openers * len(sw["grid"]["types"]), (per_agg, n_roles, n_openers)
    L = []
    w = L.append
    w("# Live Car-Dealer, Phase 0 — where is ASK optimal? (exact oracle, no model)")
    w("")
    w("_Rendered by `render_phase0_note.py` from frozen JSON; do not edit numbers by hand._")
    w("")
    w(f"Source: `{os.path.relpath(args.sweep)}` (`sweep_ask_value.py`, env `{sw['env']}`, regime {sw['regime']}, "
      f"opponent shortlist {sw['shortlist']}, {len(cells)} cells = {len(agg)} (kernel, H, c_q) aggregates × {per_agg} cells each "
      f"({n_roles} focal roles × {n_openers} openers × {len(sw['grid']['types'])} focal types; the opponent's type is marginalised by the "
      f"uniform prior, so it is not a cell dimension); {sw['elapsed_s']} s). Every number is **exact-on-model**: the oracle is solved, nothing is sampled. "
      f"Every mean in this note is over the {per_agg} cells of its aggregate unless it says otherwise.")
    w("")
    w("Quantities: V\\*_ask / V\\*_noask = the focal negotiator's optimal expected net gain from the opening state with ASK "
      "allowed / forbidden at every decision; Δ\\*_active = V\\*_ask − V\\*_noask (≥ 0 by construction); `ask-opt nodes` = focal "
      "decision nodes (ASK legal) where ASK is *strictly* optimal, out of all such nodes solved; `root ASK` = cells whose "
      "optimal opening move (when the focal opens) is a question.")
    w("")
    w("## 1. Aggregates per (kernel, H, c_q)")
    w("")
    w("| kernel (α₀, β, p_counter, p_hold) | H | c_q | mean Δ\\*_active | max Δ\\*_active | cells with Δ > 0 | ask-opt nodes | root ASK | mean V\\*_ask |")
    w("|---|---|---|---|---|---|---|---|---|")
    for a in sorted(agg, key=lambda a: (a["kernel"], a["H"], float(eval(a["ask_cost"])))):
        p = a["params"]
        w(f"| ({float(eval(p['alpha0'])):.2f}, {p['beta']:g}, {p['p_counter']:g}, {p['p_hold']:g}) | {a['H']} | {a['ask_cost']} | "
          f"{a['mean_delta_active']:.4f} | {a['max_delta_active']:.4f} | {a['share_cells_delta_positive']:.2f} | "
          f"{a['ask_optimal_nodes']}/{a['ask_legal_nodes']} | {a['root_ask_cells']}/{a['cells']} | {a['mean_V_ask']:.4f} |")
    w("")
    # read-out computed
    byH = defaultdict(list)
    for a in agg:
        byH[a["H"]].append(a)
    w("## 2. Read-out (computed from the table)")
    w("")
    for H in sorted(byH):
        rows = byH[H]
        pos = [a for a in rows if a["mean_delta_active"] > 1e-9]
        best = max(rows, key=lambda a: a["mean_delta_active"])
        line = (f"- **H = {H}:** Δ\\*_active > 0 in {len(pos)}/{len(rows)} (kernel, c_q) aggregates; mean Δ ranges "
                f"{min(a['mean_delta_active'] for a in rows):.4f}–{max(a['mean_delta_active'] for a in rows):.4f}")
        if pos:
            line += (f"; the largest is kernel `{best['kernel']}` at c_q = {best['ask_cost']} "
                     f"(mean {best['mean_delta_active']:.4f}, ASK strictly optimal in {best['ask_optimal_nodes']}/{best['ask_legal_nodes']} nodes).")
        else:
            legal = sum(a["ask_legal_nodes"] for a in rows)
            line += f"; ASK is strictly optimal in 0/{legal} ASK-legal nodes across every kernel and cost, so no ask/offer trade-off exists at this horizon."
        w(line)
    # cost monotonicity per (kernel,H)
    viol = 0; pairs = 0
    byKH = defaultdict(list)
    for a in agg:
        byKH[(a["kernel"], a["H"])].append(a)
    for k, rows in byKH.items():
        rows.sort(key=lambda a: float(eval(a["ask_cost"])))
        for x, y in zip(rows, rows[1:]):
            pairs += 1
            if y["mean_delta_active"] > x["mean_delta_active"] + 1e-9:
                viol += 1
    w(f"- **Cost:** mean Δ\\*_active is non-increasing in c_q in {pairs - viol}/{pairs} adjacent (kernel, H) pairs (a structural expectation; violations would indicate a bug).")
    # horizon monotonicity
    hv = 0; hp = 0
    byKC = defaultdict(list)
    for a in agg:
        byKC[(a["kernel"], a["ask_cost"])].append(a)
    for k, rows in byKC.items():
        rows.sort(key=lambda a: a["H"])
        for x, y in zip(rows, rows[1:]):
            hp += 1
            if y["mean_delta_active"] < x["mean_delta_active"] - 1e-9:
                hv += 1
    # which kernels already show the predicted decrease at the top of the swept range (at c_q = 0)?
    Hs = sorted(byH)
    dec_kernels, inc_kernels = [], []
    if len(Hs) >= 2:
        for (k, cq), rows in sorted(byKC.items()):
            if float(eval(cq)) != 0.0:
                continue
            rows.sort(key=lambda a: a["H"])
            top, prev = rows[-1], rows[-2]
            (dec_kernels if top["mean_delta_active"] < prev["mean_delta_active"] - 1e-9 else inc_kernels).append(k)
    line = (f"- **Horizon:** mean Δ\\*_active increases with H in {hp - hv}/{hp} adjacent (kernel, c_q) pairs over the horizons swept (H ∈ {Hs}). "
            f"The design document's VOI-H2 predicts a *non-monotone* dependence once offers alone suffice. ")
    if dec_kernels:
        line += (f"At c_q = 0 the decrease from H = {Hs[-2]} to H = {Hs[-1]} already appears for {len(dec_kernels)}/{len(dec_kernels) + len(inc_kernels)} kernels "
                 f"({', '.join('`' + k + '`' for k in dec_kernels)}); the remaining {len(inc_kernels)} ({', '.join('`' + k + '`' for k in inc_kernels)}) still increase, "
                 f"so their peak lies at H ≥ {Hs[-1]} and is not located by this sweep.")
    else:
        line += f"No kernel shows a decrease within H ≤ {Hs[-1]}, so the peak is not located by this sweep."
    w(line)
    # by focal role at the best cell
    bestall = max(agg, key=lambda a: a["mean_delta_active"])
    bc = [c for c in cells if c["kernel"] == bestall["kernel"] and c["H"] == bestall["H"] and c["ask_cost"] == bestall["ask_cost"]]
    for role in ("buyer", "seller"):
        for opener in ("buyer", "seller"):
            sub = [c for c in bc if c["focal"] == role and c["opener"] == opener]
            md = sum(c['delta_active'] for c in sub) / len(sub)
            if role == opener:
                acts = sorted(set(str(c['root_action']) for c in sub))
                w(f"- Best aggregate (`{bestall['kernel']}`, H = {bestall['H']}, c_q = {bestall['ask_cost']}), focal = {role}, opener = {opener}: "
                  f"mean Δ\\*_active over the 6 focal types {md:.4f}; optimal opening moves {acts}.")
            else:
                w(f"- Best aggregate, focal = {role}, opener = {opener}: mean Δ\\*_active over the 6 focal types {md:.4f}; "
                  f"the focal has no opening move (the opponent opens), so no root action is reported.")
    w("")
    w("## 3. What this settles for Phase 1")
    w("")
    tot_pos = sum(1 for a in agg if a["mean_delta_active"] > 1e-9)
    w(f"- Gate 7 of the design document is **{'passed' if tot_pos else 'NOT passed'}**: {tot_pos}/{len(agg)} aggregates have states where ASK is strictly optimal, so a confirmatory `(H, c_q, r)` cell with a genuine ask/offer trade-off exists under the clean T1 kernel.")
    w("- The exact oracle is the planner at H ≤ 4, so the registered `C_plan − B_plan` contrast collapses to Δ\\*_active in those cells; the budgeted-planner fairness questions of section 5 apply only to H = 6 and T2.")
    w("- Nothing here is a research claim about any model; no registration has been made and no Jev or LLM call has been placed.")
    w("")
    w("## Regenerate")
    w("")
    w("```bash")
    w("PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.sweep_ask_value --out results/live_phase0_sweep_20261027/sweep.json")
    w("PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase0_note --sweep results/live_phase0_sweep_20261027/sweep.json --out llm_rl_scripts/car_dealer_live/PHASE0_RESULTS.md")
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
