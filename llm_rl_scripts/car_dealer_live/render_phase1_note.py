"""Render a Markdown note from one or more Phase 1 ``analysis.json`` files
(``analyze_phase1.py``). Every number is formatted from the JSON; nothing is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note \
        --dirs results/a,results/b --out llm_rl_scripts/car_dealer_live/PHASE1_SCRIPTED_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ASK_NOASK = [("exact_ask", "exact_noask"), ("jev_ask_plain", "jev_noask_plain"), ("jev_ask_eig", "jev_noask_eig")]
CELL_SETS = ("all", "focal_opens", "opponent_opens")
LOSS_EPS = 1e-9


def f4(x, sign=False):
    if x is None:
        return "—"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(c):
    if not c or c.get("point") is None:
        return "—"
    lo, hi = c["ci95"]
    return f"{c['point']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def kernel_str(k: dict) -> str:
    return f"α₀={k['alpha0']}, β={k['beta']:g}, p_counter={k['p_counter']:g}, p_hold={k['p_hold']:g}, shortlist={k['shortlist']}"


def section(d: str, a: dict, w) -> None:
    cfg = a["config"]
    w(f"## H = {cfg['horizon']}, c_q = {cfg['ask_cost']} — `{d}`")
    w("")
    w(f"Config: env `{a['env']}`, regime {cfg['regime']} (r = {cfg['reservation']}), kernel {kernel_str(a['kernel'])}, "
      f"menu `{a['menu']}`, arms {a['arms']}, {a['episodes']} episodes ({a['repeats']} repeats per cell, seed {a['seed']}), "
      f"{a['error_stubs']} error stubs, {a['elapsed_s']} s. Bootstrap B = {a['bootstrap']['B']}, clusters = {a['cluster_definition']}.")
    w("")
    # per-arm table
    w("### Per arm (all cells)")
    w("")
    w("| arm | n | mean Y (normalized net gain) | mean net gain | mean gross gain | agreement | questions / episode | proposals / episode | total decision regret | zero-regret decisions | EIG regret (bits) | final entropy (bits) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in a["arms"]:
        s = a["per_arm"][arm]["all"]
        if s.get("n", 0) == 0:
            continue
        w(f"| {arm} | {s['n']} | {f4(s['mean_normalized_net_gain'])} | {f4(s['mean_net_gain'])} | {f4(s['mean_gross_gain'])} | "
          f"{f3(s['agreement_rate'])} | {f3(s['mean_questions_asked'])} | {f3(s['mean_proposals_made'])} | "
          f"{f4(s['mean_total_decision_regret'])} | {f3(s['zero_regret_decision_share'])} | {f3(s['mean_total_eig_regret_bits'])} | {f3(s['mean_final_entropy_bits'])} |")
    w("")
    w("### Per arm, mean Y by who opens")
    w("")
    w("| arm | focal opens: n | focal opens: mean Y | opponent opens: n | opponent opens: mean Y |")
    w("|---|---|---|---|---|")
    for arm in a["arms"]:
        fo, oo = a["per_arm"][arm]["focal_opens"], a["per_arm"][arm]["opponent_opens"]
        w(f"| {arm} | {fo.get('n', 0)} | {f4(fo.get('mean_normalized_net_gain'))} | {oo.get('n', 0)} | {f4(oo.get('mean_normalized_net_gain'))} |")
    w("")
    # contrasts
    w("### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)")
    w("")
    w("| contrast | all cells | n | focal opens | n | opponent opens | n |")
    w("|---|---|---|---|---|---|---|")
    for name, c in a["contrasts"].items():
        w(f"| {name} | {ci(c['all'])} | {c['all']['n_clusters']} | {ci(c['focal_opens'])} | {c['focal_opens']['n_clusters']} | "
          f"{ci(c['opponent_opens'])} | {c['opponent_opens']['n_clusters']} |")
    w("")
    # ceilings
    w("### Exact ceilings and the costlessness check")
    w("")
    w("| cell set | cells | mean Δ\\*_full | mean Δ\\*_menu | loss (ASK allowed) | loss (ASK forbidden) | loss from offer restriction only | loss from question restriction only | cells with any loss > 1e-9 | cells with Δ\\*_menu > 0 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for cs in CELL_SETS:
        c = a["ceilings"].get(cs, {})
        if not c.get("n_cells"):
            w(f"| {cs} | 0 | — | — | — | — | — | — | — | — |")
            continue
        w(f"| {cs} | {c['n_cells']} | {f4(c['mean_delta_full'])} | {f4(c['mean_delta_menu'])} | {f4(c['mean_loss_ask'])} | {f4(c['mean_loss_noask'])} | "
          f"{f4(c['mean_loss_offers_restriction_ask'])} | {f4(c['mean_loss_questions_restriction_ask'])} | {c['cells_with_any_loss']} | {c['cells_with_delta_menu_positive']} |")
    w("")
    w("Δ\\*_full = V\\*_ask − V\\*_noask under the unrestricted action set; Δ\\*_menu = the same under the shortlist menu (the ceiling for any arm "
      "that sees the menu). Losses are V\\*(full) − V\\*(restricted) at the root; a loss of 0 means the restriction is costless on the oracle's own path.")
    w("")
    # captured fraction
    rows = []
    for x, z in ASK_NOASK:
        name = f"{x}-{z}"
        if name not in a["contrasts"]:
            continue
        for cs in CELL_SETS:
            c = a["contrasts"][name][cs]
            cf = c.get("captured_fraction", {})
            m = cf.get("delta_menu", {})
            fl = cf.get("delta_full", {})
            rows.append(f"| {name} | {cs} | {ci(c)} | {f4(m.get('ceiling'))} | {ci(m) if m.get('point') is not None else '—'} | {f4(fl.get('ceiling'))} | {ci(fl) if fl.get('point') is not None else '—'} |")
    if rows:
        w("### Captured fraction of the exact ceiling (ask arm − matching no-ask arm)")
        w("")
        w("| contrast | cell set | paired ΔY | mean Δ\\*_menu (Y units) | fraction of Δ\\*_menu | mean Δ\\*_full (Y units) | fraction of Δ\\*_full |")
        w("|---|---|---|---|---|---|---|")
        for r in rows:
            w(r)
        w("")
    # selectivity
    if a["selectivity"]:
        w("### Selectivity: P(ask) split by whether asking is strictly optimal in that state")
        w("")
        w("| arm | asked / decisions where ASK optimal | P(ask \\| optimal) | asked / decisions where ASK legal but not optimal | P(ask \\| not optimal) | difference [CI] |")
        w("|---|---|---|---|---|---|")
        for arm, s in a["selectivity"].items():
            diff = s.get("difference")
            dci = s.get("difference_ci95")
            dtxt = "—" if diff is None else (f"{diff:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]" if dci else f"{diff:+.3f}")
            w(f"| {arm} | {s['asked_when_optimal']}/{s['decisions_ask_optimal']} | {f3(s['p_ask_given_optimal'])} | "
              f"{s['asked_when_not_optimal']}/{s['decisions_ask_legal_not_optimal']} | {f3(s['p_ask_given_not_optimal'])} | {dtxt} |")
        w("")
    if a.get("position"):
        w("### Jev chosen position (0 = first lettered option)")
        w("")
        for arm, dist in a["position"].items():
            w(f"- {arm}: {dist}")
        w("")
    # computed read-out
    w("### Read-out (computed)")
    w("")
    c = a["contrasts"].get("exact_ask-exact_noask")
    if c:
        for cs in CELL_SETS:
            cc = c[cs]
            cst = a["ceilings"].get(cs, {})
            ceil = cst.get("mean_delta_menu_Y")
            if cc.get("point") is None or ceil is None:
                continue
            lo, hi = cc["ci95"]
            inside = lo - 1e-12 <= ceil <= hi + 1e-12
            w(f"- exact_ask − exact_noask ({cs}): {ci(cc)} vs exact Δ\\*_menu {f4(ceil)} in Y units "
              f"({f4(cst['mean_delta_menu'])} utility / {f4(cst['y_scale'])}) — the ceiling is {'inside' if inside else 'OUTSIDE'} the CI.")
    cl = a["ceilings"].get("all", {})
    if cl.get("n_cells"):
        if cl["cells_with_any_loss"] == 0:
            w(f"- The shortlist menu is costless at the root in all {cl['n_cells']} cells (every loss < 1e-9).")
        else:
            parts = []
            if (cl.get("mean_loss_offers_restriction_ask") or 0) > LOSS_EPS:
                parts.append(f"the offer restriction (mean loss {f4(cl['mean_loss_offers_restriction_ask'])} with ASK allowed)")
            if (cl.get("mean_loss_questions_restriction_ask") or 0) > LOSS_EPS:
                parts.append(f"the question restriction (mean loss {f4(cl['mean_loss_questions_restriction_ask'])} with ASK allowed)")
            if (cl.get("mean_loss_noask") or 0) > LOSS_EPS:
                parts.append(f"the offer restriction with ASK forbidden (mean loss {f4(cl['mean_loss_noask'])})")
            w(f"- The shortlist menu is NOT costless: {cl['cells_with_any_loss']}/{cl['n_cells']} cells lose value at the root; "
              f"mean loss with ASK allowed {f4(cl['mean_loss_ask'])}, with ASK forbidden {f4(cl['mean_loss_noask'])}. "
              f"Attribution: {'; '.join(parts) if parts else 'no single restriction above 1e-9 on average'}. "
              f"Δ\\*_menu ({f4(cl['mean_delta_menu'])} utility = {f4(cl.get('mean_delta_menu_Y'))} in Y units) is the ceiling used for the captured fraction; "
              f"Δ\\*_full is {f4(cl['mean_delta_full'])} utility. Losses in this bullet and the ceilings table are in utility units.")
    more, less = [], []
    for arm, s in a["selectivity"].items():
        if s.get("difference") is None:
            continue
        (more if s["difference"] > 0 else less).append(f"{arm} ({s['difference']:+.3f})")
    if more or less:
        w(f"- Arms asking more often when asking is optimal than when it is not: {', '.join(more) if more else 'none'}; "
          f"not more often: {', '.join(less) if less else 'none'}.")
    w("")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", required=True, help="comma-separated result dirs, each holding analysis.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--costlessness", default=None, help="optional costlessness_check.py JSON to tabulate")
    args = ap.parse_args(argv)
    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    analyses = {d: json.load(open(os.path.join(d, "analysis.json"))) for d in dirs}
    L = []
    w = L.append
    w("# Live Car-Dealer, Phase 1 — scripted arms against the T1 kernel")
    w("")
    w("_Rendered by `render_phase1_note.py` from `analysis.json` files written by `analyze_phase1.py`; every number is formatted from JSON, none is typed._")
    w("")
    w("This is a scripted-arm note: it contains no model result and no registration has been made. Arms: exact_ask (C_plan, the "
      "menu-restricted exact oracle with ASK), exact_noask (B_plan, the same oracle with ASK forbidden), greedy_eig, random; any jev_* arm "
      "present ran against the offline mock backend unless the section's config says otherwise.")
    w("")
    w("Result dirs:")
    w("")
    for d, a in analyses.items():
        cfg = a["config"]
        w(f"- `{d}` — env `{a['env']}`, H = {cfg['horizon']}, c_q = {cfg['ask_cost']}, regime {cfg['regime']}, kernel {kernel_str(a['kernel'])}, menu `{a['menu']}`, {a['episodes']} episodes.")
    w("")
    w("Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r), with U_max = 1 for every type; no deal gives Y = 0. "
      "Contrasts are paired over clusters (focal, opener, focal type, opponent type) with a percentile cluster bootstrap.")
    w("")
    for d in dirs:
        section(d, analyses[d], w)
    if args.costlessness:
        cc = json.load(open(args.costlessness))
        cfg = cc["config"]
        w(f"## Menu size and costlessness — `{args.costlessness}`")
        w("")
        w(f"`costlessness_check.py`, H = {cfg['horizon']}, c_q = {cfg['ask_cost']}, kernel {kernel_str(cc['kernel'])}, {cc['elapsed_s']} s. "
          "Root values of the exact oracle under the full action set and under the shortlist menu with n_offers × n_questions candidates "
          "(the offer rule yields at most three distinct candidates: SAFE, AMBITIOUS, BALANCED). Utility units.")
        w("")
        w("| menu (offers × questions) | cells lossless | mean loss, ASK allowed | max loss, ASK allowed | mean loss, ASK forbidden | max loss, ASK forbidden | mean Δ\\*_menu | mean Δ\\*_full |")
        w("|---|---|---|---|---|---|---|---|")
        for k, ag in cc["aggregate"].items():
            w(f"| {k} | {ag['cells_lossless']}/{ag['cells']} | {f4(ag['mean_loss_ask'])} | {f4(ag['max_loss_ask'])} | {f4(ag['mean_loss_noask'])} | "
              f"{f4(ag['max_loss_noask'])} | {f4(ag['mean_delta_menu'])} | {f4(ag['mean_delta_full'])} |")
        w("")
        lossy = sorted({(k.split("|")[0] == k.split("|")[1]) for k, v in cc["cells"].items() if any(x["loss_ask"] > 1e-9 or x["loss_noask"] > 1e-9 for x in v.values() if isinstance(x, dict))})
        where = ", ".join("focal opens" if f else "opponent opens" for f in lossy) or "none"
        w(f"- Cells with any loss under any menu size: {where}.")
        w("")
    w("## Regenerate")
    w("")
    w("```bash")
    for d in dirs:
        w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_phase1 --dir {d}")
    w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_phase1_note --dirs {','.join(dirs)} --out {args.out}" + (f" --costlessness {args.costlessness}" if args.costlessness else ""))
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
