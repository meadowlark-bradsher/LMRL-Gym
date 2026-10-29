"""Render the CD-MIS-1 note from ``analysis.json`` (``analyze_mis.py``) and, optionally,
``verdicts.json``. Every number is formatted from JSON; none is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis_note --dir results/<dir> --out <md> [--verdicts <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CELL_SETS = ("all", "focal_opens", "opponent_opens")
EPS = 1e-9


def f4(x, sign=False):
    if x is None:
        return "—"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(c, nd=4):
    if not c or c.get("point") is None:
        return "—"
    if not c.get("ci95"):
        return f"{c['point']:+.{nd}f} [no CI]"
    return f"{c['point']:+.{nd}f} [{c['ci95'][0]:+.{nd}f}, {c['ci95'][1]:+.{nd}f}]"


def kernel_str(k) -> str:
    if not isinstance(k, dict):
        return str(k)
    return ", ".join(f"{a}={k[a]}" for a in ("alpha0", "beta", "p_counter", "p_hold", "shortlist", "accept_noise") if a in k)


def side(c) -> str:
    """'below 0' / 'above 0' / 'unresolved' / 'no data' from a contrast's CI."""
    if not c or c.get("point") is None or not c.get("ci95"):
        return "no data"
    lo, hi = c["ci95"]
    if hi < -EPS:
        return "below 0"
    if lo > EPS:
        return "above 0"
    return "unresolved"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdicts", default=None)
    args = ap.parse_args(argv)
    a = json.load(open(os.path.join(args.dir, "analysis.json")))
    V = json.load(open(args.verdicts)) if args.verdicts else None
    procs, arms = a["processes"], a["arms"]
    cfg = a["config"] or {}
    L = []
    w = L.append

    w("# Live Car-Dealer, CD-MIS-1 — learning with the wrong opponent model")
    w("")
    w("_Rendered by `render_mis_note.py` from `analysis.json` written by `analyze_mis.py`; every number is formatted from JSON, none is typed._")
    w("")
    w("No model is involved: every arm is scripted and every number is exact-on-model or Monte Carlo over the frozen seeds. "
      "The agent's ASSUMED kernel, planner, question bank and costs are fixed; only the TRUE opponent process and the belief filter vary.")
    w("")
    w(f"Source: `{args.dir}` — env `{a.get('env')}`, design `{a.get('design')}`, regime {cfg.get('regime')} (r = {cfg.get('reservation')}), "
      f"H = {cfg.get('horizon')}, c_q = {cfg.get('ask_cost')}; assumed kernel {kernel_str(a.get('assumed_kernel'))}; "
      f"processes {procs}; arms {arms}; {a['episodes']} episodes ({a.get('repeats')} repeats per cell, seed {a.get('seed')}), "
      f"{a['error_stubs']} error stubs, {a.get('elapsed_s')} s. Clusters = {a['cluster_definition']}; bootstrap B = {a['bootstrap']['B']} (seed {a['bootstrap']['seed']}).")
    w("")
    if a.get("filters"):
        w("Filters: " + "; ".join(f"`{k}` → {v['name']} (update={v['update']}, eta={v['eta']}, tau={v['tau']})" for k, v in a["filters"].items()) + ".")
        w("")
    w("Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r); no deal gives Y = 0. "
      "Decision regret is measured by O_true (Q under the true process at the exact true posterior). "
      "Belief diagnostics are on the agent's own posterior at the end of the episode; predictive log-loss is on every observed opponent response under the agent's effective likelihood; "
      "Brier is on the assumed model's P(accept) for the focal's own proposals.")
    w("")

    # (a) per-process per-arm tables
    for p in procs:
        w(f"## Process `{p}`")
        w("")
        w("### Per arm (all cells)")
        w("")
        w("| arm | n | mean Y [95% CI] | agreement | walk rate | asks / episode | decision regret (true) | final p(true type) | final log-loss (bits) | final entropy (bits) | argmax = true | predictive log-loss (bits) | Brier(accept) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            s = a["per_arm"].get(p, {}).get(arm, {}).get("all")
            if not s or s.get("n", 0) == 0:
                continue
            cy = s.get("ci95_normalized_net_gain")
            cytxt = f"{f4(s['mean_normalized_net_gain'])} [{f4(cy[0])}, {f4(cy[1])}]" if cy else f4(s["mean_normalized_net_gain"])
            w(f"| {arm} | {s['n']} | {cytxt} | {f3(s['agreement_rate'])} | {f3(s['walk_rate'])} | {f3(s['mean_questions_asked'])} | "
              f"{f4(s['mean_total_decision_regret_true'])} | {f3(s['mean_final_p_true'])} | {f3(s['mean_final_logloss_true_bits'])} | "
              f"{f3(s['mean_final_entropy_bits'])} | {f3(s['final_argmax_true_share'])} | {f3(s['mean_predictive_logloss_bits'])} | {f3(s['mean_brier_accept'])} |")
        w("")
        # (b) contrasts
        C = a["contrasts"].get(p, {})
        if C:
            w("### Paired contrasts in mean Y (cluster bootstrap, 95% percentile CI)")
            w("")
            w("| contrast | all cells | n | focal opens | n | opponent opens | n |")
            w("|---|---|---|---|---|---|---|")
            for name, c in C.items():
                w(f"| {name} | {ci(c['all'])} | {c['all']['n_clusters']} | {ci(c['focal_opens'])} | {c['focal_opens']['n_clusters']} | "
                  f"{ci(c['opponent_opens'])} | {c['opponent_opens']['n_clusters']} |")
            w("")
        # (d) belief error vs decision loss
        D = a.get("contrasts_diag", {})
        rows = []
        for arm in arms:
            if arm == "A_prior":
                continue
            name = f"{arm}-A_prior"
            cl = D.get("final_logloss_true_bits", {}).get(p, {}).get(name, {}).get("all")
            cr = D.get("total_decision_regret_true", {}).get(p, {}).get(name, {}).get("all")
            cy = C.get(name, {}).get("all")
            if cl or cr or cy:
                rows.append(f"| {arm} | {ci(cl, 3)} | {ci(cr)} | {ci(cy)} |")
        if rows:
            w("### Belief error vs decision loss (each arm − A_prior, all cells)")
            w("")
            w("| arm | Δ final log-loss of the true type (bits; < 0 = better posterior) | Δ decision regret (true; < 0 = better decisions) | Δ Y (> 0 = better outcome) |")
            w("|---|---|---|---|")
            for r in rows:
                w(r)
            w("")

    # (c) cross-process table
    XP = a.get("cross_process", {})
    if XP.get("mean_normalized_net_gain"):
        w("## Across true processes: mean Y per arm (all cells), with the paired `process − matched` contrast in brackets")
        w("")
        w("| arm | " + " | ".join(f"`{p}`" for p in procs) + " |")
        w("|---|" + "---|" * len(procs))
        for arm in arms:
            means = XP["mean_normalized_net_gain"].get(arm, {})
            vs = XP.get("vs_matched", {}).get(arm, {})
            cells = []
            for p in procs:
                m = means.get(p)
                if m is None:
                    cells.append("—")
                elif p in vs and vs[p].get("point") is not None:
                    cells.append(f"{f4(m)} [{ci(vs[p])}]")
                else:
                    cells.append(f4(m))
            w(f"| {arm} | " + " | ".join(cells) + " |")
        w("")

    # (e) verdicts
    if V is not None:
        w(f"## Pre-registered verdicts — `{args.verdicts}`")
        w("")
        w(f"Sanity gates pass: **{V.get('gates_pass')}**.")
        w("")
        w("| rule | quantity | prediction | estimate [95% CI] | verdict |")
        w("|---|---|---|---|---|")
        for k, v in V.get("rules", {}).items():
            e = v.get("estimate")
            etxt = f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] (n = {e['n_clusters']})" if e and e.get("point") is not None and e.get("ci95") else (
                f"{e['point']:+.4f} (n = {e.get('n_clusters')})" if e and e.get("point") is not None else "—")
            w(f"| {k} | {v.get('quantity')} | {v.get('prediction')} | {etxt} | **{v.get('verdict')}** |")
        w("")
        for gk, g in (V.get("gates") or {}).items():
            w(f"- gate `{gk}`: {json.dumps(g)}")
        if V.get("gates"):
            w("")

    # computed read-out
    w("## Read-out (computed)")
    w("")
    for p in procs:
        c = a["contrasts"].get(p, {}).get("B_naive-A_prior", {}).get("all")
        if c:
            w(f"- `{p}`: B_naive − A_prior (all cells) is **{side(c)}**: {ci(c)}.")
    cm = a["contrasts"].get("matched", {}).get("B_mix0.2-B_naive", {}).get("all")
    if cm:
        s = side(cm)
        w(f"- Cost of robustness under `matched`: B_mix0.2 − B_naive = {ci(cm)} — {'≤ 0 (a cost)' if s == 'below 0' else ('> 0 (no cost measured)' if s == 'above 0' else 'unresolved')}.")
    for p in procs:
        if p == "matched":
            continue
        c = a["contrasts"].get(p, {}).get("B_mix0.2-B_naive", {}).get("all")
        if c:
            w(f"- Robustness under `{p}`: B_mix0.2 − B_naive = {ci(c)} ({side(c)}).")
    for p in procs:
        c = a["contrasts"].get(p, {}).get("O_true-B_naive", {}).get("all")
        if c:
            w(f"- Cost of misspecification under `{p}`: O_true − B_naive = {ci(c)} — {'positive' if side(c) == 'above 0' else side(c)}.")
    w("")
    w("## Regenerate")
    w("")
    w("```bash")
    w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir {args.dir} --B {a['bootstrap']['B']} --seed {a['bootstrap']['seed']}")
    w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis_note --dir {args.dir} --out {args.out}" + (f" --verdicts {args.verdicts}" if args.verdicts else ""))
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
