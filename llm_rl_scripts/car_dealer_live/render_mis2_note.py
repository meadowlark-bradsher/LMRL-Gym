"""Render the CD-MIS-2 note (severity sweep; the evasive-answer channel) from ``analysis.json``
(``analyze_mis.py``) and ``verdicts.json`` (``verdicts_mis2.py``). Every number is formatted from
JSON; none is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis2_note --dir results/<dir> --verdicts <json> --out <md>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .render_mis_note import ci, f3, f4, kernel_str, side

MATRIX_ARMS = ["B_naive", "C_naive", "C_ans0.1", "C_ans0.2", "C_ans0.3", "O_true"]
DOSE_PAIRS = ["C_naive-B_naive", "C_ans0.1-C_naive", "C_ans0.2-C_naive", "C_ans0.3-C_naive", "O_true-B_naive", "O_true-C_naive", "O_true-C_ans0.2"]
SEV_PAIRS = ["B_naive-A_prior", "B_mix0.2-B_naive", "C_naive-B_naive", "O_true-C_naive"]


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
    S = a.get("severity") or {}
    I = a.get("trajectory_identity") or {}
    L = []
    w = L.append

    w("# Live Car-Dealer, CD-MIS-2 — severity sweep: how wrong can the opponent model be, and the evasive-answer channel")
    w("")
    w("_Rendered by `render_mis2_note.py` from `analysis.json` written by `analyze_mis.py`; every number is formatted from JSON, none is typed._")
    w("")
    w("No model is involved: every arm is scripted and every number is exact-on-model or Monte Carlo over the frozen seeds. "
      "The agent's ASSUMED behavioural kernel, planner, question bank and costs are fixed; the TRUE process, its severity, and the "
      "agent's answer model (the `C_ans` arms assume a flip rate) vary.")
    w("")
    w(f"Source: `{args.dir}` — env `{a.get('env')}`, design `{a.get('design')}`, regime {cfg.get('regime')} (r = {cfg.get('reservation')}), "
      f"H = {cfg.get('horizon')}, c_q = {cfg.get('ask_cost')}; assumed kernel {kernel_str(a.get('assumed_kernel'))}; "
      f"processes {procs}; arms {arms}; {a['episodes']} episodes ({a.get('repeats')} repeats per cell, seed {a.get('seed')}), "
      f"{a['error_stubs']} error stubs, {a.get('elapsed_s')} s. Clusters = {a['cluster_definition']}; bootstrap B = {a['bootstrap']['B']} (seed {a['bootstrap']['seed']}).")
    w("")
    if a.get("hedged_arms"):
        w("Hedged-answer arms (planner and filter both assume this flip rate; behavioural kernel = the registered T1): " +
          ", ".join(f"`{k}` → {v}" for k, v in a["hedged_arms"].items()) + ". `O_true` plans with the true process and the exact filter under it.")
        w("")
    w("Y = focal normalized net gain = (U − r − c_q × own asks) / (U_max − r); no deal gives Y = 0. "
      "'Identical' = share of seed-paired episodes in which two arms made the same decision sequence and reached the same terminal.")
    w("")

    # ---- evasive matrix ----------------------------------------------------------------------------
    ev = S.get("evasive")
    if ev:
        levels = ev["levels"]
        w("## The evasive channel: true flip rate × assumed flip rate")
        w("")
        w("Mean Y (all cells) by true flip rate (rows) and arm (columns). `C_naive` assumes truthful answers (assumed 0); "
          "`C_ans a` assumes a; `O_true` assumes the true rate (the diagonal); `B_naive` never asks.")
        w("")
        cols = [x for x in MATRIX_ARMS if x in ev["mean_normalized_net_gain"]]
        w("| true flip rate | " + " | ".join(f"`{c}`" for c in cols) + " |")
        w("|---|" + "---|" * len(cols))
        for lv in levels:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} (`{p}`) | " + " | ".join(f4(ev["mean_normalized_net_gain"][c].get(p)) for c in cols) + " |")
        w("")
        w("Questions asked per episode, same layout:")
        w("")
        w("| true flip rate | " + " | ".join(f"`{c}`" for c in cols) + " |")
        w("|---|" + "---|" * len(cols))
        for lv in levels:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} | " + " | ".join(f3(ev["mean_questions_asked"][c].get(p)) for c in cols) + " |")
        w("")
        w("### Dose-response contrasts along the true flip rate (paired cluster bootstrap, 95% CI, all cells)")
        w("")
        pairs = [q for q in DOSE_PAIRS if any(q in a["contrasts"].get(lv["process"], {}) for lv in levels)]
        w("| true flip rate | " + " | ".join(f"`{q}`" for q in pairs) + " |")
        w("|---|" + "---|" * len(pairs))
        for lv in levels:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} | " + " | ".join(ci(a["contrasts"].get(p, {}).get(q, {}).get("all")) for q in pairs) + " |")
        w("")
        w("Identical-trajectory share, same layout:")
        w("")
        w("| true flip rate | " + " | ".join(f"`{q}`" for q in pairs) + " |")
        w("|---|" + "---|" * len(pairs))
        for lv in levels:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} | " + " | ".join(f3((I.get(p, {}).get(q) or {}).get("identical_share")) for q in pairs) + " |")
        w("")

    # ---- behavioural severity ----------------------------------------------------------------------
    for fam in [f for f in S if f != "evasive"]:
        blk = S[fam]
        w(f"## Behavioural severity: `{fam}`")
        w("")
        cols = [x for x in blk["mean_normalized_net_gain"]]
        w("Mean Y (all cells) by severity level and arm:")
        w("")
        w("| level | " + " | ".join(f"`{c}`" for c in cols) + " |")
        w("|---|" + "---|" * len(cols))
        for lv in blk["levels"]:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} (`{p}`) | " + " | ".join(f4(blk["mean_normalized_net_gain"][c].get(p)) for c in cols) + " |")
        w("")
        pairs = [q for q in SEV_PAIRS if any(q in a["contrasts"].get(lv["process"], {}) for lv in blk["levels"])]
        w("| level | " + " | ".join(f"`{q}`" for q in pairs) + " |")
        w("|---|" + "---|" * len(pairs))
        for lv in blk["levels"]:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} | " + " | ".join(ci(a["contrasts"].get(p, {}).get(q, {}).get("all")) for q in pairs) + " |")
        w("")
        w("Identical-trajectory share, same layout:")
        w("")
        w("| level | " + " | ".join(f"`{q}`" for q in pairs) + " |")
        w("|---|" + "---|" * len(pairs))
        for lv in blk["levels"]:
            p = lv["process"]
            w(f"| {lv['severity']:.2f} | " + " | ".join(f3((I.get(p, {}).get(q) or {}).get("identical_share")) for q in pairs) + " |")
        w("")

    # ---- per process per arm -----------------------------------------------------------------------
    w("## Per process, per arm (all cells)")
    w("")
    w("| process | arm | n | mean Y [95% CI] | agreement | walk rate | asks / episode | decision regret (true) | final p(true type) | final log-loss (bits) | argmax = true | predictive log-loss (bits) | Brier(accept) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in procs:
        for arm in arms:
            s = a["per_arm"].get(p, {}).get(arm, {}).get("all")
            if not s or s.get("n", 0) == 0:
                continue
            cy = s.get("ci95_normalized_net_gain")
            cytxt = f"{f4(s['mean_normalized_net_gain'])} [{f4(cy[0])}, {f4(cy[1])}]" if cy else f4(s["mean_normalized_net_gain"])
            w(f"| `{p}` | {arm} | {s['n']} | {cytxt} | {f3(s['agreement_rate'])} | {f3(s['walk_rate'])} | {f3(s['mean_questions_asked'])} | "
              f"{f4(s['mean_total_decision_regret_true'])} | {f3(s['mean_final_p_true'])} | {f3(s['mean_final_logloss_true_bits'])} | "
              f"{f3(s['final_argmax_true_share'])} | {f3(s['mean_predictive_logloss_bits'])} | {f3(s['mean_brier_accept'])} |")
    w("")

    # ---- verdicts ----------------------------------------------------------------------------------
    if V is not None:
        w(f"## Pre-registered verdicts — `{args.verdicts}`")
        w("")
        w(f"Confirmatory: `{V['confirmatory']['contrast']}` under `{V['confirmatory']['process']}`. Sanity gates pass: **{V.get('gates_pass')}**.")
        w("")
        w("| rule | process | quantity | prediction | estimate [95% CI] | identical | verdict |")
        w("|---|---|---|---|---|---|---|")
        for k, v in V.get("rules", {}).items():
            e = v.get("estimate")
            etxt = f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] (n = {e['n_clusters']})" if e and e.get("point") is not None and e.get("ci95") else "—"
            w(f"| {k} | `{v['process']}` | {v.get('quantity')} | {v.get('prediction')} | {etxt} | {f3(v.get('identical_trajectory_share'))} | **{v.get('verdict')}** |")
        w("")
        for gk, g in (V.get("gates") or {}).items():
            short = {k: v for k, v in g.items() if k != "episodes_per_process_arm"}
            w(f"- gate `{gk}`: {'PASS' if g['pass'] else 'FAIL'} — {json.dumps(short)}")
        w("")

    # ---- computed read-out -------------------------------------------------------------------------
    w("## Read-out (computed)")
    w("")
    if ev:
        for lv in ev["levels"]:
            p = lv["process"]
            c = a["contrasts"].get(p, {}).get("C_naive-B_naive", {}).get("all")
            h = a["contrasts"].get(p, {}).get("C_ans0.2-C_naive", {}).get("all")
            o = a["contrasts"].get(p, {}).get("O_true-B_naive", {}).get("all")
            asks = ev["mean_questions_asked"].get("O_true", {}).get(p)
            w(f"- true flip {lv['severity']:.2f}: trusting answers (`C_naive − B_naive`) {ci(c)} ({side(c)}); the 0.2 hedge over trusting "
              f"(`C_ans0.2 − C_naive`) {ci(h)} ({side(h)}); value of questions with the right answer model (`O_true − B_naive`) {ci(o)} ({side(o)}); "
              f"O_true asks {f3(asks)} per episode.")
    for fam in [f for f in S if f != "evasive"]:
        for lv in S[fam]["levels"]:
            p = lv["process"]
            c = a["contrasts"].get(p, {}).get("B_naive-A_prior", {}).get("all")
            m = a["contrasts"].get(p, {}).get("B_mix0.2-B_naive", {}).get("all")
            o = a["contrasts"].get(p, {}).get("O_true-C_naive", {}).get("all")
            im = (I.get(p, {}).get("B_mix0.2-B_naive") or {}).get("identical_share")
            w(f"- `{p}`: learning (`B_naive − A_prior`) {ci(c)} ({side(c)}); robust filter (`B_mix0.2 − B_naive`) {ci(m)} with identical trajectories {f3(im)}; "
              f"misspecification cost for the ASK planner (`O_true − C_naive`) {ci(o)} ({side(o)}).")
    w("")
    w("## Regenerate")
    w("")
    w("```bash")
    w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir {args.dir} --B {a['bootstrap']['B']} --seed {a['bootstrap']['seed']}")
    w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_mis2_note --dir {args.dir} --out {args.out}" + (f" --verdicts {args.verdicts}" if args.verdicts else ""))
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
