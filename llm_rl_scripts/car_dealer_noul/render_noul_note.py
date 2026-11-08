"""Render the CD-NOUL-1 Stage 1 results note from ``analysis.json`` (analyze_noul.py) and
``verdicts.json`` (verdicts_noul.py). Every number is formatted from the JSON; nothing is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.render_noul_note \
        --dir results/<dir> --out llm_rl_scripts/car_dealer_noul/RESULTS_NOUL.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from llm_rl_scripts.car_dealer_live.render_phase1_note import ci, f3, f4, kernel_str

from . import arms_noul as an

CELL_SETS = ("all", "focal_opens", "opponent_opens")


def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def render(d: str, a: dict, v: dict, title: str = "CD-NOUL-1 Stage 1 — likelihood from the model, choice from the planner") -> str:
    lines = []
    w = lines.append
    w(f"# {title}")
    w("")
    w(f"_Rendered by `render_noul_note.py` from `analysis.json` (`analyze_noul.py`) and `verdicts.json` (`verdicts_noul.py`) in `{d}`; "
      "every number is formatted from JSON, none is typed. Design: `DESIGN_NOUL.md`; rules: `PREREGISTRATION_NOUL.md`._")
    w("")
    cfg, k = a["config"], a["kernel"]
    w(f"Env `{a['env']}`, regime {cfg['regime']} (r = {cfg['reservation']}), H = {cfg['horizon']}, c_q = {cfg['ask_cost']}; kernel {kernel_str(k)}. "
      f"Bank `{a['bank']}`. {a['episodes']} episodes ({a['repeats']} repeats per cell, seed {a['seed']}), {a['error_stubs']} error stubs, "
      f"{a['elapsed_s']} s. Bootstrap B = {a['bootstrap']['B']} (seed {a['bootstrap']['seed']}); clusters = {a['cluster_definition']}. "
      f"Y = net gain / (U_max − r), scale {f4(a['y_scale'])}. Primary cells: {v['primary_cells']}; primary hedge a = {v['primary_hedge']}; "
      f"agents' menu belief decoded by `{a['agent_menu_decoder']}` at a = {a['registered_hedge']}.")
    w("")
    # gates
    w("## Gates")
    w("")
    for name, g in v["gates"].items():
        w(f"- `{name}`: **{'pass' if g.get('pass') else 'FAIL'}** — " + ", ".join(f"{kk} {g[kk]}" for kk in g if kk != "pass"))
    w(f"- gates pass: **{v['gates_pass']}**")
    w("")
    # ceilings
    w("## Exact ceilings (Y units)")
    w("")
    w("| cell set | n cells | Δ\\*_menu | Δ\\*_full | V\\*_menu ask | V\\*_menu no-ask |")
    w("|---|---|---|---|---|---|")
    for cs in CELL_SETS:
        c = a["ceilings"].get(cs, {})
        if c.get("n_cells"):
            ys = a["y_scale"]
            w(f"| {cs} | {c['n_cells']} | {f4(c.get('mean_delta_menu_Y'))} | {f4(c.get('mean_delta_full_Y'))} | "
              f"{f4(c['mean_menu_ask'] / ys)} | {f4(c['mean_menu_noask'] / ys)} |")
    w("")
    # per key
    w("## Per (arm, regime)")
    w("")
    w("| arm | regime | n | mean Y (all) [95% CI] | mean Y (focal opens) | agreement | walk | questions / ep | regret (true) | zero-regret share |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for key in a["keys"]:
        arm, rg = key.split("|")
        s, sf = a["per_key"][key]["all"], a["per_key"][key]["focal_opens"]
        cci = s.get("ci95_normalized_net_gain")
        cs = f"{f4(s['mean_normalized_net_gain'])}" + (f" [{f4(cci[0])}, {f4(cci[1])}]" if cci else "")
        w(f"| {arm} | {rg} | {s['n']} | {cs} | {f4(sf.get('mean_normalized_net_gain'))} | {f3(s['agreement_rate'])} | {f3(s['walk_rate'])} | "
          f"{f3(s['mean_questions_asked'])} | {f4(s['mean_total_decision_regret_true'])} | {f3(s['zero_regret_decision_share_true'])} |")
    w("")
    # decoders in episode
    w("## Decoders in play (all cells)")
    w("")
    w("| arm | regime | decodes | answer log-loss (bits) | mean P(true type) after decode | exact filter's | confident wrong | episodes with true type zeroed | final P(true type) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for key in a["keys"]:
        arm, rg = key.split("|")
        dcs = a["per_key"][key]["all"].get("decode") or {}
        if not dcs.get("n_decodes"):
            continue
        w(f"| {arm} | {rg} | {dcs['n_decodes']} | {f3(dcs.get('mean_answer_logloss_bits'))} | {f3(dcs.get('mean_p_true_after'))} | "
          f"{f3(dcs.get('mean_p_true_exact_after'))} | {dcs.get('confident_wrong_decodes')} | {dcs.get('episodes_true_type_zeroed')} | {f3(dcs.get('mean_final_p_true'))} |")
    w("")
    # selectivity
    w("## Selectivity against the evaluator's reference (ASK strictly optimal under the a = 0 oracle at the exact belief)")
    w("")
    w("| arm | regime | asked / ASK-optimal | P(ask \\| optimal) | asked / legal-not-optimal | P(ask \\| not optimal) | difference [95% CI] |")
    w("|---|---|---|---|---|---|---|")
    for key, s in a["selectivity"].items():
        arm, rg = key.split("|")
        dci = s.get("difference_ci95")
        diff = "—" if s.get("difference") is None else f"{s['difference']:+.3f}" + (f" [{dci[0]:+.3f}, {dci[1]:+.3f}]" if dci else "")
        w(f"| {arm} | {rg} | {s['asked_when_optimal']}/{s['decisions_ask_optimal']} | {f3(s['p_ask_given_optimal'])} | "
          f"{s['asked_when_not_optimal']}/{s['decisions_ask_legal_not_optimal']} | {f3(s['p_ask_given_not_optimal'])} | {diff} |")
    w("")
    # agents
    ag = {key: a["per_key"][key]["all"].get("agent") for key in a["keys"] if a["per_key"][key]["all"].get("agent")}
    if ag:
        w("## Agent arms")
        w("")
        w("| arm | regime | decisions | well-formed | retried | positions chosen | tokens in / out |")
        w("|---|---|---|---|---|---|---|")
        for key, g in ag.items():
            arm, rg = key.split("|")
            tok = "—" if g.get("input_tokens") is None else f"{g['input_tokens']} / {g['output_tokens']}"
            w(f"| {arm} | {rg} | {g['decisions']} | {f3(g['well_formed_share'])} | {g['retried']} | {g['positions']} | {tok} |")
        w("")
    # contrasts
    w("## Paired contrasts in Y (cluster bootstrap, 95% percentile CI)")
    w("")
    w("| contrast | all cells | focal opens | opponent opens | captured fraction of Δ\\*_menu (focal opens) |")
    w("|---|---|---|---|---|")
    for name, c in a["contrasts"].items():
        cap = ((c["focal_opens"] or {}).get("captured_fraction") or {}).get("delta_menu")
        cap_s = "—" if not cap or cap.get("point") is None else f"{cap['point']:+.2f} [{cap['ci95'][0]:+.2f}, {cap['ci95'][1]:+.2f}]"
        w(f"| {name} | {ci(c['all'])} | {ci(c['focal_opens'])} | {ci(c['opponent_opens'])} | {cap_s} |")
    w("")
    w("### The same pairs in questions asked per episode")
    w("")
    w("| contrast | all cells | focal opens | opponent opens |")
    w("|---|---|---|---|")
    for name, c in a["contrasts_questions"].items():
        w(f"| {name} | {ci(c['all'])} | {ci(c['focal_opens'])} | {ci(c['opponent_opens'])} |")
    w("")
    if a.get("contrasts_answer_logloss"):
        w("### Decoder pairs in in-episode answer log-loss (bits; lower is better)")
        w("")
        w("| contrast | all cells | focal opens | opponent opens |")
        w("|---|---|---|---|")
        for name, c in a["contrasts_answer_logloss"].items():
            w(f"| {name} | {ci(c['all'])} | {ci(c['focal_opens'])} | {ci(c['opponent_opens'])} |")
        w("")
    # verdicts
    w("## Pre-registered rules and verdicts")
    w("")
    w(f"δ = {v['delta']} Y; captured minimum {v['captured_min']}; selectivity maximum {v['selectivity_max']}. Vocabulary in `verdicts_noul.py`.")
    w("")
    w("| rule | quantity | prediction | estimate [95% CI] | verdict |")
    w("|---|---|---|---|---|")
    for kk, r in v["rules"].items():
        e = r.get("estimate")
        s = "—" if not e or "point" not in e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        w(f"| {kk} | {r['quantity']} | {r['prediction']} | {s} | **{r['verdict']}** |")
    w("")
    p1 = v["rules"]["P1"]
    cap = p1.get("captured_fraction_of_delta_menu") or {}
    w("## Read-out (computed)")
    w("")
    e = p1.get("estimate")
    if e and "point" in e:
        w(f"- P1: {e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] against a ceiling of {f4(p1.get('ceiling_delta_menu_Y'))} → captured "
          f"{pct(cap.get('point'))}" + (f" [{pct(cap['ci95'][0])}, {pct(cap['ci95'][1])}]" if cap.get("ci95") else "") + f" → **{p1['verdict']}**.")
    for kk in ("P2", "P3", "P4", "P5_noul_answer", "P5_regex", "P8"):
        r = v["rules"].get(kk)
        if r and r.get("estimate") and "point" in r["estimate"]:
            ee = r["estimate"]
            w(f"- {kk}: {ee['point']:+.4f} [{ee['ci95'][0]:+.4f}, {ee['ci95'][1]:+.4f}] → **{r['verdict']}**.")
    for kk, r in v["rules"].items():
        if kk.startswith("P6") and r.get("estimate"):
            ee = r["estimate"]
            w(f"- {kk}: P(ask | optimal) {f3(r.get('p_ask_given_optimal'))}, P(ask | not optimal) {f3(r.get('p_ask_given_not_optimal'))}, "
              f"difference {ee['point']:+.3f} [{ee['ci95'][0]:+.3f}, {ee['ci95'][1]:+.3f}] → **{r['verdict']}**.")
    for kk, r in v["rules"].items():
        if kk.startswith("P7") and r.get("estimate") and "point" in r["estimate"]:
            ee = r["estimate"]
            w(f"- {kk}: {ee['point']:+.4f} [{ee['ci95'][0]:+.4f}, {ee['ci95'][1]:+.4f}] → **{r['verdict']}**.")
    desc = v.get("descriptive", {})
    oh = desc.get("P1_other_hedge")
    if oh:
        w(f"- The P1 contrast at the other hedge: {oh['point']:+.4f} [{oh['ci95'][0]:+.4f}, {oh['ci95'][1]:+.4f}] (descriptive).")
    sj = desc.get("sonnet_judge_vs_regex_oblique")
    if sj:
        w(f"- The Sonnet judge as decoder vs the parser, oblique: {sj['point']:+.4f} [{sj['ci95'][0]:+.4f}, {sj['ci95'][1]:+.4f}] (descriptive).")
    w("")
    w("## Limits")
    w("")
    w("- Stage 1's opponent is the registered T1 kernel: it never lies and its moves are typed; only its answers are text. The strategic "
      "responder, the realizer and a Noul over offers are Stage 2.")
    w("- The six-ranking type space makes the hypothesis bins free, as in 20Q; an open type space is not addressed here.")
    w("- The `oblique` verbalizer may let priorities show beyond the yes/no, so the binary-answer ceiling bounds the answer decoders only; "
      "`noul_type` can in principle exceed it there.")
    w("- The agents' menus are built from a belief decoded by `" + a["agent_menu_decoder"] + "`; the agent-vs-planner contrast isolates the chooser "
      "on identical menus, not an agent with no scaffolding.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    a = json.load(open(os.path.join(args.dir, "analysis.json")))
    v = json.load(open(os.path.join(args.dir, "verdicts.json")))
    text = render(args.dir, a, v)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
