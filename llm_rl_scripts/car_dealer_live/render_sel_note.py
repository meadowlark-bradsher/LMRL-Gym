"""Render the CD-SEL-1 results note (SEL_RESULTS.md) from the study's analysis.json
(analyze_phase1.py) and verdicts.json (verdicts_sel.py). Every number is formatted
from the JSON; nothing is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.render_sel_note \
        --dir results/live_sel_jev_h4_cq0_20261030 --out llm_rl_scripts/car_dealer_live/SEL_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .render_phase1_note import CELL_SETS, ci, f3, f4, section

SHOWN_COLUMNS = (("ask", "questions on the menu"), ("show_eig", "EIG per option"), ("show_belief", "posterior over the opponent's rankings, in words"),
                 ("accept_annotation", "P(accept) printed on every offer"))


def yesno(v) -> str:
    if v is None or v is False:
        return "no"
    if v is True:
        return "yes"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    a = json.load(open(os.path.join(args.dir, "analysis.json")))
    v = json.load(open(os.path.join(args.dir, "verdicts.json")))
    lines = []
    w = lines.append
    w("# CD-SEL-1 — Jev shown what the history implies (live Car-Dealer, design Phase 2 selector arms)")
    w("")
    w("_Rendered by `render_sel_note.py` from `analysis.json` (`analyze_phase1.py`) and `verdicts.json` (`verdicts_sel.py`) in "
      f"`{args.dir}`; every number is formatted from JSON, none is typed. Rules registered in `PREREGISTRATION_SEL.md`._")
    w("")
    w("Jev has seen the full event history (every proposal, answer, rejection, hold) in its state text since Phase 1. This study "
      "shows it, in separately named arms, what that history implies: the matched posterior over the opponent's six priority "
      "rankings in words (`belief`), the posterior probability that each offer is accepted (`pacc`, the B_passive selector), the same "
      "number under the never-updated uniform prior (`prior`, A_prior), a constant on every offer (`placebo`, P_placebo), and belief + "
      "posterior P(accept) together (`both`, with and without questions on the menu). The scripted `exact_ask` / `exact_noask` / "
      "`random` arms run in the same cells.")
    w("")
    w("## What each arm is shown")
    w("")
    w("| arm | " + " | ".join(c for _, c in SHOWN_COLUMNS) + " |")
    w("|---|" + "---|" * len(SHOWN_COLUMNS))
    specs = a.get("jev_arm_specs") or {}
    for arm in a["arms"]:
        if arm in specs:
            w(f"| {arm} | " + " | ".join(yesno(specs[arm].get(k, False if k != 'accept_annotation' else None)) for k, _ in SHOWN_COLUMNS) + " |")
    w("")
    w(f"The placebo constant is {f3(json.load(open(os.path.join(args.dir, 'summary.json'))).get('placebo_p_accept'))} on every offer. "
      "Options are lettered and shuffled in every arm; no arm sees tags, Q\\* or (except `eig` arms) EIG.")
    w("")
    # gates and drift
    w("## Sanity gates and the drift check")
    w("")
    for name, g in v["gates"].items():
        w(f"- `{name}`: **{'pass' if g.get('pass') else 'FAIL'}** — " + ", ".join(f"{k} {g[k]}" for k in g if k != "pass"))
    w(f"- gates pass: **{v['gates_pass']}**")
    d = (v.get("checks") or {}).get("plain_arm_drift")
    if d:
        w(f"- Drift check (reported, not a gate): the in-study `jev_ask_plain` mean Y {f4(d['study_mean_Y'])} vs the frozen Phase 1 run's "
          f"{f4(d['phase1_mean_Y'])} [{f4(d['phase1_ci95'][0])}, {f4(d['phase1_ci95'][1])}] — {'inside' if d['inside_phase1_ci'] else 'OUTSIDE'} the Phase 1 CI; "
          f"walk rate {f3(d['study_walk_rate'])} vs {f3(d['phase1_walk_rate'])}; questions / episode {f3(d['study_questions_per_episode'])} vs "
          f"{f3(d['phase1_questions_per_episode'])}.")
    w("")
    # the Phase 1 renderer's generic section (per arm, contrasts, ceilings, selectivity, choices, positions)
    section(args.dir, a, w)
    # walk-rate contrasts
    cw = a.get("contrasts_walk") or {}
    if cw:
        w("### Paired contrasts in walk rate (episodes ending with a walk-out; cluster bootstrap, 95% percentile CI)")
        w("")
        w("| contrast | all cells | focal opens | opponent opens |")
        w("|---|---|---|---|")
        for name, c in cw.items():
            w(f"| {name} | {ci(c['all'])} | {ci(c['focal_opens'])} | {ci(c['opponent_opens'])} |")
        w("")
    # verdicts
    w("## Pre-registered rules and verdicts")
    w("")
    w(f"Materiality δ = {v['delta']} Y. Primary cells: {v['primary_cells']}. Verdict vocabulary as in `verdicts_live.py`.")
    w("")
    w("| rule | quantity | prediction | estimate [95% CI] | verdict |")
    w("|---|---|---|---|---|")
    for k, r in v["rules"].items():
        e = r.get("estimate")
        s = "—" if not e or "point" not in e else f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
        w(f"| {k} | {r['quantity']} | {r['prediction']} | {s} | **{r['verdict']}** |")
    w("")
    # computed read-out
    w("## Read-out (computed)")
    w("")
    p1 = v["rules"]["P1"]
    e = p1.get("estimate")
    if e and "point" in e:
        w(f"- P1 `{v['primary_contrast']}` ({v['primary_cells']} cells): {e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] → **{p1['verdict']}**.")
    gc = p1.get("gap_closed_point")
    pe, be = p1.get("plain_minus_exact_ask"), p1.get("both_minus_exact_ask")
    if gc is not None and pe and be:
        w(f"- Gap to the exact planner: plain {pe['point']:+.4f} [{pe['ci95'][0]:+.4f}, {pe['ci95'][1]:+.4f}], both {be['point']:+.4f} "
          f"[{be['ci95'][0]:+.4f}, {be['ci95'][1]:+.4f}]; the shown information closes {gc:.1%} of the plain gap (point estimate; no interval on the ratio).")
    desc = v.get("descriptive", {})
    my = desc.get("mean_Y_all", {})
    if my:
        w("- Mean Y, all cells: " + ", ".join(f"{k} {f4(x)}" for k, x in my.items() if x is not None) + ".")
    wr = (v["rules"].get("S8") or {}).get("walk_rate") or {}
    if wr:
        w("- Walk rate: " + ", ".join(f"{k} {f3(x)}" for k, x in wr.items() if x is not None) + ".")
    qa = (v["rules"].get("S5") or {}).get("questions_per_episode") or {}
    if qa:
        w("- Questions per episode: " + ", ".join(f"{k} {f3(x)}" for k, x in qa.items() if x is not None) + ".")
    for key, label in (("prior_vs_placebo", "jev_ask_prior − jev_ask_placebo"), ("both_vs_pacc", "jev_ask_both − jev_ask_pacc"), ("both_vs_belief", "jev_ask_both − jev_ask_belief")):
        c = desc.get(key)
        if c:
            w(f"- {label} (all cells, descriptive): {c['point']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}].")
    w("")
    w("## Regenerate")
    w("")
    w("```bash")
    w(f"bash {args.dir}/REGENERATE.sh")
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
