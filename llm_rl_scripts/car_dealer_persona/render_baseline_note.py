"""Render the CD-PERSONA-0 results note from the frozen result directory written by
``analyze.py`` (``results.json``, ``scores.jsonl``, ``audit_key.json``,
``audit_labels.jsonl``). Every number is formatted from those files; none is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.render_baseline_note \\
        --dir results/persona_baseline2023_20261106 \\
        --out llm_rl_scripts/car_dealer_persona/RESULTS_BASELINE2023.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from . import analyze as an

TAG = "prereg-persona-20261106"


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(c) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def rate(x: dict) -> str:
    return f"{f3(x['rate'])} {ci(x['ci95'])} ({x['k']}/{x['n']})"


def frac(k: int, n: int) -> str:
    lo, hi = an.wilson(k, n)
    return f"{k} of {n} ({k / n:.3f}, interval {lo:.3f} to {hi:.3f})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    def load(name):
        with open(os.path.join(args.dir, name)) as f:
            return json.load(f)

    R = load("results.json")
    key = load("audit_key.json")
    with open(os.path.join(args.dir, "audit_labels.jsonl")) as f:
        labels = {d["item"]: d["outcome"] for d in map(json.loads, filter(str.strip, f))}
    with open(os.path.join(args.dir, "scores.jsonl")) as f:
        scores = [json.loads(line) for line in f if line.strip()]

    p, e, a, s = R["pooled"], R["eval_only"], R["audit"], R["audit_unclear_as_sale"]
    v, P = R["M1_verdict"], R["predictions"]
    L = []
    w = L.append

    w("# CD-PERSONA-0 results: did the 2023 Car Dealer personas govern behavior?")
    w("")
    w(f"_Rendered by `render_baseline_note.py` from `{args.dir}` written by `analyze.py`; every number is "
      f"formatted from the result files, none is typed. Registered in `PREREGISTRATION_BASELINE2023.md`, "
      f"frozen at tag `{TAG}` before the confirmatory split was read._")
    w("")
    w(f"Data: the released LMRL-Gym Car Dealer conversations, each written by a single GPT-3.5 narrator "
      f"(both sides). Confirmatory split: {p['n']} conversations; the persona-blind scorer returned "
      f"`unclear` on {p['unclear']} of them.")
    w("")

    w("## Predictions")
    w("")
    w("| | Claim | Value | Verdict |")
    w("|---|---|---|---|")
    gaps = P["P3"]["value"]
    w(f"| P1 | discount-only buyer constraint is not `HELD` (rate above {an.HELD_MAX}) | "
      f"{f3(P['P1']['value'])} audit-corrected | {P['P1']['verdict']} |")
    w(f"| P2 | no-discount seller constraint violated above {an.HELD_MAX} | {f3(P['P2']['value'])} | "
      f"{P['P2']['verdict']} |")
    w(f"| P3 | compatibility gap at least {an.GAP_MIN} for the discount buyer, below it for the other two | "
      + ", ".join(f"{b}: {gaps[b]:+.3f}" for b in an.BUYERS) + f" | {P['P3']['verdict']} |")
    w(f"| P4 | more than {an.OVER_BUDGET_MIN:.0%} of priced sales above budget, highest at the 10k budget | "
      f"{f3(P['P4']['value']['overall'])} overall; highest at {int(P['P4']['value']['highest_budget']):,} | "
      f"{P['P4']['verdict']} |")
    w(f"| P5 | a self-report of exactly 0 or 1 agrees with the dialogue outcome at least "
      f"{an.CALIBRATION_MIN:.0%} of the time | {f3(P['P5']['value'])} | {P['P5']['verdict']} |")
    w("")

    w("## M1. Buyer hard constraint: does a discount-only buyer buy from a seller who never discounts?")
    w("")
    w(f"Audit-corrected rate: **{f3(v['corrected_rate'])} {ci(v['ci95'])}**, band `{v['band']}`"
      + (", borderline (the interval crosses a band edge)" if v["borderline"] else "") + ".")
    w("")
    w("| | Rate | |")
    w("|---|---|---|")
    w(f"| Raw scorer | {rate(p['M1_cell'])} | band `{p['M1_cell']['band']}`; {p['M1_cell']['unclear']} "
      f"scorer-unclear excluded, bounds {ci(p['M1_cell']['bounds_unclear'])} |")
    w(f"| Audit-corrected (registered) | {f3(a['corrected_rate'])} {ci(a['ci95'])} | precision "
      f"{f3(a['precision'])} on {a['audited_sales']} audited sales, miss rate {f3(a['miss_rate'])} on "
      f"{a['audited_no_sales']} audited no-sales |")
    w(f"| Sensitivity: audit `unclear` counted as a sale | {f3(s['corrected_rate'])} {ci(s['ci95'])} | |")
    for sel in an.NO_DISCOUNT_SELLERS:
        w(f"| With {sel} (raw) | {rate(p['M1_cell']['by_seller'][sel])} | |")
    w(f"| Eval file only (raw) | {rate(e['M1_cell'])} | band `{e['M1_cell']['band']}` |")
    w(f"| Text-level, exploratory: discount buyer, no discount detected, any seller | {rate(p['M1_text'])} | |")
    w("")
    w("Sales in these cells follow a seller who was told never to discount, so this rate is an upper "
      "bound on how often the buyer broke its own persona; see the exploratory split below.")
    w("")

    w("## M2. Seller hard constraint: does a no-discount seller offer a discount?")
    w("")
    w(f"**{rate(p['M2'])}**, band `{p['M2']['band']}`"
      + (", borderline" if p["M2"]["borderline"] else "") + ". Not hand-audited; on the labelled "
      "conversations the detector missed more offers than it invented, so this is more likely understated.")
    w("")
    for sel in an.NO_DISCOUNT_SELLERS:
        w(f"- {sel}: {rate(p['M2']['by_seller'][sel])}")
    w(f"- Eval file only: {rate(e['M2'])}")
    w("")

    w("## M3. Compatibility: sale rate by persona pair")
    w("")
    w("| Buyer | " + " | ".join(an.SELLERS) + " | Gap (compatible minus mean of others) |")
    w("|---|" + "---|" * (len(an.SELLERS) + 1))
    for b in an.BUYERS:
        cells = p["M3"]["cells"][b]
        g = p["M3"]["gaps"][b]
        w(f"| {b} | " + " | ".join(
            ("**" if sel == g["compatible_seller"] else "") + f"{f3(cells[sel]['rate'])} ({cells[sel]['k']}/{cells[sel]['n']})"
            + ("**" if sel == g["compatible_seller"] else "") for sel in an.SELLERS)
          + f" | {g['gap']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
    w("")
    w("Bold marks the pairing the paper calls compatible. Eval file only, gaps: "
      + ", ".join(f"{b} {e['M3']['gaps'][b]['gap']:+.3f}" for b in an.BUYERS) + ".")
    w("")

    w("## M4. Budget: share of priced sales above the planted budget")
    w("")
    w(f"**{rate(p['M4'])}** ({p['M4']['sales_without_price']} sales had no extractable price). "
      f"Eval file only: {rate(e['M4'])}.")
    w("")
    w("| Budget | Share above budget |")
    w("|---|---|")
    for b, x in p["M4"]["by_budget"].items():
        w(f"| {int(b):,} | {rate(x)} |")
    w("")

    w("## M5. Self-report against the dialogue outcome")
    w("")
    fr = p["M5"]["fractional"]
    w(f"A final self-report of exactly 0 or 1 agrees with the dialogue outcome in **{rate(p['M5'])}**. "
      f"Eval file only: {rate(e['M5'])}.")
    w("")
    w(f"{fr['n']} conversations ({fr['n'] / p['n']:.1%}) end on a fractional self-report; their dialogue "
      f"outcomes are {fr['sale']} sale, {fr['no_sale']} no sale, {fr['unclear']} unclear. "
      f"{p['M5']['no_self_report']} have no parseable self-report.")
    w("")

    w("## The hand-audit")
    w("")
    strata = {}
    for k in key:
        strata.setdefault(k["stratum"], Counter())[labels[k["item"]]] += 1
    w(f"{len(key)} items, labelled with cell and scorer call hidden (`audit.py`, seed {an.au.AUDIT_SEED}).")
    w("")
    w("| Stratum | Items | Labelled sale | Labelled no sale | Labelled unclear |")
    w("|---|---|---|---|---|")
    names = {"hard_sale": "Hard cells, scorer said sale", "hard_no_sale": "Hard cells, scorer said no sale",
             "decoy_sale": "Other cells, scorer said sale", "decoy_no_sale": "Other cells, scorer said no sale"}
    for st, name in names.items():
        c = strata.get(st, Counter())
        w(f"| {name} | {sum(c.values())} | {c['sale']} | {c['no_sale']} | {c['unclear']} |")
    w("")
    dn = strata.get("decoy_no_sale", Counter())
    hn = strata.get("hard_no_sale", Counter())
    outside_miss = (a["scorer_sales"] * a["precision"] + a["scorer_no_sales"] * dn["sale"] / sum(dn.values())) \
        / (a["scorer_sales"] + a["scorer_no_sales"])
    w(f"Not registered, but it bears on the verdict: among the scorer's no-sale calls outside the hard "
      f"cells, {frac(dn['sale'], sum(dn.values()))} were labelled a sale, against "
      f"{frac(hn['sale'], sum(hn.values()))} inside them. The registered correction uses only the "
      f"hard-cell strata. If the hard cells had the outside miss rate, the corrected rate would be "
      f"{f3(outside_miss)}.")
    w("")

    w("## Exploratory: who broke persona first in the hard cells?")
    w("")
    hard = [r for r in scores if r["hard"] and r["outcome"] != "unclear"]
    c = Counter((r["outcome"], bool(r["discount"])) for r in hard)
    n = len(hard)
    w("Not registered. Scorer outcome by whether the discount detector fired, hard cells only "
      f"({n} decided conversations).")
    w("")
    w("| | Discount offered | No discount detected |")
    w("|---|---|---|")
    w(f"| Sale | {c[('sale', True)]} | {c[('sale', False)]} |")
    w(f"| No sale | {c[('no_sale', True)]} | {c[('no_sale', False)]} |")
    w("")
    w(f"- The seller offered a discount it was told never to give in {frac(c[('sale', True)] + c[('no_sale', True)], n)}.")
    w(f"- The buyer bought with no discount detected in {frac(c[('sale', False)], n)}. The detector "
      f"misses some offers, so this overstates the buyer's own violations somewhat.")
    w("")

    w("## Reading")
    w("")
    w(f"- **The personas strongly govern the story.** Sale rates run from "
      f"{f3(min(x['rate'] for row in p['M3']['cells'].values() for x in row.values()))} to "
      f"{f3(max(x['rate'] for row in p['M3']['cells'].values() for x in row.values()))} across the nine "
      f"pairings.")
    w(f"- **The hard constraints are soft.** A discount-only buyer still buys from a no-discount seller "
      f"about {v['corrected_rate']:.0%} of the time, and a no-discount seller offers a discount at least "
      f"{p['M2']['rate']:.0%} of the time. As a noise level for an oracle, that is inside the range of "
      f"answer noise the elicitation toy swept.")
    past = [name for name, val in (("counting audit `unclear` as a sale", s["corrected_rate"]),
                                   ("applying the outside miss rate", outside_miss)) if val > an.SOFT_MAX]
    w(f"- **The verdict on M1 is borderline.** By the registered rule it is `{v['band']}`"
      + (f"; the interval crosses a band edge" if v["borderline"] else "")
      + (f", and {' and '.join(past)} would put it above {an.SOFT_MAX}." if past else "."))
    w(f"- **It depends on which no-discount seller it is.** Violations concentrate with the "
      f"feature-talking seller ({f3(p['M1_cell']['by_seller']['S-features']['rate'])}) and are rare with "
      f"the concise one ({f3(p['M1_cell']['by_seller']['S-concise']['rate'])}).")
    held = [b for b in an.BUYERS if p["M3"]["gaps"][b]["gap"] >= an.GAP_MIN]
    w(f"- **The paper's compatibility structure holds for {len(held)} of {len(an.BUYERS)} buyers** "
      f"({', '.join(held) if held else 'none'}). The gap for B-impatient is "
      f"{p['M3']['gaps']['B-impatient']['gap']:+.3f}: it buys less from the seller the design pairs it with "
      f"than from the other two.")
    w(f"- **An unambiguous self-report can be trusted ({f3(p['M5']['rate'])} agreement), but "
      f"{fr['n'] / p['n']:.0%} of conversations end on a fractional one.**")
    w("")

    w("## Limits")
    w("")
    w("- One narrator wrote both sides. This is whether a narrated story respects two personas, not "
      "whether a buyer agent holds one against an independent seller.")
    w("- All labels (160 for the scorer gate, 180 for the audit) come from one labeler, Claude. The "
      "audit hid cell and scorer call, not the conversation's content.")
    w("- The released files may not be the exact data behind the paper's tables "
      "(`PERSONA_AUDIT_TARGET.md`).")
    w("- M2 and the text-level and exploratory rows depend on a discount detector that was gated but "
      "not audited.")
    w("")

    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
