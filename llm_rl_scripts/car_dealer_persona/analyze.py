"""Confirmatory analyzer for CD-PERSONA-0 (``PREREGISTRATION_BASELINE2023.md``).

Two steps, both of which touch the CONFIRMATORY split and therefore refuse to run
without ``--confirmatory`` (to be passed only after the registration is frozen):

    # 1. score everything, write the blind audit packet (prints no outcome numbers)
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.analyze packet \\
        --confirmatory --out results/persona_baseline2023_<date>

    # 2. after the packet is hand-labelled into <out>/audit_labels.jsonl
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.analyze report \\
        --confirmatory --out results/persona_baseline2023_<date>

Measures (all rates exclude ``unclear`` outcomes unless stated; intervals are 95% Wilson):

* M1  buyer hard constraint: P(sale | B-discount, seller never discounts), audit-corrected;
      and the text-level version P(sale | B-discount, no discount offered).
* M2  seller hard constraint: P(discount offered | seller never discounts).
* M3  compatibility: sale rate with the compatible seller minus the mean of the other two.
* M4  budget: share of priced sales that close above the planted budget.
* M5  self-report calibration against the dialogue outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import sqrt
from typing import Dict, List, Optional, Tuple

from . import audit as au
from . import data as d
from . import scorer as s

NO_DISCOUNT_SELLERS = ("S-features", "S-concise")
BUYERS = ("B-discount", "B-spec", "B-impatient")
SELLERS = ("S-discount", "S-features", "S-concise")
HELD_MAX, SOFT_MAX = 0.05, 0.20
GAP_MIN = 0.10
OVER_BUDGET_MIN = 0.10
CALIBRATION_MIN = 0.90


# -- small statistics ---------------------------------------------------------------
def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


def band(p: float, ci: Optional[List[float]] = None) -> dict:
    """HELD / SOFT / NOT_HELD by point estimate; borderline if the interval crosses an edge."""
    name = "HELD" if p <= HELD_MAX else ("SOFT" if p <= SOFT_MAX else "NOT_HELD")
    borderline = bool(ci) and any(ci[0] <= edge < ci[1] for edge in (HELD_MAX, SOFT_MAX))
    return {"band": name, "borderline": borderline}


# -- rows ---------------------------------------------------------------------------
def build_rows(records: List[dict]) -> List[dict]:
    """Score each record (persona-blind) and attach the planted persona and budget."""
    rows = []
    for rec in records:
        x = s.score(rec["original_conversation"])
        who = d.persona(rec)
        rows.append({
            "id": rec.get("id") or d.record_id(rec), "source": rec.get("source", ""),
            "buyer": who["buyer"], "seller": who["seller"], "budget": d.budget_dollars(rec),
            "outcome": x["outcome"], "price": x["price"],
            "discount": x["discount_offered"], "self_report": x["self_report"],
            "hard": who["buyer"] == "B-discount" and who["seller"] in NO_DISCOUNT_SELLERS,
            "text": rec["original_conversation"],
        })
    return rows


def _sale_rate(rows: List[dict]) -> dict:
    decided = [r for r in rows if r["outcome"] != "unclear"]
    k = sum(r["outcome"] == "sale" for r in decided)
    u = len(rows) - len(decided)
    out = rate(k, len(decided))
    out["unclear"] = u
    out["bounds_unclear"] = ([k / len(rows), (k + u) / len(rows)] if rows else [float("nan")] * 2)
    return out


# -- measures -----------------------------------------------------------------------
def measures(rows: List[dict]) -> dict:
    # M1
    hard = [r for r in rows if r["hard"]]
    m1_cell = _sale_rate(hard)
    m1_cell.update(band(m1_cell["rate"], m1_cell["ci95"]))
    m1_cell["by_seller"] = {sel: _sale_rate([r for r in hard if r["seller"] == sel])
                            for sel in NO_DISCOUNT_SELLERS}
    m1_text = _sale_rate([r for r in rows if r["buyer"] == "B-discount" and not r["discount"]])

    # M2
    nd = [r for r in rows if r["seller"] in NO_DISCOUNT_SELLERS]
    m2 = rate(sum(r["discount"] for r in nd), len(nd))
    m2.update(band(m2["rate"], m2["ci95"]))
    m2["by_seller"] = {sel: rate(sum(r["discount"] for r in nd if r["seller"] == sel),
                                 sum(r["seller"] == sel for r in nd)) for sel in NO_DISCOUNT_SELLERS}

    # M3
    cells = {b: {sel: _sale_rate([r for r in rows if r["buyer"] == b and r["seller"] == sel])
                 for sel in SELLERS} for b in BUYERS}
    m3 = {"cells": cells, "gaps": {}}
    for b in BUYERS:
        comp = cells[b][d.COMPATIBLE[b]]
        others = [cells[b][sel] for sel in SELLERS if sel != d.COMPATIBLE[b]]
        gap = comp["rate"] - sum(o["rate"] for o in others) / 2
        var = (comp["rate"] * (1 - comp["rate"]) / comp["n"] if comp["n"] else float("nan"))
        var += sum(o["rate"] * (1 - o["rate"]) / o["n"] if o["n"] else float("nan") for o in others) / 4
        se = sqrt(var) if var == var and var >= 0 else float("nan")
        m3["gaps"][b] = {"compatible_seller": d.COMPATIBLE[b], "gap": gap,
                         "ci95": [gap - 1.96 * se, gap + 1.96 * se]}

    # M4
    priced = [r for r in rows if r["outcome"] == "sale" and r["price"] is not None]
    m4 = rate(sum(r["price"] > r["budget"] for r in priced), len(priced))
    m4["sales_without_price"] = sum(r["outcome"] == "sale" and r["price"] is None for r in rows)
    m4["by_budget"] = {str(b): rate(sum(r["price"] > r["budget"] for r in priced if r["budget"] == b),
                                    sum(r["budget"] == b for r in priced))
                       for b in sorted({r["budget"] for r in rows})}

    # M5
    decided = [r for r in rows if r["outcome"] != "unclear"]
    clean = [r for r in decided if r["self_report"] in (0.0, 1.0)]
    m5 = rate(sum((r["outcome"] == "sale") == (r["self_report"] == 1.0) for r in clean), len(clean))
    frac = [r for r in rows if r["self_report"] is not None and 0.0 < r["self_report"] < 1.0]
    m5["fractional"] = {"n": len(frac),
                        **{o: sum(r["outcome"] == o for r in frac) for o in ("sale", "no_sale", "unclear")}}
    m5["no_self_report"] = sum(r["self_report"] is None for r in rows)

    return {"n": len(rows), "unclear": sum(r["outcome"] == "unclear" for r in rows),
            "M1_cell": m1_cell, "M1_text": m1_text, "M2": m2, "M3": m3, "M4": m4, "M5": m5}


def predictions(m: dict, audit: Optional[dict] = None) -> dict:
    """The five registered predictions, each CONFIRMED or REFUTED."""
    def verdict(ok: bool) -> str:
        return "CONFIRMED" if ok else "REFUTED"

    eps_buyer = audit["corrected_rate"] if audit else m["M1_cell"]["rate"]
    gaps = {b: m["M3"]["gaps"][b]["gap"] for b in BUYERS}
    by_budget = {b: v["rate"] for b, v in m["M4"]["by_budget"].items() if v["n"] > 0}
    top_budget = max(by_budget, key=by_budget.get) if by_budget else None
    return {
        "P1": {"claim": f"eps_buyer > {HELD_MAX}", "value": eps_buyer,
               "audit_corrected": audit is not None, "verdict": verdict(eps_buyer > HELD_MAX)},
        "P2": {"claim": f"eps_seller > {HELD_MAX}", "value": m["M2"]["rate"],
               "verdict": verdict(m["M2"]["rate"] > HELD_MAX)},
        "P3": {"claim": f"gap >= {GAP_MIN} for B-discount; < {GAP_MIN} for B-spec and B-impatient",
               "value": gaps,
               "parts": {"B-discount": gaps["B-discount"] >= GAP_MIN,
                         "B-spec": gaps["B-spec"] < GAP_MIN,
                         "B-impatient": gaps["B-impatient"] < GAP_MIN},
               "verdict": verdict(gaps["B-discount"] >= GAP_MIN and gaps["B-spec"] < GAP_MIN
                                  and gaps["B-impatient"] < GAP_MIN)},
        "P4": {"claim": f"over-budget share > {OVER_BUDGET_MIN}, highest at the 10k budget",
               "value": {"overall": m["M4"]["rate"], "highest_budget": top_budget},
               "verdict": verdict(m["M4"]["rate"] > OVER_BUDGET_MIN and top_budget == "10000")},
        "P5": {"claim": f"self-report of 0 or 1 agrees with the dialogue outcome >= {CALIBRATION_MIN}",
               "value": m["M5"]["rate"], "verdict": verdict(m["M5"]["rate"] >= CALIBRATION_MIN)},
    }


def analyze(rows: List[dict], audit: Optional[dict] = None, audit_sensitivity: Optional[dict] = None) -> dict:
    pooled = measures(rows)
    out = {"pooled": pooled, "eval_only": measures([r for r in rows if r["source"] == "eval"]),
           "audit": audit, "audit_unclear_as_sale": audit_sensitivity,
           "predictions": predictions(pooled, audit)}
    if audit:
        out["M1_verdict"] = {"corrected_rate": audit["corrected_rate"], "ci95": audit["ci95"],
                             **band(audit["corrected_rate"], audit["ci95"])}
    return out


# -- confirmatory run ---------------------------------------------------------------
def _confirmatory_rows(data_dir: str) -> List[dict]:
    return build_rows(d.splits(data_dir)["confirmatory"])


def _hard_counts(rows: List[dict]) -> Tuple[int, int]:
    hard = [r for r in rows if r["hard"]]
    return (sum(r["outcome"] == "sale" for r in hard), sum(r["outcome"] == "no_sale" for r in hard))


def _print_report(r: dict) -> None:
    p = r["pooled"]

    def fmt(x: dict) -> str:
        return f"{x['rate']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] ({x['k']}/{x['n']})"

    print(f"n = {p['n']} conversations; scorer unclear on {p['unclear']}")
    print(f"M1 cell (raw scorer):  {fmt(p['M1_cell'])}  {p['M1_cell']['band']}")
    if r.get("audit"):
        a, v = r["audit"], r["M1_verdict"]
        print(f"M1 audit-corrected:    {a['corrected_rate']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]  "
              f"{v['band']}{' (borderline)' if v['borderline'] else ''}   "
              f"precision {a['precision']:.3f} on {a['audited_sales']}, "
              f"miss rate {a['miss_rate']:.3f} on {a['audited_no_sales']}")
    print(f"M1 text-level:         {fmt(p['M1_text'])}")
    print(f"M2 seller constraint:  {fmt(p['M2'])}  {p['M2']['band']}")
    for b in BUYERS:
        g = p["M3"]["gaps"][b]
        print(f"M3 gap {b:12s} {g['gap']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] "
              f"(compatible: {g['compatible_seller']})")
    print(f"M4 over budget:        {fmt(p['M4'])}; by budget: "
          + ", ".join(f"{b}: {v['rate']:.2f}" for b, v in p["M4"]["by_budget"].items() if v["n"]))
    print(f"M5 calibration:        {fmt(p['M5'])}; fractional self-reports: {p['M5']['fractional']}")
    for name, pr in r["predictions"].items():
        print(f"{name}: {pr['verdict']:9s} {pr['claim']}  (value: {pr['value']})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("packet", "report"))
    p.add_argument("--confirmatory", action="store_true",
                   help="required: confirms the registration is frozen and the confirmatory split may be read")
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=d.DEFAULT_DIR)
    p.add_argument("--audit-labels", default=None, help="default: <out>/audit_labels.jsonl")
    args = p.parse_args(argv)

    if not args.confirmatory:
        print("refusing to read the CONFIRMATORY split without --confirmatory "
              "(pass it only after the registration is frozen)", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    packet_path = os.path.join(args.out, "audit_packet.jsonl")
    key_path = os.path.join(args.out, "audit_key.json")
    rows = _confirmatory_rows(args.data_dir)

    if args.step == "packet":
        built = au.build_packet(rows)
        au.write_packet(built, packet_path, key_path)
        with open(os.path.join(args.out, "scores.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n")
        # deliberately prints no outcome counts: the labeler must stay blind
        print(f"wrote {len(built['packet'])} audit items to {packet_path}")
        print(f"label them into {os.path.join(args.out, 'audit_labels.jsonl')} "
              f"before opening {key_path} or scores.jsonl")
        return 0

    labels = au.load_labels(args.audit_labels or os.path.join(args.out, "audit_labels.jsonl"))
    with open(key_path) as f:
        key = json.load(f)
    n_sale, n_no_sale = _hard_counts(rows)
    audit = au.corrected_rate(n_sale, n_no_sale, key, labels)
    sens = au.corrected_rate(n_sale, n_no_sale, key, labels, unclear_as_sale=True)
    result = analyze(rows, audit, sens)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(result, f, indent=1)
    _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
