"""Experiment B analysis — semantic salience (narrative x food-label 2x2).

B1 neutral/generic, B2 neutral/milk, B3 narrative/generic, B4 narrative/milk; all
reactive, no goal. The question is whether pickup/eat/fridge behavior is driven by
the perishability narrative, the word "milk", or their interaction — and whether
any pickup increase is *selective* for the target or just generalized food
activation (design 6).

All measures are opportunity-conditioned: a rate's denominator is the number of
steps where that action was actually admissible (logged), never raw counts, so a
condition that simply encountered more opportunities isn't mistaken for a stronger
tendency. Rates are pooled over opportunities with Wilson 95% intervals; the 2x2
marginal effects (label, narrative, interaction) are pooled differences.

    python -m llm_rl_scripts.text_nav.jev.analyze_exp_b <run_dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from typing import Dict, List

CONDITIONS = ["B1", "B2", "B3", "B4"]
# factor levels: narrative N (B3,B4 = N1) ; label L (B2,B4 = L1=milk)
N1 = {"B3", "B4"}
L1 = {"B2", "B4"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def load(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "episodes.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


# numerator, denominator keys for each opportunity-conditioned rate
RATES = {
    "take_target|offered": ("take_target", "opp_target_offered"),
    "target|both_offered": ("chose_target_when_both", "opp_target_vs_distractor"),
    "eat|offered": ("chose_eat_when_offered", "opp_eat_offered"),
}


def totals(eps: List[dict]) -> Dict[str, Dict[str, int]]:
    out = {c: collections.Counter() for c in CONDITIONS}
    for r in eps:
        c = r["condition"]
        if c in out:
            for k, v in r["diagnostics"].items():
                out[c][k] += v
            out[c]["episodes"] += 1
    return out


def analyze(run_dir: str) -> dict:
    eps = load(run_dir)
    tot = totals(eps)

    per_cond = {}
    for c in CONDITIONS:
        t = tot[c]
        rates = {}
        for name, (num, den) in RATES.items():
            rates[name] = {"k": t[num], "n": t[den], "wilson": wilson(t[num], t[den])}
        per_cond[c] = {
            "episodes": t["episodes"],
            "take_target": t["take_target"], "take_distractor": t["take_distractor"],
            "opp_target_offered": t["opp_target_offered"],
            "opp_target_vs_distractor": t["opp_target_vs_distractor"],
            "opp_eat_offered": t["opp_eat_offered"],
            "open_fridge": t["open_fridge"], "insert": t["insert"],
            "rates": rates,
        }

    def pooled_rate(conds, num, den):
        k = sum(tot[c][num] for c in conds)
        n = sum(tot[c][den] for c in conds)
        return k, n, (k / n if n else None)

    # 2x2 marginal effects on each rate (pooled difference)
    effects = {}
    for name, (num, den) in RATES.items():
        _, _, r_milk = pooled_rate(L1, num, den)
        _, _, r_gen = pooled_rate([c for c in CONDITIONS if c not in L1], num, den)
        _, _, r_narr = pooled_rate(N1, num, den)
        _, _, r_neu = pooled_rate([c for c in CONDITIONS if c not in N1], num, den)
        # interaction: (B4-B3) - (B2-B1) on the rate
        def cr(c):
            k, n = tot[c][num], tot[c][den]
            return (k / n) if n else None
        b1, b2, b3, b4 = (cr("B1"), cr("B2"), cr("B3"), cr("B4"))
        inter = None
        if None not in (b1, b2, b3, b4):
            inter = round((b4 - b3) - (b2 - b1), 4)
        effects[name] = {
            "label_effect_milk_minus_generic": (round(r_milk - r_gen, 4)
                                                if None not in (r_milk, r_gen) else None),
            "narrative_effect_N1_minus_N0": (round(r_narr - r_neu, 4)
                                             if None not in (r_narr, r_neu) else None),
            "interaction": inter,
        }
    return {"run_dir": os.path.abspath(run_dir), "per_condition": per_cond,
            "effects": effects}


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    args = p.parse_args(argv)
    a = analyze(args.run_dir)
    with open(os.path.join(args.run_dir, "exp_b_analysis.json"), "w") as f:
        json.dump(a, f, indent=2, default=str)

    print("Experiment B — semantic salience (narrative x food-label)")
    print("  run_dir:", a["run_dir"])
    print("\n  opportunity-conditioned rates [Wilson 95%] (k/n):")
    for name in RATES:
        print("   {}:".format(name))
        for c in CONDITIONS:
            r = a["per_condition"][c]["rates"][name]
            p_, lo, hi = r["wilson"]
            lab = "N{}/L{}".format(1 if c in N1 else 0, 1 if c in L1 else 0)
            if p_ is None:
                print("     {} ({}): n=0 (no opportunities)".format(c, lab))
            else:
                print("     {} ({}): {:.3f} [{:.3f},{:.3f}]  ({}/{})".format(
                    c, lab, p_, lo, hi, r["k"], r["n"]))
    print("\n  2x2 marginal effects (pooled rate differences):")
    for name, e in a["effects"].items():
        print("   {}: label(milk-gen)={} narrative(N1-N0)={} interaction={}".format(
            name, e["label_effect_milk_minus_generic"],
            e["narrative_effect_N1_minus_N0"], e["interaction"]))
    print("\n  selectivity check (take target vs distractor counts):")
    for c in CONDITIONS:
        pc = a["per_condition"][c]
        print("     {}: target={} distractor={} | fridge opens={} inserts={} | eps={}".format(
            c, pc["take_target"], pc["take_distractor"], pc["open_fridge"],
            pc["insert"], pc["episodes"]))
    print("\n  wrote exp_b_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
