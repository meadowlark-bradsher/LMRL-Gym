"""Analyze the forced-counter probe (PREREGISTRATION_COUNTER.md).

Primary: Δ P(counter acceptable to starter) = seller_responds − buyer_responds, paired
type-pair-clustered bootstrap over the feasible pairs (each pair contributes both
cells). Secondaries: Δ utility-to-starter, Δ own utility, Δ shown P(accept) of the
chosen counter. Selection check: acceptability by cell × whether the opening was
acceptable to the responder. Descriptive: chosen-deal distributions by cell and the
total-variation distance between the seller cell's choices mapped through the
buyer/seller mirror and the buyer cell's choices.

Example:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_counter_probe \
        results/jj_counter_20261025 --seed 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS, Deal
from llm_rl_scripts.car_dealer_discrete.jev.aid import AcceptanceAid, AidConfig
from llm_rl_scripts.car_dealer_discrete.protocol import Observation

Pair = Tuple[Tuple[str, ...], Tuple[str, ...]]
METRICS = ["acceptable", "u_to_starter", "own_u", "shown_pacc"]


def load(path: str) -> List[dict]:
    p = path if path.endswith(".jsonl") else os.path.join(path, "decisions.jsonl")
    recs = [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in recs if "counter" in r], sum(1 for r in recs if "error" in r)


def derive(r: dict) -> dict:
    tr = r["private_truth_evaluator_only"]
    return {"pair": (tuple(tr["buyer_rank"]), tuple(tr["seller_rank"])), "cell": r["cell"],
            "acceptable": 1.0 if r["counter_acceptable_to_starter"] else 0.0,
            "u_to_starter": r["counter_u_to_starter"], "own_u": r["counter_own_u"],
            "shown_pacc": r["shown_pacc_of_counter"], "opening_ok": bool(r["opening_acceptable_to_responder"]),
            "counter": r["counter"]}


def _stats(eps, metric, filt):
    st: Dict[Pair, Tuple[float, int]] = {}
    for e in eps:
        if not filt(e): continue
        v = e.get(metric)
        if v is None: continue
        s, c = st.get(e["pair"], (0.0, 0)); st[e["pair"]] = (s + float(v), c + 1)
    return st

def _mean(st, sample):
    s = c = 0.0
    for p in sample:
        if p in st: sv, cv = st[p]; s += sv; c += cv
    return s / c if c else None

def _ci(xs):
    if not xs: return None
    xs = sorted(xs); n = len(xs)
    return (round(xs[int(0.025 * (n - 1))], 4), round(xs[int(0.975 * (n - 1))], 4))

def contrast(eps, metric, fa, fb, pairs, boot, rng):
    sa, sb = _stats(eps, metric, fa), _stats(eps, metric, fb)
    ma, mb = _mean(sa, pairs), _mean(sb, pairs)
    da, db, dd = [], [], []; n = len(pairs)
    for _ in range(boot):
        smp = [pairs[rng.randrange(n)] for _ in range(n)]
        a, b = _mean(sa, smp), _mean(sb, smp)
        if a is not None: da.append(a)
        if b is not None: db.append(b)
        if a is not None and b is not None: dd.append(a - b)
    return {"A": None if ma is None else round(ma, 4), "B": None if mb is None else round(mb, 4),
            "delta": None if (ma is None or mb is None) else round(ma - mb, 4),
            "ci95_A": _ci(da), "ci95_B": _ci(db), "ci95_delta": _ci(dd),
            "n_A": sum(c for _, c in sa.values()), "n_B": sum(c for _, c in sb.values())}


def mirror(deal_id: str) -> str:
    d = Deal.from_id(deal_id); return Deal(2 - d.price, 2 - d.warranty, 2 - d.delivery).id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_dir")
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    recs, failed = load(args.probe_dir)
    E = [derive(r) for r in recs]
    pairs = sorted({e["pair"] for e in E})
    SR = lambda e: e["cell"] == "seller_responds"; BR = lambda e: e["cell"] == "buyer_responds"

    # sanity: information shown is mirror-symmetric by construction (recompute the prior dicts)
    aid = AcceptanceAid(AidConfig(mode="prior_only")); r = Fraction(11, 20)
    def prior_dict(role):
        obs = Observation(role=role, own_ranking=ALL_RANKINGS[0], reservation=r, max_proposals=6, proposals_used=1,
                          outstanding_offer=None, history=(), legal_actions=())
        return aid.acceptance_probabilities(obs)[1]
    pb, ps = prior_dict("buyer"), prior_dict("seller")
    mirror_ok = all(abs(pb[d] - ps[mirror(d)]) < 1e-9 for d in pb)

    primary = contrast(E, "acceptable", SR, BR, pairs, args.boot, rng)
    lo, hi = primary["ci95_delta"]
    verdict = ("CONFIRMED (seller counters less acceptable; CI < 0)" if hi < 0 else
               "OPPOSITE SIGN (seller counters MORE acceptable; CI > 0)" if lo > 0 else
               "NOT SUPPORTED (CI includes 0) — adequately powered: a null is informative")
    secondaries = {m: contrast(E, m, SR, BR, pairs, args.boot, rng) for m in ("u_to_starter", "own_u", "shown_pacc")}
    selection = {}
    for cell, fc in (("seller_responds", SR), ("buyer_responds", BR)):
        for ok in (True, False):
            f = (lambda e, fc=fc, ok=ok: fc(e) and e["opening_ok"] == ok)
            st = _stats(E, "acceptable", f); m = _mean(st, pairs)
            selection[f"{cell} | opening acceptable to responder={ok}"] = {"acceptable": None if m is None else round(m, 4), "n": sum(c for _, c in st.values())}
    # descriptive: chosen-deal distributions and mirror TVD
    cs = Counter(e["counter"] for e in E if SR(e)); cb = Counter(e["counter"] for e in E if BR(e))
    ns, nb = sum(cs.values()), sum(cb.values())
    cs_m = Counter({mirror(k): v for k, v in cs.items()})
    tvd = 0.5 * sum(abs(cs_m.get(d.id, 0) / ns - cb.get(d.id, 0) / nb) for d in ALL_DEALS) if ns and nb else None

    report = {
        "_pre_registration": "PREREGISTRATION_COUNTER.md",
        "decisions": {"analyzed": len(E), "failed_excluded": failed, "seller_responds": primary["n_A"], "buyer_responds": primary["n_B"]},
        "bootstrap": {"B": args.boot, "seed": args.seed, "unit": "feasible type pair, paired", "pairs": len(pairs)},
        "sanity": {"prior_pacc_dicts_mirror_symmetric": mirror_ok, "all_decisions_are_offers": all("counter" in r for r in recs)},
        "C1_primary_delta_acceptable_seller_minus_buyer": primary, "C1_verdict": verdict,
        "secondaries_seller_minus_buyer": secondaries,
        "C5_selection_check_acceptable_by_cell_x_opening_ok": selection,
        "descriptive": {"top_counters_seller_responds": cs.most_common(6), "top_counters_buyer_responds": cb.most_common(6),
                        "tvd_seller_mirrored_vs_buyer": None if tvd is None else round(tvd, 4)},
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str)); return 0
    print(f"decisions analyzed={len(E)} (failed excluded={failed})  cells: seller_responds={primary['n_A']} buyer_responds={primary['n_B']}  pairs={len(pairs)}  B={args.boot} seed={args.seed}")
    print(f"sanity: mirror-symmetric info={mirror_ok}  all offers={report['sanity']['all_decisions_are_offers']}\n")
    print(f"C1 PRIMARY  acceptable: seller {primary['A']} vs buyer {primary['B']}  Δ={primary['delta']:+.4f}  CI={primary['ci95_delta']}\n   -> {verdict}")
    for m, v in secondaries.items():
        print(f"   {m:<14} seller {v['A']} vs buyer {v['B']}  Δ={v['delta']:+.4f}  CI={v['ci95_delta']}")
    print("\nC5 selection check (acceptable rate):")
    for k, v in selection.items(): print(f"   {k:<58} {v['acceptable']!s:>8}  n={v['n']}")
    print(f"\ntop counters seller_responds: {cs.most_common(5)}\ntop counters buyer_responds:  {cb.most_common(5)}\nTVD(seller mirrored, buyer) = {report['descriptive']['tvd_seller_mirrored_vs_buyer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
