"""Score the INFEASIBLE type pairs (no deal is individually rational for both).

Spec 4.1/4.4: for F = ∅ the correct outcome is no deal. These pairs must never be
scored as two failures; they are scored on correct-no-deal and on how efficiently
the agents recognise incompatibility. In the hard regime (r = 0.55) there are 6
such ordered pairs. Six clusters is few: every interval here is wide by
construction and is reported as such.

Per-episode metrics (bayes − prior-only, paired type-pair cluster bootstrap):
  correct_no_deal     1 if the episode ended without an invalid acceptance
  invalid_acceptance  1 if an agent tried to ACCEPT a non-IR deal
  walk / timeout      how the no-deal was reached (recognised vs ran out)
  proposals_used      proposals burned before ending (lower = recognised sooner)
  decisions           total decisions
  final_max_pacc_own_ir
      the aid's max P(opponent accepts) at each seat's LAST decision, taken over
      the deals that are IR for THAT seat (own utility >= r), averaged over the
      two seats. This is the forward model's own verdict on "is there anything I
      want that they would take?" — the quantity a learner should drive down on
      an infeasible pair, and one the prior-only aid cannot update.

Example:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_infeasible \
        results/jj_hard_20261025/bayes results/jj_hard_20261025/prior_only --seed 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_discrete.jev.analyze_salience import load
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

Pair = Tuple[Tuple[str, ...], Tuple[str, ...]]
METRICS = ["correct_no_deal", "invalid_acceptance", "walk", "timeout", "proposals_used",
           "decisions", "final_max_pacc_own_ir"]


def derive(r: dict) -> dict:
    tr = r["private_truth_evaluator_only"]
    rank = {"buyer": tuple(tr["buyer_rank"]), "seller": tuple(tr["seller_rank"])}
    res = RESERVATIONS[r["reservation_regime"]]
    rs, sc = r["result"], r["score"]
    out = {
        "pair": (rank["buyer"], rank["seller"]),
        "correct_no_deal": 1.0 if sc["correct_no_deal"] else 0.0,
        "invalid_acceptance": 1.0 if rs["outcome_type"] == "invalid_acceptance" else 0.0,
        "walk": 1.0 if rs["outcome_type"] == "walk" else 0.0,
        "timeout": 1.0 if rs["outcome_type"] == "timeout" else 0.0,
        "proposals_used": float(rs["proposals_used"]),
        "decisions": float(rs["decisions"]),
    }
    vals = []
    for role in ("buyer", "seller"):
        steps = [s for s in r["jev_steps"][role] if s.get("acceptance")]
        if not steps:
            continue
        acc = steps[-1]["acceptance"]
        own_ir = [d for d in acc if utility(role, rank[role], DEALS_BY_ID[d]) >= res]
        if own_ir:
            vals.append(max(acc[d] for d in own_ir))
    out["final_max_pacc_own_ir"] = (sum(vals) / len(vals)) if vals else None
    return out


def _stats(eps, metric):
    st: Dict[Pair, Tuple[float, int]] = {}
    for e in eps:
        v = e.get(metric)
        if v is None:
            continue
        s, c = st.get(e["pair"], (0.0, 0)); st[e["pair"]] = (s + v, c + 1)
    return st


def _mean(st, sample):
    s = c = 0.0
    for p in sample:
        if p in st:
            sv, cv = st[p]; s += sv; c += cv
    return s / c if c else None


def _ci(xs):
    if not xs:
        return None
    xs = sorted(xs); n = len(xs)
    return (round(xs[int(0.025 * (n - 1))], 4), round(xs[int(0.975 * (n - 1))], 4))


def compare(A, B, pairs, metric, boot, rng):
    sa, sb = _stats(A, metric), _stats(B, metric)
    ma, mb = _mean(sa, pairs), _mean(sb, pairs)
    da, db, dd = [], [], []
    n = len(pairs)
    for _ in range(boot):
        smp = [pairs[rng.randrange(n)] for _ in range(n)]
        a, b = _mean(sa, smp), _mean(sb, smp)
        if a is not None: da.append(a)
        if b is not None: db.append(b)
        if a is not None and b is not None: dd.append(a - b)
    return {"bayes": None if ma is None else round(ma, 4), "prior_only": None if mb is None else round(mb, 4),
            "delta": None if (ma is None or mb is None) else round(ma - mb, 4),
            "ci95_bayes": _ci(da), "ci95_prior_only": _ci(db), "ci95_delta": _ci(dd)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bayes"); ap.add_argument("prior_only")
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    RA = [r for r in load(args.bayes) if not r["score"]["feasible"]]
    RB = [r for r in load(args.prior_only) if not r["score"]["feasible"]]
    A, B = [derive(r) for r in RA], [derive(r) for r in RB]
    pairs = sorted({e["pair"] for e in A} & {e["pair"] for e in B})
    report = {
        "_scope": "INFEASIBLE type pairs only (F = ∅); scored on correct-no-deal, never as G/Nash = 0",
        "bootstrap": {"B": args.boot, "seed": args.seed, "unit": "ordered type pair, paired", "pairs": len(pairs)},
        "power_note": f"{len(pairs)} clusters: intervals are wide by construction; report, do not over-read",
        "episodes": {"bayes": len(A), "prior_only": len(B)},
        "infeasible_pairs": [{"buyer": ">".join(p[0]), "seller": ">".join(p[1])} for p in pairs],
        "metrics": {m: compare(A, B, pairs, m, args.boot, rng) for m in METRICS},
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str)); return 0
    print(report["_scope"]); print(report["power_note"])
    print(f"episodes bayes={len(A)} prior={len(B)}  pairs={len(pairs)}  B={args.boot} seed={args.seed}\n")
    print(f"{'metric':<24}{'bayes':>9}{'prior':>9}{'Δ':>9}   95% CI(Δ)")
    for m, v in report["metrics"].items():
        print(f"{m:<24}{v['bayes']!s:>9}{v['prior_only']!s:>9}{v['delta']!s:>9}   {v['ci95_delta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
