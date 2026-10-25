"""EXPLORATORY / POST HOC — why does the responder gain resolve only when the SELLER responds?

Hard regime, feasible pairs only. Not pre-registered; hypothesis-generating. Every
quantity carries a paired type-pair-clustered bootstrap 95% interval over the 30
feasible pairs (each pair contributes both starting sides, so between-cell
contrasts are paired by pair).

Cells:  seller_responds = buyer starts;  buyer_responds = seller starts.

Candidate mechanisms (fixed before looking):
  H-a identifiability  buyer openings reveal more type -> responder posterior
                       entropy after ONE observation is lower when the seller responds
  H-b countering       the buyer-as-responder ACCEPTS the opening more often
                       instead of countering
  H-c agreement drag   buyer_responds episodes agree less (G = 0 drags Δ)
  H-d opening quality  seller openings are less acceptable to the recipient

Two structural sanity checks: the feasible pair set is closed under role swap,
and the uniform-prior P(accept) dicts shown to a buyer-starter and a
seller-starter are mirror images (so any role difference in openings is Jev's).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, Deal, utility
from llm_rl_scripts.car_dealer_discrete.jev.analyze_salience import load
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

Pair = Tuple[Tuple[str, ...], Tuple[str, ...]]
OTHER = {"buyer": "seller", "seller": "buyer"}
UNIFORM_BITS = math.log2(6)


def derive(r: dict) -> dict:
    tr = r["private_truth_evaluator_only"]
    rank = {"buyer": tuple(tr["buyer_rank"]), "seller": tuple(tr["seller_rank"])}
    res = float(RESERVATIONS[r["reservation_regime"]])
    st = r["starting_side"]; rp = OTHER[st]
    rs, sc, turns = r["result"], r["score"], r["turns"]
    g = {"buyer": float(sc["buyer_gain"] or 0.0), "seller": float(sc["seller_gain"] or 0.0)}
    d: Dict[str, object] = {
        "pair": (rank["buyer"], rank["seller"]), "start": st,
        "cell": "seller_responds" if st == "buyer" else "buyer_responds",
        "agreement": 1.0 if rs["agreement"] else 0.0,
        "G_responder": g[rp], "G_starter": g[st],
    }
    # starter's first offer (turn 0) and responder's first action (turn 1)
    t0 = turns[0]; d0 = Deal.from_id(t0["deal_id"])
    d["starter_first_own"] = float(utility(st, rank[st], d0))
    d["starter_first_recipient_u"] = float(utility(rp, rank[rp], d0))
    d["starter_first_acceptable"] = 1.0 if d["starter_first_recipient_u"] >= res else 0.0
    s0 = r["jev_steps"][st][0]
    d["starter_first_pacc"] = s0["acceptance"][t0["deal_id"]] if s0.get("acceptance") else None
    t1 = turns[1] if len(turns) > 1 else None
    d["responder_accepted_opening"] = 1.0 if (t1 and t1["action"] == "ACCEPT") else 0.0
    d["responder_countered"] = 1.0 if (t1 and t1["action"] == "OFFER") else 0.0
    d["responder_rejected"] = 1.0 if (t1 and t1["action"] == "REJECT") else 0.0
    if t1 and t1["action"] == "OFFER":
        d1 = Deal.from_id(t1["deal_id"])
        d["responder_first_own"] = float(utility(rp, rank[rp], d1))
        d["responder_first_recipient_u"] = float(utility(st, rank[st], d1))
        d["responder_first_acceptable"] = 1.0 if d["responder_first_recipient_u"] >= res else 0.0
        s1 = r["jev_steps"][rp][0]
        d["responder_first_pacc"] = s1["acceptance"][t1["deal_id"]] if s1.get("acceptance") else None
    else:
        for k in ("responder_first_own", "responder_first_recipient_u", "responder_first_acceptable", "responder_first_pacc"):
            d[k] = None
    # whose proposal closed
    proposer = acceptor = outstanding = None
    for t in turns:
        if t["action"] == "OFFER": outstanding = t["actor"]
        elif t["action"] == "ACCEPT": acceptor, proposer = t["actor"], outstanding
    d["starter_proposal_accepted"] = (1.0 if proposer == st else 0.0) if proposer else None
    # what the RESPONDER's posterior learned about the STARTER — after one observation, and finally
    steps = [s for s in r["jev_steps"][rp] if s.get("belief")]
    true_key = ">".join(tr[f"{st}_rank"])
    if steps:
        q1, qT = steps[0]["belief"], steps[-1]["belief"]
        d["resp_entropy_after_1obs"] = steps[0]["belief_entropy_bits"]
        d["resp_improvement_after_1obs"] = math.log2(max(q1.get(true_key, 0.0), 1e-300)) - math.log2(1 / 6)
        d["resp_final_improvement"] = math.log2(max(qT.get(true_key, 0.0), 1e-300)) - math.log2(1 / 6)
        d["resp_final_argmax_true"] = 1.0 if max(qT, key=qT.get) == true_key else 0.0
    else:
        d["resp_entropy_after_1obs"] = d["resp_improvement_after_1obs"] = d["resp_final_improvement"] = d["resp_final_argmax_true"] = None
    return d


def _stats(eps, metric, filt):
    st: Dict[Pair, Tuple[float, int]] = {}
    for e in eps:
        if filt and not filt(e): continue
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

def contrast(statsA, statsB, pairs, boot, rng):
    ma, mb = _mean(statsA, pairs), _mean(statsB, pairs)
    da, db, dd = [], [], []; n = len(pairs)
    for _ in range(boot):
        smp = [pairs[rng.randrange(n)] for _ in range(n)]
        a, b = _mean(statsA, smp), _mean(statsB, smp)
        if a is not None: da.append(a)
        if b is not None: db.append(b)
        if a is not None and b is not None: dd.append(a - b)
    return {"A": None if ma is None else round(ma, 4), "B": None if mb is None else round(mb, 4),
            "delta": None if (ma is None or mb is None) else round(ma - mb, 4),
            "ci95_A": _ci(da), "ci95_B": _ci(db), "ci95_delta": _ci(dd)}


METRICS = ["G_responder", "G_starter", "agreement",
           "starter_first_own", "starter_first_recipient_u", "starter_first_acceptable", "starter_first_pacc",
           "responder_accepted_opening", "responder_countered", "responder_rejected",
           "responder_first_own", "responder_first_recipient_u", "responder_first_acceptable", "responder_first_pacc",
           "starter_proposal_accepted",
           "resp_entropy_after_1obs", "resp_improvement_after_1obs", "resp_final_improvement", "resp_final_argmax_true"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bayes"); ap.add_argument("prior_only")
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    RA = [r for r in load(args.bayes) if r["score"]["feasible"]]
    RB = [r for r in load(args.prior_only) if r["score"]["feasible"]]
    A, B = [derive(r) for r in RA], [derive(r) for r in RB]
    pairs = sorted({e["pair"] for e in A} & {e["pair"] for e in B})

    # structural sanity
    swap_closed = all((p[1], p[0]) in set(pairs) for p in pairs)
    sb = next(s for r in RA if r["starting_side"] == "buyer" for s in [r["jev_steps"]["buyer"][0]])["acceptance"]
    ss = next(s for r in RA if r["starting_side"] == "seller" for s in [r["jev_steps"]["seller"][0]])["acceptance"]
    mirror = lambda d: f"P{2-int(d[1])}-W{2-int(d[4])}-D{2-int(d[7])}"  # buyer-ideal <-> seller-ideal reflection
    mirror_ok = all(abs(sb[d] - ss[mirror(d)]) < 1e-9 for d in sb)

    SR = lambda e: e["cell"] == "seller_responds"; BR = lambda e: e["cell"] == "buyer_responds"
    within = {}
    for cell, filt in (("seller_responds", SR), ("buyer_responds", BR)):
        within[cell] = {m: contrast(_stats(A, m, filt), _stats(B, m, filt), pairs, args.boot, rng) for m in METRICS}
    between = {}
    for arm_name, eps in (("bayes", A), ("prior_only", B)):
        between[arm_name] = {m: contrast(_stats(eps, m, SR), _stats(eps, m, BR), pairs, args.boot, rng) for m in METRICS}

    report = {
        "_status": "EXPLORATORY / POST HOC — hard regime, feasible pairs only; hypothesis-generating",
        "bootstrap": {"B": args.boot, "seed": args.seed, "unit": "ordered feasible type pair, paired", "pairs": len(pairs)},
        "episodes": {"bayes": len(A), "prior_only": len(B)},
        "structural_sanity": {"feasible_pairs_closed_under_role_swap": swap_closed,
                               "uniform_prior_pacc_dicts_mirror_symmetric_buyer_vs_seller_starter": mirror_ok},
        "within_cell_delta_bayes_minus_prior": within,   # A = bayes, B = prior_only
        "between_cells_within_arm_seller_responds_minus_buyer_responds": between,  # A = seller_responds, B = buyer_responds
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str)); return 0

    print(report["_status"]); print(f"pairs={len(pairs)} eps bayes={len(A)} prior={len(B)}  structural: {report['structural_sanity']}\n")
    def show(title, block, la, lb):
        print("=" * 110); print(title); print(f"{'metric':<32}{la:>10}{lb:>10}{'Δ':>9}   95% CI(Δ)")
        for m, v in block.items():
            print(f"{m:<32}{v['A']!s:>10}{v['B']!s:>10}{v['delta']!s:>9}   {v['ci95_delta']}")
    show("WITHIN CELL: seller_responds (buyer starts) — Δ = bayes − prior", within["seller_responds"], "bayes", "prior")
    show("WITHIN CELL: buyer_responds (seller starts) — Δ = bayes − prior", within["buyer_responds"], "bayes", "prior")
    show("BETWEEN CELLS within BAYES — Δ = seller_responds − buyer_responds", between["bayes"], "sell_resp", "buy_resp")
    show("BETWEEN CELLS within PRIOR_ONLY — Δ = seller_responds − buyer_responds", between["prior_only"], "sell_resp", "buy_resp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
