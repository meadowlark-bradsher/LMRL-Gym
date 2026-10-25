"""EXPLORATORY / POST HOC — why does learning raise G_S (resolved CI) but not G_B?

NOT pre-registered. Nothing here is a confirmed claim; it generates the
hypothesis a fresh-seed run would test. Every quantity still carries a paired,
type-pair-clustered bootstrap 95% interval so nothing is over-read.

Three lenses, all computable from the committed logs with no API calls
(the aid is a pure function of public history; every turn's belief is logged):

  1. Starting side — Δ G_B / Δ G_S split by who started, and re-framed by role
     in the exchange (starter vs responder) to test a proposer-advantage story.
  2. Who concedes first — with 0 ideal offers every offer is a concession, so:
     depth of each role's FIRST offer (own utility; lower = bigger concession),
     its aid-rated safety P(accept), who first offers a deal the recipient
     would actually accept (worth >= r), and whose proposal is the one accepted.
  3. What the posterior learns about each role — learning bits / entropy /
     argmax-correct by DIRECTION (buyer learns seller vs seller learns buyer),
     plus the within-arm asymmetry with its own CI (bayes only).

Pair-level: on how many of the 36 pairs is Δ a joint gain vs a transfer?

Example:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.explore_seller_asymmetry \
        results/jj_quality_20261025/bayes results/jj_quality_20261025/prior_only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import Deal, utility
from llm_rl_scripts.car_dealer_discrete.jev.analyze_salience import load
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

Pair = Tuple[Tuple[str, ...], Tuple[str, ...]]
OTHER = {"buyer": "seller", "seller": "buyer"}


def derive(r: dict) -> dict:
    tr = r["private_truth_evaluator_only"]
    rank = {"buyer": tuple(tr["buyer_rank"]), "seller": tuple(tr["seller_rank"])}
    res = float(RESERVATIONS[r["reservation_regime"]])
    turns, sc, rs = r["turns"], r["score"], r["result"]
    start = r["starting_side"]
    g = {"buyer": float(sc["buyer_gain"]) if sc["buyer_gain"] is not None else 0.0,
         "seller": float(sc["seller_gain"]) if sc["seller_gain"] is not None else 0.0}
    d: Dict[str, object] = {
        "pair": (rank["buyer"], rank["seller"]), "start": start,
        "agreement": 1.0 if rs["agreement"] else 0.0,
        "G_B": g["buyer"], "G_S": g["seller"],
        "G_starter": g[start], "G_responder": g[OTHER[start]],
        "U_B": float(rs["buyer_utility"]), "U_S": float(rs["seller_utility"]),
        "proposals": float(rs["proposals_used"]),
    }
    # whose proposal was accepted
    proposer = acceptor = outstanding = None
    for t in turns:
        if t["action"] == "OFFER":
            outstanding = t["actor"]
        elif t["action"] == "ACCEPT":
            acceptor, proposer = t["actor"], outstanding
    d["seller_proposed_accepted"] = (1.0 if proposer == "seller" else 0.0) if proposer else None
    d["starter_proposed_accepted"] = (1.0 if proposer == start else 0.0) if proposer else None
    d["split_U_S_minus_U_B"] = (d["U_S"] - d["U_B"]) if rs["agreement"] else None
    # first offers per role; first offer acceptable to its recipient
    idx = {"buyer": 0, "seller": 0}
    first: Dict[str, dict] = {}
    first_decent_role = None
    n_off = {"buyer": 0, "seller": 0}
    for t in turns:
        a = t["actor"]
        if t["action"] == "OFFER":
            n_off[a] += 1
            deal = Deal.from_id(t["deal_id"]); opp = OTHER[a]
            own = float(utility(a, rank[a], deal)); rec = float(utility(opp, rank[opp], deal))
            if a not in first:
                steps = r["jev_steps"][a]; i = idx[a]
                pacc = steps[i]["acceptance"].get(t["deal_id"]) if (i < len(steps) and steps[i].get("acceptance")) else None
                first[a] = {"own": own, "rec": rec, "pacc": pacc, "decent": 1.0 if rec >= res else 0.0}
            if first_decent_role is None and rec >= res:
                first_decent_role = a
        idx[a] += 1
    for role in ("buyer", "seller"):
        f = first.get(role, {})
        d[f"{role}_first_own"] = f.get("own")
        d[f"{role}_first_recipient_u"] = f.get("rec")
        d[f"{role}_first_pacc"] = f.get("pacc")
        d[f"{role}_first_decent"] = f.get("decent")
        d[f"n_offers_{role}"] = float(n_off[role])
    d["first_decent_by_seller"] = (1.0 if first_decent_role == "seller" else 0.0) if first_decent_role else None
    # Built-in placebo: the STARTER's first offer is made under the uniform prior in
    # BOTH arms (identical P(accept) shown), so its delta must be ~0. The RESPONDER's
    # first offer is the only decision that differs between arms (one observation).
    for who, role in (("starter", start), ("responder", OTHER[start])):
        f = first.get(role, {})
        d[f"{who}_first_own"] = f.get("own")
        d[f"{who}_first_recipient_u"] = f.get("rec")
        d[f"{who}_first_pacc"] = f.get("pacc")
        d[f"{who}_first_decent"] = f.get("decent")
    # learning by direction (only where a belief was logged and moved — bayes)
    for role in ("buyer", "seller"):
        opp = OTHER[role]
        steps = [s for s in r["jev_steps"][role] if s.get("belief")]
        key = ">".join(tr[f"{opp}_rank"])
        if steps:
            qT = steps[-1]["belief"]; q = max(qT.get(key, 0.0), 1e-300)
            d[f"{role}_learns_{opp}_bits"] = math.log2(q) - math.log2(1 / 6)
            d[f"{role}_learns_{opp}_entropy"] = steps[-1]["belief_entropy_bits"]
            d[f"{role}_learns_{opp}_argmax"] = 1.0 if max(qT, key=qT.get) == key else 0.0
        else:
            d[f"{role}_learns_{opp}_bits"] = d[f"{role}_learns_{opp}_entropy"] = d[f"{role}_learns_{opp}_argmax"] = None
    sb, bs = d["seller_learns_buyer_bits"], d["buyer_learns_seller_bits"]
    d["learn_asym_S_minus_B"] = (sb - bs) if (sb is not None and bs is not None) else None
    return d


def pair_stats(eps: List[dict], metric: str, filt: Optional[Callable[[dict], bool]]) -> Dict[Pair, Tuple[float, int]]:
    out: Dict[Pair, Tuple[float, int]] = {}
    for e in eps:
        if filt is not None and not filt(e):
            continue
        v = e.get(metric)
        if v is None:
            continue
        s, c = out.get(e["pair"], (0.0, 0))
        out[e["pair"]] = (s + float(v), c + 1)
    return out


def _mean(stats, sample) -> Optional[float]:
    s = c = 0.0
    for p in sample:
        if p in stats:
            sv, cv = stats[p]; s += sv; c += cv
    return s / c if c else None


def _ci(xs: List[float]) -> Optional[Tuple[float, float]]:
    if not xs:
        return None
    xs = sorted(xs); n = len(xs)
    return (round(xs[int(0.025 * (n - 1))], 4), round(xs[int(0.975 * (n - 1))], 4))


def compare(A: List[dict], B: List[dict], pairs: List[Pair], metric: str, filt, boot: int, rng) -> dict:
    sa, sb = pair_stats(A, metric, filt), pair_stats(B, metric, filt)
    ma, mb = _mean(sa, pairs), _mean(sb, pairs)
    da, db, dd = [], [], []
    n = len(pairs)
    for _ in range(boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a, b = _mean(sa, sample), _mean(sb, sample)
        if a is not None: da.append(a)
        if b is not None: db.append(b)
        if a is not None and b is not None: dd.append(a - b)
    return {"bayes": None if ma is None else round(ma, 4), "prior_only": None if mb is None else round(mb, 4),
            "delta": None if (ma is None or mb is None) else round(ma - mb, 4),
            "ci95_bayes": _ci(da), "ci95_prior_only": _ci(db), "ci95_delta": _ci(dd),
            "n_bayes": sum(c for _, c in sa.values()), "n_prior_only": sum(c for _, c in sb.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bayes"); ap.add_argument("prior_only")
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--feasible-only", action="store_true",
                   help="Restrict to type pairs with a feasible deal (hard regime).")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    RA, RB = load(args.bayes), load(args.prior_only)
    if args.feasible_only:
        RA = [r for r in RA if r["score"]["feasible"]]
        RB = [r for r in RB if r["score"]["feasible"]]
    A = [derive(r) for r in RA]
    B = [derive(r) for r in RB]
    pairs = sorted({e["pair"] for e in A} & {e["pair"] for e in B})
    bs = lambda e: e["start"] == "buyer"
    ss = lambda e: e["start"] == "seller"
    C = lambda m, f=None: compare(A, B, pairs, m, f, args.boot, rng)

    report = {
        "_status": "EXPLORATORY / POST HOC — not pre-registered; hypothesis-generating only",
        "feasible_only": args.feasible_only,
        "bootstrap": {"B": args.boot, "seed": args.seed, "unit": "ordered type pair, paired", "pairs": len(pairs)},
        "episodes": {"bayes": len(A), "prior_only": len(B)},
        "lens1_starting_side": {
            "G_B | start=buyer": C("G_B", bs), "G_B | start=seller": C("G_B", ss),
            "G_S | start=buyer": C("G_S", bs), "G_S | start=seller": C("G_S", ss),
            "G_starter (either role)": C("G_starter"), "G_responder (either role)": C("G_responder"),
            "proposals | start=buyer": C("proposals", bs), "proposals | start=seller": C("proposals", ss),
        },
        "lens2_who_concedes": {
            "starter's proposal was the accepted one (rate)": C("starter_proposed_accepted"),
            "seller's proposal was the accepted one (rate)": C("seller_proposed_accepted"),
            "buyer FIRST offer: own utility (lower = deeper concession)": C("buyer_first_own"),
            "seller FIRST offer: own utility": C("seller_first_own"),
            "buyer FIRST offer: utility to seller": C("buyer_first_recipient_u"),
            "seller FIRST offer: utility to buyer": C("seller_first_recipient_u"),
            "buyer FIRST offer: aid P(accept) shown": C("buyer_first_pacc"),
            "seller FIRST offer: aid P(accept) shown": C("seller_first_pacc"),
            "buyer FIRST offer acceptable to seller (rate)": C("buyer_first_decent"),
            "seller FIRST offer acceptable to buyer (rate)": C("seller_first_decent"),
            "first acceptable offer came from seller (rate)": C("first_decent_by_seller"),
            "offers per episode: buyer": C("n_offers_buyer"), "offers per episode: seller": C("n_offers_seller"),
            "split at agreement U_S - U_B": C("split_U_S_minus_U_B"),
        },
        "lens2b_placebo_starter_vs_responder_first_offer": {
            "STARTER first offer: own utility  (placebo: expect Δ≈0)": C("starter_first_own"),
            "STARTER first offer: utility to recipient (placebo)": C("starter_first_recipient_u"),
            "STARTER first offer: aid P(accept) shown (placebo)": C("starter_first_pacc"),
            "STARTER first offer acceptable to recipient (placebo)": C("starter_first_decent"),
            "RESPONDER first offer: own utility": C("responder_first_own"),
            "RESPONDER first offer: utility to recipient": C("responder_first_recipient_u"),
            "RESPONDER first offer: aid P(accept) shown": C("responder_first_pacc"),
            "RESPONDER first offer acceptable to recipient": C("responder_first_decent"),
        },
        "lens3_learning_by_direction_bayes_only": {
            "seller learns buyer: bits": C("seller_learns_buyer_bits"),
            "buyer learns seller: bits": C("buyer_learns_seller_bits"),
            "asymmetry (seller-learns - buyer-learns) bits": C("learn_asym_S_minus_B"),
            "seller learns buyer: final entropy": C("seller_learns_buyer_entropy"),
            "buyer learns seller: final entropy": C("buyer_learns_seller_entropy"),
            "seller learns buyer: argmax correct": C("seller_learns_buyer_argmax"),
            "buyer learns seller: argmax correct": C("buyer_learns_seller_argmax"),
        },
    }
    # pair-level decomposition of the delta
    def pm(eps, m, p): 
        v = [e[m] for e in eps if e["pair"] == p and e[m] is not None]; return sum(v) / len(v) if v else None
    cls = {"joint_gain(S+,B+)": 0, "transfer_to_seller(S+,B-)": 0, "transfer_to_buyer(S-,B+)": 0, "joint_loss(S-,B-)": 0, "flat": 0}
    rows = []
    for p in pairs:
        dS = pm(A, "G_S", p) - pm(B, "G_S", p); dB = pm(A, "G_B", p) - pm(B, "G_B", p)
        rows.append({"buyer": ">".join(p[0]), "seller": ">".join(p[1]), "dG_S": round(dS, 3), "dG_B": round(dB, 3)})
        if abs(dS) < 1e-9 and abs(dB) < 1e-9: cls["flat"] += 1
        elif dS >= 0 and dB >= 0: cls["joint_gain(S+,B+)"] += 1
        elif dS >= 0 and dB < 0: cls["transfer_to_seller(S+,B-)"] += 1
        elif dS < 0 and dB >= 0: cls["transfer_to_buyer(S-,B+)"] += 1
        else: cls["joint_loss(S-,B-)"] += 1
    report["pair_level"] = {"classification_of_36_pairs": cls, "per_pair": rows}

    if args.json:
        print(json.dumps(report, indent=2, default=str)); return 0
    print(report["_status"]); print(f"pairs={len(pairs)}  B={args.boot}  episodes bayes={len(A)} prior={len(B)}\n")
    def show(title, block):
        print("=" * 100); print(title)
        print(f"{'metric':<58}{'bayes':>8}{'prior':>8}{'Δ':>8}   95% CI(Δ)")
        for k, v in block.items():
            print(f"{k:<58}{v['bayes']!s:>8}{v['prior_only']!s:>8}{v['delta']!s:>8}   {v['ci95_delta']}")
    show("LENS 1 — starting side", report["lens1_starting_side"])
    show("LENS 2 — who concedes / whose proposal closes", report["lens2_who_concedes"])
    show("LENS 2b — built-in placebo: STARTER (uniform prior in both arms) vs RESPONDER (one observation)",
         report["lens2b_placebo_starter_vs_responder_first_offer"])
    print("=" * 100); print("LENS 3 — what the posterior learns, by direction (bayes arm only; CI = cluster CI of the mean)")
    for k, v in report["lens3_learning_by_direction_bayes_only"].items():
        print(f"{k:<58}{v['bayes']!s:>8}   95% CI {v['ci95_bayes']}")
    print("=" * 100); print("PAIR-LEVEL — Δ classification over 36 pairs:", cls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
