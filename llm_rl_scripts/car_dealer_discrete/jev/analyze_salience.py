"""Compare J/J conditions trajectory-by-trajectory (not just by outcome).

Reads one ``runs.jsonl`` per condition directory and reports, per condition:
  * outcome counts
  * the distinct deals each role offered, and how often
  * the decent-offer check: how many offers were worth >= r to the RECIPIENT.
    This is the confound test for any accept-side intervention (e.g. salience):
    if this is 0, no mutually-acceptable offer ever existed, so WALK was the
    correct deadline response and the binding failure is on the OFFER side.
  * who walked, relative to who started, and at which decision index
  * mean Jev confidence at first offer / last offer / walk

Utilities are recomputed exactly (Fraction) from the evaluator-only truth in
each record, so nothing here depends on numbers stored in the logs.

Example:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_salience \
        results/jj_salience_20261025/baseline results/jj_salience_20261025/salience
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Dict, List, Optional

from llm_rl_scripts.car_dealer_discrete.game import Deal, utility
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS


def load(path: str) -> List[dict]:
    runs = path if path.endswith(".jsonl") else os.path.join(path, "runs.jsonl")
    with open(runs) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _spread(xs: List[float]) -> Optional[Dict[str, float]]:
    """Mean with its observed range. Live Jev confidences are stochastic draws,
    so a mean without its spread is a point estimate masquerading as a quantity."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(sum(xs) / len(xs), 3), "min": min(xs), "max": max(xs), "n": len(xs)}


def analyze(recs: List[dict]) -> Dict[str, object]:
    outcomes = Counter(r["result"]["outcome_type"] for r in recs)
    offered: Dict[str, Counter] = defaultdict(Counter)
    walker: Counter = Counter()
    walk_idx: List[int] = []
    total_offers = decent = nonideal = 0
    conf_first: List[float] = []
    conf_last: List[float] = []
    conf_walk: List[float] = []

    for r in recs:
        truth = r["private_truth_evaluator_only"]
        rank = {"buyer": tuple(truth["buyer_rank"]), "seller": tuple(truth["seller_rank"])}
        res_r = RESERVATIONS[r["reservation_regime"]]
        for i, turn in enumerate(r["turns"]):
            actor = turn["actor"]
            if turn["action"] == "OFFER":
                d = Deal.from_id(turn["deal_id"])
                offered[actor][d.id] += 1
                total_offers += 1
                recipient = "seller" if actor == "buyer" else "buyer"
                if utility(recipient, rank[recipient], d) >= res_r:
                    decent += 1
                if utility(actor, rank[actor], d) < Fraction(1):
                    nonideal += 1
            elif turn["action"] == "WALK":
                walker[f"start={r['starting_side']} walker={actor}"] += 1
                walk_idx.append(i + 1)
        for role in ("buyer", "seller"):
            steps = r["jev_steps"][role]
            offs = [s for s in steps if s["chosen"] and s["chosen"].startswith("OFFER")]
            if offs:
                conf_first.append(offs[0]["confidence"])
                conf_last.append(offs[-1]["confidence"])
            walks = [s for s in steps if s["chosen"] == "WALK"]
            if walks:
                conf_walk.append(walks[0]["confidence"])

    return {
        "episodes": len(recs),
        "outcomes": dict(outcomes),
        "offered_by_buyer": dict(offered["buyer"]),
        "offered_by_seller": dict(offered["seller"]),
        "offers_total": total_offers,
        "offers_nonideal_own_lt_1": nonideal,
        "offers_worth_ge_r_to_recipient": decent,
        "who_walked": dict(walker),
        "walk_decision_indices": sorted(set(walk_idx)),
        "conf_first_offer": _spread(conf_first),
        "conf_last_offer": _spread(conf_last),
        "conf_walk": _spread(conf_walk),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("conditions", nargs="+", help="Condition dirs (or runs.jsonl paths).")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = p.parse_args(argv)

    report = {os.path.basename(os.path.normpath(c)): analyze(load(c)) for c in args.conditions}
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for name, a in report.items():
        print("=" * 70)
        print(f"{name.upper()}  ({a['episodes']} episodes)")
        print(f"  outcomes: {a['outcomes']}")
        print(f"  buyer offered : {a['offered_by_buyer']}")
        print(f"  seller offered: {a['offered_by_seller']}")
        print(
            f"  offers total={a['offers_total']}  non-ideal(own<1)={a['offers_nonideal_own_lt_1']}  "
            f"worth>=r to RECIPIENT={a['offers_worth_ge_r_to_recipient']}"
        )
        print(f"  who walked: {a['who_walked']}  at decision# {a['walk_decision_indices']}")
        print(
            f"  conf: first offer={a['conf_first_offer']}  last offer={a['conf_last_offer']}  "
            f"walk={a['conf_walk']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
