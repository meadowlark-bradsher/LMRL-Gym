"""Calibration of Jev's Noul world-model against commonsense ground truth.

Several attributes in the bank name a category directly ("Is it an animal?",
"Is it a musical instrument?", ...). For those the ground-truth yes/no is known
from the object dictionary, so we can ask how well Jev's graded Noul separates
in-category objects from the rest: mean Noul in vs out, the gap, and the AUC of
using the Noul to classify membership (1.0 = perfect ranking, 0.5 = chance).

This measures the *quality of the likelihood* that BED consumes, independent of
the planning loop. Run:
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_matrix \
        --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, List, Set

from .bed import load_hypotheses

# attribute id -> the set of categories whose objects should answer "yes".
ATTR_CATEGORIES: Dict[str, Set[str]] = {
    "animal": {"Animals"},
    "instrument": {"Musical Instruments"},
    "furniture": {"Furniture"},
    "electronic": {"Electronics"},
    "transport": {"Vehicles"},
    "toy_play": {"Toys"},
    "food": {"Fruits", "Vegetables"},
    "edible": {"Fruits", "Vegetables"},
    "wearable": {"Clothes", "Jewelry"},
    "decoration": {"Jewelry"},
    "garden": {"Garden Supplies"},
    "office_class": {"Office Supplies"},
    "write_draw": {"Office Supplies", "Art"},
    "sports_games": {"Sports", "Toys"},
}


def auc(pos: List[float], neg: List[float]) -> float:
    """AUC = P(score(pos) > score(neg)), ties counted as 0.5."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = json.load(open(args.matrix))
    matrix = payload["matrix"]
    hyps = {h.name: h for h in load_hypotheses()}
    hyps = {n: h for n, h in hyps.items() if n in payload["objects"]}

    rows = []
    for aid, cats in ATTR_CATEGORIES.items():
        if aid not in matrix:
            continue
        pos = [matrix[aid][n] for n, h in hyps.items() if h.category in cats]
        neg = [matrix[aid][n] for n, h in hyps.items() if h.category not in cats]
        rows.append({
            "attribute": aid,
            "categories": sorted(cats),
            "n_pos": len(pos), "n_neg": len(neg),
            "mean_in": round(statistics.mean(pos), 3),
            "mean_out": round(statistics.mean(neg), 3),
            "gap": round(statistics.mean(pos) - statistics.mean(neg), 3),
            "auc": round(auc(pos, neg), 3),
        })

    rows.sort(key=lambda r: -r["auc"])
    aucs = [r["auc"] for r in rows]
    result = {
        "matrix": args.matrix,
        "model": payload.get("model"),
        "n_attributes_checked": len(rows),
        "mean_auc": round(statistics.mean(aucs), 3),
        "min_auc": round(min(aucs), 3),
        "frac_auc_ge_0.9": round(sum(a >= 0.9 for a in aucs) / len(aucs), 3),
        "per_attribute": rows,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print("saved -> {}".format(args.out))


if __name__ == "__main__":
    main()
