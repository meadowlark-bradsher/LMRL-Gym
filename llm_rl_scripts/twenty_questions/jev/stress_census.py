"""Crowding census of an object x attribute matrix (the stress-test baseline).

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_census \
        --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json

Deterministic (no seeds, no API): binarizes each Noul row at 0.5 -- the same
threshold ``matrix_oracle`` answers with -- and reports how many objects the 44
attribute questions can tell apart, how close each object's nearest neighbour is,
and how much of the matrix sits in the ambiguous band. ``STRESS_DESIGN.md`` cites
these numbers for the real 158-object matrix; the stress studies call ``census``
on every generated pool so each grid cell carries its own crowding coordinates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .bed import ATTRIBUTE_BANK, load_hypotheses
from .matrix import load_matrix


def binarize(matrix: Dict[str, Dict[str, float]], names: Sequence[str],
             threshold: float = 0.5) -> Dict[str, tuple]:
    return {n: tuple(1 if matrix[aid][n] >= threshold else 0
                     for aid, _ in ATTRIBUTE_BANK) for n in names}


def census(matrix: Dict[str, Dict[str, float]], names: Sequence[str],
           categories: Optional[Dict[str, str]] = None,
           band: float = 0.2) -> dict:
    """Crowding coordinates of a pool. ``band``: a cell is ambiguous when
    ``band < p < 1 - band``. ``categories`` (name -> category) adds the
    within-category nearest-neighbour histogram, the distance that matters once
    a category question has narrowed the pool."""
    rows = binarize(matrix, names)
    names = list(names)

    def ham(a: str, b: str) -> int:
        return sum(x != y for x, y in zip(rows[a], rows[b]))

    nn = {a: min(ham(a, b) for b in names if b != a) for a in names}
    clusters = Counter(rows.values())
    cells = [matrix[aid][n] for aid, _ in ATTRIBUTE_BANK for n in names]
    out = {
        "n_objects": len(names),
        "n_attributes": len(ATTRIBUTE_BANK),
        "distinct_rows": len(clusters),
        "objects_sharing_a_row": sum(c for c in clusters.values() if c > 1),
        "largest_identical_cluster": max(clusters.values()),
        "nn_hamming_hist": dict(sorted(Counter(nn.values()).items())),
        "nn_within_1": sum(1 for d in nn.values() if d <= 1),
        "ambiguous_cells": sum(1 for p in cells if band < p < 1.0 - band),
        "n_cells": len(cells),
        # Expected attribute-flip rate if an answerer *samples* yes ~ Bernoulli(p)
        # instead of thresholding: the ST-4 A3 answerer, averaged uniformly over
        # cells (the planner's question mix will weight it differently).
        "mean_sampling_flip": sum(min(p, 1.0 - p) for p in cells) / len(cells),
    }
    if categories is not None:
        within = {}
        for a in names:
            same = [b for b in names if b != a and categories[b] == categories[a]]
            if same:
                within[a] = min(ham(a, b) for b in same)
        out["nn_within_category_hist"] = dict(sorted(Counter(within.values()).items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    args = ap.parse_args()
    objs = load_hypotheses()
    matrix = load_matrix(args.matrix)["matrix"]
    c = census(matrix, [h.name for h in objs],
               categories={h.name: h.category for h in objs})
    print(json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
