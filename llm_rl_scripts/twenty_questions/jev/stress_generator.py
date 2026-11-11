"""ST-0a pool generator: new objects with the real matrix's attribute structure.

    # calibrate rho_real against the real census (deterministic given seeds):
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_generator \
        --calibrate --seeds 5

A generated object picks a category and a *parent* object in it. Each of its 44
Noul cells is then independently, with probability ``rho``, replaced by a draw
from that category's empirical values for the attribute (the cells of the real
objects in the category); otherwise it keeps the parent's value. Graded values
survive, and so do the within-category correlations a parent carries. ``rho`` is
the crowding knob: small ``rho`` makes near-duplicates of real objects.

``rho_real`` is the ``rho`` at which an all-generated pool with the real
category composition matches the real matrix on ``nn_within_1`` (objects whose
nearest neighbour is <= 1 binarized attribute away; 98 of 158). The stress
studies state crowding relative to it (``STRESS_DESIGN.md``).

``lookalike`` makes ST-2's novel objects: a copy of a parent with exactly ``d``
cells flipped across 0.5 (p -> 1 - p).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .bed import ATTRIBUTE_BANK, Hypothesis, load_hypotheses
from .matrix import load_matrix
from .stress_census import census

Matrix = Dict[str, Dict[str, float]]
Row = Dict[str, float]

AIDS = [aid for aid, _ in ATTRIBUTE_BANK]


def category_values(objs: Sequence[Hypothesis], matrix: Matrix
                    ) -> Dict[str, Dict[str, List[float]]]:
    """Per category, per attribute: the real objects' cells, in object order."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for h in objs:
        per = out.setdefault(h.category, {aid: [] for aid in AIDS})
        for aid in AIDS:
            per[aid].append(matrix[aid][h.name])
    return out


def child_row(parent: Row, cat_vals: Dict[str, List[float]], rho: float,
              rng: random.Random) -> Row:
    """One generated row: each cell resampled from the category with prob rho."""
    row: Row = {}
    for aid in AIDS:
        if rng.random() < rho:
            row[aid] = rng.choice(cat_vals[aid])
        else:
            row[aid] = parent[aid]
    return row


def _row(matrix: Matrix, name: str) -> Row:
    return {aid: matrix[aid][name] for aid in AIDS}


def generate_objects(objs: Sequence[Hypothesis], matrix: Matrix, n: int, rho: float,
                     seed: int, prefix: str = "syn",
                     per_category: Optional[Dict[str, int]] = None
                     ) -> Tuple[List[Hypothesis], Dict[str, Row]]:
    """Generate ``n`` objects (or exactly ``per_category[c]`` per category).
    Categories are drawn in proportion to the real category sizes; the parent is
    uniform within the category. Names are ``{prefix}_{category}_{i}``."""
    rng = random.Random(seed)
    by_cat: Dict[str, List[Hypothesis]] = {}
    for h in objs:
        by_cat.setdefault(h.category, []).append(h)
    cats = sorted(by_cat)
    vals = category_values(objs, matrix)
    if per_category is not None:
        plan = [c for c in cats for _ in range(per_category.get(c, 0))]
    else:
        weights = [len(by_cat[c]) for c in cats]
        plan = rng.choices(cats, weights=weights, k=n)
    hyps: List[Hypothesis] = []
    rows: Dict[str, Row] = {}
    for i, cat in enumerate(plan):
        parent = rng.choice(by_cat[cat])
        name = "{}_{}_{}".format(prefix, cat, i)
        rows[name] = child_row(_row(matrix, parent.name), vals[cat], rho, rng)
        hyps.append(Hypothesis(name=name, variants=(name,), category=cat))
    return hyps, rows


def lookalike(parent: Hypothesis, matrix: Matrix, d: int, rng: random.Random,
              name: str) -> Tuple[Hypothesis, Row]:
    """A copy of ``parent`` with exactly ``d`` cells flipped across 0.5. Cells at
    exactly 0.5 cannot cross and are never chosen."""
    row = _row(matrix, parent.name)
    flippable = [aid for aid in AIDS if row[aid] != 0.5]
    for aid in rng.sample(flippable, d):
        row[aid] = 1.0 - row[aid]
    return Hypothesis(name=name, variants=(name,), category=parent.category), row


def merge(matrix: Matrix, rows: Dict[str, Row]) -> Matrix:
    """A new matrix: ``matrix``'s columns plus the generated rows (no mutation)."""
    out = {aid: dict(col) for aid, col in matrix.items()}
    for name, row in rows.items():
        for aid in AIDS:
            out[aid][name] = row[aid]
    return out


def build_pool(objs: Sequence[Hypothesis], matrix: Matrix, n_total: int, rho: float,
               seed: int, include_real: bool = True, prefix: str = "syn"
               ) -> Tuple[List[Hypothesis], Matrix]:
    """An ST-1 pool: the real objects plus ``n_total - len(objs)`` generated ones
    (or ``n_total`` generated ones when ``include_real`` is False)."""
    n_gen = n_total - len(objs) if include_real else n_total
    if n_gen < 0:
        raise ValueError("n_total {} is below the real pool {}".format(n_total, len(objs)))
    gen, rows = generate_objects(objs, matrix, n_gen, rho, seed, prefix=prefix)
    hyps = (list(objs) if include_real else []) + gen
    return hyps, merge(matrix, rows)


def calibrate(objs: Sequence[Hypothesis], matrix: Matrix, rhos: Sequence[float],
              seeds: Sequence[int]) -> dict:
    """``nn_within_1`` of all-generated pools with the real category composition,
    per rho and seed, and the interpolated rho at which the seed-mean meets the
    real value. ``nn_within_1`` falls as rho rises (rho = 0 copies parents)."""
    comp = dict(Counter(h.category for h in objs))
    real = census(matrix, [h.name for h in objs])
    target = real["nn_within_1"]
    table = []
    for rho in rhos:
        per_seed = []
        for s in seeds:
            hyps, rows = generate_objects(objs, matrix, 0, rho, s, per_category=comp)
            c = census(merge(matrix, rows), [h.name for h in hyps])
            per_seed.append({"seed": s, "nn_within_1": c["nn_within_1"],
                             "distinct_rows": c["distinct_rows"],
                             "ambiguous_cells": c["ambiguous_cells"]})
        mean = sum(r["nn_within_1"] for r in per_seed) / len(per_seed)
        table.append({"rho": rho, "nn_within_1_mean": mean,
                      "nn_within_1_min": min(r["nn_within_1"] for r in per_seed),
                      "nn_within_1_max": max(r["nn_within_1"] for r in per_seed),
                      "per_seed": per_seed})
    rho_real = None
    for lo, hi in zip(table, table[1:]):
        a, b = lo["nn_within_1_mean"], hi["nn_within_1_mean"]
        if (a - target) * (b - target) <= 0 and a != b:
            rho_real = lo["rho"] + (target - a) * (hi["rho"] - lo["rho"]) / (b - a)
            break
    return {"target_nn_within_1": target, "real_census": real,
            "seeds": list(seeds), "table": table, "rho_real": rho_real}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--rhos", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the calibration JSON here")
    args = ap.parse_args()
    objs = load_hypotheses()
    matrix = load_matrix(args.matrix)["matrix"]
    if args.calibrate:
        res = calibrate(objs, matrix, [float(x) for x in args.rhos.split(",")],
                        list(range(1, args.seeds + 1)))
        print("target nn_within_1 = {} / {}".format(res["target_nn_within_1"], len(objs)))
        for row in res["table"]:
            print("  rho {:.3f}: nn_within_1 mean {:6.1f} [{}, {}]".format(
                row["rho"], row["nn_within_1_mean"], row["nn_within_1_min"],
                row["nn_within_1_max"]))
        print("rho_real = {}".format(
            "none in range" if res["rho_real"] is None else "{:.4f}".format(res["rho_real"])))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2, sort_keys=True)
            print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
