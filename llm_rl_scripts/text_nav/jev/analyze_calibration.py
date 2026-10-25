"""Per-(seed, condition) success-rate analysis for a repeats-calibration run.

A calibration run has ``--repeats K`` independent rollouts per (seed, condition)
(same game, independent Jev choices). This script estimates, for each cell, the
success *rate* and a Wilson 95% interval, the mean greatest-ordered milestone
with its standard error, and a pooled per-condition rate. It also reports the
rollouts K needed to pin a rate to a target half-width, to inform the main run.

    python -m llm_rl_scripts.text_nav.jev.analyze_calibration <run_dir>

Reads only logged episodes; no Jev or TextWorld calls.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import os
import sys
from typing import Dict, List

Z = 1.96  # 95%


def wilson(k: int, n: int, z: float = Z):
    """Wilson score interval for a binomial proportion (no scipy)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "episodes.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def cells(eps: List[dict]) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in eps:
        out[(r["condition"], r["seed"])].append(r)
    return out


def k_for_halfwidth(p: float, half: float = 0.15, z: float = Z) -> int:
    """Rollouts needed so a Wald half-width is <= `half` at proportion p."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return max(1, math.ceil((z * z * p * (1 - p)) / (half * half)))


def analyze(run_dir: str) -> dict:
    eps = load(run_dir)
    cl = cells(eps)
    conds = sorted({c for c, _ in cl})
    seeds = sorted({s for _, s in cl})

    rows = []
    for c in conds:
        for s in seeds:
            rs = cl.get((c, s), [])
            n = len(rs)
            wins = sum(1 for r in rs if r["won"])
            p, lo, hi = wilson(wins, n)
            ordered = [r["milestones"]["greatest_ordered"] for r in rs]
            mean_ord = sum(ordered) / n if n else 0.0
            var = (sum((x - mean_ord) ** 2 for x in ordered) / (n - 1)) if n > 1 else 0.0
            se_ord = math.sqrt(var / n) if n else 0.0
            rows.append({
                "condition": c, "seed": s, "K": n, "wins": wins,
                "success_rate": round(p, 4),
                "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                "mean_ordered": round(mean_ord, 3), "se_ordered": round(se_ord, 3),
            })

    # pooled per-condition (over all rollouts across the calibration seeds)
    pooled = {}
    for c in conds:
        rs = [r for (cc, _), v in cl.items() if cc == c for r in v]
        n = len(rs)
        wins = sum(1 for r in rs if r["won"])
        p, lo, hi = wilson(wins, n)
        per_seed_rates = [row["success_rate"] for row in rows if row["condition"] == c]
        pooled[c] = {
            "n_rollouts": n, "wins": wins, "pooled_rate": round(p, 4),
            "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
            "per_seed_rate_min": min(per_seed_rates) if per_seed_rates else None,
            "per_seed_rate_max": max(per_seed_rates) if per_seed_rates else None,
        }
    return {"rows": rows, "pooled": pooled, "conditions": conds, "seeds": seeds,
            "run_dir": os.path.abspath(run_dir)}


def main(argv: List[str] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: analyze_calibration.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = argv[0]
    a = analyze(run_dir)

    with open(os.path.join(run_dir, "calibration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(a["rows"][0].keys()))
        w.writeheader()
        w.writerows(a["rows"])
    with open(os.path.join(run_dir, "calibration.json"), "w") as f:
        json.dump(a, f, indent=2, default=str)

    print("Repeats-calibration  ({} conditions x {} seeds)".format(
        len(a["conditions"]), len(a["seeds"])))
    print("  run_dir:", a["run_dir"])
    print("\n  per-(condition, seed) success rate [Wilson 95%]  (K rollouts):")
    for r in a["rows"]:
        print("    {} s{:>2}: {}/{} = {:.2f}  [{:.2f},{:.2f}]   mean_ordered={:.2f}+-{:.2f}"
              .format(r["condition"], r["seed"], r["wins"], r["K"],
                      r["success_rate"], r["wilson_lo"], r["wilson_hi"],
                      r["mean_ordered"], r["se_ordered"]))
    print("\n  pooled per-condition (across calibration seeds):")
    for c in a["conditions"]:
        p = a["pooled"][c]
        print("    {}: {}/{} = {:.3f}  [{:.3f},{:.3f}]   per-seed rate range [{},{}]"
              .format(c, p["wins"], p["n_rollouts"], p["pooled_rate"],
                      p["wilson_lo"], p["wilson_hi"],
                      p["per_seed_rate_min"], p["per_seed_rate_max"]))
    # K guidance from the most informative (largest-variance) observed rate
    rates = [r["success_rate"] for r in a["rows"] if 0 < r["success_rate"] < 1]
    if rates:
        p_mid = sorted(rates)[len(rates) // 2]
        print("\n  K guidance: a mid observed rate p={:.2f} needs K~{} per cell for a "
              "+-0.15 half-width (+-0.10 => K~{}).".format(
                  p_mid, k_for_halfwidth(p_mid, 0.15), k_for_halfwidth(p_mid, 0.10)))
    else:
        print("\n  K guidance: no intermediate rates observed (all 0 or 1); "
              "success is near-degenerate in these cells.")
    print("\n  wrote calibration.csv and calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
