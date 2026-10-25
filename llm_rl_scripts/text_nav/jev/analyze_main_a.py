"""Seed-clustered analysis for the Experiment A main run — the sole generator of
the numbers the results write-up cites (per PREREGISTRATION_A.md).

The seed is the cluster: per (seed, condition) we form a success rate over the K
rollouts and a mean greatest-ordered milestone, then aggregate across seeds with a
cluster bootstrap that resamples *seeds* (with replacement) and reuses the same
resampled seed set across every condition and contrast in an iteration, so the
pre-registered seed-paired contrasts stay paired.

    python -m llm_rl_scripts.text_nav.jev.analyze_main_a <run_dir> [--boot 10000]

Pure stdlib; reproducible (fixed bootstrap seed). No Jev/TextWorld calls.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys
from typing import Dict, List, Tuple

CONDITIONS = ["A1", "A2", "A3", "A4"]
BOOT_SEED = 20261025


def load(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "episodes.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_seed_vectors(eps: List[dict]) -> Tuple[List[int], Dict[str, Dict[int, float]],
                                               Dict[str, Dict[int, float]]]:
    """Return (seeds, rate[cond][seed], ordered[cond][seed])."""
    cells: Dict[Tuple[str, int], List[dict]] = collections.defaultdict(list)
    for r in eps:
        cells[(r["condition"], r["seed"])].append(r)
    seeds = sorted({s for _, s in cells})
    rate: Dict[str, Dict[int, float]] = {c: {} for c in CONDITIONS}
    ordered: Dict[str, Dict[int, float]] = {c: {} for c in CONDITIONS}
    for (c, s), rs in cells.items():
        if c not in rate:
            continue
        n = len(rs)
        rate[c][s] = sum(1 for r in rs if r["won"]) / n
        ordered[c][s] = sum(r["milestones"]["greatest_ordered"] for r in rs) / n
    return seeds, rate, ordered


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _ci(samples: List[float]) -> Tuple[float, float]:
    xs = sorted(samples)
    n = len(xs)
    lo = xs[max(0, int(0.025 * n))]
    hi = xs[min(n - 1, int(0.975 * n))]
    return (lo, hi)


# contrasts as functions of a {cond: per-seed value} snapshot over a seed set
def _contrasts(vals: Dict[str, float]) -> Dict[str, float]:
    a1, a2, a3, a4 = vals["A1"], vals["A2"], vals["A3"], vals["A4"]
    return {
        "A4_vs_A1": a4 - a1,
        "A4_vs_A2": a4 - a2,
        "A4_vs_A3": a4 - a3,
        "goal_main": (a2 + a4) - (a1 + a3),
        "history_main": (a3 + a4) - (a1 + a2),
        "interaction": (a4 - a3) - (a2 - a1),
    }


def analyze(run_dir: str, boot: int = 10000) -> dict:
    eps = load(run_dir)
    seeds, rate, ordered = per_seed_vectors(eps)
    rng = random.Random(BOOT_SEED)

    def summarize(metric: Dict[str, Dict[int, float]]) -> dict:
        # point estimates
        cond_mean = {c: _mean([metric[c][s] for s in seeds if s in metric[c]])
                     for c in CONDITIONS}
        point_contrasts = _contrasts(cond_mean)
        # cluster bootstrap over seeds (shared resample across conditions)
        boot_cond: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
        boot_con: Dict[str, List[float]] = {k: [] for k in point_contrasts}
        for _ in range(boot):
            draw = [rng.choice(seeds) for _ in seeds]
            cm = {c: _mean([metric[c][s] for s in draw if s in metric[c]])
                  for c in CONDITIONS}
            for c in CONDITIONS:
                boot_cond[c].append(cm[c])
            for k, v in _contrasts(cm).items():
                boot_con[k].append(v)
        return {
            "condition_mean": {c: round(cond_mean[c], 4) for c in CONDITIONS},
            "condition_ci": {c: [round(x, 4) for x in _ci(boot_cond[c])]
                             for c in CONDITIONS},
            "contrasts": {k: {"estimate": round(point_contrasts[k], 4),
                              "ci": [round(x, 4) for x in _ci(boot_con[k])],
                              "excludes_zero": (0 < _ci(boot_con[k])[0]
                                                or 0 > _ci(boot_con[k])[1])}
                          for k in point_contrasts},
        }

    return {
        "run_dir": os.path.abspath(run_dir),
        "n_seeds": len(seeds),
        "git_commit": json.load(open(os.path.join(run_dir, "run.json"))).get(
            "git_commit"),
        "repeats": json.load(open(os.path.join(run_dir, "run.json"))).get("repeats"),
        "boot": boot,
        "success": summarize(rate),
        "milestone_ordered": summarize(ordered),
        "per_seed_success": {c: {s: round(rate[c].get(s, float("nan")), 3)
                                 for s in seeds} for c in CONDITIONS},
    }


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--boot", type=int, default=10000)
    args = p.parse_args(argv)

    a = analyze(args.run_dir, boot=args.boot)
    with open(os.path.join(args.run_dir, "main_analysis.json"), "w") as f:
        json.dump(a, f, indent=2, default=str)
    # flat CSV of per-condition point + CI for both endpoints
    with open(os.path.join(args.run_dir, "main_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "condition", "mean", "ci_lo", "ci_hi"])
        for ep in ("success", "milestone_ordered"):
            for c in CONDITIONS:
                m = a[ep]["condition_mean"][c]
                lo, hi = a[ep]["condition_ci"][c]
                w.writerow([ep, c, m, lo, hi])

    print("Experiment A main analysis  ({} seeds x K={}, commit {}, {} boot)".format(
        a["n_seeds"], a["repeats"], (a["git_commit"] or "?")[:8], a["boot"]))
    for ep in ("success", "milestone_ordered"):
        s = a[ep]
        print("\n  == {} ==".format(ep))
        for c in CONDITIONS:
            lo, hi = s["condition_ci"][c]
            print("    {}: {:.3f}  [{:.3f}, {:.3f}]".format(
                c, s["condition_mean"][c], lo, hi))
        print("    contrasts (95% cluster-bootstrap CI; * excludes 0):")
        for k, v in s["contrasts"].items():
            star = "*" if v["excludes_zero"] else " "
            print("      {} {:>13}: {:+.3f}  [{:+.3f}, {:+.3f}]".format(
                star, k, v["estimate"], v["ci"][0], v["ci"][1]))
    print("\n  wrote main_analysis.json and main_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
