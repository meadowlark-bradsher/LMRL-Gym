"""Cluster-bootstrap CIs for the Wordle Jev-vs-optimal benchmark.

Reads the per-(target, repeat) JSONL written by eval_jev.py --jsonl. The unit of
resampling is the TARGET WORD: repeats within a target are correlated (same
hidden word, same posterior trajectory), so the bootstrap resamples whole
targets with replacement and keeps all of a drawn target's repeats. That makes
the CIs robust to both between-target heterogeneity (the dominant source) and
Jev's within-target selection noise.

Point estimates come from the full data; CIs are the 2.5/97.5 percentiles of the
bootstrap distribution. The headline is the paired gap optimal_solve - jev_solve
and whether its CI clears zero.

    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.wordle.jev.analyze_wordle PATH/TO/games.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple


def load_by_target(path: str) -> Dict[str, List[dict]]:
    by_target: Dict[str, List[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_target[rec["target"]].append(rec)
    return dict(by_target)


def _mean_or_none(xs: List[float]) -> Optional[float]:
    return mean(xs) if xs else None


def point_estimates(by_target: Dict[str, List[dict]]) -> Dict[str, Optional[float]]:
    games = [g for recs in by_target.values() for g in recs]
    jev_solate = [1.0 if g["jev_solved"] else 0.0 for g in games]
    opt_solate = [1.0 if g["opt_solved"] else 0.0 for g in games]
    jev_g = [g["jev_guesses"] for g in games if g["jev_solved"]]
    opt_g = [g["opt_guesses"] for g in games if g["opt_solved"]]
    turns = [t for g in games for t in g.get("jev_turns", [])]
    jev_solve = mean(jev_solate)
    opt_solve = mean(opt_solate)
    return {
        "jev_solve": jev_solve,
        "opt_solve": opt_solve,
        "gap": opt_solve - jev_solve,
        "jev_avg_guesses": _mean_or_none(jev_g),
        "opt_avg_guesses": _mean_or_none(opt_g),
        "match_at_1": _mean_or_none([1.0 if t["match"] else 0.0 for t in turns]),
        "mean_regret_bits": _mean_or_none([t["regret"] for t in turns]),
    }


def _agg(sample_targets: List[List[dict]]) -> Dict[str, Optional[float]]:
    games = [g for recs in sample_targets for g in recs]
    jev_solate = [1.0 if g["jev_solved"] else 0.0 for g in games]
    opt_solate = [1.0 if g["opt_solved"] else 0.0 for g in games]
    jev_g = [g["jev_guesses"] for g in games if g["jev_solved"]]
    turns = [t for g in games for t in g.get("jev_turns", [])]
    jev_solve = mean(jev_solate)
    opt_solve = mean(opt_solate)
    return {
        "jev_solve": jev_solve,
        "opt_solve": opt_solve,
        "gap": opt_solve - jev_solve,
        "jev_avg_guesses": _mean_or_none(jev_g),
        "match_at_1": _mean_or_none([1.0 if t["match"] else 0.0 for t in turns]),
        "mean_regret_bits": _mean_or_none([t["regret"] for t in turns]),
    }


def bootstrap(by_target: Dict[str, List[dict]], b: int, seed: int
              ) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    clusters = list(by_target.values())
    t = len(clusters)
    keys = ["jev_solve", "opt_solve", "gap", "jev_avg_guesses",
            "match_at_1", "mean_regret_bits"]
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(b):
        sample = [clusters[rng.randrange(t)] for _ in range(t)]
        agg = _agg(sample)
        for k in keys:
            v = agg[k]
            if v is not None:
                draws[k].append(v)
    cis = {}
    for k, xs in draws.items():
        if not xs:
            cis[k] = (float("nan"), float("nan"))
            continue
        xs.sort()
        lo = xs[int(0.025 * len(xs))]
        hi = xs[min(len(xs) - 1, int(0.975 * len(xs)))]
        cis[k] = (lo, hi)
    return cis


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="per-(target,repeat) records from eval_jev --jsonl")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    by_target = load_by_target(args.jsonl)
    n_targets = len(by_target)
    n_games = sum(len(v) for v in by_target.values())
    reps = n_games // n_targets if n_targets else 0

    pe = point_estimates(by_target)
    ci = bootstrap(by_target, args.bootstrap, args.seed)

    def fmt(key, scale=1.0, unit=""):
        v = pe[key]
        lo, hi = ci[key]
        if v is None:
            return "n/a"
        return "{:.3f}{} [{:.3f}, {:.3f}]".format(
            v * scale, unit, lo * scale, hi * scale)

    print("Wordle x Jev  cluster bootstrap over targets")
    print("  targets={}  games={}  (~{} reps/target)  B={}".format(
        n_targets, n_games, reps, args.bootstrap))
    print("  jev solve      : {}".format(fmt("jev_solve")))
    print("  optimal solve  : {}".format(fmt("opt_solve")))
    print("  GAP (opt-jev)  : {}".format(fmt("gap")))
    print("  jev avg guesses: {}".format(fmt("jev_avg_guesses")))
    print("  match@1        : {}".format(fmt("match_at_1")))
    print("  mean regret    : {} bits".format(fmt("mean_regret_bits")))
    gap_lo, gap_hi = ci["gap"]
    verdict = "clears 0 (optimal beats Jev)" if gap_lo > 0 else "includes 0 (no separation)"
    print("  -> gap 95% CI {}".format(verdict))


if __name__ == "__main__":
    main()
