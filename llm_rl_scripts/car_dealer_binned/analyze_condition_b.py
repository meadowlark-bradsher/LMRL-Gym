"""Analyse Condition B logs: information efficiency, decision quality, outcome.

Per arm x role (design doc section 4):
  * outcome        mean learner utility, mean gain G = (U - r)/(U* - r),
                   agreement rate, Nash ratio, Pareto share
  * decision       mean total decision regret, share of zero-regret decisions,
                   mean optimality gap V*(prior) - realised U
  * information    mean total EIG regret (bits), questions asked, cumulative
                   and final entropy
  * reference      V*(prior) per cell (identical across cells by symmetry)

Differences vs the ``exact`` arm use a paired, type-pair-clustered bootstrap
(cluster = (role, own type, opponent type); B and seed are arguments). For the
deterministic arms (exact, greedy_eig, myopic under zero noise) every number is
exact-on-logs; the interval then only expresses variation across the type
space, not sampling noise. The ``random`` arm and any Jev arm are stochastic.

For Jev arms it also reports H3 quantities: Spearman rank correlation between
the choice entropy and the decision regret, overall and within terciles of the
best-minus-second Q gap, and EIG regret / decision regret vs the random arm.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b \
        results/binned_b_scripted_20261026 --seed 0 --json results/binned_b_scripted_20261026/analysis.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


def load(path: str) -> List[dict]:
    """Read runs.jsonl or runs.jsonl.gz from a file path or a result directory."""
    if os.path.isdir(path):
        cands = [os.path.join(path, "runs.jsonl"), os.path.join(path, "runs.jsonl.gz")]
        runs = next((c for c in cands if os.path.exists(c)), cands[0])
    else:
        runs = path
    opener = gzip.open if runs.endswith(".gz") else open
    with opener(runs, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fl(v):
    return None if v is None else v["float"]


def cluster_key(r):
    return (r["role"], tuple(r["own_ranking"]), tuple(r["private_truth_evaluator_only"]["opp_ranking"]))


def paired_cluster_bootstrap(a: Dict[tuple, List[float]], b: Dict[tuple, List[float]], B: int, seed: int):
    """Mean over clusters of (mean_a - mean_b), resampling clusters with replacement."""
    # clusters where the quantity is undefined on either side (e.g. gain on an
    # infeasible type pair) are dropped; the returned cluster count says how many remain
    keys = [k for k in sorted(set(a) & set(b)) if mean(a[k]) is not None and mean(b[k]) is not None]
    if not keys:
        return None
    diffs = [mean(a[k]) - mean(b[k]) for k in keys]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[min(B - 1, int(0.975 * B))]
    return {"delta": point, "ci95": [lo, hi], "clusters": len(keys)}


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def summarise(recs: List[dict]) -> dict:
    n = len(recs)
    decs = [d for r in recs for d in r["decisions"]]
    out = {
        "episodes": n,
        "decisions": len(decs),
        "mean_learner_utility": mean(fl(r["learner_utility"]) for r in recs),
        "mean_opponent_utility": mean(fl(r["opponent_utility"]) for r in recs),
        "mean_gain": mean(fl(r["gain"]) for r in recs),
        "agreement_rate": mean(1.0 if r["agreement"] else 0.0 for r in recs),
        "mean_nash_ratio": mean(fl(r["nash_ratio"]) for r in recs),
        "pareto_share_of_agreements": mean(1.0 if r["on_pareto_frontier"] else 0.0 for r in recs if r["agreement"]),
        "mean_reference_prior_value": mean(fl(r["reference_prior_value"]) for r in recs),
        "mean_optimality_gap": _clean(mean(fl(r["reference_prior_value"]) - fl(r["learner_utility"]) for r in recs)),
        "mean_total_decision_regret": mean(fl(r["total_decision_regret"]) for r in recs),
        "zero_regret_decision_share": mean(1.0 if d["decision_regret"]["exact"] == "0" else 0.0 for d in decs),
        "mean_total_eig_regret_bits": mean(r["total_eig_regret_bits"] for r in recs),
        "mean_eig_regret_per_decision_bits": mean(d["eig_regret_bits"] for d in decs),
        "mean_questions_asked": mean(r["questions_asked"] for r in recs),
        "mean_offers_made": mean(r["offers_made"] for r in recs),
        "mean_decisions": mean(r["n_decisions"] for r in recs),
        "mean_cumulative_entropy_bits": mean(r["cumulative_entropy_bits"] for r in recs),
        "mean_final_entropy_bits": mean(r["final_entropy_bits"] for r in recs),
        "chosen_kind_counts": dict(_count(d["chosen_kind"] for d in decs)),
        "chosen_tag_counts": dict(_count(d["chosen_tag"] for d in decs if d["chosen_tag"])),
        "accepted_deal_counts": dict(_count(r["accepted_deal_id"] for r in recs if r["accepted_deal_id"])),
    }
    jev = [d for d in decs if d.get("jev")]
    if jev:
        ent = [d["jev"]["choice_entropy_bits"] for d in jev if d["jev"]["choice_entropy_bits"] is not None]
        reg = [fl(d["decision_regret"]) for d in jev if d["jev"]["choice_entropy_bits"] is not None]
        gaps = [fl(d["q_gap_best_second"]) for d in jev if d["jev"]["choice_entropy_bits"] is not None]
        tert = _terciles(gaps)
        by_t = {}
        for name, (lo, hi) in tert.items():
            idx = [i for i, g in enumerate(gaps) if lo <= g <= hi]
            by_t[name] = {
                "n": len(idx),
                "gap_range": [lo, hi],
                "spearman_entropy_vs_regret": spearman([ent[i] for i in idx], [reg[i] for i in idx]),
                "mean_entropy_bits": mean(ent[i] for i in idx),
                "mean_regret": mean(reg[i] for i in idx),
            }
        out["jev"] = {
            "well_formed_share": mean(1.0 if d["jev"]["well_formed"] else 0.0 for d in jev),
            "mean_choice_entropy_bits": mean(ent),
            "mean_confidence": mean(d["jev"]["confidence"] for d in jev),
            "spearman_entropy_vs_decision_regret": spearman(ent, reg),
            "by_gap_tercile": by_t,
            "chosen_position_counts": dict(_count(d["jev"]["chosen_position"] for d in jev)),
        }
    return out


def _clean(x, eps=1e-12):
    """Snap float noise around an exact zero (sums of Fraction-derived floats)."""
    return None if x is None else (0.0 if abs(x) < eps else x)


def _count(it):
    c = defaultdict(int)
    for x in it:
        c[str(x)] += 1
    return dict(sorted(c.items()))


def _terciles(xs: List[float]) -> Dict[str, Tuple[float, float]]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    a, b = s[n // 3], s[(2 * n) // 3]
    return {"small_gap": (s[0], a), "mid_gap": (a, b), "large_gap": (b, s[-1])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--json", default=None, help="Write the analysis here (also printed).")
    args = p.parse_args(argv)

    recs = load(args.path)
    arms = sorted({r["arm"] for r in recs}, key=lambda a: (a != "exact", a))
    roles = sorted({r["role"] for r in recs})
    report = {"source": os.path.abspath(args.path), "episodes": len(recs), "arms": arms, "bootstrap": {"B": args.B, "seed": args.seed}, "by_arm": {}, "by_arm_role": {}, "vs_exact": {}}
    for a in arms:
        report["by_arm"][a] = summarise([r for r in recs if r["arm"] == a])
        for ro in roles:
            report["by_arm_role"][f"{a}/{ro}"] = summarise([r for r in recs if r["arm"] == a and r["role"] == ro])

    if "exact" in arms:
        def by_cluster(arm, field):
            d = defaultdict(list)
            for r in recs:
                if r["arm"] == arm:
                    d[cluster_key(r)].append(field(r))
            return d
        fields = {
            "learner_utility": lambda r: fl(r["learner_utility"]),
            "gain": lambda r: fl(r["gain"]),
            "agreement": lambda r: 1.0 if r["agreement"] else 0.0,
            "total_eig_regret_bits": lambda r: r["total_eig_regret_bits"],
            "total_decision_regret": lambda r: fl(r["total_decision_regret"]),
            "questions_asked": lambda r: float(r["questions_asked"]),
            "cumulative_entropy_bits": lambda r: r["cumulative_entropy_bits"],
        }
        for a in arms:
            if a == "exact":
                continue
            report["vs_exact"][a] = {
                name: paired_cluster_bootstrap(by_cluster(a, fn), by_cluster("exact", fn), args.B, args.seed)
                for name, fn in fields.items()
            }
    if "random" in arms:
        report["vs_random"] = {}
        for a in arms:
            if a in ("random",):
                continue
            report["vs_random"][a] = {
                name: paired_cluster_bootstrap(by_cluster(a, fn), by_cluster("random", fn), args.B, args.seed)
                for name, fn in fields.items()
            } if "exact" in arms else None

    text = json.dumps(report, indent=2)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    # compact table
    print(f"{'arm':12s} {'n':>5s} {'U':>7s} {'gain':>7s} {'agree':>6s} {'Vgap':>7s} {'Dregret':>8s} {'IGregret':>9s} {'asked':>6s} {'C_H':>6s}")
    for a in arms:
        s = report["by_arm"][a]
        print(f"{a:12s} {s['episodes']:5d} {s['mean_learner_utility']:7.4f} {s['mean_gain'] if s['mean_gain'] is not None else float('nan'):7.4f} {s['agreement_rate']:6.3f} {s['mean_optimality_gap']:7.4f} {s['mean_total_decision_regret']:8.4f} {s['mean_total_eig_regret_bits']:9.4f} {s['mean_questions_asked']:6.3f} {s['mean_cumulative_entropy_bits']:6.3f}")
    for a, d in report.get("vs_exact", {}).items():
        u = d["learner_utility"]
        print(f"  {a} - exact: ΔU = {u['delta']:+.4f} [{u['ci95'][0]:+.4f}, {u['ci95'][1]:+.4f}] over {u['clusters']} clusters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
