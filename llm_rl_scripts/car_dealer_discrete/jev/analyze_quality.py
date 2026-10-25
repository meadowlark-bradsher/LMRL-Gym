"""Paired, type-pair-clustered bootstrap comparison of two J/J conditions.

Built for PREREGISTRATION_QUALITY.md (bayes vs prior-only), but generic: pass any
two condition directories and the first minus the second is reported.

Unit of inference is the ordered type pair (buyer_rank, seller_rank). For each
bootstrap draw the pairs are resampled with replacement and BOTH conditions'
episodes for the drawn pairs are pooled, so the comparison is paired at the
cluster level and respects within-pair dependence across repeats and starts.

Per-episode quality metrics follow the spec's section 4:
  nash_ratio      Q_N (no-deal = 0)                 -- PRIMARY (Δ)
  min_gain        min(G_B, G_S) (no-deal = 0)
  buyer_gain, seller_gain
  pareto          1/0, over valid agreements only
  proposals       proposals_used, over valid agreements only
  agreement       1/0

Error stubs (episodes that failed after retries) are excluded by the loader and
the count is reported.

Example:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_quality \
        results/jj_quality_20261025/bayes results/jj_quality_20261025/prior_only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.jev.analyze_aid import epistemic
from llm_rl_scripts.car_dealer_discrete.jev.analyze_salience import analyze, load

PairKey = Tuple[Tuple[str, ...], Tuple[str, ...]]
METRICS = ["nash_ratio", "min_gain", "buyer_gain", "seller_gain", "pareto", "proposals", "agreement"]
PRIMARY = "nash_ratio"


def _count_stubs(path: str) -> int:
    runs = path if path.endswith(".jsonl") else os.path.join(path, "runs.jsonl")
    with open(runs) as f:
        return sum(1 for line in f if line.strip() and "\"error\"" in line and "\"result\"" not in line)


def episode_metrics(r: dict) -> Dict[str, Optional[float]]:
    sc, res = r["score"], r["result"]
    valid = bool(sc["valid_agreement"])
    return {
        "nash_ratio": float(sc["nash_ratio"]) if sc["nash_ratio"] is not None else 0.0,
        "min_gain": float(sc["min_gain"]) if sc["min_gain"] is not None else 0.0,
        "buyer_gain": float(sc["buyer_gain"]) if sc["buyer_gain"] is not None else 0.0,
        "seller_gain": float(sc["seller_gain"]) if sc["seller_gain"] is not None else 0.0,
        "pareto": (1.0 if sc["on_pareto_frontier"] else 0.0) if valid else None,
        "proposals": float(res["proposals_used"]) if valid else None,
        "agreement": 1.0 if res["agreement"] else 0.0,
    }


def by_pair(recs: List[dict]) -> Dict[PairKey, List[Dict[str, Optional[float]]]]:
    out: Dict[PairKey, List[Dict[str, Optional[float]]]] = {}
    for r in recs:
        t = r["private_truth_evaluator_only"]
        key = (tuple(t["buyer_rank"]), tuple(t["seller_rank"]))
        out.setdefault(key, []).append(episode_metrics(r))
    return out


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pooled_mean(groups: Dict[PairKey, List[dict]], pairs: List[PairKey], metric: str) -> Optional[float]:
    vals: List[Optional[float]] = []
    for p in pairs:
        vals.extend(e[metric] for e in groups[p])
    return _mean(vals)


def cluster_bootstrap(
    a: Dict[PairKey, List[dict]], b: Dict[PairKey, List[dict]], pairs: List[PairKey],
    B: int, seed: int,
) -> Dict[str, dict]:
    rng = random.Random(seed)
    draws: Dict[str, Dict[str, List[float]]] = {m: {"a": [], "b": [], "delta": []} for m in METRICS}
    n = len(pairs)
    for _ in range(B):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        for m in METRICS:
            ma, mb = pooled_mean(a, sample, m), pooled_mean(b, sample, m)
            if ma is None or mb is None:
                continue
            draws[m]["a"].append(ma); draws[m]["b"].append(mb); draws[m]["delta"].append(ma - mb)

    def ci(xs: List[float]) -> Optional[Tuple[float, float]]:
        if not xs:
            return None
        xs = sorted(xs)
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        return (round(lo, 4), round(hi, 4))

    out: Dict[str, dict] = {}
    for m in METRICS:
        pa, pb = pooled_mean(a, pairs, m), pooled_mean(b, pairs, m)
        out[m] = {
            "mean_a": round(pa, 4) if pa is not None else None,
            "mean_b": round(pb, 4) if pb is not None else None,
            "delta": round(pa - pb, 4) if (pa is not None and pb is not None) else None,
            "ci95_a": ci(draws[m]["a"]),
            "ci95_b": ci(draws[m]["b"]),
            "ci95_delta": ci(draws[m]["delta"]),
            "n_boot": len(draws[m]["delta"]),
        }
    return out


def per_pair_delta(a, b, pairs, metric):
    rows = []
    for p in pairs:
        ma, mb = _mean([e[metric] for e in a[p]]), _mean([e[metric] for e in b[p]])
        if ma is None or mb is None:
            continue
        rows.append({"buyer": ">".join(p[0]), "seller": ">".join(p[1]),
                     "a": round(ma, 3), "b": round(mb, 3), "delta": round(ma - mb, 3),
                     "n_a": len(a[p]), "n_b": len(b[p])})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cond_a", help="Condition A directory (reported as A - B).")
    ap.add_argument("cond_b", help="Condition B directory.")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    recs_a, recs_b = load(args.cond_a), load(args.cond_b)
    stubs = {"a": _count_stubs(args.cond_a), "b": _count_stubs(args.cond_b)}
    ga, gb = by_pair(recs_a), by_pair(recs_b)
    pairs = sorted(set(ga) & set(gb))
    only_a, only_b = sorted(set(ga) - set(gb)), sorted(set(gb) - set(ga))

    boot = cluster_bootstrap(ga, gb, pairs, args.boot, args.seed)
    pp = per_pair_delta(ga, gb, pairs, PRIMARY)
    signs = {"pos": sum(1 for r in pp if r["delta"] > 0), "neg": sum(1 for r in pp if r["delta"] < 0),
             "zero": sum(1 for r in pp if r["delta"] == 0)}
    traj_a, traj_b = analyze(recs_a), analyze(recs_b)

    name_a, name_b = os.path.basename(os.path.normpath(args.cond_a)), os.path.basename(os.path.normpath(args.cond_b))
    report = {
        "a": name_a, "b": name_b,
        "episodes": {"a": len(recs_a), "b": len(recs_b)},
        "failed_episodes_excluded": stubs,
        "pairs_compared": len(pairs), "pairs_only_in_a": len(only_a), "pairs_only_in_b": len(only_b),
        "bootstrap": {"B": args.boot, "seed": args.seed, "unit": "ordered type pair (paired resampling)"},
        "primary": PRIMARY,
        "metrics": boot,
        "per_pair_primary_delta": pp,
        "primary_sign_count": signs,
        "sanity": {
            "ideal_offers_a": f"{traj_a['offers_total'] - traj_a['offers_nonideal_own_lt_1']}/{traj_a['offers_total']}",
            "ideal_offers_b": f"{traj_b['offers_total'] - traj_b['offers_nonideal_own_lt_1']}/{traj_b['offers_total']}",
            "agreement_a": boot["agreement"]["mean_a"], "agreement_b": boot["agreement"]["mean_b"],
        },
        "epistemic_a": epistemic(recs_a), "epistemic_b": epistemic(recs_b),
    }
    prim = boot[PRIMARY]["ci95_delta"]
    report["decision"] = (
        "undetermined" if prim is None else
        "learning improves quality (primary CI excludes 0, positive)" if prim[0] > 0 else
        "learning HURTS quality (primary CI excludes 0, negative)" if prim[1] < 0 else
        "no resolved benefit (primary CI includes 0)"
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str)); return 0

    print(f"A = {name_a} ({len(recs_a)} eps, {stubs['a']} failed excluded)   B = {name_b} ({len(recs_b)} eps, {stubs['b']} failed excluded)")
    print(f"pairs compared: {len(pairs)}   bootstrap: B={args.boot} seed={args.seed}, paired over type pairs")
    print(f"sanity: ideal offers A={report['sanity']['ideal_offers_a']} B={report['sanity']['ideal_offers_b']}   agreement A={boot['agreement']['mean_a']} B={boot['agreement']['mean_b']}")
    if report["epistemic_a"]:
        e = report["epistemic_a"]; print(f"learning (A): improvement bits={e['logscore_improvement_over_uniform_bits']}  argmax=true={e['argmax_is_true_type_rate']}")
    print()
    print(f"{'metric':<13}{'A':>9}{'B':>9}{'Δ(A-B)':>10}   {'95% CI for Δ':<22}{'':<3}")
    for m in METRICS:
        r = boot[m]; tag = "  <- PRIMARY" if m == PRIMARY else ""
        print(f"{m:<13}{r['mean_a']!s:>9}{r['mean_b']!s:>9}{r['delta']!s:>10}   {str(r['ci95_delta']):<22}{tag}")
    print(f"\nper-pair Δ{PRIMARY} sign count: +{signs['pos']} / -{signs['neg']} / 0:{signs['zero']}  (of {len(pp)})")
    print(f"\nDECISION (pre-registered rule): {report['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
