"""Cluster-bootstrap CIs for the selector arms.

Jev's Choice is stochastic at near-ties, so each selector arm is run as a census
with K independent rollouts per object. The resampling unit is the **object**
(each object contributes all its rollouts), matching the Wordle census analyzer,
because objects are the independent draws and rollouts within an object are not.

Metrics per arm, with a percentile bootstrap CI (B resamples of the objects):
solve rate, avg questions, selection match@1 (over selector turns), mean regret
in bits. The point estimate is the pooled value on the full data.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_selector \
        --arms none=results/selector_rep_census_.../episodes.jsonl \
               norepeat=... history=... --out results/.../selector_ci.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import Dict, List, Tuple


def load_arm(path: str) -> Dict[str, List[dict]]:
    """object name -> list of its episode records."""
    by_obj: Dict[str, List[dict]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_obj.setdefault(rec["true_name"], []).append(rec)
    return by_obj


def _pooled_metrics(objs: List[str], by_obj: Dict[str, List[dict]]) -> Dict[str, float]:
    won = tot = 0
    qsum = 0.0
    picks = matched = 0
    regret_sum = 0.0
    for o in objs:
        for ep in by_obj[o]:
            tot += 1
            won += 1 if ep["won"] else 0
            qsum += ep["n_questions"]
            for st in ep["steps"]:
                if st.get("sel_match_top") is not None:
                    picks += 1
                    matched += 1 if st["sel_match_top"] else 0
                    regret_sum += st.get("sel_regret_bits") or 0.0
    return {
        "solve_rate": won / tot if tot else float("nan"),
        "avg_questions": qsum / tot if tot else float("nan"),
        "match_at_1": matched / picks if picks else float("nan"),
        "mean_regret_bits": regret_sum / picks if picks else float("nan"),
        "_n_ep": tot, "_n_picks": picks,
    }


def bootstrap_arm(by_obj: Dict[str, List[dict]], B: int, seed: int) -> dict:
    objs = sorted(by_obj)
    point = _pooled_metrics(objs, by_obj)
    rng = random.Random(seed)
    keys = ["solve_rate", "avg_questions", "match_at_1", "mean_regret_bits"]
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    n = len(objs)
    for _ in range(B):
        sample = [objs[rng.randrange(n)] for _ in range(n)]
        m = _pooled_metrics(sample, by_obj)
        for k in keys:
            draws[k].append(m[k])

    def ci(vals: List[float]) -> Tuple[float, float]:
        s = sorted(v for v in vals if v == v)  # drop nan
        if not s:
            return (float("nan"), float("nan"))
        lo = s[int(0.025 * len(s))]
        hi = s[min(len(s) - 1, int(0.975 * len(s)))]
        return (lo, hi)

    out = {
        "n_objects": n,
        "n_episodes": point["_n_ep"],
        "n_selector_picks": point["_n_picks"],
        "rollouts_per_object": point["_n_ep"] // n if n else None,
    }
    for k in keys:
        lo, hi = ci(draws[k])
        out[k] = {"point": round(point[k], 4) if point[k] == point[k] else None,
                  "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=path/to/episodes.jsonl ...")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {"boot": args.boot, "seed": args.seed, "arms": {}}
    for spec in args.arms:
        label, path = spec.split("=", 1)
        by_obj = load_arm(path)
        result["arms"][label] = {"path": path, **bootstrap_arm(by_obj, args.boot, args.seed)}

    print(json.dumps(result, indent=2))
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print("saved -> {}".format(args.out))


if __name__ == "__main__":
    main()
