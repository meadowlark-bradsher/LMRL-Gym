"""Analyze one CD-MIS-1 result dir written by ``run_mis.py``: per-arm statistics per true process,
paired cluster-bootstrap contrasts between arms (outcome Y and two diagnostics), and cross-process
contrasts (each process minus ``matched``) per arm. Skips error stubs. Writes ``analysis.json``.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.analyze_mis --dir results/<dir> [--B 5000 --seed 41]

Cluster = the private type pair (buyer ranking, seller ranking): 36 clusters (design section 8).
Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple

CELL_SETS = ("all", "focal_opens", "opponent_opens")
CLUSTER_DEFINITION = "(buyer ranking, seller ranking) type pair; 36 clusters"


def _clean(x):
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return 0.0 if x == 0 else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def load_runs(path: str) -> Tuple[List[dict], int]:
    recs, stubs = [], 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                stubs += 1
            else:
                recs.append(r)
    return recs, stubs


def cluster_of(r: dict) -> tuple:
    f, o = tuple(r["focal_ranking"]), tuple(r["private_truth_evaluator_only"]["opp_ranking"])
    b, s = (f, o) if r["focal"] == "buyer" else (o, f)
    return ("/".join(b), "/".join(s))


def in_set(r: dict, cell_set: str) -> bool:
    if cell_set == "all":
        return True
    return (r["focal"] == r["opener"]) if cell_set == "focal_opens" else (r["focal"] != r["opener"])


# ---- outcomes -------------------------------------------------------------------------------

def y(r: dict) -> float:
    return r["normalized_net_gain"]["float"]


def final_logloss(r: dict) -> float:
    return r["final_agent"]["logloss_true_bits"]


def regret_true(r: dict) -> float:
    return r["total_decision_regret_true"]


OUTCOMES: Dict[str, Callable[[dict], float]] = {
    "normalized_net_gain": y,
    "final_logloss_true_bits": final_logloss,
    "total_decision_regret_true": regret_true,
}


# ---- bootstrap -------------------------------------------------------------------------------

def cluster_means(recs: List[dict], cell_set: str, fn: Callable[[dict], float]) -> Dict[tuple, float]:
    by: Dict[tuple, List[float]] = defaultdict(list)
    for r in recs:
        if in_set(r, cell_set):
            by[cluster_of(r)].append(fn(r))
    return {k: mean(v) for k, v in by.items()}


def _percentile_ci(draws: List[float]) -> List[float]:
    draws = sorted(draws)
    B = len(draws)
    return [draws[int(0.025 * B)], draws[min(B - 1, int(0.975 * B))]]


def cluster_ci(recs: List[dict], fn: Callable[[dict], float], B: int, seed: int) -> Optional[List[float]]:
    by: Dict[tuple, List[float]] = defaultdict(list)
    for r in recs:
        by[cluster_of(r)].append(fn(r))
    vals = [mean(v) for v in by.values()]
    if len(vals) < 2 or B <= 0:
        return None
    rng = random.Random(seed)
    n = len(vals)
    return _percentile_ci([sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)])


def paired_bootstrap(a: Dict[tuple, float], b: Dict[tuple, float], B: int, seed: int) -> dict:
    keys = sorted(set(a) & set(b))
    skipped = len(set(a) ^ set(b))
    if not keys:
        return {"point": None, "ci95": None, "n_clusters": 0, "skipped_clusters": skipped, "B": B}
    diffs = [a[k] - b[k] for k in keys]
    point = mean(diffs)
    n = len(diffs)
    if n < 2 or B <= 0:
        return {"point": point, "ci95": None, "n_clusters": n, "skipped_clusters": skipped, "B": B}
    rng = random.Random(seed)
    ci = _percentile_ci([sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)])
    return {"point": point, "ci95": ci, "n_clusters": n, "skipped_clusters": skipped, "B": B}


# ---- per-arm statistics -----------------------------------------------------------------------

def _mean_opt(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def per_arm_stats(recs: List[dict], B: int, seed: int) -> dict:
    if not recs:
        return {"n": 0}
    n_dec = sum(r["n_decisions"] for r in recs)
    return {
        "n": len(recs),
        "n_clusters": len({cluster_of(r) for r in recs}),
        "mean_normalized_net_gain": mean(y(r) for r in recs),
        "ci95_normalized_net_gain": cluster_ci(recs, y, B, seed),
        "mean_net_gain": mean(r["net_gain"]["float"] for r in recs),
        "mean_gross_gain": mean(r["gross_gain"]["float"] for r in recs),
        "agreement_rate": mean(1.0 if r["agreement"] else 0.0 for r in recs),
        "mean_questions_asked": mean(r["questions_asked"] for r in recs),
        "mean_total_decision_regret_true": mean(regret_true(r) for r in recs),
        "zero_regret_decision_share": (sum(r["zero_regret_decisions"] for r in recs) / n_dec) if n_dec else None,
        "mean_final_p_true": mean(r["final_agent"]["p_true"] for r in recs),
        "mean_final_logloss_true_bits": mean(final_logloss(r) for r in recs),
        "mean_final_entropy_bits": mean(r["final_agent"]["entropy_bits"] for r in recs),
        "final_argmax_true_share": mean(1.0 if r["final_agent"]["argmax_is_true"] else 0.0 for r in recs),
        "mean_final_max_wrong": mean(r["final_agent"]["max_wrong"] for r in recs),
        "mean_true_filter_p_true": mean(r["final_true_filter"]["p_true"] for r in recs),
        "mean_predictive_logloss_bits": _mean_opt([r.get("mean_predictive_logloss_bits") for r in recs]),
        "mean_brier_accept": _mean_opt([r.get("brier_accept") for r in recs]),
        "walk_rate": mean(1.0 if r["terminal"]["reason"] == "walk" else 0.0 for r in recs),
    }


# ---- pairs ------------------------------------------------------------------------------------

def arm_pairs(arms: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if "A_prior" in arms:
        pairs += [(a, "A_prior") for a in arms if a != "A_prior"]
    if "B_naive" in arms:
        pairs += [(a, "B_naive") for a in arms if a not in ("B_naive", "A_prior")]
    if "C_mix0.2" in arms and "C_naive" in arms:
        pairs.append(("C_mix0.2", "C_naive"))
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args(argv)

    summary = json.load(open(os.path.join(args.dir, "summary.json")))
    recs, stubs = load_runs(os.path.join(args.dir, "runs.jsonl.gz"))
    processes = [p for p in summary.get("processes", []) if any(r["process"] == p for r in recs)]
    for p in sorted({r["process"] for r in recs}):
        if p not in processes:
            processes.append(p)
    arms = [a for a in summary.get("arms", []) if any(r["arm"] == a for r in recs)]
    for a in sorted({r["arm"] for r in recs}):
        if a not in arms:
            arms.append(a)
    by: Dict[str, Dict[str, List[dict]]] = {p: {a: [] for a in arms} for p in processes}
    for r in recs:
        by[r["process"]][r["arm"]].append(r)

    per_arm = {p: {a: {cs: per_arm_stats([r for r in by[p][a] if in_set(r, cs)], args.B, args.seed) for cs in CELL_SETS}
                   for a in arms if by[p][a]} for p in processes}

    pairs = arm_pairs(arms)
    contrasts = {}
    contrasts_diag = {o: {} for o in OUTCOMES if o != "normalized_net_gain"}
    for p in processes:
        contrasts[p] = {}
        for o in contrasts_diag:
            contrasts_diag[o][p] = {}
        for x, z in pairs:
            if not by[p][x] or not by[p][z]:
                continue
            name = f"{x}-{z}"
            contrasts[p][name] = {cs: paired_bootstrap(cluster_means(by[p][x], cs, y), cluster_means(by[p][z], cs, y), args.B, args.seed)
                                  for cs in CELL_SETS}
            for o, fn in OUTCOMES.items():
                if o == "normalized_net_gain":
                    continue
                contrasts_diag[o][p][name] = {cs: paired_bootstrap(cluster_means(by[p][x], cs, fn), cluster_means(by[p][z], cs, fn), args.B, args.seed)
                                              for cs in CELL_SETS}

    cross_process = {"mean_normalized_net_gain": {}, "vs_matched": {}}
    for a in arms:
        cross_process["mean_normalized_net_gain"][a] = {p: per_arm[p][a]["all"]["mean_normalized_net_gain"] for p in processes if a in per_arm[p]}
        if "matched" in processes and by["matched"].get(a):
            base = cluster_means(by["matched"][a], "all", y)
            cross_process["vs_matched"][a] = {}
            for p in processes:
                if p == "matched" or not by[p].get(a):
                    continue
                c = paired_bootstrap(cluster_means(by[p][a], "all", y), base, args.B, args.seed)
                cross_process["vs_matched"][a][p] = {"point": c["point"], "ci95": c["ci95"], "n_clusters": c["n_clusters"]}

    analysis = {
        "env": summary.get("env"), "design": summary.get("design"), "config": summary.get("config"),
        "assumed_kernel": summary.get("assumed_kernel"), "processes": processes, "arms": arms, "filters": summary.get("filters"),
        "repeats": summary.get("repeats"), "seed": summary.get("seed"), "episodes": len(recs), "error_stubs": stubs,
        "elapsed_s": summary.get("elapsed_s"), "bootstrap": {"B": args.B, "seed": args.seed}, "cluster_definition": CLUSTER_DEFINITION,
        "per_arm": per_arm, "contrasts": contrasts, "contrasts_diag": contrasts_diag, "cross_process": cross_process,
    }
    out = os.path.join(args.dir, "analysis.json")
    with open(out, "w") as f:
        json.dump(_clean(analysis), f, indent=1)
    print(f"wrote {out}: {len(recs)} episodes, {stubs} stubs, processes {processes}, arms {arms}")
    for p in processes:
        for name, c in contrasts[p].items():
            a = c["all"]
            if a["point"] is not None:
                ci = f"[{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]" if a["ci95"] else "[no CI]"
                print(f"  {p:12s} {name:24s} all: {a['point']:+.4f} {ci} n={a['n_clusters']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
