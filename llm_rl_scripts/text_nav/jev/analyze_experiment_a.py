"""Single source of truth for every number cited from an Experiment A run.

The report (PILOT_A_RESULTS.md and any later write-up) must read its figures
from this script's output, not carry hand-typed copies, so that regenerating is
the way to check whether a claim still holds (see the cited-numbers discipline).

Reads an experiment run directory's ``episodes.jsonl`` and writes ``summary.csv``
and ``claims.json`` beside it, and prints a human-readable summary. It computes
only from logged episode records; it does not call Jev or TextWorld.

    python -m llm_rl_scripts.text_nav.jev.analyze_experiment_a <run_dir>

Caveat on reproducibility: jev-latest's argmax was deterministic in the
determinism probe but its probabilities are noisy at small margins, so these
per-condition counts are single-run pilot values until a re-run audit measures
their spread. Treat them as stochastic (seed-varying at borderline decisions)
rather than exact.
"""

from __future__ import annotations

import collections
import csv
import json
import os
import sys
from typing import Dict, List

from llm_rl_scripts.text_nav.jev.milestones import MILESTONES

CONDITION_ORDER = ["A1", "A2", "A3", "A4"]


def load(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "episodes.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def by_condition(eps: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in eps:
        out[r["condition"]].append(r)
    return out


def condition_rows(eps: List[dict]) -> List[dict]:
    bc = by_condition(eps)
    rows = []
    for c in CONDITION_ORDER:
        rs = bc.get(c, [])
        n = len(rs)
        if n == 0:
            continue
        dg = collections.Counter()
        for r in rs:
            for k, v in r["diagnostics"].items():
                dg[k] += v
        wins = sum(1 for r in rs if r["won"])
        both = dg["opp_target_vs_distractor"]
        chose = dg["chose_target_when_both"]
        eat_opp = dg["opp_eat_offered"]
        rows.append({
            "condition": c,
            "n": n,
            "wins": wins,
            "success_rate": round(wins / n, 4),
            "mean_greatest_ordered": round(
                sum(r["milestones"]["greatest_ordered"] for r in rs) / n, 4),
            **{"ever_" + m.split("_")[0]: sum(
                1 for r in rs if r["milestones"]["ever"][m]) for m in MILESTONES},
            "seeds_with_target_opportunity": sum(
                1 for r in rs if r["diagnostics"]["opp_target_offered"] > 0),
            "take_target": dg["take_target"],
            "take_distractor": dg["take_distractor"],
            "p_target_given_both_offered": (round(chose / both, 4) if both else None),
            "n_both_offered": both,
            "eats": dg["eat"],
            "eat_opportunities": eat_opp,
            "p_eat_given_offered": (round(dg["chose_eat_when_offered"] / eat_opp, 4)
                                    if eat_opp else None),
        })
    return rows


def per_seed_table(eps: List[dict]) -> Dict[str, Dict[int, dict]]:
    """seed -> condition -> {ordered, won} for the paired view."""
    out: Dict[int, Dict[str, dict]] = collections.defaultdict(dict)
    for r in eps:
        out[r["seed"]][r["condition"]] = {
            "ordered": r["milestones"]["greatest_ordered"],
            "won": r["won"],
        }
    return out


def claims(run_dir: str) -> dict:
    eps = load(run_dir)
    rows = condition_rows(eps)
    rowmap = {r["condition"]: r for r in rows}
    seeds = per_seed_table(eps)
    a4_wins = sorted(s for s, cc in seeds.items()
                     if cc.get("A4", {}).get("won"))
    return {
        "run_dir": os.path.abspath(run_dir),
        "git_commit": json.load(open(os.path.join(run_dir, "run.json"))).get(
            "git_commit"),
        "n_seeds": max((r["n"] for r in rows), default=0),
        "conditions": rows,
        "a4_winning_seeds": a4_wins,
        "success_by_condition": {c: rowmap[c]["success_rate"] for c in rowmap},
        "mean_ordered_by_condition": {
            c: rowmap[c]["mean_greatest_ordered"] for c in rowmap},
        "only_condition_past_M1": [
            c for c in rowmap if rowmap[c]["ever_M2"] > 0],
        "eats_total": sum(r["eats"] for r in rows),
        "eat_opportunities_total": sum(r["eat_opportunities"] for r in rows),
    }


def write_outputs(run_dir: str) -> dict:
    rows = condition_rows(load(run_dir))
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    c = claims(run_dir)
    with open(os.path.join(run_dir, "claims.json"), "w") as f:
        json.dump(c, f, indent=2, default=str)
    return c


def main(argv: List[str] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: analyze_experiment_a.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = argv[0]
    c = write_outputs(run_dir)
    print("Experiment A analysis  ({} seeds, commit {})".format(
        c["n_seeds"], (c["git_commit"] or "?")[:8]))
    print("  run_dir:", c["run_dir"])
    print("\n  success / mean-ordered-milestone by condition:")
    for r in c["conditions"]:
        print("    {}: success={:.3f} ({}/{})  mean_ordered={:.2f}  "
              "everM1-5={}".format(
                  r["condition"], r["success_rate"], r["wins"], r["n"],
                  r["mean_greatest_ordered"],
                  [r["ever_" + m.split("_")[0]] for m in MILESTONES]))
    print("\n  A4 winning seeds:", c["a4_winning_seeds"])
    print("  only condition(s) ever past M1 (reach M2+):", c["only_condition_past_M1"])
    print("  eating: {} eats / {} opportunities (suppression across all cond)".format(
        c["eats_total"], c["eat_opportunities_total"]))
    print("\n  wrote summary.csv and claims.json to run_dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
