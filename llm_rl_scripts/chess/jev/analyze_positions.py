"""Analyse P1 position-eval scores into per-arm, per-stratum statistics.

Reports, for each arm and each phase stratum: N, mean/median centipawn loss,
best-move and top-3 match rates, materially-losing-blunder rate, missed-forced-
mate rate, and illegal rate. Then the paired comparisons that are the actual
result -- Jev minus each baseline on the *same* positions -- with a bootstrap
95% CI clustered by source game (so several plies from one game don't count as
independent evidence). No single position can dominate: distributions are
summarised, not just pooled means.

Usage:
    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.chess.jev.analyze_positions \
      --scores position_scores.jsonl --out-dir results/primary --report report.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

PHASES = ("opening", "middlegame", "tactical", "endgame")


def load(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def arm_summary(records: List[dict]) -> Dict:
    legal = [r for r in records if r.get("is_legal")]
    cp = [r["cp_loss"] for r in legal if r.get("cp_loss") is not None]
    # Centipawn means are only meaningful on non-mate positions: a forced-mate
    # best move maps to a huge magnitude that would swamp the average. Mate
    # positions are reported via the missed-mate rate instead.
    cp_nonmate = [r["cp_loss"] for r in legal
                  if r.get("cp_loss") is not None and not r.get("best_is_mate")]
    return {
        "n": len(records),
        "n_legal": len(legal),
        "illegal_rate": 1.0 - len(legal) / len(records) if records else None,
        "mean_cp_loss_nonmate": _mean(cp_nonmate),
        "median_cp_loss": _median(cp),
        "median_cp_loss_nonmate": _median(cp_nonmate),
        "match_best": _mean([1.0 if r.get("match_best") else 0.0 for r in legal]),
        "match_top3": _mean([1.0 if r.get("match_top3") else 0.0 for r in legal]),
        "blunder_rate": _mean([1.0 if r.get("blunder") else 0.0 for r in legal]),
        "missed_mate_rate": _mean([1.0 if r.get("lost_forced_mate") else 0.0
                                   for r in legal if r.get("best_is_mate")]),
        "n_mate_positions": sum(1 for r in legal if r.get("best_is_mate")),
    }


def bootstrap_paired_cluster(
    pairs: List[Tuple[float, object]], iters: int = 5000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean paired difference + 95% CI, resampling clusters (source games)."""
    by_cluster: Dict[object, List[float]] = defaultdict(list)
    for diff, cluster in pairs:
        by_cluster[cluster].append(diff)
    clusters = list(by_cluster.keys())
    all_diffs = [d for d, _ in pairs]
    point = statistics.fmean(all_diffs) if all_diffs else 0.0
    if len(clusters) < 2:
        return point, point, point
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample: List[float] = []
        for _ in range(len(clusters)):
            sample.extend(by_cluster[rng.choice(clusters)])
        if sample:
            means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return point, lo, hi


def paired_diffs(records: List[dict], arm_a: str, arm_b: str, field: str):
    """Per-position (arm_a - arm_b) differences on `field`, with cluster id."""
    by_pos: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_pos[r["position_id"]][r["arm"]] = r
    pairs = []
    for pid, byarm in by_pos.items():
        a, b = byarm.get(arm_a), byarm.get(arm_b)
        if not a or not b:
            continue
        va = a.get(field)
        vb = b.get(field)
        if va is None or vb is None:
            continue
        pairs.append((float(va) - float(vb), a.get("source_game", pid)))
    return pairs


def fmt(x: Optional[float], pct: bool = False, nd: int = 1) -> str:
    if x is None:
        return "  -  "
    return ("{:.{}f}%".format(100 * x, nd) if pct else "{:.{}f}".format(x, nd))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--report", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = load(args.scores)
    arms = sorted({r["arm"] for r in records})
    lines: List[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("# P1 fixed-position move quality\n")
    out("Records: {}  |  arms: {}\n".format(len(records), ", ".join(arms)))

    # Overall per-arm table. Centipawn means exclude forced-mate positions (see
    # arm_summary); mate positions are summarised by missed-mate rate.
    out("## Per-arm (all positions)\n")
    out("cpL columns exclude forced-mate positions; mate handled via 'missed mate'.\n")
    out("| arm | n | illegal | mean cpL* | median cpL* | match@1 | match@3 | blunder | missed mate |")
    out("|-----|---|---------|-----------|-------------|---------|---------|---------|-------------|")
    overall = {}
    for arm in arms:
        s = arm_summary([r for r in records if r["arm"] == arm])
        overall[arm] = s
        out("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm, s["n"], fmt(s["illegal_rate"], pct=True), fmt(s["mean_cp_loss_nonmate"]),
            fmt(s["median_cp_loss_nonmate"]), fmt(s["match_best"], pct=True),
            fmt(s["match_top3"], pct=True), fmt(s["blunder_rate"], pct=True),
            "{} (n={})".format(fmt(s["missed_mate_rate"], pct=True), s["n_mate_positions"])))
    out("\n*cpL = centipawn loss vs best legal move, non-mate positions only.\n")

    # Per-phase median cpL (robust to the few mate positions in each phase).
    out("## Median centipawn loss by phase (non-mate positions)\n")
    out("| arm | " + " | ".join(PHASES) + " |")
    out("|-----|" + "|".join(["---"] * len(PHASES)) + "|")
    for arm in arms:
        cells = []
        for ph in PHASES:
            s = arm_summary([r for r in records if r["arm"] == arm and r["phase"] == ph])
            cells.append("{} (n={})".format(fmt(s["median_cp_loss_nonmate"]), s["n_legal"]))
        out("| {} | {} |".format(arm, " | ".join(cells)))
    out()

    # Paired Jev-vs-baseline comparisons.
    if "jev" in arms:
        out("## Paired comparison: jev - baseline (same positions)\n")
        out("Negative mean cpL difference = Jev loses *fewer* centipawns than the baseline (better).\n")
        out("| vs | metric | mean diff | 95% CI (cluster bootstrap) |")
        out("|----|--------|-----------|----------------------------|")
        nonmate = [r for r in records if not r.get("best_is_mate")]
        for base in [a for a in arms if a != "jev"]:
            for field, label, recs in [("cp_loss", "cp loss (non-mate)", nonmate),
                                       ("match_best", "match@1", records),
                                       ("blunder", "blunder", records)]:
                pairs = paired_diffs(recs, "jev", base, field)
                pt, lo, hi = bootstrap_paired_cluster(pairs, seed=args.seed)
                out("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}]  (n={})|".format(
                    base, label, pt, lo, hi, len(pairs)))
        out()

    report_text = "\n".join(lines) + "\n"
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(overall, f, indent=2)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report_text)
        print("\nwrote report -> {}".format(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
