"""Summarise a full-game ladder JSONL under the fixed termination accounting.

Reads the per-game records written by ``eval_jev.py --jsonl`` and reports, per
opponent rung: finished vs truncated counts, W/D/L over *finished* games, score
(W+D/2 over finished only), and the termination-reason histogram. The point is
to separate genuine draws from move-cap truncations -- which the original runner
collapsed into one "draw" bucket.

    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.chess.jev.analyze_ladder --ladder results/ladder_20261025/ladder.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

RUNG_ORDER = ["random", "sf_depth1", "sf_skill0", "sf_elo1320"]


def load(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(rows: List[dict]) -> Dict:
    finished = [r for r in rows if r["result"] != "truncated"]
    w = int(sum(r["win"] for r in finished))
    d = int(sum(r["draw"] for r in finished))
    lo = int(sum(r["loss"] for r in finished))
    trunc = int(sum(r["truncated"] for r in rows))
    nf = len(finished)
    return {
        "n": len(rows),
        "finished": nf,
        "truncated": trunc,
        "W": w, "D": d, "L": lo,
        "score_finished": (w + 0.5 * d) / nf if nf else None,
        "real_draws": d,
        "terminations": dict(Counter(r["termination"] for r in rows)),
        "avg_moves": sum(r["moves"] for r in rows) / len(rows) if rows else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ladder", required=True)
    args = p.parse_args()
    rows = load(args.ladder)
    opps = [o for o in RUNG_ORDER if any(r["opponent"] == o for r in rows)]
    opps += sorted({r["opponent"] for r in rows} - set(opps))

    print("| rung | n | finished | truncated | W/D/L (finished) | real draws | score(fin) |")
    print("|------|---|----------|-----------|------------------|------------|------------|")
    for opp in opps:
        s = summarize([r for r in rows if r["opponent"] == opp])
        sc = "{:.3f}".format(s["score_finished"]) if s["score_finished"] is not None else "-"
        print("| {} | {} | {} | {} | {}/{}/{} | {} | {} |".format(
            opp, s["n"], s["finished"], s["truncated"], s["W"], s["D"], s["L"],
            s["real_draws"], sc))
    print()
    for opp in opps:
        s = summarize([r for r in rows if r["opponent"] == opp])
        print("{:<11s} terminations: {}  avg_moves={:.1f}".format(
            opp, s["terminations"], s["avg_moves"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
