"""ST-3 exploratory budget-end split (NOT a registered analysis).

Every ``wrong`` in ST-3 is a game that used all 20 questions without a confirmed
guess and ended with a known object, not UNKNOWN, on top
(``openworld.run_openworld_episode``: "expert iff UNKNOWN is on top, else a wrong
guess"). This splits each cell's games by outcome and, for ``wrong`` games, by
pool (in-pool or novel) and by the live-set size when the budget ran out
(``n_live_final``; -1 for ``soft``, which has no live set). It also counts expert
calls made at the budget end (20 questions asked).

The split motivates ``STRESS_ST3_ENDGAME_PREREG.md``. It reads ST-3's committed
per-game columns only; no game is played. Writes ``budget_end.json`` into the run
directory.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.st3_budget_end \
        --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_20261113
"""
import argparse
import glob
import gzip
import json
import os
from collections import Counter, defaultdict

WON, EXPERT, WRONG, FALSE_CONFIRM = 0, 1, 2, 3
BUDGET = 20
ARMS = [("primary", "soft"), ("primary", "modeswitch_guard"), ("norepeat", "modeswitch_guard")]


def split(run_dir: str, prior: str = "true") -> dict:
    acc = defaultdict(Counter)
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        j, g = rec["job"], rec["games"]
        if j["prior"] != prior or (j["arm"], j["backend"]) not in ARMS:
            continue
        a = acc[(j["sigma"], j["flip"], j["arm"], j["backend"])]
        for i in range(len(g["obj"])):
            pool = "in_pool" if g["in_pool"][i] else "novel"
            o = g["outcome"][i]
            a["games"] += 1
            a["{}_{}".format(pool, ("won", "expert", "wrong", "false_confirm")[o])] += 1
            if o == WRONG:
                nl = g["n_live_final"][i]
                a["wrong_{}_live_{}".format(pool, "na" if nl < 0 else min(nl, 3))] += 1
            if o == EXPERT and g["n_questions"][i] == BUDGET:
                a["budget_end_expert_{}".format(pool)] += 1
    cells = []
    for key in sorted(acc):
        a = acc[key]
        n = a.pop("games")
        cells.append({"sigma": key[0], "flip": key[1], "arm": key[2], "backend": key[3],
                      "games": n, "share": {k: v / n for k, v in sorted(a.items())}})
    return {"schema": "twenty_questions_st3_budget_end_v1", "run_dir": run_dir,
            "prior": prior, "note": "exploratory, not registered; live_3 means >= 3",
            "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    out = split(a.run_dir)
    with open(os.path.join(a.run_dir, "budget_end.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for c in out["cells"]:
        s = c["share"]
        if c["flip"] != 0.0 or c["sigma"] < 0.5:
            continue
        print("s={sigma} {arm}/{backend}".format(**c))
        for k in sorted(s):
            if k.startswith("wrong_") or k.startswith("budget_end"):
                print("   {:28s} {:.4f}".format(k, s[k]))
    print("wrote {}".format(os.path.join(a.run_dir, "budget_end.json")))


if __name__ == "__main__":
    main()
