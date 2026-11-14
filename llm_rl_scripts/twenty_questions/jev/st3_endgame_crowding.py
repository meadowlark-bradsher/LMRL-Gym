"""ST-3 end-game: where the won and FPR changes come from (NOT a registered analysis).

Each end-game job plays the same stream of objects as its ST-3 `true` pair
(``analyze_st3_endgame`` checks this). So game i of the two jobs has the same secret,
and each game can be classed by its pool status under `top` and under the rule:

* ``in->in``: in-pool under both. A won game under `top` that the rule job does not
  win is *crowded*. The extra admissions made it harder. A non-won game the rule job
  wins is *rescued*.
* ``nov->in``: novel under `top`, in-pool under the rule. These are repeats of a
  species that only the rule's escalations admitted.
* ``nov->nov`` and ``in->nov``: the rest.

It reads committed per-game columns only; no game is played. It writes
``crowding.json`` into the end-game run directory.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.st3_endgame_crowding \
        --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_endgame_20261114 \
        --base-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_20261113
"""
import argparse
import glob
import gzip
import json
import os
from collections import Counter

WON, EXPERT, WRONG, FALSE_CONFIRM = 0, 1, 2, 3
ARMS = [("primary", "soft"), ("norepeat", "modeswitch_guard")]
RULES = ["half", "conf", "always"]
SIGMAS = [(0.5, "s50"), (0.75, "s75")]


def _load(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)["games"]


def cell(run_dir: str, base_dir: str, tag: str, arm: str, backend: str, rule: str) -> dict:
    c = Counter()
    n = 0
    pat = os.path.join(run_dir, "jobs", "{}_true_f0_ss*_ns0_{}_{}_end{}.json.gz".format(
        tag, arm, backend, rule))
    paths = sorted(glob.glob(pat))
    for p in paths:
        bid = os.path.basename(p)[:-len(".json.gz")].rsplit("_end", 1)[0]
        r = _load(p)
        b = _load(os.path.join(base_dir, "jobs", bid + ".json.gz"))
        if r["obj"] != b["obj"]:
            raise ValueError("stream differs from its pair: " + p)
        for i in range(len(r["obj"])):
            n += 1
            cls = ("in" if b["in_pool"][i] else "nov") + "->" + ("in" if r["in_pool"][i] else "nov")
            c[cls] += 1
            tw, rw = b["outcome"][i] == WON, r["outcome"][i] == WON
            if cls == "in->in":
                c["crowded"] += tw and not rw
                c["rescued"] += rw and not tw
                c["crowded_to_expert"] += tw and r["outcome"][i] == EXPERT
            elif cls == "nov->in":
                c["new_in_pool_won"] += rw
                c["new_in_pool_expert"] += r["outcome"][i] == EXPERT
            c["won_top"] += tw
            c["won_rule"] += rw
    out = {k: v / n for k, v in sorted(c.items())} if n else {}
    if n:
        out["won_change"] = out["won_rule"] - out["won_top"]
    return {"jobs": len(paths), "games": n, "share_of_all_games": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    a = ap.parse_args()
    cells = []
    for sigma, tag in SIGMAS:
        for arm, backend in ARMS:
            for rule in RULES:
                d = cell(a.run_dir, a.base_dir, tag, arm, backend, rule)
                d.update({"sigma": sigma, "flip": 0.0, "arm": arm, "backend": backend, "rule": rule})
                cells.append(d)
                s = d["share_of_all_games"]
                print("s={} {}/{} {:6s} new in-pool {:.3f} (won {:.3f})  crowded {:.3f} "
                      "rescued {:.3f}  won change {:+.3f}".format(
                          sigma, arm, backend, rule, s.get("nov->in", 0), s.get("new_in_pool_won", 0),
                          s.get("crowded", 0), s.get("rescued", 0), s["won_change"]))
    out = {"schema": "twenty_questions_st3_endgame_crowding_v1", "run_dir": a.run_dir,
           "base_dir": a.base_dir, "note": "exploratory, not registered; flip 0, shares of all games",
           "cells": cells}
    path = os.path.join(a.run_dir, "crowding.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote " + path)


if __name__ == "__main__":
    main()
