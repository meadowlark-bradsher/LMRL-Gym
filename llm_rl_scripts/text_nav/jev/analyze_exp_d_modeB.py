"""Experiment D, Mode B — a zero-API yardstick on the logged choices.

Design D.4 scores Jev's logged option-probabilities against the information-optimal
(EIG) action. But this game family has a fixed map and *unique room descriptions*
(verified: 0 of the main run's distinct observations map to >1 room), so the agent
is localized the moment it reads a room, and with deterministic observations EIG(a)
collapses to ~0 for every action after the opening — the EIG yardstick is vacuous
here (design D.1's b0 regime). This script (a) demonstrates that localization fact,
and (b) applies the yardstick design D.5 names for the exploit regime instead:
**goal-directedness** — alignment of Jev's choice with the shortest-path-optimal
action toward the goal under the known map. Both are zero-API reanalyses of the
logged steps.

The map, directions, and goal (milk on the living-room table -> into the kitchen
fridge -> close) are taken from env/make_game.py. Ground-truth world fields
(player_room, carrying, fridge state) are used for *scoring only*; they were never
in the policy prompt.

    python -m llm_rl_scripts.text_nav.jev.analyze_exp_d_modeB \
        llm_rl_scripts/text_nav/jev/results/main_A_20261025/steps.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
from typing import Dict, List, Optional, Set

# (room, direction) -> neighbour, from make_game.py's M.connect calls.
EDGES = {
    ("bedroom", "west"): "living room", ("bedroom", "south"): "bathroom",
    ("living room", "east"): "bedroom", ("living room", "west"): "dining room",
    ("living room", "south"): "office",
    ("dining room", "east"): "living room", ("dining room", "west"): "kitchen",
    ("kitchen", "east"): "dining room", ("kitchen", "north"): "garden",
    ("kitchen", "west"): "backyard",
    ("garden", "south"): "kitchen",
    ("office", "north"): "living room", ("office", "east"): "bathroom",
    ("bathroom", "west"): "office", ("bathroom", "north"): "bedroom",
    ("backyard", "east"): "kitchen",
}
ROOMS = sorted({r for (r, _) in EDGES} | set(EDGES.values()))
TARGET = "milk"


def neighbours(room: str):
    return [(d, dst) for (r, d), dst in EDGES.items() if r == room]


def bfs_dist(goal: str) -> Dict[str, int]:
    dist = {goal: 0}
    q = collections.deque([goal])
    while q:
        cur = q.popleft()
        # reverse edges: who can step INTO cur
        for (r, d), dst in EDGES.items():
            if dst == cur and r not in dist:
                dist[r] = dist[cur] + 1
                q.append(r)
    return dist


DIST_TO = {room: bfs_dist(room) for room in ("living room", "kitchen")}


def optimal_moves_toward(room: str, goal_room: str) -> Set[str]:
    """All 'go <dir>' moves that lie on a shortest path from room to goal_room."""
    if room == goal_room:
        return set()
    dist = DIST_TO[goal_room]
    here = dist.get(room)
    if here is None:
        return set()
    out = set()
    for d, dst in neighbours(room):
        if dist.get(dst, 1e9) == here - 1:
            out.add("go {}".format(d))
    return out


def optimal_actions(world: dict, admissible: List[str]) -> Optional[Set[str]]:
    """The goal-optimal admissible action(s) for this state, or None if undefined."""
    room = world.get("extra", {}).get("player_room")
    carrying = world.get("carrying_target")
    in_fridge = world.get("target_in_fridge")
    fridge_open = world.get("fridge_open")
    if room is None:
        return None

    want: Set[str] = set()
    if in_fridge and fridge_open:
        want = {c for c in admissible if c == "close fridge"}
    elif carrying:
        if room == "kitchen":
            if fridge_open:
                want = {c for c in admissible if c.startswith("insert") and TARGET in c}
            else:
                want = {c for c in admissible if c == "open fridge"}
        else:
            moves = optimal_moves_toward(room, "kitchen")
            want = {c for c in admissible if c in moves}
    else:  # need to acquire the target
        if room == "living room":
            want = {c for c in admissible if c.startswith("take") and TARGET in c}
        else:
            moves = optimal_moves_toward(room, "living room")
            want = {c for c in admissible if c in moves}
    return want or None


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def analyze(path: str) -> dict:
    rows = [json.loads(l) for l in _open(path) if l.strip()]

    # (a) localization / EIG-degeneracy fact
    obs2room = collections.defaultdict(set)
    for r in rows:
        pr = r["world"]["extra"].get("player_room")
        if pr:
            obs2room[r["observation"]].add(pr)
    ambiguous = sum(1 for v in obs2room.values() if len(v) > 1)

    # (b) goal-directedness yardstick, per condition
    by = collections.defaultdict(lambda: {"n": 0, "defined": 0, "argmax_opt": 0,
                                           "mass": 0.0, "chance": 0.0})
    for r in rows:
        c = r["condition"]
        by[c]["n"] += 1
        opt = optimal_actions(r["world"], r["admissible"])
        if not opt:
            continue
        by[c]["defined"] += 1
        if r["chosen"] in opt:
            by[c]["argmax_opt"] += 1
        probs = r.get("probabilities") or {}
        by[c]["mass"] += sum(probs.get(a, 0.0) for a in opt)
        # chance baseline: a uniform-random admissible pick lands on an optimal
        # action with probability |optimal| / |admissible|.
        by[c]["chance"] += len(opt) / len(r["admissible"])

    conds = sorted(by)
    rows_out = []
    for c in conds:
        d = by[c]
        defined = d["defined"]
        rows_out.append({
            "condition": c,
            "steps": d["n"],
            "steps_with_defined_optimal": defined,
            "coverage": round(defined / d["n"], 4) if d["n"] else None,
            "goal_optimal_argmax_rate": round(d["argmax_opt"] / defined, 4) if defined else None,
            "chance_optimal_rate": round(d["chance"] / defined, 4) if defined else None,
            "argmax_lift_over_chance": round(
                d["argmax_opt"] / defined - d["chance"] / defined, 4) if defined else None,
            "mean_prob_mass_on_optimal": round(d["mass"] / defined, 4) if defined else None,
        })
    return {
        "source": os.path.abspath(path),
        "localization": {
            "distinct_observations": len(obs2room),
            "observations_mapping_to_multiple_rooms": ambiguous,
            "eig_note": ("unique room descriptions => immediate localization => "
                         "EIG(a)~0 after the opening; EIG yardstick vacuous, "
                         "goal-directedness used instead"),
        },
        "goal_directedness": rows_out,
    }


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("steps")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    a = analyze(args.steps)
    out = args.out or os.path.dirname(os.path.abspath(args.steps))
    with open(os.path.join(out, "exp_d_modeB.json"), "w") as f:
        json.dump(a, f, indent=2, default=str)

    print("Experiment D Mode B — EIG-degeneracy + goal-directedness yardstick")
    loc = a["localization"]
    print("  localization: {}/{} distinct observations map to >1 room".format(
        loc["observations_mapping_to_multiple_rooms"], loc["distinct_observations"]))
    print("    => {}".format(loc["eig_note"]))
    print("  goal-directedness (shortest-path-optimal under known map):")
    print("    cond | cover | argmax-opt | chance | lift  | prob-mass-opt")
    for r in a["goal_directedness"]:
        print("     {}  | {:.2f}  |   {:.3f}    | {:.3f}  | {:+.3f}|    {:.3f}".format(
            r["condition"], r["coverage"], r["goal_optimal_argmax_rate"],
            r["chance_optimal_rate"], r["argmax_lift_over_chance"],
            r["mean_prob_mass_on_optimal"]))
    print("\n  wrote exp_d_modeB.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
