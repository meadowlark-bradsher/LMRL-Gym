"""Horizon x regime sweep of the scripted arms (exact-on-logs except ``random``).

For rounds in 1..6 and both regimes, over all 72 (role, own type, opponent
type) cells: the menu and unconstrained reference values, the realised mean
utility of the exact / greedy-EIG / myopic arms, the seeded random arm's mean
over ``--random-repeats`` plays, the arms' question and offer counts, and the
number of cells in which greedy-EIG strictly underperforms the exact policy.
This is the design evidence for the choice of the pre-registered horizon.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.sweep_horizon \
        --out results/binned_b_sweep_20261026/sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .arms import ARMS
from .env import ASK, OFFER, Config, ScriptedOpponent, annotate, initial_state, menu, other_role, transition
from .reference import ExactReference


def play(arm_fn, cfg, role, own, opp, ref, seed=0):
    rng = random.Random(seed)
    s = initial_state(role, own)
    o = ScriptedOpponent(other_role(role), opp, cfg, random.Random(seed))
    asked = offers = 0
    while True:
        rows = annotate(s, cfg)
        q = ref.q_values(s)
        key = arm_fn(s, rows, q, rng)
        opt = next(x for x in menu(s, cfg) if x.key == key)
        asked += opt.kind == ASK
        offers += opt.kind == OFFER
        out = o.respond(opt, role)
        nxt, term = transition(s, opt, out, cfg)
        if nxt is None:
            return term, asked, offers
        s = nxt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", default="1,2,3,4,5,6")
    p.add_argument("--regimes", default="easy,hard")
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    t0 = time.time()
    rows = []
    for regime in args.regimes.split(","):
        for rounds in (int(x) for x in args.rounds.split(",")):
            cfg = Config(regime=regime, rounds=rounds)
            acc = {k: Fraction(0) for k in ("V_menu", "V_full", "exact", "greedy_eig", "myopic")}
            counts = {"exact_asked": 0, "exact_offers": 0, "greedy_asked": 0, "greedy_offers": 0, "myopic_offers": 0}
            rnd = 0.0
            worse = 0
            n = 0
            for role in ("buyer", "seller"):
                for own in ALL_RANKINGS:
                    ref = ExactReference(cfg, role, own)
                    full = ExactReference(cfg, role, own, full_menu=True)
                    for opp in ALL_RANKINGS:
                        n += 1
                        acc["V_menu"] += ref.prior_value()
                        acc["V_full"] += full.prior_value()
                        u, a, of = play(ARMS["exact"], cfg, role, own, opp, ref)
                        acc["exact"] += u
                        counts["exact_asked"] += a
                        counts["exact_offers"] += of
                        g, ga, go = play(ARMS["greedy_eig"], cfg, role, own, opp, ref)
                        acc["greedy_eig"] += g
                        counts["greedy_asked"] += ga
                        counts["greedy_offers"] += go
                        m, _, mo = play(ARMS["myopic"], cfg, role, own, opp, ref)
                        acc["myopic"] += m
                        counts["myopic_offers"] += mo
                        worse += g < u
                        rnd += sum(
                            float(play(ARMS["random"], cfg, role, own, opp, ref, seed=args.seed * 1000 + k)[0])
                            for k in range(args.random_repeats)
                        ) / args.random_repeats
            rows.append(
                {
                    "regime": regime,
                    "reservation": str(cfg.reservation),
                    "rounds": rounds,
                    "decisions_max": cfg.final_t,
                    "cells": n,
                    "mean": {k: {"exact": str(v / n), "float": float(v / n)} for k, v in acc.items()},
                    "mean_random": rnd / n,
                    "random_repeats": args.random_repeats,
                    "per_cell_counts": {k: v / n for k, v in counts.items()},
                    "cells_greedy_below_exact": worse,
                    "greedy_equals_exact_everywhere": worse == 0 and acc["greedy_eig"] == acc["exact"],
                    "menu_rule_costless": acc["V_menu"] == acc["V_full"],
                }
            )
            print(f"{regime:5s} rounds={rounds} V={float(acc['V_menu']/n):.4f} greedy={float(acc['greedy_eig']/n):.4f} myopic={float(acc['myopic']/n):.4f} random={rnd/n:.4f} greedy<exact in {worse}/{n}", file=sys.stderr)
    out = {"game": cfg.to_json()["game"], "seed": args.seed, "elapsed_s": round(time.time() - t0, 1), "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
