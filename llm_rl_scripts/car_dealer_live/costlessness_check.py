"""Costlessness check of the Phase 1 menu rule: for every (focal role, opener,
focal type) cell solve the exact oracle under the full action set and under the
shortlist menu, with ASK allowed and forbidden, for several menu sizes
(n_offers, n_questions). Writes one JSON with every root value and the losses.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.costlessness_check \
        --horizon 4 --ask-cost 0 --sizes "2x2,3x2,2x3,3x3" --out results/<dir>/costlessness.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .menu_live import Menu
from .opponent import KernelParams, T1Kernel
from .oracle import Oracle
from .protocol_live import ENV_VERSION, LiveConfig, other


def _tag(r):
    return "/".join(a[0] for a in r)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--regime", default="easy"); p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--ask-cost", default="0"); p.add_argument("--alpha0", default="4/5"); p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--p-counter", type=float, default=1.0); p.add_argument("--p-hold", type=float, default=1.0); p.add_argument("--shortlist", type=int, default=3)
    p.add_argument("--sizes", default="2x2,3x2,3x2c", help="n_offers x n_questions variants (2 = SAFE+AMBITIOUS, 3 = +PLAN; suffix c = + PLAN_ASK/PLAN_Q completeness)")
    p.add_argument("--registered", default="3x2c", help="the menu the registration uses; the stopping rule is evaluated on it")
    p.add_argument("--stop-threshold", type=float, default=0.005, help="|mean Δ*_menu − mean Δ*_full| above this in any cell set -> STOP")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = LiveConfig(regime=args.regime, horizon=args.horizon, ask_cost=Fraction(args.ask_cost))
    params = KernelParams(alpha0=Fraction(args.alpha0), beta=args.beta, p_counter=args.p_counter, p_hold=args.p_hold, shortlist=args.shortlist)
    def parse(sz):   # "3x2" = heuristic candidates only; "3x2c" = plus planner completeness (PLAN_ASK, PLAN_Q)
        complete = sz.endswith("c")
        a, b = sz.rstrip("c").split("x")
        return int(a), int(b), complete
    sizes = [parse(x) for x in args.sizes.split(",")]

    def label(no, nq, c):
        return f"{no}x{nq}{'c' if c else ''}"
    t0 = time.time()
    cells = {}
    for role in ("buyer", "seller"):
        for rk in ALL_RANKINGS:
            kernel = T1Kernel(other(role), cfg, params)
            full_ask = Oracle(cfg, role, rk, kernel, True); full_noask = Oracle(cfg, role, rk, kernel, False)
            menus = {label(no, nq, c): Menu(cfg, role, rk, kernel, n_offers=no, n_questions=nq, plan_oracle=full_noask,
                                            plan_ask_oracle=full_ask if c else None) for no, nq, c in sizes}
            oracles = {k: (Oracle(cfg, role, rk, kernel, True, menu=m.actions(True)), Oracle(cfg, role, rk, kernel, False, menu=m.actions(False)))
                       for k, m in menus.items()}
            for opener in ("buyer", "seller"):
                row = {"full_ask": full_ask.root_value(opener), "full_noask": full_noask.root_value(opener)}
                row["delta_full"] = row["full_ask"] - row["full_noask"]
                for k, (oa, on) in oracles.items():
                    va, vn = oa.root_value(opener), on.root_value(opener)
                    row[k] = {"menu_ask": va, "menu_noask": vn, "delta_menu": va - vn,
                              "loss_ask": row["full_ask"] - va, "loss_noask": row["full_noask"] - vn}
                cells[f"{role}|{opener}|{_tag(rk)}"] = row
                print(f"[{time.time()-t0:6.1f}s] {role} opener={opener} {_tag(rk)} Δfull={row['delta_full']:.4f} " +
                      " ".join(f"{k}:lossA={v['loss_ask']:.4f},lossN={v['loss_noask']:.4f}" for k, v in row.items() if isinstance(v, dict)), file=sys.stderr)
    agg = {}
    for k in [label(no, nq, c) for no, nq, c in sizes]:
        rows = list(cells.values())
        agg[k] = {"cells": len(rows),
                  "mean_loss_ask": sum(r[k]["loss_ask"] for r in rows) / len(rows), "max_loss_ask": max(r[k]["loss_ask"] for r in rows),
                  "mean_loss_noask": sum(r[k]["loss_noask"] for r in rows) / len(rows), "max_loss_noask": max(r[k]["loss_noask"] for r in rows),
                  "cells_lossless": sum(1 for r in rows if r[k]["loss_ask"] < 1e-9 and r[k]["loss_noask"] < 1e-9),
                  "mean_delta_menu": sum(r[k]["delta_menu"] for r in rows) / len(rows),
                  "mean_delta_full": sum(r["delta_full"] for r in rows) / len(rows)}
    # stopping rule (reviewer, 2026-10-29): the registered menu's artefact = mean Δ*_menu − mean Δ*_full, per cell set
    stop = {"registered": args.registered, "threshold": args.stop_threshold, "cell_sets": {}}
    for cs, pred in (("all", lambda k: True), ("focal_opens", lambda k: k.split("|")[0] == k.split("|")[1]),
                     ("opponent_opens", lambda k: k.split("|")[0] != k.split("|")[1])):
        rows = [v for k, v in cells.items() if pred(k)]
        art = sum(r[args.registered]["delta_menu"] - r["delta_full"] for r in rows) / len(rows)
        stop["cell_sets"][cs] = {"cells": len(rows), "mean_delta_full": sum(r["delta_full"] for r in rows) / len(rows),
                                 "mean_delta_menu": sum(r[args.registered]["delta_menu"] for r in rows) / len(rows),
                                 "artefact": art, "mean_loss_ask": sum(r[args.registered]["loss_ask"] for r in rows) / len(rows),
                                 "mean_loss_noask": sum(r[args.registered]["loss_noask"] for r in rows) / len(rows),
                                 "exceeds": abs(art) > args.stop_threshold}
    stop["verdict"] = "STOP" if any(v["exceeds"] for v in stop["cell_sets"].values()) else "PROCEED"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"env": ENV_VERSION, "config": cfg.to_json(), "kernel": params.to_json(), "sizes": [label(a, b, c) for a, b, c in sizes],
                   "aggregate": agg, "stopping_rule": stop, "cells": cells, "elapsed_s": round(time.time() - t0, 1)}, f, indent=1)
    print(json.dumps(agg, indent=1))
    print(json.dumps(stop, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
