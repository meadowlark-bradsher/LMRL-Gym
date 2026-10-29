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
    p.add_argument("--sizes", default="2x2,3x2,2x3,3x3", help="n_offers x n_questions variants")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = LiveConfig(regime=args.regime, horizon=args.horizon, ask_cost=Fraction(args.ask_cost))
    params = KernelParams(alpha0=Fraction(args.alpha0), beta=args.beta, p_counter=args.p_counter, p_hold=args.p_hold, shortlist=args.shortlist)
    sizes = [tuple(int(x) for x in s.split("x")) for s in args.sizes.split(",")]
    t0 = time.time()
    cells = {}
    for role in ("buyer", "seller"):
        for rk in ALL_RANKINGS:
            kernel = T1Kernel(other(role), cfg, params)
            full_ask = Oracle(cfg, role, rk, kernel, True); full_noask = Oracle(cfg, role, rk, kernel, False)
            menus = {f"{no}x{nq}": Menu(cfg, role, rk, kernel, n_offers=no, n_questions=nq) for no, nq in sizes}
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
    for k in [f"{no}x{nq}" for no, nq in sizes]:
        rows = list(cells.values())
        agg[k] = {"cells": len(rows),
                  "mean_loss_ask": sum(r[k]["loss_ask"] for r in rows) / len(rows), "max_loss_ask": max(r[k]["loss_ask"] for r in rows),
                  "mean_loss_noask": sum(r[k]["loss_noask"] for r in rows) / len(rows), "max_loss_noask": max(r[k]["loss_noask"] for r in rows),
                  "cells_lossless": sum(1 for r in rows if r[k]["loss_ask"] < 1e-9 and r[k]["loss_noask"] < 1e-9),
                  "mean_delta_menu": sum(r[k]["delta_menu"] for r in rows) / len(rows),
                  "mean_delta_full": sum(r["delta_full"] for r in rows) / len(rows)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"env": ENV_VERSION, "config": cfg.to_json(), "kernel": params.to_json(), "sizes": [f"{a}x{b}" for a, b in sizes],
                   "aggregate": agg, "cells": cells, "elapsed_s": round(time.time() - t0, 1)}, f, indent=1)
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
