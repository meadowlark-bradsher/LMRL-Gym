"""Phase 0 step 7: where is ASK optimal? Sweep the T1 kernel and (H, c_q).

For every cell (kernel params, H, c_q, focal role, opener, focal type) solve the
exact oracle twice (ASK allowed / forbidden) and record
  V_ask, V_noask, Delta*_active = V_ask - V_noask, the root action, and the share
  of focal decision nodes (with ASK legal) where ASK is strictly optimal.
Aggregates per (kernel, H, c_q) over the 2 roles x 2 openers x 6 types.
Everything is exact-on-model (no sampling, no API).

    PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.sweep_ask_value \
        --out results/live_phase0_sweep_20261027/sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .opponent import KernelParams, T1Kernel
from .oracle import Oracle
from .protocol_live import ENV_VERSION, LiveConfig, new_game, other
from . import belief_live as bl


def kernel_id(p: KernelParams) -> str:
    return f"a{float(p.alpha0):.2f}_b{p.beta:g}_pc{p.p_counter:g}_ph{p.p_hold:g}_k{p.shortlist}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha0", default="3/5,4/5")
    ap.add_argument("--beta", default="4,8")
    ap.add_argument("--p-counter", default="0.5,1.0")
    ap.add_argument("--p-hold", default="1.0")
    ap.add_argument("--shortlist", type=int, default=3)
    ap.add_argument("--horizons", default="2,3,4")
    ap.add_argument("--ask-costs", default="0,1/100,3/100")
    ap.add_argument("--regime", default="easy")
    ap.add_argument("--types", default="all", help="'all' or comma indices into ALL_RANKINGS")
    args = ap.parse_args(argv)

    alphas = [Fraction(x) for x in args.alpha0.split(",")]
    betas = [float(x) for x in args.beta.split(",")]
    pcs = [float(x) for x in args.p_counter.split(",")]
    phs = [float(x) for x in args.p_hold.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    costs = [Fraction(x) for x in args.ask_costs.split(",")]
    types = list(range(len(ALL_RANKINGS))) if args.types == "all" else [int(x) for x in args.types.split(",")]

    t0 = time.time()
    cells = []
    agg = {}
    for a0, b, pc, ph in itertools.product(alphas, betas, pcs, phs):
        params = KernelParams(alpha0=a0, beta=b, p_counter=pc, p_hold=ph, shortlist=args.shortlist)
        kid = kernel_id(params)
        for H in horizons:
            # V_noask does not depend on c_q: solve once per (kernel, H, role, opener, type)
            noask = {}
            for focal in ("buyer", "seller"):
                for ti in types:
                    cfg0 = LiveConfig(regime=args.regime, horizon=H, ask_cost=Fraction(0))
                    k = T1Kernel(other(focal), cfg0, params)
                    on = Oracle(cfg0, focal, ALL_RANKINGS[ti], k, ask_allowed=False)
                    for opener in ("buyer", "seller"):
                        noask[(focal, ti, opener)] = on.root_value(opener)
            for cq in costs:
                cfg = LiveConfig(regime=args.regime, horizon=H, ask_cost=cq)
                deltas, root_asks, opt_nodes, legal_nodes = [], 0, 0, 0
                for focal in ("buyer", "seller"):
                    for ti in types:
                        k = T1Kernel(other(focal), cfg, params)
                        oa = Oracle(cfg, focal, ALL_RANKINGS[ti], k, ask_allowed=True)
                        for opener in ("buyer", "seller"):
                            v = oa.root_value(opener)
                            vn = noask[(focal, ti, opener)]
                            root = new_game(cfg, opener)
                            root_action = oa.best(root, bl.uniform())[0].key if root.turn == focal else None
                            d = v - vn
                            assert d >= -1e-12, (kid, H, cq, focal, ti, opener, d)
                            cells.append({
                                "kernel": kid, "H": H, "ask_cost": str(cq), "focal": focal, "opener": opener,
                                "focal_type": "/".join(a[0] for a in ALL_RANKINGS[ti]),
                                "V_ask": v, "V_noask": vn, "delta_active": d, "root_action": root_action,
                            })
                            deltas.append(d)
                            root_asks += 1 if (root_action or "").startswith("ASK") else 0
                        st = oa.stats()
                        opt_nodes += st["ask_optimal_nodes"]
                        legal_nodes += st["ask_legal_nodes"]
                agg[f"{kid}|H={H}|cq={cq}"] = {
                    "kernel": kid, "params": params.to_json(), "H": H, "ask_cost": str(cq),
                    "cells": len(deltas),
                    "mean_delta_active": sum(deltas) / len(deltas),
                    "max_delta_active": max(deltas),
                    "share_cells_delta_positive": sum(1 for d in deltas if d > 1e-9) / len(deltas),
                    "root_ask_cells": root_asks,
                    "ask_optimal_nodes": opt_nodes, "ask_legal_nodes": legal_nodes,
                    "ask_optimal_share": (opt_nodes / legal_nodes) if legal_nodes else None,
                    "mean_V_ask": sum(c["V_ask"] for c in cells if c["kernel"] == kid and c["H"] == H and c["ask_cost"] == str(cq)) / len(deltas),
                }
                a = agg[f"{kid}|H={H}|cq={cq}"]
                print(f"[{time.time()-t0:6.0f}s] {kid} H={H} cq={cq}: meanΔ={a['mean_delta_active']:.4f} maxΔ={a['max_delta_active']:.4f} cells>0={a['share_cells_delta_positive']:.2f} askopt={opt_nodes}/{legal_nodes} rootASK={root_asks}/{len(deltas)}", file=sys.stderr)
    out = {
        "env": ENV_VERSION, "regime": args.regime, "shortlist": args.shortlist,
        "grid": {"alpha0": [str(a) for a in alphas], "beta": betas, "p_counter": pcs, "p_hold": phs,
                 "horizons": horizons, "ask_costs": [str(c) for c in costs], "types": types},
        "elapsed_s": round(time.time() - t0, 1),
        "aggregate": agg, "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}: {len(cells)} cells, {len(agg)} aggregates, {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
