"""How the focal's exact value grows with the budget H under the registered kernel (buyer focal):
V*_ask, V*_noask and the perfect-information value (point-mass belief on each opponent type,
averaged) per opener, for the horizons given. Exploratory; feeds the horizon-mechanism paragraph
of FINDINGS.md section 3.7.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.horizon_growth --horizons 1,2,3,4 --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from .opponent import KernelParams, T1Kernel
from .oracle import Oracle
from .protocol_live import LiveConfig, new_game


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizons", default="1,2,3,4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    params = KernelParams()
    out = {}
    for H in [int(x) for x in args.horizons.split(",")]:
        t0 = time.time()
        cfg = LiveConfig(horizon=H)
        row = {}
        for opener in ("buyer", "seller"):
            va = vn = vp = 0.0
            for rk in ALL_RANKINGS:
                kernel = T1Kernel("seller", cfg, params)
                oa = Oracle(cfg, "buyer", rk, kernel, True)
                on = Oracle(cfg, "buyer", rk, kernel, False)
                va += oa.root_value(opener) / 6
                vn += on.root_value(opener) / 6
                for i in range(6):   # perfect information: the opponent's type known
                    b = tuple(1.0 if j == i else 0.0 for j in range(6))
                    vp += on.value(new_game(cfg, opener), b) / 36
            row[opener] = {"V_ask": va, "V_noask": vn, "V_perfect_info": vp, "delta": va - vn, "info_gap": vp - va}
        row["elapsed_s"] = round(time.time() - t0, 1)
        out[str(H)] = row
        print(H, json.dumps(row), file=sys.stderr, flush=True)
    json.dump({"kernel": params.to_json(), "focal": "buyer", "regime": "easy", "by_H": out}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
