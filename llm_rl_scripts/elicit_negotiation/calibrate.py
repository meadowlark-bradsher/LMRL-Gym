"""Calibration: does the toy have the intended regime -- meaningful inter-communicative
VOI, a token cost at which asking stops paying, and an EIG acquisition signal that
diverges from decision value (so the later EIG-vs-value experiment has teeth)?

    PYTHONPATH=. python3 -m llm_rl_scripts.elicit_negotiation.calibrate

Reports (truthful answer kernel, uniform prior over the 45 buyer types):
* V*, V*_noask and VOI across a few token prices (gross -> net).
* per-question EIG (bits) vs myopic decision value (ask one, then offer): if EIG is
  ~flat while myopic value is not, EIG is a poor guide to what is worth asking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import game as g
from . import oracle as o


def _myopic_gain(s: o.Solver, b: np.ndarray, q: int) -> float:
    """Value of asking q then offering immediately (no further questions), minus
    offering now -- the decision value of a single question."""
    v = -s.token_price * g.BANK[q].cost
    for yes in (True, False):
        p, post = g.posterior(b, g.answer_likelihood(s._pyes, q, yes))
        if p > 0:
            v += p * s.offer_value(post)
    return v - s.offer_value(b)


def calibrate(token_prices=(0.0, 0.01, 0.03, 0.1)) -> dict:
    voi = [o.summarize(token_price=tp) for tp in token_prices]

    s = o.Solver(token_price=0.0)
    b = g.uniform()
    per_q = [{"id": g.BANK[q].id, "cost": g.BANK[q].cost,
              "eig_bits": s.eig_bits(b, q), "myopic_gain": _myopic_gain(s, b, q)}
             for q in range(g.N_QUESTIONS)]
    eig = np.array([r["eig_bits"] for r in per_q])
    val = np.array([r["myopic_gain"] for r in per_q])
    eig_spread = float(eig.max() - eig.min())
    val_spread = float(val.max() - val.min())
    high_eig_zero_value = [r["id"] for r in per_q
                           if r["eig_bits"] >= 0.5 and r["myopic_gain"] <= 1e-9]
    zero_price_voi = next(v["VOI"] for v in voi if v["token_price"] == 0.0)
    net_zero_price = next((v["token_price"] for v in voi if v["VOI"] <= 1e-9), None)
    teeth = zero_price_voi > 1e-3 and len(high_eig_zero_value) > 0

    return {"token_prices": list(token_prices), "voi": voi, "per_question": per_q,
            "eig_spread_bits": eig_spread, "myopic_value_spread": val_spread,
            "high_eig_zero_value_questions": high_eig_zero_value,
            "gross_voi": zero_price_voi, "token_price_voi_zero": net_zero_price,
            "teeth": teeth}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="optional dir to freeze results.json into")
    args = p.parse_args(argv)
    r = calibrate()

    print("VOI across token prices (truthful kernel, uniform prior):")
    for v in r["voi"]:
        print(f"  token_price={v['token_price']:<5} V*={v['V_star']:.4f}  noask={v['V_noask']:.4f}  "
              f"VOI={v['VOI']:.4f}  first={v['best_first_action']}")
    print("\nPer-question EIG vs myopic decision value (prior):")
    print(f"  {'question':20s} {'cost':>4} {'EIG(bits)':>10} {'myopic_gain':>12}")
    for q in r["per_question"]:
        print(f"  {q['id']:20s} {q['cost']:>4} {q['eig_bits']:>10.3f} {q['myopic_gain']:>+12.4f}")
    print(f"\nEIG spread {r['eig_spread_bits']:.3f} bits vs myopic-value spread {r['myopic_value_spread']:.4f}.")
    print(f"High-EIG, zero decision-value questions: {r['high_eig_zero_value_questions']}")
    print(f"Gross VOI {r['gross_voi']:.4f}; VOI -> 0 at token_price {r['token_price_voi_zero']}.")
    print(f"\nVerdict: regime {'OK' if r['teeth'] else 'INSUFFICIENT'} -- "
          + ("meaningful VOI, and EIG is ~flat while decision value is not (EIG is a poor acquisition "
             "guide here, so the EIG-vs-value experiment has teeth)." if r["teeth"]
             else "VOI or EIG/value divergence too small; revisit the design."))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "calibration.json"), "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
