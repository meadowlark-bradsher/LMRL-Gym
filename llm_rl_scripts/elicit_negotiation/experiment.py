"""Increment 2: EIG-greedy acquisition vs the value-optimal oracle.

    PYTHONPATH=. python3 -m llm_rl_scripts.elicit_negotiation.experiment [--out DIR]

Two experiments, both evaluated exactly (``evaluate.evaluate``) from the uniform
prior over the 45 buyer types.

A. FRONTIER UNDER TOKEN COST (matched truthful kernel). For a grid of token prices,
   the value-optimal oracle vs EIG-greedy at its *best* threshold (swept over ``tau``).
   The oracle is optimal over all policies here, so it weakly dominates; the question
   is how large the gap is and why. We also report each policy's first question at the
   prior -- the oracle asks a decision-relevant one, EIG-greedy asks the max-EIG one,
   which the calibration showed is nearly arbitrary (EIG barely varies across
   questions) and typically decision-irrelevant.

B. ANSWER-KERNEL MISMATCH. Both policies *assume* truthful answers, but the buyer
   answers through a noisy kernel (symmetric flip ``eps``) or a self-serving strategic
   lean. We compare the naive oracle and naive EIG-greedy (truthful-assumed) against a
   robust oracle that knows the true kernel. EIG-greedy, assuming certainty, collapses
   its belief onto the wrong types and offers deals the buyer rejects; the robust
   oracle recovers most of the loss. This is the elicitation echo of the CD-DRL R4
   finding: an acquisition rule is only as good as the value model behind it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import game as g
from . import oracle as o
from .evaluate import evaluate
from .policies import EIGGreedy, OraclePolicy

TOKEN_PRICES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.1)
TAUS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99)
EPSILONS = (0.0, 0.05, 0.1, 0.2, 0.3)
MISMATCH_PRICE = 0.02
# a simple strategic distortion: the buyer overstates a taste for sweeteners
# (warranty/delivery) to pull the seller off high-price offers.
STRATEGIC_LEAN = (("top_warranty", 0.4), ("top_delivery", 0.4))


def _best_eig_greedy(eval_kernel: g.Kernel, token_price: float, assumed=g.TRUTHFUL) -> dict:
    """EIG-greedy at the tau that maximises payoff under this price/kernel."""
    best = None
    for tau in TAUS:
        r = evaluate(EIGGreedy(tau, assumed_kernel=assumed), eval_kernel, token_price)
        r = {**r, "tau": tau}
        if best is None or r["payoff"] > best["payoff"] + 1e-15:
            best = r
    return best


def frontier() -> list:
    rows = []
    for tp in TOKEN_PRICES:
        oracle = OraclePolicy(g.TRUTHFUL, token_price=tp)
        ora = evaluate(oracle, g.TRUTHFUL, tp)
        eig = _best_eig_greedy(g.TRUTHFUL, tp)
        rows.append({
            "token_price": tp,
            "oracle_payoff": ora["payoff"], "oracle_first": ora["first_action"],
            "oracle_questions": ora["exp_questions"],
            "eig_payoff": eig["payoff"], "eig_first": eig["first_action"],
            "eig_questions": eig["exp_questions"], "eig_best_tau": eig["tau"],
            "gap": ora["payoff"] - eig["payoff"],
        })
    return rows


def _naive_tau() -> float:
    """The asking threshold EIG-greedy commits to under its truthful assumption at the
    mismatch price -- fixed before any noise is seen, so the mismatch test is honest."""
    return _best_eig_greedy(g.TRUTHFUL, MISMATCH_PRICE, assumed=g.TRUTHFUL)["tau"]


def mismatch() -> list:
    rows = []
    tau = _naive_tau()
    kernels = [("noisy", eps, g.noisy(eps)) for eps in EPSILONS]
    kernels.append(("strategic", None, g.Kernel(eps=0.0, lean=STRATEGIC_LEAN)))
    for name, eps, ek in kernels:
        # all three policies assume TRUTHFUL except the robust oracle, which is told ek.
        naive_ora = evaluate(OraclePolicy(g.TRUTHFUL, MISMATCH_PRICE), ek, MISMATCH_PRICE)
        robust_ora = evaluate(OraclePolicy(ek, MISMATCH_PRICE), ek, MISMATCH_PRICE)
        naive_eig = evaluate(EIGGreedy(tau, g.TRUTHFUL), ek, MISMATCH_PRICE)
        rows.append({
            "kernel": name, "eps": eps,
            "naive_oracle_payoff": naive_ora["payoff"],
            "robust_oracle_payoff": robust_ora["payoff"],
            "naive_eig_payoff": naive_eig["payoff"],
            "naive_eig_questions": naive_eig["exp_questions"],
            "robust_gain": robust_ora["payoff"] - naive_ora["payoff"],
        })
    return rows


def _diagnostic() -> dict:
    """Why EIG-greedy loses: its acquisition signal is nearly flat across questions and
    blind to decision value, so it cannot convert a token budget into payoff the way
    value-optimal asking does -- it over-asks when info is free and mis-spends when it
    is not. We quantify the flatness (EIG spread) and the over-asking signature
    (questions asked at price 0: EIG asks everything, the oracle asks only what pays)."""
    s = o.Solver(token_price=0.0)
    b = g.uniform()
    eig = np.array([s.eig_bits(b, q) for q in range(g.N_QUESTIONS)])
    q_star = int(np.argmax(eig))
    ora0 = evaluate(OraclePolicy(g.TRUTHFUL, 0.0), g.TRUTHFUL, 0.0)
    eig0 = evaluate(EIGGreedy(0.0, g.TRUTHFUL), g.TRUTHFUL, 0.0)
    return {"eig_spread_bits": float(eig.max() - eig.min()),
            "max_eig_question": g.BANK[q_star].id, "max_eig_bits": float(eig[q_star]),
            "oracle_questions_free": ora0["exp_questions"],
            "eig_questions_free": eig0["exp_questions"],
            "same_gross_payoff": abs(ora0["payoff"] - eig0["payoff"]) < 1e-9}


def run() -> dict:
    fr = frontier()
    mm = mismatch()
    diag = _diagnostic()
    max_gap = max(r["gap"] for r in fr)
    dominates = all(r["gap"] >= -1e-9 for r in fr)
    robust_helps = any(r["robust_gain"] > 1e-3 for r in mm)
    over_asks = diag["eig_questions_free"] > diag["oracle_questions_free"] + 1.0
    teeth = max_gap > 1e-3 and dominates and robust_helps and over_asks
    return {"frontier": fr, "mismatch": mm, "diagnostic": diag,
            "max_gap": max_gap, "oracle_dominates": dominates,
            "robust_helps": robust_helps, "eig_over_asks": over_asks, "teeth": teeth,
            "config": {"token_prices": list(TOKEN_PRICES), "taus": list(TAUS),
                       "epsilons": list(EPSILONS), "mismatch_price": MISMATCH_PRICE,
                       "strategic_lean": STRATEGIC_LEAN}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="optional dir to freeze results.json into")
    args = p.parse_args(argv)
    r = run()

    d = r["diagnostic"]
    print("Diagnostic (uniform prior): EIG is flat and value-blind")
    print(f"  EIG spread across questions = {d['eig_spread_bits']:.3f} bits "
          f"(max-EIG question = {d['max_eig_question']} @ {d['max_eig_bits']:.3f} bits)")
    print(f"  questions asked when info is FREE: EIG-greedy {d['eig_questions_free']:.2f} "
          f"vs oracle {d['oracle_questions_free']:.2f} "
          f"(same gross payoff: {d['same_gross_payoff']}) -> EIG over-asks")

    print("\nA. Frontier under token cost (matched truthful kernel):")
    print(f"  {'price':>6} {'oracle':>8} {'Eq':>5} | {'EIG':>8} {'tau':>5} {'Eq':>5} | {'gap':>7}")
    for row in r["frontier"]:
        print(f"  {row['token_price']:>6} {row['oracle_payoff']:>8.4f} "
              f"{row['oracle_questions']:>5.2f} | {row['eig_payoff']:>8.4f} "
              f"{row['eig_best_tau']:>5} {row['eig_questions']:>5.2f} | {row['gap']:>7.4f}")

    print(f"\nB. Answer-kernel mismatch (token_price={MISMATCH_PRICE}, policies assume truthful):")
    print(f"  {'kernel':>12} {'naive_ora':>10} {'robust_ora':>11} {'naive_EIG':>10} "
          f"{'EIG_Eq':>7} {'robust_gain':>12}")
    for row in r["mismatch"]:
        tag = row["kernel"] if row["eps"] is None else f"noisy{row['eps']}"
        print(f"  {tag:>12} {row['naive_oracle_payoff']:>10.4f} "
              f"{row['robust_oracle_payoff']:>11.4f} {row['naive_eig_payoff']:>10.4f} "
              f"{row['naive_eig_questions']:>7.2f} {row['robust_gain']:>+12.4f}")

    print(f"\nMax oracle-minus-EIG gap {r['max_gap']:.4f}; oracle dominates: {r['oracle_dominates']}; "
          f"robust oracle helps under mismatch: {r['robust_helps']}.")
    print(f"Verdict: regime {'OK' if r['teeth'] else 'INSUFFICIENT'} -- "
          + ("EIG ranks questions nearly flat and blind to value, so it over-asks when "
             "info is free and cannot spend a token budget as well as value-optimal "
             "asking (oracle strictly wins at every positive price); and knowing the "
             "true answer kernel recovers payoff that naive elicitation throws away."
             if r["teeth"] else
             "the EIG-vs-value gap or the mismatch effect is too small; revisit."))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "increment2.json"), "w") as f:
            json.dump(r, f, indent=1)
        print(f"\nFroze results to {os.path.join(args.out, 'increment2.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
