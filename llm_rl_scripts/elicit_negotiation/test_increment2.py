"""Gates for increment 2: EIG-greedy vs the value-optimal oracle, and mismatch.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/elicit_negotiation/test_increment2.py

1. the exact evaluator agrees with the oracle's own DP value under a matched kernel
   (evaluate(oracle) == summarize V*);
2. under a matched kernel the oracle weakly dominates EIG-greedy at every token price
   and threshold (it is optimal over all policies there);
3. the gap is real: at some token price the oracle strictly beats EIG-greedy's best;
4. EIG-greedy's acquisition signal is nearly flat and value-blind, so when information
   is free it over-asks -- asks every question where the oracle asks only what pays,
   for the identical gross payoff;
5. answer-kernel mismatch bites and value knowledge recovers it: for some noise level
   the robust oracle (which knows the kernel) strictly beats the naive oracle, and
   naive EIG-greedy both pays for questions and trails the robust oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import game as g
from . import oracle as o
from .evaluate import evaluate
from .experiment import MISMATCH_PRICE, TOKEN_PRICES, TAUS, run
from .policies import EIGGreedy, OraclePolicy

R = run()


def test_evaluator_matches_oracle_dp():
    for tp in (0.0, 0.02, 0.1):
        got = evaluate(OraclePolicy(g.TRUTHFUL, tp), g.TRUTHFUL, tp)["payoff"]
        want = o.summarize(token_price=tp)["V_star"]
        assert got == pytest.approx(want, abs=1e-9), (tp, got, want)


def test_oracle_dominates_eig_matched_kernel():
    for tp in TOKEN_PRICES:
        ora = evaluate(OraclePolicy(g.TRUTHFUL, tp), g.TRUTHFUL, tp)["payoff"]
        for tau in TAUS:
            eig = evaluate(EIGGreedy(tau, g.TRUTHFUL), g.TRUTHFUL, tp)["payoff"]
            assert ora >= eig - 1e-9, (tp, tau, ora, eig)


def test_gap_is_real():
    assert R["oracle_dominates"]
    assert R["max_gap"] > 1e-3, R["max_gap"]


def test_eig_is_flat_and_over_asks_when_free():
    d = R["diagnostic"]
    assert d["eig_spread_bits"] < 0.1, d                    # EIG barely ranks questions
    assert d["same_gross_payoff"], d                        # same payoff at price 0
    assert d["eig_questions_free"] > d["oracle_questions_free"] + 1.0, d
    assert R["eig_over_asks"]


def test_mismatch_bites_and_value_knowledge_recovers():
    assert R["robust_helps"], R["mismatch"]
    # at the strongest noise, the robust oracle beats naive EIG-greedy, which still
    # spends tokens asking.
    worst = max((r for r in R["mismatch"] if r["eps"] is not None), key=lambda r: r["eps"])
    assert worst["robust_oracle_payoff"] > worst["naive_eig_payoff"] + 1e-9, worst
    assert worst["naive_eig_questions"] > 0.0, worst
