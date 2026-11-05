"""Gates for the elicited-negotiation game + VOI oracle.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/elicit_negotiation

1. belief posteriors stay normalized;
2. under the truthful kernel, asking never hurts GROSS value (V* >= V*_noask), and
   once every question is asked the only move left is to offer (value == offer_value);
3. with the type known, the oracle's offer is the exact full-information optimum;
4. a high enough token price drives VOI to 0 and makes the first move OFFER;
5. asking strictly helps at token_price 0 (VOI > 0); and the EIG signal diverges
   from decision value -- some questions are high-EIG yet decision-irrelevant.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import game as g
from . import oracle as o
from .calibrate import calibrate


def test_posteriors_normalized():
    b = g.uniform()
    assert abs(b.sum() - 1.0) < 1e-12
    pyes = g.TRUTHFUL.p_yes()
    for q in range(g.N_QUESTIONS):
        for yes in (True, False):
            p, post = g.posterior(b, g.answer_likelihood(pyes, q, yes))
            if p > 0:
                assert abs(post.sum() - 1.0) < 1e-9
                assert (post >= -1e-12).all()


def test_more_info_never_hurts_and_terminal_offers():
    s = o.Solver(token_price=0.0)
    b = g.uniform()
    assert s.value(b) >= s.offer_value(b) - 1e-12           # asking is optional
    all_asked = frozenset(range(g.N_QUESTIONS))
    assert s.value(b, all_asked) == pytest.approx(s.offer_value(b))  # nothing left but to offer


def test_full_information_offer_is_optimal():
    s = o.Solver(token_price=0.0)
    for t in (0, 7, 23, 44):
        b = np.zeros(g.N_TYPES); b[t] = 1.0
        want = float((g.U_FOCAL * g.ACCEPT[t]).max())
        assert s.offer_value(b) == pytest.approx(want)
        # a known type needs no questions
        assert s.best_action(b) == "OFFER"


def test_high_token_price_kills_voi():
    s = o.summarize(token_price=1.0)
    assert s["VOI"] == pytest.approx(0.0, abs=1e-9)
    assert s["best_first_action"] == "OFFER"


def test_voi_positive_and_eig_value_divergence():
    r = calibrate()
    assert r["gross_voi"] > 1e-3, r["gross_voi"]
    assert r["token_price_voi_zero"] is not None            # some price kills VOI
    # EIG is ~flat across questions while decision value is not -> EIG a poor guide
    assert r["eig_spread_bits"] < 0.1
    assert r["myopic_value_spread"] > 1e-3
    assert len(r["high_eig_zero_value_questions"]) > 0, "expected high-EIG decision-irrelevant questions"
