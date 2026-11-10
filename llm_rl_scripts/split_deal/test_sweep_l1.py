"""Checks for the gate logic of the L1 sweep and regeneration of its results and note.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_sweep_l1.py
"""

import json
import os
from fractions import Fraction

import pytest

from .sweep_l1 import FLOOR, GO, SCARCITY, compute, gate, render, spearman, tier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = "results/split_l1_20261110/sweep.json"
OUT = os.path.join(ROOT, *REL.split("/"))
NOTE = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "SWEEP_SPLIT_L1.md")


def test_tiers_at_the_registered_bars():
    assert (GO, FLOOR, SCARCITY) == (Fraction(1, 10), Fraction(2, 25), Fraction(1, 25))
    assert tier(Fraction(1, 10)) == "go"
    assert tier(Fraction(1, 10) - Fraction(1, 10 ** 9)) == "conditional"
    assert tier(Fraction(2, 25)) == "conditional"
    assert tier(Fraction(2, 25) - Fraction(1, 10 ** 9)) == "no-go"


def _block(w, d_re):
    words = [Fraction(1, 2), None, None, Fraction(1, 2) + w]
    return {"v_words": {"0": words}, "v_reoffer": {"0": {"offers": Fraction(1, 2), "asks_and_offers": Fraction(1, 2) + d_re}}}


def test_gate_fallback_and_scarcity_rules():
    g = gate({"easy": _block(Fraction(3, 25), Fraction(1, 20)), "hard": _block(Fraction(0), Fraction(0))})
    assert (g["verdict"], g["primary_regime"], g["fallback_used"], g["scarcity_conditional"]) == ("go", "easy", False, False)
    # easy no-go, hard passes: hard becomes primary
    g = gate({"easy": _block(Fraction(1, 20), Fraction(1, 20)), "hard": _block(Fraction(9, 100), Fraction(1, 100))})
    assert (g["verdict"], g["primary_regime"], g["fallback_used"], g["scarcity_conditional"]) == (
        "conditional", "hard", True, True)
    # easy conditional: easy stays primary whatever hard shows
    g = gate({"easy": _block(Fraction(9, 100), Fraction(1, 20)), "hard": _block(Fraction(1, 5), Fraction(1, 20))})
    assert (g["verdict"], g["primary_regime"]) == ("conditional", "easy")
    # both no-go: no scarcity flag
    g = gate({"easy": _block(Fraction(0), Fraction(0)), "hard": _block(Fraction(0), Fraction(0))})
    assert (g["verdict"], g["scarcity_conditional"]) == ("no-go", False)


def test_spearman():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254, abs=1e-6)


@pytest.mark.skipif(not os.path.exists(OUT), reason="sweep results missing")
def test_regeneration():
    with open(OUT) as f:
        stored = json.load(f)
    fresh = compute(workers=max(1, (os.cpu_count() or 2) - 2))
    fresh["commit"], fresh["seconds"] = stored["commit"], stored["seconds"]
    assert json.loads(json.dumps(fresh)) == stored
    if os.path.exists(NOTE):
        with open(NOTE) as f:
            assert f.read() == render(stored, REL)
