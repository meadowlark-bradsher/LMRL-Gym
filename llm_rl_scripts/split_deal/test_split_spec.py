"""Checks for the Split Deal game definitions and the numbers its design note cites.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_split_spec.py
"""

import json
import os
from fractions import Fraction

import pytest

from .game import ITEMS, TOTAL, Instance, Question, answer, bank, canonical_instances, outside_option, y_units
from .spec_numbers import BEGIN, END, compute, instance_numbers, render_block

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = "results/split_spec_20261110/spec_numbers.json"
OUT = os.path.join(ROOT, *REL.split("/"))
NOTE = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "DESIGN_SPLIT.md")


def test_instances_and_types():
    insts = canonical_instances()
    assert [i.key for i in insts] == ["1-1-3", "1-2-2", "1-1-4", "1-2-3", "2-2-2", "1-1-5", "1-2-4", "1-3-3",
                                      "2-2-3"]
    for inst in insts:
        types = inst.types()
        assert len(set(types)) == len(types) > 0
        assert all(inst.utility(v, inst.counts) == TOTAL and min(v) >= 0 for v in types)
    # hand counts: v1 + v2 + v3 = 5 has C(7,2) solutions; v1 + v2 + 3 v3 = 10 has 11 + 8 + 5 + 2
    assert len(Instance((2, 2, 2)).types()) == 21
    assert len(Instance((1, 1, 3)).types()) == 26


def test_allocations_split_the_pool():
    inst = Instance((1, 2, 3))
    allocs = inst.allocations()
    assert len(allocs) == 2 * 3 * 4
    for v in inst.types():
        for a in allocs:
            assert inst.utility(v, a) + inst.utility(v, inst.rest(a)) == TOTAL


def test_bank_answers():
    qs = bank()
    assert len(qs) == 12 and len({q.id for q in qs}) == 12
    assert Question("PREF", (0, 2)).id == f"PREF({ITEMS[0]},{ITEMS[2]})"
    for inst in canonical_instances():
        for v in inst.types():
            assert any(answer(inst, v, Question("TOP", (k,))) for k in range(3))
            assert any(answer(inst, v, Question("HALF", (k,))) for k in range(3)) or max(
                c * x for c, x in zip(inst.counts, v)) < 5
            assert not all(answer(inst, v, Question("ZERO", (k,))) for k in range(3))
            for j, k in ((0, 1), (0, 2), (1, 2)):
                if answer(inst, v, Question("PREF", (j, k))):
                    assert v[j] > v[k] and not answer(inst, v, Question("TOP", (k,)))


def test_units():
    assert outside_option("easy") == 4 and outside_option("hard") == Fraction(11, 2)
    assert y_units("easy", TOTAL) == 1 and y_units("easy", 4) == 0 and y_units("hard", 10) == 1


def test_structural_bounds():
    for inst in canonical_instances():
        r = instance_numbers(inst)
        n = r["n_pairs"]
        assert TOTAL * n <= r["max_joint_sum"] <= 2 * TOTAL * n       # giving one side everything is worth TOTAL
        assert r["pairs_with_deal"]["hard"] <= r["pairs_with_deal"]["easy"] <= n
        assert r["bank_classes"] <= r["n_types"] and r["largest_class"] >= 1
        # a pair needs joint value >= 12 for both sides to clear the hard outside option of 5.5
        assert r["pairs_with_deal"]["hard"] <= r["pairs_integrative"]


@pytest.mark.skipif(not os.path.exists(OUT), reason="spec numbers missing")
def test_regeneration():
    with open(OUT) as f:
        stored = json.load(f)
    fresh = compute()
    fresh["commit"] = stored["commit"]
    assert json.loads(json.dumps(fresh)) == stored
    if os.path.exists(NOTE):
        with open(NOTE) as f:
            text = f.read()
        assert text[text.index(BEGIN):text.index(END) + len(END)] == render_block(stored, REL)
