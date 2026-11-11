"""Checks for the v0.3 sweeps (gates SD3-W and SD3-D): internal consistency on a small instance, the
gate logic on synthetic inputs, and regeneration of the stored results and notes.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_sweep_v03.py
"""

import json
import os
from fractions import Fraction

import pytest

from . import sweep_l1_v03 as l1
from . import sweep_l2_v03 as l2
from .belief import Cell
from .game import Instance
from .private_d import PrivateTables
from .twosided import STRUCTURES, Program, Shared

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMALL = (2, 2, 3)
WORKERS = max(1, (os.cpu_count() or 2) - 2)


# -- SD3-W --------------------------------------------------------------------------------

def test_l1_play_matches_rollout():
    tab = PrivateTables(Instance(SMALL))
    for ia in range(0, len(tab.types), 5):
        cell = Cell(tab, tab.types[ia], tab.full)
        for b in range(len(tab.types)):
            for q in (0, 1, 3):
                asked, agreed, walked = l1.play(cell, tab, b, q)
                assert (len(asked), agreed, walked) == cell.rollout(b, q, "exact")


@pytest.mark.parametrize("bank", l1.BANKS)
def test_l1_level_pairs_average_to_the_words_value(bank):
    g = l1.solve_group((list(SMALL), bank, "independent"))
    pairs = g["by_level_pair"]
    assert sorted(pairs) == sorted(l1.PAIRS) and len({p["pairs"] for p in pairs.values()}) == 1
    k = len(pairs)
    assert sum(p["V_none"] for p in pairs.values()) / k == g["v_words"]["0"][0]
    assert sum(p["W"] for p in pairs.values()) / k == l1.d_words(g)
    assert sum(p["W_common"] for p in pairs.values()) / k == g["v_words_common"][3] - g["v_words_common"][0]


def _block(w, d_re):
    words = [Fraction(1, 2), None, None, Fraction(1, 2) + w]
    return {"v_words": {"0": words}, "v_words_common": words,
            "v_reoffer": {"0": {"offers": Fraction(1, 2), "asks_and_offers": Fraction(1, 2) + d_re}}}


def test_l1_gate_has_no_fallback():
    ctrl = _block(Fraction(1, 5), Fraction(1, 5))
    g = l1.gate(_block(Fraction(3, 25), Fraction(1, 20)), ctrl)
    assert (g["verdict"], g["scarcity_conditional"]) == ("go", False)
    g = l1.gate(_block(Fraction(9, 100), Fraction(1, 100)), ctrl)
    assert (g["verdict"], g["scarcity_conditional"]) == ("conditional", True)
    g = l1.gate(_block(Fraction(1, 20), Fraction(0)), ctrl)       # the control never rescues a no-go
    assert (g["verdict"], g["scarcity_conditional"]) == ("no-go", False)


# -- SD3-D --------------------------------------------------------------------------------

def test_l2_decision_rounds():
    tab = PrivateTables(Instance(SMALL))
    sh = Shared(tab)
    for ia in range(0, len(tab.types), 3):
        prog = Program(sh, ia, STRUCTURES["T3"], 3, Fraction(1, 2))
        for b in range(len(tab.types)):
            out = prog.rollout(b)
            r = out["decision_rounds"]
            assert len(r) == len(out["decisions"]) and r == sorted(set(r)) and all(1 <= x <= 3 for x in r)


@pytest.fixture(scope="module")
def small_l2():
    groups = [l2.solve_group((list(SMALL), bank, "1/2")) for bank in l2.BANKS]
    pooled = {bank: {"1/2": l2._pool([g for g in groups if g["bank"] == bank])} for bank in l2.BANKS}
    return groups, pooled


def test_l2_levels_average_to_all(small_l2):
    groups, _ = small_l2
    for g in groups:
        for arm, d in g["arms"].items():
            for f in l2.FIELDS + ("root",):
                assert (d["lo"][f] + d["hi"][f]) / 2 == d["all"][f], (arm, f)
        for cell, d in g["game_exact_b"].items():
            for f in ("YA", "YB"):
                assert (d["lo"][f] + d["hi"][f]) / 2 == d["all"][f]


def test_l2_root_matches_rollouts_and_rent_identity(small_l2):
    groups, pooled = small_l2
    for g in groups:
        for arm, d in g["arms"].items():
            for s in l2.SCOPES:
                assert d[s]["root"] == d[s]["YA"], (arm, s)     # A1's value is what its rollouts realize
    gt = l2.gate(pooled, groups)
    for key in ("D2", "D2_lo", "D2_hi"):
        assert gt["readings"][key]["identity_exact"], key


def _row(lv, high, ans, strict=True, rd=1):
    return [0, lv, 12 if high else 0, high, ans, strict, rd]


def test_l2_d4_d5_logic():
    def groups(rows):
        return [{"instance": "x", "high_round_t2": {"2": 4}, "t3_decisions": rows}]

    # hi answers 2/2, lo answers 1/4 (declines 3/4); value questions answered 3/3 > HIGH 3/6
    rows = [_row("hi", True, True)] * 2 + [_row("lo", True, True)] + [_row("lo", True, False)] * 3 + \
        [_row("lo", False, True)] * 3
    d = l2._d45(groups(rows))
    assert d["D4"]["holds"] and d["D5"]["holds"]
    assert d["D4"]["lo_decline_share"] == Fraction(3, 4) and list(d["D4"]["by_t2_round"]) == ["2"]
    # lo declines exactly half: not "most"
    rows = [_row("hi", True, True)] * 2 + [_row("lo", True, True)] * 2 + [_row("lo", True, False)] * 2
    assert not l2._d45(groups(rows))["D4"]["holds"]
    # a tie counts as a decline, and the hi rate must be strictly higher
    rows = [_row("hi", True, False, strict=False)] + [_row("lo", True, False, strict=False)]
    d = l2._d45(groups(rows))
    assert not d["D4"]["holds"] and d["D4"]["by_level"]["lo"]["declined_tie"] == 1


# -- regeneration -------------------------------------------------------------------------

CASES = [(l1, "results/split_v03_l1_20261110/sweep.json", "SWEEP_SPLIT_V03_L1.md"),
         (l2, "results/split_v03_l2_20261110/sweep.json", "SWEEP_SPLIT_V03_L2.md")]


@pytest.mark.parametrize("mod,rel,note", CASES, ids=["l1", "l2"])
def test_regeneration(mod, rel, note):
    path = os.path.join(ROOT, *rel.split("/"))
    if not os.path.exists(path):
        pytest.skip("sweep results missing")
    with open(path) as f:
        stored = json.load(f)
    fresh = mod.compute(workers=WORKERS)
    fresh["commit"], fresh["seconds"] = stored["commit"], stored["seconds"]
    assert json.loads(json.dumps(fresh)) == stored
    note_path = os.path.join(ROOT, "llm_rl_scripts", "split_deal", note)
    if os.path.exists(note_path):
        with open(note_path) as f:
            assert f.read() == mod.render(stored, rel)
