"""Checks for gate SD3-K (``sweep_k.py``): the per-job reduction, the gate logic on synthetic inputs, a
smoke run end to end on a small test instance, and regeneration of the stored results and note.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_sweep_k.py
"""

import json
import os
from fractions import Fraction

import pytest

from . import sweep_k as sk
from .game import Instance
from .sweep_l2 import _jsonable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = "results/split_v03_k_20261110/sweep.json"
NOTE = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "SWEEP_SPLIT_K.md")
TINY = [2, 3, 4]                       # the non-canonical 10-type test instance of test_readers.py


# -- the reduction ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_t3():
    return sk.run_job((TINY, "primary", "T3"))


def test_summary_levels_average_to_all_and_roots_realize(tiny_t3):
    g = tiny_t3
    assert g["checks"] == {"b2_roots_realized": True, "a3_roots_realized": True}
    assert set(g["pairings"]) == {"A1/B0", "A1/B2", "A1/B2d", "A3/B2"}
    for name, blk in g["pairings"].items():
        for f in sk.FIELDS:
            assert (blk["lo"][f] + blk["hi"][f]) / 2 == blk["all"][f], (name, f)
        for lv in sk.LVS:
            c = blk["counts"][lv]
            assert 0 <= c["high"][0] <= c["high"][1] and 0 <= c["value"][0] <= c["value"][1]
        if name != "A3/B2":
            assert blk["counts"]["lo"]["misread"] == blk["counts"]["hi"]["misread"] == 0


def test_b2_never_worse_for_b_than_b0(tiny_t3):
    blk = tiny_t3["pairings"]
    assert blk["A1/B2"]["all"]["YB"] >= blk["A1/B0"]["all"]["YB"]
    assert blk["A3/B2"]["all"]["YA"] >= blk["A1/B2"]["all"]["YA"]


# -- the gate logic ---------------------------------------------------------------------------

def _counts(lo_ans, lo_n, hi_ans, hi_n):
    z = {"value": [0, 0], "asked_high": 0, "misread": 0}
    return {"lo": dict(z, high=[lo_ans, lo_n]), "hi": dict(z, high=[hi_ans, hi_n])}


def test_k3_support_rule_and_ordering():
    assert sk.SUPPORT == 100
    assert sk._k3(_counts(40, 100, 80, 100))["verdict"] == "holds"
    assert sk._k3(_counts(50, 100, 80, 100))["verdict"] == "fails"          # declines exactly half: not most
    assert sk._k3(_counts(40, 100, 30, 100))["verdict"] == "fails"          # hi must answer more than lo
    assert sk._k3(_counts(40, 99, 80, 100))["verdict"] == "not testable"
    assert sk._k3(_counts(0, 0, 0, 0))["verdict"] == "not testable"


def _blk(ya_lo, ya_hi, yb, counts=None):
    sc = lambda ya: {"YA": Fraction(ya), "YB": Fraction(yb), "YA_c": Fraction(ya), "YB_c": Fraction(yb)}  # noqa: E731
    lo, hi = sc(ya_lo), sc(ya_hi)
    return {"lo": lo, "hi": hi, "all": sc((Fraction(ya_lo) + Fraction(ya_hi)) / 2),
            "counts": counts or _counts(0, 0, 0, 0)}


def _pd_cells(t, r, p, s):
    """2 x 2 arms whose pooled Y_A and Y_B give temptation t, reward r, punishment p, sucker s to both."""
    vals = {"none/all": (t, s), "all/all": (r, r), "none/none": (p, p), "all/none": (s, t)}
    return {f"game:{c}": {name: {"all": {"YA": Fraction(a), "YB": Fraction(b), "YA_c": Fraction(a),
                                         "YB_c": Fraction(b)}}
                          for name in ("A1/B2", "A3/B2", "A1/B0")} for c, (a, b) in vals.items()}


def test_readings_k1_k2_k4_k5():
    P = {"T2": {"A1/B2": _blk("0.50", "0.40", "0.40"), "A1/B0": _blk("0.46", "0.42", "0.42"),
                "A3/B2": _blk("0.48", "0.40", "0.40")},
         "T3": {"A1/B2": _blk("0.45", "0.40", "0.45"), "A1/B0": _blk("0.50", "0.44", "0.41"),
                "A3/B2": _blk("0.50", "0.41", "0.43", _counts(40, 100, 80, 100))}}
    P.update(_pd_cells("0.5", "0.4", "0.3", "0.2"))                       # a strict PD for both
    k = sk.readings(P)
    assert k["K1"]["holds"]                                              # 0.45 > 0.41
    assert k["K2"]["holds"]                                              # 0.45 < 0.50
    assert k["K3"]["verdict"] == "holds"
    assert k["K4"]["holds"]                                              # 0.02 < 0.04
    assert not k["K5"]["holds"]                                          # a strict PD: K5 fails
    P.update(_pd_cells("0.3", "0.4", "0.2", "0.25"))                      # disclosure dominant: no PD
    assert sk.readings(P)["K5"]["holds"]


# -- end to end on the small instance ---------------------------------------------------------

def test_smoke_compute_and_render():
    doc = sk.compute(instances=[Instance(tuple(TINY))], workers=1)
    doc["commit"], doc["seconds"] = "smoke", 0
    doc = json.loads(json.dumps(doc))
    assert doc["checks"] == {"b2_roots_realized": True, "a3_roots_realized": True}
    assert doc["gate"]["sd3d_match"] is None                             # only checked on the nine instances
    assert len(doc["groups"]) == len(sk.jobs([Instance(tuple(TINY))]))
    text = sk.render(doc, "smoke.json")
    assert text.startswith("# Split Deal v0.3 — B2, A3 and gate SD3-K")
    par = json.loads(json.dumps(sk.compute(instances=[Instance(tuple(TINY))], workers=2)))   # the run's pool path
    par["commit"], par["seconds"] = "smoke", 0
    assert par == doc


# -- regeneration -----------------------------------------------------------------------------

def _stored():
    path = os.path.join(ROOT, *REL.split("/"))
    if not os.path.exists(path):
        pytest.skip("sweep results missing")
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("arm", ["T2", "game:all/none"])
def test_regenerate_groups(arm):
    stored = _stored()
    fresh = json.loads(json.dumps(_jsonable(sk.run_job(([2, 2, 3], "primary", arm)))))
    match = [g for g in stored["groups"] if g["instance"] == [2, 2, 3] and g["config"] == "primary" and g["arm"] == arm]
    assert match == [fresh]


def test_regenerate_note():
    stored = _stored()
    if os.path.exists(NOTE):
        with open(NOTE) as f:
            assert f.read() == sk.render(stored, REL)
