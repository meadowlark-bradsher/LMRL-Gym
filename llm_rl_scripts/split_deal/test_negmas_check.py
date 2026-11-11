"""Checks for the NegMAS cross-check (``negmas_check.py``) and for the boundary around it.

The boundary test needs no NegMAS and always runs: NegMAS (AGPL-3.0-or-later) may be imported only by
``negmas_check.py``, nothing may import that module, and no dependency file of the repository may list
it. The others run only where NegMAS is installed (see ``negmas_check.py`` for the environment).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_negmas_check.py
"""

import json
import os
import re

import pytest

from .game import Instance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REL = "results/split_negmas_check_20261111/check.json"
ALLOWED = {"negmas_check.py", "test_negmas_check.py"}
IMPORTS_NEGMAS = re.compile(r"^\s*(import|from)\s+negmas\b", re.M)
IMPORTS_CHECK = re.compile(r"^\s*(import|from)\s+\S*negmas_check\b|^\s*from\s+\.\s+import\s+.*\bnegmas_check\b", re.M)


def test_boundary():
    for dirpath, _, files in os.walk(os.path.join(ROOT, "llm_rl_scripts")):
        for name in files:
            if not name.endswith(".py") or (dirpath == HERE and name in ALLOWED):
                continue
            with open(os.path.join(dirpath, name)) as f:
                src = f.read()
            assert not IMPORTS_NEGMAS.search(src), f"{name} imports negmas"
            assert not IMPORTS_CHECK.search(src), f"{name} imports negmas_check"
    for name in ("setup.py", "requirements.txt", "environment.yml"):
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            with open(path) as f:
                assert "negmas" not in f.read().lower(), f"{name} lists negmas"


def test_agrees_on_small_instance():
    pytest.importorskip("negmas")
    from . import negmas_check as nc
    doc = nc.compute([Instance((2, 2, 3))])
    assert doc["agree"] and doc["mismatches"] == 0
    assert all(g["pairs"] == 81 for g in doc["v02"]) and doc["v03"][0]["pairs"] == 324


def test_detects_a_wrong_frontier(monkeypatch):
    """The comparison is not vacuous: our frontier computed at the wrong outside option disagrees."""
    pytest.importorskip("negmas")
    from . import negmas_check as nc
    from .frontier import pair_frontier
    monkeypatch.setattr(nc, "pair_frontier", lambda inst, va, vb, d: pair_frontier(inst, va, vb, d + 1))
    doc = nc.compute([Instance((2, 2, 3))])
    m = doc["v02"][0]["mismatch"]
    assert not doc["agree"] and m["acceptable"] > 0 and m["s_star"] > 0


def test_regenerate():
    pytest.importorskip("negmas")
    from . import negmas_check as nc
    path = os.path.join(ROOT, *REL.split("/"))
    if not os.path.exists(path):
        pytest.skip("check results missing")
    with open(path) as f:
        stored = json.load(f)
    fresh = json.loads(json.dumps(nc.compute()))
    for k in ("commit", "seconds"):
        stored.pop(k)
    assert fresh == stored
