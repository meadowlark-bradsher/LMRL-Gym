"""The frozen CD-NOUL-1 Stage 1 result regenerates from the committed (gzipped) runs.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_noul/test_regeneration_noul.py

Skipped when the frozen result directory is absent. The analysis is deterministic (seeded cluster
bootstrap, ``PYTHONHASHSEED=0`` as registered), so every comparison is exact: ``analysis.json`` from
the runs, ``verdicts.json`` from the analysis, ``RESULTS_NOUL.md`` from both, and the verdict table
pinned so a drift in any file is loud."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from . import analyze_noul as an
from . import render_noul_note as rn
from . import verdicts_noul as vn

FROZEN = os.path.join("results", "noul_stage1_20261108")
NOTE = os.path.join(os.path.dirname(__file__), "RESULTS_NOUL.md")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(FROZEN, "analysis.json")) and os.path.exists(an.runs_path(FROZEN))),
    reason="frozen results absent")


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_gzipped_runs_load_complete():
    recs, stubs = an.load_runs(an.runs_path(FROZEN))
    assert stubs == 0 and len(recs) == 18000
    assert len({an.key_of(r) for r in recs}) == 25


def test_analysis_regenerates(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for name in ("summary.json", "ceilings.json"):
        shutil.copy(os.path.join(FROZEN, name), out / name)
    shutil.copy(an.runs_path(FROZEN), out / os.path.basename(an.runs_path(FROZEN)))
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=os.getcwd())
    subprocess.run([sys.executable, "-m", "llm_rl_scripts.car_dealer_noul.analyze_noul", "--dir", str(out),
                    "--B", "5000", "--seed", "0"], check=True, env=env, capture_output=True)
    assert _load(out / "analysis.json") == _load(os.path.join(FROZEN, "analysis.json"))


def test_verdicts_regenerate(tmp_path):
    shutil.copy(os.path.join(FROZEN, "analysis.json"), tmp_path / "analysis.json")
    assert vn.main(["--analysis", str(tmp_path / "analysis.json")]) == 0
    assert _load(tmp_path / "verdicts.json") == _load(os.path.join(FROZEN, "verdicts.json"))


def test_note_matches_frozen_results(tmp_path):
    out = tmp_path / "note.md"
    assert rn.main(["--dir", FROZEN, "--out", str(out)]) == 0
    with open(NOTE) as f:
        assert f.read() == out.read_text()


def test_registered_verdicts_pinned():
    V = _load(os.path.join(FROZEN, "verdicts.json"))
    assert V["gates_pass"] is True
    assert {k: v["verdict"] for k, v in V["rules"].items()} == {
        "P1": "positive but below captured minimum",
        "P2": "confirmed", "P3": "not supported", "P4": "confirmed",
        "P5_noul_answer": "confirmed", "P5_regex": "confirmed",
        "P6_plain": "confirmed", "P6_oblique": "confirmed",
        "P7_jev_agent_plain": "confirmed", "P7_jev_agent_oblique": "confirmed",
        "P7_sonnet_agent_plain": "confirmed", "P7_sonnet_agent_oblique": "confirmed",
        "P8": "confirmed"}
    cap = V["rules"]["P1"]["captured_fraction_of_delta_menu"]["point"]
    assert 0.499 < cap < vn.CAPTURED_MIN, cap
