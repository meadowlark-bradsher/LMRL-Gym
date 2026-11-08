"""The frozen CD-PERSONA-1 result regenerates from the committed transcripts and audit labels.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_persona/test_results_current_frozen.py

Skipped when the frozen result directory is absent. Everything is deterministic (seeded packet,
seeded bootstrap), so the comparison is exact.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from . import analyze_current as ac
from . import audit as au

FROZEN = os.path.join("results", "persona_current_20261107")
BASELINE = os.path.join("results", "persona_baseline2023_20261106", "results.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(FROZEN, "results.json")) and os.path.exists(BASELINE)),
    reason="frozen results absent")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("arm", ac.ARMS)
def test_audit_packet_regenerates(arm):
    rows = ac.build_rows(ac.load_results(os.path.join(FROZEN, f"arm_{arm}.jsonl")), arm)
    built = ac.build_packet(rows, arm)
    assert built["key"] == _load(os.path.join(FROZEN, f"audit_key_{arm}.json"))
    assert built["packet"] == _load_jsonl(os.path.join(FROZEN, f"audit_packet_{arm}.jsonl"))


def test_report_regenerates(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for arm in ac.ARMS:
        for name in (f"arm_{arm}.jsonl", f"audit_key_{arm}.json", f"audit_labels_{arm}.jsonl"):
            shutil.copy(os.path.join(FROZEN, name), out / name)
    assert ac.main(["report", "--confirmatory", "--dir", str(out), "--baseline", BASELINE]) == 0
    assert _load(out / "results.json") == _load(os.path.join(FROZEN, "results.json"))


def test_note_matches_frozen_results(tmp_path):
    from . import render_current_note as rn
    out = tmp_path / "note.md"
    assert rn.main(["--dir", FROZEN, "--out", str(out)]) == 0
    committed = os.path.join(os.path.dirname(__file__), "RESULTS_CURRENT.md")
    with open(committed) as f:
        assert f.read() == out.read_text()


def test_registered_numbers_in_results():
    # the verdicts quoted in RESULTS_CURRENT.md's headline, pinned so a drift in either file is loud
    R = _load(os.path.join(FROZEN, "results.json"))
    assert R["M1_verdict"] == {"A": {"corrected_rate": 0.0, "ci95": [0.0, 0.0], "band": "HELD", "borderline": False},
                               "N": {"corrected_rate": 0.0, "ci95": [0.0, 0.0], "band": "HELD", "borderline": False}}
    assert {k: v["verdict"] for k, v in R["predictions"].items() if k.startswith("P")} == {
        "P1": "REFUTED", "P2": "REFUTED", "P3": "REFUTED", "P4": "CONFIRMED", "P5": "CONFIRMED",
        "P6": "REFUTED", "P7": "REFUTED", "P8": "VOID"}
    assert au.AUDIT_SEED == ac.AUDIT_SEED
