"""The frozen confirmatory result regenerates from the pinned data and the committed
audit labels.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_persona/test_results_frozen.py

Skipped when the dataset has not been fetched (``python3 -m ...car_dealer_persona.data --fetch``).
Everything is deterministic (seeded packet, seeded bootstrap), so the comparison is exact.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from . import analyze as an
from . import audit as au
from . import data as d

FROZEN = os.path.join("results", "persona_baseline2023_20261106")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(d.DEFAULT_DIR, "train.json")) and os.path.exists(FROZEN)),
    reason="dataset not fetched or frozen results absent")


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_audit_packet_regenerates():
    rows = an._confirmatory_rows(d.DEFAULT_DIR)
    built = au.build_packet(rows)
    assert built["key"] == _load(os.path.join(FROZEN, "audit_key.json"))
    with open(os.path.join(FROZEN, "audit_packet.jsonl")) as f:
        assert [json.loads(line) for line in f if line.strip()] == built["packet"]


def test_report_regenerates(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for name in ("audit_key.json", "audit_labels.jsonl"):
        shutil.copy(os.path.join(FROZEN, name), out / name)
    assert an.main(["report", "--confirmatory", "--out", str(out)]) == 0
    assert _load(out / "results.json") == _load(os.path.join(FROZEN, "results.json"))


def test_note_matches_frozen_results(tmp_path):
    from . import render_baseline_note as rn
    out = tmp_path / "note.md"
    assert rn.main(["--dir", FROZEN, "--out", str(out)]) == 0
    committed = os.path.join(os.path.dirname(__file__), "RESULTS_BASELINE2023.md")
    with open(committed) as f:
        assert f.read() == out.read_text()
