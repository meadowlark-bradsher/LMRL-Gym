"""Answer probe (gates 2, 3, 9): the item grid, an offline collection with the fake client, a balanced blind packet whose
key regenerates, and the gate logic on planted labels and Noul reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import answer_probe as ap
from . import arms_noul2 as a2
from .run_noul2 import FakeSonnet

CAL = Path("results/noul_bank_20261108/calibration_noul2.json")
needs_cal = pytest.mark.skipif(not CAL.exists(), reason="frozen Stage 1 calibration file not present")


def test_item_grid_covers_every_brief_role_ranking_question():
    its = ap.items()
    assert len(its) == 3 * 2 * 6 * 9 and len({i["id"] for i in its}) == len(its)
    assert all(i["truthful"] in ("yes", "no") for i in its)
    st = ap.ask_state("seller", "top:price")
    assert st.turn == "buyer" and st.terminal is None and not st.events


@pytest.fixture(scope="module")
def probe_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("probe")
    prov = ap.collect(d, FakeSonnet(), workers=2)
    assert prov["n_rows"] == 324 and prov["n_written"] == 324
    # resumable: a second collect writes nothing
    assert ap.collect(d, FakeSonnet(), workers=2)["n_written"] == 0
    return d


def test_packet_is_balanced_blind_and_sealed(probe_dir):
    rows = ap.load_bank(probe_dir / "probe_bank.jsonl")
    built = ap.build_packet(rows, seed=5, n_per_brief=20)
    assert len(built["packet"]) == 60
    for b in ap.BRIEFS:
        ks = [k for k in built["key"] if k["regime"] == b]
        assert len(ks) == 20 and sum(k["truthful"] == "yes" for k in ks) == 10
    assert all(set(p) == {"item", "role", "question", "reply"} for p in built["packet"])   # nothing that identifies truth
    assert ap.build_packet(rows, seed=5, n_per_brief=20)["key"] == built["key"]
    ap.write_packet(built, probe_dir)
    assert (probe_dir / "audit_view.txt").exists() and len(json.load((probe_dir / "audit_key.json").open())) == 60


@needs_cal
def test_report_gates_on_planted_labels_and_reads(probe_dir):
    rows = ap.load_bank(probe_dir / "probe_bank.jsonl")
    built = ap.build_packet(rows, seed=5, n_per_brief=20)
    key = built["key"]
    truthful = {k["item"]: k["truthful"] for k in key}
    # perfect labels: candid/guarded never lie; adversarial does not mislead -> inert
    rep = ap.report(rows, key, dict(truthful), None, None)
    assert rep["gates"]["2_candid_never_lies"] and rep["gates"]["2_guarded_never_lies"] and not rep["gates"]["2_adversarial_misleads"]
    assert "9_candid_no_confident_wrong" not in rep["gates"]
    # flip every adversarial label and one candid label
    flipped = dict(truthful)
    for k in key:
        if k["regime"] == "adversarial":
            flipped[k["item"]] = "no" if k["truthful"] == "yes" else "yes"
    first_candid = next(k["item"] for k in key if k["regime"] == "candid")
    flipped[first_candid] = "no" if truthful[first_candid] == "yes" else "yes"
    rep = ap.report(rows, key, flipped, None, None)
    assert not rep["gates"]["2_candid_never_lies"] and rep["gates"]["2_adversarial_misleads"]
    assert rep["briefs"]["candid"]["contradictions"] == [first_candid]
    # the fake client answers "A" to everything: regex parses nothing, so guarded accuracy is 0 -> gate 3 passes vacuously here
    assert rep["gates"]["3_guarded_is_guarded"] and rep["briefs"]["guarded"]["regex_unparsed_bank"] == rep["briefs"]["guarded"]["n_bank"]
    # Noul reads: a confident wrong raw read on one audited candid item survives Platt+clamp -> gate 9 fails; sharp right reads pass
    cal = a2.Calibration(CAL)
    good = {r["item"]["id"]: {"p_yes": 0.999 if r["item"]["truthful"] == "yes" else 0.001} for r in rows}
    rep = ap.report(rows, key, dict(truthful), good, cal)
    assert rep["gates"]["9_candid_no_confident_wrong"] and rep["briefs"]["candid"]["noul"]["n_read"] == 108
    cid = next(k["id"] for k in key if k["regime"] == "candid")
    bad = dict(good)
    bad[cid] = {"p_yes": 0.001 if good[cid]["p_yes"] > 0.5 else 0.999}
    rep = ap.report(rows, key, dict(truthful), bad, cal)
    assert not rep["gates"]["9_candid_no_confident_wrong"] and rep["briefs"]["candid"]["noul"]["confident_wrong_audited"] == [cid]


def test_cli_packet_and_report_round_trip(probe_dir):
    assert ap.main(["packet", "--out", str(probe_dir), "--n", "10"]) == 0
    key = json.load((probe_dir / "audit_key.json").open())
    with (probe_dir / "audit_labels.jsonl").open("w") as f:
        for k in key:
            f.write(json.dumps({"item": k["item"], "label": k["truthful"]}) + "\n")
    assert ap.main(["report", "--out", str(probe_dir), "--labeller", "test"]) == 0
    rep = json.load((probe_dir / "gates_answers.json").open())
    assert rep["labeller"] == "test" and rep["gates"]["2_candid_never_lies"] and "inputs" in rep


PROBE = Path("results/noul2_probe_20261108")
needs_probe = pytest.mark.skipif(not (PROBE / "gates_answers.json").exists(), reason="probe result dir not present")


@needs_probe
def test_probe_gate_files_regenerate():
    """The sealed key regenerates from the probe bank, and the committed gate verdicts regenerate from bank + key + labels
    + Noul matrix under the committed calibration."""
    from . import decode as dc
    rows = ap.load_bank(PROBE / "probe_bank.jsonl")
    assert len(rows) == 324 and all(not r["defaulted"] and not r["retried"] for r in rows)
    key = json.load((PROBE / "audit_key.json").open())
    assert ap.build_packet(rows)["key"] == key
    labels = ap.load_labels(PROBE / "audit_labels.jsonl")
    noul = dc.load_matrix(dc.matrix_path(PROBE, "noul"))
    rep = ap.report(rows, key, labels, noul, a2.Calibration(CAL))
    stored = json.load((PROBE / "gates_answers.json").open())
    assert rep["gates"] == stored["gates"] and all(stored["gates"].values())
    for b in ap.BRIEFS:
        assert rep["briefs"][b]["contradictions"] == stored["briefs"][b]["contradictions"]
        assert rep["briefs"][b]["noul"]["confident_wrong_audited"] == stored["briefs"][b]["noul"]["confident_wrong_audited"]
