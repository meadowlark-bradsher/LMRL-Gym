"""Phase 0 gates for CD-NOUL-1: the bank enumeration, prompts, resumable generation, the
regex decoder, the likelihood plumbing and the blind-audit packet. No model is called."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

import pytest

from llm_rl_scripts.car_dealer_binned.questions import BANK, BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_persona.run import Completion

from . import audit as au
from . import decode as dc
from . import verbalize as vb


class FakeClient:
    """Replies with a canned sentence that encodes the truthful answer, or with nothing
    (``empty_first``) on the first call to exercise the retry path."""

    def __init__(self, empty_first: bool = False, refuse_ids: Optional[set] = None):
        self.calls: List[dict] = []
        self.empty_first = empty_first

    def complete(self, system: Optional[str], messages: List[dict], max_tokens: int) -> Completion:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        if self.empty_first and len(self.calls) == 1:
            return Completion(text="", model="fake", stop_reason="max_tokens", input_tokens=1, output_tokens=max_tokens)
        truthful = "YES" if "answer is YES" in messages[0]["content"] else "NO"
        text = "Yes, that is what I care about most." if truthful == "YES" else "No, not really."
        return Completion(text=f'"{text}"', model="fake", stop_reason="end_turn", input_tokens=10, output_tokens=12)


# ---- bank enumeration and prompts ---------------------------------------------------------

def test_items_enumerate_every_cell_once():
    its = vb.items()
    assert len(its) == 2 * 2 * 9 * 6 * vb.K_VARIANTS == 4320
    assert len({it.id for it in its}) == len(its)
    cells = {(it.regime, it.role, it.question_id, it.ranking) for it in its}
    assert len(cells) == 2 * 2 * 9 * 6
    for it in its[:200]:
        assert it.truthful == BANK_BY_ID[it.question_id].truthful_answer(it.ranking)


def test_prompts_carry_ranking_question_and_truth_and_regime_rules():
    it = vb.items(["oblique"], k=1)[5]
    sysm, user = vb.system_prompt(it), vb.user_prompt(it)
    a, b, c = it.ranking
    assert f"{a} > {b} > {c}" in sysm and it.role in sysm and vb.other(it.role) in sysm
    assert "do not say yes or no outright" in sysm.lower()
    assert it.question_text in user and f"truthful answer is {it.truthful.upper()}" in user
    plain = vb.system_prompt(vb.items(["plain"], k=1)[5])
    assert "unmistakable" in plain and "beyond what the question itself asks" in plain
    for regime in vb.REGIMES:
        s = vb.system_prompt(vb.items([regime], k=1)[0])
        assert "Never state anything false" in s and "no numbers, no offers" in s


# ---- generation ---------------------------------------------------------------------------

def test_generate_is_resumable_and_records_usage(tmp_path: Path):
    its = vb.items(["plain"], k=1)[:7]
    path = tmp_path / "bank.jsonl"
    c = FakeClient()
    assert vb.generate(its[:4], c, path) == 4
    assert vb.generate(its, c, path) == 3          # only the missing three
    assert vb.generate(its, c, path) == 0
    rows = vb.load_bank(path)
    assert [r["item"]["id"] for r in rows] == sorted(it.id for it in its)
    r = rows[0]
    assert r["text"] and not r["text"].startswith('"')   # quotes stripped
    assert r["usage"]["input_tokens"] == 10 and r["model"] == "fake" and not r["refusal"] and not r["empty"]
    assert r["item"]["truthful"] in ("yes", "no") and r["item"]["question_text"]


def test_generate_threaded_matches_sequential(tmp_path: Path):
    its = vb.items(["oblique"], k=1)[:12]
    vb.generate(its, FakeClient(), tmp_path / "a.jsonl", workers=1)
    vb.generate(its, FakeClient(), tmp_path / "b.jsonl", workers=4)
    ra, rb = vb.load_bank(tmp_path / "a.jsonl"), vb.load_bank(tmp_path / "b.jsonl")
    assert [r["item"]["id"] for r in ra] == [r["item"]["id"] for r in rb]
    assert [r["text"] for r in ra] == [r["text"] for r in rb]


def test_empty_reply_is_retried_once_with_more_tokens(tmp_path: Path):
    it = vb.items(["plain"], k=1)[0]
    c = FakeClient(empty_first=True)
    row = vb.verbalize(it, c)
    assert row["retried"] and not row["empty"] and len(c.calls) == 2
    assert c.calls[0]["max_tokens"] == vb.MAX_TOKENS and c.calls[1]["max_tokens"] == vb.RETRY_MAX_TOKENS


# ---- decoders -----------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Yes, price is what I care about most.", "yes"),
    ("Absolutely — that's the first thing I look at.", "yes"),
    ("No, not really; the warranty matters more to me.", "no"),
    ("I wouldn't say so. Delivery is way down my list.", "no"),
    ("Hmm, that depends on what you can do for me.", None),
    ("Well, it is what it is — no, warranty comes first.", "no"),
    ("It is not my first concern, honestly.", "no"),
    ("That's not where my head is. Yes, I hear you, but no.", "no"),   # earliest cue wins (a known limit)
])
def test_regex_answer(text, expected):
    assert dc.regex_answer(text) == expected


def test_answer_vector_matches_bank_likelihoods_and_is_uninformative_at_half():
    q = BANK[3]   # above:price>warranty, 3 of 6 yes
    yes = dc.answer_vector(q.id, 1.0)
    no = dc.answer_vector(q.id, 0.0)
    assert yes == tuple(float(x) for x in q.likelihoods(Fraction(0))["yes"])
    assert no == tuple(float(x) for x in q.likelihoods(Fraction(0))["no"])
    half = dc.answer_vector(q.id, 0.5)
    assert all(abs(h - 0.5) < 1e-12 for h in half)
    hedged = dc.answer_vector(q.id, 1.0, Fraction(1, 5))
    assert set(round(h, 12) for h in hedged) == {0.2, 0.8}
    assert dc.regex_p_yes("Yes.") == 1.0 and dc.regex_p_yes("No.") == 0.0 and dc.regex_p_yes("Maybe.") == 0.5
    assert dc.oracle_p_yes("yes") == 1.0 and dc.oracle_p_yes("no") == 0.0


# ---- audit packet -------------------------------------------------------------------------

def _fake_bank(tmp_path: Path, k: int = 3) -> List[dict]:
    path = tmp_path / "bank.jsonl"
    vb.generate(vb.items(k=k), FakeClient(), path, workers=4)
    return vb.load_bank(path)


def test_packet_is_blind_balanced_and_keyed(tmp_path: Path):
    rows = _fake_bank(tmp_path)
    built = au.build_packet(rows, seed=1, n_per_regime=20)
    packet, key = built["packet"], built["key"]
    assert len(packet) == len(key) == 40
    assert set(packet[0]) == {"item", "role", "question", "reply"}      # nothing that reveals truth or regime
    assert [p["item"] for p in packet] == list(range(40)) == [k["item"] for k in key]
    for regime in vb.REGIMES:
        ks = [k for k in key if k["regime"] == regime]
        assert len(ks) == 20 and sum(k["truthful"] == "yes" for k in ks) == 10
    again = au.build_packet(rows, seed=1, n_per_regime=20)
    assert again == built                                                 # seeded
    au.write_packet(built, tmp_path)
    assert (tmp_path / "audit_packet.jsonl").exists() and (tmp_path / "audit_key.json").exists()
    view = (tmp_path / "audit_view.txt").read_text()
    assert "### item 0" in view and "truthful" not in view


def test_report_counts_contradictions_and_regex_accuracy(tmp_path: Path):
    rows = _fake_bank(tmp_path)
    built = au.build_packet(rows, seed=2, n_per_regime=20)
    key = built["key"]
    labels = {k["item"]: k["truthful"] for k in key}
    # flip one plain item to the wrong answer, mark one oblique item unclear
    plain = [k for k in key if k["regime"] == "plain"][0]
    obl = [k for k in key if k["regime"] == "oblique"][0]
    labels[plain["item"]] = "no" if plain["truthful"] == "yes" else "yes"
    labels[obl["item"]] = "unclear"
    rep = au.report(rows, key, labels)
    assert rep["regimes"]["plain"]["contradictions"] == [plain["item"]]
    assert rep["gates"]["plain"]["contradictions_ok"] is False
    assert rep["regimes"]["oblique"]["contradictions"] == [] and rep["gates"]["oblique"]["contradictions_ok"]
    assert rep["regimes"]["oblique"]["labels"]["unclear"] == 1
    # the fake bank answers plainly in both regimes, so regex is perfect and oblique is not oblique
    assert rep["regimes"]["plain"]["regex_accuracy_bank"] == 1.0
    assert rep["gates"]["oblique"]["oblique_enough"] is False
    lab_path = tmp_path / "audit_labels.jsonl"
    with lab_path.open("w") as f:
        for item, label in labels.items():
            f.write(json.dumps({"item": item, "label": label}) + "\n")
    assert au.load_labels(lab_path) == labels


# ---- decoder matrices ---------------------------------------------------------------------

from LLM_RL.jev.client import JevClient, MockJevBackend  # noqa: E402


def _mock_noul(state: str, qid: str, instr: str) -> float:
    """Reads the fake bank's canned replies: 'Yes, ...' or 'No, ...'."""
    said_yes = 'replied: "Yes' in state
    if qid == "yes":
        return 0.9 if said_yes else 0.1
    if qid == "no":
        return 0.1 if said_yes else 0.9
    # type questions: favour rankings whose first attribute is named in the question
    return 0.8 if instr.split("are ")[1].split(" >")[0] in state else 0.2


class FakeJudge:
    """Sonnet stand-in: JSON probability from the canned reply; malformed on the first call
    when asked, to exercise the retry."""

    def __init__(self, malformed_first: bool = False):
        self.calls = 0
        self.malformed_first = malformed_first

    def complete(self, system, messages, max_tokens):
        self.calls += 1
        if self.malformed_first and self.calls == 1:
            return Completion(text="I think probably yes.", model="fake", stop_reason="end_turn")
        p = 0.95 if 'replied: "Yes' in messages[0]["content"] else 0.05
        return Completion(text=f'{{"p_yes": {p}}}', model="fake", stop_reason="end_turn", input_tokens=5, output_tokens=5)


def test_noul_item_reads_both_decoders_from_one_call(tmp_path: Path):
    rows = _fake_bank(tmp_path, k=1)
    backend = MockJevBackend(noul_fn=_mock_noul)
    row = rows[0]
    out = dc.noul_item(JevClient(backend), row)
    assert len(backend.calls) == 1 and len(backend.calls[0]["questions"]) == 8
    assert set(out) >= {"id", "regime", "truthful", "p_yes", "type_vector", "raw"}
    assert (out["p_yes"] > 0.5) == (row["item"]["truthful"] == "yes")
    assert len(out["type_vector"]) == 6 and max(out["type_vector"]) == 1.0
    assert dc.p_yes_two_way(0.0, 0.0) == 0.5 and dc.p_yes_two_way(0.9, 0.1) == pytest.approx(0.9)
    assert dc.type_vector({dc.ranking_key(r): 0.0 for r in ALL_RANKINGS}) == tuple([1.0] * 6)


def test_sonnet_item_parses_json_and_retries_once(tmp_path: Path):
    rows = _fake_bank(tmp_path, k=1)
    j = FakeJudge(malformed_first=True)
    out = dc.sonnet_item(j, rows[0])
    assert j.calls == 2 and out["raw"]["retried"] and not out["malformed"]
    assert (out["p_yes"] > 0.5) == (rows[0]["item"]["truthful"] == "yes")
    assert dc.parse_p_yes('{"p_yes": 0.25}') == 0.25 and dc.parse_p_yes('{"p_yes": 1.5}') is None
    assert dc.parse_p_yes("no json here") is None


def test_build_matrix_is_resumable_and_calibration_reports(tmp_path: Path):
    rows = _fake_bank(tmp_path, k=1)
    path = dc.matrix_path(tmp_path, "noul")
    client = JevClient(MockJevBackend(noul_fn=_mock_noul))
    assert dc.build_matrix("noul", rows[:10], client, path, workers=3) == 10
    assert dc.build_matrix("noul", rows, client, path, workers=3) == len(rows) - 10
    assert dc.build_matrix("noul", rows, client, path) == 0
    noul = dc.load_matrix(path)
    assert set(noul) == {r["item"]["id"] for r in rows}
    spath = dc.matrix_path(tmp_path, "sonnet")
    dc.build_matrix("sonnet", rows, FakeJudge(), spath, workers=2)
    rep = dc.calibration(rows, {"noul": noul, "sonnet": dc.load_matrix(spath)})
    for regime in ("plain", "oblique"):
        r = rep[regime]
        assert r["oracle"]["accuracy"] == 1.0 and r["oracle"]["logloss_bits"] < 1e-4
        assert r["noul_answer"]["accuracy"] == 1.0 and r["sonnet_answer"]["accuracy"] == 1.0
        assert r["regex"]["accuracy"] == 1.0            # the fake bank answers plainly
        assert 0 < r["noul_type"]["posterior_on_true_mean"] <= 1
        assert r["noul_answer"]["logloss_bits"] == pytest.approx(-__import__("math").log2(0.9))


def test_calibration_reports_audited_unclear_and_inferred(tmp_path: Path):
    rows = _fake_bank(tmp_path, k=1)
    noul = {}
    for r in rows:
        noul[r["item"]["id"]] = dc.noul_item(JevClient(MockJevBackend(noul_fn=_mock_noul)), r)
    ids = [r["item"]["id"] for r in rows if r["item"]["regime"] == "plain"][:6]
    audit = {ids[0]: "unclear", ids[1]: "unclear", ids[2]: "yes", ids[3]: "no", ids[4]: "yes", ids[5]: "no"}
    rep = dc.calibration(rows, {"noul": noul}, audit)
    a = rep["plain"]["audited"]["noul_answer"]
    assert a["n_unclear"] == 2 and a["n_inferred"] == 4
    assert a["unclear_share_near_half"] == 0.0            # the mock is confident on every canned reply
    assert rep["plain"]["audited"]["regex"]["n_inferred"] == 4
    assert "audited" not in rep["oblique"] or rep["oblique"]["audited"]["noul_answer"]["n_unclear"] == 0
