"""Gates for the CD-PERSONA-1 analyzer (`analyze_current.py`) on synthetic conversations.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_persona/test_analysis_current.py

The Arm A conversations are produced by `run.run_agent` with scripted buyer replies, so
their seller lines are the real script and the frozen scorer reads real transcripts; the
Arm N conversations are templated 2023-style stories. Outcomes are assigned by position
within each cell, so every registered rate below is exact, not sampled.

1. rows carry the persona, the factors, both cell flags and the harness channels;
2. M1 / M1b / M3 / M4 / M5 / M6 come out at the planted rates;
3. the packet is the CD-PERSONA-0 180 plus the H_spec strata, text only, no reuse of an
   id across strata; the corrected rates reduce to the raw rate under perfect labels and
   move as the formula says under imperfect ones;
4. the decomposition arithmetic and the prediction verdicts;
5. the pilot split and the gate; the CLI refuses the confirmatory steps without the flag
   and the packet -> labels -> report round trip writes results.json.
"""

from __future__ import annotations

import json
import os

import pytest

from llm_rl_scripts.car_dealer_persona import analyze_current as ac
from llm_rl_scripts.car_dealer_persona import audit as au
from llm_rl_scripts.car_dealer_persona import run
from llm_rl_scripts.car_dealer_persona.test_harness import Fake

BUY = "That works for me, I'll take it.\nDECISION: BUY ${price}"
LEAVE = "No thanks, I'll pass.\nDECISION: LEAVE"
GO_ON = "Tell me more.\nDECISION: CONTINUE"
MALFORMED = "Hmm, let me think."


def _replies(buy_turn=None, leave_turn=None, malformed_first=False):
    """Scripted buyer replies: CONTINUE until the decision turn."""
    out = [MALFORMED] if malformed_first else []
    n = (buy_turn or leave_turn or 6) - 1 - len(out)
    out += [GO_ON] * max(n, 0)
    if buy_turn:
        out.append(BUY.format(price="0"))
    elif leave_turn:
        out.append(LEAVE)
    else:
        out.append(GO_ON)
    return out


# Per cell, the (buy_turn, leave_turn) pattern by position i within the cell (50 rows each).
def _plan_a(r, i):
    b, s = r.buyer, r.seller
    if b == "B-discount" and s == "S-discount":
        return (2, None) if i % 10 < 2 else ((3, None) if i % 10 < 9 else (None, 4))
    if b == "B-discount" and s == "S-features":
        return (2, None) if i % 10 < 2 else (None, 3)                  # 0.20 violation
    if b == "B-discount" and s == "S-concise":
        return (2, None) if i % 25 == 0 else (None, 2)                 # 0.04 violation
    if b == "B-spec":
        if r.feature_match == "missing-one":
            return (2, None) if i % 10 < 3 else (None, 2)              # ~0.3 spec violation
        return (2, None) if i % 10 < 8 else (None, 2)
    # B-impatient
    p = {"S-discount": 9, "S-features": 5, "S-concise": 3}[s]
    return (2, None) if i % 10 < p else (None, 2)


@pytest.fixture(scope="module")
def results_a():
    recs = run.sample_records("A", 450, 11)
    by_cell = {}
    for r in sorted(recs, key=lambda r: r.id):
        by_cell.setdefault(r.cell, []).append(r)
    out = []
    for cell, rows in by_cell.items():
        for i, r in enumerate(rows):
            buy, leave = _plan_a(r, i)
            malformed = (i % 50 == 7)
            res = run.run_agent(r, Fake(_replies(buy, leave, malformed_first=malformed)))
            out.append(res)
    return out


def _story(r, sale: bool, discount: bool, price: int, prob) -> str:
    agent = f"Agent: Welcome! This {r.brand} {r.classification} is ${price:,}."
    if discount:
        agent += " I can offer you a 10% discount today."
    buyer = "Buyer: Great, I'll take it." if sale else "Buyer: I'll pass, thanks. I'm looking elsewhere."
    tail = f"\n\n(Probability of purchase: {prob}, Purchase price: ${price:,})"
    return f"{agent}\n\n{buyer}{tail}"


@pytest.fixture(scope="module")
def results_n():
    recs = run.sample_records("N", 450, 12)
    by_cell = {}
    for r in sorted(recs, key=lambda r: r.id):
        by_cell.setdefault(r.cell, []).append(r)
    out = []
    for cell, rows in by_cell.items():
        for i, r in enumerate(rows):
            if r.buyer == "B-discount" and r.seller != "S-discount":
                sale, discount = i % 10 < 3, i % 10 < 1                 # 0.30 raw violation, 0.10 seller leak
            elif r.seller == "S-discount":
                sale, discount = i % 10 < 9, True
            else:
                sale, discount = i % 10 < 5, i % 10 < 1
            price = int(r.budget * (1.2 if i % 5 == 0 else 0.9))
            prob = 1.0 if sale else (0.0 if i % 4 else 0.5)
            out.append({"record": run.asdict(r), "original_conversation": _story(r, sale, discount, price, prob),
                        "refusal": False, "stop_reason": "end_turn", "models": ["fake"]})
    return out


# 1. rows --------------------------------------------------------------------------------

def test_rows_a(results_a):
    rows = ac.build_rows(results_a, "A")
    assert len(rows) == 450 and all(r["seller_ok"] for r in rows)
    r = rows[0]
    assert {"multiplier", "feature_match", "token_outcome", "sale_price", "hard", "hard_spec", "text"} <= set(r)
    assert sum(r["hard"] for r in rows) == 100 and 75 <= sum(r["hard_spec"] for r in rows) <= 78
    assert all((r["hard"] and r["buyer"] == "B-discount") or not r["hard"] for r in rows)
    with pytest.raises(ValueError):
        ac.build_rows(results_a[:1], "N")


def test_refusal_row_is_unclear(results_a):
    res = dict(results_a[0], refusal=True, token_outcome="refusal")
    row = ac.build_rows([res], "A")[0]
    assert row["refusal"] and row["outcome"] == "unclear"
    m = ac.measures([row], "A")
    assert m["refusals"] == 1 and m["M1_cell"]["n"] == 0


# 2. measures ----------------------------------------------------------------------------

def test_measures_a_hit_the_planted_rates(results_a):
    m = ac.measures(ac.build_rows(results_a, "A"), "A")
    m1 = m["M1_cell"]
    assert m1["by_seller"]["S-features"]["rate"] == pytest.approx(0.20)
    assert m1["by_seller"]["S-concise"]["rate"] == pytest.approx(0.04)
    assert m1["rate"] == pytest.approx(12 / 100) and m1["band"] == "SOFT"
    assert m["M1b_cell"]["rate"] == pytest.approx(0.30, abs=0.1)
    assert m["M1b_cell"]["all_features"]["rate"] == pytest.approx(0.80, abs=0.1)
    assert m["M2"] == {"by_construction": True, "seller_ok_all": True, "seller_violations": 0}
    cells = m["M3"]["cells"]
    assert cells["B-discount"]["S-discount"]["rate"] == pytest.approx(0.90)
    assert m["M3"]["gaps"]["B-impatient"]["gap"] == pytest.approx(0.3 - (0.9 + 0.5) / 2)
    assert m["M3"]["gaps"]["B-discount"]["gap"] == pytest.approx(0.9 - (0.2 + 0.04) / 2)


def test_m4_by_multiplier(results_a):
    m = ac.measures(ac.build_rows(results_a, "A"), "A")
    m4 = m["M4"]
    assert m4["rate"] == 1.0 and m4["n"] > 0                   # every sale at 1.25 is over budget
    assert m4["by_multiplier"]["0.8"]["over_budget"]["rate"] == 0.0
    assert m4["by_multiplier"]["1.0"]["over_budget"]["rate"] == 0.0
    assert set(m4["by_multiplier"]) == {"0.8", "1.0", "1.25"}
    assert m4["price_matches_close"]["rate"] == 1.0 and m4["sales_without_price"] == 0


def test_m5_token_vs_dialogue(results_a):
    m = ac.measures(ac.build_rows(results_a, "A"), "A")
    m5 = m["M5"]
    assert m5["rate"] == 1.0 and m5["disagreements"] == []
    assert m5["timeout"]["k"] == 0
    assert m5["malformed_replies"]["k"] == 9                    # one per cell at i == 7
    assert m5["malformed_replies"]["n"] > m5["malformed_replies"]["k"]


def test_m6_cumulative_and_per_offer(results_a):
    m6 = ac.measures(ac.build_rows(results_a, "A"), "A")["M6"]
    assert m6["n"] == 50
    c = {k: v["rate"] for k, v in m6["cumulative"].items()}
    assert c == pytest.approx({"0%": 0.0, "5%": 0.2, "10%": 0.9, "15%": 0.9, "close": 0.9})
    assert m6["per_offer"]["10%"]["rate"] == pytest.approx(35 / 40)
    assert m6["per_offer"]["15%"]["rate"] == 0.0 and m6["per_offer"]["15%"]["n"] == 5
    assert m6["left_without_buying"]["rate"] == pytest.approx(0.1)


def test_measures_n_are_the_2023_measures(results_n):
    m = ac.measures(ac.build_rows(results_n, "N"), "N")
    assert m["M1_cell"]["rate"] == pytest.approx(0.30)
    assert m["M2"]["rate"] == pytest.approx(0.10) and m["M2"]["band"] == "SOFT"
    assert m["M1_text"]["n"] < m["M1_cell"]["n"] + 50          # discount rows excluded
    assert m["M4"]["rate"] == pytest.approx(0.2, abs=0.03)
    assert set(m["M4"]["by_budget"]) == {"10000", "30000", "50000", "70000", "90000"}
    assert m["M5"]["rate"] == 1.0 and m["M5"]["fractional"]["n"] > 0
    assert "M1b_cell" not in m and "M6" not in m


# 3. packet and correction ---------------------------------------------------------------

def test_packet_a_has_both_cell_sets(results_a):
    rows = ac.build_rows(results_a, "A")
    built = ac.build_packet(rows, "A")
    key, packet = built["key"], built["packet"]
    assert len(key) == len(packet) and [p["item"] for p in packet] == list(range(len(packet)))
    assert all(set(p) == {"item", "text"} for p in packet)
    strata = {}
    for k in key:
        strata[k["stratum"]] = strata.get(k["stratum"], 0) + 1
    assert strata["hard_sale"] == 12 and strata["decoy_sale"] == 108
    assert strata["hard_no_sale"] == 30 and strata["decoy_no_sale"] == 30
    assert strata["spec_sale"] <= ac.N_SPEC_SALE and strata["spec_no_sale"] == ac.N_SPEC_NO_SALE
    assert len({k["id"] for k in key}) == len(key)            # no id in two strata
    by_id = {r["id"]: r for r in rows}
    assert all(by_id[k["id"]]["hard_spec"] for k in key if k["stratum"].startswith("spec_"))
    assert all(k["scorer_outcome"] == by_id[k["id"]]["outcome"] for k in key)
    assert built == ac.build_packet(rows, "A")                 # deterministic


def test_packet_n_is_the_180_design(results_n):
    built = ac.build_packet(ac.build_rows(results_n, "N"), "N")
    assert len(built["packet"]) == 180
    assert not any(k["stratum"].startswith("spec_") for k in built["key"])


def test_corrected_rates(results_a):
    rows = ac.build_rows(results_a, "A")
    key = ac.build_packet(rows, "A")["key"]
    perfect = {k["item"]: k["scorer_outcome"] for k in key}
    c = ac.corrected(rows, key, perfect, "A")
    assert c["M1"]["corrected_rate"] == pytest.approx(0.12)
    assert c["M1b"]["corrected_rate"] == pytest.approx(ac.measures(rows, "A")["M1b_cell"]["rate"])
    assert c["M1"]["ci95"][0] < 0.12 < c["M1"]["ci95"][1]
    # two audited hard sales relabelled no_sale: precision 10/12
    flipped = dict(perfect)
    hs = [k["item"] for k in key if k["stratum"] == "hard_sale"][:2]
    for item in hs:
        flipped[item] = "no_sale"
    c2 = ac.corrected(rows, key, flipped, "A")
    assert c2["M1"]["precision"] == pytest.approx(10 / 12)
    assert c2["M1"]["corrected_rate"] == pytest.approx((12 * 10 / 12 + 88 * 0) / 100)
    assert c2["M1b"]["corrected_rate"] == c["M1b"]["corrected_rate"]   # spec strata untouched
    # an unclear audit label counts as no sale unless the sensitivity says otherwise
    flipped[hs[0]] = "unclear"
    c3 = ac.corrected(rows, key, flipped, "A")
    assert c3["M1"]["precision"] == pytest.approx(10 / 12)
    assert c3["M1_unclear_as_sale"]["precision"] == pytest.approx(11 / 12)
    with pytest.raises(ValueError):
        ac.corrected(rows, key, {k: v for k, v in perfect.items() if k != 0}, "A")


# 4. decomposition and predictions -------------------------------------------------------

def test_decomposition_arithmetic():
    d = ac.decomposition({"rate": 0.20, "ci95": [0.17, 0.23]}, {"rate": 0.15, "ci95": [0.12, 0.18]},
                         {"rate": 0.05, "ci95": [0.03, 0.07]})
    assert d["model_effect"]["diff"] == pytest.approx(0.05)
    assert d["protocol_effect"]["diff"] == pytest.approx(0.10)
    se = ((0.06 / 3.92) ** 2 + (0.06 / 3.92) ** 2) ** 0.5
    assert d["model_effect"]["ci95"] == pytest.approx([0.05 - 1.96 * se, 0.05 + 1.96 * se])
    assert set(d["cells"]) == {"2023_narrator", "current_narrator", "current_agent"}


def _baseline():
    path = "results/persona_baseline2023_20261106/results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"audit": {"corrected_rate": 0.198, "ci95": [0.174, 0.223]}}


def test_predictions_on_the_synthetic_arms(results_a, results_n):
    rowsA, rowsN = ac.build_rows(results_a, "A"), ac.build_rows(results_n, "N")
    keyA, keyN = ac.build_packet(rowsA, "A")["key"], ac.build_packet(rowsN, "N")["key"]
    audA = ac.corrected(rowsA, keyA, {k["item"]: k["scorer_outcome"] for k in keyA}, "A")
    audN = ac.corrected(rowsN, keyN, {k["item"]: k["scorer_outcome"] for k in keyN}, "N")
    r = ac.analyze(rowsA, rowsN, audA, audN, _baseline())
    p = r["predictions"]
    assert p["P1"]["verdict"] == "CONFIRMED"                   # concise 0.04 held, features 0.20 not
    assert p["P2"]["verdict"] == "REFUTED"                     # Arm N 0.30 > 0.198
    assert p["P3"]["verdict"] == "CONFIRMED"                   # 0.30 - 0.12 >= 0.05
    assert p["P4"]["verdict"] == "CONFIRMED"
    assert p["P5"]["verdict"] == "CONFIRMED"
    assert p["P6"]["verdict"] == "CONFIRMED"
    assert p["P7"]["verdict"] == "CONFIRMED" and p["P7"]["value"]["15%"] == pytest.approx(0.9)
    assert p["P8"]["verdict"] == "VOID"
    assert r["M1_verdict"]["A"]["band"] == "SOFT" and r["M1_verdict"]["N"]["band"] == "NOT_HELD"
    assert r["decomposition"]["protocol_effect"]["diff"] == pytest.approx(0.18)


# 5. pilot, gate, CLI --------------------------------------------------------------------

def test_pilot_split_and_gate(results_a):
    rows = ac.build_rows(results_a[:170], "A")
    split = ac.pilot_split(rows)
    assert len(split["dev"]) == 80 and len(split["gate"]) == 80
    assert not {x["id"] for x in split["dev"]} & {x["id"] for x in split["gate"]}
    assert set(split["dev"][0]) == {"index", "id", "original_conversation"}
    assert split == ac.pilot_split(rows)
    by_id = {r["id"]: r for r in rows}
    labels = [{"id": x["id"], "outcome": by_id[x["id"]]["outcome"],
               "discount_offered": by_id[x["id"]]["discount"], "price": by_id[x["id"]]["price"]}
              for x in split["gate"] if by_id[x["id"]]["outcome"] != "unclear"]
    g = ac.gate(rows, labels)
    assert g["sale_gate"] and g["discount_gate"] and g["accuracy"] == 1.0


def _write(results, path):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def test_cli_round_trip(tmp_path, results_a, results_n, capsys):
    d = str(tmp_path)
    _write(results_a, os.path.join(d, "arm_A.jsonl"))
    _write(results_n, os.path.join(d, "arm_N.jsonl"))
    assert ac.main(["packet", "--dir", d]) == 2
    assert ac.main(["report", "--dir", d]) == 2
    assert not os.path.exists(os.path.join(d, "audit_packet_A.jsonl"))

    assert ac.main(["pilot-packet", "--dir", d]) == 0
    assert os.path.exists(os.path.join(d, "pilot_gate_N.jsonl"))

    assert ac.main(["packet", "--dir", d, "--confirmatory"]) == 0
    out = capsys.readouterr().out
    assert "sale" not in out.replace("no_sale", "").replace("audit_labels", "")  # prints no counts
    for arm in ("A", "N"):
        with open(os.path.join(d, f"audit_key_{arm}.json")) as f:
            key = json.load(f)
        labels = [{"item": k["item"], "outcome": k["scorer_outcome"]} for k in key]
        _write(labels, os.path.join(d, f"audit_labels_{arm}.jsonl"))
    baseline = os.path.join(d, "baseline.json")
    with open(baseline, "w") as f:
        json.dump(_baseline(), f)
    assert ac.main(["report", "--dir", d, "--confirmatory", "--baseline", baseline]) == 0
    with open(os.path.join(d, "results.json")) as f:
        res = json.load(f)
    assert res["predictions"]["P1"]["verdict"] == "CONFIRMED"
    assert res["A"]["M1_cell"]["rate"] == pytest.approx(0.12)
    assert "P7: CONFIRMED" in capsys.readouterr().out
