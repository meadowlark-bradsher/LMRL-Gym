"""Gates for the confirmatory analyzer and the hand-audit, on synthetic data only.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_persona

Nothing here reads the released dataset: the confirmatory split stays untouched until
the registration is frozen.

1. Wilson intervals and the verdict bands behave at the edges;
2. on a synthetic corpus with known personas the five measures come out as constructed;
3. the predictions are scored from the measures, and P1 uses the audit when given;
4. the audit packet has a fixed size, hides cell and scorer outcome, and is seeded;
5. the audit-corrected rate is the stratified estimate;
6. the analyzer refuses to read the confirmatory split without ``--confirmatory``.
"""

from __future__ import annotations

import os

import pytest

from . import analyze as an
from . import audit as au
from . import data as d

_B = {v: k for k, v in d.BUYER.items()}
_S = {v: k for k, v in d.SELLER.items()}

SALE = "Agent: The price is ${p}.\nBuyer: I'll take it.\nAgent: Congratulations on your new car!\n{note}"
SALE_DISC = ("Agent: It is listed at $99,000, but I can offer you a discount of $1,000.\n" + SALE)
NO_SALE = "Agent: The price is fixed at ${p}.\nBuyer: Then I'll have to look elsewhere.\n{note}"


def _rec(i, buyer, seller, budget, template, price, note="", source="train"):
    text = template.format(p=f"{price:,}", note=note) + f"[item {i}]"
    return {"id": f"r{i}", "source": source, "original_conversation": text,
            "buyer_info": {"background_buyer": _B[buyer], "background_seller": _S[seller],
                           "budget": f"{budget // 1000}k"}}


def _corpus():
    recs, i = [], 0

    def add(n, *args, **kw):
        nonlocal i
        for _ in range(n):
            recs.append(_rec(i, *args, **kw))
            i += 1

    # hard cells: 2 sales in 10 with S-features, 0 in 5 with S-concise -> M1 = 2/15
    add(2, "B-discount", "S-features", 50000, SALE, 60000, note="Probability: 100%")
    add(8, "B-discount", "S-features", 50000, NO_SALE, 60000, note="Probability: 0%")
    add(5, "B-discount", "S-concise", 50000, NO_SALE, 60000, note="Probability: 0%")
    # compatible discount pair: 10 discounted sales under budget
    add(10, "B-discount", "S-discount", 50000, SALE_DISC, 40000, note="Probability: 100%")
    # spec buyer: 5/10 with every seller -> gap 0
    for sel in an.SELLERS:
        add(5, "B-spec", sel, 30000, SALE, 20000, note="Probability: 0.8", source="eval")
        add(5, "B-spec", sel, 30000, NO_SALE, 20000, note="Probability: 0.2", source="eval")
    return recs


def test_wilson_and_bands():
    lo, hi = an.wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = an.wilson(5, 10)
    assert lo == pytest.approx(1 - hi)
    assert an.band(0.05)["band"] == "HELD" and an.band(0.051)["band"] == "SOFT"
    assert an.band(0.20)["band"] == "SOFT" and an.band(0.21)["band"] == "NOT_HELD"
    assert an.band(0.04, [0.01, 0.09])["borderline"] and not an.band(0.10, [0.07, 0.15])["borderline"]


def test_measures_on_synthetic_corpus():
    rows = an.build_rows(_corpus())
    m = an.measures(rows)
    assert m["n"] == 55 and m["unclear"] == 0

    assert (m["M1_cell"]["k"], m["M1_cell"]["n"]) == (2, 15) and m["M1_cell"]["band"] == "SOFT"
    assert m["M1_cell"]["by_seller"]["S-features"]["k"] == 2
    assert m["M1_cell"]["by_seller"]["S-concise"]["k"] == 0
    assert (m["M1_text"]["k"], m["M1_text"]["n"]) == (2, 15)        # B-discount, no discount offered
    assert (m["M2"]["k"], m["M2"]["n"]) == (0, 35) and m["M2"]["band"] == "HELD"

    gaps = m["M3"]["gaps"]
    assert gaps["B-discount"]["gap"] == pytest.approx(1.0 - (0.2 + 0.0) / 2)
    assert gaps["B-spec"]["gap"] == pytest.approx(0.0)

    # over budget: only the 2 hard-cell sales (60k on a 50k budget) of 27 priced sales
    assert (m["M4"]["k"], m["M4"]["n"]) == (2, 27)
    assert m["M4"]["by_budget"]["50000"]["k"] == 2 and m["M4"]["by_budget"]["30000"]["k"] == 0

    # self-report of exactly 0/1 agrees everywhere it is given; the rest is fractional
    assert (m["M5"]["k"], m["M5"]["n"]) == (25, 25)
    assert m["M5"]["fractional"] == {"n": 30, "sale": 15, "no_sale": 15, "unclear": 0}

    assert an.measures([r for r in rows if r["source"] == "eval"])["n"] == 30


def test_predictions_use_the_audit_for_p1():
    rows = an.build_rows(_corpus())
    m = an.measures(rows)
    assert an.predictions(m)["P1"]["verdict"] == "CONFIRMED"                     # raw 2/15 > 0.05
    audited = an.predictions(m, {"corrected_rate": 0.01})
    assert audited["P1"]["verdict"] == "REFUTED" and audited["P1"]["audit_corrected"]
    assert an.predictions(m)["P2"]["verdict"] == "REFUTED"                       # no discounts offered
    assert an.predictions(m)["P5"]["verdict"] == "CONFIRMED"


def _audit_rows(n_hard_sale, n_hard_no, n_out_sale=300, n_out_no=300):
    rows, i = [], 0
    for hard, outcome, n in ((True, "sale", n_hard_sale), (True, "no_sale", n_hard_no),
                             (False, "sale", n_out_sale), (False, "no_sale", n_out_no)):
        for _ in range(n):
            rows.append({"id": f"a{i:04d}", "text": f"conversation {i}", "outcome": outcome, "hard": hard})
            i += 1
    return rows


def test_audit_packet_is_fixed_size_and_blind():
    size = 2 * au.N_SALE + 2 * au.N_NO_SALE
    for n_hard_sale in (5, 60, 200):                       # few, exactly enough, many
        built = au.build_packet(_audit_rows(n_hard_sale, 100))
        assert len(built["packet"]) == size
        assert all(set(p) == {"item", "text"} for p in built["packet"])
        strata = [k["stratum"] for k in built["key"]]
        assert strata.count("hard_sale") == min(n_hard_sale, au.N_SALE)
        assert strata.count("hard_sale") + strata.count("decoy_sale") == 2 * au.N_SALE
        assert strata.count("hard_no_sale") == au.N_NO_SALE
    again = au.build_packet(_audit_rows(200, 100))
    assert [k["id"] for k in again["key"]] == [k["id"] for k in built["key"]]      # seeded


def test_corrected_rate_is_the_stratified_estimate():
    built = au.build_packet(_audit_rows(20, 80))
    hard_sales = [k["item"] for k in built["key"] if k["stratum"] == "hard_sale"]
    labels = {k["item"]: k["scorer_outcome"] for k in built["key"]}
    for item in hard_sales[:10]:                           # half the audited sales were not sales
        labels[item] = "no_sale"
    r = au.corrected_rate(20, 80, built["key"], labels)
    assert r["raw_rate"] == pytest.approx(0.2)
    assert r["precision"] == pytest.approx(0.5) and r["miss_rate"] == 0.0
    assert r["corrected_rate"] == pytest.approx(0.1)
    assert r["ci95"][0] <= 0.1 <= r["ci95"][1]

    for item in hard_sales[:10]:
        labels[item] = "unclear"
    assert au.corrected_rate(20, 80, built["key"], labels)["corrected_rate"] == pytest.approx(0.1)
    assert au.corrected_rate(20, 80, built["key"], labels, unclear_as_sale=True)["corrected_rate"] \
        == pytest.approx(0.2)

    del labels[hard_sales[0]]
    with pytest.raises(ValueError):
        au.corrected_rate(20, 80, built["key"], labels)


def test_analyzer_refuses_without_confirmatory_flag(tmp_path):
    out = tmp_path / "run"
    assert an.main(["packet", "--out", str(out)]) == 2
    assert an.main(["report", "--out", str(out)]) == 2
    assert not os.path.exists(out)
