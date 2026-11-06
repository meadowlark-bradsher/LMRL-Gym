"""Gates for the persona-audit data layer and outcome scorer.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_persona

1. the scorer is persona-blind: it takes the conversation text and nothing else;
2. the dialogue outcome ignores the narrator's self-report (annotations are stripped);
3. dialogue acts: acceptance, refusal and deferral are scored as sale / no_sale / no_sale,
   and a story with no decisive act is ``unclear``;
4. prices skip stated reductions; discount detection respects negation;
5. the self-report parser reads percentages and never reads through a "$" into a price;
6. the committed RESERVE labels (the gate set) pass the registered validity gate, and a
   deferral is never scored as a sale (the error the first gate exposed);
7. the splits are deterministic, disjoint and of the registered sizes (needs the data).
"""

from __future__ import annotations

import inspect
import os

import pytest

from . import data as d
from . import scorer as s
from . import validate as v


def test_scorer_is_persona_blind():
    assert list(inspect.signature(s.score).parameters) == ["text"]
    assert "buyer_info" not in inspect.getsource(s)


def test_outcome_ignores_self_report():
    sale = ("Agent: The price is $20,000.\n"
            "Buyer: Alright, I'll take it.\n"
            "Agent: Congratulations on your new car!\n"
            "Probability of buyer purchasing a car: 0%\n")
    x = s.score(sale)
    assert x["outcome"] == "sale" and x["self_report"] == 0.0

    walk = ("Agent: The price is fixed at $20,000.\n"
            "Buyer: Then I'll have to look elsewhere. (Probability: 100%, Price: $20,000)\n"
            "[Probability of purchase: 100%]\n")
    x = s.score(walk)
    assert x["outcome"] == "no_sale" and x["self_report"] == 1.0


def test_dialogue_acts():
    defer = "Agent: The price is $20,000.\nBuyer: I need some time to think about it.\n"
    assert s.score(defer)["outcome"] == "no_sale"
    nothing = "Agent: The price is $20,000.\nBuyer: What colors does it come in?\n"
    assert s.score(nothing)["outcome"] == "unclear"
    # a later refusal overrides an earlier acceptance
    flip = ("Buyer: I'll take it for $15,000.\nAgent: The price is $20,000.\n"
            "Buyer: Then I won't be able to make the purchase.\n")
    assert s.score(flip)["outcome"] == "no_sale"
    question = "Agent: It is $20,000.\nBuyer: Would I be able to take it today?\n"
    assert s.score(question)["outcome"] == "unclear"


def test_price_and_discount():
    text = ("Agent: It is listed at $35,000, but I can offer you a discount of $5,000, "
            "bringing the price down to $30,000.\n"
            "Buyer: I'll take it.\n"
            "Agent: Wonderful, let's proceed with the paperwork.\n")
    x = s.score(text)
    assert x["outcome"] == "sale" and x["price"] == 30000 and x["discount_offered"] is True
    assert s.score("Agent: The price is $79.06k.\nBuyer: I'll take it.\n")["price"] == pytest.approx(79060)

    no_disc = "Agent: I'm sorry, but we do not offer any discounts. Our prices are fixed.\n"
    assert s.score(no_disc)["discount_offered"] is False


def test_self_report_parser():
    assert s.self_report("Probability of buying: 90%") == pytest.approx(0.9)
    assert s.self_report("Probability: 0.65, Price: $46,000") == pytest.approx(0.65)
    assert s.self_report("(0% probability of purchase, $12,000 purchase price)") == 0.0
    # "very high" is not a number; must not read 29 out of the price
    assert s.self_report("The probability is very high, and the purchase price is $29,500.") is None
    # the last statement wins
    assert s.self_report("Probability: 0.3\nProbability: 1.0") == 1.0


def test_reserve_passes_registered_gate():
    gate = v.load_gold(v.GATE_PATH)
    assert len(gate) == d.N_RESERVE
    r = v.validate(gate)
    assert r["sale_gate"], (r["accuracy"], r["scorer_unclear_rate"])
    assert r["discount_gate"], r["discount_agreement"]
    assert r["missed_sales"] == 0


def test_deferral_is_not_a_sale():
    # the two errors on the first gate set: a conditional close and "if I decide to purchase"
    a = ("Agent: I can offer it at $48,000.\n"
         "Buyer: I'll need some time to think it over.\n"
         "Agent: If you decide to proceed with the purchase, we can complete the paperwork today.\n"
         "Buyer: I'll definitely keep your offer in mind. I'll let you know my decision soon.\n")
    b = ("Agent: I cannot lower the price.\n"
         "Buyer: Okay, I'll have to think about it.\n"
         "Agent: Take your time.\n"
         "Buyer: Thank you, I'll let you know if I decide to purchase.\n")
    assert s.score(a)["outcome"] == "no_sale"
    assert s.score(b)["outcome"] == "no_sale"
    # a speaker label with a parenthetical is still a turn
    c = "Agent (after a pause): We have a special offer: $50,000.\nBuyer: I'll take it.\n"
    x = s.score(c)
    assert x["outcome"] == "sale" and x["discount_offered"] is True


@pytest.mark.skipif(not os.path.exists(os.path.join(d.DEFAULT_DIR, "train.json")),
                    reason="dataset not fetched")
def test_splits_frozen():
    a, b = d.splits(), d.splits()
    assert {k: len(x) for k, x in a.items()} == {
        "dev": d.N_DEV, "gold": d.N_GOLD, "reserve": d.N_RESERVE, "confirmatory": 3889}
    assert [r["id"] for r in a["gold"]] == [r["id"] for r in b["gold"]]
    ids = [r["id"] for split in a.values() for r in split]
    assert len(ids) == len(set(ids))                       # disjoint, de-duplicated
    assert [g["id"] for g in v.load_gold(v.DEV_LABELS_PATH)] == [r["id"] for r in a["gold"]]
    assert [g["id"] for g in v.load_gold(v.GATE_PATH)] == [r["id"] for r in a["reserve"]]
