"""Tests for ``scorer_current``: the DEV-half failure modes it was revised on, and no
regression against the frozen scorer on the 2023 hand labels."""

import json
import os

import pytest

from . import analyze_current as ac
from . import run
from . import scorer as s0
from . import scorer_current as s1
from . import validate as va
from .test_harness import Fake, rec

_GOLD = os.path.join(os.path.dirname(__file__), "gold")
_PILOT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "persona_current_pilot_20261107")


# -- markdown normalisation ---------------------------------------------------------
MD_STORY = """# Car Dealership Conversation

**Agent:** Welcome in! What brings you here?

**Buyer:** A Ford coupe, under $12,000. `[Probability: 20% | Price: $12,000]`

**Agent:** This one is $11,500, and I can't go lower.

**Buyer:** Fine, I'll take it. *[Probability: 90% | Price: $11,500]*

---
## Final Outcome
**Probability of purchase: 90% | Final purchase price: $11,500**
"""


def test_frozen_scorer_sees_no_turns_in_markdown():
    assert s0.score(MD_STORY)["n_turns"] == 0


def test_normalize_restores_2023_layout():
    turns = s1.split_turns(MD_STORY)
    assert [r for r, _ in turns] == ["dealer", "buyer", "dealer", "buyer"]
    assert turns[1][1].startswith("A Ford coupe")
    assert "[" not in turns[1][1] and "`" not in turns[1][1]


def test_markdown_story_scores():
    x = s1.score(MD_STORY)
    assert x["outcome"] == "sale" and x["price"] == 11500.0
    assert x["self_report"] == 0.9 and x["discount_offered"] is False


@pytest.mark.parametrize("label", ["**Buyer**: hi", "**Buyer:** hi", "> **Buyer:** hi",
                                   "**Buyer (Sam):** hi", "Buyer: hi"])
def test_speaker_label_variants(label):
    assert s1.split_turns(label) == [("buyer", "hi")]


# -- buyer-agent phrasing (Arm A DEV) -----------------------------------------------
@pytest.mark.parametrize("line", [
    "You've dodged my question four times. I'm done here.",
    "That's the third time. I'm not wasting more time, so I'm leaving.",
    "I'll take that as a no. I'm walking.",
    "I'm going to look at other options.",
    "If you change your mind, you have my number. Otherwise I'll check with other dealers.",
    "Then we're at an impasse. I'll take my business elsewhere.",
    "I'm not signing anything without knowing whether it has CarPlay, so I'm leaving.",
])
def test_walk_outs_are_no_sale(line):
    text = f"Agent: The price is $20,000.\n\nBuyer: {line}"
    assert s1.score(text)["outcome"] == "no_sale", line


@pytest.mark.parametrize("line", [
    "$8,000 works for me. Let's just get the paperwork done.",
    "Let's skip to the paperwork and get this done at that price.",
    "Fine, $11,500, get the paperwork.",
    "Yes, do it now, and please keep it short.",
    "Fine. Where do I sign?",
    "Get the keys, and have the paperwork ready when I'm done.",
])
def test_commitments_are_sale(line):
    text = f"Agent: The price is $20,000.\n\nBuyer: {line}"
    assert s1.score(text)["outcome"] == "sale", line


def test_temporal_im_done_is_not_a_walk_out():
    assert s1._buyer_act("Have the paperwork ready when I'm done.") == "accept"
    assert s1._buyer_act("I'm done here.") == "reject"


def test_no_thank_you_after_a_sale_is_not_a_walk_out():
    text = ("Agent: It's $42,500.\n\nBuyer: I'll take it.\n\n"
            "Agent: Would you like the extended warranty?\n\nBuyer: No, thank you.")
    assert s1.score(text)["outcome"] == "sale"


def test_conditional_offer_in_close_shape_is_not_a_close():
    text = ("Agent: Tell you what, $11,000 and we can start the paperwork.\n\n"
            "Buyer: My budget is about $10,000, and I need the third row.")
    assert s1.score(text)["outcome"] == "unclear"
    text2 = "Agent: Deal. I'll get the paperwork started at $30,000.\n\nBuyer: Fine, hurry up."
    assert s1.score(text2)["outcome"] == "sale" and s1.score(text2)["price"] == 30000.0


def test_if_another_dealer_is_not_an_acceptance():
    text = ("Agent: The price is $50,000.\n\nBuyer: Without a discount I'll have to pass.\n\n"
            "Agent: I'm sorry to hear that.\n\n"
            "Buyer: Thanks for your time. If another dealer offers a discount, I'll go with them.")
    assert s1.score(text)["outcome"] == "no_sale"


def test_difference_amounts_are_not_prices():
    text = ("Agent: The price is $71,500.\n\nBuyer: Ugh, that's $1,500 over what I wanted to spend. "
            "But I don't want to keep shopping.\n\nAgent: I can have the paperwork ready.\n\n"
            "Buyer: Fine, I'll take it.")
    assert s1.score(text)["price"] == 71500.0


# -- no regression on the 2023 hand labels ------------------------------------------
@pytest.mark.parametrize("name", ["gold", "reserve"])
def test_matches_frozen_scorer_on_2023_labels(name):
    gold = va.load_gold(os.path.join(_GOLD, f"{name}_labels.jsonl"))
    r0, r1 = va.validate(gold), va.validate(gold, scorer=s1)
    assert r1["accuracy"] >= r0["accuracy"]
    assert r1["scorer_unclear_rate"] <= r0["scorer_unclear_rate"]
    assert r1["discount_agreement"] >= r0["discount_agreement"]
    assert r1["price_exact"] >= r0["price_exact"]
    assert r1["sale_gate"] and r1["discount_gate"]


# -- the pilot gate, as recorded in the registration, regenerates from the labels ---------
@pytest.mark.parametrize("arm", ["A", "N"])
def test_pilot_gate_reproduces(arm):
    if not os.path.exists(os.path.join(_PILOT, f"gate_{arm}.json")):
        pytest.skip("pilot results not present")
    rows = [{"id": r["id"], "text": r["original_conversation"]}
            for r in ac.load_results(os.path.join(_PILOT, f"pilot_gate_{arm}.jsonl"))]
    labels = ac.load_results(os.path.join(_PILOT, f"pilot_gate_labels_{arm}.jsonl"))
    with open(os.path.join(_PILOT, f"gate_{arm}.json")) as f:
        stored = json.load(f)
    got = ac.gate(rows, labels)
    assert {k: v for k, v in got.items() if not isinstance(v, (list, dict))} == \
           {k: v for k, v in stored.items() if not isinstance(v, (list, dict))}
    assert got["sale_gate"] and got["discount_gate"]
    # the numbers quoted in PREREGISTRATION_CURRENT.md §Scorer re-gate
    expect = {"A": (5, 66, 0, 13, 12, 13), "N": (0, 22, 1, 45, 42, 45)}[arm]
    assert (got["false_sales"], got["n_labelled_no_sale"], got["missed_sales"],
            got["n_labelled_sale"], got["price_exact"], got["n_priced"]) == expect


# -- Arm A: the scorer never sees the token's echo ---------------------------------
def test_close_line_excluded_from_scored_text():
    r = rec("B-discount", "S-concise", 1.0, "all")
    res = run.run_agent(r, Fake(["Sure.\nDECISION: CONTINUE", "Okay.\nDECISION: BUY $30,000"]))
    full, scored = run.render_transcript(res), run.render_transcript(res, include_close=False)
    assert "let's finalize it" in full and "let's finalize it" not in scored
    assert "DECISION" not in scored
    # the words alone say nothing decisive here; with the echo the frozen rules would call a sale
    assert s1.score(scored)["outcome"] == "unclear"
