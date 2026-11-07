"""Gates for the CD-PERSONA-1 harness (`sellers.py`, `buyer.py`, `narrator.py`, `run.py`).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/car_dealer_persona/test_harness.py

No model is called: `run.py` is exercised with a scripted fake client.

1. the scripted sellers keep their persona in every line: the no-discount sellers never
   concede on price, the no-features sellers never name a feature, S-concise is short;
2. prices follow the registered schedule and the close uses the seller's price;
3. the buyer prompt carries its 2023 background verbatim and never the seller's;
4. the decision token parses, falls back to CONTINUE when malformed, and is stripped
   from the text the scorer reads;
5. the narrator constants and prompt function are identical to `personality_convos.py`;
6. record sampling is deterministic, balanced over cells and factors, and Arm N carries
   no Arm A factors;
7. `run_agent` ends on BUY / LEAVE / timeout / refusal as registered, and
   `verify_seller_lines` catches a tampered seller line;
8. the confirmatory phase refuses to run without the registration tag on HEAD.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm_rl_scripts.car_dealer_persona import buyer, narrator, run, sellers

CONCESSION = re.compile(r"discount|\boff\b|%|cheaper|reduce|lower the price|deal", re.I)
FEATURE_WORDS = re.compile("|".join(re.escape(f.lower()) for f in narrator.FEATURES)
                           + r"|feature|leather|seat|camera|bluetooth|navigation|sunroof", re.I)


def rec(buyer_="B-discount", seller="S-features", m=1.0, fm="all", features=None, budget="30k",
        arm="A"):
    features = features or ["Leather seats", "Backup camera", "Bluetooth"]
    missing = features[-1] if fm == "missing-one" else None
    dollars = narrator.BUDGET_DOLLARS[budget]
    if arm == "N":
        return run.Record("n0", "N", buyer_, seller, "Honda", "Sedan", features, budget, dollars)
    return run.Record("a0", "A", buyer_, seller, "Honda", "Sedan", features, budget, dollars,
                      multiplier=m, feature_match=fm, missing_feature=missing,
                      list_price=int(round(m * dollars)),
                      offered_features=[f for f in features if f != missing])


# 1. persona in every line ------------------------------------------------------------

@pytest.mark.parametrize("n_features", [1, 2, 3, 10])
@pytest.mark.parametrize("fm", ["all", "missing-one"])
def test_no_discount_sellers_never_concede(n_features, fm):
    r = rec(features=narrator.FEATURES[:n_features], fm=fm)
    for seller in ("S-features", "S-concise"):
        for turn in range(1, sellers.MAX_TURNS + 1):
            line = sellers.seller_line(seller, r, turn)
            if seller == "S-concise" and turn == 3:
                assert line == "No discounts."
            else:
                assert not CONCESSION.search(line), (seller, turn, line)
            assert sellers.current_price(seller, r, turn) == r.list_price
        close, price = sellers.close_line(seller, r, 3, "BUY")
        assert price == r.list_price and "$30,000" in close


def test_no_feature_sellers_never_name_a_feature():
    r = rec(features=list(narrator.FEATURES))
    for seller in ("S-discount", "S-concise"):
        for turn in range(1, sellers.MAX_TURNS + 1):
            assert not FEATURE_WORDS.search(sellers.seller_line(seller, r, turn)), (seller, turn)


def test_features_seller_names_only_offered_features():
    r = rec(features=["Leather seats", "Backup camera", "Bluetooth"], fm="missing-one")
    assert r.missing_feature == "Bluetooth"
    text = " ".join(sellers.seller_line("S-features", r, t) for t in range(1, 6))
    assert "Leather seats" in text and "Backup camera" in text
    assert "Bluetooth" not in text          # the omitted feature is simply absent
    assert "lack" not in text.lower() and "without" not in text.lower()


def test_concise_seller_is_concise():
    r = rec(seller="S-concise")
    for turn in range(1, 6):
        assert len(sellers.seller_line("S-concise", r, turn).split()) <= 6


def test_all_blurbs_cover_all_features_and_are_clean():
    assert set(sellers.FEATURE_BLURBS) == set(narrator.FEATURES)
    for blurb in sellers.FEATURE_BLURBS.values():
        assert not CONCESSION.search(blurb), blurb


# 2. prices ---------------------------------------------------------------------------

def test_discount_schedule_and_close_price():
    r = rec(seller="S-discount", m=1.0, budget="30k")
    assert [sellers.current_price("S-discount", r, t) for t in range(1, 6)] == \
        [30000, 28500, 27000, 25500, 25500]
    for t, p in [(2, "$28,500"), (3, "$27,000"), (4, "$25,500"), (5, "$25,500")]:
        assert p in sellers.seller_line("S-discount", r, t)
    close, price = sellers.close_line("S-discount", r, 3, "BUY")
    assert price == 27000 and "$27,000" in close
    close, price = sellers.close_line("S-discount", r, 3, "LEAVE")
    assert price is None and "$" not in close


def test_lines_are_deterministic():
    r = rec()
    for seller in sellers.SELLERS:
        a = [sellers.seller_line(seller, r, t) for t in range(1, 6)]
        b = [sellers.seller_line(seller, r, t) for t in range(1, 6)]
        assert a == b


# 3. buyer prompt ---------------------------------------------------------------------

def test_buyer_prompt_has_own_background_not_sellers():
    r = rec(buyer_="B-impatient", seller="S-features")
    p = buyer.system_prompt(r)
    assert narrator.BACKGROUND_BUYERS[2] in p
    for s in narrator.BACKGROUND_SELLERS:
        assert s not in p
    assert "$30,000" in p and "Honda Sedan" in p
    assert "Leather seats, Backup camera, Bluetooth" in p
    assert "DECISION: CONTINUE" in p and "DECISION: BUY $<price>" in p and "DECISION: LEAVE" in p
    assert buyer.SYSTEM_TEMPLATE.count("{") == 5


# 4. decision token -------------------------------------------------------------------

@pytest.mark.parametrize("text,kind,price,malformed", [
    ("I'd like to hear more.\nDECISION: CONTINUE", "CONTINUE", None, False),
    ("That works for me.\nDECISION: BUY $28,500", "BUY", 28500, False),
    ("Fine.\n\ndecision: buy 27000.", "BUY", 27000, False),
    ("Deal.\n`DECISION: BUY $25,500`", "BUY", 25500, False),
    ("No thanks.\nDECISION: LEAVE", "LEAVE", None, False),
    ("DECISION: LEAVE\nActually, wait.\nDECISION: CONTINUE", "CONTINUE", None, False),
    ("I'll think about it.", "CONTINUE", None, True),
    ("I'll take it!\nDECISION: PURCHASE", "CONTINUE", None, True),
])
def test_parse_decision(text, kind, price, malformed):
    d = buyer.parse_decision(text)
    assert (d.kind, d.price, d.malformed) == (kind, price, malformed)


def test_strip_decisions_removes_only_token_lines():
    text = "I like it, but I need a discount.\nDECISION: CONTINUE\n"
    assert buyer.strip_decisions(text) == "I like it, but I need a discount."
    assert buyer.strip_decisions("Deal at $27,000.\n`DECISION: BUY $27,000`") == "Deal at $27,000."
    assert buyer.strip_decisions("No token here.") == "No token here."


def test_messages_alternate_from_seller():
    m = buyer.messages(["s1", "s2"], ["b1"])
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[-1]["content"] == "s2"
    with pytest.raises(ValueError):
        buyer.messages(["s1"], ["b1"])


# 5. narrator is the 2023 script ------------------------------------------------------

def test_narrator_constants_match_personality_convos():
    assert narrator.SOURCE.exists()
    src = narrator.source_constants()
    assert src["background_buyers"] == narrator.BACKGROUND_BUYERS
    assert src["background_sellers"] == narrator.BACKGROUND_SELLERS
    assert src["brands"] == narrator.BRANDS
    assert src["features"] == narrator.FEATURES
    assert src["budgets"] == narrator.BUDGETS
    assert src["classifications"] == narrator.CLASSIFICATIONS


def test_narrator_prompt_matches_personality_convos():
    src = narrator.source_constants()
    r = rec(arm="N", buyer_="B-spec", seller="S-concise", budget="50k")
    ours = narrator.conversation_prompt(r)
    theirs = src["get_conversation_prompt"](
        narrator.BACKGROUND_BUYERS[1], narrator.BACKGROUND_SELLERS[2], "50k", "Honda", "Sedan",
        "Leather seats, Backup camera, Bluetooth")
    assert ours == theirs
    assert ours[0] == {"role": "system", "content": ""}
    assert "$50k" in ours[1]["content"]
    assert narrator.BACKGROUND_SELLERS[2] + narrator.BACKGROUND_BUYERS[1] in ours[1]["content"]


def test_strip_quotes_is_the_2023_rule():
    assert narrator.strip_quotes('  "Agent: hi\nBuyer: hello"  ') == "Agent: hi\nBuyer: hello"
    assert narrator.strip_quotes("Agent: hi") == "Agent: hi"
    assert narrator.strip_quotes('"') == ""


# 6. record sampling ------------------------------------------------------------------

def test_records_balanced_and_deterministic():
    a = run.sample_records("A", 2250, 20261109)
    b = run.sample_records("A", 2250, 20261109)
    assert [r.id for r in a] == [r.id for r in b]
    assert len({r.id for r in a}) == 2250
    cells = {}
    for r in a:
        cells.setdefault(r.cell, []).append(r)
    assert set(cells) == {f"{b_}|{s}" for b_, s in run.CELLS} and all(len(v) == 250 for v in cells.values())
    for rows in cells.values():
        combos = {}
        for r in rows:
            combos[(r.multiplier, r.feature_match)] = combos.get((r.multiplier, r.feature_match), 0) + 1
        assert len(combos) == 6 and max(combos.values()) - min(combos.values()) <= 1
    for r in a:
        assert r.list_price == int(round(r.multiplier * r.budget))
        assert 1 <= len(r.features) <= 10 and len(set(r.features)) == len(r.features)
        if r.feature_match == "missing-one":
            assert r.missing_feature in r.features and r.missing_feature not in r.offered_features
            assert len(r.offered_features) == len(r.features) - 1
        else:
            assert r.missing_feature is None and r.offered_features == r.features
    assert run.sample_records("A", 2250, 1)[0].id != a[0].id


def test_pilot_size_spreads_160_over_nine_cells():
    recs = run.sample_records("N", 160, 20261108)
    counts = {}
    for r in recs:
        counts[r.cell] = counts.get(r.cell, 0) + 1
    assert sum(counts.values()) == 160 and max(counts.values()) - min(counts.values()) <= 1
    assert all(r.multiplier is None and r.list_price is None and r.offered_features == []
               for r in recs)
    assert all(r.budget == narrator.BUDGET_DOLLARS[r.budget_label] for r in recs)


# 7. the agent loop with a fake client -------------------------------------------------

class Fake:
    """Replies in order; the last reply repeats."""

    def __init__(self, replies, stop="end_turn"):
        self.replies, self.stop, self.calls = list(replies), stop, []

    def complete(self, system, messages, max_tokens):
        self.calls.append((system, messages, max_tokens))
        i = min(len(self.calls) - 1, len(self.replies) - 1)
        return run.Completion(self.replies[i], "fake-model", self.stop)


def test_agent_buys_at_turn_three():
    r = rec(seller="S-discount")
    fake = Fake(["Tell me more.\nDECISION: CONTINUE", "Still too much.\nDECISION: CONTINUE",
                 "Deal.\nDECISION: BUY $27,000"])
    res = run.run_agent(r, fake)
    assert res["token_outcome"] == "sale" and res["sale_price"] == 27000 and res["buyer_price"] == 27000
    assert res["n_seller_turns"] == 3 and len(fake.calls) == 3
    assert res["seller_ok"] and res["malformed_replies"] == 0 and not res["refusal"]
    assert res["turns"][-1]["close"] and "$27,000" in res["turns"][-1]["text"]
    # the buyer saw its own system prompt and the seller's lines as user turns
    system, messages, _ = fake.calls[-1]
    assert narrator.BACKGROUND_BUYERS[0] in system
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == sellers.seller_line("S-discount", r, 1)
    assert "DECISION" not in res["text"] and res["text"].startswith("Agent: ")
    assert "Buyer: Deal." in res["text"] and res["text"].count("Agent: ") == 4


def test_agent_leaves_at_turn_one_and_sale_price_is_the_sellers():
    r = rec(seller="S-concise")
    res = run.run_agent(r, Fake(["No.\nDECISION: LEAVE"]))
    assert res["token_outcome"] == "no_sale" and res["sale_price"] is None
    assert res["n_seller_turns"] == 1 and res["seller_ok"]
    res = run.run_agent(r, Fake(["Fine.\nDECISION: BUY $20,000"]))
    assert res["sale_price"] == 30000 and res["buyer_price"] == 20000   # seller's price wins


def test_agent_times_out_after_five_and_counts_malformed():
    r = rec()
    fake = Fake(["Hmm.", "Hmm.\nDECISION: CONTINUE"])
    res = run.run_agent(r, fake)
    assert res["token_outcome"] == "timeout" and res["n_seller_turns"] == sellers.MAX_TURNS
    assert len(fake.calls) == 5 and res["malformed_replies"] == 1
    assert res["seller_ok"] and not any(t.get("close") for t in res["turns"])


def test_agent_records_a_refusal():
    res = run.run_agent(rec(), Fake([""], stop="refusal"))
    assert res["token_outcome"] == "refusal" and res["refusal"] and res["n_seller_turns"] == 1


def test_verify_catches_a_tampered_seller_line():
    res = run.run_agent(rec(seller="S-features"), Fake(["Ok.\nDECISION: CONTINUE", "Deal.\nDECISION: BUY $30,000"]))
    assert run.verify_seller_lines(res)
    res["turns"][0]["text"] += " And I can knock 10% off."
    assert not run.verify_seller_lines(res)


def test_narrator_run_strips_quotes_and_uses_2023_prompt():
    r = rec(arm="N")
    fake = Fake(['"Agent: Hello!\n\nBuyer: Hi. 0.5, $30,000"'])
    res = run.run_narrator(r, fake)
    assert res["original_conversation"] == "Agent: Hello!\n\nBuyer: Hi. 0.5, $30,000"
    system, messages, max_tokens = fake.calls[0]
    assert system is None and max_tokens == narrator.MAX_TOKENS_2023
    assert messages == narrator.conversation_prompt(r)[1:]


def test_generate_resumes_by_id(tmp_path):
    recs = run.sample_records("A", 3, 5)
    path = tmp_path / "arm_A.jsonl"
    assert run.generate("A", recs, Fake(["Bye.\nDECISION: LEAVE"]), path) == 3
    assert run.generate("A", recs, Fake(["Bye.\nDECISION: LEAVE"]), path) == 0
    assert len(path.read_text().splitlines()) == 3


# 8. the confirmatory guard -----------------------------------------------------------

def test_confirmatory_refuses_without_tag(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run.main(["--arm", "A", "--phase", "confirmatory", "--out", str(tmp_path), "--dry-run"])
    monkeypatch.setattr(run, "head_has_tag", lambda tag: False)
    with pytest.raises(SystemExit):
        run.main(["--arm", "A", "--phase", "confirmatory", "--out", str(tmp_path),
                  "--tag", "prereg-persona1-x", "--dry-run"])
    monkeypatch.setattr(run, "head_has_tag", lambda tag: tag == "prereg-persona1-x")
    assert run.main(["--arm", "A", "--phase", "confirmatory", "--out", str(tmp_path),
                     "--tag", "prereg-persona1-x", "--dry-run"]) == 0
    assert run.main(["--arm", "N", "--phase", "pilot", "--out", str(tmp_path), "--dry-run"]) == 0
    assert not (Path(tmp_path) / "arm_N.jsonl").exists()
