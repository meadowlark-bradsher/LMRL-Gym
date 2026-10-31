"""CD-SEL-1 gates: the shown-information treatments are what the registration says they
are, the plain arm's prompt is unchanged, and the runner / analyzer / verdict pipeline
runs end to end on the new arms (mock backend, no API)."""

import gzip
import json
import os
import random
import tempfile
from fractions import Fraction

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from . import belief_live as bl
from .analyze_phase1 import SELECTOR_PAIRS, main as analyze_main
from .jev_policy_live import ACCEPT_ANNOTATIONS, PLACEBO_P_ACCEPT, JevLivePolicy
from .opponent import KernelParams
from .protocol_live import ASK, PROPOSE, LiveConfig, apply, new_game
from .run_phase1 import JEV_ARMS, NOASK_ARMS, Cell, main as run_main, shown_p_accept
from .verdicts_sel import REQUIRED_ARMS, main as verdicts_main

CFG = LiveConfig(horizon=3)
PARAMS = KernelParams(alpha0=Fraction(4, 5), beta=8.0, p_counter=1.0, p_hold=1.0, shortlist=3)


def _client():
    return JevClient(MockJevBackend())


def test_belief_text_states_the_support_with_probabilities_and_the_ruled_out_count():
    pol = JevLivePolicy(_client(), CFG, "buyer", ALL_RANKINGS[0], show_belief=True)
    txt = pol.belief_text(bl.uniform())
    assert txt.startswith("Nothing the seller has said or done so far distinguishes its 6 possible priority rankings") and "0.17" in txt
    b = (0.0, 0.0, 0.75, 0.25, 0.0, 0.0)
    txt = pol.belief_text(b)
    assert "warranty > price > delivery (0.75); warranty > delivery > price (0.25)" in txt
    assert "The other 4 rankings are ruled out." in txt
    assert "price > warranty > delivery" not in txt
    txt1 = pol.belief_text((0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    assert "price > delivery > warranty (1.00)" in txt1 and "The other 5 rankings are ruled out." in txt1
    # the state text carries it only when the treatment is on
    st = new_game(CFG, "buyer")
    assert pol.belief_text(b) in pol.state_text(st, b)
    plain = JevLivePolicy(_client(), CFG, "buyer", ALL_RANKINGS[0])
    assert "imply" not in plain.state_text(st) and "distinguishes" not in plain.state_text(st)


def test_annotations_follow_the_arm_spec_and_the_plain_prompt_is_unchanged():
    cell = Cell(CFG, PARAMS, "seller", ALL_RANKINGS[2], "shortlist")
    st = new_game(CFG, "seller")
    belief = bl.uniform()
    rows = cell.menu.rows(st, belief, include_questions=True)
    offer = next(r for r in rows if r.action.kind == PROPOSE)
    plain = JevLivePolicy(_client(), CFG, "seller", ALL_RANKINGS[2])
    base_instr, base_desc = plain.instructions(), plain.describe(offer)
    assert "accepts with probability" not in base_desc and "imply" not in base_instr and "estimated probability" not in base_instr
    for arm, spec in JEV_ARMS.items():
        pol = JevLivePolicy(_client(), CFG, "seller", ALL_RANKINGS[2], show_eig=spec.get("show_eig", False),
                            show_belief=spec.get("show_belief", False), accept_annotation=spec.get("accept_annotation"))
        instr = pol.instructions()
        assert ("imply about its priority ranking" in instr) == bool(spec.get("show_belief"))
        assert ("estimated probability that the other party accepts" in instr) == (spec.get("accept_annotation") is not None)
        shown = shown_p_accept(cell, st, rows, belief, spec.get("accept_annotation"))
        if spec.get("accept_annotation") is None:
            assert shown is None
            if not spec.get("show_eig"):
                assert pol.describe(offer) == base_desc
            else:
                assert pol.describe(offer).startswith(base_desc) and "bits." in pol.describe(offer)
        else:
            d = pol.describe(offer, shown[offer.key])
            assert f"the buyer accepts with probability {shown[offer.key]:.2f}." in d
            assert d.startswith(base_desc.split(" This uses one opportunity.")[0])
    # the arm table is consistent with the policy's vocabulary and the no-ask set
    assert all(spec.get("accept_annotation") in ACCEPT_ANNOTATIONS for spec in JEV_ARMS.values())
    assert NOASK_ARMS == {"exact_noask", "jev_noask_plain", "jev_noask_eig", "jev_noask_both"}
    assert set(REQUIRED_ARMS) - {"exact_ask", "exact_noask", "random"} <= set(JEV_ARMS)


def test_shown_acceptance_numbers_are_posterior_prior_or_placebo():
    cell = Cell(CFG, PARAMS, "buyer", ALL_RANKINGS[0], "shortlist")
    rng = random.Random(3)
    st = new_game(CFG, "seller")
    belief = bl.uniform()
    # the seller opens; then take a non-uniform belief (the kernel's opening move need not be informative)
    act = cell.kernel.sample(st, ALL_RANKINGS[4], rng)
    st = apply(st, act)
    belief = (0.5, 0.3, 0.2, 0.0, 0.0, 0.0)
    rows = cell.menu.rows(st, belief, include_questions=True)
    offers = [r for r in rows if r.action.kind == PROPOSE]
    assert offers
    post = shown_p_accept(cell, st, rows, belief, "posterior")
    prior = shown_p_accept(cell, st, rows, belief, "prior")
    plac = shown_p_accept(cell, st, rows, belief, "placebo")
    assert set(post) == set(prior) == set(plac) == {r.key for r in offers}
    for r in offers:
        assert abs(post[r.key] - r.p_accept) < 1e-12
        assert abs(prior[r.key] - cell.menu.p_accept(st, bl.uniform(), r.action.deal_id)) < 1e-12
        assert plac[r.key] == PLACEBO_P_ACCEPT
    assert any(abs(post[k] - prior[k]) > 1e-9 for k in post), "posterior and prior P(accept) coincide after an informative move"


def test_runner_analyzer_and_verdicts_run_end_to_end_on_the_selector_arms():
    arms = "exact_ask,exact_noask,random,jev_ask_plain,jev_ask_belief,jev_ask_pacc,jev_ask_prior,jev_ask_placebo,jev_ask_both,jev_noask_both"
    with tempfile.TemporaryDirectory() as d:
        run_main(["--horizon", "3", "--types", "1", "--opp-types", "0,4", "--roles", "seller", "--openers", "buyer,seller",
                  "--repeats", "2", "--arms", arms, "--skip-ceilings", "--out", d])
        recs = [json.loads(l) for l in gzip.open(os.path.join(d, "runs.jsonl.gz"), "rt")]
        summ = json.load(open(os.path.join(d, "summary.json")))
        assert summ["error_stubs"] == 0 and len(recs) == 10 * 2 * 2 * 2
        assert set(summ["jev_arm_specs"]) == set(arms.split(",")) - {"exact_ask", "exact_noask", "random"}
        assert summ["placebo_p_accept"] == PLACEBO_P_ACCEPT
        for r in recs:
            assert "error" not in r
            spec = JEV_ARMS.get(r["arm"])
            for dd in r["decisions"]:
                if r["arm"] in NOASK_ARMS:
                    assert dd["chosen_kind"] != ASK and not any(m["kind"] == ASK for m in dd["menu"])
                if spec is None:
                    assert dd["jev"] is None
                    continue
                j = dd["jev"]
                assert j["well_formed"]
                assert j["shown"]["belief"] == spec.get("show_belief", False)
                assert j["shown"]["accept_annotation"] == spec.get("accept_annotation")
                offers = {m["key"]: m for m in dd["menu"] if m["kind"] == PROPOSE}
                if spec.get("accept_annotation") is None:
                    assert j["shown"]["p_accept_shown"] is None
                else:
                    assert set(j["shown"]["p_accept_shown"]) == set(offers)
                    if spec["accept_annotation"] == "posterior":
                        assert all(abs(j["shown"]["p_accept_shown"][k] - offers[k]["p_accept"]) < 1e-12 for k in offers)
                    if spec["accept_annotation"] == "placebo":
                        assert all(p == PLACEBO_P_ACCEPT for p in j["shown"]["p_accept_shown"].values())
        analyze_main(["--dir", d, "--B", "200", "--seed", "1"])
        a = json.load(open(os.path.join(d, "analysis.json")))
        for x, z in SELECTOR_PAIRS:
            assert f"{x}-{z}" in a["contrasts"] and f"{x}-{z}" in a["contrasts_walk"]
        assert "jev_ask_both-jev_noask_both" in a["contrasts"] and "jev_ask_both-exact_ask" in a["contrasts"]
        assert a["jev_arm_specs"] == summ["jev_arm_specs"]
        verdicts_main(["--analysis", os.path.join(d, "analysis.json")])
        v = json.load(open(os.path.join(d, "verdicts.json")))
        for k in ("P1", "S1", "S2", "S3", "S4", "S5", "S5b", "S6", "S7", "S8", "S8b", "S9_jev_ask_belief", "S9_jev_ask_both"):
            assert k in v["rules"], k
        assert v["gates"]["pipeline"]["missing_arms"] == [] and v["gates"]["pipeline"]["error_stubs"] == 0
