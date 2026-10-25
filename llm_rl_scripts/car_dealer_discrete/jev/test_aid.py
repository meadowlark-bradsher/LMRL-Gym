"""Offline checks for the acceptance aid and its wiring into the Jev policy.

Pure stdlib, no API. Runnable directly or under pytest.
"""

from __future__ import annotations

import inspect
from fractions import Fraction

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, Deal
from llm_rl_scripts.car_dealer_discrete.jev.aid import (
    AcceptanceAid,
    AidConfig,
    entropy_bits,
    ranking_key,
)
from llm_rl_scripts.car_dealer_discrete.jev.policy import JevNegotiationPolicy
from llm_rl_scripts.car_dealer_discrete.protocol import (
    ACCEPT,
    OFFER,
    REJECT,
    WALK,
    Action,
    Observation,
    TurnRecord,
)

EASY = Fraction(2, 5)
HARD = Fraction(11, 20)
BUYER_RANK = ("price", "warranty", "delivery")


def _obs(role, history=(), outstanding=None, used=None, r=EASY, legal=None):
    used = len([t for t in history if t.action.kind == OFFER]) if used is None else used
    if legal is None:
        legal = []
        if outstanding is not None:
            legal += [Action(ACCEPT), Action(REJECT)]
        legal += [Action(OFFER, d.id) for d in __import__(
            "llm_rl_scripts.car_dealer_discrete.game", fromlist=["ALL_DEALS"]).ALL_DEALS]
        legal.append(Action(WALK))
    return Observation(
        role=role, own_ranking=BUYER_RANK, reservation=r, max_proposals=6,
        proposals_used=used, outstanding_offer=outstanding, history=tuple(history),
        legal_actions=tuple(legal),
    )


def test_uniform_prior_is_a_distribution():
    q = AcceptanceAid().posterior(_obs("buyer"))
    assert len(q) == 6 and set(q) == set(ALL_RANKINGS)
    assert abs(sum(q.values()) - 1.0) < 1e-12
    assert all(abs(p - 1 / 6) < 1e-12 for p in q.values())
    assert abs(entropy_bits(q) - 2.584962500721156) < 1e-9  # log2(6)


def test_uniform_prior_already_separates_ideal_from_compromise():
    aid = AcceptanceAid(AidConfig(noise=0.05))
    _, pacc = aid.acceptance_probabilities(_obs("buyer"))
    # Buyer's ideal is worth 0 to every seller ranking -> never IR -> noise floor.
    assert abs(pacc["P0-W2-D2"] - 0.05) < 1e-12
    # The universal 0.5 compromise is IR for every ranking in the easy regime.
    assert abs(pacc["P1-W1-D1"] - 0.95) < 1e-12
    # ...but not in the hard regime (0.5 < 0.55).
    _, pacc_hard = aid.acceptance_probabilities(_obs("buyer", r=HARD))
    assert abs(pacc_hard["P1-W1-D1"] - 0.05) < 1e-12
    assert all(0.0 <= p <= 1.0 for p in pacc.values()) and len(pacc) == 27


def test_ideal_offer_carries_no_information():
    # Seller's ideal P2-W0-D0 has utility 1.0 under every seller ranking, so
    # observing it must leave the posterior exactly uniform.
    hist = [TurnRecord("buyer", Action(OFFER, "P0-W2-D2")),
            TurnRecord("seller", Action(OFFER, "P2-W0-D0"))]
    # NB: the seller's counter also non-accepts our P0-W2-D2, which is non-IR for
    # every type -> likelihood 1-eps for all -> also uninformative. Net: uniform.
    q = AcceptanceAid().posterior(_obs("buyer", hist, outstanding="P2-W0-D0"))
    assert all(abs(p - 1 / 6) < 1e-9 for p in q.values())


def test_concession_reveals_priority():
    # Seller concedes on PRICE (P1 instead of P2). Giving up price is cheap for
    # a seller who ranks price LAST, expensive for one who ranks it FIRST, so
    # the posterior must shift toward price-last rankings.
    hist = [TurnRecord("buyer", Action(OFFER, "P0-W2-D2")),
            TurnRecord("seller", Action(OFFER, "P1-W0-D0"))]
    q = AcceptanceAid().posterior(_obs("buyer", hist, outstanding="P1-W0-D0"))
    price_last = sum(p for th, p in q.items() if th[2] == "price")
    price_first = sum(p for th, p in q.items() if th[0] == "price")
    assert price_last > price_first
    assert entropy_bits(q) < 2.584962500721156  # strictly informative


def test_rejection_softly_downweights_types_that_would_accept():
    # We (buyer) offer P1-W1-D1 (IR for all seller types in easy) and the seller
    # REJECTs. Under the threshold model every type "should" have accepted, so
    # all are down-weighted equally -> posterior stays uniform (no type singled
    # out), and crucially nothing collapses to zero thanks to the noise floor.
    hist = [TurnRecord("buyer", Action(OFFER, "P1-W1-D1")),
            TurnRecord("seller", Action(REJECT))]
    q = AcceptanceAid().posterior(_obs("buyer", hist))
    assert all(p > 0.0 for p in q.values())
    assert all(abs(p - 1 / 6) < 1e-9 for p in q.values())
    # Now a deal IR for only SOME types: rejection should favour the others.
    # P1-W0-D0 is IR for a seller only if price is not its top priority
    # (top=price -> 8.5/13=0.65 >= .55? yes in hard; use easy: all >= .4). Use hard.
    hist2 = [TurnRecord("buyer", Action(OFFER, "P1-W0-D0")),
             TurnRecord("seller", Action(REJECT))]
    q2 = AcceptanceAid().posterior(_obs("buyer", hist2, r=HARD))
    assert all(p > 0.0 for p in q2.values())
    assert abs(sum(q2.values()) - 1.0) < 1e-12


def test_aid_api_is_leakage_safe():
    # The aid can only ever see an Observation; there is no parameter through
    # which the opponent's true ranking could be passed.
    sig = inspect.signature(AcceptanceAid.posterior)
    assert list(sig.parameters) == ["self", "obs"]
    assert "opponent" not in " ".join(Observation.__dataclass_fields__)


def test_policy_annotates_offers_and_logs_belief():
    aid = AcceptanceAid()
    pol = JevNegotiationPolicy(JevClient(MockJevBackend()), aid=aid)
    obs = _obs("buyer")
    options, back = pol.build_options(obs)
    offers = {k: v for k, v in options.items() if k.startswith("OFFER ")}
    assert len(offers) == 27
    assert all("chance the other party accepts" in v for v in offers.values())
    # Jev is NOT handed the posterior or any expected value.
    joined = " ".join(options.values()) + pol.build_state(obs) + pol.build_instructions(obs)
    assert "expected value" not in joined.lower()
    assert ">" not in pol.build_state(obs)  # no ranking strings leaked into state
    # The policy exposes the belief for the evaluator/recorder.
    assert pol.last_belief is not None and abs(sum(pol.last_belief.values()) - 1) < 1e-12
    assert pol.last_acceptance is not None and len(pol.last_acceptance) == 27
    assert "estimate of how likely" in pol.build_instructions(obs)


def test_aid_and_salience_compose():
    pol = JevNegotiationPolicy(JevClient(MockJevBackend()), aid=AcceptanceAid(), salience=True)
    hist = [TurnRecord("buyer", Action(OFFER, "P0-W2-D2")),
            TurnRecord("seller", Action(OFFER, "P1-W0-D0"))]
    obs = _obs("buyer", hist, outstanding="P1-W0-D0")
    state = pol.build_state(obs)
    assert "no-deal payoff" in state           # salience block present
    options, _ = pol.build_options(obs)
    assert any("chance the other party accepts" in (v or "") for v in options.values())
    action = pol(obs)                           # end-to-end with the mock backend
    assert action in obs.legal_actions


def test_ranking_key_roundtrip():
    for th in ALL_RANKINGS:
        assert tuple(ranking_key(th).split(">")) == th


def test_prior_only_never_updates():
    # Feed the concession history that moves the bayes posterior; prior_only
    # must stay exactly uniform and emit the bayes aid's t=0 P(accept) values.
    hist = [TurnRecord("buyer", Action(OFFER, "P0-W2-D2")),
            TurnRecord("seller", Action(OFFER, "P1-W0-D0"))]
    obs = _obs("buyer", hist, outstanding="P1-W0-D0")
    prior_aid = AcceptanceAid(AidConfig(mode="prior_only"))
    q = prior_aid.posterior(obs)
    assert all(abs(p - 1 / 6) < 1e-12 for p in q.values())
    _, p_prior = prior_aid.acceptance_probabilities(obs)
    _, p_bayes0 = AcceptanceAid().acceptance_probabilities(_obs("buyer"))
    assert all(abs(p_prior[k] - p_bayes0[k]) < 1e-12 for k in p_prior)
    # ...while the real filter HAS moved on this history.
    _, p_bayes = AcceptanceAid().acceptance_probabilities(obs)
    assert any(abs(p_bayes[k] - p_prior[k]) > 1e-6 for k in p_prior)


def test_placebo_is_constant_and_prompt_identical_to_real_aid():
    import re
    obs = _obs("buyer")
    plc = AcceptanceAid(AidConfig(mode="placebo", placebo_value=0.5))
    _, pacc = plc.acceptance_probabilities(obs)
    assert len(pacc) == 27 and all(p == 0.5 for p in pacc.values())
    pol_b = JevNegotiationPolicy(JevClient(MockJevBackend()), aid=AcceptanceAid())
    pol_p = JevNegotiationPolicy(JevClient(MockJevBackend()), aid=plc)
    # The control differs from the real aid ONLY in the emitted numbers.
    assert pol_b.build_instructions(obs) == pol_p.build_instructions(obs)
    assert pol_b.build_state(obs) == pol_p.build_state(obs)
    ob, _ = pol_b.build_options(obs)
    op, _ = pol_p.build_options(obs)
    assert list(ob) == list(op)
    strip = lambda t: re.sub(r"accepts: \d\.\d\d", "accepts: X", t or "")
    assert all(strip(ob[k]) == strip(op[k]) for k in ob)
    assert all("accepts: 0.50" in (op[k] or "") for k in op if k.startswith("OFFER "))
    # Logged belief under placebo is the uniform prior (zero learning, by design).
    assert all(abs(v - 1 / 6) < 1e-12 for v in pol_p.last_belief.values())


def test_aid_config_rejects_bad_mode():
    try:
        AidConfig(mode="oracle")
    except ValueError:
        pass
    else:
        raise AssertionError("bad mode accepted")


def _run_all():
    import sys
    mod = sys.modules[__name__]
    tests = [f for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    fails = 0
    for fn in tests:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
