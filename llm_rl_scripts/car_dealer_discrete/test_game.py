"""Sanity checks for the discretized Car Dealer engine (spec section 6.3).

Pure stdlib; runnable directly (``python test_game.py``) or under pytest.
The headline checks reproduce the spec's exact-v0 feasibility counts:
36/36 (easy, r=0.40) and 30/36 (hard, r=0.55).
"""

from __future__ import annotations

import random
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import (
    ALL_DEALS,
    ALL_RANKINGS,
    ATTRIBUTES,
    LEVELS,
    Deal,
    all_type_pairs,
    buyer_satisfaction,
    satisfaction,
    utility,
)
from llm_rl_scripts.car_dealer_discrete.protocol import (
    ACCEPT,
    NegotiationGame,
    Observation,
    random_policy,
)
from llm_rl_scripts.car_dealer_discrete.scoring import (
    EASY_RESERVATION,
    HARD_RESERVATION,
    analyze_type_pair,
    feasibility_counts,
    score_episode,
)


def test_catalog_cardinalities():
    assert len(ALL_DEALS) == 27
    assert len({d.id for d in ALL_DEALS}) == 27
    assert len(ALL_RANKINGS) == 6
    assert len(all_type_pairs()) == 36


def test_deal_id_roundtrip():
    for d in ALL_DEALS:
        assert Deal.from_id(d.id) == d
    # Canonical worked-example id from the spec.
    assert Deal.from_id("P1-W2-D0") == Deal(price=1, warranty=2, delivery=0)


def test_worked_example_utilities():
    # buyer: price > warranty > delivery ; seller: delivery > price > warranty
    buyer_rank = ("price", "warranty", "delivery")
    seller_rank = ("delivery", "price", "warranty")
    d = Deal.from_id("P1-W2-D0")  # medium price, long warranty, slow delivery
    assert utility("buyer", buyer_rank, d) == Fraction(15, 26)   # 7.5/13
    assert utility("seller", seller_rank, d) == Fraction(21, 26)  # 10.5/13
    # Both pass even the hard reservation.
    assert utility("buyer", buyer_rank, d) >= HARD_RESERVATION
    assert utility("seller", seller_rank, d) >= HARD_RESERVATION


def test_utilities_in_unit_interval():
    for rank in ALL_RANKINGS:
        for d in ALL_DEALS:
            for role in ("buyer", "seller"):
                u = utility(role, rank, d)
                assert Fraction(0) <= u <= Fraction(1)


def test_satisfaction_complementarity():
    # Seller satisfaction is the exact complement of the buyer's (section 6.3).
    for attr in ATTRIBUTES:
        for level in LEVELS:
            assert satisfaction("seller", attr, level) == 1 - buyer_satisfaction(attr, level)


def test_feasibility_counts_match_spec():
    assert feasibility_counts(EASY_RESERVATION) == (36, 36)
    assert feasibility_counts(HARD_RESERVATION) == (30, 36)


def test_pareto_and_nash_well_formed():
    for buyer_rank, seller_rank in all_type_pairs():
        a = analyze_type_pair(buyer_rank, seller_rank, HARD_RESERVATION)
        if not a.is_feasible:
            assert a.pareto_frontier == () and a.buyer_u_star is None
            continue
        assert len(a.pareto_frontier) >= 1
        assert set(a.pareto_frontier).issubset(set(a.feasible))
        # U* is attained inside F.
        assert a.buyer_u_star == max(utility("buyer", buyer_rank, d) for d in a.feasible)
        assert a.seller_u_star == max(utility("seller", seller_rank, d) for d in a.feasible)
        assert a.nash_max is not None and a.nash_max >= 0
        # The best buyer and best seller deals are both on the frontier.
        best_b = max(a.feasible, key=lambda d: utility("buyer", buyer_rank, d))
        best_s = max(a.feasible, key=lambda d: utility("seller", seller_rank, d))
        assert best_b in set(a.pareto_frontier)
        assert best_s in set(a.pareto_frontier)


def test_observation_hides_opponent_ranking():
    # Structural leakage check: the observation dataclass exposes only the
    # policy's own ranking, never a field naming the opponent's type.
    fields = set(Observation.__dataclass_fields__)
    assert "own_ranking" in fields
    assert not any("opponent" in f or "other" in f for f in fields)


def test_random_selfplay_invariants():
    # Many random self-play episodes must terminate and never produce an
    # invalid (non-IR) agreement; no-deal payoff equals the outside option.
    for seed in range(200):
        rng = random.Random(seed)
        b, s = rng.choice(ALL_RANKINGS), rng.choice(ALL_RANKINGS)
        r = rng.choice([EASY_RESERVATION, HARD_RESERVATION])
        start = rng.choice(["buyer", "seller"])
        game = NegotiationGame(
            b, s, r, random_policy(rng), random_policy(rng), starting_side=start
        )
        result = game.play()
        assert result.proposals_used <= game.max_proposals
        if result.agreement:
            assert result.individually_rational is True
            d = Deal.from_id(result.accepted_deal_id)
            assert utility("buyer", b, d) >= r and utility("seller", s, d) >= r
            assert result.outcome_type == "agreement"
        else:
            assert result.buyer_utility == r and result.seller_utility == r
        # Scoring is consistent with the truth table.
        analysis = analyze_type_pair(b, s, r)
        score = score_episode(result, analysis)
        if score.valid_agreement:
            assert score.feasible  # a valid IR agreement implies F non-empty
            assert score.on_pareto_frontier in (True, False)
        if not score.feasible:
            assert score.buyer_gain is None and score.seller_gain is None


def test_invalid_acceptance_is_not_a_deal():
    # An agent that accepts a non-IR deal gets no agreement and the no-deal payoff.
    hard = HARD_RESERVATION
    buyer_rank = ("price", "warranty", "delivery")
    seller_rank = ("price", "warranty", "delivery")
    # Find a deal that is NOT IR for both under the hard regime.
    non_ir = next(
        d
        for d in ALL_DEALS
        if not (
            utility("buyer", buyer_rank, d) >= hard
            and utility("seller", seller_rank, d) >= hard
        )
    )

    from llm_rl_scripts.car_dealer_discrete.protocol import Action, OFFER

    def scripted_buyer(obs: Observation) -> Action:
        return Action(OFFER, non_ir.id)

    def eager_seller(obs: Observation) -> Action:
        if obs.outstanding_offer is not None:
            return Action(ACCEPT)
        return Action("WALK")

    game = NegotiationGame(
        buyer_rank, seller_rank, hard, scripted_buyer, eager_seller, starting_side="buyer"
    )
    result = game.play()
    assert result.outcome_type == "invalid_acceptance"
    assert result.agreement is False
    assert result.buyer_utility == hard and result.seller_utility == hard


def _run_all():
    import inspect
    import sys

    module = sys.modules[__name__]
    tests = [
        fn for name, fn in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
