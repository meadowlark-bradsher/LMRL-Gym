"""Phase 0 gates for live-car-dealer-v0.1 (design document section 6, Phase 0).

  1. the kernel's action distribution sums to 1 at every reachable opponent node
  2. terminal rules: accepted deals are IR for the opponent (it never accepts
     below r); deadline gives the non-proposer the final decision; HOLD cannot loop
  3. the focal belief never updates from the focal's own actions, stays
     normalised, and shrinks support only through opponent evidence
  4. oracle value == exhaustive policy evaluation of the oracle policy (no max,
     full expectation) to < 1e-9, and Monte Carlo of that policy within 4 SE
  5. V_ask >= V_noask at every root; ASK-forbidden oracle never asks
"""

from __future__ import annotations

import random
import statistics
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS, DEALS_BY_ID, utility

from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.oracle import Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import (
    ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, Action, LiveConfig, apply, legal_actions, new_game, other,
)
from llm_rl_scripts.car_dealer_live.simulate import run_episode

CFG = LiveConfig(horizon=3)


def _walk_tree(cfg, kernel, focal, max_nodes=20000):
    """Enumerate reachable public states under ALL focal actions and ALL positive-prob
    opponent actions (answers both ways). Yields (state, is_opponent_node)."""
    seen = set()
    stack = [new_game(cfg, "buyer"), new_game(cfg, "seller")]
    while stack and len(seen) < max_nodes:
        s = stack.pop()
        if s.terminal is not None or s.key() in seen:
            continue
        seen.add(s.key())
        yield s
        if s.turn == focal:
            for a in legal_actions(s):
                if a.kind == ASK:
                    stack += [apply(s, a, "yes"), apply(s, a, "no")]
                else:
                    stack.append(apply(s, a))
        else:
            acts = set()
            for th in ALL_RANKINGS:
                acts.update(kernel.distribution(s, th))
            stack += [apply(s, a) for a in acts]


def test_kernel_sums_to_one_and_is_legal():
    for params in (KernelParams(), KernelParams(p_counter=0.5, p_hold=0.5, accept_noise=0.1, shortlist=0)):
        k = T1Kernel("seller", CFG, params)
        n = 0
        for s in _walk_tree(CFG, k, "buyer", max_nodes=3000):
            if s.turn != "seller":
                continue
            legal = set(legal_actions(s))
            for th in ALL_RANKINGS:
                d = k.distribution(s, th)
                assert abs(sum(d.values()) - 1.0) < 1e-9
                assert set(d) <= legal, (s.key(), set(d) - legal)
                assert all(p > 0 for p in d.values())
                n += 1
        assert n > 50


def test_opponent_never_accepts_below_reservation_and_deadline_rule():
    k = T1Kernel("seller", CFG)
    for s in _walk_tree(CFG, k, "buyer", max_nodes=3000):
        if s.turn != "seller" or s.standing is None or s.standing.proposer != "buyer":
            continue
        for th in ALL_RANKINGS:
            d = k.distribution(s, th)
            if Action(ACCEPT) in d:
                assert utility("seller", th, DEALS_BY_ID[s.standing.deal_id]) >= CFG.reservation
            if s.at_deadline:
                assert set(d) <= {Action(ACCEPT), Action(WALK)}
    # deadline: after the last opportunity the non-proposer decides
    s = new_game(LiveConfig(horizon=1), "buyer")
    s = apply(s, Action(PROPOSE, deal_id=ALL_DEALS[0].id))
    assert s.at_deadline and s.turn == "seller" and set(legal_actions(s)) == {Action(ACCEPT), Action(WALK)}
    # ASK as the last opportunity with the OTHER side's offer standing -> asker decides
    s = new_game(LiveConfig(horizon=2), "seller")
    s = apply(s, Action(PROPOSE, deal_id=ALL_DEALS[0].id))         # seller proposes, buyer's move
    s = apply(s, Action(ASK, question_id="top:price"), "yes")       # buyer asks, budget -> 0
    assert s.at_deadline and s.turn == "buyer" and set(legal_actions(s)) == {Action(ACCEPT), Action(WALK)}


def test_hold_only_for_own_offer_and_cannot_loop():
    s = new_game(CFG, "buyer")
    s = apply(s, Action(PROPOSE, deal_id=ALL_DEALS[5].id))   # buyer proposes
    assert Action(HOLD) not in legal_actions(s)              # seller faces buyer's offer: no hold
    s = apply(s, Action(ASK, question_id="top:price"), "no")  # seller asks instead
    assert s.turn == "buyer" and Action(HOLD) in legal_actions(s)
    s = apply(s, Action(HOLD))
    assert s.turn == "seller" and Action(HOLD) not in legal_actions(s)


def test_belief_updates_only_from_opponent_evidence():
    k = T1Kernel("seller", CFG)
    b = bl.uniform()
    s = new_game(CFG, "buyer")
    # focal PROPOSE: no update by construction (the simulator never touches belief here)
    s2 = apply(s, Action(PROPOSE, deal_id=ALL_DEALS[0].id))
    # opponent response: likelihood over types
    liks = bl.action_likelihoods(k, s2)
    for a, (p, post) in bl.branches(b, liks).items():
        assert abs(sum(post) - 1.0) < 1e-12 and 0 < p <= 1
    # an answer shrinks support exactly under zero noise
    ans = bl.answer_likelihoods(k, "above:price>warranty")
    post = bl.posterior(b, ans["yes"])
    assert len(bl.support(post)) == 3 and abs(sum(post) - 1.0) < 1e-12


def _policy_eval(oracle, state, belief):
    """Exhaustive expectation of the oracle's greedy policy (no max anywhere)."""
    if state.terminal is not None:
        return oracle.terminal_gain(state)
    if state.turn == oracle.focal:
        a, _ = oracle.best(state, belief)
        if a.kind == ASK:
            liks = bl.answer_likelihoods(oracle.kernel, a.question_id)
            return sum(p * _policy_eval(oracle, apply(state, a, ans), post) for ans, (p, post) in bl.branches(belief, liks).items()) - float(oracle.cfg.ask_cost)
        return _policy_eval(oracle, apply(state, a), belief)
    liks = bl.action_likelihoods(oracle.kernel, state)
    return sum(p * _policy_eval(oracle, apply(state, act), post) for act, (p, post) in bl.branches(belief, liks).items())


def test_oracle_equals_exhaustive_policy_evaluation_and_monte_carlo():
    cfg = LiveConfig(horizon=3, ask_cost=Fraction(1, 100))
    for focal in ("buyer", "seller"):
        k = T1Kernel(other(focal), cfg, KernelParams(p_counter=0.7, p_hold=0.6))
        o = Oracle(cfg, focal, ALL_RANKINGS[2], k)
        for opener in ("buyer", "seller"):
            root = new_game(cfg, opener)
            v = o.value(root, bl.uniform())
            pe = _policy_eval(o, root, bl.uniform())
            assert abs(v - pe) < 1e-9, (v, pe)
            # Monte Carlo over a uniform opponent type
            rng = random.Random(7)
            pol = lambda s, b: o.best(s, b)[0]
            vals = [run_episode(cfg, focal, ALL_RANKINGS[2], opp, k, pol, opener, rng).focal_net_gain
                    for opp in ALL_RANKINGS for _ in range(150)]
            m = statistics.mean(vals)
            se = statistics.pstdev(vals) / len(vals) ** 0.5
            assert abs(m - v) <= 4 * se + 1e-9, (m, v, se)


def test_ask_value_nonnegative_and_forbidden_oracle_never_asks():
    cfg = LiveConfig(horizon=3)
    k = T1Kernel("seller", cfg)
    for own in ALL_RANKINGS[:3]:
        oa = Oracle(cfg, "buyer", own, k, ask_allowed=True)
        on = Oracle(cfg, "buyer", own, k, ask_allowed=False)
        for opener in ("buyer", "seller"):
            assert oa.root_value(opener) >= on.root_value(opener) - 1e-12
        assert on.ask_legal_nodes == 0


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("ok  ", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
