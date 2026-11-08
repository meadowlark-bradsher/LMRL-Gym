"""Synthetic-recovery gates for ``kernel_fit`` (NEG-7): records replay exactly; the likelihood of the generating
parameters beats a wrong kernel; the fit recovers planted parameters from logged moves; the held-out gain is
large when the opponent is not T1 and small when it is. Offline; the focal is a random legal mover so that
every branch of the kernel (accept / counter / reject / hold / concede) is exercised."""

from __future__ import annotations

import random
from dataclasses import replace
from fractions import Fraction

import pytest

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, LiveConfig, apply, legal_actions, new_game, other, truthful_answer

from . import kernel_fit as kf

CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))
PLANTED = replace(REFERENCE, alpha0=Fraction(3, 5), beta=3.0, p_counter=0.6, p_hold=0.4, accept_noise=0.15)


def simulate(params: KernelParams, n_per_cell: int, seed: int, brief: str = "x") -> list:
    """Random-legal focal vs a T1 opponent with ``params``; both roles, both openers, all type pairs."""
    rng = random.Random(seed)
    recs = []
    for opp_role in ("buyer", "seller"):
        focal = other(opp_role)
        kernel = T1Kernel(opp_role, CFG, params)
        for opener in ("buyer", "seller"):
            for fr in ALL_RANKINGS:
                for orank in ALL_RANKINGS:
                    for rep in range(n_per_cell):
                        ep = f"{opp_role}|{opener}|{'>'.join(fr)}|{'>'.join(orank)}|{rep}"
                        state = new_game(CFG, opener)
                        while state.terminal is None:
                            if state.turn == focal:
                                act = rng.choice(legal_actions(state, focal))
                                ans = truthful_answer(act.question_id, orank) if act.kind == ASK else None
                                state = apply(state, act, ans)
                            else:
                                act = kernel.sample(state, orank, rng)
                                recs.append(kf.record_move(ep, state, opp_role, orank, act, brief))
                                ans = truthful_answer(act.question_id, fr) if act.kind == ASK else None
                                state = apply(state, act, ans)
    return recs


@pytest.fixture(scope="module")
def planted_records():
    return simulate(PLANTED, 1, seed=11)


def test_records_replay_to_the_state_they_were_taken_in(planted_records):
    prep = kf.prepare(planted_records[:50], CFG)
    for r, p in zip(planted_records[:50], prep):
        assert p.state.turn == r["role"] and p.state.terminal is None and p.action.key == r["action"]
        assert p.action in legal_actions(p.state, r["role"])


def test_generating_parameters_beat_the_reference_in_likelihood(planted_records):
    prep = kf.prepare(planted_records, CFG)
    ll_true = kf.mean_ll(prep, PLANTED, CFG)
    ll_ref = kf.mean_ll(prep, REFERENCE, CFG)
    assert ll_true > ll_ref + 0.1, (ll_true, ll_ref)
    _, zeros = kf.log_likelihood(prep, PLANTED, CFG)
    assert zeros == 0


def test_fit_recovers_planted_parameters(planted_records):
    prep = kf.prepare(planted_records, CFG)
    fitted, ll = kf.fit_best(prep, CFG)
    assert ll >= kf.mean_ll(prep, PLANTED, CFG) - 1e-6          # at least as likely as the truth
    assert abs(float(fitted.alpha0) - 0.6) < 0.08, fitted
    assert abs(fitted.beta - 3.0) < 2.0, fitted
    assert abs(fitted.p_counter - 0.6) < 0.15, fitted
    assert abs(fitted.p_hold - 0.4) < 0.2, fitted
    assert abs(fitted.accept_noise - 0.15) < 0.08, fitted
    assert fitted.shortlist == REFERENCE.shortlist


def test_heldout_gain_large_for_a_non_t1_opponent(planted_records):
    rep = kf.heldout_report(planted_records, CFG, folds=2, seed=3)
    assert rep["adopt_fit"] and rep["heldout_gain_nats_per_move"] > kf.GATE_GAIN_NATS
    # an episode contributes only if the opponent moved at all (a random focal can open and walk at once)
    assert 130 <= rep["n_episodes"] <= 2 * 2 * 36
    assert all(f["zero_prob_fitted"] <= f["zero_prob_reference"] for f in rep["folds"])


def test_heldout_gain_small_when_the_opponent_is_t1():
    recs = simulate(REFERENCE, 1, seed=5)
    rep = kf.heldout_report(recs, CFG, folds=2, seed=1)
    assert not rep["adopt_fit"] and rep["heldout_gain_nats_per_move"] < kf.GATE_GAIN_NATS
    f = rep["fitted_all"]
    assert abs(f["beta"] - 8.0) < 4.0 or abs(rep["mean_ll_fitted_all"] - rep["mean_ll_reference_all"]) < 0.02


def test_cli_round_trip(tmp_path, planted_records):
    import json
    path = tmp_path / "moves.jsonl"
    with path.open("w") as f:
        for r in planted_records:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "fit.json"
    assert kf.main(["--records", str(path), "--out", str(out), "--brief", "x", "--folds", "1"]) == 0
    rep = json.load(out.open())
    assert rep["brief"] == "x" and rep["n_moves"] == len(planted_records) and "fitted_all" in rep
