"""CD-MIS-1 gates: true processes are valid kernels; filters stay normalised and never divide by
zero; under the matched process the naive filter tracks the exact filter and O_true equals the
assumed ASK planner; the runner's records are complete."""

import gzip
import json
import os
import random
import tempfile

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from . import belief_live as bl
from . import filters_mis as fm
from .opponents_mis import PROCESSES, assumed_kernel, true_kernel
from .oracle import Oracle
from .protocol_live import ASK, LiveConfig, apply, legal_actions, new_game, other
from .run_mis import main as run_main

CFG = LiveConfig(horizon=3)


def test_true_processes_are_valid_kernels_everywhere_reachable():
    rng = random.Random(5)
    for proc in PROCESSES:
        nodes = 0
        for ep in range(80):
            focal = rng.choice(("buyer", "seller")); opp = rng.choice(ALL_RANKINGS)
            k = true_kernel(proc, other(focal), CFG)
            st = new_game(CFG, rng.choice(("buyer", "seller")))
            while st.terminal is None:
                if st.turn == focal:
                    a = rng.choice(legal_actions(st, focal))
                    st = apply(st, a, k.sample_answer(a.question_id, opp, rng) if a.kind == ASK else None)
                else:
                    dist = k.distribution(st, opp); nodes += 1
                    legal = set(legal_actions(st, k.role))
                    assert abs(sum(dist.values()) - 1) < 1e-9 and all(x in legal for x in dist) and all(p >= 0 for p in dist.values()), proc
                    if proc == "withholding" and st.remaining > 1:
                        assert all(x.kind != "ACCEPT" for x in dist)
                    st = apply(st, k.sample(st, opp, rng))
        assert nodes > 100


def test_filters_normalised_and_floor_survives_impossible_observations():
    focal, rk = "buyer", ALL_RANKINGS[0]
    assumed = assumed_kernel("seller", CFG)
    st = new_game(CFG, "buyer")
    # the focal proposes its ideal deal; under the assumed kernel the seller never WALKs early
    from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS
    from .protocol_live import Action, PROPOSE, WALK
    st = apply(st, Action(PROPOSE, deal_id=ALL_DEALS[0].id))
    assert st.turn == "seller"
    impossible = Action(WALK)
    assert all(impossible not in assumed.distribution(st, th) for th in ALL_RANKINGS)
    for name, spec in fm.FILTERS.items():
        lik = fm.opponent_action_likelihood(assumed, st, impossible, spec)
        b = fm.update(bl.uniform(), lik, spec)
        assert abs(sum(b) - 1) < 1e-12 and all(x >= 0 for x in b), name
        if not spec.update:
            assert b == bl.uniform()
    # tempering with tau < 1 moves less than naive from the same observation
    liks = bl.action_likelihoods(assumed, st)
    act = next(a for a in liks if a.kind != WALK)
    naive = fm.update(bl.uniform(), fm.opponent_action_likelihood(assumed, st, act, fm.FILTERS["B_naive"]), fm.FILTERS["B_naive"])
    temp = fm.update(bl.uniform(), fm.opponent_action_likelihood(assumed, st, act, fm.FILTERS["B_temp0.5"]), fm.FILTERS["B_temp0.5"])
    assert bl.entropy_bits(temp) >= bl.entropy_bits(naive) - 1e-12


def test_matched_process_naive_tracks_exact_and_o_true_equals_assumed_planner():
    focal, rk = "seller", ALL_RANKINGS[3]
    assumed = assumed_kernel("buyer", CFG); truth = true_kernel("matched", "buyer", CFG)
    for opener in ("buyer", "seller"):
        assert abs(Oracle(CFG, focal, rk, truth, True).root_value(opener) - Oracle(CFG, focal, rk, assumed, True).root_value(opener)) < 1e-12
    rng = random.Random(9)
    for ep in range(60):
        opp = rng.choice(ALL_RANKINGS)
        st, b_naive, b_true = new_game(CFG, rng.choice(("buyer", "seller"))), bl.uniform(), bl.uniform()
        while st.terminal is None:
            if st.turn == focal:
                st = apply(st, rng.choice([a for a in legal_actions(st, focal) if a.kind != ASK]))
            else:
                act = truth.sample(st, opp, rng)
                b_naive = fm.update(b_naive, fm.opponent_action_likelihood(assumed, st, act, fm.FILTERS["B_naive"]), fm.FILTERS["B_naive"])
                b_true = bl.posterior(b_true, bl.action_likelihoods(truth, st)[act])
                st = apply(st, act)
        assert max(abs(x - y) for x, y in zip(b_naive, b_true)) < 0.02   # the 1e-3 floor only


def test_runner_records_complete():
    with tempfile.TemporaryDirectory() as d:
        run_main(["--process", "history,evasive", "--horizon", "3", "--roles", "buyer", "--types", "1", "--opp-types", "0,5",
                  "--repeats", "1", "--arms", "A_prior,B_naive,B_mix0.2,C_naive,O_true", "--seed", "1", "--out", d])
        recs = [json.loads(l) for l in gzip.open(os.path.join(d, "runs.jsonl.gz"), "rt")]
        summ = json.load(open(os.path.join(d, "summary.json")))
        assert summ["error_stubs"] == 0 and len(recs) == 2 * 5 * 2 * 2
        for r in recs:
            assert "error" not in r and r["n_decisions"] == len(r["decisions"]) >= 1
            assert 0.0 <= r["final_agent"]["p_true"] <= 1.0 and r["final_true_filter"]["p_true"] >= 0.0
            assert r["total_decision_regret_true"] >= -1e-9
            if r["arm"] == "O_true":
                assert r["total_decision_regret_true"] <= 1e-9
            if r["arm"] == "A_prior":
                assert abs(r["final_agent"]["entropy_bits"] - bl.entropy_bits(bl.uniform())) < 1e-12
            if r["arm"] in ("A_prior", "B_naive", "B_mix0.2"):
                assert r["questions_asked"] == 0
