"""CD-MIS-2 gates: severity ids parse to valid kernels at the stated parameter; the hedged answer
likelihood is the stated mixture and stays normalised; the hedged arm whose assumed flip rate equals
the true rate has zero true-model regret on the evasive family (it is O_true up to tie-breaks); the no-ask arms are identical
across evasive levels (family-seeded streams); a bare family id reproduces the CD-MIS-1 kernel."""

import gzip
import json
import os
import random
import tempfile
from fractions import Fraction

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from . import belief_live as bl
from . import filters_mis as fm
from .opponents_mis import MIS1_LEVEL, assumed_kernel, family, parse_process, severity, true_kernel
from .protocol_live import ASK, BANK_BY_ID, LiveConfig, apply, legal_actions, new_game, other
from .run_mis import main as run_main

CFG = LiveConfig(horizon=4)


def test_severity_ids_parse_and_bind_the_parameter():
    assert parse_process("history@0.4") == ("history", Fraction(2, 5)) and parse_process("history") == ("history", None)
    assert family("evasive@1/2") == "evasive" and severity("evasive") == MIS1_LEVEL["evasive"] and severity("matched") is None
    assert true_kernel("history@0.4", "seller", CFG).step == Fraction(2, 5)
    assert true_kernel("history", "seller", CFG).step == MIS1_LEVEL["history"]
    assert true_kernel("stochastic@0.5", "buyer", CFG).p.accept_noise == 0.5
    assert true_kernel("anchoring@6", "buyer", CFG).power == 6
    assert true_kernel("evasive@0.1", "seller", CFG).cfg.answer_noise == Fraction(1, 10)
    assert true_kernel("evasive@0", "seller", CFG).cfg.answer_noise == 0
    q = next(iter(BANK_BY_ID))
    d = true_kernel("evasive@0.5", "seller", CFG).answer_distribution(q, ALL_RANKINGS[0])
    assert abs(d["yes"] - 0.5) < 1e-12 and abs(d["no"] - 0.5) < 1e-12
    # severity kernels are valid distributions at reachable nodes
    rng = random.Random(3)
    for proc in ("history@0.4", "stochastic@0.5", "anchoring@6"):
        k = true_kernel(proc, "buyer", CFG)
        for ep in range(40):
            opp = rng.choice(ALL_RANKINGS)
            st = new_game(CFG, rng.choice(("buyer", "seller")))
            while st.terminal is None:
                if st.turn == "seller":
                    a = rng.choice(legal_actions(st, "seller"))
                    st = apply(st, a, k.sample_answer(a.question_id, opp, rng) if a.kind == ASK else None)
                else:
                    dist = k.distribution(st, opp)
                    assert abs(sum(dist.values()) - 1) < 1e-9 and all(p >= 0 for p in dist.values()), proc
                    st = apply(st, k.sample(st, opp, rng))


def test_hedged_answer_likelihood_is_the_stated_mixture():
    assumed = assumed_kernel("seller", CFG)
    q = next(iter(BANK_BY_ID))
    truthful = bl.answer_likelihoods(assumed, q)["yes"]
    spec = fm.FILTERS["B_ans0.2"]
    lik = fm.answer_likelihood(assumed, q, "yes", spec)
    for p, l in zip(truthful, lik):
        expected = (1 - spec.eta) * (0.8 * p + 0.2 * (1 - p)) + spec.eta * 0.5
        assert abs(l - expected) < 1e-12
    # a hedged kernel used by the planner gives the same per-type answer model (before the floor)
    hedged = assumed_kernel("seller", CFG, answer_noise=Fraction(1, 5))
    for p, h in zip(truthful, bl.answer_likelihoods(hedged, q)["yes"]):
        assert abs(h - (0.8 * p + 0.2 * (1 - p))) < 1e-12
    b = fm.update(bl.uniform(), lik, spec)
    assert abs(sum(b) - 1) < 1e-12 and bl.entropy_bits(b) > bl.entropy_bits(fm.update(bl.uniform(), fm.answer_likelihood(assumed, q, "yes", fm.FILTERS["B_naive"]), fm.FILTERS["B_naive"]))


def test_runner_diagonal_identity_and_noask_invariance():
    with tempfile.TemporaryDirectory() as d:
        run_main(["--process", "evasive@0.3,evasive@0.1,history@0.4", "--horizon", "4", "--roles", "buyer", "--types", "2", "--opp-types", "0,3",
                  "--repeats", "2", "--arms", "A_prior,B_naive,C_naive,C_ans0.3,O_true", "--seed", "7", "--design", "CD-MIS-2", "--out", d])
        recs = [json.loads(l) for l in gzip.open(os.path.join(d, "runs.jsonl.gz"), "rt")]
        summ = json.load(open(os.path.join(d, "summary.json")))
        assert summ["error_stubs"] == 0 and summ["design"] == "CD-MIS-2" and len(recs) == 3 * 5 * 2 * 2 * 2
        assert summ["process_meta"]["evasive@0.3"] == {"family": "evasive", "severity": "3/10"}
        assert summ["hedged_arms"] == {"C_ans0.3": "3/10"}
        key = lambda r: (r["process"], r["arm"], r["opener"], tuple(r["private_truth_evaluator_only"]["opp_ranking"]), r["repeat"])
        by = {key(r): r for r in recs}
        sig = lambda r: ([x["chosen"] for x in r["decisions"]], r["terminal"])
        for r in recs:
            assert "error" not in r and r["n_decisions"] == len(r["decisions"]) >= 1
            p, arm, opener, opp, k = key(r)
            if p == "evasive@0.3" and arm == "C_ans0.3":
                # the diagonal is the true-model planner: zero regret under the truth (trajectories may differ at
                # exact ties, which the 1e-3 floor breaks differently from the exact filter)
                assert r["total_decision_regret_true"] <= 1e-9
                assert len(r["decisions"]) == len(by[(p, "O_true", opener, opp, k)]["decisions"])
            if p == "evasive@0.1" and arm in ("A_prior", "B_naive"):
                assert sig(r) == sig(by[("evasive@0.3", arm, opener, opp, k)])  # no-ask arms never see an answer
            if arm in ("A_prior", "B_naive"):
                assert r["questions_asked"] == 0


def test_bare_family_reproduces_mis1_kernel_and_seed_stream():
    from .run_mis import HEDGED_ARMS
    assert set(HEDGED_ARMS) == {"C_ans0.1", "C_ans0.2", "C_ans0.3"}
    k1, k2 = true_kernel("evasive", "seller", CFG), true_kernel("evasive@0.3", "seller", CFG)
    assert k1.cfg.answer_noise == k2.cfg.answer_noise == Fraction(3, 10) and k1.p == k2.p
    h1, h2 = true_kernel("history", "seller", CFG), true_kernel("history@0.1", "seller", CFG)
    assert h1.step == h2.step and family("history@0.1") == family("history") == "history"
