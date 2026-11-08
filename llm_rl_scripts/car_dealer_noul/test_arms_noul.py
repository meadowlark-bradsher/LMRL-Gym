"""Gates for the Stage 1 arms and runner: arm parsing, common random numbers, the text bank,
parity of ``plan_ask-oracle-0`` / ``plan_noask`` with CD-VOI-1's ``exact_ask`` / ``exact_noask`` on
the frozen Phase 1 seeds, ceilings unchanged, the agents' text, and a resumable offline run."""

from __future__ import annotations

import gzip
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, LiveConfig

from . import arms_noul as an
from . import run_noul as rn
from .agents_noul import SonnetAgent, TextJevPolicy, parse_letter

BANK = Path("results/noul_bank_20261108")
PHASE1 = Path("results/live_phase1_jev_h4_cq0_20261029")
CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))

needs_bank = pytest.mark.skipif(not (BANK / "matrix_sonnet.jsonl").exists(), reason="frozen bank + matrices absent")
needs_phase1 = pytest.mark.skipif(not (PHASE1 / "runs.jsonl.gz").exists(), reason="frozen Phase 1 run absent")


@pytest.fixture(scope="module")
def bank():
    return an.TextBank(BANK)


@pytest.fixture(scope="module")
def cell_buyer0():
    return an.NoulCell(CFG, "buyer", ALL_RANKINGS[0])


def test_parse_arm_round_trips():
    names = an.planner_arms()
    assert len(names) == len(an.DECODERS) * len(an.HEDGES) + 1
    for n in names + list(an.AGENT_ARMS):
        a = an.parse_arm(n)
        assert a.name == n
    a = an.parse_arm("plan_ask-noul_type-0.2")
    assert a.kind == "plan_ask" and a.decoder == "noul_type" and a.hedge == Fraction(1, 5) and a.asks
    assert not an.parse_arm("plan_noask").asks and not an.parse_arm("plan_noask").uses_text
    assert an.parse_arm("sonnet_agent").decoder == an.AGENT_MENU_DECODER
    with pytest.raises(ValueError):
        an.parse_arm("plan_ask-oracle-0.5")


def test_variant_plan_is_arm_independent_and_episode_specific():
    a = an.variant_plan(1, "buyer", "seller", 0, 3, 2)
    b = an.variant_plan(1, "buyer", "seller", 0, 3, 2)
    c = an.variant_plan(1, "buyer", "seller", 0, 3, 3)
    assert a == b and a != c and len(a) == 9 and all(0 <= v < 20 for v in a.values())


@needs_bank
def test_text_bank_serves_truthful_utterances_and_likelihoods(bank):
    row = bank.utterance("plain", "seller", "top:price", ALL_RANKINGS[0], 7)
    assert row["item"]["truthful"] == "yes" and row["item"]["variant"] == 7 and row["item"]["role"] == "seller"
    lik, read = bank.likelihood("oracle", row, Fraction(0))
    assert read == {"p_yes": 1.0}
    # truthful-yes types get the high likelihood, the rest the floor
    assert max(lik) > 0.99 and min(lik) < 1e-2
    lik2, read2 = bank.likelihood("noul_type", row, Fraction(1, 5))
    assert len(lik2) == 6 and len(read2["type_vector"]) == 6
    for d in an.DECODERS:
        if d != "noul_type":
            assert 0.0 <= bank.p_yes(d, row) <= 1.0


@needs_bank
def test_ceilings_match_frozen_phase1(cell_buyer0):
    frozen = json.load(open(PHASE1 / "ceilings.json"))["cells"]["buyer|buyer|p/w/d"]
    c = cell_buyer0.ceilings("buyer")
    for k in ("menu_ask", "menu_noask", "full_ask", "full_noask", "delta_menu"):
        assert c[k] == pytest.approx(frozen[k], abs=1e-9), k


def _phase1_records(arm: str, focal: str, ranking):
    out = {}
    with gzip.open(PHASE1 / "runs.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] == arm and r["focal"] == focal and r["focal_ranking"] == list(ranking):
                out[(r["opener"], tuple(r["private_truth_evaluator_only"]["opp_ranking"]), r["repeat"])] = r
    return out


@needs_bank
@needs_phase1
def test_plan_ask_oracle_0_replays_exact_ask_and_plan_noask_replays_exact_noask(bank, cell_buyer0):
    seed = json.load(open(PHASE1 / "summary.json"))["seed"]
    for arm_name, ref_arm in (("plan_ask-oracle-0", "exact_ask"), ("plan_noask", "exact_noask")):
        arm = an.parse_arm(arm_name)
        ref = _phase1_records(ref_arm, "buyer", ALL_RANKINGS[0])
        assert ref, "frozen Phase 1 run has no records for this cell"
        n = 0
        for opener in ("buyer", "seller"):
            for j in range(6):
                for k in range(2):
                    rng = random.Random(f"{seed}|buyer|{opener}|0|{j}|{k}")
                    variants = an.variant_plan(seed, "buyer", opener, 0, j, k)
                    rec = an.play_episode(cell_buyer0, arm, "plain" if arm.uses_text else an.NO_TEXT_REGIME, opener,
                                          ALL_RANKINGS[j], rng, variants, bank, k)
                    r = ref[(opener, ALL_RANKINGS[j], k)]
                    assert rec["normalized_net_gain"]["exact"] == r["normalized_net_gain"]["exact"], (arm_name, opener, j, k)
                    assert rec["questions_asked"] == r["questions_asked"]
                    assert [d["chosen"] for d in rec["decisions"]] == [d["chosen"] for d in r["decisions"]]
                    n += 1
        assert n == 24


@needs_bank
def test_plan_noask_never_asks_and_decoded_arms_record_text(bank, cell_buyer0):
    rng = random.Random("x")
    rec = an.play_episode(cell_buyer0, an.parse_arm("plan_noask"), an.NO_TEXT_REGIME, "buyer", ALL_RANKINGS[3], rng,
                          an.variant_plan(0, "buyer", "buyer", 0, 3, 0), bank, 0)
    assert rec["questions_asked"] == 0 and all(d["chosen_kind"] != ASK for d in rec["decisions"])
    for d_name in ("regex", "noul_answer", "noul_type"):
        arm = an.parse_arm(f"plan_ask-{d_name}-0.2")
        rng = random.Random("y")
        rec = an.play_episode(cell_buyer0, arm, "oblique", "buyer", ALL_RANKINGS[3], rng,
                              an.variant_plan(0, "buyer", "buyer", 0, 3, 0), bank, 0)
        asks = [d for d in rec["decisions"] if d["chosen_kind"] == ASK]
        for d in asks:
            dec = d["decode"]
            assert dec["decoder"] == d_name and dec["bank_id"].startswith("oblique|seller|")
            assert 0.0 <= dec["p_true_after"] <= 1.0 and len(dec["likelihood"]) == 6
        assert rec["hedge"] == "1/5" and rec["regime"] == "oblique"


@needs_bank
def test_common_random_numbers_across_arms(bank, cell_buyer0):
    """Two text arms asking the same question in the same episode hear the same utterance."""
    heard = {}
    for d_name in ("oracle", "regex"):
        rng = random.Random(f"5|buyer|seller|0|2|0")
        rec = an.play_episode(cell_buyer0, an.parse_arm(f"plan_ask-{d_name}-0"), "plain", "seller", ALL_RANKINGS[2], rng,
                              an.variant_plan(5, "buyer", "seller", 0, 2, 0), bank, 0)
        for d in rec["decisions"]:
            if d["decode"]:
                heard.setdefault(d["decode"]["bank_id"].split("|")[2], set()).add(d["decode"]["bank_id"])
    for q, ids in heard.items():
        assert len(ids) == 1, q


def test_text_jev_policy_prints_utterances_and_parse_letter():
    pol = TextJevPolicy(JevClient(MockJevBackend()), CFG, "buyer", ALL_RANKINGS[0])
    assert "in its own words" in pol.instructions()
    assert parse_letter("I choose B because...", "ABC") == "B"
    assert parse_letter("Option C.", "ABC") == "C"
    assert parse_letter("PROPOSE the deal", "ABC") is None
    assert parse_letter("nothing", "ABC") is None


@needs_bank
def test_agent_arms_run_offline_and_log_choice(bank, cell_buyer0):
    jev = TextJevPolicy(JevClient(MockJevBackend()), CFG, "buyer", ALL_RANKINGS[0], shuffle_rng=random.Random(1))
    rng = random.Random("a")
    rec = an.play_episode(cell_buyer0, an.parse_arm("jev_agent"), "plain", "buyer", ALL_RANKINGS[1], rng,
                          an.variant_plan(0, "buyer", "buyer", 0, 1, 0), bank, 0, agent=jev)
    assert rec["decisions"] and all(d["agent"]["well_formed"] for d in rec["decisions"])
    son = SonnetAgent(rn.FakeSonnet(), CFG, "buyer", ALL_RANKINGS[0], shuffle_rng=random.Random(2))
    rng = random.Random("a")
    rec2 = an.play_episode(cell_buyer0, an.parse_arm("sonnet_agent"), "oblique", "buyer", ALL_RANKINGS[1], rng,
                           an.variant_plan(0, "buyer", "buyer", 0, 1, 0), bank, 0, agent=son)
    assert rec2["decisions"] and all(d["agent"]["chosen_letter"] == "A" for d in rec2["decisions"])
    # an ASK by the agent puts the utterance into the next state text
    asked = [d for d in rec2["decisions"] if d["chosen_kind"] == ASK]
    if asked:
        assert son.utterances and all(isinstance(v, str) and v for v in son.utterances.values())


@needs_bank
def test_runner_is_resumable(tmp_path: Path, bank):
    out = tmp_path / "run"
    s1 = rn.run(CFG, ["plan_noask", "plan_ask-regex-0.2", "sonnet_agent"], ["plain"], bank, out, repeats=1, seed=3,
                sonnet_client=rn.FakeSonnet(), roles=("buyer",), openers=("buyer",), types=[0], opp_types=[0, 1], log=None)
    assert s1["episodes_written"] == 2 + 2 + 2 and s1["error_stubs"] == 0
    s2 = rn.run(CFG, ["plan_noask", "plan_ask-regex-0.2", "sonnet_agent"], ["plain"], bank, out, repeats=1, seed=3,
                sonnet_client=rn.FakeSonnet(), roles=("buyer",), openers=("buyer",), types=[0], opp_types=[0, 1], log=None)
    assert s2["episodes_written"] == 0 and s2["episodes_resumed"] == 6
    lines = [json.loads(l) for l in (out / "runs.jsonl").open()]
    assert len(lines) == 6 and len({l["episode_key"] for l in lines}) == 6
    assert (out / "ceilings.json").exists() and (out / "summary.json").exists()
