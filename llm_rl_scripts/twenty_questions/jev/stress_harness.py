"""ST-0d snapshot mode and the shared stress-test scorecard.

    # mechanics self-test for every ST-0 piece (offline, ~1 min):
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_harness --self-test

**Snapshot mode** seeds ``H_known`` with a whole pool and plays a fixed mix of
in-pool games (secrets uniform over the pool) and novelty games (secrets from a
held-out set the planner has never seen). Nothing is admitted, so the pool does
not grow: crowding is held fixed for the whole cell, isolated from the discovery
dynamics ST-3 studies.

**Hash seed.** The survivor-set backends (``hard``, ``retract``, ``tombstone``,
``modeswitch``) iterate Python ``set``s of names, whose order depends on
``PYTHONHASHSEED``; it moves questions-per-game in the third decimal (the
in-pool failure rate, AUC and wrong count are stable). Stress runs therefore
refuse to start without a fixed ``PYTHONHASHSEED`` and record it, so every
stress number regenerates exactly. Run them under the ``LLM_RL`` env (Python
3.9): Python 3.12's compensated float ``sum`` shifts posteriors by an ulp and
breaks bit-for-bit parity with the committed runs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .analyze_openworld import auc
from .bed import GUESS, Hypothesis, OracleModel, matrix_oracle
from .openworld import run_openworld_episode

Matrix = Dict[str, Dict[str, float]]


def require_hash_seed() -> str:
    """The pinned ``PYTHONHASHSEED``; raises if it is unset or ``random``."""
    hs = os.environ.get("PYTHONHASHSEED")
    if hs is None or hs == "random":
        raise RuntimeError(
            "stress runs need a fixed PYTHONHASHSEED (e.g. PYTHONHASHSEED=0): the "
            "survivor-set backends iterate sets of names, so results depend on it")
    return hs


@dataclass
class SnapshotRecord:
    game: int
    true_name: str
    true_category: str
    in_pool: bool
    outcome: str                 # won | expert | wrong | false_confirm
    n_questions: int
    n_guesses: int               # by-name guesses asked
    n_repeat_guesses: int        # guesses of a name already guessed this game
    p_unknown_max: float
    fire_turn: Optional[int]
    n_retractions: int
    n_live_final: int            # live knowns at budget exhaustion (-1: n/a)


def run_snapshot(pool: Sequence[Hypothesis], matrix: Matrix,
                 novel: Sequence[Hypothesis], n_in_pool: int,
                 oracle: Optional[OracleModel] = None, backend: str = "hard",
                 seed: int = 0, p_unknown_prior: Optional[float] = None,
                 max_questions: int = 20, confidence: float = 0.9,
                 eps: float = 0.05, margin: float = 2.0,
                 tolerance: int = 3, no_repeat: bool = False) -> List[SnapshotRecord]:
    """Play ``n_in_pool`` in-pool games and one game per ``novel`` object, in a
    seeded shuffled order, against a fixed ``H_known = pool``. ``matrix`` must
    hold rows for the pool and the novel objects (the answerer reads both). The
    catch-all prior defaults to the novelty share of the game mix. Above ~0.1
    the survivor-set backends loop on a repeated question unless ``no_repeat``
    is set (``STRESS_ST0.md``)."""
    rng = random.Random(seed)
    secrets = [rng.choice(pool) for _ in range(n_in_pool)] + list(novel)
    rng.shuffle(secrets)
    if p_unknown_prior is None:
        p_unknown_prior = len(novel) / len(secrets)
    if oracle is None:
        oracle = matrix_oracle(matrix)
    begin_game = getattr(oracle, "begin_game", None)
    pool = list(pool)
    out: List[SnapshotRecord] = []
    for g, true_h in enumerate(secrets):
        if begin_game is not None:
            begin_game(g)
        ep = run_openworld_episode(
            true_h, pool, matrix, p_unknown_prior, oracle,
            max_questions=max_questions, confidence=confidence, eps=eps,
            backend=backend, margin=margin, tolerance=tolerance, no_repeat=no_repeat)
        guesses = [s.qid for s in ep.steps if s.kind == GUESS]
        out.append(SnapshotRecord(
            game=g, true_name=true_h.name, true_category=true_h.category,
            in_pool=ep.in_pool, outcome=ep.outcome, n_questions=ep.n_questions,
            n_guesses=len(guesses), n_repeat_guesses=len(guesses) - len(set(guesses)),
            p_unknown_max=ep.p_unknown_max, fire_turn=ep.fire_turn,
            n_retractions=ep.n_retractions, n_live_final=ep.n_live_final))
    return out


def scorecard(records: Sequence, confidence: float = 0.9) -> Dict[str, float]:
    """The shared stress scorecard (``STRESS_DESIGN.md``). Works on snapshot
    records and on ``openworld.GameRecord`` alike; fields a record type lacks
    are skipped."""
    inp = [r for r in records if r.in_pool]
    nov = [r for r in records if not r.in_pool]

    def frac(rs, pred):
        return sum(1 for r in rs if pred(r)) / len(rs) if rs else float("nan")

    out: Dict[str, float] = {
        "games": len(records), "in_pool_games": len(inp), "novel_games": len(nov),
        "in_pool_won": frac(inp, lambda r: r.outcome == "won"),
        "in_pool_safe_escalation": frac(inp, lambda r: r.outcome == "expert"),
        "in_pool_wrong": frac(inp, lambda r: r.outcome == "wrong"),
        "in_pool_false_confirm": frac(inp, lambda r: r.outcome == "false_confirm"),
        "novel_expert": frac(nov, lambda r: r.outcome == "expert"),
        "novel_wrong": frac(nov, lambda r: r.outcome == "wrong"),
        "novel_false_confirm": frac(nov, lambda r: r.outcome == "false_confirm"),
        "mean_questions": (sum(r.n_questions for r in records) / len(records)
                           if records else float("nan")),
        "expert_calls": sum(1 for r in records if r.outcome == "expert"),
    }
    if inp and nov:
        out["auc"] = auc([r.p_unknown_max for r in records],
                         [0 if r.in_pool else 1 for r in records])
        out["tpr"] = frac(nov, lambda r: r.p_unknown_max >= confidence)
        out["fpr"] = frac(inp, lambda r: r.p_unknown_max >= confidence)
    fired = [r.fire_turn for r in nov if r.fire_turn is not None]
    out["novel_median_fire_turn"] = statistics.median(fired) if fired else float("nan")
    if records and hasattr(records[0], "n_guesses"):
        out["mean_guesses"] = sum(r.n_guesses for r in records) / len(records)
        out["mean_repeat_guesses"] = sum(r.n_repeat_guesses for r in records) / len(records)
    if records and hasattr(records[0], "n_live_final"):
        wrong = [r for r in records if r.outcome == "wrong" and r.n_live_final >= 0]
        out["budget_exhaustion_share"] = frac(wrong, lambda r: r.n_live_final >= 2)
    return out


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def self_test(matrix_path: str) -> None:
    from .bed import ATTRIBUTE, ATTRIBUTE_BANK, Question, load_hypotheses
    from .matrix import load_matrix
    from .openworld import run_accumulation
    from .stress_census import census
    from .stress_generator import (AIDS, build_pool, category_values, generate_objects,
                                   lookalike, merge)
    from .stress_oracles import (FlipCounter, bursty_oracle, guess_noise, hamming,
                                 lookalike_confusion, sampled_oracle, sticky_oracle)

    require_hash_seed()
    objs = load_hypotheses()
    matrix = load_matrix(matrix_path)["matrix"]
    names = [h.name for h in objs]

    # ST-0a generator ------------------------------------------------------- #
    gen0, rows0 = generate_objects(objs, matrix, 50, 0.0, seed=1)
    parents = {tuple(matrix[a][n] for a in AIDS) for n in names}
    assert all(tuple(r[a] for a in AIDS) in parents for r in rows0.values()), \
        "rho=0 must copy a parent row exactly"
    gen1, rows1 = generate_objects(objs, matrix, 50, 1.0, seed=1)
    vals = category_values(objs, matrix)
    for h in gen1:
        assert all(rows1[h.name][a] in vals[h.category][a] for a in AIDS)
    again, _ = generate_objects(objs, matrix, 50, 1.0, seed=1)
    assert [h.name for h in again] == [h.name for h in gen1], "generator not seeded"
    pool, pm = build_pool(objs, matrix, 300, 0.25, seed=2)
    assert len(pool) == 300 and pool[:len(objs)] == list(objs)
    assert all(pm[a][n] == matrix[a][n] for a in AIDS for n in names), "real rows changed"
    assert all(n not in matrix[AIDS[0]] for n in [h.name for h in pool[len(objs):]]), \
        "build_pool mutated the input matrix"
    rng = random.Random(3)
    for d in (1, 2, 6, 12):
        la, row = lookalike(objs[d], matrix, d, rng, "la_{}".format(d))
        lm = merge(matrix, {la.name: row})
        assert hamming(lm, la.name, objs[d].name) == d, "lookalike distance != d"
        assert la.category == objs[d].category
    print("[self-test] ST-0a generator OK (rho=0 copies, rho=1 within category, "
          "seeded, real rows untouched, lookalike at exact d)")

    # ST-0b answer channels -------------------------------------------------- #
    secret = objs[0]
    st = sticky_oracle(matrix, 2, seed=5)
    cells = st.wrong_cells(secret.name)
    assert len(cells) == 2 and cells == st.wrong_cells(secret.name)
    for aid, text in ATTRIBUTE_BANK:
        q = Question(ATTRIBUTE, aid, text)
        expect = (matrix[aid][secret.name] >= 0.5) != (aid in cells)
        assert st(q, secret) == expect and st(q, secret) == expect, "sticky not sticky"
    aid_mid = min(AIDS, key=lambda a: abs(matrix[a][secret.name] - 0.3))
    p = matrix[aid_mid][secret.name]
    sam = sampled_oracle(matrix, seed=7)
    qm = Question(ATTRIBUTE, aid_mid, "")
    freq = sum(sam(qm, secret) for _ in range(4000)) / 4000
    assert abs(freq - p) < 0.03, "sampled answerer rate {} vs p {}".format(freq, p)
    bo = bursty_oracle(matrix, burst_p=0.5, inner_flip=1.0, seed=9)
    calls = {"n": 0}
    real_begin = bo.begin_game

    def counting_begin(g):
        calls["n"] += 1
        real_begin(g)

    bo.begin_game = counting_begin
    stream = [i % 10 for i in range(30)]
    run_accumulation(objs[:10], matrix, stream, 0.2, 20.0, backend="hard", oracle=bo)
    assert calls["n"] == 30, "begin_game called {} times for 30 games".format(calls["n"])
    gq = Question(GUESS, secret.name, "")
    gn = guess_noise(matrix_oracle(matrix), matrix, false_yes=0.0, false_no=1.0, seed=1)
    assert gn(gq, secret) is False
    near = min((h for h in objs if h.name != secret.name),
               key=lambda h: hamming(matrix, h.name, secret.name))
    gy = guess_noise(matrix_oracle(matrix), matrix, false_yes=1.0, false_no=0.0,
                     seed=1, near=hamming(matrix, near.name, secret.name))
    assert gy(Question(GUESS, near.name, ""), secret) is True
    lc = lookalike_confusion(matrix_oracle(matrix), {secret.name: near.name}, 1.0, seed=1)
    assert lc(Question(GUESS, near.name, ""), secret) is True
    fc = FlipCounter(st, matrix)
    for aid, text in ATTRIBUTE_BANK:
        fc(Question(ATTRIBUTE, aid, text), secret)
    assert fc.rates()["attr_flip_rate"] == 2 / len(ATTRIBUTE_BANK)
    print("[self-test] ST-0b answer channels OK (sticky, sampled rate, bursty "
          "begin_game per game, guess false-yes/no, look-alike confusion, flip counter)")

    # ST-0c false_confirm ----------------------------------------------------- #
    def yes_to_any_guess(q, true_h):
        return True if q.kind == GUESS else matrix_oracle(matrix)(q, true_h)

    known = objs[:40]
    novel_h = objs[60]
    ep = run_openworld_episode(novel_h, known, matrix, 0.1, yes_to_any_guess,
                               backend="hard")
    assert ep.outcome == "false_confirm", ep.outcome
    ep_in = run_openworld_episode(known[5], known, matrix, 0.1, matrix_oracle(matrix),
                                  backend="hard")
    assert ep_in.outcome == "won", ep_in.outcome
    print("[self-test] ST-0c scoring OK (a 'yes' to the wrong name is false_confirm; "
          "the exact oracle still wins)")

    # ST-0d snapshot ---------------------------------------------------------- #
    pool_s, pm_s = build_pool(objs, matrix, 200, 0.25, seed=4)
    nov_h, nov_rows = generate_objects(objs, matrix, 20, 0.25, seed=99, prefix="nov")
    full = merge(pm_s, nov_rows)
    for bk in ("soft", "hard", "modeswitch"):
        recs = run_snapshot(pool_s, full, nov_h, n_in_pool=60, backend=bk, seed=1,
                            p_unknown_prior=0.05)
        assert len(recs) == 80 and sum(1 for r in recs if not r.in_pool) == 20
        assert not any(r.outcome == "false_confirm" for r in recs), \
            "exact oracle produced a false_confirm"
        assert not any(r.outcome == "won" for r in recs if not r.in_pool)
        sc = scorecard(recs)
        assert 0.0 <= sc["auc"] <= 1.0
        if bk != "soft":
            assert all(r.n_live_final >= 0 for r in recs if r.outcome == "wrong")
        print("  snapshot {:10s} in-pool won {:.3f}  wrong {:.3f}  novel expert {:.3f}"
              "  AUC {:.3f}  q/game {:.2f}".format(
                  bk, sc["in_pool_won"], sc["in_pool_wrong"], sc["novel_expert"],
                  sc["auc"], sc["mean_questions"]))
    # The repeat-question loop: at prior 0.25 hard re-asks a question that splits
    # UNKNOWN from the knowns; no_repeat breaks the loop.
    loop = run_snapshot(pool_s, full, [], n_in_pool=30, backend="hard", seed=2,
                        p_unknown_prior=0.25)
    guarded = run_snapshot(pool_s, full, [], n_in_pool=30, backend="hard", seed=2,
                           p_unknown_prior=0.25, no_repeat=True)
    won_loop = sum(r.outcome == "won" for r in loop) / 30
    won_guarded = sum(r.outcome == "won" for r in guarded) / 30
    assert won_loop < 0.5 < won_guarded, (won_loop, won_guarded)
    print("  repeat-question loop at prior 0.25: hard won {:.2f} -> {:.2f} with "
          "no_repeat".format(won_loop, won_guarded))
    c = census(full, [h.name for h in pool_s])
    assert c["n_objects"] == 200
    print("[self-test] ST-0d snapshot OK (fixed pool, held-out novelty, no admissions, "
          "census per cell)")
    print("[self-test] all ST-0 pieces OK (PYTHONHASHSEED={})".format(require_hash_seed()))


# --------------------------------------------------------------------------- #
# ST-0 diagnostics (cited in STRESS_ST0.md)                                    #
# --------------------------------------------------------------------------- #

def repeat_probe(matrix_path: str, games: int = 150, seed: int = 0) -> dict:
    """The repeat-question loop. (a) Closed real pool, ``games`` in-pool games
    per (backend, prior), with and without ``no_repeat``. (b) The committed
    clean accumulation stream under ``hard``: the games that re-asked a
    non-guess question, their outcome and the prior they were played at."""
    from collections import Counter
    from .bed import load_hypotheses
    from .matrix import load_matrix
    from . import openworld as ow
    from .stress_regression import _setup

    require_hash_seed()
    objs = load_hypotheses()
    matrix = load_matrix(matrix_path)["matrix"]
    oracle = matrix_oracle(matrix)

    def repeated(ep) -> bool:
        c = Counter(s.qid for s in ep.steps)
        return any(v > 1 for q, v in c.items() if not q.startswith("guess"))

    closed = []
    for bk in ("soft", "hard", "modeswitch"):
        for no_repeat in (False, True):
            for prior in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25):
                rng = random.Random(seed)
                won = rep = 0
                for _ in range(games):
                    ep = ow.run_openworld_episode(rng.choice(objs), objs, matrix, prior,
                                                  oracle, backend=bk, tolerance=3,
                                                  no_repeat=no_repeat)
                    won += ep.outcome == "won"
                    rep += repeated(ep)
                closed.append({"backend": bk, "no_repeat": no_repeat, "prior": prior,
                               "games": games, "won": won / games,
                               "repeated": rep / games})
    objs2, m2, stream, _, sigma, theta = _setup(2000, 1.1, 7)
    orig = ow.run_openworld_episode
    hits = []

    def wrapped(true_h, known, matrix_, prior, *a, **k):
        ep = orig(true_h, known, matrix_, prior, *a, **k)
        if repeated(ep):
            hits.append({"k_known": len(known), "prior": prior, "outcome": ep.outcome,
                         "in_pool": ep.in_pool})
        return ep

    ow.run_openworld_episode = wrapped
    try:
        recs = ow.run_accumulation(objs2, m2, stream, sigma, theta, backend="hard")
    finally:
        ow.run_openworld_episode = orig
    return {"hashseed": require_hash_seed(), "closed_pool": closed,
            "committed_stream_hard": {
                "games": len(recs), "repeat_games": len(hits),
                "repeat_in_pool_expert": sum(1 for h in hits if h["in_pool"]
                                             and h["outcome"] == "expert"),
                "won": sum(1 for r in recs if r.outcome == "won"),
                "prior_min": min((h["prior"] for h in hits), default=None),
                "prior_max": max((h["prior"] for h in hits), default=None),
                "hits": hits}}


def _hash_cell(args) -> dict:
    hs, bk, ns = args
    import subprocess
    import sys
    code = (
        "import json,sys\n"
        "from llm_rl_scripts.twenty_questions.jev.stress_regression import _setup, _retraction_metrics\n"
        "from llm_rl_scripts.twenty_questions.jev.openworld import run_accumulation\n"
        "o,m,s,_,sg,th=_setup(2000,1.1,7)\n"
        "r=run_accumulation(o,m,s,sg,th,oracle_noise=0.1,noise_seed={ns},backend='{bk}',tolerance=3)\n"
        "print(json.dumps(_retraction_metrics(r)))\n").format(ns=ns, bk=bk)
    env = dict(os.environ, PYTHONHASHSEED=str(hs))
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True,
                         capture_output=True, text=True).stdout
    row = json.loads(out.strip().splitlines()[-1])
    row.update(hashseed=hs, backend=bk, noise_seed=ns)
    return row


def hash_spread(hashseeds: Sequence[int] = (0, 1, 2, 3, 4), workers: int = 0) -> dict:
    """The retraction study's headline cells (hard, modeswitch@3 at 10% flips,
    noise seeds 1-2), each replayed under several PYTHONHASHSEEDs in fresh
    interpreters: how far the string-hash order moves each metric."""
    from multiprocessing import Pool
    jobs = [(hs, bk, ns) for hs in hashseeds for bk in ("hard", "modeswitch")
            for ns in (1, 2)]
    with Pool(workers or os.cpu_count() or 4) as pool:
        rows = pool.map(_hash_cell, jobs)
    agg = {}
    for bk in ("hard", "modeswitch"):
        for k in ("in_pool_fail_rate", "auc", "wrong", "false_positive_calls",
                  "mean_questions"):
            per_h = [statistics.mean(r[k] for r in rows if r["backend"] == bk
                                     and r["hashseed"] == hs) for hs in hashseeds]
            agg["{}.{}".format(bk, k)] = {"min": min(per_h), "max": max(per_h),
                                          "per_hashseed": per_h}
    return {"hashseeds": list(hashseeds), "rows": rows, "seed_mean_range": agg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repeat-probe", action="store_true")
    ap.add_argument("--hash-spread", action="store_true")
    ap.add_argument("--out", default=None, help="write the diagnostic JSON here")
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    args = ap.parse_args()
    if args.self_test:
        self_test(args.matrix)
        return
    res = None
    if args.repeat_probe:
        res = repeat_probe(args.matrix)
        for row in res["closed_pool"]:
            print("  {:10s} no_repeat={!s:5s} prior {:.2f}: won {:.3f}  repeated {:.3f}".format(
                row["backend"], row["no_repeat"], row["prior"], row["won"], row["repeated"]))
        cs = res["committed_stream_hard"]
        print("committed stream, hard: {} repeat games ({} in-pool -> expert), won {}, "
              "prior {:.3f}-{:.3f}".format(cs["repeat_games"], cs["repeat_in_pool_expert"],
                                         cs["won"], cs["prior_min"], cs["prior_max"]))
    elif args.hash_spread:
        res = hash_spread()
        for k, v in res["seed_mean_range"].items():
            print("  {:36s} {:.4f} .. {:.4f}".format(k, v["min"], v["max"]))
    if res is not None and args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
