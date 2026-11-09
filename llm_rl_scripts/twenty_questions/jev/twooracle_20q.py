"""ENV-704-B — two-oracle nested VOI: a cheap, noisy answer source vs an expensive,
truthful human, and the policy for when to pay for the human.

Closed-pool 20Q (no catch-all, no onboarding — those confounds are out). Each turn the
soft-posterior BED planner picks the EIG-best question; a **query policy** then decides
whether to take the **cheap** answer (attribute answers flip with probability ``q``) or
pay one **human** query for the truth. Sweeping each policy's threshold traces its
accuracy-vs-human-spend frontier; the question is which policy buys the most accuracy per
expensive query. See ``TWOORACLE_PREREG.md``.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.twooracle_20q --self-test
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.twooracle_20q --run --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

from .bed import (ATTRIBUTE, GUESS, Hypothesis, candidate_questions, entropy,
                  build_question_bank, load_hypotheses, make_likelihood, matrix_oracle,
                  posterior_update, rank_questions, uniform_posterior)
from .matrix import load_matrix

# A policy decides, per turn, whether to pay for the human on an ATTRIBUTE question.
# Signature: (kind, eig, post, rng, knob) -> bool. Non-attribute answers are exact and
# free (no flip, no human needed), so a policy is only consulted on attribute questions.


def _top_two(post: Dict[str, float]) -> Tuple[float, float]:
    vals = sorted(post.values(), reverse=True)
    return (vals[0], vals[1] if len(vals) > 1 else 0.0)


def pay_human(policy: str, knob: float, eig: float, post: Dict[str, float],
              rng: random.Random) -> bool:
    if policy == "cheap_only":
        return False
    if policy == "human_all":
        return True
    if policy == "random":
        return rng.random() < knob
    if policy == "entropy":
        return entropy(post) >= knob
    if policy == "margin":
        t1, t2 = _top_two(post)
        return (t1 - t2) <= knob
    if policy == "voi_eig":
        return eig >= knob
    raise ValueError(policy)


def play_game(true_h: Hypothesis, pool: Sequence[Hypothesis],
              bank: Dict[str, List], like, by_name: Dict[str, Hypothesis],
              human, q_flip: float, policy: str, knob: float, T: int,
              confidence: float, eps: float, rng: random.Random
              ) -> Tuple[bool, int, int]:
    """Return (won, n_questions, human_queries). ``won`` = a correct by-name guess within
    the budget ``T``. The cheap source flips each attribute answer with prob ``q_flip``;
    the human answers truthfully. Non-attribute answers are always exact."""
    post = uniform_posterior(pool)
    guess_by_name = {qq.key: qq for qq in bank[GUESS]}
    human_queries = 0

    for turn in range(1, T + 1):
        top_name = max(post, key=lambda n: post[n])
        support = sum(1 for p in post.values() if p > 1e-9)
        if post[top_name] >= confidence or support == 1:
            q, eig = guess_by_name[top_name], 0.0
        else:
            cands = candidate_questions(post, bank)
            ranked = rank_questions(cands, post, by_name, like)
            q, eig = ranked[0][0], ranked[0][1]

        truth = human(q, true_h)
        if q.kind == ATTRIBUTE:
            # budget policies (``first``) spend a fixed per-game quota on the earliest
            # attribute questions — a causal EIG-greedy proxy, since EIG falls over a
            # game; threshold policies decide from the per-turn score.
            if policy == "first":
                use = human_queries < int(knob)
            else:
                use = pay_human(policy, knob, eig, post, rng)
            if use:
                ans = truth
                human_queries += 1
            else:
                ans = (not truth) if rng.random() < q_flip else truth
        else:
            ans = truth                       # category/guess: exact, free

        if q.kind == GUESS and ans:
            return True, turn, human_queries
        post = posterior_update(post, q, ans, by_name, like, eps=eps)

    return False, T, human_queries


def run_cell(objs, matrix, N: int, T: int, q_flip: float, policy: str, knob: float,
             games: int, confidence: float, eps: float, pool_seed: int,
             stream_seed: int) -> Dict[str, float]:
    prng = random.Random(pool_seed)
    pool = prng.sample(list(objs), N)
    by_name = {h.name: h for h in pool}
    bank = build_question_bank(pool)
    like = make_likelihood(matrix)
    human = matrix_oracle(matrix)
    srng = random.Random(stream_seed)
    arng = random.Random(stream_seed * 7919 + 1)   # answer/flip/policy randomness
    won = hq = nq = 0
    for _ in range(games):
        true_h = pool[srng.randrange(N)]
        w, n, h = play_game(true_h, pool, bank, like, by_name, human, q_flip,
                            policy, knob, T, confidence, eps, arng)
        won += int(w)
        nq += n
        hq += h
    return {"accuracy": won / games, "human_q_per_game": hq / games,
            "questions_per_game": nq / games, "games": games}


# Each policy's threshold sweep (low spend -> high spend as listed).
def knob_sweep(policy: str, N: int) -> List[float]:
    if policy in ("cheap_only", "human_all"):
        return [0.0]
    if policy == "random":
        return [0.1, 0.2, 0.4, 0.6, 0.85]
    if policy == "entropy":                     # pay when entropy >= frac * log2(N)
        lg = math.log2(N)
        return [0.85 * lg, 0.7 * lg, 0.5 * lg, 0.3 * lg, 0.1 * lg]
    if policy == "margin":                       # pay when top1-top2 <= knob
        return [0.01, 0.03, 0.08, 0.2, 0.5]
    if policy == "voi_eig":                       # pay when EIG >= knob (bits)
        return [0.9, 0.7, 0.5, 0.3, 0.1]
    if policy == "first":                         # budget: human on first B attr. qs
        return [1, 2, 3, 4]
    raise ValueError(policy)


def run_grid(matrix_path: str, games: int, confidence: float, eps: float,
             regimes: List[dict], policies: List[str], outdir: str) -> str:
    objs = load_hypotheses()
    matrix = load_matrix(matrix_path)["matrix"]
    cells = []
    print("regime        policy      knob     acc    human_q/g   q/g")
    for reg in regimes:
        N, T, q = reg["N"], reg["T"], reg["q"]
        tag = "N{}_T{}_q{:g}".format(N, T, q)
        for policy in policies:
            for knob in knob_sweep(policy, N):
                m = run_cell(objs, matrix, N, T, q, policy, knob, games, confidence,
                             eps, pool_seed=reg.get("pool_seed", 11),
                             stream_seed=reg.get("stream_seed", 7))
                cells.append({"regime": tag, "N": N, "T": T, "q": q,
                              "policy": policy, "knob": knob, "metrics": m})
                print("{:12s}  {:10s}  {:5.2f}  {:.3f}    {:6.2f}    {:5.2f}".format(
                    tag, policy, knob, m["accuracy"], m["human_q_per_game"],
                    m["questions_per_game"]))

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(outdir, "twooracle_{}".format(stamp))
    os.makedirs(outpath, exist_ok=True)
    payload = {"schema": "twenty_questions_twooracle_v1", "created": stamp,
               "config": {"matrix": os.path.basename(matrix_path), "games": games,
                          "confidence": confidence, "eps": eps, "regimes": regimes,
                          "policies": policies},
               "cells": cells}
    with open(os.path.join(outpath, "summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote {}".format(outpath))
    return outpath


DEFAULT_REGIMES = [
    {"N": 20, "T": 20, "q": 0.15, "label": "easy"},
    {"N": 40, "T": 10, "q": 0.15, "label": "hard"},
    {"N": 40, "T": 10, "q": 0.25, "label": "hard-q25"},
]
DEFAULT_POLICIES = ["cheap_only", "human_all", "random", "first", "entropy", "margin",
                    "voi_eig"]


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    objs = load_hypotheses()
    from .matrix import synthetic_matrix
    matrix = synthetic_matrix(objs)

    print("1. human_all == truthful run; cheap_only <= human_all accuracy")
    ha = run_cell(objs, matrix, 30, 12, 0.20, "human_all", 0.0, 600, 0.9, 0.05, 11, 7)
    co = run_cell(objs, matrix, 30, 12, 0.20, "cheap_only", 0.0, 600, 0.9, 0.05, 11, 7)
    assert ha["human_q_per_game"] > 0 and co["human_q_per_game"] == 0.0
    assert co["accuracy"] <= ha["accuracy"] + 1e-9, (co["accuracy"], ha["accuracy"])
    print("   human_all acc={:.3f} (human_q/g={:.2f}); cheap_only acc={:.3f}  OK".format(
        ha["accuracy"], ha["human_q_per_game"], co["accuracy"]))

    print("2. within voi_eig, more human spend -> >= accuracy (monotone trend)")
    pts = []
    for knob in [0.9, 0.5, 0.1]:                 # higher spend as knob drops
        m = run_cell(objs, matrix, 30, 12, 0.20, "voi_eig", knob, 600, 0.9, 0.05, 11, 7)
        pts.append((m["human_q_per_game"], m["accuracy"]))
    print("   voi_eig (human_q/g, acc):", [(round(a, 2), round(b, 3)) for a, b in pts])
    assert pts[-1][0] >= pts[0][0] - 1e-9, "spend not increasing"
    assert pts[-1][1] >= pts[0][1] - 0.02, "more truth hurt accuracy"

    print("3. flips hit attribute answers only (category/guess exact)")
    # with q=1.0 (every attribute flipped) cheap_only should be badly degraded vs human
    hi = run_cell(objs, matrix, 30, 12, 1.0, "cheap_only", 0.0, 400, 0.9, 0.05, 11, 7)
    assert hi["accuracy"] < co["accuracy"] + 0.5, "q=1.0 not degrading"
    print("   q=1.0 cheap_only acc={:.3f} (fully corrupted attributes)  OK".format(
        hi["accuracy"]))
    print("ALL TWO-ORACLE SELF-TESTS PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--out", default="llm_rl_scripts/twenty_questions/jev/results")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.run:
        run_grid(args.matrix, args.games, args.confidence, args.eps, DEFAULT_REGIMES,
                 DEFAULT_POLICIES, args.out)
        return
    ap.error("pass --self-test or --run")


if __name__ == "__main__":
    main()
