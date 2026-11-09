"""ENV-704 (Phase A) — expert-onboarding cost in the open-world loop.

The §3.11 / ENV-702 loop admits a novel object for free: the expert is called and the
object's full Noul row appears in ``H_known``. This module makes admission cost
something and models the elicited row as *partial* — the expert supplies only ``E`` of
the 44 attributes (by a ``diagnostic`` or ``random`` policy); the rest are left at 0.5
(inert). The world still answers truthfully (the oracle uses the true matrix); only the
planner's belief uses the partial row, through the ``like_matrix`` seam in
``run_openworld_episode``.

It then measures, as ``E`` grows, how well recurrences of an admitted species resolve and
at what total cost (elicitation + repeat expert calls) — see ``OPENWORLD_ONBOARDING_
PREREG.md`` for the model and predictions, ``analyze_onboarding.py`` for the figures.

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.onboarding_openworld --self-test
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.onboarding_openworld --run --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
from typing import Dict, List, Sequence

from .analyze_openworld import auc
from .bed import ATTRIBUTE_BANK, Hypothesis, load_hypotheses, matrix_oracle
from .matrix import load_matrix
from .openworld import (GameRecord, fit_pitman_yor, make_stream, noisy_matrix_oracle,
                        py_p_new, run_openworld_episode)

ATTR_KEYS = [aid for aid, _ in ATTRIBUTE_BANK]
INERT = 0.5   # an un-elicited attribute: the planner does not know the object's answer


def elicit_row(true_matrix: Dict[str, Dict[str, float]], name: str, E: int,
               policy: str, rng: random.Random) -> Dict[str, float]:
    """Return the attribute row the expert supplies for ``name`` at effort ``E``: ``E``
    attributes at their true value (``diagnostic`` = most distinctive first, ``random`` =
    a random subset), the rest left inert (0.5)."""
    if E >= len(ATTR_KEYS):
        chosen = set(ATTR_KEYS)
    elif policy == "diagnostic":
        order = sorted(ATTR_KEYS, key=lambda k: -abs(true_matrix[k][name] - 0.5))
        chosen = set(order[:E])
    elif policy == "random":
        chosen = set(rng.sample(ATTR_KEYS, E))
    else:
        raise ValueError(policy)
    return {k: (true_matrix[k][name] if k in chosen else INERT) for k in ATTR_KEYS}


def run_onboarding(objs: Sequence[Hypothesis], true_matrix: Dict[str, Dict[str, float]],
                   stream: Sequence[int], sigma: float, theta: float, E: int,
                   policy: str, confidence: float = 0.9, eps: float = 0.05,
                   oracle_noise: float = 0.0, noise_seed: int = 1,
                   elicit_seed: int = 7) -> List[GameRecord]:
    """Play the stream with partial onboarding. ``H_known`` grows by expert admission as
    in §3.11, but each admitted object enters with an ``E``-attribute elicited row; the
    planner's belief uses that partial row while the oracle answers from the true
    matrix. No re-elicitation on a recurrence — a thin row that re-escalates simply pays
    the expert again (that cost is the point)."""
    if oracle_noise > 0.0:
        oracle = noisy_matrix_oracle(true_matrix, oracle_noise, noise_seed)
    else:
        oracle = matrix_oracle(true_matrix)
    rng = random.Random(elicit_seed)
    elicited: Dict[str, Dict[str, float]] = {k: {} for k in ATTR_KEYS}
    known: List[Hypothesis] = []
    known_set: set = set()
    records: List[GameRecord] = []

    for g, idx in enumerate(stream):
        true_h = objs[idx]
        in_pool = true_h.name in known_set       # a recurrence of an admitted species
        p_unknown_prior = py_p_new(len(known), g, sigma, theta)
        if len(known) == 0:
            p_unknown_prior = 1.0

        ep = run_openworld_episode(
            true_h, known, true_matrix, p_unknown_prior, oracle,
            confidence=confidence, eps=eps, backend="soft", like_matrix=elicited,
        )
        expert_called = ep.outcome == "expert"
        if expert_called and not in_pool:
            row = elicit_row(true_matrix, true_h.name, E, policy, rng)
            for k in ATTR_KEYS:
                elicited[k][true_h.name] = row[k]
            known.append(true_h)
            known_set.add(true_h.name)

        records.append(GameRecord(
            game=g, true_name=true_h.name, rank=0, in_pool=in_pool, is_new=not in_pool,
            outcome=ep.outcome, n_questions=ep.n_questions,
            p_unknown_max=ep.p_unknown_max, fire_turn=ep.fire_turn,
            k_before=len(known_set) - (1 if (expert_called and not in_pool) else 0),
            expert_called=expert_called, false_positive=expert_called and in_pool,
        ))
    return records


def metrics(records: List[GameRecord], E: int) -> Dict[str, float]:
    recur = [r for r in records if r.in_pool]                 # admitted species, again
    admitted = len({r.true_name for r in records if r.expert_called and not r.in_pool})
    rr = len(recur)
    won = sum(1 for r in recur if r.outcome == "won")
    reesc = sum(1 for r in recur if r.outcome == "expert")
    wrong = sum(1 for r in recur if r.outcome == "wrong")
    labels = [0 if r.in_pool else 1 for r in records]
    scores = [r.p_unknown_max for r in records]
    return {
        "admitted": admitted,
        "onboarding_attrs": E * admitted,
        "recur_games": rr,
        "recur_resolution_rate": won / rr if rr else float("nan"),
        "recur_reescalation_rate": reesc / rr if rr else float("nan"),
        "recur_wrong_rate": wrong / rr if rr else float("nan"),
        "recur_mean_questions": (sum(r.n_questions for r in recur) / rr
                                 if rr else float("nan")),
        "expert_calls": sum(1 for r in records if r.expert_called),
        "auc": auc(scores, labels),
    }


def total_cost(m: Dict[str, float], c_elicit: float, c_expert: float) -> float:
    return c_elicit * m["onboarding_attrs"] + c_expert * m["expert_calls"]


def run_grid(matrix_path: str, games: int, zipf_s: float, seed: int, confidence: float,
             eps: float, efforts: List[int], policies: List[str], levels: List[float],
             noise_seed: int, outdir: str) -> str:
    objs = load_hypotheses()
    true_matrix = load_matrix(matrix_path)["matrix"]
    stream, _ = make_stream(objs, games, zipf_s, seed)
    seen: set = set()
    flags = []
    for idx in stream:
        flags.append(objs[idx].name not in seen)
        seen.add(objs[idx].name)
    sigma, theta, _ = fit_pitman_yor(flags)

    cells = []
    print("policy      noise   E   resolve  reesc   wrong   q/recur  onbd_attrs  experts")
    for policy in policies:
        for lvl in levels:
            for E in efforts:
                recs = run_onboarding(objs, true_matrix, stream, sigma, theta, E, policy,
                                      confidence=confidence, eps=eps, oracle_noise=lvl,
                                      noise_seed=noise_seed)
                m = metrics(recs, E)
                cells.append({"policy": policy, "oracle_noise": lvl, "effort": E,
                              "metrics": m})
                print("{:10s} {:4.0f}%  {:2d}   {:.3f}   {:.3f}   {:.3f}   {:6.2f}   {:8d}   {:6d}".format(
                    policy, lvl * 100, E, m["recur_resolution_rate"],
                    m["recur_reescalation_rate"], m["recur_wrong_rate"],
                    m["recur_mean_questions"], m["onboarding_attrs"], m["expert_calls"]))

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(outdir, "openworld_onboarding_{}".format(stamp))
    os.makedirs(outpath, exist_ok=True)
    payload = {
        "schema": "twenty_questions_openworld_onboarding_v1",
        "created": stamp,
        "config": {"matrix": os.path.basename(matrix_path), "games": games,
                   "zipf_s": zipf_s, "seed": seed, "confidence": confidence, "eps": eps,
                   "efforts": efforts, "policies": policies, "levels": levels,
                   "n_attributes": len(ATTR_KEYS)},
        "pitman_yor_fit": {"sigma": sigma, "theta": theta},
        "cells": cells,
    }
    with open(os.path.join(outpath, "summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote {}".format(outpath))
    return outpath


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    """Partial-row semantics: an un-elicited attribute is inert, and a full row (E=44)
    reproduces the free-admission baseline (run_accumulation soft) bit-for-bit."""
    from .openworld import run_accumulation
    objs = load_hypotheses()
    from .matrix import synthetic_matrix
    matrix = synthetic_matrix(objs)
    stream, _ = make_stream(objs, 400, 1.1, seed=7)
    seen: set = set()
    flags = []
    for idx in stream:
        flags.append(objs[idx].name not in seen)
        seen.add(objs[idx].name)
    sigma, theta, _ = fit_pitman_yor(flags)

    print("1. elicit_row: E attributes true (most distinctive), the rest inert (0.5)")
    name = objs[0].name
    row = elicit_row(matrix, name, 8, "diagnostic", random.Random(0))
    picked = [k for k in ATTR_KEYS if row[k] != INERT]
    unpicked = [k for k in ATTR_KEYS if row[k] == INERT]
    assert len(picked) <= 8, len(picked)
    assert all(abs(row[k] - matrix[k][name]) < 1e-12 for k in picked), "picked != true"
    # diagnostic: every picked attribute is at least as distinctive as every unpicked.
    assert (not picked or not unpicked or
            min(abs(matrix[k][name] - 0.5) for k in picked)
            >= max(abs(matrix[k][name] - 0.5) for k in unpicked) - 1e-9), "not top-E"
    print("   E=8: {} true-valued, {} inert  OK".format(len(picked), len(unpicked)))

    print("2. full elicitation (E=44) == free-admission baseline, bit-for-bit")
    base = run_accumulation(objs, matrix, stream, sigma, theta, backend="soft")
    full = run_onboarding(objs, matrix, stream, sigma, theta, E=len(ATTR_KEYS),
                          policy="diagnostic")
    assert len(base) == len(full)
    mism = [(b.game, b.outcome, f.outcome) for b, f in zip(base, full)
            if b.outcome != f.outcome or b.n_questions != f.n_questions]
    assert not mism, mism[:5]
    print("   {} games identical (outcome + questions)  OK".format(len(base)))

    print("3. thin rows resolve recurrences worse than rich rows")
    m0 = metrics(run_onboarding(objs, matrix, stream, sigma, theta, 0, "diagnostic"), 0)
    m44 = metrics(run_onboarding(objs, matrix, stream, sigma, theta, len(ATTR_KEYS),
                                 "diagnostic"), len(ATTR_KEYS))
    print("   E=0  resolve={:.3f} q/recur={:.2f}".format(
        m0["recur_resolution_rate"], m0["recur_mean_questions"]))
    print("   E=44 resolve={:.3f} q/recur={:.2f}".format(
        m44["recur_resolution_rate"], m44["recur_mean_questions"]))
    assert m44["recur_resolution_rate"] >= m0["recur_resolution_rate"], "rich !>= thin"
    print("ALL ONBOARDING SELF-TESTS PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--efforts", default="0,2,4,8,16,44")
    ap.add_argument("--policies", default="diagnostic,random")
    ap.add_argument("--levels", default="0.0,0.10")
    ap.add_argument("--noise-seed", type=int, default=1)
    ap.add_argument("--out", default="llm_rl_scripts/twenty_questions/jev/results")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.run:
        run_grid(args.matrix, args.games, args.zipf, args.seed, args.confidence,
                 args.eps, [int(x) for x in args.efforts.split(",")],
                 args.policies.split(","), [float(x) for x in args.levels.split(",")],
                 args.noise_seed, args.out)
        return
    ap.error("pass --self-test or --run")


if __name__ == "__main__":
    main()
