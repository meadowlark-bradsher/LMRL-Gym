"""Run Condition B episodes for one or more arms and stream exact per-decision logs.

Every (learner role, learner type, opponent type) cell is enumerated — 2 x 6 x 6
= 72 cells — and repeated ``--repeats`` times (stochastic arms only need > 1).
For every decision the record stores the full annotated menu (EIG, P(accept),
own utility), the exact Q* of every option, the chosen option, the outcome, and
the two regrets (EIG regret in bits, decision regret in utility) plus the
best-minus-second-best Q gap. Nothing in the record depends on floats except
the EIG values; probabilities, utilities and Q are stored as exact Fractions
(strings) with float companions.

Arms: random, greedy_eig, myopic, exact (``arms.py``); jev_plain, jev_eig
(``jev_policy.py``; ``--jev-backend mock`` exercises the code path offline,
``--jev-backend http`` needs TYPESAFE_API_KEY and is gated by the
pre-registration note).

Example (scripted arms, all cells, random repeated 20x):
    PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_binned.run_condition_b \
        --arms exact,greedy_eig,myopic,random --repeats 20 --seed 0 \
        --out results/binned_b_scripted_20261026
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_discrete.scoring import analyze_type_pair, nash_product

from . import beliefs
from .arms import ARMS
from .env import (
    ACCEPT,
    ASK,
    OFFER,
    Config,
    Option,
    ScriptedOpponent,
    State,
    annotate,
    initial_state,
    menu,
    option_likelihoods,
    other_role,
    transition,
)
from .jev_policy import JevConditionBPolicy
from .reference import ExactReference, regrets

JEV_ARMS = {"jev_plain": False, "jev_eig": True}


def _f(x: Optional[Fraction]):
    return None if x is None else {"exact": str(x), "float": float(x)}


def _row_json(row: dict) -> dict:
    out = dict(row)
    for k in ("p_accept", "own_utility", "myopic_value"):
        if k in out:
            out[k] = _f(out[k])
    return out


def _choose_with_retry(jev, state, rows, max_retries: int, backoff: float) -> str:
    """Retry transient Jev failures; re-raise after ``max_retries`` so the
    runner can write an error stub and continue with the next episode."""
    from LLM_RL.jev.client import JevError

    for attempt in range(1, max_retries + 1):
        try:
            return jev.choose(state, rows)
        except JevError as e:
            if attempt == max_retries:
                raise
            print(f"  jev error (attempt {attempt}/{max_retries}): {e}; retrying", file=sys.stderr)
            time.sleep(backoff * attempt)
    raise RuntimeError("unreachable")


def play_episode(
    cfg: Config,
    arm: str,
    role: str,
    own,
    opp,
    ref: ExactReference,
    rng: random.Random,
    jev: Optional[JevConditionBPolicy],
    repeat: int,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> dict:
    state = initial_state(role, own)
    opponent = ScriptedOpponent(other_role(role), opp, cfg, random.Random(rng.random()))
    decisions: List[dict] = []
    entropies: List[float] = []
    learner_u: Optional[Fraction] = None
    accepted: Optional[str] = None
    n_q = n_o = 0
    while True:
        rows = annotate(state, cfg)
        if not rows:
            raise RuntimeError("empty menu before termination")
        qvals = ref.q_values(state)
        h = beliefs.entropy_bits(state.belief)
        entropies.append(h)
        info = None
        if arm in JEV_ARMS:
            assert jev is not None
            key = _choose_with_retry(jev, state, rows, max_retries, retry_backoff)
            info = jev.last_info
        else:
            key = ARMS[arm](state, rows, qvals, rng)
        opt = next(o for o in menu(state, cfg) if o.key == key)
        outcome = opponent.respond(opt, role)
        max_eig = max(r["eig_bits"] for r in rows)
        chosen_row = next(r for r in rows if r["key"] == key)
        d_regret, gap = regrets(qvals, key)
        decisions.append(
            {
                "t": state.t,
                "entropy_before_bits": h,
                "support_size": len(beliefs.support(state.belief)),
                "menu": [_row_json(r) for r in rows],
                "q_values": {k: _f(v) for k, v in qvals.items()},
                "chosen": key,
                "chosen_kind": opt.kind,
                "chosen_tag": opt.tag,
                "chosen_position": rows.index(chosen_row),
                "outcome": outcome,
                "eig_chosen_bits": chosen_row["eig_bits"],
                "eig_regret_bits": max_eig - chosen_row["eig_bits"],
                "decision_regret": _f(d_regret),
                "q_gap_best_second": _f(gap),
                "jev": info,
            }
        )
        if opt.kind == ASK:
            n_q += 1
        else:
            n_o += 1
        nxt, terminal = transition(state, opt, outcome, cfg)
        if nxt is None:
            learner_u = terminal
            if opt.kind == OFFER and outcome == ACCEPT:
                accepted = opt.target
            # posterior given the terminal outcome (acceptance is informative too)
            final_belief = beliefs.update(state.belief, option_likelihoods(opt, state, cfg)[outcome])
            break
        state = nxt
    entropies.append(beliefs.entropy_bits(final_belief))

    # evaluator-only scoring against the exact truth table
    r = cfg.reservation
    b_rank, s_rank = (own, opp) if role == "buyer" else (opp, own)
    analysis = analyze_type_pair(b_rank, s_rank, r)
    u_star = analysis.buyer_u_star if role == "buyer" else analysis.seller_u_star
    gain = None
    if u_star is not None and u_star != r:
        gain = (learner_u - r) / (u_star - r)
    nash_ratio = None
    pareto = None
    if analysis.is_feasible and analysis.nash_max:
        if accepted is not None:
            d = DEALS_BY_ID[accepted]
            nash_ratio = nash_product(d, b_rank, s_rank, r) / analysis.nash_max
            pareto = d in set(analysis.pareto_frontier)
        else:
            nash_ratio = Fraction(0)
    opp_u = utility(other_role(role), opp, DEALS_BY_ID[accepted]) if accepted else r
    return {
        "arm": arm,
        "role": role,
        "own_ranking": list(own),
        "repeat": repeat,
        "private_truth_evaluator_only": {"opp_ranking": list(opp)},
        "decisions": decisions,
        "n_decisions": len(decisions),
        "questions_asked": n_q,
        "offers_made": n_o,
        "accepted_deal_id": accepted,
        "agreement": accepted is not None,
        "learner_utility": _f(learner_u),
        "opponent_utility": _f(opp_u),
        "reference_prior_value": _f(ref.prior_value()),
        "attainable_u_star": _f(u_star),
        "feasible_pair": analysis.is_feasible,
        "gain": _f(gain),
        "nash_ratio": _f(nash_ratio),
        "on_pareto_frontier": pareto,
        "entropy_trajectory_bits": entropies,
        "cumulative_entropy_bits": sum(entropies),
        "final_entropy_bits": entropies[-1],
        "total_eig_regret_bits": sum(d["eig_regret_bits"] for d in decisions),
        "total_decision_regret": _f(sum((Fraction(d["decision_regret"]["exact"]) for d in decisions), Fraction(0))),
        "zero_regret_decisions": sum(1 for d in decisions if d["decision_regret"]["exact"] == "0"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", default="exact,greedy_eig,myopic,random")
    p.add_argument("--roles", default="buyer,seller")
    p.add_argument("--regime", default="easy", choices=("easy", "hard"))
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--answer-noise", default="0", help="Fraction, e.g. 1/10")
    p.add_argument("--accept-noise", default="0", help="Fraction, e.g. 1/20")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per cell for stochastic arms (random, jev_*); deterministic arms play once.")
    p.add_argument("--seed", type=int, default=0, help="Seeds scripted randomness and option shuffling; provenance only for jev-latest.")
    p.add_argument("--out", required=True)
    p.add_argument("--jev-backend", default="mock", choices=("mock", "http"))
    p.add_argument("--no-shuffle", action="store_true", help="Present Jev options in menu order.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-backoff", type=float, default=2.0, help="Seconds; multiplied by the attempt number.")
    p.add_argument("--cells", default="all", help="'all' or 'own_idx,opp_idx' pairs like '0:1;2:3' (indices into ALL_RANKINGS).")
    args = p.parse_args(argv)

    cfg = Config(
        regime=args.regime,
        rounds=args.rounds,
        answer_noise=Fraction(args.answer_noise),
        accept_noise=Fraction(args.accept_noise),
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS and a not in JEV_ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(ARMS)} + {sorted(JEV_ARMS)}")
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    if args.cells == "all":
        cells = [(i, j) for i in range(len(ALL_RANKINGS)) for j in range(len(ALL_RANKINGS))]
    else:
        cells = [tuple(int(x) for x in c.split(":")) for c in args.cells.split(";")]

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    jev_client = None
    if any(a in JEV_ARMS for a in arms):
        backend = MockJevBackend() if args.jev_backend == "mock" else HttpJevBackend()
        jev_client = JevClient(backend)

    refs: Dict[tuple, ExactReference] = {}
    full_values: Dict[str, dict] = {}
    for role in roles:
        for own in ALL_RANKINGS:
            refs[(role, own)] = ExactReference(cfg, role, own)
            full = ExactReference(cfg, role, own, full_menu=True)
            full_values[f"{role}/{'/'.join(a[0] for a in own)}"] = {
                "menu": _f(refs[(role, own)].prior_value()),
                "unconstrained": _f(full.prior_value()),
            }

    n = 0
    errors = 0
    t0 = time.time()
    path = os.path.join(args.out, "runs.jsonl")
    with open(path, "w") as f:
        for arm in arms:
            jev = None
            if arm in JEV_ARMS and jev_client is not None:
                jev = JevConditionBPolicy(
                    jev_client, cfg, show_eig=JEV_ARMS[arm],
                    shuffle_rng=None if args.no_shuffle else random.Random(rng.random()),
                )
            for role in roles:
                for i, j in cells:
                    own, opp = ALL_RANKINGS[i], ALL_RANKINGS[j]
                    # deterministic arms under zero noise are played once per cell
                    deterministic = arm in ARMS and arm != "random" and cfg.answer_noise == 0 and cfg.accept_noise == 0
                    for k in range(1 if deterministic else args.repeats):
                        try:
                            rec = play_episode(cfg, arm, role, own, opp, refs[(role, own)], rng, jev, k,
                                               args.max_retries, args.retry_backoff)
                        except Exception as e:  # error stub: carries no result, analyzers skip it
                            rec = {"arm": arm, "role": role, "own_ranking": list(own), "repeat": k,
                                   "private_truth_evaluator_only": {"opp_ranking": list(opp)},
                                   "error": f"{type(e).__name__}: {e}"}
                            errors += 1
                            print(f"  episode failed after retries: {rec['error']}", file=sys.stderr)
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        n += 1
            print(f"[{time.time()-t0:6.1f}s] arm={arm} done ({n} episodes so far)", file=sys.stderr)

    summary = {
        "config": cfg.to_json(),
        "arms": arms,
        "roles": roles,
        "cells": len(cells),
        "repeats": args.repeats,
        "seed": args.seed,
        "jev_backend": args.jev_backend if jev_client else None,
        "jev_shuffle_options": not args.no_shuffle,
        "episodes": n,
        "error_stubs": errors,
        "reference_values": full_values,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "reference_values"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
