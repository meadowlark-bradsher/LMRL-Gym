"""Phase 1 runner for the live Car-Dealer: focal arms against the T1 kernel in one
(H, c_q, kernel) regime, with exact per-decision annotations.

Cells: every (focal role, opener, focal type, opponent type) — 2 x 2 x 6 x 6 = 144 —
repeated ``--repeats`` times (the opponent is stochastic, so every arm repeats);
the design's K = 5 gives 720 episodes per arm. Episode randomness is seeded from
(seed, cell, repeat) so arms share random streams where the play allows.

Per decision the record stores the menu the arm saw (exact EIG, P(accept), own
utility), the exact Q* of every ask-menu option under the menu-restricted oracle
with ASK allowed (C_plan) and of every non-ASK option with ASK forbidden (B_plan),
whether ASK was strictly optimal in that state, the chosen option and its
regrets. Arms: exact_ask, exact_noask, greedy_eig, random (``arms_live``);
jev_ask_plain, jev_ask_eig, jev_noask_plain, jev_noask_eig (``jev_policy_live``;
``--jev-backend mock`` runs offline; ``http`` needs TYPESAFE_API_KEY and is gated
by the pre-registration).

Ceilings written to ``ceilings.json`` per (role, opener, focal type): the root
values of six exact oracles — full menu with/without ASK, shortlist menu
with/without ASK, offers-restricted only, questions-restricted only — so the
shortlist's cost is attributed to each restriction (the costlessness check).
"""

from __future__ import annotations

import argparse
import gzip
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

from . import belief_live as bl
from .arms_live import arms_for
from .jev_policy_live import JevLivePolicy
from .menu_live import Menu
from .opponent import KernelParams, T1Kernel
from .oracle import EPS, Oracle
from .protocol_live import ASK, PROPOSE, ENV_VERSION, Action, LiveConfig, apply, asks_by, new_game, other

JEV_ARMS = {"jev_ask_plain": (True, False), "jev_ask_eig": (True, True),
            "jev_noask_plain": (False, False), "jev_noask_eig": (False, True)}
NOASK_ARMS = {"exact_noask", "jev_noask_plain", "jev_noask_eig"}


def _f(x):
    return None if x is None else {"exact": str(x), "float": float(x)}


def _tag(r):
    return "/".join(a[0] for a in r)


class Cell:
    """Everything exact for one (focal role, opener, focal type): menus and oracles."""

    def __init__(self, cfg: LiveConfig, params: KernelParams, focal: str, ranking, menu_mode: str, n_offers: int = 3, n_questions: int = 2):
        self.cfg, self.focal, self.ranking = cfg, focal, ranking
        self.kernel = T1Kernel(other(focal), cfg, params)
        full = (menu_mode == "full")
        self.n_offers, self.n_questions = n_offers, n_questions
        self.full_noask = Oracle(cfg, focal, ranking, self.kernel, ask_allowed=False)   # menu-free; defines PLAN
        self.full_ask = Oracle(cfg, focal, ranking, self.kernel, ask_allowed=True)
        self.menu = Menu(cfg, focal, ranking, self.kernel, restrict_offers=not full, restrict_questions=not full,
                         n_offers=n_offers, n_questions=n_questions, plan_oracle=self.full_noask, plan_ask_oracle=self.full_ask)
        self.ask = Oracle(cfg, focal, ranking, self.kernel, ask_allowed=True, menu=self.menu.actions(True))
        self.noask = Oracle(cfg, focal, ranking, self.kernel, ask_allowed=False, menu=self.menu.actions(False))

    def ceilings(self, opener: str) -> dict:
        out = {"menu_ask": self.ask.root_value(opener), "menu_noask": self.noask.root_value(opener)}
        out["full_ask"] = self.full_ask.root_value(opener)
        out["full_noask"] = self.full_noask.root_value(opener)
        if self.menu.restrict_offers or self.menu.restrict_questions:
            m_off = Menu(self.cfg, self.focal, self.ranking, self.kernel, restrict_offers=True, restrict_questions=False,
                         n_offers=self.n_offers, n_questions=self.n_questions, plan_oracle=self.full_noask, plan_ask_oracle=self.full_ask)
            m_q = Menu(self.cfg, self.focal, self.ranking, self.kernel, restrict_offers=False, restrict_questions=True,
                         n_offers=self.n_offers, n_questions=self.n_questions, plan_oracle=self.full_noask, plan_ask_oracle=self.full_ask)
            out["offers_restricted_ask"] = Oracle(self.cfg, self.focal, self.ranking, self.kernel, True, menu=m_off.actions(True)).root_value(opener)
            out["questions_restricted_ask"] = Oracle(self.cfg, self.focal, self.ranking, self.kernel, True, menu=m_q.actions(True)).root_value(opener)
        out["delta_full"] = out["full_ask"] - out["full_noask"]
        out["delta_menu"] = out["menu_ask"] - out["menu_noask"]
        out["loss_ask"] = out["full_ask"] - out["menu_ask"]
        out["loss_noask"] = out["full_noask"] - out["menu_noask"]
        return out


def play_episode(cell: Cell, arm: str, opener: str, opp_ranking, rng: random.Random, jev: Optional[JevLivePolicy],
                 arms: dict, repeat: int, max_retries: int, backoff: float) -> dict:
    cfg, focal = cell.cfg, cell.focal
    state = new_game(cfg, opener)
    belief = bl.uniform()
    decisions: List[dict] = []
    entropies: List[float] = []
    ask_arm = arm not in NOASK_ARMS
    while state.terminal is None:
        if state.turn != focal:
            act = cell.kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(cell.kernel, state)[act]
            state = apply(state, act)
            belief = bl.posterior(belief, lik)
            continue
        rows_ask = cell.menu.rows(state, belief, include_questions=True)
        rows = rows_ask if ask_arm else cell.menu.rows(state, belief, include_questions=False)
        q_ask = {r.key: cell.ask.q_value(state, belief, r.action) for r in rows_ask}
        q_noask = {r.key: cell.noask.q_value(state, belief, r.action) for r in rows_ask if r.action.kind != ASK}
        ask_q = [q_ask[r.key] for r in rows_ask if r.action.kind == ASK]
        non_ask_q = [q_ask[r.key] for r in rows_ask if r.action.kind != ASK]
        ask_optimal = bool(ask_q) and max(ask_q) > max(non_ask_q) + EPS
        h = bl.entropy_bits(belief)
        entropies.append(h)
        info = None
        if arm in JEV_ARMS:
            key = _choose_with_retry(jev, state, rows, max_retries, backoff)
            info = jev.last_info
        else:
            key = arms[arm](rows, q_ask, q_noask, rng).key
        chosen = next(r for r in rows if r.key == key)
        ref = q_ask if ask_arm else q_noask
        best = max(ref[r.key] for r in rows)
        srt = sorted((ref[r.key] for r in rows), reverse=True)
        max_eig = max(r.eig_bits for r in rows)
        decisions.append({
            "remaining": state.remaining, "facing_offer": bool(state.standing and state.standing.proposer != focal),
            "own_offer_standing": bool(state.standing and state.standing.proposer == focal), "at_deadline": state.at_deadline,
            "entropy_before_bits": h, "support_size": len(bl.support(belief)), "belief": list(belief),
            "menu": [r.to_json() for r in rows],
            "q_ask": q_ask, "q_noask": q_noask, "ask_legal": bool(ask_q), "ask_optimal": ask_optimal,
            "best_ask_minus_best_nonask": (max(ask_q) - max(non_ask_q)) if ask_q else None,
            "chosen": key, "chosen_kind": chosen.action.kind, "chosen_tag": chosen.tag, "chosen_position": rows.index(chosen),
            "eig_chosen_bits": chosen.eig_bits, "eig_regret_bits": max_eig - chosen.eig_bits,
            "decision_regret": best - ref[key], "q_gap_best_second": (srt[0] - srt[1]) if len(srt) > 1 else None,
            "jev": info,
        })
        act = chosen.action
        if act.kind == ASK:
            ans = cell.kernel.sample_answer(act.question_id, opp_ranking, rng)
            lik = bl.answer_likelihoods(cell.kernel, act.question_id)[ans]
            state = apply(state, act, ans)
            belief = bl.posterior(belief, lik)
        else:
            state = apply(state, act)
    entropies.append(bl.entropy_bits(belief))
    t = state.terminal
    r = cfg.reservation
    deal = DEALS_BY_ID[t.deal_id] if t.deal_id else None
    u = utility(focal, cell.ranking, deal) if deal else r
    gross = u - r
    n_ask = asks_by(state, focal)
    net = gross - cfg.ask_cost * n_ask
    u_max = max(utility(focal, cell.ranking, d) for d in DEALS_BY_ID.values())
    assert u_max > r
    y = net / (u_max - r)
    b_rank, s_rank = (cell.ranking, opp_ranking) if focal == "buyer" else (opp_ranking, cell.ranking)
    analysis = analyze_type_pair(b_rank, s_rank, r)
    nash_ratio = pareto = None
    if analysis.is_feasible and analysis.nash_max:
        if deal is not None:
            nash_ratio = nash_product(deal, b_rank, s_rank, r) / analysis.nash_max
            pareto = deal in set(analysis.pareto_frontier)
        else:
            nash_ratio = Fraction(0)
    opp_u = utility(other(focal), opp_ranking, deal) if deal else r
    return {
        "arm": arm, "focal": focal, "opener": opener, "focal_ranking": list(cell.ranking), "repeat": repeat,
        "private_truth_evaluator_only": {"opp_ranking": list(opp_ranking)},
        "decisions": decisions, "n_decisions": len(decisions),
        "questions_asked": n_ask, "proposals_made": sum(1 for e in state.events if e.actor == focal and e.action.kind == PROPOSE),
        "opponent_questions": asks_by(state, other(focal)),
        "terminal": {"deal_id": t.deal_id, "reason": t.reason}, "agreement": deal is not None,
        "focal_utility": _f(u), "opponent_utility": _f(opp_u), "gross_gain": _f(gross), "net_gain": _f(net),
        "normalized_net_gain": _f(y), "u_max": _f(u_max), "feasible_pair": analysis.is_feasible,
        "nash_ratio": _f(nash_ratio), "on_pareto_frontier": pareto,
        "entropy_trajectory_bits": entropies, "final_entropy_bits": entropies[-1],
        "total_decision_regret": sum(d["decision_regret"] for d in decisions),
        "zero_regret_decisions": sum(1 for d in decisions if d["decision_regret"] <= EPS),
        "total_eig_regret_bits": sum(d["eig_regret_bits"] for d in decisions),
        "asks_when_ask_optimal": sum(1 for d in decisions if d["ask_optimal"] and d["chosen_kind"] == ASK),
        "decisions_ask_optimal": sum(1 for d in decisions if d["ask_optimal"]),
        "asks_when_ask_not_optimal": sum(1 for d in decisions if d["ask_legal"] and not d["ask_optimal"] and d["chosen_kind"] == ASK),
        "decisions_ask_legal_not_optimal": sum(1 for d in decisions if d["ask_legal"] and not d["ask_optimal"]),
    }


def _choose_with_retry(jev, state, rows, max_retries: int, backoff: float) -> str:
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", default="exact_ask,exact_noask,greedy_eig,random")
    p.add_argument("--regime", default="easy", choices=("easy", "hard"))
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--ask-cost", default="0", help="Fraction, e.g. 3/100")
    p.add_argument("--alpha0", default="4/5"); p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--p-counter", type=float, default=1.0); p.add_argument("--p-hold", type=float, default=1.0)
    p.add_argument("--shortlist", type=int, default=3)
    p.add_argument("--menu", default="shortlist", choices=("shortlist", "full"))
    p.add_argument("--n-offers", type=int, default=3, help="offer candidates in the shortlist (SAFE, AMBITIOUS, PLAN, then BALANCED fill)")
    p.add_argument("--n-questions", type=int, default=2)
    p.add_argument("--roles", default="buyer,seller"); p.add_argument("--openers", default="buyer,seller")
    p.add_argument("--types", default="all", help="'all' or focal type indices '0,3'")
    p.add_argument("--opp-types", default="all", help="'all' or opponent type indices")
    p.add_argument("--repeats", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--jev-backend", default="mock", choices=("mock", "http"))
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--max-retries", type=int, default=3); p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--skip-ceilings", action="store_true", help="Do not solve the six ceiling oracles per cell (tests).")
    args = p.parse_args(argv)

    cfg = LiveConfig(regime=args.regime, horizon=args.horizon, ask_cost=Fraction(args.ask_cost))
    params = KernelParams(alpha0=Fraction(args.alpha0), beta=args.beta, p_counter=args.p_counter, p_hold=args.p_hold, shortlist=args.shortlist)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scripted = arms_for(float(cfg.reservation))
    for a in arms:
        if a not in scripted and a not in JEV_ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(scripted)} + {sorted(JEV_ARMS)}")
    roles = args.roles.split(","); openers = args.openers.split(",")
    types = list(range(len(ALL_RANKINGS))) if args.types == "all" else [int(x) for x in args.types.split(",")]
    opp_types = list(range(len(ALL_RANKINGS))) if args.opp_types == "all" else [int(x) for x in args.opp_types.split(",")]
    os.makedirs(args.out, exist_ok=True)
    jev_client = None
    if any(a in JEV_ARMS for a in arms):
        jev_client = JevClient(MockJevBackend() if args.jev_backend == "mock" else HttpJevBackend())

    t0 = time.time()
    cells: Dict[tuple, Cell] = {}
    ceilings: Dict[str, dict] = {}
    for role in roles:
        for i in types:
            c = Cell(cfg, params, role, ALL_RANKINGS[i], args.menu, args.n_offers, args.n_questions)
            cells[(role, i)] = c
            for opener in openers:
                if not args.skip_ceilings:
                    ceilings[f"{role}|{opener}|{_tag(ALL_RANKINGS[i])}"] = c.ceilings(opener)
                    print(f"[{time.time()-t0:6.1f}s] ceilings {role} opener={opener} type={_tag(ALL_RANKINGS[i])}: "
                          f"Δfull={ceilings[f'{role}|{opener}|{_tag(ALL_RANKINGS[i])}']['delta_full']:.4f} "
                          f"Δmenu={ceilings[f'{role}|{opener}|{_tag(ALL_RANKINGS[i])}']['delta_menu']:.4f}", file=sys.stderr)
    with open(os.path.join(args.out, "ceilings.json"), "w") as f:
        json.dump({"config": cfg.to_json(), "kernel": params.to_json(), "menu": args.menu, "n_offers": args.n_offers,
                   "n_questions": args.n_questions, "cells": ceilings}, f, indent=1)

    n = errors = 0
    path = os.path.join(args.out, "runs.jsonl.gz")
    shuffle_master = random.Random(args.seed)
    with gzip.open(path, "wt") as f:
        for arm in arms:
            for role in roles:
                for i in types:
                    cell = cells[(role, i)]
                    jev = None
                    if arm in JEV_ARMS:
                        jev = JevLivePolicy(jev_client, cfg, role, ALL_RANKINGS[i], show_eig=JEV_ARMS[arm][1],
                                            shuffle_rng=None if args.no_shuffle else random.Random(shuffle_master.random()))
                    for opener in openers:
                        for j in opp_types:
                            for k in range(args.repeats):
                                rng = random.Random(f"{args.seed}|{role}|{opener}|{i}|{j}|{k}")
                                try:
                                    rec = play_episode(cell, arm, opener, ALL_RANKINGS[j], rng, jev, scripted, k, args.max_retries, args.retry_backoff)
                                except Exception as e:
                                    rec = {"arm": arm, "focal": role, "opener": opener, "focal_ranking": list(ALL_RANKINGS[i]), "repeat": k,
                                           "private_truth_evaluator_only": {"opp_ranking": list(ALL_RANKINGS[j])}, "error": f"{type(e).__name__}: {e}"}
                                    errors += 1
                                    print(f"  episode failed: {rec['error']}", file=sys.stderr)
                                f.write(json.dumps(rec) + "\n")
                                n += 1
            print(f"[{time.time()-t0:6.1f}s] arm={arm} done ({n} episodes so far)", file=sys.stderr)
    summary = {"env": ENV_VERSION, "config": cfg.to_json(), "kernel": params.to_json(), "menu": args.menu,
               "n_offers": args.n_offers, "n_questions": args.n_questions, "arms": arms,
               "roles": roles, "openers": openers, "focal_types": [_tag(ALL_RANKINGS[i]) for i in types],
               "opp_types": [_tag(ALL_RANKINGS[j]) for j in opp_types], "repeats": args.repeats, "seed": args.seed,
               "jev_backend": args.jev_backend if jev_client else None, "jev_shuffle_options": not args.no_shuffle,
               "episodes": n, "error_stubs": errors, "elapsed_s": round(time.time() - t0, 2)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
