"""CD-MIS-1 / CD-MIS-2 runner: learning with the wrong opponent model (design document section 7).

Fixed: the agent's ASSUMED kernel (the registered T1), its planner (the exact oracle under the
assumed kernel, over the full legal action set; ASK forbidden for A/B arms, allowed for C arms),
its question bank and action costs. Varied: the TRUE opponent process (``opponents_mis``) and the
belief filter (``filters_mis``). ``O_true`` plans with the true process and the exact filter under it.

Per episode the record holds Y (normalized net gain), gross/net gain, agreement, asks, and the
belief-vs-decision diagnostics: the agent's posterior on the true type and its log-loss, entropy,
mass on the best wrong type, whether the argmax is the true type (final and per decision); the
exact posterior under the TRUE kernel for comparison; the Brier score of the assumed model's
P(accept) for the focal's own proposals; the predictive log-loss of every observed opponent
response under the agent's effective likelihood; and the decision regret of every focal choice
measured by O_true (Q under the true process and the exact true belief).

    PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_live.run_mis --process history --horizon 3 \
        --arms A_prior,B_naive,B_temp0.5,B_temp0.75,B_mix0.05,B_mix0.2,C_naive,C_mix0.2,O_true \
        --repeats 5 --seed 41 --out results/<dir>

CD-MIS-2 adds severity levels (``--process evasive@0.1,evasive@0.5``; see ``opponents_mis``) and the
hedged-answer ASK arms ``C_ans0.1`` / ``C_ans0.2`` / ``C_ans0.3`` (planner and filter both assume that
answer flip rate; the behavioural kernel stays the registered T1). Random streams are seeded by the
process FAMILY, so a family's levels share episodes wherever the play allows (the no-ask arms are
identical across evasive levels; the CD-MIS-1 level of each family replicates CD-MIS-1 exactly).
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

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_discrete.scoring import analyze_type_pair

from . import belief_live as bl
from . import filters_mis as fm
from .opponents_mis import PROCESSES, assumed_kernel, family, parse_process, severity, true_kernel
from .oracle import EPS, Oracle
from .protocol_live import ACCEPT, ASK, PROPOSE, ENV_VERSION, Action, LiveConfig, apply, asks_by, new_game, other

ARMS = ["A_prior", "B_naive", "B_temp0.5", "B_temp0.75", "B_mix0.05", "B_mix0.2", "C_naive", "C_mix0.2",
        "C_ans0.1", "C_ans0.2", "C_ans0.3", "O_true"]
ARM_FILTER = {"A_prior": "A_prior", "B_naive": "B_naive", "B_temp0.5": "B_temp0.5", "B_temp0.75": "B_temp0.75",
              "B_mix0.05": "B_mix0.05", "B_mix0.2": "B_mix0.2", "C_naive": "B_naive", "C_mix0.2": "B_mix0.2",
              "C_ans0.1": "B_ans0.1", "C_ans0.2": "B_ans0.2", "C_ans0.3": "B_ans0.3"}
ASK_ARMS = {"C_naive", "C_mix0.2", "C_ans0.1", "C_ans0.2", "C_ans0.3", "O_true"}
HEDGED_ARMS = {"C_ans0.1": Fraction(1, 10), "C_ans0.2": Fraction(1, 5), "C_ans0.3": Fraction(3, 10)}


def _f(x):
    return None if x is None else {"exact": str(x), "float": float(x)}


def _tag(r):
    return "/".join(a[0] for a in r)


class MisCell:
    """One (focal role, focal type) under one true process: assumed planners, true oracle, kernels."""

    def __init__(self, cfg: LiveConfig, focal: str, ranking, process: str):
        self.cfg, self.focal, self.ranking, self.process = cfg, focal, ranking, process
        self.assumed = assumed_kernel(other(focal), cfg)
        self.truth = true_kernel(process, other(focal), cfg)
        self.plan_noask = Oracle(cfg, focal, ranking, self.assumed, ask_allowed=False)
        self.plan_ask = Oracle(cfg, focal, ranking, self.assumed, ask_allowed=True)
        self.o_true = Oracle(cfg, focal, ranking, self.truth, ask_allowed=True)
        self._hedged: Dict[str, Oracle] = {}

    def planner(self, arm: str) -> Oracle:
        if arm == "O_true":
            return self.o_true
        if arm in HEDGED_ARMS:
            if arm not in self._hedged:
                hedged = assumed_kernel(other(self.focal), self.cfg, answer_noise=HEDGED_ARMS[arm])
                self._hedged[arm] = Oracle(self.cfg, self.focal, self.ranking, hedged, ask_allowed=True)
            return self._hedged[arm]
        return self.plan_ask if arm in ASK_ARMS else self.plan_noask


def play_episode(cell: MisCell, arm: str, opener: str, opp_ranking, rng: random.Random, repeat: int) -> dict:
    cfg, focal = cell.cfg, cell.focal
    true_idx = ALL_RANKINGS.index(opp_ranking)
    state = new_game(cfg, opener)
    agent_b = bl.uniform()          # the arm's belief (assumed kernel through its filter)
    true_b = bl.uniform()           # exact posterior under the TRUE kernel (diagnostic; O_true's belief)
    spec = fm.FILTERS[ARM_FILTER.get(arm, "B_naive")]
    planner = cell.planner(arm)
    decisions: List[dict] = []
    pred_ll = []        # predictive log-loss of observed opponent responses under the agent's model
    brier = []          # (P_assumed(accept) - 1[accepted])^2 for the focal's proposals
    pending_accept_pred: Optional[float] = None
    while state.terminal is None:
        if state.turn != focal:
            act = cell.truth.sample(state, opp_ranking, rng)
            if pending_accept_pred is not None:
                brier.append((pending_accept_pred - (1.0 if act.kind == ACCEPT else 0.0)) ** 2)
                pending_accept_pred = None
            lik_agent = fm.opponent_action_likelihood(cell.assumed, state, act, spec)
            if spec.update:
                pred_ll.append(fm.predictive_logloss_bits(agent_b, lik_agent))
            agent_b = fm.update(agent_b, lik_agent, spec)
            lik_true = bl.action_likelihoods(cell.truth, state).get(act)
            true_b = bl.posterior(true_b, lik_true) if lik_true is not None else true_b
            state = apply(state, act)
            continue
        belief_used = true_b if arm == "O_true" else agent_b
        act, _ = planner.best(state, belief_used)
        # decision regret measured by the TRUE model at the TRUE belief
        q_true = cell.o_true.q_values(state, true_b)
        best_true = max(q_true.values())
        regret = best_true - q_true.get(act, float("nan")) if act in q_true else None
        decisions.append({"remaining": state.remaining, "chosen": act.key, "chosen_kind": act.kind,
                          "decision_regret_true": regret,
                          "agent": fm.belief_diagnostics(agent_b, true_idx), "true_filter": fm.belief_diagnostics(true_b, true_idx)})
        if act.kind == PROPOSE:
            # the assumed model's predicted acceptance of this proposal, scored against the truth when it responds
            nxt = apply(state, act)
            liks = bl.action_likelihoods(cell.assumed, nxt)
            acc = liks.get(Action(ACCEPT))
            pending_accept_pred = bl.marginal(agent_b, acc) if acc is not None else 0.0
        if act.kind == ASK:
            ans = cell.truth.sample_answer(act.question_id, opp_ranking, rng)
            lik_agent = fm.answer_likelihood(cell.assumed, act.question_id, ans, spec)
            if spec.update:
                pred_ll.append(fm.predictive_logloss_bits(agent_b, lik_agent))
            agent_b = fm.update(agent_b, lik_agent, spec)
            true_b = bl.posterior(true_b, bl.answer_likelihoods(cell.truth, act.question_id)[ans])
            state = apply(state, act, ans)
        else:
            state = apply(state, act)
    t = state.terminal
    r = cfg.reservation
    deal = DEALS_BY_ID[t.deal_id] if t.deal_id else None
    u = utility(focal, cell.ranking, deal) if deal else r
    gross = u - r
    n_ask = asks_by(state, focal)
    net = gross - cfg.ask_cost * n_ask
    u_max = max(utility(focal, cell.ranking, d) for d in DEALS_BY_ID.values())
    y = net / (u_max - r)
    b_rank, s_rank = (cell.ranking, opp_ranking) if focal == "buyer" else (opp_ranking, cell.ranking)
    analysis = analyze_type_pair(b_rank, s_rank, r)
    regrets = [d["decision_regret_true"] for d in decisions if d["decision_regret_true"] is not None]
    return {
        "arm": arm, "process": cell.process, "filter": spec.to_json() if arm != "O_true" else {"name": "exact_true"},
        "focal": focal, "opener": opener, "focal_ranking": list(cell.ranking), "repeat": repeat,
        "private_truth_evaluator_only": {"opp_ranking": list(opp_ranking)},
        "decisions": decisions, "n_decisions": len(decisions), "questions_asked": n_ask,
        "proposals_made": sum(1 for e in state.events if e.actor == focal and e.action.kind == PROPOSE),
        "terminal": {"deal_id": t.deal_id, "reason": t.reason}, "agreement": deal is not None,
        "focal_utility": _f(u), "gross_gain": _f(gross), "net_gain": _f(net), "normalized_net_gain": _f(y), "u_max": _f(u_max),
        "feasible_pair": analysis.is_feasible,
        "total_decision_regret_true": sum(regrets), "zero_regret_decisions": sum(1 for x in regrets if x <= EPS),
        "final_agent": fm.belief_diagnostics(agent_b, true_idx), "final_true_filter": fm.belief_diagnostics(true_b, true_idx),
        "mean_predictive_logloss_bits": (sum(pred_ll) / len(pred_ll)) if pred_ll else None, "n_predictions": len(pred_ll),
        "brier_accept": (sum(brier) / len(brier)) if brier else None, "n_brier": len(brier),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--process", default="all", help="one of %s or 'all'" % (PROCESSES,))
    p.add_argument("--regime", default="easy"); p.add_argument("--horizon", type=int, default=3); p.add_argument("--ask-cost", default="0")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--roles", default="buyer,seller"); p.add_argument("--openers", default="buyer,seller")
    p.add_argument("--types", default="all"); p.add_argument("--opp-types", default="all")
    p.add_argument("--repeats", type=int, default=5); p.add_argument("--seed", type=int, default=41)
    p.add_argument("--design", default="CD-MIS-1", help="label written to summary.json (CD-MIS-1 or CD-MIS-2)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = LiveConfig(regime=args.regime, horizon=args.horizon, ask_cost=Fraction(args.ask_cost))
    procs = list(PROCESSES) if args.process == "all" else args.process.split(",")
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; known {ARMS}")
    roles = args.roles.split(","); openers = args.openers.split(",")
    types = list(range(len(ALL_RANKINGS))) if args.types == "all" else [int(x) for x in args.types.split(",")]
    opp_types = list(range(len(ALL_RANKINGS))) if args.opp_types == "all" else [int(x) for x in args.opp_types.split(",")]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time(); n = errors = 0
    with gzip.open(os.path.join(args.out, "runs.jsonl.gz"), "wt") as f:
        for proc in procs:
            for role in roles:
                for i in types:
                    cell = MisCell(cfg, role, ALL_RANKINGS[i], proc)
                    for arm in arms:
                        for opener in openers:
                            for j in opp_types:
                                for k in range(args.repeats):
                                    rng = random.Random(f"{args.seed}|{family(proc)}|{role}|{opener}|{i}|{j}|{k}")
                                    try:
                                        rec = play_episode(cell, arm, opener, ALL_RANKINGS[j], rng, k)
                                    except Exception as e:
                                        rec = {"arm": arm, "process": proc, "focal": role, "opener": opener, "focal_ranking": list(ALL_RANKINGS[i]),
                                               "repeat": k, "private_truth_evaluator_only": {"opp_ranking": list(ALL_RANKINGS[j])}, "error": f"{type(e).__name__}: {e}"}
                                        errors += 1
                                        print(f"  episode failed: {rec['error']}", file=sys.stderr)
                                    f.write(json.dumps(rec) + "\n"); n += 1
                print(f"[{time.time()-t0:7.1f}s] process={proc} role={role} done ({n} episodes)", file=sys.stderr)
    summary = {"env": ENV_VERSION, "design": args.design, "config": cfg.to_json(), "assumed_kernel": assumed_kernel("seller", cfg).p.to_json(),
               "processes": procs, "process_meta": {q: {"family": family(q), "severity": (None if severity(q) is None else str(severity(q)))} for q in procs},
               "arms": arms, "hedged_arms": {a: str(v) for a, v in HEDGED_ARMS.items() if a in arms}, "filters": {a: fm.FILTERS[ARM_FILTER[a]].to_json() for a in arms if a in ARM_FILTER},
               "roles": roles, "openers": openers, "focal_types": [_tag(ALL_RANKINGS[i]) for i in types], "opp_types": [_tag(ALL_RANKINGS[j]) for j in opp_types],
               "repeats": args.repeats, "seed": args.seed, "episodes": n, "error_stubs": errors, "elapsed_s": round(time.time() - t0, 2)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "filters"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
