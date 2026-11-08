"""CD-NOUL-1 Stage 1 arms and the episode loop (`DESIGN_NOUL.md` §3).

The game, kernel, menu and oracles are `car_dealer_live`'s, untouched: the registered T1
opponent plays and answers exactly as in CD-VOI-1, so `exact_ask` remains the ceiling. What
is new is the **text layer**: every answer the opponent gives is served to the focal as a bank
utterance (`verbalize.py`), and the focal's belief is updated by a **decoder** reading of that
utterance instead of the typed yes/no.

Arms (``parse_arm``):

* ``plan_ask-<decoder>-<a>`` — the exact planner with ASK (C_plan over the planner-complete
  shortlist menu) under the hedged answer model with flip rate ``a`` in BOTH planner and
  filter (CD-MIS-2's ``C_ans a``); the belief updates on typed moves through the matched
  filter and on answers through ``decoder`` ∈ {oracle, regex, noul_answer, noul_type,
  sonnet_answer}. ``plan_ask-oracle-0`` is CD-VOI-1's ``exact_ask`` on the same seeds (a gate).
* ``plan_noask`` — the exact planner with ASK forbidden (B_plan): never sees text.
* ``jev_agent`` / ``sonnet_agent`` — `agents_noul`: the LLM reads the conversation (answers as
  text) and chooses from the same lettered menu. The belief that builds THEIR menu (the
  scaffolding, not the chooser) is updated on answers by ``AGENT_MENU_DECODER`` at the
  registered hedge, so an agent arm and the corresponding planner arm face identical menus
  and the contrast isolates the chooser.

Common random numbers: the opponent's move stream is seeded by (seed, cell, repeat) as in
Phase 1, and the bank variant served for a question is a function of (seed, cell, repeat,
question) only — never of the arm — so two arms that ask the same question in the same
episode hear the same words (``variant_plan``).

Every decision records, beside the Phase 1 annotations under the arm's own model, the
evaluator's references: the exact matched belief under the true kernel, whether ASK is
strictly optimal there (``ask_optimal_true``) and the decision regret under the true model at
the true belief (``decision_regret_true``), so selectivity and regret are measured against the
same reference for every arm.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_discrete.scoring import analyze_type_pair
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live import filters_mis as fm
from llm_rl_scripts.car_dealer_live.arms_live import exact_ask, exact_noask
from llm_rl_scripts.car_dealer_live.menu_live import Menu
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.oracle import EPS, Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, PROPOSE, Action, LiveConfig, apply, asks_by, new_game, other

from . import decode as dc
from .verbalize import K_VARIANTS, REGIMES, load_bank

DECODERS = ("oracle", "regex", "noul_answer", "noul_type", "sonnet_answer")
HEDGES = (Fraction(0), Fraction(1, 5))
REGISTERED_HEDGE = Fraction(1, 5)          # CD-MIS-2's recovering flip rate
AGENT_MENU_DECODER = "noul_answer"         # the belief behind the agents' menus (scaffolding only)
ETA = 0.0                                  # exact Bayes, as Phase 1's matched filter: no background floor.
                                           # (CD-MIS's 1e-3 floor leaves mass on ruled-out types and makes the
                                           # planner ask once more; it existed for misspecified kernels, which
                                           # Stage 1 does not have. A confident wrong decode at hedge 0 can zero
                                           # the true type; filters_mis.update then keeps the belief rather than
                                           # dividing by zero, which is that arm's honest failure mode.)
PLAN_NOASK, JEV_AGENT, SONNET_AGENT = "plan_noask", "jev_agent", "sonnet_agent"
AGENT_ARMS = (JEV_AGENT, SONNET_AGENT)
NO_TEXT_REGIME = "none"

_ARM_RE = re.compile(r"^plan_ask-(" + "|".join(DECODERS) + r")-(0|0\.2)$")


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                      # plan_ask | plan_noask | jev_agent | sonnet_agent
    decoder: Optional[str] = None  # plan_ask only
    hedge: Fraction = Fraction(0)  # planner + filter answer flip rate

    @property
    def asks(self) -> bool:
        return self.kind != "plan_noask"

    @property
    def uses_text(self) -> bool:
        return self.kind != "plan_noask"


def parse_arm(name: str) -> Arm:
    if name == PLAN_NOASK:
        return Arm(name, "plan_noask")
    if name in AGENT_ARMS:
        return Arm(name, name, decoder=AGENT_MENU_DECODER, hedge=REGISTERED_HEDGE)
    m = _ARM_RE.match(name)
    if not m:
        raise ValueError(f"unknown arm {name!r}")
    return Arm(name, "plan_ask", decoder=m.group(1), hedge=Fraction(m.group(2)))


def arm_name(decoder: str, hedge: Fraction) -> str:
    return f"plan_ask-{decoder}-{'0' if hedge == 0 else '0.2'}"


def planner_arms() -> List[str]:
    return [arm_name(d, a) for d in DECODERS for a in HEDGES] + [PLAN_NOASK]


# ---- the text bank and its decoders ---------------------------------------------------------

class TextBank:
    """Bank utterances indexed by (regime, answering role, question, ranking) and the frozen
    decoder readings for each utterance."""

    def __init__(self, bank_dir, require_matrices: bool = True):
        from pathlib import Path
        self.dir = Path(bank_dir)
        rows = load_bank(self.dir / "bank.jsonl")
        self.by_cell: Dict[tuple, List[dict]] = {}
        for r in rows:
            it = r["item"]
            self.by_cell.setdefault((it["regime"], it["role"], it["question_id"], dc.ranking_key(it["ranking"])), []).append(r)
        for k, v in self.by_cell.items():
            v.sort(key=lambda r: r["item"]["variant"])
        self.noul = dc.load_matrix(dc.matrix_path(self.dir, "noul"))
        self.sonnet = dc.load_matrix(dc.matrix_path(self.dir, "sonnet"))
        if require_matrices and (len(self.noul) != len(rows) or len(self.sonnet) != len(rows)):
            raise ValueError("decoder matrices do not cover the bank; run decode.py build for both decoders")
        self.k = max(len(v) for v in self.by_cell.values())

    def utterance(self, regime: str, role: str, question_id: str, ranking, variant: int) -> dict:
        cell = self.by_cell[(regime, role, question_id, dc.ranking_key(ranking))]
        return cell[variant % len(cell)]

    def p_yes(self, decoder: str, row: dict) -> float:
        it = row["item"]
        if decoder == "oracle":
            return dc.oracle_p_yes(it["truthful"])
        if decoder == "regex":
            return dc.regex_p_yes(row["text"])
        if decoder == "noul_answer":
            return float(self.noul[it["id"]]["p_yes"])
        if decoder == "sonnet_answer":
            return float(self.sonnet[it["id"]]["p_yes"])
        raise ValueError(decoder)

    def likelihood(self, decoder: str, row: dict, hedge: Fraction) -> Tuple[Tuple[float, ...], dict]:
        """Per-type effective likelihood of the utterance, and what the decoder read."""
        it = row["item"]
        if decoder == "noul_type":
            vec = tuple(self.noul[it["id"]]["type_vector"])
            return fm.effective_likelihood(vec, 1.0, ETA), {"type_vector": list(vec)}
        p = self.p_yes(decoder, row)
        vec = dc.answer_vector(it["question_id"], p, hedge)
        return fm.effective_likelihood(vec, 0.5, ETA), {"p_yes": p}


def variant_plan(seed: int, role: str, opener: str, i: int, j: int, k: int, n_variants: int = K_VARIANTS) -> Dict[str, int]:
    """The bank variant served for each question in this episode; arm-independent (CRN)."""
    rng = random.Random(f"{seed}|text|{role}|{opener}|{i}|{j}|{k}")
    return {q.id: rng.randrange(n_variants) for q in BANK}


# ---- cells: kernels, menus, oracles per hedge -------------------------------------------------

class NoulCell:
    """Everything exact for one (focal role, focal ranking): the true kernel, and per hedge the
    assumed hedged kernel with its planner-complete menu and menu oracles (ASK / no-ASK)."""

    def __init__(self, cfg: LiveConfig, focal: str, ranking, params: KernelParams = REFERENCE):
        self.cfg, self.focal, self.ranking, self.params = cfg, focal, ranking, params
        self.true_kernel = T1Kernel(other(focal), cfg, params)
        self._per_hedge: Dict[Fraction, dict] = {}

    def hedged(self, a: Fraction) -> dict:
        if a not in self._per_hedge:
            cfg_a = LiveConfig(regime=self.cfg.regime, horizon=self.cfg.horizon, ask_cost=self.cfg.ask_cost, answer_noise=a)
            kernel = T1Kernel(other(self.focal), cfg_a, self.params)
            full_noask = Oracle(self.cfg, self.focal, self.ranking, kernel, ask_allowed=False)
            full_ask = Oracle(self.cfg, self.focal, self.ranking, kernel, ask_allowed=True)
            menu = Menu(self.cfg, self.focal, self.ranking, kernel, plan_oracle=full_noask, plan_ask_oracle=full_ask)
            self._per_hedge[a] = {
                "kernel": kernel, "full_noask": full_noask, "full_ask": full_ask, "menu": menu,
                "ask": Oracle(self.cfg, self.focal, self.ranking, kernel, ask_allowed=True, menu=menu.actions(True)),
                "noask": Oracle(self.cfg, self.focal, self.ranking, kernel, ask_allowed=False, menu=menu.actions(False)),
            }
        return self._per_hedge[a]

    def ceilings(self, opener: str) -> dict:
        h = self.hedged(Fraction(0))
        out = {"menu_ask": h["ask"].root_value(opener), "menu_noask": h["noask"].root_value(opener),
               "full_ask": h["full_ask"].root_value(opener), "full_noask": h["full_noask"].root_value(opener)}
        out["delta_full"] = out["full_ask"] - out["full_noask"]
        out["delta_menu"] = out["menu_ask"] - out["menu_noask"]
        return out


def _f(x):
    return None if x is None else {"exact": str(x), "float": float(x)}


# ---- the episode ------------------------------------------------------------------------------

def play_episode(cell: NoulCell, arm: Arm, regime: str, opener: str, opp_ranking, rng: random.Random,
                 variants: Dict[str, int], bank: Optional[TextBank], repeat: int, agent=None) -> dict:
    cfg, focal = cell.cfg, cell.focal
    true_idx = ALL_RANKINGS.index(opp_ranking)
    state = new_game(cfg, opener)
    exact0 = cell.hedged(Fraction(0))                       # the evaluator's reference (true kernel, truthful answers)
    own = cell.hedged(arm.hedge)                             # the arm's planner and menu
    spec = fm.FilterSpec(f"B_ans{float(arm.hedge)}", eta=ETA, answer_noise=float(arm.hedge))
    belief = bl.uniform()        # the arm's belief (decoded answers, matched moves, hedged)
    true_b = bl.uniform()        # exact posterior under the true kernel and the typed answers
    utterances: Dict[int, str] = {}
    decisions: List[dict] = []
    while state.terminal is None:
        if state.turn != focal:
            act = cell.true_kernel.sample(state, opp_ranking, rng)
            lik_true = bl.action_likelihoods(cell.true_kernel, state)[act]
            lik_own = fm.opponent_action_likelihood(own["kernel"], state, act, spec)
            state = apply(state, act)
            true_b = bl.posterior(true_b, lik_true)
            belief = fm.update(belief, lik_own, spec)
            continue
        rows_ask = own["menu"].rows(state, belief, include_questions=True)
        rows = rows_ask if arm.asks else own["menu"].rows(state, belief, include_questions=False)
        q_ask = {r.key: own["ask"].q_value(state, belief, r.action) for r in rows_ask}
        q_noask = {r.key: own["noask"].q_value(state, belief, r.action) for r in rows_ask if r.action.kind != ASK}
        # evaluator's reference at the true belief, over the same options
        q_true = {r.key: exact0["ask"].q_value(state, true_b, r.action) for r in rows_ask}
        ask_q_true = [q_true[r.key] for r in rows_ask if r.action.kind == ASK]
        non_ask_q_true = [q_true[r.key] for r in rows_ask if r.action.kind != ASK]
        ask_optimal_true = bool(ask_q_true) and max(ask_q_true) > max(non_ask_q_true) + EPS
        info = None
        if arm.kind == "plan_ask":
            key = exact_ask(rows, q_ask, q_noask, rng).key
        elif arm.kind == "plan_noask":
            key = exact_noask(rows, q_ask, q_noask, rng).key
        else:
            agent.utterances = utterances
            key = agent.choose(state, rows)
            info = agent.last_info
        chosen = next(r for r in rows if r.key == key)
        ref = q_ask if arm.asks else q_noask
        best_own = max(ref[r.key] for r in rows)
        best_true = max(q_true[r.key] for r in rows)
        dec = {
            "remaining": state.remaining, "at_deadline": state.at_deadline,
            "facing_offer": bool(state.standing and state.standing.proposer != focal),
            "belief": list(belief), "belief_true": list(true_b),
            "agent_belief": fm.belief_diagnostics(belief, true_idx), "true_belief": fm.belief_diagnostics(true_b, true_idx),
            "menu": [r.to_json() for r in rows], "q_ask": q_ask, "q_noask": q_noask, "q_true": q_true,
            "ask_legal": bool(ask_q_true), "ask_optimal_true": ask_optimal_true,
            "best_ask_minus_best_nonask_true": (max(ask_q_true) - max(non_ask_q_true)) if ask_q_true else None,
            "chosen": key, "chosen_kind": chosen.action.kind, "chosen_tag": chosen.tag, "chosen_position": rows.index(chosen),
            "decision_regret_own": best_own - ref[key], "decision_regret_true": best_true - q_true[key],
            "agent": info, "decode": None,
        }
        act = chosen.action
        if act.kind == ASK:
            ans = cell.true_kernel.sample_answer(act.question_id, opp_ranking, rng)
            true_b = bl.posterior(true_b, bl.answer_likelihoods(cell.true_kernel, act.question_id)[ans])
            if arm.uses_text:
                row = bank.utterance(regime, other(focal), act.question_id, opp_ranking, variants[act.question_id])
                assert row["item"]["truthful"] == ans, "bank utterance must verbalize the drawn answer"
                lik, read = bank.likelihood(arm.decoder, row, arm.hedge)
                utterances[len(state.events)] = row["text"]
                before = belief[true_idx]
                belief = fm.update(belief, lik, spec)
                dec["decode"] = {"decoder": arm.decoder, "bank_id": row["item"]["id"], "variant": row["item"]["variant"],
                                 "truthful": ans, "read": read, "likelihood": list(lik),
                                 "p_true_before": before, "p_true_after": belief[true_idx],
                                 "p_true_exact_after": true_b[true_idx]}
            else:   # unreachable for a non-asking arm, kept for symmetry
                belief = fm.update(belief, fm.answer_likelihood(own["kernel"], act.question_id, ans, spec), spec)
            state = apply(state, act, ans)
        else:
            state = apply(state, act)
        decisions.append(dec)
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
    return {
        "arm": arm.name, "kind": arm.kind, "decoder": arm.decoder, "hedge": str(arm.hedge), "regime": regime,
        "focal": focal, "opener": opener, "focal_ranking": list(cell.ranking), "repeat": repeat,
        "private_truth_evaluator_only": {"opp_ranking": list(opp_ranking)},
        "variants": variants, "decisions": decisions, "n_decisions": len(decisions),
        "questions_asked": n_ask, "proposals_made": sum(1 for e in state.events if e.actor == focal and e.action.kind == PROPOSE),
        "terminal": {"deal_id": t.deal_id, "reason": t.reason}, "agreement": deal is not None,
        "focal_utility": _f(u), "gross_gain": _f(gross), "net_gain": _f(net), "normalized_net_gain": _f(y), "u_max": _f(u_max),
        "feasible_pair": analyze_type_pair(b_rank, s_rank, r).is_feasible,
        "total_decision_regret_true": sum(d["decision_regret_true"] for d in decisions),
        "zero_regret_decisions_true": sum(1 for d in decisions if d["decision_regret_true"] <= EPS),
        "asks_when_ask_optimal": sum(1 for d in decisions if d["ask_optimal_true"] and d["chosen_kind"] == ASK),
        "decisions_ask_optimal": sum(1 for d in decisions if d["ask_optimal_true"]),
        "asks_when_ask_not_optimal": sum(1 for d in decisions if d["ask_legal"] and not d["ask_optimal_true"] and d["chosen_kind"] == ASK),
        "decisions_ask_legal_not_optimal": sum(1 for d in decisions if d["ask_legal"] and not d["ask_optimal_true"]),
        "final_agent_belief": fm.belief_diagnostics(belief, true_idx), "final_true_belief": fm.belief_diagnostics(true_b, true_idx),
        "decode_logloss_bits": [-math.log2(max(d["decode"]["p_true_after"], 1e-300)) for d in decisions if d["decode"]],
    }
