"""CD-NOUL-2 Stage 2 arms and episode loop (`DESIGN_NOUL2.md` §4; NEG-8).

What changes from Stage 1 (`arms_noul.py`): the opponent is an object (`ScriptedT1Opponent` for the parity gate,
`opponent_llm.KernelOpponent` offline, `opponent_llm.LLMOpponent` live) whose typed moves come with words; the
focal's moves are put into words by a realizer (`realizer.TemplateRealizer` or `realizer.Realizer`) that the
opponent hears; answers are decoded **live** (no matrix) by `oracle` / `regex` / `noul` / `noulcal` / `sonnet`;
the `adaptive` hedge uses the calibrated read directly in the filter and a per-question flip-rate map in the
planner (`LiveConfig.answer_noise_by_q`); `+offer` multiplies a `noul_offer` type likelihood on the opponent's
PROPOSE / REJECT / HOLD words into the move update; the planner's kernel is the fitted one or `-T1`; and
`plan_reveal` is the clairvoyant reference (belief = point mass on the true ranking).

Arm grammar: ``plan_ask-<decoder>[+offer]-<hedge>[-T1][-template]`` with decoder ∈ {oracle, regex, noul,
noulcal, sonnet}, hedge ∈ {0, 0.2, adaptive} (`adaptive` only with noul / noulcal / sonnet; `0` is for the
parity gate); ``plan_noask``; ``plan_reveal``; ``jev_agent``; ``sonnet_agent``.

The evaluator's reference (``ask_optimal_true``, ``decision_regret_true``) is the planner's **assumed**
kernel at the exact matched belief under the typed truthful answers — a reference, not a truth, now that the
opponent is not the kernel. ``plan_ask-oracle-0`` with ``ScriptedT1Opponent`` + ``TemplateRealizer`` + T1
reproduces Stage 1's ``plan_ask-oracle-0`` decision for decision (gate 7).
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_discrete.scoring import analyze_type_pair
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live import filters_mis as fm
from llm_rl_scripts.car_dealer_live.arms_live import exact_ask, exact_noask
from llm_rl_scripts.car_dealer_live.menu_live import Menu
from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.oracle import EPS, Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import (ASK, HOLD, PROPOSE, REJECT, Action, LiveConfig, PublicState, apply,
                                                          asks_by, legal_actions, new_game, other, truthful_answer)

from . import calibrate_noul as cn
from . import decode as dc
from .arms_noul import ETA
from .kernel_fit import record_move
from .opponent_llm import BRIEFS
from .realizer import Realization, TemplateRealizer

DECODERS = ("oracle", "regex", "noul", "noulcal", "sonnet")
HEDGES = ("0", "0.2", "adaptive")
ADAPTIVE_DECODERS = ("noul", "noulcal", "sonnet")
PLAN_ASK, PLAN_NOASK, PLAN_REVEAL, JEV_AGENT, SONNET_AGENT = "plan_ask", "plan_noask", "plan_reveal", "jev_agent", "sonnet_agent"
AGENT_ARMS = (JEV_AGENT, SONNET_AGENT)
AGENT_MENU_DECODER, AGENT_MENU_HEDGE = "noulcal", "adaptive"   # the belief behind the agents' menus (scaffolding)
BRIEF_REGIME = {"candid": "plain", "guarded": "oblique", "adversarial": "oblique"}   # which Stage 1 regime's a_q
DEFAULT_MAP, DEFAULT_CLAMP = "platt", 0.95                                          # §3.1a recommendation; registration decides
OFFER_TEMPER = 1.0

_ARM_RE = re.compile(r"^plan_ask-(" + "|".join(DECODERS) + r")(\+offer)?-(0|0\.2|adaptive)(-T1)?(-template)?$")


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    decoder: Optional[str] = None
    offer: bool = False
    hedge: Optional[str] = None
    kernel: str = "fitted"        # fitted | T1
    realizer: str = "llm"         # llm | template

    @property
    def asks(self) -> bool:
        return self.kind in (PLAN_ASK, JEV_AGENT, SONNET_AGENT)


def parse_arm(name: str) -> Arm:
    if name == PLAN_NOASK:
        return Arm(name, PLAN_NOASK)
    if name == PLAN_REVEAL:
        return Arm(name, PLAN_REVEAL)
    if name in AGENT_ARMS:
        return Arm(name, name, AGENT_MENU_DECODER, False, AGENT_MENU_HEDGE)
    m = _ARM_RE.match(name)
    if not m:
        raise ValueError(f"unknown arm {name!r}")
    decoder, offer, hedge, t1, template = m.group(1), bool(m.group(2)), m.group(3), bool(m.group(4)), bool(m.group(5))
    if hedge == "adaptive" and decoder not in ADAPTIVE_DECODERS:
        raise ValueError(f"{name}: the adaptive hedge needs a calibrated decoder ({', '.join(ADAPTIVE_DECODERS)})")
    if hedge == "0" and decoder != "oracle":
        raise ValueError(f"{name}: hedge 0 is the parity gate's and is allowed with the oracle decoder only")
    return Arm(name, PLAN_ASK, decoder, offer, hedge, "T1" if t1 else "fitted", "template" if template else "llm")


def design_arms() -> List[str]:
    """The §4 table (without the parity-gate arm)."""
    return ["plan_ask-noulcal-adaptive", "plan_ask-noulcal-0.2", "plan_ask-noul-adaptive", "plan_ask-sonnet-adaptive",
            "plan_ask-regex-0.2", "plan_ask-noulcal+offer-adaptive", "plan_ask-noulcal-adaptive-T1",
            "plan_ask-noulcal-adaptive-template", PLAN_NOASK, PLAN_REVEAL, SONNET_AGENT, JEV_AGENT]


ARM_BRIEFS = {"plan_ask-noulcal-adaptive": BRIEFS, "plan_ask-noulcal-0.2": BRIEFS, PLAN_NOASK: BRIEFS, PLAN_REVEAL: BRIEFS,
              "plan_ask-noulcal-adaptive-T1": ("guarded",), "plan_ask-noulcal-adaptive-template": ("guarded",)}


def briefs_for(arm_name: str, briefs: Tuple[str, ...]) -> List[str]:
    allowed = ARM_BRIEFS.get(arm_name, ("candid", "guarded"))
    return [b for b in briefs if b in allowed]


# ---- calibration map -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MapSpec:
    name: str = DEFAULT_MAP           # raw | pooled | oblique_fit | safe | platt | platt_oblique
    clamp: float = DEFAULT_CLAMP      # reads are clipped to [1 - clamp, clamp] in the adaptive filter; 1.0 = none


class Calibration:
    """Reads ``calibration_noul2.json``: per decoder the (T, b) of the chosen map and the a_q vectors per
    Stage 1 regime under it. The `sonnet` decoder is read raw (its map is a Stage 2 secondary); `noul` raw;
    `noulcal` under the chosen map."""

    def __init__(self, path, spec: MapSpec = MapSpec()):
        self.spec = spec
        self.data = json.load(open(path))
        self.path = str(path)

    def params(self, decoder: str) -> Tuple[float, float]:
        if decoder == "noulcal":
            c = self.data["decoders"]["noul"]["candidates"][self.spec.name]
            return float(c["temperature"]), float(c["bias"])
        return 1.0, 0.0

    def read(self, decoder: str, p_raw: float) -> float:
        T, b = self.params(decoder)
        p = cn.read(p_raw, T, b)
        c = self.spec.clamp
        return min(max(p, 1.0 - c), c)

    def noise_map(self, decoder: str, brief: str) -> Tuple[Tuple[str, Fraction], ...]:
        src = "noul" if decoder in ("noul", "noulcal") else "sonnet"
        cand = self.spec.name if decoder == "noulcal" else "raw"
        a_q = self.data["decoders"][src]["candidates"][cand]["a_q"][BRIEF_REGIME[brief]]
        return cn.as_noise_map(a_q)

    def to_json(self) -> dict:
        return {"path": self.path, "map": self.spec.name, "clamp": self.spec.clamp,
                "noulcal_T_b": list(self.params("noulcal"))}


# ---- cells --------------------------------------------------------------------------------------------

def params_from_json(d: dict) -> KernelParams:
    return KernelParams(alpha0=Fraction(d["alpha0"]), beta=float(d["beta"]), p_counter=float(d["p_counter"]),
                        accept_noise=float(d["accept_noise"]), p_hold=float(d["p_hold"]), shortlist=int(d["shortlist"]))


class Stage2Cell:
    """Planners for one (focal role, focal ranking): per (hedge, kernel, noise map) the assumed kernel, its
    planner-complete menu and the ASK / no-ASK oracles; plus the evaluator's reference (assumed fitted kernel,
    truthful answers, a = 0)."""

    def __init__(self, cfg: LiveConfig, focal: str, ranking, fitted: KernelParams = REFERENCE, t1: KernelParams = REFERENCE):
        self.cfg, self.focal, self.ranking = cfg, focal, tuple(ranking)
        self.params = {"fitted": fitted, "T1": t1}
        self._cache: Dict[tuple, dict] = {}

    def planner(self, hedge: str, kernel: str, noise_map: Optional[Tuple[Tuple[str, Fraction], ...]] = None) -> dict:
        key = (hedge, kernel, noise_map)
        if key not in self._cache:
            a = Fraction(0) if hedge in ("0", "adaptive") else Fraction(1, 5)
            cfg_a = LiveConfig(regime=self.cfg.regime, horizon=self.cfg.horizon, ask_cost=self.cfg.ask_cost, answer_noise=a,
                               answer_noise_by_q=noise_map if hedge == "adaptive" else None)
            k = T1Kernel(other(self.focal), cfg_a, self.params[kernel])
            full_noask = Oracle(self.cfg, self.focal, self.ranking, k, ask_allowed=False)
            full_ask = Oracle(self.cfg, self.focal, self.ranking, k, ask_allowed=True)
            menu = Menu(self.cfg, self.focal, self.ranking, k, plan_oracle=full_noask, plan_ask_oracle=full_ask)
            self._cache[key] = {"kernel": k, "menu": menu, "full_noask": full_noask, "full_ask": full_ask,
                                "ask": Oracle(self.cfg, self.focal, self.ranking, k, ask_allowed=True, menu=menu.actions(True)),
                                "noask": Oracle(self.cfg, self.focal, self.ranking, k, ask_allowed=False, menu=menu.actions(False))}
        return self._cache[key]

    def reference(self) -> dict:
        return self.planner("0", "fitted")

    def ceilings(self, opener: str) -> dict:
        h = self.reference()
        out = {"menu_ask": h["ask"].root_value(opener), "menu_noask": h["noask"].root_value(opener),
               "full_ask": h["full_ask"].root_value(opener), "full_noask": h["full_noask"].root_value(opener)}
        out["delta_full"] = out["full_ask"] - out["full_noask"]
        out["delta_menu"] = out["menu_ask"] - out["menu_noask"]
        return out


# ---- opponents ------------------------------------------------------------------------------------------

class ScriptedT1Opponent:
    """Stage 1's opponent exactly: the T1 kernel sampled on the Stage 1 rng stream, typed answers as words,
    no speech, deaf to the realizer. For the parity gate and for cheap offline runs."""

    def __init__(self, kernel: T1Kernel, ranking, rng: random.Random):
        self.kernel, self.ranking, self.rng = kernel, tuple(ranking), rng
        self.said: Dict[int, str] = {}
        self.heard: Dict[int, str] = {}
        self.last_info = None
        self.calls: List[dict] = []

    def move(self, state: PublicState) -> Tuple[Action, str]:
        return self.kernel.sample(state, self.ranking, self.rng), ""

    def answer(self, state: PublicState, question_id: str) -> str:
        ans = self.kernel.sample_answer(question_id, self.ranking, self.rng)
        self.said[len(state.events)] = ans
        return ans


# ---- live decoders ----------------------------------------------------------------------------------------

class Decoders:
    """In-episode answer and offer reads. ``jev_client`` (`LLM_RL.jev.client.JevClient`) for noul / noulcal /
    offer; ``sonnet_client`` (persona ``Client``) for the judge."""

    def __init__(self, calibration: Calibration, jev_client=None, sonnet_client=None):
        self.cal, self.jev, self.sonnet = calibration, jev_client, sonnet_client

    def p_yes(self, decoder: str, role: str, question_id: str, reply: str, truthful: str) -> Tuple[float, dict]:
        q = BANK_BY_ID[question_id]
        if decoder == "oracle":
            return dc.oracle_p_yes(truthful), {"p_raw": dc.oracle_p_yes(truthful)}
        if decoder == "regex":
            p = dc.regex_p_yes(reply)
            return p, {"p_raw": p}
        if decoder in ("noul", "noulcal"):
            qs = {k: v for k, v in dc.noul_questions(role, q.text).items() if k in ("yes", "no")}
            raw = self.jev.nouls(dc.context_state(role, q.text, reply), qs)
            p_raw = dc.p_yes_two_way(raw["yes"], raw["no"])
            p = self.cal.read("noulcal", p_raw) if decoder == "noulcal" else p_raw
            return p, {"p_raw": p_raw, "raw": raw}
        if decoder == "sonnet":
            messages = [{"role": "user", "content": dc.sonnet_user(role, q.text, reply)}]
            r = self.sonnet.complete(dc.SONNET_SYSTEM, messages, dc.SONNET_MAX_TOKENS)
            p_raw = dc.parse_p_yes(r.text)
            malformed = p_raw is None
            if malformed:
                r = self.sonnet.complete(dc.SONNET_SYSTEM, messages, dc.SONNET_RETRY_MAX_TOKENS)
                p_raw = dc.parse_p_yes(r.text)
                malformed = p_raw is None
            p_raw = 0.5 if p_raw is None else p_raw
            c = self.cal.spec.clamp
            return min(max(p_raw, 1.0 - c), c), {"p_raw": p_raw, "malformed": malformed, "reply": r.text[:200]}
        raise ValueError(decoder)

    def offer_vector(self, role: str, action: Action, say: str) -> Tuple[Tuple[float, ...], dict]:
        label = DEALS_BY_ID[action.deal_id].label() if action.kind == PROPOSE else None
        item = dc.offer_item(self.jev, role, action.key, say, label)
        vec = tuple(item["type_vector"])
        if OFFER_TEMPER != 1.0:
            vec = tuple(v ** OFFER_TEMPER for v in vec)
        return vec, item


def answer_likelihood(question_id: str, p: float, hedge: str) -> Tuple[float, ...]:
    """The per-type likelihood the filter multiplies in: Stage 1's hedged answer model at 0.2, the read itself
    at adaptive (and at 0, the parity gate)."""
    noise = Fraction(1, 5) if hedge == "0.2" else Fraction(0)
    return fm.effective_likelihood(dc.answer_vector(question_id, p, noise), 0.5, ETA)


def _f(x):
    return None if x is None else {"exact": str(x), "float": float(x)}


def _onehot(idx: int) -> tuple:
    return tuple(1.0 if i == idx else 0.0 for i in range(len(ALL_RANKINGS)))


# ---- the episode -----------------------------------------------------------------------------------------

def play_episode(cell: Stage2Cell, arm: Arm, brief: str, opener: str, opp_ranking, rng: random.Random, opponent,
                 realizer, decoders: Optional[Decoders], calibration: Optional[Calibration], repeat: int,
                 agent=None, move_log: Optional[list] = None, episode_id: str = "") -> dict:
    cfg, focal = cell.cfg, cell.focal
    opp_role = other(focal)
    true_idx = ALL_RANKINGS.index(opp_ranking)
    state = new_game(cfg, opener)
    ref = cell.reference()
    noise_map = None
    if arm.kind == PLAN_ASK and arm.hedge == "adaptive":
        noise_map = calibration.noise_map(arm.decoder, brief)
    elif arm.kind in AGENT_ARMS:
        noise_map = calibration.noise_map(AGENT_MENU_DECODER, brief)
    own = cell.planner(arm.hedge or "0", arm.kernel, noise_map) if arm.kind not in (PLAN_NOASK, PLAN_REVEAL) else cell.planner("0", arm.kernel)
    spec = fm.FilterSpec(f"S2-{arm.hedge}", eta=ETA, answer_noise=0.2 if arm.hedge == "0.2" else 0.0)
    belief = bl.uniform()
    true_b = bl.uniform()
    answers_text: Dict[int, str] = {}       # event index -> the opponent's words when answering (for the agents)
    decisions: List[dict] = []
    opp_infos: List[dict] = []
    real_infos: List[dict] = []
    offer_decodes: List[dict] = []
    while state.terminal is None:
        if state.turn != focal:
            act, say = opponent.move(state)
            if move_log is not None:
                move_log.append(record_move(episode_id, state, opp_role, opp_ranking, act, brief))
            info = getattr(opponent, "last_info", None)
            if info is not None:
                opp_infos.append({"kind": "move", "well_formed": info.well_formed, "retried": info.retried,
                                  "defaulted": info.defaulted, "say": say[:300]})
            lik_ref = bl.action_likelihoods(ref["kernel"], state).get(act)
            lik_own = fm.opponent_action_likelihood(own["kernel"], state, act, spec)
            if arm.offer and say and act.kind in (PROPOSE, REJECT, HOLD):
                vec, item = decoders.offer_vector(opp_role, act, say)
                lik_own = tuple(l * v for l, v in zip(lik_own, vec))
                offer_decodes.append({"event": len(state.events), "action": act.key, "type_vector": list(vec),
                                      "p_true_before": belief[true_idx]})
            state = apply(state, act)
            if lik_ref is not None:
                true_b = bl.posterior(true_b, lik_ref) if sum(p * l for p, l in zip(true_b, lik_ref)) > 0 else true_b
            belief = fm.update(belief, lik_own, spec)
            if offer_decodes and offer_decodes[-1]["event"] == len(state.events) - 1:
                offer_decodes[-1]["p_true_after"] = belief[true_idx]
            continue
        menu_belief = _onehot(true_idx) if arm.kind == PLAN_REVEAL else belief
        rows_ask = own["menu"].rows(state, menu_belief, include_questions=True)
        rows = rows_ask if arm.asks else own["menu"].rows(state, menu_belief, include_questions=False)
        q_ask = {r.key: own["ask"].q_value(state, menu_belief, r.action) for r in rows_ask}
        q_noask = {r.key: own["noask"].q_value(state, menu_belief, r.action) for r in rows_ask if r.action.kind != ASK}
        q_true = {r.key: ref["ask"].q_value(state, true_b, r.action) for r in rows_ask}
        ask_q_true = [q_true[r.key] for r in rows_ask if r.action.kind == ASK]
        non_ask_q_true = [q_true[r.key] for r in rows_ask if r.action.kind != ASK]
        ask_optimal_true = bool(ask_q_true) and max(ask_q_true) > max(non_ask_q_true) + EPS
        info = None
        if arm.kind == PLAN_ASK:
            key = exact_ask(rows, q_ask, q_noask, rng).key
        elif arm.kind in (PLAN_NOASK, PLAN_REVEAL):
            key = exact_noask(rows, q_ask, q_noask, rng).key
        else:
            agent.utterances = answers_text
            key = agent.choose(state, rows)
            info = agent.last_info
        chosen = next(r for r in rows if r.key == key)
        refq = q_ask if arm.asks else q_noask
        best_own = max(refq[r.key] for r in rows)
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
            "decision_regret_own": best_own - refq[key], "decision_regret_true": best_true - q_true[key],
            "agent": info, "decode": None, "realized": None,
        }
        act = chosen.action
        legal = legal_actions(state, focal)
        real: Realization = realizer.realize(state, act, legal, {}, opponent.said)
        opponent.heard[len(state.events)] = real.text
        real_infos.append({"ok": real.ok, "fallback": real.fallback, "retried": real.retried, "reasons": real.reasons})
        dec["realized"] = real.text[:300]
        if act.kind == ASK:
            truthful = truthful_answer(act.question_id, opp_ranking)
            reply = opponent.answer(state, act.question_id)
            oinfo = getattr(opponent, "last_info", None)
            if oinfo is not None and oinfo.kind == "answer":
                opp_infos.append({"kind": "answer", "well_formed": oinfo.well_formed, "retried": oinfo.retried,
                                  "defaulted": oinfo.defaulted, "say": reply[:300]})
            answers_text[len(state.events)] = reply
            true_b = bl.posterior(true_b, bl.answer_likelihoods(ref["kernel"], act.question_id)[truthful])
            dcode = arm.decoder if arm.kind == PLAN_ASK else AGENT_MENU_DECODER
            hedge = arm.hedge if arm.kind == PLAN_ASK else AGENT_MENU_HEDGE
            p, read = decoders.p_yes(dcode, opp_role, act.question_id, reply, truthful)
            lik = answer_likelihood(act.question_id, p, hedge)
            before = belief[true_idx]
            belief = fm.update(belief, lik, spec)
            dec["decode"] = {"decoder": dcode, "truthful": truthful, "reply": reply[:300], "p_yes": p, "read": read,
                             "likelihood": list(lik), "p_true_before": before, "p_true_after": belief[true_idx],
                             "p_true_exact_after": true_b[true_idx],
                             "confident_wrong": (p > 0.9 and truthful == "no") or (p < 0.1 and truthful == "yes")}
            state = apply(state, act, truthful)
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
        "arm": arm.name, "kind": arm.kind, "decoder": arm.decoder, "hedge": arm.hedge, "offer": arm.offer, "kernel": arm.kernel,
        "realizer": arm.realizer, "brief": brief, "focal": focal, "opener": opener, "focal_ranking": list(cell.ranking), "repeat": repeat,
        "private_truth_evaluator_only": {"opp_ranking": list(opp_ranking)},
        "decisions": decisions, "n_decisions": len(decisions),
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
        "confident_wrong_reads": sum(1 for d in decisions if d["decode"] and d["decode"]["confident_wrong"]),
        "opponent": opp_infos, "realizer_calls": real_infos, "offer_decodes": offer_decodes,
    }
