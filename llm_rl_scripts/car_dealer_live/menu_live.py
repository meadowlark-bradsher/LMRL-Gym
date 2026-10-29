"""Phase 1 menu rule for the live game: what a model-driven focal negotiator is
offered at each decision, and the exact annotations (EIG, P(accept), own utility)
of every option.

Shortlist rule (primary; mirrors Condition B so the numbers sit side by side):

* free moves that are legal: ACCEPT / REJECT (facing the other side's offer),
  HOLD (own offer standing), WALK (always);
* up to ``n_questions`` = 2 unasked questions with the highest exact EIG under
  the current belief (ties by bank order);
* up to ``n_offers`` = 3 offer candidates built from the CURRENT belief, which
  updates on answers and on the opponent's moves (the belief is maintained
  outside the model):
      SAFE       argmax P_b(accept), ties -> own utility, ties -> deal order
      AMBITIOUS  argmax own utility among deals with P_b(accept) > 0
      PLAN       argmax over proposals of the exact NO-ASK planner's Q at this
                 (state, belief) — "what the planner would offer if it could not
                 ask"; ties -> own utility, ties -> deal order. Needs ``plan_oracle``
                 (a menu-free Oracle with ask_allowed=False). Added 2026-10-29 after
                 the costlessness check showed the two-candidate menu cost the
                 no-ask planner 0.015 in every focal-opens cell at H = 4 and that no
                 EIG / P(accept) / utility rule reproduces its opening offer (two
                 deals tie on EIG and P(accept); the planner prefers the one whose
                 accepting types leave the more useful residual belief). With PLAN
                 in the menu the no-ask planner is never handicapped, so a menu
                 artefact can only DEFLATE the measured value of asking.
      BALANCED   (fill only) argmax P_b(accept) * (U_own - r)
  over the pool of individually-rational deals (U_own >= r) minus the focal's
  own currently standing deal. Candidates are deduplicated in that order.
* planner completeness (``plan_ask_oracle``, a menu-free Oracle with ASK allowed;
  added 2026-10-29 when the reviewer's stopping rule fired at H = 3: the two
  highest-EIG questions tie on information with the question the ASK planner
  wants, and the planner's question has the higher decision value, so the
  question restriction cost the ASK planner 0.015-0.018 and removed the whole
  value of asking in the focal-opens cells): the ASK planner's best proposal
  (``PLAN_ASK``) and best question (``PLAN_Q``) are appended whenever they are
  not already present. With PLAN, PLAN_ASK and PLAN_Q in the menu neither
  planner is ever handicapped, so V_menu = V_full for both by construction and
  the menu artefact on Delta* is exactly zero; the costlessness check then
  verifies the code rather than measuring a loss. P_b(accept) is the kernel's exact probability
  of ACCEPT at the opponent node that follows the proposal, marginalised over
  the belief; it therefore reflects the opponent's aspiration at that point in
  the budget and the deadline rule.

``restrict_offers`` / ``restrict_questions`` switch each restriction off (all 27
deals / all unasked questions), so the costlessness check can attribute any lost
value to one restriction or the other. At the deadline the menu is {ACCEPT, WALK}.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, DEALS_BY_ID, Deal, Ranking, utility

from . import belief_live as bl
from .opponent import T1Kernel
from .protocol_live import ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, Action, LiveConfig, PublicState, apply, legal_actions

SAFE, AMBITIOUS, PLAN, PLAN_ASK, PLAN_Q, BALANCED = "SAFE", "AMBITIOUS", "PLAN", "PLAN_ASK", "PLAN_Q", "BALANCED"


@dataclass(frozen=True)
class Row:
    action: Action
    tag: Optional[str]                 # offers: SAFE | AMBITIOUS | PLAN | BALANCED (None under the full menu)
    eig_bits: float                    # exact expected information gain of the opponent's response / answer
    p_accept: Optional[float]          # offers only
    own_utility: Optional[Fraction]    # offers and ACCEPT

    @property
    def key(self) -> str:
        return self.action.key

    def to_json(self) -> dict:
        return {"key": self.key, "kind": self.action.kind, "deal_id": self.action.deal_id,
                "question_id": self.action.question_id, "tag": self.tag, "eig_bits": self.eig_bits,
                "p_accept": self.p_accept,
                "own_utility": None if self.own_utility is None else {"exact": str(self.own_utility), "float": float(self.own_utility)}}


class Menu:
    def __init__(self, cfg: LiveConfig, focal: str, focal_ranking: Ranking, kernel: T1Kernel, *,
                 restrict_offers: bool = True, restrict_questions: bool = True, n_offers: int = 3, n_questions: int = 2,
                 plan_oracle=None, plan_ask_oracle=None):
        """``plan_oracle``: a menu-free ``Oracle(..., ask_allowed=False)`` for the same focal; it
        defines the PLAN candidate. ``plan_ask_oracle``: a menu-free ``Oracle(..., ask_allowed=True)``;
        it defines PLAN_ASK and PLAN_Q. ``None`` drops the corresponding candidates (``None, None``
        is the pre-2026-10-29 heuristic-only rule)."""
        self.plan_oracle = plan_oracle
        self.plan_ask_oracle = plan_ask_oracle
        if plan_oracle is not None:
            assert plan_oracle.menu is None and not plan_oracle.ask_allowed and plan_oracle.focal == focal
        if plan_ask_oracle is not None:
            assert plan_ask_oracle.menu is None and plan_ask_oracle.ask_allowed and plan_ask_oracle.focal == focal
        self.cfg = cfg
        self.focal = focal
        self.ranking = focal_ranking
        self.kernel = kernel
        self.restrict_offers = restrict_offers
        self.restrict_questions = restrict_questions
        self.n_offers = n_offers
        self.n_questions = n_questions
        self._u = {d.id: utility(focal, focal_ranking, d) for d in ALL_DEALS}
        self._cache: Dict[tuple, Tuple[Row, ...]] = {}

    # -- exact annotations ---------------------------------------------------------

    def response_likelihoods(self, state: PublicState, action: Action):
        """Likelihoods (per opponent type) of the opponent's next observable, or None
        if the action ends the game."""
        if action.kind == ASK:
            return bl.answer_likelihoods(self.kernel, action.question_id)
        nxt = apply(state, action)
        if nxt.terminal is not None:
            return None
        return bl.action_likelihoods(self.kernel, nxt)

    def eig(self, state: PublicState, belief: bl.Belief, action: Action) -> float:
        liks = self.response_likelihoods(state, action)
        return 0.0 if liks is None else bl.eig_bits(belief, liks)

    def p_accept(self, state: PublicState, belief: bl.Belief, deal_id: str) -> float:
        liks = bl.action_likelihoods(self.kernel, apply(state, Action(PROPOSE, deal_id=deal_id)))
        acc = liks.get(Action(ACCEPT))
        return 0.0 if acc is None else bl.marginal(belief, acc)

    # -- candidates ------------------------------------------------------------------

    def offer_candidates(self, state: PublicState, belief: bl.Belief) -> List[Tuple[Deal, Optional[str]]]:
        r = self.cfg.reservation
        own_standing = state.standing.deal_id if (state.standing and state.standing.proposer == self.focal) else None
        pool = [d for d in ALL_DEALS if self._u[d.id] >= r and d.id != own_standing] or [d for d in ALL_DEALS if d.id != own_standing]
        if not self.restrict_offers:
            return [(d, None) for d in pool]
        pa = {d.id: self.p_accept(state, belief, d.id) for d in pool}
        u = self._u
        idx = {d.id: i for i, d in enumerate(ALL_DEALS)}
        tagged: List[Tuple[str, Deal]] = []
        tagged.append((SAFE, max(pool, key=lambda d: (pa[d.id], u[d.id], -idx[d.id]))))
        positive = [d for d in pool if pa[d.id] > 1e-12]
        if positive:
            tagged.append((AMBITIOUS, max(positive, key=lambda d: (u[d.id], pa[d.id], -idx[d.id]))))
        if self.plan_oracle is not None:
            q = {d.id: self.plan_oracle.q_value(state, belief, Action(PROPOSE, deal_id=d.id)) for d in pool}
            tagged.append((PLAN, max(pool, key=lambda d: (round(q[d.id], 9), u[d.id], -idx[d.id]))))
        tagged.append((BALANCED, max(pool, key=lambda d: (pa[d.id] * float(u[d.id] - r), u[d.id], -idx[d.id]))))
        out, seen = [], set()
        for tag, d in tagged:
            if d.id in seen:
                continue
            seen.add(d.id)
            out.append((d, tag))
            if len(out) == self.n_offers:
                break
        if self.plan_ask_oracle is not None:   # planner completeness: the ASK planner's best proposal
            q = {d.id: self.plan_ask_oracle.q_value(state, belief, Action(PROPOSE, deal_id=d.id)) for d in pool}
            best = max(pool, key=lambda d: (round(q[d.id], 9), u[d.id], -idx[d.id]))
            if best.id not in seen:
                out.append((best, PLAN_ASK))
        return out

    def question_candidates(self, state: PublicState, belief: bl.Belief) -> List[str]:
        asked = state.asked_by(self.focal)
        unasked = [q.id for q in BANK if q.id not in asked]
        if not self.restrict_questions:
            return unasked
        scored = [(-bl.eig_bits(belief, bl.answer_likelihoods(self.kernel, qid)), i, qid) for i, qid in enumerate(unasked)]
        scored.sort()
        out = [qid for _, _, qid in scored[: self.n_questions]]
        if self.plan_ask_oracle is not None and unasked:   # planner completeness: the ASK planner's best question
            q = {qid: self.plan_ask_oracle.q_value(state, belief, Action(ASK, question_id=qid)) for qid in unasked}
            best = max(unasked, key=lambda qid: (round(q[qid], 9), -unasked.index(qid)))
            if best not in out:
                out.append(best)
        return out

    def question_tag(self, state: PublicState, belief: bl.Belief, qid: str) -> Optional[str]:
        """PLAN_Q if the question is in the menu only through planner completeness."""
        if self.plan_ask_oracle is None:
            return None
        asked = state.asked_by(self.focal)
        unasked = [q.id for q in BANK if q.id not in asked]
        scored = sorted((-bl.eig_bits(belief, bl.answer_likelihoods(self.kernel, q)), i, q) for i, q in enumerate(unasked))
        top = [q for _, _, q in scored[: self.n_questions]]
        return None if (qid in top or not self.restrict_questions) else PLAN_Q

    # -- the menu ----------------------------------------------------------------------

    def rows(self, state: PublicState, belief: bl.Belief, include_questions: bool = True) -> List[Row]:
        key = (state.key(), bl.round_belief(belief), include_questions)
        if key in self._cache:
            return list(self._cache[key])
        legal = set(legal_actions(state, self.focal))
        st = state.standing
        out: List[Row] = []
        if Action(ACCEPT) in legal:
            out.append(Row(Action(ACCEPT), None, 0.0, None, self._u[st.deal_id]))
        if Action(REJECT) in legal:
            a = Action(REJECT)
            out.append(Row(a, None, self.eig(state, belief, a), None, None))
        if Action(HOLD) in legal:
            a = Action(HOLD)
            out.append(Row(a, None, self.eig(state, belief, a), None, None))
        if not state.at_deadline:
            if include_questions:
                for qid in self.question_candidates(state, belief):
                    a = Action(ASK, question_id=qid)
                    out.append(Row(a, self.question_tag(state, belief, qid), self.eig(state, belief, a), None, None))
            for d, tag in self.offer_candidates(state, belief):
                a = Action(PROPOSE, deal_id=d.id)
                out.append(Row(a, tag, self.eig(state, belief, a), self.p_accept(state, belief, d.id), self._u[d.id]))
        out.append(Row(Action(WALK), None, 0.0, None, None))
        assert all(r.action in legal for r in out)
        self._cache[key] = tuple(out)
        return out

    def actions(self, include_questions: bool = True) -> Callable[[PublicState, bl.Belief], List[Action]]:
        """The oracle hook: ``menu(state, belief) -> [Action]``."""
        return lambda state, belief: [r.action for r in self.rows(state, belief, include_questions)]
