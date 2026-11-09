"""Tier-1 opponent: a history-responsive scripted negotiator with a KNOWN kernel.

It is not a frozen threshold responder (design document section 3.3). Its
behaviour depends on its private ranking, the standing offer, the remaining
budget, and the public history (it never re-proposes a deal the other side has
rejected, nor the deal it currently has standing). Every probability it uses is
exposed through :meth:`T1Kernel.distribution`, so the oracle and the matched
belief filter can use the exact same numbers. It never asks questions in T1.

Kernel (parameters in :class:`KernelParams`):

* **Aspiration** a(remaining) = r + (alpha0 - r) * remaining / H: it concedes
  linearly as the budget is spent and accepts anything individually rational
  at the deadline.
* **Facing the other side's offer d:** if U(d) >= a it ACCEPTs (with prob
  1 - accept_noise; the noise branch behaves like a rejection); otherwise it
  COUNTERs with prob p_counter (a PROPOSE from its proposal distribution; only
  if an opportunity remains) and REJECTs otherwise.
* **Proposal distribution:** candidates = deals with U >= a, minus deals the
  other side has rejected, minus its own currently-standing deal; if empty,
  fall back to U >= r, then to all deals. Keep the ``shortlist`` best by own
  utility (0 = all) and draw with softmax(beta * U). Ties by deal order.
* **Own offer standing (the other side asked instead of answering):** HOLD with
  prob p_hold, otherwise re-PROPOSE (a concession draw) if an opportunity
  remains.
* **No standing offer, its move:** PROPOSE from the distribution.
* **Deadline decision:** ACCEPT iff U(d) >= r, else WALK. It never walks early.
* **Answers:** truthful, flipped with prob ``cfg.answer_noise``.

Widened family (CD-NOUL-2 Phase 0, DESIGN_NOUL2 §3.3a; both default 0 = the kernel above, bit for bit):

* **eps_propose:** at every non-deadline node, mix weight eps onto a uniform draw over ALL legal
  proposals (every deal), so no proposal has zero probability under any type.
* **p_ask:** at every non-deadline node with an unasked question, ask with prob p_ask, uniformly over the
  unasked questions — the same for every type, so an observed opponent question carries no type evidence.

  P(a | θ) = p_ask·U_ask(a) + (1 − p_ask)·[(1 − eps)·P_T1(a | θ) + eps·U_propose(a)]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Dict, List, Optional

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, DEALS_BY_ID, Deal, Ranking, utility

from .protocol_live import (
    ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, Action, LiveConfig, PublicState, legal_actions, other, truthful_answer,
)


@dataclass(frozen=True)
class KernelParams:
    alpha0: Fraction = Fraction(4, 5)   # initial aspiration
    beta: float = 8.0                   # softmax sharpness over own utility
    p_counter: float = 1.0              # counter (vs plain reject) when the offer is below aspiration
    accept_noise: float = 0.0           # P(reject an acceptable offer)
    p_hold: float = 1.0                 # hold (vs concede) when own offer stands after being asked
    shortlist: int = 3                  # top-k own-utility candidates proposed from (0 = all)
    eps_propose: float = 0.0            # widened family: mix weight on a uniform draw over all legal proposals
    p_ask: float = 0.0                  # widened family: type-independent P(ask), uniform over unasked questions

    WIDE_FIELDS = ("eps_propose", "p_ask")

    def to_json(self) -> dict:
        d = asdict(self)
        d["alpha0"] = str(self.alpha0)
        for k in self.WIDE_FIELDS:      # omitted at the default so every pre-widening JSON (and hash) is unchanged
            if d[k] == 0.0:
                del d[k]
        return d


class T1Kernel:
    def __init__(self, role: str, cfg: LiveConfig, params: KernelParams = KernelParams()):
        self.role = role
        self.cfg = cfg
        self.p = params
        self._cache: Dict[tuple, Dict[Action, float]] = {}

    def aspiration(self, remaining: int) -> Fraction:
        r = self.cfg.reservation
        return r + (self.p.alpha0 - r) * Fraction(remaining, self.cfg.horizon)

    # -- proposal distribution -------------------------------------------------

    def proposal_distribution(self, state: PublicState, ranking: Ranking) -> Dict[Action, float]:
        a = self.aspiration(state.remaining)
        rejected = state.rejected_by(other(self.role))
        own_standing = state.standing.deal_id if (state.standing and state.standing.proposer == self.role) else None
        u = {d: utility(self.role, ranking, d) for d in ALL_DEALS}

        def pool(threshold: Fraction) -> List[Deal]:
            return [d for d in ALL_DEALS if u[d] >= threshold and d.id not in rejected and d.id != own_standing]

        cands = pool(a) or pool(self.cfg.reservation) or [d for d in ALL_DEALS if d.id != own_standing] or list(ALL_DEALS)
        cands.sort(key=lambda d: (-u[d], ALL_DEALS.index(d)))
        if self.p.shortlist > 0:
            cands = cands[: self.p.shortlist]
        w = [math.exp(self.p.beta * float(u[d])) for d in cands]
        z = sum(w)
        return {Action(PROPOSE, deal_id=d.id): wi / z for d, wi in zip(cands, w)}

    # -- full response distribution ---------------------------------------------

    def distribution(self, state: PublicState, ranking: Ranking) -> Dict[Action, float]:
        """P(action | state, type) over this opponent's legal actions; sums to 1."""
        key = (state.key(), ranking)
        if key in self._cache:
            return self._cache[key]
        assert state.turn == self.role and state.terminal is None
        st = state.standing
        out: Dict[Action, float] = {}
        if state.at_deadline:
            if st is not None and st.proposer != self.role:
                ok = utility(self.role, ranking, DEALS_BY_ID[st.deal_id]) >= self.cfg.reservation
                out = {Action(ACCEPT): 1.0} if ok else {Action(WALK): 1.0}
            else:
                out = {Action(WALK): 1.0}
        elif st is not None and st.proposer != self.role:
            u = utility(self.role, ranking, DEALS_BY_ID[st.deal_id])
            p_acc = (1.0 - self.p.accept_noise) if u >= self.aspiration(state.remaining) else 0.0
            p_rej_branch = 1.0 - p_acc
            if p_acc > 0:
                out[Action(ACCEPT)] = p_acc
            if p_rej_branch > 0:
                pc = self.p.p_counter if state.remaining > 0 else 0.0
                for act, pr in self.proposal_distribution(state, ranking).items():
                    out[act] = out.get(act, 0.0) + p_rej_branch * pc * pr
                if 1.0 - pc > 0:
                    out[Action(REJECT)] = out.get(Action(REJECT), 0.0) + p_rej_branch * (1.0 - pc)
        elif st is not None and st.proposer == self.role:
            ph = self.p.p_hold if state.remaining > 0 else 1.0
            if ph > 0:
                out[Action(HOLD)] = ph
            if 1.0 - ph > 0:
                for act, pr in self.proposal_distribution(state, ranking).items():
                    out[act] = out.get(act, 0.0) + (1.0 - ph) * pr
        else:
            out = dict(self.proposal_distribution(state, ranking))
        if not state.at_deadline and (self.p.eps_propose > 0 or self.p.p_ask > 0):
            out = self._widen(state, out)
        total = sum(out.values())
        assert abs(total - 1.0) < 1e-9, total
        self._cache[key] = out
        return out

    def _widen(self, state: PublicState, core: Dict[Action, float]) -> Dict[Action, float]:
        legal = legal_actions(state, self.role)
        props = [a for a in legal if a.kind == PROPOSE]
        asks = [a for a in legal if a.kind == ASK]
        out = dict(core)
        eps = self.p.eps_propose if props else 0.0
        if eps > 0:
            out = {a: (1.0 - eps) * p for a, p in out.items()}
            for a in props:
                out[a] = out.get(a, 0.0) + eps / len(props)
        q = self.p.p_ask if asks else 0.0
        if q > 0:
            out = {a: (1.0 - q) * p for a, p in out.items()}
            for a in asks:
                out[a] = out.get(a, 0.0) + q / len(asks)
        return out

    def answer_distribution(self, question_id: str, ranking: Ranking) -> Dict[str, float]:
        t = truthful_answer(question_id, ranking)
        n = float(self.cfg.noise_for(question_id))
        o = "no" if t == "yes" else "yes"
        return {t: 1.0 - n, o: n} if n > 0 else {t: 1.0}

    # -- sampling ---------------------------------------------------------------

    def sample(self, state: PublicState, ranking: Ranking, rng: random.Random) -> Action:
        dist = self.distribution(state, ranking)
        x = rng.random()
        acc = 0.0
        for act, pr in dist.items():
            acc += pr
            if x < acc:
                return act
        return act  # float slack

    def sample_answer(self, question_id: str, ranking: Ranking, rng: random.Random) -> str:
        dist = self.answer_distribution(question_id, ranking)
        x = rng.random()
        acc = 0.0
        for ans, pr in dist.items():
            acc += pr
            if x < acc:
                return ans
        return ans
