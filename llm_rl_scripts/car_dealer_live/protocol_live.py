"""Live Car-Dealer protocol (``live-car-dealer-v0.1``): the discretized bilateral
game plus a public ASK action. Pure functions over an immutable public state.

Rules (design document section 3.2, with the decisions Phase 0 had to make):

* Actions: ``PROPOSE d`` (any of the 27 deals; replaces the standing offer),
  ``ACCEPT`` (the other side's standing offer; ends with that deal), ``REJECT``
  (clears the other side's standing offer), ``ASK q`` (one of the nine binary
  TOP/ABOVE/BOTTOM questions; answered immediately by the other side, answer is
  public), ``WALK`` (ends with no deal), ``HOLD`` (legal only while your OWN
  offer stands: pass without changing it).
* Budget: ``horizon`` H opportunities shared by both sides; PROPOSE and ASK each
  consume one; ACCEPT / REJECT / WALK / HOLD are free.
* Initiative passes to the other side after every action (the respondent sees
  which question was asked). A question does not cancel a standing offer, so the
  party whose own offer stands may face its own offer; it can HOLD, re-PROPOSE
  (concede), ASK or WALK. HOLD cannot loop: only the proposer may hold, and after
  a hold the other side faces the offer.
* Deadline: when a consuming move leaves 0 opportunities, the non-proposer of a
  standing offer (if any) gets one final decision, ACCEPT or WALK; with no
  standing offer the episode ends with no deal.
* Payoff: an accepted deal pays each side its exact utility; no deal pays r.
  The oracle's objective is *net gain* U - r - (ask_cost x own asks); no deal = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK, BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, DEALS_BY_ID, Ranking, utility
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

ENV_VERSION = "live-car-dealer-v0.1"

PROPOSE, ACCEPT, REJECT, ASK, WALK, HOLD = "PROPOSE", "ACCEPT", "REJECT", "ASK", "WALK", "HOLD"
ROLES = ("buyer", "seller")


def other(role: str) -> str:
    return "seller" if role == "buyer" else "buyer"


@dataclass(frozen=True)
class LiveConfig:
    regime: str = "easy"
    horizon: int = 4                       # shared opportunities
    ask_cost: Fraction = Fraction(0)       # charged to the asker, utility units
    answer_noise: Fraction = Fraction(0)   # P(untruthful answer)

    @property
    def reservation(self) -> Fraction:
        return RESERVATIONS[self.regime]

    def to_json(self) -> dict:
        return {"env": ENV_VERSION, "regime": self.regime, "reservation": str(self.reservation),
                "horizon": self.horizon, "ask_cost": str(self.ask_cost), "answer_noise": str(self.answer_noise)}


@dataclass(frozen=True)
class Action:
    kind: str
    deal_id: Optional[str] = None
    question_id: Optional[str] = None

    @property
    def key(self) -> str:
        if self.kind == PROPOSE:
            return f"PROPOSE {self.deal_id}"
        if self.kind == ASK:
            return f"ASK {self.question_id}"
        return self.kind

    @property
    def consumes(self) -> bool:
        return self.kind in (PROPOSE, ASK)


@dataclass(frozen=True)
class Standing:
    proposer: str
    deal_id: str


@dataclass(frozen=True)
class Event:
    actor: str
    action: Action
    answer: Optional[str] = None  # ASK only



@dataclass(frozen=True)
class Terminal:
    deal_id: Optional[str]
    reason: str  # accept | walk | deadline


@dataclass(frozen=True)
class PublicState:
    opener: str
    turn: str                      # who decides next (deadline rule already applied)
    remaining: int
    standing: Optional[Standing]
    events: Tuple[Event, ...] = ()
    terminal: Optional[Terminal] = None

    # -- derived public facts --------------------------------------------
    def asked_by(self, role: str) -> FrozenSet[str]:
        return frozenset(e.action.question_id for e in self.events if e.actor == role and e.action.kind == ASK)

    def rejected_by(self, role: str) -> FrozenSet[str]:
        """Deals this role has REJECTed (the kernel never re-proposes them)."""
        out = set()
        standing: Optional[Standing] = None
        for e in self.events:
            if e.action.kind == PROPOSE:
                standing = Standing(e.actor, e.action.deal_id)
            elif e.action.kind == REJECT and standing is not None and e.actor == role:
                out.add(standing.deal_id)
                standing = None
        return frozenset(out)

    def proposed_by(self, role: str) -> FrozenSet[str]:
        return frozenset(e.action.deal_id for e in self.events if e.actor == role and e.action.kind == PROPOSE)

    @property
    def at_deadline(self) -> bool:
        return self.remaining == 0

    def key(self):
        """Public, history-sufficient key for memoisation (kernel + legality depend
        only on these)."""
        return (self.turn, self.remaining, self.standing, self.asked_by("buyer"), self.asked_by("seller"),
                self.rejected_by("buyer"), self.rejected_by("seller"), self.terminal)


def initial_state(opener: str) -> PublicState:
    return PublicState(opener=opener, turn=opener, remaining=0, standing=None)  # remaining set by new_game


def new_game(cfg: LiveConfig, opener: str) -> PublicState:
    return PublicState(opener=opener, turn=opener, remaining=cfg.horizon, standing=None)


def legal_actions(state: PublicState, actor: Optional[str] = None) -> List[Action]:
    if state.terminal is not None:
        return []
    actor = actor or state.turn
    assert actor == state.turn, "not this actor's move"
    st = state.standing
    if state.at_deadline:
        if st is not None and st.proposer != actor:
            return [Action(ACCEPT), Action(WALK)]
        return [Action(WALK)]
    acts: List[Action] = []
    if st is not None and st.proposer != actor:
        acts += [Action(ACCEPT), Action(REJECT)]
    if st is not None and st.proposer == actor:
        acts.append(Action(HOLD))
    acts += [Action(PROPOSE, deal_id=d.id) for d in ALL_DEALS]
    asked = state.asked_by(actor)
    acts += [Action(ASK, question_id=q.id) for q in BANK if q.id not in asked]
    acts.append(Action(WALK))
    return acts


def truthful_answer(question_id: str, ranking: Ranking) -> str:
    return BANK_BY_ID[question_id].truthful_answer(ranking)


def apply(state: PublicState, action: Action, answer: Optional[str] = None) -> PublicState:
    """Apply ``action`` by ``state.turn``. ASK needs the (already drawn) answer."""
    actor = state.turn
    assert action in legal_actions(state, actor), f"illegal: {action.key} at {state.key()}"
    ev = Event(actor, action, answer if action.kind == ASK else None)
    events = state.events + (ev,)
    nxt = other(actor)
    if action.kind == ACCEPT:
        return PublicState(state.opener, nxt, state.remaining, state.standing, events, Terminal(state.standing.deal_id, "accept"))
    if action.kind == WALK:
        return PublicState(state.opener, nxt, state.remaining, state.standing, events, Terminal(None, "walk"))
    if action.kind == REJECT:
        return PublicState(state.opener, nxt, state.remaining, None, events)
    if action.kind == HOLD:
        return PublicState(state.opener, nxt, state.remaining, state.standing, events)
    # consuming moves
    assert answer is not None or action.kind == PROPOSE
    remaining = state.remaining - 1
    standing = Standing(actor, action.deal_id) if action.kind == PROPOSE else state.standing
    if remaining == 0:
        if standing is None:
            return PublicState(state.opener, nxt, 0, None, events, Terminal(None, "deadline"))
        # final decision belongs to the non-proposer of the standing offer
        return PublicState(state.opener, other(standing.proposer), 0, standing, events)
    return PublicState(state.opener, nxt, remaining, standing, events)


def gain(role: str, ranking: Ranking, terminal: Terminal, r: Fraction) -> Fraction:
    """Gross gain above the outside option: U(d) - r on a deal, 0 otherwise."""
    if terminal.deal_id is None:
        return Fraction(0)
    return utility(role, ranking, DEALS_BY_ID[terminal.deal_id]) - r


def asks_by(state: PublicState, role: str) -> int:
    return sum(1 for e in state.events if e.actor == role and e.action.kind == ASK)
