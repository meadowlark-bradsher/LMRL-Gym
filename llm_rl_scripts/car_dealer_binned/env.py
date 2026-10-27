"""Binned Car Dealer, Condition B: information acquisition with opportunity cost.

One *learner* (buyer or seller, private ranking known to itself) negotiates
against a *frozen scripted opponent* whose private ranking is hidden. Each
decision the learner picks ONE option from a small public menu:

  * ASK q      consume the turn; the opponent answers under the public response
               model (``questions.py``); the belief updates exactly.
  * OFFER d    the opponent ACCEPTS iff U_opp(d) >= r (individually rational),
               else REJECTS; acceptance closes the game at U_learner(d);
               rejection consumes the turn and is itself informative (it
               eliminates every type for which d was acceptable).

Horizon: ``rounds`` regular decisions (default 6) where the menu mixes
questions and offers, then ONE final decision (t = rounds + 1) whose menu is
offers only (the design doc's "six rounds, followed by a final offer"). A
rejected final offer ends the game at the reservation value r for both.

Frozen opponent (public, deliberately simple): truthful answers with known
noise; deterministic IR acceptance (optionally with a noise floor); never
counter-offers, never asks, maintains no belief. This keeps the learner's
problem a single-agent POMDP with an exact dynamic-programming reference
(``reference.py``). Bilateral belief tracking (design doc section 5) is
therefore degenerate in this version: only the learner's entropy moves.

Menu rule (a deterministic, public function of the public state, so the
reference policy and every arm face identical menus):
  * questions: the ``n_questions`` unasked questions with the highest current
    EIG (ties -> bank order). Zero-EIG questions are NOT removed; they become
    the natural low-information distractors.
  * offers: two named candidates computed from the public belief and the
    learner's own utility, using the public acceptance model, drawn from the
    deals worth at least r to the learner itself (own individual rationality),
      SAFE       argmax P_b(accept), ties -> own utility, ties -> deal id
      AMBITIOUS  argmax own utility among deals with P_b(accept) > 0,
                 ties -> P_b(accept), ties -> deal id
    and, only if those coincide, BALANCED = argmax of the myopic value
      m(d) = P_b(accept d) * U_own(d) + (1 - P_b(accept d)) * r.
    The reference policy optimises *within* these menus (design doc section 4,
    Q*(b_t, a) for a in A_t). ``reference.py`` can also solve the unconstrained
    game (all questions, all 27 deals) to bound what the menu rule costs.

Everything numeric is exact (Fractions); EIG is a float in bits.
"""

from __future__ import annotations

import random
from functools import lru_cache
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from llm_rl_scripts.car_dealer_discrete.game import (
    ALL_DEALS,
    ALL_RANKINGS,
    DEALS_BY_ID,
    Deal,
    Ranking,
    utility,
)
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

from . import beliefs
from .beliefs import Belief, Likelihood
from .questions import BANK, BANK_BY_ID, Question

GAME_VERSION = "binned-car-dealer-b-v0.1"

ASK = "ASK"
OFFER = "OFFER"
ACCEPT = "accept"
REJECT = "reject"


def other_role(role: str) -> str:
    return "seller" if role == "buyer" else "buyer"


@dataclass(frozen=True)
class Config:
    regime: str = "easy"
    rounds: int = 6                      # regular decisions; t = rounds+1 is offers-only
    n_questions: int = 2                 # question slots per regular menu
    n_offers: int = 2                    # offer slots per menu
    answer_noise: Fraction = Fraction(0)  # P(untruthful answer)
    accept_noise: Fraction = Fraction(0)  # P(IR deal rejected) = P(non-IR deal accepted)

    @property
    def reservation(self) -> Fraction:
        return RESERVATIONS[self.regime]

    @property
    def final_t(self) -> int:
        return self.rounds + 1

    def to_json(self) -> dict:
        return {
            "game": GAME_VERSION,
            "regime": self.regime,
            "reservation": str(self.reservation),
            "rounds": self.rounds,
            "final_decision_t": self.final_t,
            "n_questions": self.n_questions,
            "n_offers": self.n_offers,
            "answer_noise": str(self.answer_noise),
            "accept_noise": str(self.accept_noise),
            "bank": [q.id for q in BANK],
        }


@dataclass(frozen=True)
class Option:
    kind: str                      # ASK | OFFER
    target: str                    # question id | deal id
    tag: Optional[str] = None      # offers: SAFE | AMBITIOUS | BALANCED

    @property
    def key(self) -> str:
        return f"{self.kind} {self.target}"


@dataclass(frozen=True)
class Turn:
    option: Option
    outcome: str  # yes | no | accept | reject


@dataclass(frozen=True)
class State:
    """Public state of the learner's decision problem (nothing hidden here)."""

    role: str
    own_ranking: Ranking
    t: int
    belief: Belief
    asked: Tuple[str, ...]
    history: Tuple[Turn, ...] = ()

    def key(self, cfg: "Config"):
        """Memo key for the reference policy.

        The menu depends only on (t, belief, asked). Under zero answer noise a
        question whose answer is already determined by the support has EIG 0
        whether or not it has been asked, so it is dropped from the key: states
        that differ only in such questions have identical menus (the menu ranks
        unasked questions by EIG, ties by bank order) and identical values.
        With answer noise every unasked question stays informative and the
        full asked set is the key.
        """
        if cfg.answer_noise == 0:
            sup = beliefs.support(self.belief)
            live = frozenset(
                q for q in self.asked
                if len({BANK_BY_ID[q].truthful_answer(th) for th in sup}) > 1
            )
            return (self.t, self.belief, live)
        return (self.t, self.belief, frozenset(self.asked))


# ---------------------------------------------------------------- likelihoods


@lru_cache(maxsize=None)
def acceptance_likelihood(role: str, deal: Deal, r: Fraction, noise: Fraction) -> Dict[str, Likelihood]:
    """{accept: (P(accept | theta_opp)), reject: ...} under the public model."""
    opp = other_role(role)
    acc = tuple(
        (1 - noise) if utility(opp, th, deal) >= r else noise for th in ALL_RANKINGS
    )
    rej = tuple(1 - a for a in acc)
    return {ACCEPT: acc, REJECT: rej}


def option_likelihoods(opt: Option, state: State, cfg: Config) -> Dict[str, Likelihood]:
    if opt.kind == ASK:
        return BANK_BY_ID[opt.target].likelihoods(cfg.answer_noise)
    return acceptance_likelihood(state.role, DEALS_BY_ID[opt.target], cfg.reservation, cfg.accept_noise)


def p_accept(state: State, deal: Deal, cfg: Config) -> Fraction:
    lik = acceptance_likelihood(state.role, deal, cfg.reservation, cfg.accept_noise)[ACCEPT]
    return beliefs.marginal(state.belief, lik)


def own_utility(state: State, deal: Deal) -> Fraction:
    return utility(state.role, state.own_ranking, deal)


def myopic_value(state: State, deal: Deal, cfg: Config) -> Fraction:
    p = p_accept(state, deal, cfg)
    return p * own_utility(state, deal) + (1 - p) * cfg.reservation


# ---------------------------------------------------------------- menu rule


def offer_candidates(state: State, cfg: Config) -> List[Option]:
    """SAFE, AMBITIOUS (and BALANCED only to fill), deduplicated, in that order."""
    pa = {d: p_accept(state, d, cfg) for d in ALL_DEALS}
    uo = {d: own_utility(state, d) for d in ALL_DEALS}
    # Candidates are restricted to deals the learner itself would accept
    # (U_own >= r): a rational agent never proposes a deal it would refuse, and
    # the discretized game treats such acceptances as invalid. The fallback to
    # all deals only matters if no own-IR deal exists, which cannot happen (the
    # learner's ideal deal is worth 1).
    pool = [d for d in ALL_DEALS if uo[d] >= cfg.reservation] or list(ALL_DEALS)
    safe = max(pool, key=lambda d: (pa[d], uo[d], -ALL_DEALS.index(d)))
    tagged: List[Tuple[str, Deal]] = [("SAFE", safe)]
    positive = [d for d in pool if pa[d] > 0]
    if positive:
        amb = max(positive, key=lambda d: (uo[d], pa[d], -ALL_DEALS.index(d)))
        tagged.append(("AMBITIOUS", amb))
    bal = max(pool, key=lambda d: (myopic_value(state, d, cfg), uo[d], -ALL_DEALS.index(d)))
    tagged.append(("BALANCED", bal))
    out: List[Option] = []
    seen = set()
    for tag, d in tagged:
        if d.id in seen:
            continue
        seen.add(d.id)
        out.append(Option(OFFER, d.id, tag))
        if len(out) == cfg.n_offers:
            break
    return out


def question_candidates(state: State, cfg: Config) -> List[Option]:
    unasked = [q for q in BANK if q.id not in state.asked]
    scored = [
        (-beliefs.expected_information_gain(state.belief, q.likelihoods(cfg.answer_noise)), i, q)
        for i, q in enumerate(unasked)
    ]
    scored.sort(key=lambda x: (x[0], x[1]))
    return [Option(ASK, q.id) for _, _, q in scored[: cfg.n_questions]]


def menu(state: State, cfg: Config) -> List[Option]:
    """The legal options at ``state``: questions first, then offers."""
    if state.t > cfg.final_t:
        return []
    opts: List[Option] = []
    if state.t <= cfg.rounds:
        opts += question_candidates(state, cfg)
    opts += offer_candidates(state, cfg)
    return opts


# ---------------------------------------------------------------- dynamics


def initial_state(role: str, own_ranking: Ranking) -> State:
    return State(role=role, own_ranking=own_ranking, t=1, belief=beliefs.uniform(), asked=())


def transition(state: State, opt: Option, outcome: str, cfg: Config) -> Tuple[Optional[State], Optional[Fraction]]:
    """Apply an outcome. Returns (next_state, None) or (None, terminal_utility)."""
    liks = option_likelihoods(opt, state, cfg)
    if outcome not in liks:
        raise ValueError(f"outcome {outcome!r} not valid for {opt.key}")
    if opt.kind == OFFER and outcome == ACCEPT:
        return None, own_utility(state, DEALS_BY_ID[opt.target])
    if opt.kind == OFFER and state.t >= cfg.final_t:
        return None, cfg.reservation
    new_belief = beliefs.update(state.belief, liks[outcome])
    asked = state.asked + ((opt.target,) if opt.kind == ASK else ())
    nxt = State(
        role=state.role,
        own_ranking=state.own_ranking,
        t=state.t + 1,
        belief=new_belief,
        asked=asked,
        history=state.history + (Turn(opt, outcome),),
    )
    return nxt, None


class ScriptedOpponent:
    """Frozen responder of hidden type ``ranking``. Public model, private type."""

    def __init__(self, role: str, ranking: Ranking, cfg: Config, rng: Optional[random.Random] = None):
        self.role = role
        self.ranking = ranking
        self.cfg = cfg
        self.rng = rng or random.Random(0)

    def respond(self, opt: Option, learner_role: str) -> str:
        assert other_role(learner_role) == self.role
        if opt.kind == ASK:
            truthful = BANK_BY_ID[opt.target].truthful_answer(self.ranking)
            if self.cfg.answer_noise > 0 and self.rng.random() < float(self.cfg.answer_noise):
                return "no" if truthful == "yes" else "yes"
            return truthful
        d = DEALS_BY_ID[opt.target]
        ir = utility(self.role, self.ranking, d) >= self.cfg.reservation
        flip = self.cfg.accept_noise > 0 and self.rng.random() < float(self.cfg.accept_noise)
        return ACCEPT if (ir != flip) else REJECT


@dataclass
class EpisodeResult:
    role: str
    own_ranking: Ranking
    opp_ranking: Ranking
    turns: List[Turn] = field(default_factory=list)
    accepted_deal_id: Optional[str] = None
    learner_utility: Optional[Fraction] = None
    final_belief: Optional[Belief] = None

    @property
    def agreement(self) -> bool:
        return self.accepted_deal_id is not None


def annotate(state: State, cfg: Config) -> List[dict]:
    """Per-option public annotations: EIG, P(accept), own utility, myopic value."""
    rows = []
    for opt in menu(state, cfg):
        liks = option_likelihoods(opt, state, cfg)
        row = {
            "key": opt.key,
            "kind": opt.kind,
            "target": opt.target,
            "tag": opt.tag,
            "eig_bits": beliefs.expected_information_gain(state.belief, liks),
        }
        if opt.kind == OFFER:
            d = DEALS_BY_ID[opt.target]
            row["p_accept"] = p_accept(state, d, cfg)
            row["own_utility"] = own_utility(state, d)
            row["myopic_value"] = myopic_value(state, d, cfg)
        rows.append(row)
    return rows
