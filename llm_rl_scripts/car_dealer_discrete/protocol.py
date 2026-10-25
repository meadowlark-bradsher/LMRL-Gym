"""Legal action / turn state machine for the proposal-only protocol (spec section 3).

Design decisions encoded here (v0):

* Action space: ``OFFER(deal_id)``, ``ACCEPT``, ``REJECT``, ``WALK``.
* Six total *proposal opportunities* per episode (``max_proposals``). Only
  ``OFFER`` consumes this budget; ``ACCEPT`` / ``REJECT`` / ``WALK`` are
  decisions recorded separately, so counteroffers never silently extend it.
* A counteroffer (``OFFER`` while an offer is outstanding) replaces the
  outstanding offer.
* ``REJECT`` clears the outstanding offer and passes the turn.
* ``ACCEPT`` resolves the episode. Accepting a deal that is not individually
  rational for *both* sides is an invalid action: it is logged, yields no
  enforceable agreement, and the episode ends with the no-deal payoff.
* At the deadline (proposal budget exhausted) a still-outstanding offer may
  still be ``ACCEPT``-ed, but no new ``OFFER`` may be made; an unaccepted
  outstanding offer does not count as a deal.
* The no-deal payoff for each side equals its outside option ``r``.

The evaluator holds both hidden rankings but the observation handed to a
policy exposes only that policy's *own* ranking plus public information.
A policy is any callable ``observation -> action``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

from .game import ALL_DEALS, DEALS_BY_ID, Deal, Ranking, utility

# ---- actions ---------------------------------------------------------------

OFFER, ACCEPT, REJECT, WALK = "OFFER", "ACCEPT", "REJECT", "WALK"


@dataclass(frozen=True)
class Action:
    kind: str                      # OFFER / ACCEPT / REJECT / WALK
    deal_id: Optional[str] = None  # set iff kind == OFFER

    def __post_init__(self) -> None:
        if self.kind == OFFER:
            if self.deal_id not in DEALS_BY_ID:
                raise ValueError(f"OFFER references unknown deal id {self.deal_id!r}")
        elif self.kind in (ACCEPT, REJECT, WALK):
            if self.deal_id is not None:
                raise ValueError(f"{self.kind} takes no deal_id")
        else:
            raise ValueError(f"unknown action kind {self.kind!r}")


def offer(deal: Deal) -> Action:
    return Action(OFFER, deal.id)


# ---- observation (what a policy is allowed to see) -------------------------

@dataclass(frozen=True)
class Observation:
    """Everything a policy may condition on. Carries NO opponent private info."""

    role: str                                  # "buyer" or "seller"
    own_ranking: Ranking                       # this policy's own private type
    reservation: Fraction                      # public outside option r
    max_proposals: int                         # public turn budget
    proposals_used: int                        # proposals consumed so far
    outstanding_offer: Optional[str]           # deal id on the table, or None
    history: Tuple["TurnRecord", ...]          # public sequence of (actor, action)
    legal_actions: Tuple[Action, ...]          # exactly the actions legal right now

    def utility_of(self, deal_id: str) -> Fraction:
        """Convenience: this policy's own utility for a deal (own ranking only)."""
        return utility(self.role, self.own_ranking, DEALS_BY_ID[deal_id])


# ---- turn / episode records ------------------------------------------------

@dataclass(frozen=True)
class TurnRecord:
    actor: str
    action: Action


@dataclass
class EpisodeResult:
    buyer_rank: Ranking
    seller_rank: Ranking
    reservation: Fraction
    starting_side: str
    agreement: bool
    accepted_deal_id: Optional[str]
    buyer_utility: Fraction            # realized payoff (deal utility, or r if no deal)
    seller_utility: Fraction
    individually_rational: Optional[bool]
    outcome_type: str                  # agreement / walk / timeout / invalid_acceptance
    proposals_used: int
    decisions: int
    turns: List[TurnRecord] = field(default_factory=list)


Policy = Callable[[Observation], Action]


class NegotiationGame:
    """Deterministic referee for one episode of the discretized game."""

    def __init__(
        self,
        buyer_rank: Ranking,
        seller_rank: Ranking,
        reservation: Fraction,
        buyer_policy: Policy,
        seller_policy: Policy,
        *,
        max_proposals: int = 6,
        starting_side: str = "buyer",
    ) -> None:
        if starting_side not in ("buyer", "seller"):
            raise ValueError("starting_side must be 'buyer' or 'seller'")
        self.buyer_rank = buyer_rank
        self.seller_rank = seller_rank
        self.r = reservation
        self.policies: Dict[str, Policy] = {"buyer": buyer_policy, "seller": seller_policy}
        self.max_proposals = max_proposals
        self.starting_side = starting_side

    def _ranking(self, role: str) -> Ranking:
        return self.buyer_rank if role == "buyer" else self.seller_rank

    def _is_ir(self, deal: Deal) -> bool:
        return (
            utility("buyer", self.buyer_rank, deal) >= self.r
            and utility("seller", self.seller_rank, deal) >= self.r
        )

    def _legal_actions(self, outstanding: Optional[Deal], proposals_used: int) -> Tuple[Action, ...]:
        legal: List[Action] = []
        can_offer = proposals_used < self.max_proposals
        if outstanding is not None:
            legal.append(Action(ACCEPT))
            legal.append(Action(REJECT))
        if can_offer:
            legal.extend(offer(d) for d in ALL_DEALS)
        legal.append(Action(WALK))
        return tuple(legal)

    def play(self) -> EpisodeResult:
        outstanding: Optional[Deal] = None
        proposals_used = 0
        decisions = 0
        turns: List[TurnRecord] = []
        actor = self.starting_side
        # Hard safety cap: each OFFER/REJECT advances state; budget bounds OFFERs,
        # and a REJECT with no re-offer forces an OFFER-or-WALK next turn.
        max_turns = 4 * self.max_proposals + 8

        outcome_type = "timeout"
        agreement = False
        accepted_deal: Optional[Deal] = None

        for _ in range(max_turns):
            legal = self._legal_actions(outstanding, proposals_used)
            obs = Observation(
                role=actor,
                own_ranking=self._ranking(actor),
                reservation=self.r,
                max_proposals=self.max_proposals,
                proposals_used=proposals_used,
                outstanding_offer=outstanding.id if outstanding else None,
                history=tuple(turns),
                legal_actions=legal,
            )
            action = self.policies[actor](obs)
            if action not in legal:
                # A policy must return a legal action. Treat anything else as a
                # walk (recorded), rather than silently repairing it.
                action = Action(WALK)
            turns.append(TurnRecord(actor, action))
            decisions += 1

            if action.kind == WALK:
                outcome_type = "walk"
                break

            if action.kind == ACCEPT:
                # ACCEPT is only legal with an outstanding offer (enforced above).
                assert outstanding is not None
                if self._is_ir(outstanding):
                    agreement = True
                    accepted_deal = outstanding
                    outcome_type = "agreement"
                else:
                    outcome_type = "invalid_acceptance"
                break

            if action.kind == REJECT:
                outstanding = None
                actor = _other(actor)
                continue

            if action.kind == OFFER:
                outstanding = DEALS_BY_ID[action.deal_id]
                proposals_used += 1
                actor = _other(actor)
                continue

        return self._finalize(
            agreement, accepted_deal, outcome_type, proposals_used, decisions, turns
        )

    def _finalize(
        self,
        agreement: bool,
        accepted_deal: Optional[Deal],
        outcome_type: str,
        proposals_used: int,
        decisions: int,
        turns: List[TurnRecord],
    ) -> EpisodeResult:
        if agreement and accepted_deal is not None:
            buyer_u = utility("buyer", self.buyer_rank, accepted_deal)
            seller_u = utility("seller", self.seller_rank, accepted_deal)
            ir: Optional[bool] = True
            accepted_id: Optional[str] = accepted_deal.id
        else:
            # No enforceable deal: each side realizes its outside option.
            buyer_u = self.r
            seller_u = self.r
            ir = False if outcome_type == "invalid_acceptance" else None
            accepted_id = None
        return EpisodeResult(
            buyer_rank=self.buyer_rank,
            seller_rank=self.seller_rank,
            reservation=self.r,
            starting_side=self.starting_side,
            agreement=agreement,
            accepted_deal_id=accepted_id,
            buyer_utility=buyer_u,
            seller_utility=seller_u,
            individually_rational=ir,
            outcome_type=outcome_type,
            proposals_used=proposals_used,
            decisions=decisions,
            turns=turns,
        )


def _other(role: str) -> str:
    return "seller" if role == "buyer" else "buyer"


# ---- a minimal policy, only to exercise the engine in tests ----------------

def random_policy(rng: random.Random) -> Policy:
    """Uniform choice over the legal actions. For self-test / smoke only."""

    def _policy(obs: Observation) -> Action:
        return rng.choice(obs.legal_actions)

    return _policy
