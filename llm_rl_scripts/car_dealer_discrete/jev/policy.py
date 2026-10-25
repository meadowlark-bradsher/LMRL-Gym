"""A Jev-driven policy for the discretized Car Dealer negotiation.

The policy is a drop-in for the :class:`~llm_rl_scripts.car_dealer_discrete.protocol`
``Policy`` type: it maps an :class:`Observation` to a legal :class:`Action`.

By construction the Choice options are *exactly* the actions the state machine
reports as legal, so the policy can never emit an illegal action — the same
legality-by-construction property Jev has on Chess and the mazes. The open
question is negotiation *quality*, not legality.

Each option is described in plain terms. Because the public utility formula and
the agent's own ranking are known to that agent, the description may also state
the deal's value *to this agent* (``reveal_own_utility``, default on): that is
information the agent could compute itself, not a leak of the opponent's private
type. Turn it off to ablate how much Jev relies on the precomputed number.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from LLM_RL.jev.client import ChoiceAnswer, JevClient
from llm_rl_scripts.car_dealer_discrete.game import Deal
from llm_rl_scripts.car_dealer_discrete.jev.aid import AcceptanceAid, ranking_key
from llm_rl_scripts.car_dealer_discrete.protocol import (
    ACCEPT,
    OFFER,
    REJECT,
    WALK,
    Action,
    Observation,
)

_DIRECTIONS = {
    "buyer": "lower price, longer warranty, and faster delivery",
    "seller": "higher price, shorter warranty, and slower delivery",
}


def _option_key(action: Action, obs: Observation) -> str:
    """A stable token Jev returns; carries the deal id where one applies."""
    if action.kind == OFFER:
        return f"OFFER {action.deal_id}"
    if action.kind == ACCEPT:
        return f"ACCEPT {obs.outstanding_offer}"
    return action.kind  # REJECT / WALK


class JevNegotiationPolicy:
    """Chooses one legal action per turn via a single Choice question."""

    def __init__(
        self,
        client: JevClient,
        *,
        reveal_own_utility: bool = True,
        salience: bool = False,
        aid: Optional[AcceptanceAid] = None,
        shuffle_rng: Optional[random.Random] = None,
    ) -> None:
        self.client = client
        self.reveal_own_utility = reveal_own_utility
        # Salience condition: restate the deadline and the standing/best-seen
        # offer in *value* terms relative to the no-deal payoff. Pure framing of
        # outcomes — no strategic advice — so it isolates "does Jev pivot once
        # the opportunity cost is visible?" from the policy-aid condition.
        self.salience = salience
        # Policy-aid condition: an external forward model annotates each OFFER
        # with P(opponent accepts). The aid never ranks, scores, or prunes
        # options; Jev integrates the number with its own utility itself.
        self.aid = aid
        self.shuffle_rng = shuffle_rng
        self.last_answer: Optional[ChoiceAnswer] = None
        # Evaluator-facing belief report (section 4.3); NOT shown to Jev.
        self.last_belief: Optional[Dict[str, float]] = None
        self.last_acceptance: Optional[Dict[str, float]] = None

    # -- prompt construction -------------------------------------------------

    def build_instructions(self, obs: Observation) -> str:
        ranking = " > ".join(obs.own_ranking)
        r = float(obs.reservation)
        text = (
            f"You are the {obs.role} negotiating the sale of one car. A deal sets "
            f"three attributes — price, warranty, delivery — each at one of three "
            f"levels. As the {obs.role} you prefer {_DIRECTIONS[obs.role]}. Your "
            f"PRIVATE priority ranking, most to least important, is: {ranking}. "
            f"Every deal is worth a score between 0 (worst for you) and 1 (best for "
            f"you). Both you and the other party have a reservation value of {r:.3f}: "
            f"a deal only holds if it is worth at least {r:.3f} to BOTH sides, and if "
            f"no deal is reached you each receive exactly {r:.3f}. You have a limited "
            f"number of proposals. Choose the single action giving the best outcome "
            f"for YOU — you do not know the other party's private ranking."
        )
        if self.aid is not None:
            text += (
                " Each proposal also shows an estimate of how likely the other party "
                "is to accept it, based on their behaviour so far."
            )
        return text

    def build_state(self, obs: Observation) -> str:
        lines: List[str] = []
        lines.append(
            f"Proposals used: {obs.proposals_used} of {obs.max_proposals}."
        )
        if obs.outstanding_offer is not None:
            d = Deal.from_id(obs.outstanding_offer)
            own = float(obs.utility_of(obs.outstanding_offer))
            note = f" (worth {own:.3f} to you)" if self.reveal_own_utility else ""
            lines.append(
                f"There is an offer on the table from the other party: "
                f"{d.label()} [{d.id}]{note}."
            )
        else:
            lines.append("There is no offer currently on the table.")
        if self.salience:
            lines.extend(self._salience_lines(obs))
        if obs.history:
            lines.append("Negotiation so far:")
            for turn in obs.history:
                act = turn.action
                if act.kind == OFFER:
                    lines.append(f"  - {turn.actor} proposed {act.deal_id}")
                elif act.kind == ACCEPT:
                    lines.append(f"  - {turn.actor} accepted")
                elif act.kind == REJECT:
                    lines.append(f"  - {turn.actor} rejected the offer")
                else:
                    lines.append(f"  - {turn.actor} walked away")
        else:
            lines.append("No actions have been taken yet; it is your move.")
        return "\n".join(lines)

    def _salience_lines(self, obs: Observation) -> List[str]:
        """Outcome values made explicit (salience condition). No advice."""
        r = float(obs.reservation)
        remaining = obs.max_proposals - obs.proposals_used
        other = "seller" if obs.role == "buyer" else "buyer"
        lines = [
            f"Proposals remaining for both sides combined: {remaining}. If they run "
            f"out with no accepted deal, the negotiation ends and you receive exactly "
            f"{r:.3f} — your no-deal payoff."
        ]
        # Best offer the other party has actually put on the table so far.
        their_offers = [
            t.action.deal_id
            for t in obs.history
            if t.actor == other and t.action.kind == OFFER
        ]
        if their_offers:
            best_id = max(their_offers, key=lambda d: obs.utility_of(d))
            best_u = float(obs.utility_of(best_id))
            rel = "above" if best_u >= r else "below"
            lines.append(
                f"The best offer the {other} has made to you so far is "
                f"{Deal.from_id(best_id).label()} [{best_id}], worth {best_u:.3f} to you "
                f"— {rel} your no-deal payoff of {r:.3f}."
            )
        if obs.outstanding_offer is not None:
            own = float(obs.utility_of(obs.outstanding_offer))
            lines.append(
                f"Accepting the offer on the table now gives you {own:.3f}; walking "
                f"away or running out of proposals gives you {r:.3f}."
            )
        return lines

    def _describe(self, action: Action, obs: Observation) -> str:
        if action.kind == OFFER:
            d = Deal.from_id(action.deal_id)
            acc = ""
            if self.last_acceptance is not None:
                p = self.last_acceptance[action.deal_id]
                acc = f"; estimated chance the other party accepts: {p:.2f}"
            if self.reveal_own_utility:
                own = float(obs.utility_of(action.deal_id))
                return f"Propose {d.label()} [{d.id}]; this deal is worth {own:.3f} to you{acc}."
            return f"Propose {d.label()} [{d.id}]{acc}."
        if action.kind == ACCEPT:
            d = Deal.from_id(obs.outstanding_offer)
            if self.reveal_own_utility:
                own = float(obs.utility_of(obs.outstanding_offer))
                return (
                    f"Accept the offer on the table ({d.label()}), worth {own:.3f} to "
                    f"you. This ends the negotiation with an agreed deal."
                )
            return (
                f"Accept the offer on the table ({d.label()}). This ends the "
                f"negotiation with an agreed deal."
            )
        if action.kind == REJECT:
            return (
                "Reject the offer on the table and keep negotiating; the other party "
                "moves next."
            )
        return (
            f"Walk away with no deal. You then receive only your reservation payoff "
            f"({float(obs.reservation):.3f})."
        )

    def build_options(
        self, obs: Observation
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Action]]:
        """Return (option_descriptions, key->Action) built from legal actions."""
        actions = list(obs.legal_actions)
        if self.shuffle_rng is not None:
            self.shuffle_rng.shuffle(actions)
        if self.aid is not None:
            q, pacc = self.aid.acceptance_probabilities(obs)
            self.last_belief = {ranking_key(th): p for th, p in q.items()}
            self.last_acceptance = pacc
        else:
            self.last_belief = None
            self.last_acceptance = None
        descriptions: Dict[str, Optional[str]] = {}
        back: Dict[str, Action] = {}
        for action in actions:
            key = _option_key(action, obs)
            descriptions[key] = self._describe(action, obs)
            back[key] = action
        return descriptions, back

    # -- the Policy interface ------------------------------------------------

    def __call__(self, obs: Observation) -> Action:
        options, back = self.build_options(obs)
        answer = self.client.choice(
            state=self.build_state(obs),
            instructions=self.build_instructions(obs),
            options=options,
            question_id="negotiation_action",
        )
        self.last_answer = answer
        # options were built from legal actions, so a well-formed answer maps
        # back to a legal action; the state machine still guards the rest.
        return back.get(answer.choice, Action(WALK))
