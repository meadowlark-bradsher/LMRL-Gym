"""The Jev arm for the live game: one Choice over the lettered menu at every focal
decision. Carried over from Condition B unchanged in treatment structure:

  * options are lettered A, B, C ... in a randomised order (positional-bias control);
    the letter -> option mapping is logged;
  * ``show_eig=False`` (plain) shows no information number; ``show_eig=True``
    (eig_shown) annotates every option with its exact expected information gain in
    bits (offers included, since an offer's response is informative too);
  * neither condition shows P(accept), the SAFE/AMBITIOUS tags, or the posterior.
    The focal's own utility of a deal is shown (public to the focal).

The belief that builds the menu is maintained outside the model; the model only
chooses among the options it is shown.
"""

from __future__ import annotations

import math
import random
import string
from typing import Dict, List, Optional

from LLM_RL.jev.client import ChoiceAnswer, JevClient
from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID

from .menu_live import Row
from .protocol_live import ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, LiveConfig, PublicState, other

_DIRECTIONS = {
    "buyer": "lower price, longer warranty, and faster delivery",
    "seller": "higher price, shorter warranty, and slower delivery",
}


def entropy_bits(probs: Dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in probs.values() if p > 0))


class JevLivePolicy:
    def __init__(self, client: JevClient, cfg: LiveConfig, focal: str, focal_ranking, *,
                 show_eig: bool = False, shuffle_rng: Optional[random.Random] = None) -> None:
        self.client = client
        self.cfg = cfg
        self.focal = focal
        self.ranking = focal_ranking
        self.show_eig = show_eig
        self.shuffle_rng = shuffle_rng
        self.last_answer: Optional[ChoiceAnswer] = None
        self.last_info: Optional[dict] = None

    def instructions(self) -> str:
        r = float(self.cfg.reservation)
        ranking = " > ".join(self.ranking)
        c = float(self.cfg.ask_cost)
        text = (
            f"You are the {self.focal} negotiating the sale of one car with the {other(self.focal)}. A deal sets "
            f"three attributes (price, warranty, delivery), each at one of three levels. As the {self.focal} you "
            f"prefer {_DIRECTIONS[self.focal]}. Your PRIVATE priority ranking, most to least important, is: {ranking}. "
            f"Every deal is worth a score between 0 (worst for you) and 1 (best for you). The other party has its own "
            f"private ranking, which you do not know, and its own walk-away value. Both of you take turns. On your turn "
            f"you may propose a deal, ask the other party one yes/no question about its priorities (it answers "
            f"truthfully), accept or reject its standing proposal, hold your own standing proposal, or walk away. "
            f"Proposals and questions share a budget of {self.cfg.horizon} opportunities between the two of you; "
            f"accepting, rejecting, holding and walking are free. When the budget runs out, whoever faces a standing "
            f"proposal makes a final accept-or-walk decision; with no proposal standing the negotiation ends with no "
            f"deal. An accepted proposal ends the negotiation with that deal. No deal is worth exactly {r:.3f} to you."
        )
        if c > 0:
            text += f" Each question you ask costs you {c:.3f} of score."
        text += " Choose the single option giving the best outcome for YOU."
        if self.show_eig:
            text += " Each option also shows how much the other party's response is expected to reveal about its priorities, in bits."
        return text

    def state_text(self, state: PublicState) -> str:
        me, them = self.focal, other(self.focal)
        lines = [f"Opportunities remaining in the shared budget: {state.remaining} of {self.cfg.horizon}."]
        if state.events:
            lines.append("So far:")
            for e in state.events:
                who = "You" if e.actor == me else f"The {them}"
                a = e.action
                if a.kind == PROPOSE:
                    d = DEALS_BY_ID[a.deal_id]
                    lines.append(f"  - {who} proposed {d.label()} [{d.id}].")
                elif a.kind == ASK:
                    q = BANK_BY_ID[a.question_id]
                    if e.actor == me:
                        lines.append(f"  - You asked: '{q.text}' The {them} answered: {e.answer}.")
                    else:
                        lines.append(f"  - The {them} asked you: '{q.text}' You answered: {e.answer}.")
                elif a.kind == ACCEPT:
                    lines.append(f"  - {who} accepted.")
                elif a.kind == REJECT:
                    lines.append(f"  - {who} rejected the standing proposal.")
                elif a.kind == HOLD:
                    lines.append(f"  - {who} held the standing proposal.")
                elif a.kind == WALK:
                    lines.append(f"  - {who} walked away.")
        else:
            lines.append("Nothing has happened yet; you move first.")
        st = state.standing
        if st is not None:
            d = DEALS_BY_ID[st.deal_id]
            owner = "your" if st.proposer == me else f"the {them}'s"
            lines.append(f"Standing proposal: {d.label()} [{d.id}] ({owner}).")
        else:
            lines.append("No proposal is standing.")
        if state.at_deadline:
            lines.append("The budget is exhausted: this is the final decision.")
        return "\n".join(lines)

    def describe(self, row: Row) -> str:
        eig = f" Expected information about their priorities: {row.eig_bits:.2f} bits." if self.show_eig else ""
        a = row.action
        if a.kind == ASK:
            return f"Ask the other party: '{BANK_BY_ID[a.question_id].text}' This uses one opportunity.{eig}"
        if a.kind == PROPOSE:
            d = DEALS_BY_ID[a.deal_id]
            return (f"Propose {d.label()} [{d.id}]; this deal is worth {float(row.own_utility):.3f} to you. "
                    f"This uses one opportunity. If accepted, the negotiation ends with this deal.{eig}")
        if a.kind == ACCEPT:
            return f"Accept the standing proposal; it is worth {float(row.own_utility):.3f} to you. The negotiation ends with it."
        if a.kind == REJECT:
            return f"Reject the standing proposal without proposing anything.{eig}"
        if a.kind == HOLD:
            return f"Hold your standing proposal unchanged and let the other party respond to it.{eig}"
        return "Walk away: the negotiation ends with no deal."

    def choose(self, state: PublicState, rows: List[Row]) -> str:
        order = list(range(len(rows)))
        if self.shuffle_rng is not None:
            self.shuffle_rng.shuffle(order)
        letters = string.ascii_uppercase[: len(rows)]
        letter_to_key: Dict[str, str] = {}
        options: Dict[str, Optional[str]] = {}
        for letter, idx in zip(letters, order):
            letter_to_key[letter] = rows[idx].key
            options[letter] = self.describe(rows[idx])
        answer = self.client.choice(state=self.state_text(state), instructions=self.instructions(),
                                    options=options, question_id="negotiation_action")
        self.last_answer = answer
        chosen_letter = answer.choice if answer.choice in letter_to_key else letters[0]
        probs_by_key = {letter_to_key[l]: float(p) for l, p in answer.probabilities.items() if l in letter_to_key}
        self.last_info = {
            "letter_to_key": letter_to_key,
            "chosen_letter": chosen_letter,
            "chosen_position": letters.index(chosen_letter),
            "probabilities": probs_by_key,
            "choice_entropy_bits": entropy_bits(probs_by_key) if probs_by_key else None,
            "confidence": answer.confidence,
            "well_formed": answer.choice in letter_to_key,
        }
        return letter_to_key[chosen_letter]
