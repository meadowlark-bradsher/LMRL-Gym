"""The Jev arm for Condition B: one Choice over the lettered menu.

Options are presented as letters (A, B, C, D ...) in a randomised order
(design doc section 2: randomise presentation to test positional bias); the
letter -> option mapping is logged. Jev returns a probability for every option,
so the action-selection entropy needed for H3 is one call, not repeated samples
(design doc section 6).

Two Jev conditions share this code path and differ in one annotation:
  * ``show_eig=False``  (plain)      options carry no information number
  * ``show_eig=True``   (eig_shown)  every option is annotated with its exact
                                     expected information gain in bits
Neither condition shows P(accept), the SAFE/AMBITIOUS tags, or the posterior:
those are analyst quantities. Own utility of an offer is shown (public to the
learner, as in the discretized Car Dealer policy).
"""

from __future__ import annotations

import math
import random
import string
from typing import Dict, List, Optional, Tuple

from LLM_RL.jev.client import ChoiceAnswer, JevClient
from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID

from .env import ASK, OFFER, Config, State
from .questions import BANK_BY_ID

_DIRECTIONS = {
    "buyer": "lower price, longer warranty, and faster delivery",
    "seller": "higher price, shorter warranty, and slower delivery",
}


def entropy_bits(probs: Dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in probs.values() if p > 0))


class JevConditionBPolicy:
    def __init__(
        self,
        client: JevClient,
        cfg: Config,
        *,
        show_eig: bool = False,
        shuffle_rng: Optional[random.Random] = None,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.show_eig = show_eig
        self.shuffle_rng = shuffle_rng
        self.last_answer: Optional[ChoiceAnswer] = None
        self.last_info: Optional[dict] = None

    def instructions(self, state: State) -> str:
        r = float(self.cfg.reservation)
        ranking = " > ".join(state.own_ranking)
        text = (
            f"You are the {state.role} negotiating the sale of one car. A deal sets three "
            f"attributes (price, warranty, delivery), each at one of three levels. As the "
            f"{state.role} you prefer {_DIRECTIONS[state.role]}. Your PRIVATE priority ranking, "
            f"most to least important, is: {ranking}. Every deal is worth a score between 0 "
            f"(worst for you) and 1 (best for you). The other party has its own private "
            f"ranking, which you do not know; it accepts a proposal only if the deal is worth "
            f"at least {r:.3f} to it, and answers questions about its priorities truthfully. "
            f"You have {self.cfg.rounds} turns in which you may either ask the other party one "
            f"question or propose one deal, followed by one final turn in which you must "
            f"propose. An accepted proposal ends the negotiation with that deal. If no proposal "
            f"is accepted you receive exactly {r:.3f}. Choose the single option giving the best "
            f"outcome for YOU."
        )
        if self.show_eig:
            text += (
                " Each option also shows how much it is expected to reveal about the other "
                "party's priorities, in bits."
            )
        return text

    def state_text(self, state: State) -> str:
        lines = [f"Turn {state.t} of {self.cfg.final_t}."]
        if state.t == self.cfg.final_t:
            lines.append("This is the final turn: you must propose a deal.")
        if state.history:
            lines.append("So far:")
            for turn in state.history:
                o = turn.option
                if o.kind == ASK:
                    q = BANK_BY_ID[o.target]
                    lines.append(f"  - You asked: '{q.text}' The other party answered: {turn.outcome}.")
                else:
                    d = DEALS_BY_ID[o.target]
                    lines.append(f"  - You proposed {d.label()} [{d.id}]; the other party {turn.outcome}ed it.")
        else:
            lines.append("Nothing has happened yet; it is your move.")
        return "\n".join(lines)

    def describe(self, row: dict, state: State) -> str:
        eig = f" Expected information about their priorities: {row['eig_bits']:.2f} bits." if self.show_eig else ""
        if row["kind"] == ASK:
            q = BANK_BY_ID[row["target"]]
            return f"Ask the other party: '{q.text}' This uses one turn.{eig}"
        d = DEALS_BY_ID[row["target"]]
        own = float(row["own_utility"])
        return (
            f"Propose {d.label()} [{d.id}]; this deal is worth {own:.3f} to you. If accepted, "
            f"the negotiation ends with this deal.{eig}"
        )

    def choose(self, state: State, rows: List[dict]) -> str:
        """Return the chosen option key; details in ``self.last_info``."""
        order = list(range(len(rows)))
        if self.shuffle_rng is not None:
            self.shuffle_rng.shuffle(order)
        letters = string.ascii_uppercase[: len(rows)]
        letter_to_key: Dict[str, str] = {}
        options: Dict[str, Optional[str]] = {}
        for letter, idx in zip(letters, order):
            letter_to_key[letter] = rows[idx]["key"]
            options[letter] = self.describe(rows[idx], state)
        answer = self.client.choice(
            state=self.state_text(state),
            instructions=self.instructions(state),
            options=options,
            question_id="negotiation_action",
        )
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
