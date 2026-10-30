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

CD-SEL-1 (design Phase 2, the selector arms) adds two kinds of shown information, each
a separately named treatment; the event history itself has been in the state text
since Phase 1:

  * ``show_belief=True``: the state text ends with what the opponent's answers and
    moves so far imply about its priority ranking — the matched posterior over the
    six rankings, in words, with probabilities; rankings ruled out are counted;
  * ``accept_annotation``: every PROPOSE option carries the probability that the
    opponent accepts it — ``"posterior"`` (B_passive: the matched posterior),
    ``"prior"`` (A_prior: the same number under the uniform prior, never updated),
    ``"placebo"`` (P_placebo: the constant ``PLACEBO_P_ACCEPT`` on every offer).
    The runner computes the numbers (``run_phase1.shown_p_accept``); the policy
    only formats what it is handed.
"""

from __future__ import annotations

import math
import random
import string
from typing import Dict, List, Optional

from LLM_RL.jev.client import ChoiceAnswer, JevClient
from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID

from .menu_live import Row
from .protocol_live import ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, LiveConfig, PublicState, other

ACCEPT_ANNOTATIONS = (None, "posterior", "prior", "placebo")
PLACEBO_P_ACCEPT = 0.5          # P_placebo: the same number on every offer, carrying no information
_SUPPORT_EPS = 1e-9

_DIRECTIONS = {
    "buyer": "lower price, longer warranty, and faster delivery",
    "seller": "higher price, shorter warranty, and slower delivery",
}


def entropy_bits(probs: Dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in probs.values() if p > 0))


class JevLivePolicy:
    def __init__(self, client: JevClient, cfg: LiveConfig, focal: str, focal_ranking, *,
                 show_eig: bool = False, show_belief: bool = False, accept_annotation: Optional[str] = None,
                 shuffle_rng: Optional[random.Random] = None) -> None:
        if accept_annotation not in ACCEPT_ANNOTATIONS:
            raise ValueError(f"accept_annotation must be one of {ACCEPT_ANNOTATIONS}, got {accept_annotation!r}")
        self.client = client
        self.cfg = cfg
        self.focal = focal
        self.ranking = focal_ranking
        self.show_eig = show_eig
        self.show_belief = show_belief
        self.accept_annotation = accept_annotation
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
        if self.show_belief:
            text += " You are also told what the other party's answers and moves so far imply about its priority ranking."
        if self.accept_annotation is not None:
            text += " Each proposal also shows the estimated probability that the other party accepts it."
        return text

    def belief_text(self, belief) -> str:
        """The matched posterior over the opponent's rankings, in words (``show_belief``)."""
        them = other(self.focal)
        n = len(ALL_RANKINGS)
        probs = [(p, i, r) for i, (r, p) in enumerate(zip(ALL_RANKINGS, belief)) if p > _SUPPORT_EPS]
        if len(probs) == n and max(p for p, _, _ in probs) - min(p for p, _, _ in probs) < _SUPPORT_EPS:
            return (f"Nothing the {them} has said or done so far distinguishes its {n} possible priority rankings; "
                    f"each has probability {1.0 / n:.2f}.")
        probs.sort(key=lambda t: (-t[0], t[1]))
        parts = [f"{' > '.join(r)} ({p:.2f})" for p, _, r in probs]
        ruled = n - len(probs)
        text = (f"What the {them}'s answers and moves so far imply about its priority ranking (most to least important), "
                f"with probabilities: " + "; ".join(parts) + ".")
        if ruled:
            text += f" The other {ruled} ranking{'s are' if ruled > 1 else ' is'} ruled out."
        return text

    def state_text(self, state: PublicState, belief=None) -> str:
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
        if self.show_belief:
            if belief is None:
                raise ValueError("show_belief policy needs the belief")
            lines.append(self.belief_text(belief))
        return "\n".join(lines)

    def describe(self, row: Row, p_accept_shown: Optional[float] = None) -> str:
        eig = f" Expected information about their priorities: {row.eig_bits:.2f} bits." if self.show_eig else ""
        a = row.action
        if a.kind == ASK:
            return f"Ask the other party: '{BANK_BY_ID[a.question_id].text}' This uses one opportunity.{eig}"
        if a.kind == PROPOSE:
            d = DEALS_BY_ID[a.deal_id]
            acc = ""
            if self.accept_annotation is not None:
                if p_accept_shown is None:
                    raise ValueError(f"accept_annotation={self.accept_annotation!r} needs a shown P(accept) for {row.key}")
                acc = f" If you propose it now, the {other(self.focal)} accepts with probability {p_accept_shown:.2f}."
            return (f"Propose {d.label()} [{d.id}]; this deal is worth {float(row.own_utility):.3f} to you. "
                    f"This uses one opportunity. If accepted, the negotiation ends with this deal.{acc}{eig}")
        if a.kind == ACCEPT:
            return f"Accept the standing proposal; it is worth {float(row.own_utility):.3f} to you. The negotiation ends with it."
        if a.kind == REJECT:
            return f"Reject the standing proposal without proposing anything.{eig}"
        if a.kind == HOLD:
            return f"Hold your standing proposal unchanged and let the other party respond to it.{eig}"
        return "Walk away: the negotiation ends with no deal."

    def choose(self, state: PublicState, rows: List[Row], belief=None,
               p_accept_shown: Optional[Dict[str, float]] = None) -> str:
        """``belief``: the matched posterior (used only when ``show_belief``); ``p_accept_shown``:
        {PROPOSE row key: probability} to print (used only with an ``accept_annotation``)."""
        order = list(range(len(rows)))
        if self.shuffle_rng is not None:
            self.shuffle_rng.shuffle(order)
        letters = string.ascii_uppercase[: len(rows)]
        letter_to_key: Dict[str, str] = {}
        options: Dict[str, Optional[str]] = {}
        shown = p_accept_shown or {}
        for letter, idx in zip(letters, order):
            letter_to_key[letter] = rows[idx].key
            options[letter] = self.describe(rows[idx], shown.get(rows[idx].key))
        answer = self.client.choice(state=self.state_text(state, belief), instructions=self.instructions(),
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
            "shown": {"eig": self.show_eig, "belief": self.show_belief, "accept_annotation": self.accept_annotation,
                      "p_accept_shown": dict(shown) if self.accept_annotation is not None else None},
        }
        return letter_to_key[chosen_letter]
