"""The two LLM-as-agent arms for CD-NOUL-1 Stage 1: the model reads the conversation (with the
opponent's answers as free text) and picks one option from the same lettered menu the planner
arms optimise over.

* ``jev_agent``   — CD-VOI-1's ``jev_ask_plain`` (`jev_policy_live.JevLivePolicy`) with one change:
                    where the state text used to print the typed answer ("answered: yes") it now
                    prints the verbalized utterance. Instructions, option descriptions, lettering
                    and shuffling are unchanged.
* ``sonnet_agent`` — ``claude-sonnet-5-5`` given the identical instructions, state text and
                    lettered options, asked for the single letter of its choice. A reply with no
                    parseable letter is retried once, then falls back to the first option (as the
                    Jev policy does for a malformed Choice) and is flagged.

Neither agent sees EIG, P(accept), tags or the posterior: the plain treatment of Phase 1.
"""

from __future__ import annotations

import random
import re
import string
from typing import Dict, List, Optional

from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID
from llm_rl_scripts.car_dealer_live.jev_policy_live import JevLivePolicy
from llm_rl_scripts.car_dealer_live.menu_live import Row
from llm_rl_scripts.car_dealer_live.protocol_live import ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, PublicState, other

SONNET_AGENT_MAX_TOKENS = 600
SONNET_AGENT_RETRY_MAX_TOKENS = 2000


class TextJevPolicy(JevLivePolicy):
    """`JevLivePolicy` whose state text carries the opponent's verbalized answers.

    ``self.utterances`` maps an event index (position in ``state.events``) to the utterance
    served for the focal's ASK at that index; the runner sets it before each decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utterances: Dict[int, str] = {}

    def instructions(self) -> str:
        text = super().instructions()
        return text.replace("(it answers truthfully)", "(it answers in its own words, truthfully but not necessarily directly)")

    def state_text(self, state: PublicState, belief=None) -> str:
        me, them = self.focal, other(self.focal)
        lines = [f"Opportunities remaining in the shared budget: {state.remaining} of {self.cfg.horizon}."]
        if state.events:
            lines.append("So far:")
            for idx, e in enumerate(state.events):
                who = "You" if e.actor == me else f"The {them}"
                a = e.action
                if a.kind == PROPOSE:
                    d = DEALS_BY_ID[a.deal_id]
                    lines.append(f"  - {who} proposed {d.label()} [{d.id}].")
                elif a.kind == ASK:
                    q = BANK_BY_ID[a.question_id]
                    if e.actor == me:
                        said = self.utterances.get(idx, e.answer)
                        lines.append(f"  - You asked: '{q.text}' The {them} replied: \"{said}\"")
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


_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def parse_letter(text: str, letters: str) -> Optional[str]:
    """The first standalone capital letter in the reply that names an option."""
    for m in _LETTER_RE.finditer(text):
        if m.group(1) in letters:
            return m.group(1)
    return None


class SonnetAgent:
    """`claude-sonnet-5-5` choosing from the lettered menu. Text is produced by a `TextJevPolicy`
    with a `None` client so the instructions and option descriptions are byte-identical to the
    Jev arm's; only the chooser differs."""

    def __init__(self, client, cfg, focal: str, focal_ranking, *, shuffle_rng: Optional[random.Random] = None):
        self.client = client
        self.text = TextJevPolicy(None, cfg, focal, focal_ranking, shuffle_rng=None)
        self.shuffle_rng = shuffle_rng
        self.last_info: Optional[dict] = None

    @property
    def utterances(self) -> Dict[int, str]:
        return self.text.utterances

    @utterances.setter
    def utterances(self, value: Dict[int, str]) -> None:
        self.text.utterances = value

    def prompt(self, state: PublicState, rows: List[Row], letter_to_key: Dict[str, str], options: Dict[str, str]) -> str:
        body = [self.text.state_text(state), "", "Your options:"]
        for letter in letter_to_key:
            body.append(f"{letter}. {options[letter]}")
        body.append("")
        body.append("Reply with the single letter of the option you choose, and nothing else.")
        return "\n".join(body)

    def choose(self, state: PublicState, rows: List[Row]) -> str:
        order = list(range(len(rows)))
        if self.shuffle_rng is not None:
            self.shuffle_rng.shuffle(order)
        letters = string.ascii_uppercase[: len(rows)]
        letter_to_key = {letter: rows[idx].key for letter, idx in zip(letters, order)}
        options = {letter: self.text.describe(rows[idx]) for letter, idx in zip(letters, order)}
        user = self.prompt(state, rows, letter_to_key, options)
        messages = [{"role": "user", "content": user}]
        reply = self.client.complete(self.text.instructions(), messages, SONNET_AGENT_MAX_TOKENS)
        letter = parse_letter(reply.text, letters)
        retried = False
        if letter is None:
            reply = self.client.complete(self.text.instructions(), messages, SONNET_AGENT_RETRY_MAX_TOKENS)
            letter = parse_letter(reply.text, letters)
            retried = True
        well_formed = letter is not None
        chosen = letter if well_formed else letters[0]
        self.last_info = {
            "letter_to_key": letter_to_key, "chosen_letter": chosen, "chosen_position": letters.index(chosen),
            "well_formed": well_formed, "retried": retried, "reply": reply.text[:400], "stop_reason": reply.stop_reason,
            "model": reply.model, "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens},
            "shown": {"eig": False, "belief": False, "accept_annotation": None},
        }
        return letter_to_key[chosen]
