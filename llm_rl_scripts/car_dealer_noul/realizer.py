"""CD-NOUL-2: the realizer. Puts the planner's chosen move into words for the LLM opponent from the **action
and the public history only** — never the belief, never a probability, never another deal. The model is
told the move's letter among the lettered legal moves and must echo it:

    DECISION: <letter>
    SAY: <one to three sentences>

so the same decision-token parser the opponent uses checks the round trip. A realization fails the check
when the letter differs, the SAY is empty, it names another deal, or it mentions probability-like
language; a failed realization falls back to the templated text and is counted. ``DESIGN_NOUL2.md`` §2.4."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS
from llm_rl_scripts.car_dealer_live.protocol_live import ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, Action, LiveConfig, PublicState, other

from .opponent_llm import deal_words, lettered, moves_text, parse_reply, state_text

MAX_TOKENS = 400
RETRY_MAX_TOKENS = 1200

SYSTEM = """You put one negotiating move into words for the {role} in a negotiation over a used car with a {other}. The deal will set the price, the warranty and the delivery time. You do not choose the move; it has been chosen and you are told which one it is. Your job is to say it naturally, in one to three sentences, as the {role} would to the {other}.

Say only what the move says. If the move proposes a deal, propose exactly that deal and no other. If it asks a question, ask that question (you may rephrase it, but it must ask the same thing). If it accepts, rejects, holds or walks away, say so plainly. Never mention probabilities, likelihoods, odds, percentages or what you believe about the {other}'s priorities; never offer or hint at any other deal or concession.

Reply with exactly one line
DECISION: <letter>
repeating the letter of the move you were given, followed by one line
SAY: <your words>"""

FORBIDDEN = ("probab", "likel", "odds", "percent", "%", "chance", "i believe you", "i think you care", "your priorit")


def realizer_hash() -> str:
    return hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]


def template_text(action: Action) -> str:
    """Stage 1's templated phrasing: the question's text, or one plain sentence."""
    if action.kind == ASK:
        return BANK_BY_ID[action.question_id].text
    if action.kind == PROPOSE:
        return f"I propose {deal_words(action.deal_id)}."
    return {ACCEPT: "I accept.", REJECT: "I can't accept that.", HOLD: "My offer stands.", WALK: "I'm walking away."}[action.kind]


def other_deal_mentioned(say: str, action: Action) -> Optional[str]:
    low = say.lower()
    for d in ALL_DEALS:
        if action.kind == PROPOSE and d.id == action.deal_id:
            continue
        if d.id.lower() in low or d.label().lower() in low:
            return d.id
    return None


def check(reply_text: str, expected_letter: str, letters: str, action: Action) -> Tuple[bool, List[str], str]:
    letter, say = parse_reply(reply_text, letters)
    reasons = []
    if letter != expected_letter:
        reasons.append("letter" if letter is not None else "no_decision")
    if not say:
        reasons.append("empty_say")
    low = say.lower()
    hits = [w for w in FORBIDDEN if w in low]
    if hits:
        reasons.append("forbidden:" + ",".join(hits))
    od = other_deal_mentioned(say, action)
    if od:
        reasons.append("other_deal:" + od)
    return (not reasons), reasons, say


@dataclass
class Realization:
    text: str
    ok: bool
    fallback: bool
    retried: bool
    reasons: List[str] = field(default_factory=list)
    reply: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""


class Realizer:
    def __init__(self, client, role: str, cfg: LiveConfig, *, rng: Optional[random.Random] = None):
        self.client, self.role, self.cfg, self.rng = client, role, cfg, rng
        self.system = SYSTEM.format(role=role, other=other(role))
        self.calls: List[dict] = []

    def realize(self, state: PublicState, action: Action, legal: List[Action], said: Dict[int, str],
                heard: Dict[int, str]) -> Realization:
        """``said`` = the focal's own earlier realized words by event index; ``heard`` = the opponent's."""
        letters = lettered(legal, self.rng)
        expected = next(L for L, a in letters.items() if a == action)
        user = (state_text(state, self.role, self.cfg, said, heard) + "\n\nThe legal moves:\n" + moves_text(letters) +
                f"\n\nThe move to put into words is {expected}. Say it.")
        reply = self.client.complete(self.system, [{"role": "user", "content": user}], MAX_TOKENS)
        ok, reasons, say = check(reply.text, expected, "".join(letters), action)
        retried = False
        if not ok:
            reply = self.client.complete(self.system, [{"role": "user", "content": user}], RETRY_MAX_TOKENS)
            ok, reasons, say = check(reply.text, expected, "".join(letters), action)
            retried = True
        self.calls.append({"user": user, "reply": reply.text, "ok": ok, "reasons": reasons})
        text = say if ok else template_text(action)
        return Realization(text, ok, not ok, retried, reasons, reply.text[:600],
                           {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}, reply.model)


class TemplateRealizer:
    """The ``template`` control arm: no model, Stage 1's phrasing."""

    def realize(self, state, action, legal, said, heard) -> Realization:
        return Realization(template_text(action), True, False, False)


class EchoClient:
    """Offline realizer client: echoes the requested letter with a templated SAY (``mode='good'``), a wrong
    letter (``'bad_letter'``), a leak (``'leak'``) or another deal (``'other_deal'``)."""

    def __init__(self, mode: str = "good"):
        self.mode, self.calls = mode, 0

    def complete(self, system, messages, max_tokens):
        from .opponent_llm import Completion
        self.calls += 1
        user = messages[-1]["content"]
        m = re.search(r"The move to put into words is ([A-Z])\.", user)
        letter = m.group(1)
        desc = dict(re.findall(r"^\s+([A-Z])\. (.*)$", user, re.M))[letter]
        say = {"good": f"Here is my position: {desc.lower()}.",
               "bad_letter": None, "leak": "I'd say there's a 70% chance you care most about price.",
               "other_deal": f"{desc.lower()}, or failing that P0-W0-D0."}[self.mode]
        if self.mode == "bad_letter":
            wrong = "Z"
            return Completion(f"DECISION: {wrong}\nSAY: {desc.lower()}.")
        return Completion(f"DECISION: {letter}\nSAY: {say}")
