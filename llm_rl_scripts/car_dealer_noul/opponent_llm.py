"""CD-NOUL-2: the LLM opponent. ``claude-sonnet-5-5`` plays a planted ranking under one of three frozen
briefs, chooses its move from the lettered legal set with a decision token and speaks in its own words.

    DECISION: <letter>
    SAY: <one to three sentences>

Moves are typed (the engine, the menu and the evaluator are unchanged); only the words are free. When the
opponent is asked a question it replies with ``SAY:`` only. A reply without a usable token is retried once
with a larger budget, then recorded malformed and the move defaults to the fallback kernel's sample (the
registered T1), counted. Every call's text is kept for the audit.

Offline this module runs against ``ScriptedClient`` (fixed replies) or ``KernelClient`` (a client that
answers the way the T1 kernel would, always well-formed) — no network. ``DESIGN_NOUL2.md`` §2.1."""

from __future__ import annotations

import hashlib
import random
import re
import string
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ATTRIBUTES, DEALS_BY_ID, LEVELS, Deal, utility
from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
from llm_rl_scripts.car_dealer_live.protocol_live import (ACCEPT, ASK, HOLD, PROPOSE, REJECT, WALK, Action, LiveConfig,
                                                          PublicState, legal_actions, other, truthful_answer)

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 700
RETRY_MAX_TOKENS = 2000
BRIEFS = ("candid", "guarded", "adversarial")

# ---- briefs (frozen text; the registration pins ``brief_hash``) ------------------------------------

_COMMON = """You are the {role} in a negotiation over one used car with a {other}. The deal will set three things: the price, the length of the warranty and the delivery time. Each has three levels.

Your private priorities, from most to least important, are: {a} > {b} > {c}. Keep them private unless you choose to reveal them.

How much a deal is worth to you: each attribute's level has a score for you (listed below); your satisfaction with a deal is (9 x score of your top attribute's level + 3 x score of your second + 1 x score of your third) / 13, between 0 and 1. A deal is worth taking only if your satisfaction is at least {reservation}; below that you are better off with no deal.
{scores}
Rules: the two of you share a budget of {horizon} opportunities; every proposal and every question uses one. When the budget is exhausted, whoever faces a standing proposal must accept it or walk away; with no proposal standing, nobody gets a deal. Either side may ask the other yes/no questions about its priorities. You negotiate for yourself and want the most satisfying deal you can get, but no deal below your threshold.

{answers}

Format. At your turn, think as much as you like, then end with exactly one line
DECISION: <label>
naming one of the listed moves by its label (a letter, or two letters), followed by one line
SAY: <what you say to the {other}, one to three sentences>
When you are only answering a question, reply with the SAY: line alone."""

_ANSWERS = {
    "candid": ("Answering questions: answer truthfully. You may answer in your own words and need not be blunt, "
               "but everything you say about your priorities must be true."),
    "guarded": ("Answering questions: you are under no obligation to answer directly. You may deflect, hedge, decline, "
                "answer a different question than the one asked, or let something show on purpose. You must never state "
                "anything false about your priorities."),
    "adversarial": ("Answering questions: you may mislead the {other} about your priorities whenever you judge it serves "
                    "you, including by stating things that are not true. Nothing obliges you to be truthful about them."),
}


def level_scores(role: str) -> Dict[str, Dict[int, Fraction]]:
    """Per-attribute, per-level score for ``role`` as the exact utility implies (min level scores 0)."""
    out: Dict[str, Dict[int, Fraction]] = {}
    others = {a: [x for x in ATTRIBUTES if x != a] for a in ATTRIBUTES}
    for a in ATTRIBUTES:
        b, c = others[a]
        ranking = (a, b, c)
        us = {}
        for lv in LEVELS:
            kwargs = {a: lv, b: 0, c: 0}
            us[lv] = utility(role, ranking, Deal(**kwargs))
        lo = min(us.values())
        out[a] = {lv: (us[lv] - lo) * 13 / 9 for lv in LEVELS}
    return out


def scores_text(role: str) -> str:
    from llm_rl_scripts.car_dealer_discrete.game import LEVEL_LABELS
    sc = level_scores(role)
    lines = []
    for a in ATTRIBUTES:
        parts = ", ".join(f"{LEVEL_LABELS[a][lv]} = {sc[a][lv]}" for lv in LEVELS)
        lines.append(f"  {a}: {parts}")
    return "\n".join(lines)


def brief(name: str, role: str, ranking, cfg: LiveConfig) -> str:
    if name not in BRIEFS:
        raise ValueError(f"unknown brief {name!r}")
    a, b, c = ranking
    return _COMMON.format(role=role, other=other(role), a=a, b=b, c=c, reservation=str(cfg.reservation),
                          scores=scores_text(role), horizon=cfg.horizon,
                          answers=_ANSWERS[name].format(other=other(role)))


def brief_hash() -> str:
    blob = _COMMON + "|" + "|".join(_ANSWERS[k] for k in BRIEFS)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---- rendering the public state for the opponent ------------------------------------------------

def deal_words(deal_id: str) -> str:
    return DEALS_BY_ID[deal_id].label()


def state_text(state: PublicState, me: str, cfg: LiveConfig, said: Dict[int, str], heard: Dict[int, str]) -> str:
    """The public history from the opponent's seat. ``said`` = the opponent's own earlier SAY lines by event
    index; ``heard`` = what the focal said (realized text) by event index; typed descriptions otherwise."""
    them = other(me)
    lines = [f"Opportunities remaining in the shared budget: {state.remaining} of {cfg.horizon}."]
    if state.events:
        lines.append("So far:")
        for idx, e in enumerate(state.events):
            mine = e.actor == me
            who = "You" if mine else f"The {them}"
            a = e.action
            words = said.get(idx) if mine else heard.get(idx)
            tail = f' You said: "{words}"' if (mine and words) else (f' They said: "{words}"' if words else "")
            if a.kind == PROPOSE:
                lines.append(f"  - {who} proposed {deal_words(a.deal_id)} [{a.deal_id}].{tail}")
            elif a.kind == ASK:
                q = BANK_BY_ID[a.question_id]
                if mine:
                    lines.append(f"  - You asked: '{q.text}' The {them} answered: {e.answer}.")
                else:
                    ans = said.get(idx, e.answer)
                    lines.append(f"  - The {them} asked you: '{q.text}' You replied: \"{ans}\"")
            elif a.kind == ACCEPT:
                lines.append(f"  - {who} accepted.{tail}")
            elif a.kind == REJECT:
                lines.append(f"  - {who} rejected the standing proposal.{tail}")
            elif a.kind == HOLD:
                lines.append(f"  - {who} held the standing proposal.{tail}")
            elif a.kind == WALK:
                lines.append(f"  - {who} walked away.{tail}")
    else:
        lines.append("Nothing has happened yet; you move first.")
    st = state.standing
    if st is not None:
        owner = "your" if st.proposer == me else f"the {them}'s"
        lines.append(f"Standing proposal: {deal_words(st.deal_id)} [{st.deal_id}] ({owner}).")
    else:
        lines.append("No proposal is standing.")
    if state.at_deadline:
        lines.append("The budget is exhausted: this is the final decision.")
    return "\n".join(lines)


def describe(action: Action) -> str:
    if action.kind == PROPOSE:
        return f"Propose {deal_words(action.deal_id)}"
    if action.kind == ASK:
        return f"Ask: '{BANK_BY_ID[action.question_id].text}'"
    return {ACCEPT: "Accept the standing proposal", REJECT: "Reject the standing proposal",
            HOLD: "Hold your standing proposal", WALK: "Walk away"}[action.kind]


# up to 39 moves can be legal (27 proposals + 9 questions + accept / reject / hold / walk): one- then two-letter labels
LABELS = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase[:2] for b in string.ascii_uppercase]


def lettered(legal: List[Action], rng: Optional[random.Random]) -> Dict[str, Action]:
    acts = sorted(legal, key=lambda a: a.key)
    if rng is not None:
        rng.shuffle(acts)
    assert len(acts) <= len(LABELS)
    return {label: a for label, a in zip(LABELS, acts)}


def moves_text(letters: Dict[str, Action]) -> str:
    return "\n".join(f"  {L}. {describe(a)}" for L, a in letters.items())


# ---- parsing ---------------------------------------------------------------------------------------

_DECISION_RE = re.compile(r"^\s*`?\s*DECISION\s*:\s*\(?([A-Z]{1,2})\)?\s*`?\s*\.?\s*$", re.M)
_SAY_RE = re.compile(r"^\s*`?\s*SAY\s*:\s*(.*)$", re.M | re.S)


def parse_reply(text: str, letters) -> Tuple[Optional[str], str]:
    """(decision label or None, what was said). ``letters`` is the collection of valid labels (a string of
    single letters or any iterable of labels). The LAST well-formed DECISION line counts; the SAY text runs
    from the SAY tag to the end of the reply, trailing fences stripped."""
    valid = set(letters)
    letter = None
    for m in _DECISION_RE.finditer(text):
        if m.group(1) in valid:
            letter = m.group(1)
    say = ""
    m = _SAY_RE.search(text)
    if m:
        say = m.group(1).strip().rstrip("`").strip()
        # a DECISION line after SAY (rare) is not speech
        say = _DECISION_RE.split(say)[0].strip()
    return letter, say


def parse_answer(text: str) -> str:
    m = _SAY_RE.search(text)
    t = (m.group(1) if m else text).strip().rstrip("`").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t


# ---- the opponent ------------------------------------------------------------------------------------

@dataclass
class OpponentInfo:
    kind: str                    # move | answer
    well_formed: bool
    retried: bool
    defaulted: bool              # fell back to the kernel
    letter: Optional[str] = None
    letter_to_key: Dict[str, str] = field(default_factory=dict)
    reply: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""


class LLMOpponent:
    def __init__(self, client, brief_name: str, role: str, ranking, cfg: LiveConfig, *, fallback: T1Kernel,
                 rng: random.Random, shuffle: bool = True):
        self.client, self.brief_name, self.role, self.ranking, self.cfg = client, brief_name, role, tuple(ranking), cfg
        self.fallback, self.rng, self.shuffle = fallback, rng, shuffle
        self.system = brief(brief_name, role, self.ranking, cfg)
        self.said: Dict[int, str] = {}        # event index -> my SAY
        self.heard: Dict[int, str] = {}       # event index -> the focal's realized words
        self.calls: List[dict] = []
        self.last_info: Optional[OpponentInfo] = None

    def _complete(self, user: str, max_tokens: int):
        reply = self.client.complete(self.system, [{"role": "user", "content": user}], max_tokens)
        self.calls.append({"user": user, "reply": reply.text, "model": reply.model, "stop_reason": reply.stop_reason,
                           "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens})
        return reply

    def move(self, state: PublicState) -> Tuple[Action, str]:
        assert state.turn == self.role and state.terminal is None
        legal = legal_actions(state, self.role)
        letters = lettered(legal, self.rng if self.shuffle else None)
        user = (state_text(state, self.role, self.cfg, self.said, self.heard) + "\n\nYour legal moves:\n" +
                moves_text(letters) + "\n\nChoose one move and say something to the " + other(self.role) + ".")
        reply = self._complete(user, MAX_TOKENS)
        letter, say = parse_reply(reply.text, list(letters))
        retried = False
        if letter is None:
            reply = self._complete(user, RETRY_MAX_TOKENS)
            letter, say = parse_reply(reply.text, list(letters))
            retried = True
        defaulted = letter is None
        action = letters[letter] if letter is not None else self.fallback.sample(state, self.ranking, self.rng)
        if defaulted:
            say = ""
        self.said[len(state.events)] = say
        self.last_info = OpponentInfo("move", not defaulted, retried, defaulted, letter, {L: a.key for L, a in letters.items()},
                                      reply.text[:600], {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens},
                                      reply.model)
        return action, say

    def answer(self, state: PublicState, question_id: str) -> str:
        q = BANK_BY_ID[question_id]
        user = (state_text(state, self.role, self.cfg, self.said, self.heard) +
                f"\n\nThe {other(self.role)} asks you: '{q.text}'\nReply with the SAY: line only.")
        reply = self._complete(user, MAX_TOKENS)
        text = parse_answer(reply.text)
        retried = False
        if not text:
            reply = self._complete(user, RETRY_MAX_TOKENS)
            text = parse_answer(reply.text)
            retried = True
        defaulted = not text
        if defaulted:
            text = truthful_answer(question_id, self.ranking)
        self.said[len(state.events)] = text
        self.last_info = OpponentInfo("answer", not defaulted, retried, defaulted, None, {}, reply.text[:600],
                                      {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}, reply.model)
        return text


# ---- offline clients ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Completion:
    text: str
    model: str = "scripted"
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


class ScriptedClient:
    """Returns the given replies in order (the last one repeats)."""

    def __init__(self, replies: List[str]):
        self.replies, self.i, self.calls = list(replies), 0, []

    def complete(self, system, messages, max_tokens) -> Completion:
        self.calls.append((system, messages, max_tokens))
        text = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return Completion(text)


class KernelClient:
    """A client that behaves like the T1 kernel: picks the letter of the kernel's sampled move and emits a
    templated SAY; answers questions with the truthful word. Always well-formed. For offline pilots."""

    def __init__(self, kernel: T1Kernel, ranking, rng: random.Random, cfg: LiveConfig):
        self.kernel, self.ranking, self.rng, self.cfg = kernel, tuple(ranking), rng, cfg

    def complete(self, system, messages, max_tokens) -> Completion:
        user = messages[-1]["content"]
        m = re.search(r"asks you: '(.*)'\nReply with the SAY", user)
        if m:
            qid = next(q.id for q in BANK_BY_ID.values() if q.text == m.group(1))
            return Completion(f"SAY: {truthful_answer(qid, self.ranking).capitalize()}.")
        # a move: recover the lettered keys from the prompt and pick the kernel's sample among them
        keys = dict(re.findall(r"^\s+([A-Z]{1,2})\. (.*)$", user, re.M))
        # the kernel needs the state; ``KernelOpponent`` binds it before each move
        if self._state is None:
            raise RuntimeError("KernelClient.bind_state(state) must be called before a move")
        act = self.kernel.sample(self._state, self.ranking, self.rng)
        letter = next(L for L, d in keys.items() if d == describe(act))
        return Completion(f"DECISION: {letter}\nSAY: {self._say(act)}")

    _state: Optional[PublicState] = None

    def bind_state(self, state: PublicState) -> None:
        self._state = state

    @staticmethod
    def _say(act: Action) -> str:
        if act.kind == PROPOSE:
            return f"How about {deal_words(act.deal_id)}?"
        return {ACCEPT: "That works for me.", REJECT: "I can't do that.", HOLD: "My offer stands.",
                WALK: "Then we have no deal.", ASK: "Let me ask you something."}[act.kind]


class KernelOpponent(LLMOpponent):
    """``LLMOpponent`` driven by a ``KernelClient``: binds the state before every move so the offline
    opponent sees what the kernel would."""

    def move(self, state: PublicState):
        if isinstance(self.client, KernelClient):
            self.client.bind_state(state)
        return super().move(state)
