"""Persona-blind outcome scorer for narrated Car Dealer conversations.

``score(text)`` takes the raw conversation text and nothing else -- never the planted
persona or budget -- and returns:

* ``outcome``: ``"sale"``, ``"no_sale"`` or ``"unclear"``, decided from the dialogue
  turns alone. The narrator's probability/price annotations (bracketed or parenthesised
  asides, unprefixed narration lines, and any sentence that states a probability) are
  stripped first, so the outcome does not depend on the narrator's self-report.
* ``price``: the dollar amount the sale closes at, when one is stated (else ``None``).
* ``discount_offered``: whether the dealer offers a price reduction in any turn.
* ``self_report``: the narrator's last stated purchase probability in [0, 1], re-parsed
  from the raw text (percentages read as percentages), or ``None``. Kept separate: it is
  the thing the dialogue outcome is later compared against, not an input to it.

The rules are deliberately simple and their validity is measured, not assumed: see
``validate.py`` and the gate in ``PREREGISTRATION_BASELINE2023.md``.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

SPEAKER = re.compile(
    r"^\s*(agent|dealer|seller|salesperson|buyer|customer)\s*(?:\([^)]*\))?\s*:\s*", re.I)
_BUYER_NAMES = {"buyer", "customer"}

_BRACKETED = re.compile(r"\[[^\]]*\]")
_PAREN_NOTE = re.compile(r"\([^)]*(probab|price|predict|likelihood|chance)[^)]*\)", re.I)
_PROB_WORD = re.compile(r"probab|likelihood|predicted", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# -- buyer dialogue acts -----------------------------------------------------------
_REJECT = re.compile(
    r"won'?t be able|will not be able|not be able to (make|purchase|buy|afford|proceed|go)"
    r"|(can'?t|cannot|can not|unable to) (afford|make (a|the) purchase|purchase|buy|proceed|go (over|above|beyond|higher))"
    r"|look(ing)? elsewhere|explore other options|other dealerships?|shop around|keep looking"
    r"|have to pass|i'?ll pass|going to pass|pass on (this|it|the)"
    r"|not (interested|ready to (buy|purchase|commit|make))|no longer interested"
    r"|(won'?t|will not|not going to|don'?t think i (can|will)) (be )?(buy|purchas|tak|proceed|mak)"
    r"|without a discount|have to decline|must decline|i decline|walk away"
    r"|not (going to|gonna) work|doesn'?t work for me|not within my budget, so"
    r"|(need|have|like|want) (some |more )?(time )?to think|think (about it|it over)|sleep on it"
    r"|get back to you|consider (my|other) options|keep [^.!?]{0,25}in mind"
    r"|let you know|(need|take) (some|more) time|exploring other options",
    re.I)
_ACCEPT = re.compile(
    r"i'?ll take (it|the|this|that)|i will take (it|the|this|that)|i'?ll buy|i will buy"
    r"|(we|you) have a deal|it'?s a deal|\bdeal[.!]|let'?s (do (it|this)|go (with|ahead)|proceed|move forward|make it happen|finalize|get (this|it) done)"
    r"|(go ahead|proceed|move forward|go forward) (and|with) (purchas|buy|the purchase|the paperwork|the deal|it|that)"
    r"|ready to (buy|purchase|proceed|move forward|make (a|the) purchase|sign)|i'?m sold|sold!"
    r"|willing to (buy|purchase|go with|pay|proceed|accept)|happy to (buy|purchase|proceed|accept)"
    r"|i (accept|agree)|i'?ll pay|i will pay|i'?ll go (with|for|ahead)|i will go (with|for|ahead)"
    r"|(like|want|love) to (purchase|buy|proceed|move forward|go ahead)"
    r"|decided to (buy|purchase|go|take)|satisfied with (the|my) purchase|where do i sign"
    r"|let'?s (finalize|complete|close) |i'?ll (purchase|proceed)|make the purchase\b"
    r"|\b(i'?ll|i will|i'?m going to|i am going to|i'?d like to|i would like to|i want to"
    r"|i'?m ready to|i am ready to|i'?ve decided to|i have decided to)\b[^.!?]{0,45}"
    r"\b(buy|purchase|take (it|the|this|that)|proceed|go ahead|move forward)"
    r"|excited about (this|the|my) (purchase|new car)|thrilled with the (price|deal|purchase)",
    re.I)
_THANKS_EXIT = re.compile(
    r"thank(s| you)( so much| very much)? for (your|the) (time|help|assistance|information|understanding)"
    r"|i appreciate your (help|time|assistance)", re.I)
_ACCEPT_NEGATED = re.compile(
    r"(not|n't|never|unless|if you|if we|if i|only if|would .* if|before i)\b[^.!?]{0,40}"
    r"(take it|buy|purchase|deal|proceed|go ahead|sign|pay)", re.I)

# -- dealer dialogue acts ----------------------------------------------------------
_DEALER_CLOSE = re.compile(
    r"congratulations|enjoy your new|thank you for your purchase"
    r"|(proceed with|start|begin|prepare|finalize|complete|take care of|get started (on|with)|move forward with)"
    r" (the |your |all the |all of the )?(necessary |required )?paperwork"
    r"|glad we (could|were able to) (reach|come to|find|make) (an agreement|a deal|terms)"
    r"|pleasure doing business|you'?ve made (a|an) (great|excellent|wise|wonderful) (choice|decision)",
    re.I)
_CONDITIONAL = re.compile(
    r"\b(if|when|once|whenever|should|in case) (you|we)\b|\bif you'?re\b|\bwhenever you'?re\b", re.I)
_DEALER_NO_SALE = re.compile(
    r"sorry to hear|if you change your mind|good luck (with|in|on) (your|finding)"
    r"|sorry (i|we) couldn'?t|wish you (the best|luck)|hope you find|understand your decision"
    r"|(don'?t|do not) have any (options|cars|vehicles|models) (within|in) your (budget|price)",
    re.I)

# -- discount ----------------------------------------------------------------------
_DISCOUNT = re.compile(
    r"discount of|offer you (a|an) [^.!?]{0,30}discount|give you (a|an) [^.!?]{0,30}discount"
    r"|\d+\s?% (discount|off)|\$[\d,.]+k? (off|discount)|(discount|reduction) (of|worth) \$"
    r"|(reduce|lower|drop|bring down|knock|cut|take) [^.!?]{0,40}(price|\$)"
    r"|special (discount|price|offer|promotion|deal)|discounted price|price (reduction|cut)"
    r"|additional discount|bigger discount|further discount|extra discount|apply (a|the) discount"
    r"|bring(ing)? (the|it|that) (price )?down to|sale price|promotional"
    r"|(normally|usually|originally|typically) [^.!?]{0,40}(priced|listed|sells?)[^.!?]{0,60}\bbut\b"
    r"|for you, i can|make an exception",
    re.I)
_DISCOUNT_INQUIRY = re.compile(
    r"(check|see|find out) (if|whether|what)|see what (kind|i can)|if (we|there) (have|are|is) any", re.I)
_DISCOUNT_NEGATED = re.compile(
    r"(no|not|n't|never|cannot|unable|without|against|instead of)\b[^.!?]{0,60}"
    r"(discount|reduc|lower|negotiat|drop|cut|bring)"
    r"|prices? (are|is) (fixed|firm|final|non-?negotiable|set|as listed)|non-?negotiable",
    re.I)

# -- money -------------------------------------------------------------------------
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(k\b)?", re.I)
_REDUCTION_BEFORE = re.compile(
    r"(discount of|reduction of|by( an additional| another)?|save( you)?|knock|take|off of"
    r"|additional|another|extra|deduct|rebate of|worth)\s*$", re.I)
_REDUCTION_AFTER = re.compile(r"^\s*(off|discount|rebate|reduction|in savings|less)", re.I)

# "probability ...: 0.8" / "probability is 90%" (never reading through a "$" into a price),
# or the number-first form "0% probability of purchase"
_PROB = re.compile(
    r"(?:probab\w*|likelihood|chance)[^\d\n%$]{0,80}?(?<![\d$,.])(\d+(?:\.\d+)?)\s*(%|percent)?"
    r"|(?<![\d$,.])(\d+(?:\.\d+)?)\s*(%|percent)\s*(?:probab\w*|likelihood|chance)", re.I)


def split_turns(text: str) -> List[Tuple[str, str]]:
    """[(role, utterance)] with role in {"buyer", "dealer"}; annotations removed."""
    turns: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        m = SPEAKER.match(raw)
        if not m:
            continue                                   # narration / annotation line
        role = "buyer" if m.group(1).lower() in _BUYER_NAMES else "dealer"
        utt = raw[m.end():]
        utt = _BRACKETED.sub(" ", utt)
        utt = _PAREN_NOTE.sub(" ", utt)
        sents = [s for s in _SENT_SPLIT.split(utt) if s.strip() and not _PROB_WORD.search(s)]
        utt = " ".join(sents).strip()
        if utt:
            turns.append((role, utt))
    return turns


def _buyer_act(utt: str) -> Optional[str]:
    """Last decisive act in a buyer turn: "accept", "reject" or None."""
    act = None
    for s in _SENT_SPLIT.split(utt):
        if _REJECT.search(s):
            act = "reject"
        elif _ACCEPT.search(s) and not _ACCEPT_NEGATED.search(s) and not s.strip().endswith("?"):
            act = "accept"
    return act


def _dealer_act(utt: str) -> Optional[str]:
    if _DEALER_NO_SALE.search(utt):
        return "no_sale"
    for sent in _SENT_SPLIT.split(utt):
        if _DEALER_CLOSE.search(sent) and not _CONDITIONAL.search(sent):
            return "close"
    return None


def _amounts(utt: str) -> List[float]:
    """Dollar amounts that read as prices, skipping stated reductions."""
    out = []
    for m in _MONEY.finditer(utt):
        val = float(m.group(1).replace(",", ""))
        if m.group(2):
            val *= 1000
        if _REDUCTION_BEFORE.search(utt[max(0, m.start() - 25):m.start()]):
            continue
        if _REDUCTION_AFTER.search(utt[m.end():m.end() + 14]):
            continue
        if val >= 1000:
            out.append(val)
    return out


def self_report(text: str) -> Optional[float]:
    last = None
    for m in _PROB.finditer(text):
        num, pct = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        val = float(num)
        if pct or val > 1:
            val /= 100.0
        if 0.0 <= val <= 1.0:
            last = val
    return last


def score(text: str) -> dict:
    turns = split_turns(text)

    # the chronologically last decisive cue wins, whoever says it
    state, close_idx = None, None                      # "sale" / "no_sale" / None
    for i, (role, utt) in enumerate(turns):
        if role == "buyer":
            act = _buyer_act(utt)
            if act == "accept":
                state, close_idx = "sale", i
            elif act == "reject":
                state, close_idx = "no_sale", None
        else:
            act = _dealer_act(utt)
            if act == "close":
                close_idx = i if state != "sale" else close_idx
                state = "sale"
            elif act == "no_sale":
                state, close_idx = "no_sale", None
    if state is None:
        # no decisive cue anywhere: a buyer who thanks the dealer and leaves did not buy
        buyer_turns = [u for r, u in turns if r == "buyer"]
        if buyer_turns and _THANKS_EXIT.search(buyer_turns[-1]):
            state = "no_sale"

    price = None
    if state == "sale" and close_idx is not None:
        # last price stated up to the closing turn, or in the turn right after it
        for role, utt in turns[:close_idx + 2]:
            amts = _amounts(utt)
            if amts:
                price = amts[-1]

    discount = False
    for role, utt in turns:
        if role != "dealer":
            continue
        for s in _SENT_SPLIT.split(utt):
            if (_DISCOUNT.search(s) and not _DISCOUNT_NEGATED.search(s)
                    and not _DISCOUNT_INQUIRY.search(s)):
                discount = True

    return {"outcome": state or "unclear", "price": price, "discount_offered": discount,
            "self_report": self_report(text), "n_turns": len(turns)}
