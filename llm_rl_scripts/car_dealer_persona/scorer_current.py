"""Scorer revision for CD-PERSONA-1: the frozen 2023 rules, applied to current-model text.

``scorer.py`` is frozen (it is what the CD-PERSONA-0 results regenerate from), so this module
wraps it rather than editing it. Same contract -- ``score(text)`` reads the words alone, never
the persona, budget or decision token -- and the same return keys. Two things change, both
developed on the pilot DEV halves only (``PREREGISTRATION_CURRENT.md`` §Scorer re-gate):

1. **Normalisation.** `claude-sonnet-5-5` answers the 2023 narrator prompt in markdown
   (``**Agent:**``, ``# Car Dealership Conversation``, ``---``, backticked annotations). The
   frozen ``SPEAKER`` regex anchors on a bare ``Agent:``, so every such story split into zero
   turns. ``normalize`` strips the markup; the frozen splitter then applies unchanged.

2. **Extra dialogue acts.** The buyer agent (Arm A) leaves in words the 2023 narrator never
   used -- "I'm done here", "I'm leaving", "I'll take that as a no", "check with other
   dealers" -- which the frozen rules scored ``unclear`` or, through "sign"/"take that",
   as a sale. ``_REJECT_MORE`` / ``_ACCEPT_MORE`` extend the frozen patterns; a reject
   still wins over an accept in the same sentence, as before.

Everything else (dealer acts, discount detector, price, self-report, last-cue-wins) is the
frozen code, called directly.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import scorer as s

# -- markdown normalisation ----------------------------------------------------------
_BOLD_SPEAKER = re.compile(
    r"^\s*(?:[>*_`\s]*)\**\s*(agent|dealer|seller|salesperson|buyer|customer)"
    r"(\s*\([^)]*\))?\s*\**\s*:\s*\**\s*", re.I)
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+.*$", re.M)
_MD_RULE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", re.M)
_INLINE_MARK = re.compile(r"[*_`]+")


def normalize(text: str) -> str:
    """Markdown story -> the 2023 plain layout. Speaker lines become ``Agent: ...`` /
    ``Buyer: ...``; headers and horizontal rules go; bold/italic/code marks are removed
    everywhere else (they only ever decorate annotations and stage directions)."""
    text = _MD_HEADER.sub("", text)
    text = _MD_RULE.sub("", text)
    out = []
    for line in text.splitlines():
        m = _BOLD_SPEAKER.match(line)
        if m:
            role = "Buyer" if m.group(1).lower() in s._BUYER_NAMES else "Agent"
            line = f"{role}: {_INLINE_MARK.sub('', line[m.end():])}"
        else:
            line = _INLINE_MARK.sub("", line)
        out.append(line)
    return "\n".join(out)


# -- buyer dialogue acts, extended --------------------------------------------------
_REJECT_MORE = re.compile(
    # "I'm done" as a walk-out, not the temporal "have it ready when I'm done"
    r"(?<!when )(?<!once )(?<!after )(?<!until )(?<!till )\bi'?m done\b(?! (with (the|my) (walk|test|look|inspection)|checking|looking at))"
    r"|\bi'?m (leaving|walking|out|moving on|going to (look|check|go)|gonna (look|check|go))\b"
    r"|\bwe'?re done here\b|\bi'?ll (look|check|go) (at|with|to|into) (other|another|some other|a different)"
    r"|\bgo with (them|another dealer|the other (dealer|dealership|guy|place))\b"
    r"|\b(look|check|going) (at|with|into|for) (other|another|some other|a different) (options?|dealers?|dealerships?|cars?|places?)"
    r"|\b(my )?business elsewhere|\btake that as a no\b|\bthat'?s a no\b|\bcount me out\b"
    r"|\bnot worth it\b|\bi'?ll keep (looking|shopping)"
    r"|\b(i'?m|i am) (not|no longer) (buying|signing|paying|interested|going to (buy|sign|pay))"
    r"|\bi (won'?t|will not|can'?t|cannot) (be )?(buy|sign|pay)"
    r"|\bwalk(ing)? (out|away)|\bi'?d rather (walk|leave|pass)|\bgoing to walk\b|\bi'?ll walk\b"
    r"|\bnot (going to|gonna) (move forward|sign|buy|pay|do (it|this|that))"
    r"|\bthis (isn'?t|is not|won'?t) (going to )?work( out)?\b|\bwe'?re at an impasse\b"
    r"|\bgood ?bye\b|\bhave a (good|nice|great) (day|one)\b",
    re.I)
_ACCEPT_MORE = re.compile(
    r"\bwrite it up\b|\blet'?s (sign|write|do the paperwork|finalize|get it done|wrap (it|this) up)"
    r"|\bi'?ll sign\b|\bwhere do i sign\b|\byou'?ve got a deal\b|\byou have a deal\b"
    r"|\bthat works for me\b|\bthat'?ll work\b|\bthat works\.|\bsold\b|\bi'?m in\b"
    r"|\b(yes|yeah|okay|ok|fine|alright|all right),? (let'?s|i'?ll|we can|go ahead|write|do it)"
    r"|\bi'?ll take it at\b|\bi accept\b|\bdone deal\b|\bthat'?s a deal\b|\bmake it happen\b"
    # the buyer agent's way of closing: it asks for the paperwork
    r"|\b(get|do|start|finish|handle|skip to|bring|have|draw up|prepare) (the|this|that|my|your) paperwork\b"
    r"|\bpaperwork (done|ready|started|going|out)\b|\bget (this|it) (done|signed|finalized)\b",
    re.I)
_ACCEPT_NEGATED_MORE = re.compile(
    r"(not|n't|never|unless|if you|if we|if i|only if|would .* if|before i|until you|until i|^\s*if\b)"
    r"\b[^.!?]{0,40}(write|sign|deal|take|work|paperwork|in\b)", re.I)
# a question that is itself the commitment ("Where do I sign?"), exempt from the no-questions rule
_ACCEPT_QUESTION = re.compile(r"\bwhere do i sign\b|\bwhen can i (sign|pick it up|take it home)\b", re.I)

# -- dealer act, extended ------------------------------------------------------------
# offers in the shape of a close ("$11,000 and we can start the paperwork", "shall I get the
# paperwork started?") are conditional, not closes
_CONDITIONAL_MORE = re.compile(
    r"\b(we|i) (can|could|would|will) (start|begin|get|do)\b.*\bpaperwork\b|\bshall (i|we)\b"
    r"|\bwould you like\b|\bif (that|this|it) works\b|\b\$[\d,]+ and (we|i)\b|\?\s*$", re.I)
_DEALER_CLOSE_MORE = re.compile(
    r"\b(get|have) (the|your) paperwork (started|ready|going|drawn up)\b"
    r"|\bi'?ll (get|have|bring) the (paperwork|keys|contract)\b|\bright this way\b", re.I)


def _buyer_act(utt: str) -> Optional[str]:
    act = None
    for sent in s._SENT_SPLIT.split(utt):
        if s._REJECT.search(sent) or _REJECT_MORE.search(sent):
            act = "reject"
        elif sent.strip().endswith("?") and not _ACCEPT_QUESTION.search(sent):
            continue
        elif s._ACCEPT.search(sent) and not s._ACCEPT_NEGATED.search(sent):
            act = "accept"
        elif _ACCEPT_MORE.search(sent) and not _ACCEPT_NEGATED_MORE.search(sent) \
                and not s._ACCEPT_NEGATED.search(sent):
            act = "accept"
    return act


def _dealer_act(utt: str) -> Optional[str]:
    if s._DEALER_NO_SALE.search(utt):
        return "no_sale"
    for sent in s._SENT_SPLIT.split(utt):
        if _CONDITIONAL_MORE.search(sent) or s._CONDITIONAL.search(sent):
            continue
        if s._DEALER_CLOSE.search(sent) or _DEALER_CLOSE_MORE.search(sent):
            return "close"
    return None


# -- money, extended -------------------------------------------------------------------
# "$1,500 over what I wanted to spend" / "$3,000 under budget" are differences, not prices
_DIFFERENCE_AFTER = re.compile(
    r"^\s?(over|above|under|below|past|beyond|more|less|higher|lower|short|shy|too (much|high))\b", re.I)


def _amounts(utt: str) -> List[float]:
    out = []
    for m in s._MONEY.finditer(utt):
        val = float(m.group(1).replace(",", ""))
        if m.group(2):
            val *= 1000
        if s._REDUCTION_BEFORE.search(utt[max(0, m.start() - 25):m.start()]):
            continue
        if s._REDUCTION_AFTER.search(utt[m.end():m.end() + 14]):
            continue
        if _DIFFERENCE_AFTER.search(utt[m.end():m.end() + 14]):
            continue
        if val >= 1000:
            out.append(val)
    return out


def split_turns(text: str) -> List[Tuple[str, str]]:
    return s.split_turns(normalize(text))


def score(text: str) -> dict:
    """Same keys as ``scorer.score``; same last-cue-wins logic with the extended buyer acts."""
    turns = split_turns(text)
    state, close_idx = None, None
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
        buyer_turns = [u for r, u in turns if r == "buyer"]
        if buyer_turns and s._THANKS_EXIT.search(buyer_turns[-1]):
            state = "no_sale"

    price = None
    if state == "sale" and close_idx is not None:
        for role, utt in turns[:close_idx + 2]:
            amts = _amounts(utt)
            if amts:
                price = amts[-1]

    discount = False
    for role, utt in turns:
        if role != "dealer":
            continue
        for sent in s._SENT_SPLIT.split(utt):
            if (s._DISCOUNT.search(sent) and not s._DISCOUNT_NEGATED.search(sent)
                    and not s._DISCOUNT_INQUIRY.search(sent)):
                discount = True

    return {"outcome": state or "unclear", "price": price, "discount_offered": discount,
            "self_report": s.self_report(normalize(text)), "n_turns": len(turns)}
