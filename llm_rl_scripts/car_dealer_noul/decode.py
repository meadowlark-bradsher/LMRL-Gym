"""Decoders: an utterance -> a per-type likelihood vector (`DESIGN_NOUL.md` §3).

Phase 0 ships the two decoders that need no model: ``oracle`` (the true yes/no, the
ceiling) and ``regex`` (a keyword parser, the control that asks whether a model is needed
at all). Both end in ``answer_vector``: given P(the answer was YES), the per-type likelihood
is the mixture of the bank's two known answer likelihoods under the hedged answer model.
The Noul and Sonnet decoders are added when their matrices are built; they feed the same
function.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID

YES, NO = "yes", "no"

# Leading-clause cues. The first cue to appear in the reply decides; none -> unparsed.
_YES_CUES = (
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\babsolutely\b", r"\bdefinitely\b", r"\bcertainly\b",
    r"\bof course\b", r"\bindeed\b", r"\bthat'?s right\b", r"\bthat'?s correct\b", r"\bcorrect\b",
    r"\bi would say so\b", r"\bexactly\b", r"\bvery much so\b",
    r"\bwithout a doubt\b", r"\byou'?ve got it\b", r"\bspot on\b", r"\bit sure is\b", r"\bsure is\b",
)
_NO_CUES = (
    r"\bno\b", r"\bnope\b", r"\bnot really\b", r"\bnot at all\b", r"\bi don'?t\b", r"\bit isn'?t\b",
    r"\bit'?s not\b", r"\bhardly\b", r"\bi wouldn'?t say\b", r"\bnot the\b", r"\bnot my\b",
    r"\bafraid not\b", r"\bnot quite\b", r"\bnot exactly\b", r"\bthat'?s not\b", r"\bnot really\b",
    r"\bisn'?t\b", r"\bdoesn'?t\b", r"\bnot what\b", r"\bnot especially\b", r"\bnot particularly\b",
    r"\bi do not\b", r"\bit is not\b", r"\bnot the case\b", r"\bwouldn'?t go that far\b",
)
_YES_RE = re.compile("|".join(_YES_CUES), re.IGNORECASE)
_NO_RE = re.compile("|".join(_NO_CUES), re.IGNORECASE)


def regex_answer(text: str) -> Optional[str]:
    """``yes`` / ``no`` by the earliest cue in the text, else ``None`` (unparsed)."""
    y = _YES_RE.search(text)
    n = _NO_RE.search(text)
    if y is None and n is None:
        return None
    if n is None or (y is not None and y.start() < n.start()):
        return YES
    return NO


def answer_vector(question_id: str, p_yes: float, noise: Fraction = Fraction(0)) -> Tuple[float, ...]:
    """Per-type likelihood of the observed reply given P(reply means YES) = ``p_yes`` and the
    hedged answer model with flip rate ``noise`` (0 = truthful). ``p_yes`` = 0.5 is uninformative."""
    liks = BANK_BY_ID[question_id].likelihoods(Fraction(noise))
    yes, no = liks[YES], liks[NO]
    return tuple(float(p_yes * y + (1.0 - p_yes) * n) for y, n in zip(yes, no))


def oracle_p_yes(truthful: str) -> float:
    return 1.0 if truthful == YES else 0.0


def regex_p_yes(text: str) -> float:
    a = regex_answer(text)
    return 0.5 if a is None else (1.0 if a == YES else 0.0)
