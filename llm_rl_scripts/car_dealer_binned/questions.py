"""The question bank for Binned Car Dealer, Condition B.

Every question is asked *of the opponent* and is about the opponent's private
priority ranking over the three deal attributes. Each question has an explicit,
public response distribution conditional on the opponent's type, so expected
information gain is exact (design doc section 2, "a crucial experimental
control": truthful questions with known response probabilities).

The bank is nine binary questions over the six strict rankings:

  * TOP(a)      "Is <a> the attribute you care about most?"   yes for 2 of 6 types
  * ABOVE(a,b)  "Do you care more about <a> than about <b>?"  yes for 3 of 6 types
  * BOTTOM(a)   "Is <a> the attribute you care about least?"  yes for 2 of 6 types

From the uniform prior an ABOVE question is worth exactly 1 bit and a TOP or
BOTTOM question H(1/3) ~= 0.918 bits; three ABOVE answers identify the type.
A question's EIG falls to 0 once the current support already fixes its answer,
which is how low-information "distractor" options arise naturally late in an
episode (design doc section 2, option C).

Response model (public):
    P(answer = truthful(theta) | q, theta) = 1 - noise
    P(answer = other          | q, theta) = noise
with ``noise = 0`` (deterministic, the default) or a Fraction in (0, 1/2).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from typing import Dict, List, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, ATTRIBUTES, Ranking

ANSWERS: Tuple[str, str] = ("yes", "no")


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    # truthful yes/no per ranking, aligned with ALL_RANKINGS
    truthful_yes: Tuple[bool, ...]

    def truthful_answer(self, ranking: Ranking) -> str:
        return "yes" if self.truthful_yes[ALL_RANKINGS.index(ranking)] else "no"

    @lru_cache(maxsize=None)
    def likelihoods(self, noise: Fraction) -> Dict[str, Tuple[Fraction, ...]]:
        """{answer: (P(answer | theta) for theta in ALL_RANKINGS)}."""
        if not (Fraction(0) <= noise < Fraction(1, 2)):
            raise ValueError("noise must be a Fraction in [0, 1/2)")
        yes = tuple((1 - noise) if t else noise for t in self.truthful_yes)
        no = tuple(noise if t else (1 - noise) for t in self.truthful_yes)
        return {"yes": yes, "no": no}


def _top(a: str) -> Question:
    return Question(
        id=f"top:{a}",
        text=f"Is {a} the attribute you care about most?",
        truthful_yes=tuple(r[0] == a for r in ALL_RANKINGS),
    )


def _above(a: str, b: str) -> Question:
    return Question(
        id=f"above:{a}>{b}",
        text=f"Do you care more about {a} than about {b}?",
        truthful_yes=tuple(r.index(a) < r.index(b) for r in ALL_RANKINGS),
    )


def _bottom(a: str) -> Question:
    return Question(
        id=f"bottom:{a}",
        text=f"Is {a} the attribute you care about least?",
        truthful_yes=tuple(r[2] == a for r in ALL_RANKINGS),
    )


def build_bank() -> List[Question]:
    """The fixed, ordered bank. Order is the tie-break for equal EIG."""
    bank: List[Question] = [_top(a) for a in ATTRIBUTES]
    pairs = [(ATTRIBUTES[0], ATTRIBUTES[1]), (ATTRIBUTES[0], ATTRIBUTES[2]), (ATTRIBUTES[1], ATTRIBUTES[2])]
    bank += [_above(a, b) for a, b in pairs]
    bank += [_bottom(a) for a in ATTRIBUTES]
    return bank


BANK: Tuple[Question, ...] = tuple(build_bank())
BANK_BY_ID: Dict[str, Question] = {q.id: q for q in BANK}
