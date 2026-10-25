"""A Jev-driven policy for Wordle, choosing from the BED-reduced shortlist.

The BED posterior (words still consistent with the feedback) is Jev's Choice
option set. Jev never generates a word -- it only selects one from a list it is
handed, so by construction it can only ever play a still-possible word. The open
question this isolates is *selection quality*: given the exact posterior, does
Jev pick as informatively as the exact-EIG optimum?

The Choice primitive caps at 255 options. When the posterior is larger than that
(early turns on the big vocabularies), Jev cannot see the whole set, so this
policy plays a *fixed, model-free opener* on those turns instead -- the same word
regardless of arm -- and only hands the decision to Jev once the posterior has
shrunk to <= the cap. That keeps the opener from smuggling in any EIG signal:
the thing under test is Jev on the shortlist, not the opener.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from LLM_RL.jev.client import MAX_CHOICE_OPTIONS, ChoiceAnswer, JevClient
from llm_rl_scripts.wordle.jev.bed import FixedTargetWordle

INSTRUCTIONS = (
    "You are playing Wordle. A hidden five-letter word must be found in six "
    "guesses. After each guess every letter is marked green (correct letter in "
    "the correct position), yellow (the letter is in the word but in a "
    "different position), or gray (the letter is not in the word). You are "
    "given the board so far and the complete list of words still consistent "
    "with all the feedback. Choose the single word that will most quickly "
    "narrow the possibilities down to the hidden word."
)

_COLOR = {"<g>": "green", "<y>": "yellow", "<b>": "gray"}

# Model-free openers used only while the posterior exceeds the Choice cap. These
# are the repo's curated start words (scripted_policies.StartWordPolicy); the
# first one still consistent with the board is played. They are a fixed policy,
# identical across arms, never chosen by EIG.
DEFAULT_OPENERS = [
    "slate", "crane", "raise", "arose", "stare", "slice", "share", "saute",
    "trace", "crate", "adieu", "audio", "roast", "ratio",
]


def render_board(game: FixedTargetWordle) -> str:
    """The guesses and their color feedback, one line per guess."""
    if not game.guesses:
        return "(no guesses yet)"
    lines = []
    for guess, pattern in zip(game.guesses, game.patterns):
        marks = [pattern[i : i + 3] for i in range(0, len(pattern), 3)]
        cells = ["{}={}".format(c, _COLOR[m]) for c, m in zip(guess, marks)]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def build_state(game: FixedTargetWordle, shortlist: List[str]) -> str:
    """The state string handed to Jev: the board, the turn, and the candidates."""
    parts = [
        "Wordle. Guess {} of 6.".format(len(game.guesses) + 1),
        "Board so far:\n" + render_board(game),
        "{} words remain consistent with the feedback.".format(len(shortlist)),
    ]
    return "\n\n".join(parts)


@dataclass
class Decision:
    """One turn's decision, with enough detail for the benchmark to score it."""

    word: str
    source: str  # "jev" or "opener"
    shortlist_size: int
    confidence: Optional[float] = None
    answer: Optional[ChoiceAnswer] = None


def choose_opener(shortlist: List[str], openers: List[str]) -> str:
    """The fixed opener: the first curated word still possible, else a
    deterministic fallback (alphabetically-first candidate) so the choice is
    reproducible and model-free."""
    possible = set(shortlist)
    for w in openers:
        if w in possible:
            return w
    return min(shortlist)


class JevWordlePolicy:
    """Selects Wordle guesses from the posterior via Jev's Choice primitive."""

    def __init__(
        self,
        client: JevClient,
        instructions: str = INSTRUCTIONS,
        openers: Optional[List[str]] = None,
        max_options: int = MAX_CHOICE_OPTIONS,
    ):
        self.client = client
        self.instructions = instructions
        self.openers = list(openers) if openers is not None else list(DEFAULT_OPENERS)
        self.max_options = max_options

    def choose(self, game: FixedTargetWordle) -> Decision:
        shortlist = game.posterior
        if not shortlist:
            raise ValueError("empty posterior: no word is consistent with the board")
        n = len(shortlist)

        # Trivial posterior: one word left. Play it; no need to spend a call.
        if n == 1:
            return Decision(word=shortlist[0], source="jev", shortlist_size=n,
                            confidence=1.0)

        if n > self.max_options:
            return Decision(word=choose_opener(shortlist, self.openers),
                            source="opener", shortlist_size=n)

        options = {w: None for w in shortlist}
        answer = self.client.choice(
            state=build_state(game, shortlist),
            instructions=self.instructions,
            options=options,
            question_id="guess",
        )
        if answer.choice not in options:
            raise ValueError(
                "Jev returned {!r}, not one of the {} candidates".format(
                    answer.choice, n
                )
            )
        return Decision(word=answer.choice, source="jev", shortlist_size=n,
                        confidence=answer.confidence, answer=answer)
