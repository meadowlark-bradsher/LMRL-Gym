"""Gate B/C: explicit game-termination and reward accounting.

The original runner collapsed every non-win, non-loss game into ``reward == 0``
and reported it as a "draw" -- so a hit move-cap (an *unfinished* game) was
silently indistinguishable from a real stalemate or a threefold repetition. This
module replaces that scalar with an explicit taxonomy so that:

  * natural draws (stalemate / insufficient material / repetition / 50-move) are
    each named and counted as draws,
  * a hit move-cap or an API/runtime failure is marked ``truncated`` and reported
    separately -- never silently counted as a chess draw,
  * the terminal loss, any illegal-move penalty, and truncation are distinct.

``classify`` is the single source of truth: given a board and how the loop left
it, it returns a :class:`Termination` from which both the reward and the W/D/L
bucket are derived, so the three can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import chess


class Reason(str, Enum):
    """Why a game ended, from White's (the agent's) point of view."""

    CHECKMATE_WIN = "checkmate_win"          # agent delivered mate
    CHECKMATE_LOSS = "checkmate_loss"        # agent was mated
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    SEVENTYFIVE_MOVES = "seventyfive_moves"  # automatic (no claim needed)
    FIVEFOLD_REPETITION = "fivefold_repetition"
    FIFTY_MOVES = "fifty_moves"              # claimable
    THREEFOLD_REPETITION = "threefold_repetition"
    VARIANT_WIN = "variant_win"
    VARIANT_LOSS = "variant_loss"
    VARIANT_DRAW = "variant_draw"
    # Non-board terminations -- these are NOT chess results:
    ILLEGAL_MOVE = "illegal_move"            # agent emitted an illegal move
    MOVE_CAP = "move_cap"                    # hit the ply cap, game unfinished
    API_ERROR = "api_error"                  # Jev/provider failure mid-game
    RUNNING = "running"                      # not terminated


# Board-rule reasons that are genuine draws.
_NATURAL_DRAWS = frozenset({
    Reason.STALEMATE,
    Reason.INSUFFICIENT_MATERIAL,
    Reason.SEVENTYFIVE_MOVES,
    Reason.FIVEFOLD_REPETITION,
    Reason.FIFTY_MOVES,
    Reason.THREEFOLD_REPETITION,
    Reason.VARIANT_DRAW,
})

# Map python-chess's Termination enum to our reason, needing the winner to split
# checkmate/variant outcomes into win vs loss for White.
_TERMINATION_MAP = {
    chess.Termination.STALEMATE: Reason.STALEMATE,
    chess.Termination.INSUFFICIENT_MATERIAL: Reason.INSUFFICIENT_MATERIAL,
    chess.Termination.SEVENTYFIVE_MOVES: Reason.SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: Reason.FIVEFOLD_REPETITION,
    chess.Termination.FIFTY_MOVES: Reason.FIFTY_MOVES,
    chess.Termination.THREEFOLD_REPETITION: Reason.THREEFOLD_REPETITION,
}


@dataclass(frozen=True)
class Termination:
    """A fully-accounted game result."""

    reason: Reason
    # Reward on the ChessEnv scale: +1 win, -1 loss-or-illegal, 0 otherwise.
    reward: float
    # W/D/L bucket, or None when the game did not finish (truncated / running).
    result: Optional[str]          # "win" | "loss" | "draw" | None
    is_truncated: bool             # True for MOVE_CAP / API_ERROR
    is_decisive: bool              # True for a real win or loss
    detail: str = ""               # free-text note (e.g. exception type)

    @property
    def is_finished(self) -> bool:
        return self.reason not in (Reason.RUNNING, Reason.MOVE_CAP, Reason.API_ERROR)


def _draw(reason: Reason) -> Termination:
    return Termination(reason=reason, reward=0.0, result="draw",
                       is_truncated=False, is_decisive=True)


def classify(
    board: chess.Board,
    *,
    illegal: bool = False,
    move_cap_hit: bool = False,
    api_error: Optional[str] = None,
) -> Termination:
    """Classify how a game ended.

    Precedence: a loop-level failure (illegal move, API error) or a hit move-cap
    is reported as itself, *not* as whatever the board happens to look like. Only
    a genuinely game-over board yields a chess result.
    """
    if api_error is not None:
        return Termination(Reason.API_ERROR, reward=0.0, result=None,
                           is_truncated=True, is_decisive=False, detail=api_error)
    if illegal:
        # Matches ChessEnv: an illegal move is an immediate loss (-1).
        return Termination(Reason.ILLEGAL_MOVE, reward=-1.0, result="loss",
                           is_truncated=False, is_decisive=True)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Board is not over. Either we stopped at the move cap or we're still
        # running -- both are "unfinished", never a draw.
        if move_cap_hit:
            return Termination(Reason.MOVE_CAP, reward=0.0, result=None,
                               is_truncated=True, is_decisive=False)
        return Termination(Reason.RUNNING, reward=0.0, result=None,
                           is_truncated=False, is_decisive=False)

    if outcome.termination == chess.Termination.CHECKMATE:
        # winner is the side that delivered mate; agent is White.
        if outcome.winner == chess.WHITE:
            return Termination(Reason.CHECKMATE_WIN, reward=1.0, result="win",
                               is_truncated=False, is_decisive=True)
        return Termination(Reason.CHECKMATE_LOSS, reward=-1.0, result="loss",
                           is_truncated=False, is_decisive=True)

    if outcome.termination == chess.Termination.VARIANT_WIN:
        win = outcome.winner == chess.WHITE
        return Termination(Reason.VARIANT_WIN if win else Reason.VARIANT_LOSS,
                           reward=1.0 if win else -1.0,
                           result="win" if win else "loss",
                           is_truncated=False, is_decisive=True)
    if outcome.termination == chess.Termination.VARIANT_LOSS:
        return Termination(Reason.VARIANT_LOSS, reward=-1.0, result="loss",
                           is_truncated=False, is_decisive=True)
    if outcome.termination == chess.Termination.VARIANT_DRAW:
        return _draw(Reason.VARIANT_DRAW)

    reason = _TERMINATION_MAP.get(outcome.termination)
    if reason is None:  # pragma: no cover - python-chess added a new draw reason
        # Unknown board termination: still a finished game, record what it was.
        return Termination(Reason.VARIANT_DRAW, reward=0.0, result="draw",
                           is_truncated=False, is_decisive=True,
                           detail=str(outcome.termination))
    return _draw(reason)
