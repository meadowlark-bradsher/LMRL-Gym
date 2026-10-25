"""A Jev-driven policy for the Chess task.

The agent plays White and must output a move in SAN. python-chess enumerates the
legal moves of any position, so the policy offers exactly those as Jev's Choice
options and returns the selected one. Because every option is legal by
construction, this policy *cannot* emit an illegal move -- the failure that costs
the paper's GPT-4 baseline a 24-35% illegal-move rate and a score of 0. The open
question Jev must answer is move *quality*, not legality.
"""

from __future__ import annotations

from typing import List, Optional

import chess

from LLM_RL.jev.client import ChoiceAnswer, JevClient

INSTRUCTIONS = (
    "You are playing the White pieces in a game of chess against an opponent. "
    "Given the current position and the legal moves available, choose the "
    "single strongest move for White."
)


def legal_san_moves(board: chess.Board) -> List[str]:
    """SAN strings for every legal move in the position."""
    return [board.san(m) for m in board.legal_moves]


class JevChessPolicy:
    """Chooses White's move from the position's legal SAN moves."""

    def __init__(self, client: JevClient, instructions: str = INSTRUCTIONS):
        self.client = client
        self.instructions = instructions
        self.last_answer: Optional[ChoiceAnswer] = None

    def build_state(self, board: chess.Board, history_san: str = "") -> str:
        # Give Jev the board three ways: an ASCII diagram, the FEN, and the SAN
        # move list so far (maintained by the caller). All are standard,
        # unambiguous chess representations.
        parts = [
            "You are White. Current position (White to move):",
            "Board:\n" + str(board),
            "FEN: " + board.fen(),
        ]
        if history_san.strip():
            parts.append("Moves so far (SAN): " + history_san.strip())
        return "\n\n".join(parts)

    def choose(self, board: chess.Board, history_san: str = "") -> str:
        sans = legal_san_moves(board)
        if not sans:
            raise ValueError("no legal moves in this position")
        options = {san: None for san in sans}
        answer = self.client.choice(
            state=self.build_state(board, history_san),
            instructions=self.instructions,
            options=options,
            question_id="move",
        )
        self.last_answer = answer
        return answer.choice
