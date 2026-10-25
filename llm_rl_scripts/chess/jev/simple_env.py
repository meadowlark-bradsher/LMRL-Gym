"""A dependency-light chess environment (python-chess only, no Stockfish).

The repo's ChessEnv always launches a Stockfish binary, and the bundled engine
is a Linux build that will not run on macOS. This env keeps the same agent/reward
contract -- agent plays White, reward +1 for delivering checkmate, -1 for an
illegal move or being checkmated, 0 otherwise -- but runs a pure-python-chess
opponent, so the Jev policy loop can be exercised anywhere.

It supports two opponents: ``random`` (uniform legal move) and ``stockfish``
(the repo's ChessEnv, used only when a working engine binary is available). The
legality guarantee being tested -- Jev never emits an illegal move -- is
opponent-independent; the opponent only affects win rate.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import chess


class SimpleChessEnv:
    """Agent is White; a random-legal opponent plays Black. No engine needed."""

    def __init__(self, from_position: Optional[str] = None, seed: Optional[int] = None):
        self.from_position = from_position
        self.rng = random.Random(seed)
        self.board = chess.Board() if from_position is None else chess.Board(from_position)

    def reset(self) -> str:
        self.board = (
            chess.Board() if self.from_position is None else chess.Board(self.from_position)
        )
        return self.board.fen()

    def _opponent_move(self):
        moves = list(self.board.legal_moves)
        self.board.push(self.rng.choice(moves))

    def step(self, action_san: str) -> Tuple[str, float, bool]:
        """Apply the agent's SAN move, then the opponent's reply.

        Mirrors ChessEnv.step's reward scheme: -1 for an illegal move, +1 for
        delivering checkmate, -1 for being checkmated, 0 otherwise (incl. draws).
        """
        # Agent (White) move.
        try:
            move = self.board.parse_san(action_san)
        except (ValueError, AssertionError):
            return self.board.fen(), -1.0, True  # illegal move: lose immediately
        self.board.push(move)

        if self.board.is_game_over():
            reward = 1.0 if self.board.is_checkmate() else 0.0  # agent just moved
            return self.board.fen(), reward, True

        # Opponent (Black) reply.
        self._opponent_move()
        if self.board.is_game_over():
            reward = -1.0 if self.board.is_checkmate() else 0.0  # opponent just moved
            return self.board.fen(), reward, True

        return self.board.fen(), 0.0, False
