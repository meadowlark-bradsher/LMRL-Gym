"""Random theoretical-endgame positions, via python-chess.

Mirrors the repo's large_piece_random_endgame (chess/env/env.py) but without
importing it (that module pulls in JaxSeq). ``pieces`` is a symbol string like
"kQK" -> black king, white queen, white king. White is always to move, and the
position is valid, not in check, and not already terminal.
"""

from __future__ import annotations

import random
from typing import Optional

import chess


def random_endgame_fen(pieces: str = "kQK", rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random()
    while True:
        board = chess.Board.empty()
        squares = list(range(64))
        rng.shuffle(squares)
        for i, sym in enumerate(pieces):
            board.set_piece_at(squares[i], chess.Piece.from_symbol(sym))
        board.turn = chess.WHITE
        if board.is_valid() and not board.is_check() and not board.is_game_over():
            return board.fen()
