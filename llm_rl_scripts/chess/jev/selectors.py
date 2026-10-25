"""P1 baselines: legal-action selectors that use no model and no search.

Both are shown exactly the same legal-move menu as Jev, so any gap between Jev
and these is attributable to Jev's decision quality, not to the legality
guarantee (which all three share by construction).

  * ``uniform_legal``  -- the minimum baseline: pick a legal move at random.
  * ``heuristic_legal`` -- cheap, fixed rules: deliver mate if possible, else
    give check, else take the most valuable piece (MVV-LVA), else promote, else
    a random legal move. No lookahead, no engine.

A selector takes (board, rng) and returns a move in UCI, so the runner can score
it with the same evaluator it uses for Jev.
"""

from __future__ import annotations

import random
from typing import List

import chess

_PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def uniform_legal(board: chess.Board, rng: random.Random) -> str:
    return rng.choice(list(board.legal_moves)).uci()


def _delivers_mate(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    mate = board.is_checkmate()
    board.pop()
    return mate


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA-ish: value of the captured piece minus a fraction of the mover."""
    if not board.is_capture(move):
        return -1
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        victim_piece = board.piece_at(move.to_square)
        victim = victim_piece.piece_type if victim_piece else chess.PAWN
    attacker_piece = board.piece_at(move.from_square)
    attacker = attacker_piece.piece_type if attacker_piece else chess.PAWN
    return _PIECE_VALUE[victim] * 10 - _PIECE_VALUE[attacker]


def heuristic_legal(board: chess.Board, rng: random.Random) -> str:
    moves: List[chess.Move] = list(board.legal_moves)

    # 1. Checkmate in one.
    mates = [m for m in moves if _delivers_mate(board, m)]
    if mates:
        return rng.choice(mates).uci()

    # 2. Best capture by MVV-LVA (only if it actually wins material-ish).
    captures = [(m, _capture_value(board, m)) for m in moves if board.is_capture(m)]
    captures = [(m, v) for m, v in captures if v > 0]
    if captures:
        best = max(v for _, v in captures)
        return rng.choice([m for m, v in captures if v == best]).uci()

    # 3. Giving check.
    checks = [m for m in moves if board.gives_check(m)]
    if checks:
        return rng.choice(checks).uci()

    # 4. Promotion (to queen preferred).
    promos = [m for m in moves if m.promotion == chess.QUEEN]
    if promos:
        return rng.choice(promos).uci()

    # 5. Otherwise random.
    return rng.choice(moves).uci()


SELECTORS = {
    "uniform_legal": uniform_legal,
    "heuristic_legal": heuristic_legal,
}
