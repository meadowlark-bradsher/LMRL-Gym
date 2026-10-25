"""Adapter exposing the repo's Stockfish-backed ChessEnv with the same surface
as SimpleChessEnv (``.board``, ``reset()``, ``step(san) -> (fen, reward, done)``).

Imported lazily by the runner only when ``--opponent stockfish`` is used, so the
default random-opponent path needs neither the stockfish package nor a binary.
The bundled engine is a Linux build; on macOS install one (e.g. ``brew install
stockfish``) and pass its path via ``engine_path`` or the CHESS_ENGINE_PATH env.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


class StockfishChessEnv:
    def __init__(self, from_position: Optional[str] = None, engine_path: Optional[str] = None,
                 engine_elo: int = 1200):
        if engine_path:
            os.environ["CHESS_ENGINE_PATH"] = engine_path
        # Imported here so the dependency is only required on this path.
        from llm_rl_scripts.chess.env.env import ChessEnv

        self.from_position = from_position
        self.env = ChessEnv(fen=True, from_position=from_position, stockfish_elo=engine_elo)

    @property
    def board(self):
        return self.env.board

    def reset(self) -> str:
        fen, _ = self.env.reset()
        return fen

    def step(self, action_san: str) -> Tuple[str, float, bool]:
        fen, reward, done, _ = self.env.step(action_san)
        return fen, float(reward), bool(done)
