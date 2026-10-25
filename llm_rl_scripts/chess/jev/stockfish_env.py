"""Stockfish-backed opponent for the Chess task, via python-chess's UCI engine.

This deliberately does NOT use the repo's ChessEnv (which imports JaxSeq and a
hard-coded Linux engine path). It subclasses the pure-python-chess SimpleChessEnv
and only swaps the opponent's move for a Stockfish reply, so the agent/reward
contract is identical and the only new dependency is a Stockfish binary on PATH
(or an explicit path).

Note on strength: Stockfish's ``UCI_Elo`` has a floor of 1320, so a requested
1200 is clamped up to 1320 — close to, but not exactly, the paper's SF-1200
opponent. ``effective_elo`` records what was actually set.
"""

from __future__ import annotations

from typing import Optional

import chess
import chess.engine

from llm_rl_scripts.chess.jev.simple_env import SimpleChessEnv

SF_MIN_ELO = 1320  # Stockfish UCI_Elo floor.


class StockfishChessEnv(SimpleChessEnv):
    def __init__(
        self,
        from_position: Optional[str] = None,
        engine_path: Optional[str] = None,
        engine_elo: int = 1200,
        skill_level: Optional[int] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        move_time: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(from_position=from_position, seed=seed)
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path or "stockfish")
        self.skill_level = skill_level
        self.depth = depth
        self.nodes = nodes
        self.move_time = move_time
        # Weakening ladder, in order of use:
        #  * skill_level (0-20): Stockfish's own handicap knob, reaches well
        #    below the UCI_Elo floor of 1320.
        #  * otherwise UCI_Elo (clamped to >= 1320).
        # Search is additionally capped by depth/nodes/time to weaken further.
        if skill_level is not None:
            self.label = "skill={}".format(skill_level)
            try:
                self.engine.configure({"Skill Level": int(skill_level)})
            except chess.engine.EngineError:
                pass
            self.effective_elo = None
        else:
            self.requested_elo = engine_elo
            self.effective_elo = max(engine_elo, SF_MIN_ELO)
            self.label = "elo={}".format(self.effective_elo)
            try:
                self.engine.configure(
                    {"UCI_LimitStrength": True, "UCI_Elo": self.effective_elo}
                )
            except chess.engine.EngineError:
                pass

    def _limit(self) -> chess.engine.Limit:
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        if self.nodes is not None:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(time=self.move_time)

    def _opponent_move(self):
        result = self.engine.play(self.board, self._limit())
        self.board.push(result.move)

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass
