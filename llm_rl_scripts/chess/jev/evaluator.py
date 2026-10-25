"""P1 evaluator: a cached Stockfish oracle that scores every legal move.

For each FEN, one ``MultiPV = (number of legal moves)`` analysis call scores all
legal moves at once (the engine reuses its search tree, so this is far cheaper
than analysing each move separately). From that we derive, for any candidate
move, its **centipawn loss** versus the best legal move, whether it matches the
best / top-3 moves, and whether it throws away a forced mate.

Scores are always from the side-to-move's perspective, so the evaluator is
colour-agnostic. Mate scores are handled separately from ordinary centipawns:
they are mapped to a large magnitude only for ordering, and the mate distance is
kept so "missed a forced mate" can be reported without pretending a mate is
worth some number of pawns.

The evaluator is independent of the 0.1 s/move *opponent*: fix its engine
version and a deterministic depth/node budget, and cache every result keyed by
(fen, engine id, budget) so a rerun is free and reproducible. Jev is never shown
these scores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess
import chess.engine

# Ordering-only magnitude for a mate score; mate distance is kept separately.
MATE_CP = 100000


@dataclass
class MoveScore:
    uci: str
    san: str
    cp: int                 # mover-POV, mate-mapped for ordering
    is_mate: bool
    mate_in: Optional[int]  # +N = mate for mover in N, -N = mated in N


@dataclass
class PositionEval:
    fen: str
    best_uci: str
    best_cp: int
    best_is_mate: bool
    moves: Dict[str, MoveScore] = field(default_factory=dict)  # uci -> MoveScore
    engine_id: str = ""
    budget: str = ""

    def rank_of(self, uci: str) -> Optional[int]:
        order = sorted(self.moves.values(), key=lambda m: -m.cp)
        for i, m in enumerate(order):
            if m.uci == uci:
                return i  # 0 = best
        return None

    def cp_loss(self, uci: str) -> Optional[int]:
        m = self.moves.get(uci)
        if m is None:
            return None
        return max(0, self.best_cp - m.cp)

    def lost_forced_mate(self, uci: str) -> bool:
        m = self.moves.get(uci)
        return bool(self.best_is_mate and m is not None and not (m.is_mate and (m.mate_in or 0) > 0))


def _to_cp(score: chess.engine.PovScore, pov: chess.Color):
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate() or 0
        cp = MATE_CP - abs(m) if m > 0 else -MATE_CP + abs(m)
        return cp, True, m
    return s.score(mate_score=MATE_CP), False, None


class MoveEvaluator:
    def __init__(
        self,
        engine_path: str,
        depth: Optional[int] = 16,
        nodes: Optional[int] = None,
        move_time: Optional[float] = None,
        cache_path: Optional[str] = None,
        multipv_cap: int = 256,
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine_id = self.engine.id.get("name", "unknown")
        self.depth = depth
        self.nodes = nodes
        self.move_time = move_time
        self.multipv_cap = multipv_cap
        if move_time is not None:
            self.budget = "time={}".format(move_time)
        elif nodes is not None:
            self.budget = "nodes={}".format(nodes)
        else:
            self.budget = "depth={}".format(depth)
        self.cache_path = cache_path
        self._cache: Dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def _limit(self) -> chess.engine.Limit:
        if self.move_time is not None:
            return chess.engine.Limit(time=self.move_time)
        if self.nodes is not None:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(depth=self.depth)

    def _key(self, fen: str) -> str:
        return "{}|{}|{}".format(fen, self.engine_id, self.budget)

    def evaluate(self, fen: str) -> PositionEval:
        key = self._key(fen)
        if key in self._cache:
            return _eval_from_dict(self._cache[key])

        board = chess.Board(fen)
        pov = board.turn
        legal = list(board.legal_moves)
        n = len(legal)
        infos = self.engine.analyse(board, self._limit(), multipv=min(n, self.multipv_cap))

        moves: Dict[str, MoveScore] = {}
        best_uci, best_cp, best_is_mate = None, None, False
        for rank, info in enumerate(infos):
            pv = info.get("pv")
            if not pv:
                continue
            mv = pv[0]
            cp, is_mate, mate_in = _to_cp(info["score"], pov)
            ms = MoveScore(uci=mv.uci(), san=board.san(mv), cp=cp, is_mate=is_mate, mate_in=mate_in)
            moves[ms.uci] = ms
            if rank == 0:
                best_uci, best_cp, best_is_mate = ms.uci, cp, is_mate

        # MultiPV capped below the legal count leaves some moves unscored; score
        # the rest by pushing the move and negating the reply's eval.
        if n > self.multipv_cap:
            for mv in legal:
                if mv.uci() in moves:
                    continue
                board.push(mv)
                info = self.engine.analyse(board, self._limit())
                cp, is_mate, mate_in = _to_cp(info["score"], pov)  # already mover-POV via pov
                board.pop()
                moves[mv.uci()] = MoveScore(uci=mv.uci(), san=chess.Board(fen).san(mv),
                                            cp=cp, is_mate=is_mate, mate_in=mate_in)

        pe = PositionEval(fen=fen, best_uci=best_uci or legal[0].uci(),
                          best_cp=best_cp if best_cp is not None else 0,
                          best_is_mate=best_is_mate, moves=moves,
                          engine_id=self.engine_id, budget=self.budget)
        self._cache[key] = _eval_to_dict(pe)
        return pe

    def flush(self) -> None:
        if self.cache_path:
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._cache, f)
            os.replace(tmp, self.cache_path)

    def close(self) -> None:
        self.flush()
        try:
            self.engine.quit()
        except Exception:
            pass


def _eval_to_dict(pe: PositionEval) -> dict:
    return {
        "fen": pe.fen, "best_uci": pe.best_uci, "best_cp": pe.best_cp,
        "best_is_mate": pe.best_is_mate, "engine_id": pe.engine_id, "budget": pe.budget,
        "moves": {u: vars(m) for u, m in pe.moves.items()},
    }


def _eval_from_dict(d: dict) -> PositionEval:
    moves = {u: MoveScore(**m) for u, m in d["moves"].items()}
    return PositionEval(fen=d["fen"], best_uci=d["best_uci"], best_cp=d["best_cp"],
                        best_is_mate=d["best_is_mate"], moves=moves,
                        engine_id=d.get("engine_id", ""), budget=d.get("budget", ""))
