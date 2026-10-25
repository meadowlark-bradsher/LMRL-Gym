"""Stockfish-backed opponent for the Chess task, via python-chess's UCI engine.

This deliberately does NOT use the repo's ChessEnv (which imports JaxSeq and a
hard-coded Linux engine path). It subclasses the pure-python-chess SimpleChessEnv
and only swaps the opponent's move for a Stockfish reply, so the agent/reward
contract is identical and the only new dependency is a Stockfish binary on PATH
(or an explicit path).

Gate A (preflight): strength knobs are no longer set silently. ``configure``
raises on any engine error, the exact options sent are recorded, and
:func:`verify_engine_config` fails closed if the requested opponent mode cannot
actually be honoured by the binary (missing option, or an out-of-range value).

Note on strength: Stockfish's ``UCI_Elo`` has a floor of 1320, so a requested
1200 is clamped up to 1320 -- close to, but not exactly, the paper's SF-1200
opponent. ``effective_elo`` records what was actually set. ``UCI_Elo`` only
limits strength when ``UCI_LimitStrength`` is true, and it overrides Skill Level
when both are given, so the two ladders (Skill Level vs UCI_Elo) are kept
mutually exclusive here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import chess
import chess.engine

from llm_rl_scripts.chess.jev.simple_env import SimpleChessEnv

SF_MIN_ELO = 1320  # Stockfish UCI_Elo floor.


class EngineConfigError(RuntimeError):
    """Raised when the requested opponent mode cannot be honoured by the binary."""


@dataclass
class EngineConfig:
    """A verifiable record of exactly how the opponent engine was set up."""

    engine_path: str
    engine_id: Dict[str, str]              # name/author (version lives in name)
    options_sent: Dict[str, object]        # the exact configure() payload
    mode: str                              # "elo" | "skill"
    requested_elo: Optional[int] = None
    effective_elo: Optional[int] = None
    skill_level: Optional[int] = None
    search_limit: str = ""                 # e.g. "time=0.1", "depth=1"
    declared_ranges: Dict[str, str] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.engine_id.get("name", "unknown")


def verify_engine_config(cfg: EngineConfig, engine: chess.engine.SimpleEngine) -> None:
    """Fail closed if the engine cannot honour what we asked for.

    Checks that every option we rely on is actually declared by this binary, and
    that the value we sent is inside the engine's advertised range. This is the
    self-test the protocol requires before any live evaluation: a requested mode
    that the engine would silently ignore raises instead of being trusted.
    """
    opts = engine.options  # name -> chess.engine.Option metadata
    for name, value in cfg.options_sent.items():
        if name not in opts:
            raise EngineConfigError(
                "engine {!r} does not declare UCI option {!r}; requested mode "
                "cannot be honoured".format(cfg.version, name)
            )
        opt = opts[name]
        if opt.type in ("spin", "check") and isinstance(value, bool) is False:
            lo, hi = opt.min, opt.max
            if lo is not None and hi is not None and not (lo <= int(value) <= hi):
                raise EngineConfigError(
                    "option {!r}={} out of engine range [{}, {}]".format(
                        name, value, lo, hi)
                )
        cfg.declared_ranges[name] = "{}[{},{}]".format(opt.type, opt.min, opt.max)


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
        verify: bool = True,
    ):
        super().__init__(from_position=from_position, seed=seed)
        path = engine_path or "stockfish"
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.skill_level = skill_level
        self.depth = depth
        self.nodes = nodes
        self.move_time = move_time

        # Weakening ladder (mutually exclusive):
        #  * skill_level (0-20): Stockfish's own handicap, reaches below the
        #    UCI_Elo floor of 1320.
        #  * otherwise UCI_Elo (clamped to >= 1320), with UCI_LimitStrength on.
        # Search is additionally capped by depth/nodes/time.
        if skill_level is not None:
            options: Dict[str, object] = {"Skill Level": int(skill_level)}
            mode, self.effective_elo = "skill", None
            self.label = "skill={}".format(skill_level)
        else:
            self.requested_elo = engine_elo
            self.effective_elo = max(engine_elo, SF_MIN_ELO)
            options = {"UCI_LimitStrength": True, "UCI_Elo": self.effective_elo}
            mode = "elo"
            self.label = "elo={}".format(self.effective_elo)

        self.config = EngineConfig(
            engine_path=path,
            engine_id=dict(self.engine.id),
            options_sent=options,
            mode=mode,
            requested_elo=engine_elo if mode == "elo" else None,
            effective_elo=self.effective_elo,
            skill_level=skill_level,
            search_limit=self._limit_desc(),
        )
        if verify:
            verify_engine_config(self.config, self.engine)
        # No silent `pass`: a configure error now propagates.
        self.engine.configure(options)

    def _limit_desc(self) -> str:
        if self.depth is not None:
            return "depth={}".format(self.depth)
        if self.nodes is not None:
            return "nodes={}".format(self.nodes)
        return "time={}".format(self.move_time)

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
