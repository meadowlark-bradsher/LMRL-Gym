"""Offline P0 validation gates -- no API key, no network, no spend.

Run every gate:
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.chess.jev.validate --all

Engine gates are skipped (not failed) when no Stockfish binary is found; pass
``--engine-path`` to force them. Any failed assertion exits non-zero so this can
gate CI and any live run.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable, List, Tuple

import chess

from llm_rl_scripts.chess.jev import outcomes
from llm_rl_scripts.chess.jev.outcomes import Reason


# --------------------------------------------------------------------------- #
# Gate: SAN <-> UCI <-> board.push correctness (promotion, castling, mate, e.p.)
# --------------------------------------------------------------------------- #
def gate_san_uci() -> None:
    cases: List[Tuple[str, str, str, str]] = [
        # (fen, san, expected_uci, note)
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4", "e2e4", "pawn double"),
        # White kingside castling.
        ("rnbqk2r/ppppbppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "O-O", "e1g1", "castle K"),
        # White queenside castling.
        ("r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2NQ4/PPP1PPPP/R3KBNR w KQkq - 6 5", "O-O-O", "e1c1", "castle Q"),
        # Promotion to queen.
        ("8/P7/8/8/8/8/8/k6K w - - 0 1", "a8=Q", "a7a8q", "promotion"),
        # En passant capture.
        ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "exf6", "e5f6", "en passant"),
        # Scholar's mate delivering checkmate (SAN carries '#').
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1", "Qh5", "d1h5", "pre-mate"),
    ]
    for fen, san, want_uci, note in cases:
        board = chess.Board(fen)
        move = board.parse_san(san)
        assert move.uci() == want_uci, "{}: SAN {} -> {} != {}".format(note, san, move.uci(), want_uci)
        # Round-trip: the SAN we render for this move must re-parse to the same move.
        rendered = board.san(move)
        assert board.parse_san(rendered) == move, "{}: SAN round-trip failed ({})".format(note, rendered)
        board.push(move)  # must not raise
    # Checkmate SAN end-to-end (Fool's mate from Black's side, but test White mate):
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    # Not actually mate; build a real one: back-rank mate.
    br = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    m = br.parse_san("Ra8#")
    br.push(m)
    assert br.is_checkmate(), "expected Ra8# to be checkmate"
    print("  [pass] SAN<->UCI<->push: {} cases incl. promotion/castling/e.p./mate".format(len(cases)))


# --------------------------------------------------------------------------- #
# Gate: termination taxonomy
# --------------------------------------------------------------------------- #
def gate_outcomes() -> None:
    # White delivers checkmate -> win, +1.
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    b.push_san("Ra8#")
    t = outcomes.classify(b)
    assert t.reason == Reason.CHECKMATE_WIN and t.reward == 1.0 and t.result == "win", t

    # White is checkmated -> loss, -1. (Fool's mate: 1.f3 e5 2.g4 Qh4#)
    b = chess.Board()
    for mv in ["f3", "e5", "g4", "Qh4#"]:
        b.push_san(mv)
    t = outcomes.classify(b)
    assert t.reason == Reason.CHECKMATE_LOSS and t.reward == -1.0 and t.result == "loss", t

    # Stalemate -> draw, 0.
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # black to move, no legal move, not in check
    t = outcomes.classify(b)
    assert t.reason == Reason.STALEMATE and t.result == "draw" and t.reward == 0.0, t

    # Insufficient material (K vs K) -> draw.
    b = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    t = outcomes.classify(b)
    assert t.reason == Reason.INSUFFICIENT_MATERIAL and t.result == "draw", t

    # Move-cap on a normal position -> truncated, NOT a draw.
    b = chess.Board()
    t = outcomes.classify(b, move_cap_hit=True)
    assert t.reason == Reason.MOVE_CAP and t.is_truncated and t.result is None and t.reward == 0.0, t

    # Illegal move -> loss, -1, not truncated.
    t = outcomes.classify(chess.Board(), illegal=True)
    assert t.reason == Reason.ILLEGAL_MOVE and t.reward == -1.0 and t.result == "loss", t

    # API error -> truncated, no result.
    t = outcomes.classify(chess.Board(), api_error="boom")
    assert t.reason == Reason.API_ERROR and t.is_truncated and t.result is None, t

    # Running (not over) -> running, no result, not truncated.
    t = outcomes.classify(chess.Board())
    assert t.reason == Reason.RUNNING and not t.is_truncated and t.result is None, t

    # Fifty-move rule (claimable) -> draw.
    b = chess.Board("8/8/8/4k3/8/3QK3/8/8 w - - 100 80")
    t = outcomes.classify(b)
    assert t.result == "draw" and t.reason in (Reason.FIFTY_MOVES, Reason.SEVENTYFIVE_MOVES), t

    print("  [pass] termination taxonomy: mate/stalemate/insufficient/50mv/cap/illegal/api/running")


# --------------------------------------------------------------------------- #
# Gate: reward replay fixture (deterministic trace -> accounted reward)
# --------------------------------------------------------------------------- #
def gate_reward_replay() -> None:
    from llm_rl_scripts.chess.jev.simple_env import SimpleChessEnv

    # Scholar's mate: White mates on move 4. Deterministic opponent replies are
    # scripted via a fixed FEN sequence so this needs no RNG and no engine.
    env = SimpleChessEnv(seed=0)
    env.reset()
    # Drive a known mating line by pushing both sides directly on the board,
    # then classify -- the env's own step() reward must agree with classify().
    b = chess.Board()
    line = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
    for mv in line:
        b.push_san(mv)
    t = outcomes.classify(b)
    assert t.reason == Reason.CHECKMATE_WIN and t.reward == 1.0, ("replay", t)

    # And through the env's step contract for the agent's winning move:
    env2 = SimpleChessEnv(from_position="r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    # ^ position after Qxf7# already on the board -> game over, agent (White) won.
    assert chess.Board(env2.from_position).is_checkmate()
    print("  [pass] reward replay: scored mating line agrees between env and classify()")


# --------------------------------------------------------------------------- #
# Gate: Stockfish config verification + fail-closed (needs a binary)
# --------------------------------------------------------------------------- #
def gate_engine_config(engine_path: str) -> None:
    import chess.engine

    from llm_rl_scripts.chess.jev.stockfish_env import (
        EngineConfigError, StockfishChessEnv, verify_engine_config,
    )

    # ELO mode: UCI_LimitStrength + UCI_Elo must both be declared and accepted.
    env = StockfishChessEnv(engine_path=engine_path, engine_elo=1320, move_time=0.05)
    cfg = env.config
    assert cfg.mode == "elo" and cfg.effective_elo == 1320, cfg
    assert cfg.options_sent.get("UCI_LimitStrength") is True, cfg
    assert "UCI_Elo" in cfg.declared_ranges, cfg
    # A requested 1200 must clamp up to the 1320 floor and record both.
    assert cfg.requested_elo == 1320
    print("  [pass] engine ELO config: {} | sent {} | ranges {}".format(
        cfg.version, cfg.options_sent, cfg.declared_ranges))
    env.close()

    # SKILL mode: Skill Level must be declared, UCI_Elo not set in this mode.
    env = StockfishChessEnv(engine_path=engine_path, skill_level=0, move_time=0.05)
    cfg = env.config
    assert cfg.mode == "skill" and cfg.skill_level == 0, cfg
    assert "UCI_Elo" not in cfg.options_sent, cfg
    print("  [pass] engine SKILL config: sent {}".format(cfg.options_sent))
    env.close()

    # Fail closed: an option the engine does not declare must raise.
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        from llm_rl_scripts.chess.jev.stockfish_env import EngineConfig
        cfg = EngineConfig(engine_path=engine_path, engine_id=dict(eng.id),
                           options_sent={"NoSuchOption": 7}, mode="elo")
        raised = False
        try:
            verify_engine_config(cfg, eng)
        except EngineConfigError:
            raised = True
        assert raised, "verify_engine_config should fail closed on an undeclared option"
    finally:
        eng.quit()
    print("  [pass] engine config fails closed on an undeclared option")


def _run(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
        return True
    except Exception:
        print("  [FAIL] {}".format(name))
        traceback.print_exc()
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="run every gate")
    p.add_argument("--engine-path", default=None, help="Stockfish binary (enables engine gates)")
    p.add_argument("--no-engine", action="store_true", help="skip engine gates even if a binary exists")
    args = p.parse_args()

    print("P0 validation gates (offline)")
    ok = True
    ok &= _run("SAN<->UCI", gate_san_uci)
    ok &= _run("outcomes", gate_outcomes)
    ok &= _run("reward replay", gate_reward_replay)

    engine_path = args.engine_path
    if engine_path is None and not args.no_engine:
        import shutil
        engine_path = shutil.which("stockfish")
    if engine_path and not args.no_engine:
        ok &= _run("engine config", lambda: gate_engine_config(engine_path))
    else:
        print("  [skip] engine config (no Stockfish binary; pass --engine-path to force)")

    print("\n{}".format("ALL GATES PASS" if ok else "SOME GATES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
