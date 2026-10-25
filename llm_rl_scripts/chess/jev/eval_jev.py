"""Evaluate Jev as a zero-shot policy on the Chess task.

Jev plays White and picks each move from the position's legal SAN moves via the
Choice primitive -- no RL training. The headline metric is the illegal-move
rate: the paper's trained models play 22-35% illegal moves in full games and
GPT-4 effectively 0 score from illegal play, whereas Jev's constrained Choice
makes an illegal move impossible by construction. The open question is move
quality (win rate against the opponent).

Opponents:
  * random    -- a uniform legal-move opponent (pure python-chess, no engine).
  * stockfish -- the repo's ChessEnv + Stockfish engine (needs a working binary;
                 the bundled one is Linux-only, so on macOS install one and pass
                 --engine-path or set CHESS_ENGINE_PATH).

Examples:
    # Offline logic check (LLM_RL env for python-chess; no key, no engine):
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.chess.jev.eval_jev --self-test --episodes 3

    # Live, random opponent (needs TYPESAFE_API_KEY):
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.chess.jev.eval_jev --episodes 10 --max-moves 60

    # Endgame position (e.g. K+Q vs K); pass a FEN:
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.chess.jev.eval_jev --from-position "<FEN>" --episodes 10
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Callable, Dict, List, Optional

import chess

from LLM_RL.jev.client import HttpJevBackend, JevClient, JevError, MockJevBackend
from llm_rl_scripts.chess.jev import outcomes
from llm_rl_scripts.chess.jev.policy import JevChessPolicy, legal_san_moves
from llm_rl_scripts.chess.jev.simple_env import SimpleChessEnv


def san_history(start_fen: Optional[str], move_stack: List[chess.Move]) -> str:
    """Reconstruct the SAN move list by replaying the stack from the start."""
    tmp = chess.Board() if start_fen is None else chess.Board(start_fen)
    out = []
    for i, mv in enumerate(move_stack):
        prefix = "{}.".format(i // 2 + 1) if i % 2 == 0 else ""
        out.append(prefix + tmp.san(mv))
        tmp.push(mv)
    return " ".join(out)


def run_game(env, policy: JevChessPolicy, max_moves: int, verbose: bool = False) -> Dict[str, float]:
    """Play one full game and return a fully-accounted result.

    Gate D (fail closed): the move Jev returns is asserted to be one of the
    legal SANs before it is applied; a provider failure or a malformed response
    is recorded as ``api_error`` (a truncated, non-draw outcome) rather than
    being silently mapped to a legal move or to a draw.
    Gate B/C: termination is classified explicitly via :mod:`outcomes`, so a hit
    move-cap is reported as truncated, never as a chess draw.
    """
    env.reset()
    illegal = 0
    moves = 0
    move_cap_hit = False
    term = None
    for ply in range(max_moves):
        board = env.board
        if board.is_game_over():
            break
        legal = set(legal_san_moves(board))
        history = san_history(env.from_position, list(board.move_stack))
        try:
            san = policy.choose(board, history)
        except JevError as e:
            term = outcomes.classify(board, api_error="JevError: {}".format(e))
            if verbose:
                print("  !! provider error at ply {}: {}".format(moves + 1, e))
            break
        moves += 1
        # Fail closed: Jev must return a member of the presented legal menu.
        if san not in legal:
            illegal += 1
            term = outcomes.classify(board, illegal=True)
            if verbose:
                print("  !! non-legal choice {!r} (not in {} legal SANs)".format(
                    san, len(legal)))
            break
        if verbose:
            conf = policy.last_answer.confidence if policy.last_answer else None
            print("  {:2d}. {} (conf={}, {} legal)".format(moves, san, conf, len(legal)))
        _, _, done = env.step(san)
        if done:
            break
        if ply == max_moves - 1:
            move_cap_hit = True

    if term is None:
        term = outcomes.classify(env.board, move_cap_hit=move_cap_hit)
    return {
        "reward": term.reward,
        "win": float(term.result == "win"),
        "loss": float(term.result == "loss"),
        "draw": float(term.result == "draw"),
        "truncated": float(term.is_truncated),
        "moves": float(moves),
        "illegal": float(illegal),
        "result": term.result or "truncated",
        "termination": term.reason.value,
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    n = len(results)
    if not n:
        return {}
    total_moves = sum(r["moves"] for r in results) or 1.0
    keys = ["reward", "win", "loss", "draw", "truncated", "moves", "illegal"]
    agg = {k: sum(r[k] for r in results) / n for k in keys}
    agg["illegal_pct"] = 100.0 * sum(r["illegal"] for r in results) / total_moves
    # Count each distinct termination reason for the report.
    agg["_terminations"] = {}
    for r in results:
        key = r.get("termination", "?")
        agg["_terminations"][key] = agg["_terminations"].get(key, 0) + 1
    return agg


def build_env(opponent, from_position, seed, engine_path, engine_elo, skill, depth, nodes):
    if opponent == "stockfish":
        # Lazy import so the default path needs no engine/stockfish package.
        from llm_rl_scripts.chess.jev.stockfish_env import StockfishChessEnv

        return StockfishChessEnv(
            from_position=from_position, engine_path=engine_path, engine_elo=engine_elo,
            skill_level=skill, depth=depth, nodes=nodes, seed=seed,
        )
    return SimpleChessEnv(from_position=from_position, seed=seed)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-moves", type=int, default=60, help="agent plies before declaring a draw")
    p.add_argument("--opponent", choices=["random", "stockfish"], default="random")
    p.add_argument("--from-position", default=None, help="starting FEN (e.g. an endgame)")
    p.add_argument("--engine-path", default=None, help="path to a Stockfish binary")
    p.add_argument("--engine-elo", type=int, default=1200)
    p.add_argument("--engine-skill", type=int, default=None, help="Stockfish Skill Level 0-20 (weakening knob below the 1320 UCI_Elo floor)")
    p.add_argument("--engine-depth", type=int, default=None, help="cap opponent search to this depth (very weak at 1-2)")
    p.add_argument("--engine-nodes", type=int, default=None, help="cap opponent search to this many nodes")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--backend", choices=["http", "mock"], default="http")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--self-test", action="store_true", help="offline: mock Jev + random opponent, no key")
    args = p.parse_args()

    use_mock = args.backend == "mock" or args.self_test
    opponent = "random" if args.self_test else args.opponent
    verbose = not args.quiet

    rng = random.Random(args.seed)
    mock_chooser = (lambda state, qid, options: rng.choice(options)) if use_mock else None
    backend = MockJevBackend(mock_chooser) if use_mock else HttpJevBackend()
    policy = JevChessPolicy(JevClient(backend=backend, model=args.model))

    mode = ["mock-jev" if use_mock else "jev:{}".format(args.model), "opp={}".format(opponent)]
    if args.from_position:
        mode.append("from_fen")
    print("Chess x Jev  [{}]  episodes={} max_moves={}".format(
        " ".join(mode), args.episodes, args.max_moves))

    results = []
    for i in range(args.episodes):
        env = build_env(opponent, args.from_position, args.seed + i, args.engine_path,
                        args.engine_elo, args.engine_skill, args.engine_depth, args.engine_nodes)
        if verbose:
            print("game {}".format(i + 1))
        try:
            res = run_game(env, policy, args.max_moves, verbose=verbose)
        finally:
            env.close()
        if verbose:
            print("  -> {} (reward={}, moves={}, illegal={})".format(
                res["result"], res["reward"], int(res["moves"]), int(res["illegal"])))
        results.append(res)

    agg = summarize(results)
    # Score (W+D/2) is well-defined only over *finished* games; truncated games
    # are excluded from it rather than silently counted as draws.
    finished = [r for r in results if r.get("result") not in (None, "truncated")]
    nf = len(finished) or 1
    wins = sum(r["win"] for r in finished)
    draws = sum(r["draw"] for r in finished)
    score = (wins + 0.5 * draws) / nf
    print("\n=== summary over {} games ({} finished, {} truncated) ===".format(
        len(results), len(finished), int(sum(r["truncated"] for r in results))))
    print("win rate     : {:.3f}  (of finished)".format(wins / nf))
    print("draw rate    : {:.3f}  (of finished)".format(draws / nf))
    print("loss rate    : {:.3f}  (of finished)".format(sum(r['loss'] for r in finished) / nf))
    print("score (W+D/2): {:.3f}  (finished games only)".format(score))
    print("avg reward   : {:.3f}".format(agg.get("reward", 0.0)))
    print("avg moves    : {:.2f}".format(agg.get("moves", 0.0)))
    print("illegal moves: {:.3f}%  (0 by construction: Jev only picks legal SAN)".format(
        agg.get("illegal_pct", 0.0)))
    print("terminations :")
    for reason, count in sorted(agg.get("_terminations", {}).items(), key=lambda kv: -kv[1]):
        print("    {:<24s} {}".format(reason, count))


if __name__ == "__main__":
    main()
