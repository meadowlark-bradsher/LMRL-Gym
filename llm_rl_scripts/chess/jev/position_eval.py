"""P1 primary experiment: fixed-position move quality.

Holds the board state and the legal-action menu constant and asks: once legality
is guaranteed, does Jev choose *better* moves than a random or a cheap heuristic
legal-action selector? Every arm sees the identical FEN, side to move and sorted
legal SAN menu (order randomised per position with a recorded seed); the move it
returns is scored by the cached Stockfish evaluator (centipawn loss vs the best
legal move, best/top-3 match, missed forced mate, materially-losing blunder).

Arms:
  * uniform_legal    -- random legal move (no model, no engine)
  * heuristic_legal  -- cheap fixed rules (no model, no engine)
  * jev              -- Jev Choice over the legal SAN menu (one API call/position)

Results append to a JSONL (``position_scores.jsonl``) and the run is resumable:
an (arm, position_id) already present is skipped, so a crash or an interrupt
never re-spends API credits. Jev is never shown the evaluator's scores.

Blunder thresholds (pre-registered):
  * materially-losing move: centipawn loss >= 300
  * missed forced mate: best move is a forced mate, chosen move is not

Usage:
    # Free baselines (no key):
    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.chess.jev.position_eval \
      --positions positions.jsonl --out position_scores.jsonl \
      --arms uniform_legal,heuristic_legal --engine-path /opt/homebrew/bin/stockfish

    # Live Jev arm (needs TYPESAFE_API_KEY); add --limit N for a smoke test:
    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.chess.jev.position_eval \
      --positions positions.jsonl --out position_scores.jsonl \
      --arms jev --engine-path /opt/homebrew/bin/stockfish
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Set, Tuple


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's str hash is randomised per run)."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

import chess

from LLM_RL.jev.client import HttpJevBackend, JevClient, JevError, MockJevBackend
from llm_rl_scripts.chess.jev import llm_arms
from llm_rl_scripts.chess.jev.evaluator import MoveEvaluator
from llm_rl_scripts.chess.jev.selectors import SELECTORS


def resolve_uci(board: chess.Board, san: Optional[str]) -> Optional[str]:
    """SAN -> UCI via python-chess (tolerant of spelling); None if illegal."""
    if not san:
        return None
    try:
        return board.parse_san(san).uci()
    except (ValueError, AssertionError):
        return None

BLUNDER_CP = 300

JEV_INSTRUCTIONS = (
    "You are playing the {side} pieces in a game of chess. Given the current "
    "position and the legal moves available, choose the single strongest move "
    "for {side}."
)


def load_positions(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done_keys(out_path: str) -> Set[Tuple[str, str]]:
    """(arm, position_id) pairs already recorded, for resumability."""
    seen: Set[Tuple[str, str]] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    seen.add((r["arm"], r["position_id"]))
                except (ValueError, KeyError):
                    continue
    return seen


def jev_state(board: chess.Board, side: str) -> str:
    return "\n\n".join([
        "You are {}. Current position ({} to move):".format(side, side),
        "Board:\n" + str(board),
        "FEN: " + board.fen(),
    ])


def choose_jev(client: JevClient, board: chess.Board, sans: List[str], side: str):
    """Return (selected_san, confidence, exception_type)."""
    options = {san: None for san in sans}
    side_cap = side.capitalize()
    try:
        answer = client.choice(
            state=jev_state(board, side_cap),
            instructions=JEV_INSTRUCTIONS.format(side=side_cap),
            options=options,
            question_id="move",
        )
    except JevError as e:
        return None, None, "JevError: {}".format(e)
    return answer.choice, answer.confidence, None


def score_record(run_id, arm, pos, board, selected_uci, selected_san,
                 presented_order, pe, confidence, exception_type) -> dict:
    is_legal = selected_uci is not None and selected_uci in pe.moves
    cp_loss = pe.cp_loss(selected_uci) if is_legal else None
    rank = pe.rank_of(selected_uci) if is_legal else None
    return {
        "run_id": run_id,
        "arm": arm,
        "position_id": pos["position_id"],
        "fen": pos["fen"],
        "phase": pos["phase"],
        "tactic_type": pos.get("tactic_type"),
        "side_to_move": pos["side_to_move"],
        "source_game": pos.get("source_game"),
        "best_uci": pe.best_uci,
        "best_cp": pe.best_cp,
        "best_is_mate": pe.best_is_mate,
        "presented_order": presented_order,
        "selected_san": selected_san,
        "selected_uci": selected_uci,
        "cp_loss": cp_loss,
        "rank": rank,
        "match_best": bool(rank == 0) if rank is not None else None,
        "match_top3": bool(rank is not None and rank < 3),
        "lost_forced_mate": pe.lost_forced_mate(selected_uci) if is_legal else None,
        "blunder": bool(cp_loss is not None and cp_loss >= BLUNDER_CP),
        "is_legal": is_legal,
        "confidence": confidence,
        "exception_type": exception_type,
        "evaluator_engine": pe.engine_id,
        "evaluator_budget": pe.budget,
        "ts": time.time(),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="uniform_legal,heuristic_legal",
                   help="comma list of uniform_legal,heuristic_legal,jev")
    p.add_argument("--engine-path", default=None)
    p.add_argument("--eval-depth", type=int, default=16)
    p.add_argument("--eval-cache", default=None, help="JSON cache for evaluator scores")
    p.add_argument("--limit", type=int, default=None, help="only the first N positions (smoke test)")
    p.add_argument("--seed", type=int, default=0, help="option-order + baseline RNG seed")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--backend", choices=["http", "mock"], default="http")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    engine_path = args.engine_path or shutil.which("stockfish")
    if not engine_path:
        raise SystemExit("no Stockfish binary; pass --engine-path")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_id = args.run_id or "run_{}".format(int(time.time()))
    cache = args.eval_cache or (args.out + ".evalcache.json")

    positions = load_positions(args.positions)
    if args.limit:
        positions = positions[: args.limit]
    seen = done_keys(args.out)

    evaluator = MoveEvaluator(engine_path, depth=args.eval_depth, cache_path=cache)
    jev_client = None
    if "jev" in arms:
        backend = MockJevBackend() if args.backend == "mock" else HttpJevBackend()
        jev_client = JevClient(backend=backend, model=args.model)

    print("position_eval [{}]  run_id={}  positions={}  evaluator={} {}".format(
        ",".join(arms), run_id, len(positions), evaluator.engine_id, evaluator.budget))

    out_f = open(args.out, "a")
    try:
        for pi, pos in enumerate(positions):
            board = chess.Board(pos["fen"])
            side = pos["side_to_move"]
            sans = [board.san(m) for m in board.legal_moves]
            # Randomise the presented order, seeded per position for reproducibility.
            rng = random.Random(stable_seed(args.seed, pos["position_id"]))
            presented = sans[:]
            rng.shuffle(presented)

            pe = evaluator.evaluate(pos["fen"])

            for arm in arms:
                if (arm, pos["position_id"]) in seen:
                    continue
                confidence, exc = None, None
                if arm == "jev":
                    selected_san, confidence, exc = choose_jev(jev_client, board, presented, side)
                elif arm in llm_arms.ARM_SPECS:
                    selected_san, exc = llm_arms.choose(arm, board, side, presented)
                else:
                    sel_rng = random.Random(stable_seed(args.seed, arm, pos["position_id"]))
                    selected_san = board.san(chess.Move.from_uci(SELECTORS[arm](board, sel_rng)))
                selected_uci = resolve_uci(board, selected_san)
                rec = score_record(run_id, arm, pos, board, selected_uci, selected_san,
                                   presented, pe, confidence, exc)
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                seen.add((arm, pos["position_id"]))
            if (pi + 1) % 10 == 0:
                evaluator.flush()
                print("  {}/{} positions".format(pi + 1, len(positions)))
    finally:
        out_f.close()
        evaluator.close()

    print("done -> {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
