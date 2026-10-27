"""Shared helpers for the Noul-vs-LLM confidence head-to-head.

The grader is the FROZEN P1 eval cache (`results/primary_20261025/eval_cache.json`):
per-move centipawn scores at depth 16, read-only. `cp_lost(uci) = best_cp - cp`
(mover POV). Events (pre-registered): acceptable = cp_lost <= 50; best = cp_lost == 0.

The legal menu is presented identically to every model as an enumerated label list,
shuffled per (position, model) with a fixed seed and logged. Labels A..Z are single
tokens (clean for gpt-4o logprobs); moves past 26 get two-letter labels (floored in the
logprob readout, reported as coverage).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = os.path.normpath(os.path.join(HERE, "..", "results", "primary_20261025"))
POSITIONS = os.path.join(PRIMARY, "positions.jsonl")
GRADER = os.path.join(PRIMARY, "eval_cache.json")

ACCEPTABLE_CP = 50          # primary event threshold
APPENDIX_THRESHOLDS = (25, 100)

# Billed price per 1M tokens (input, output). Jev has no public price -> latency only.
PRICES = {
    "gpt-4o": (2.50, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def stable_seed(*parts) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def label_for(i: int) -> str:
    """A..Z, then AA, AB, ... (only A..Z are single tokens for logprobs)."""
    if i < 26:
        return chr(65 + i)
    return chr(65 + i // 26 - 1) + chr(65 + i % 26)


def load_positions() -> List[dict]:
    with open(POSITIONS) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_grader() -> Dict[str, dict]:
    """fen -> {best_cp, best_is_mate, moves:{uci:{cp,...}}}, indexed by FEN."""
    with open(GRADER) as f:
        cache = json.load(f)
    by_fen = {}
    for entry in cache.values():
        by_fen[entry["fen"]] = entry
    return by_fen


def grader_hash() -> str:
    with open(GRADER, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def cp_lost(entry: dict, uci: str) -> Optional[int]:
    m = entry["moves"].get(uci)
    if m is None:
        return None
    return max(0, entry["best_cp"] - m["cp"])


def is_acceptable(entry: dict, uci: str, thresh: int = ACCEPTABLE_CP) -> Optional[bool]:
    cl = cp_lost(entry, uci)
    return None if cl is None else cl <= thresh


def is_best(entry: dict, uci: str) -> Optional[bool]:
    cl = cp_lost(entry, uci)
    return None if cl is None else cl == 0


def n_acceptable(entry: dict, legal_ucis: List[str], thresh: int = ACCEPTABLE_CP) -> int:
    return sum(1 for u in legal_ucis if is_acceptable(entry, u, thresh))


class Menu:
    """The shuffled, labelled legal menu for one (position, model)."""

    def __init__(self, fen: str, seed: int, model: str):
        board = chess.Board(fen)
        moves = [(board.san(m), m.uci()) for m in board.legal_moves]
        import random
        rng = random.Random(stable_seed(seed, model, fen))
        rng.shuffle(moves)
        self.fen = fen
        self.board = board
        self.order = moves                                  # [(san, uci)] shuffled
        self.labels = [label_for(i) for i in range(len(moves))]
        self.label_to_uci = {lab: uci for lab, (_, uci) in zip(self.labels, moves)}
        self.label_to_san = {lab: san for lab, (san, _) in zip(self.labels, moves)}
        self.san_to_uci = {san: uci for san, uci in moves}
        self.uci_to_label = {uci: lab for lab, uci in self.label_to_uci.items()}

    def enumerated(self) -> str:
        return "\n".join("{}. {}".format(lab, san)
                         for lab, (san, _) in zip(self.labels, self.order))

    def permutation(self) -> List[str]:
        return [uci for _, uci in self.order]


def llm_cost(model: str, usage: dict) -> Optional[float]:
    if model not in PRICES or not usage:
        return None
    pin, pout = PRICES[model]
    it = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    ot = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return it / 1e6 * pin + ot / 1e6 * pout
