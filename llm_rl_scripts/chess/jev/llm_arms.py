"""P2 baseline arms: a general LLM playing chess, two providers x two modes.

This is deliberately a *multi-provider* comparison file (not an Anthropic-SDK
app): it calls OpenAI and Anthropic through raw HTTP (urllib) so the harness
stays dependency-free and symmetric across providers, the same way the Jev
client does. Keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY.

Modes:
  * constrained -- the model is shown the identical legal SAN menu and told to
    return exactly one of those moves (arm B: same legality guarantee as Jev).
  * free        -- the model is asked for the best move with no menu; whatever
    SAN it emits is validated against the board (arm C: measures the illegal-
    move rate the legality wrapper removes -- the original GPT-4-style interface).

Each chooser returns (san_text, exception_type). The runner resolves the SAN to
a move via python-chess (tolerant of spelling), so an illegal or unparseable
free-mode answer simply resolves to None and is recorded as illegal.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

import chess

# arm name -> (provider, model, mode)
ARM_SPECS = {
    "gpt4o_constrained": ("openai", "gpt-4o", "constrained"),
    "gpt4o_free": ("openai", "gpt-4o", "free"),
    "claude_constrained": ("anthropic", "claude-opus-5-5", "constrained"),
    "claude_free": ("anthropic", "claude-opus-5-5", "free"),
}

_SYS_CONSTRAINED = (
    "You are a strong chess engine. Choose the single best move for {side}. "
    "You MUST pick exactly one move from the provided list of legal moves. "
    "Respond with ONLY that move in SAN notation and nothing else."
)
_SYS_FREE = (
    "You are a strong chess engine. Output the single best move for {side} in "
    "standard algebraic notation (SAN). Respond with ONLY the move and nothing "
    "else (e.g. Nf3, exd5, O-O, e8=Q)."
)


def _http_post(url: str, headers: dict, body: dict, timeout: float = 60.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _with_retry(fn, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529) and i < tries - 1:
                time.sleep(2.0 * (i + 1))
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            if i < tries - 1:
                time.sleep(2.0 * (i + 1))
                continue
            raise
    raise last  # pragma: no cover


def call_openai(model: str, system: str, user: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 32, "temperature": 0,
    }
    headers = {"Authorization": "Bearer {}".format(key), "Content-Type": "application/json"}
    resp = _with_retry(lambda: _http_post("https://api.openai.com/v1/chat/completions", headers, body))
    return resp["choices"][0]["message"]["content"] or ""


def call_anthropic(model: str, system: str, user: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = {
        "model": model, "max_tokens": 2048, "system": system,
        "messages": [{"role": "user", "content": user}],
        # opus-5-5 has thinking always on; a small max_tokens gets consumed by
        # thinking and returns empty text (stop_reason=max_tokens), so give it
        # room. effort=low keeps the thinking (and cost) modest.
        "output_config": {"effort": "low"},
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    resp = _with_retry(lambda: _http_post("https://api.anthropic.com/v1/messages", headers, body))
    # Concatenate any text blocks (thinking blocks carry empty text by default).
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


def _board_text(board: chess.Board, side: str) -> str:
    return "You are {}.\n\nBoard:\n{}\n\nFEN: {}".format(side, board, board.fen())


_TOKEN_RE = re.compile(r"[OKQRBNa-h0-9x+#=\-]+")


def extract_san(text: str, board: chess.Board, menu: Optional[List[str]] = None) -> Optional[str]:
    """Pull a SAN move out of a model reply, tolerant of chatter/formatting."""
    if not text:
        return None
    cleaned = text.strip().strip("`\"' .").replace("\n", " ")
    # Drop a leading move number like "12." or "7..." if present.
    cleaned = re.sub(r"^\d+\.(\.\.)?\s*", "", cleaned)
    cands = [cleaned] + _TOKEN_RE.findall(cleaned)
    if menu:
        mset = set(menu)
        for c in cands:
            if c in mset:
                return c
        # Fall back: a menu move appearing anywhere in the reply.
        for m in menu:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(m) + r"(?![A-Za-z0-9])", text):
                return m
        return None
    # Free mode: return the first candidate python-chess can parse as legal SAN.
    for c in cands:
        try:
            board.parse_san(c)
            return c
        except (ValueError, AssertionError):
            continue
    # Nothing parsed: return the first bare token so it's recorded as illegal.
    toks = _TOKEN_RE.findall(cleaned)
    return toks[0] if toks else cleaned[:12]


def choose(arm: str, board: chess.Board, side: str,
           presented_sans: List[str]) -> Tuple[Optional[str], Optional[str]]:
    provider, model, mode = ARM_SPECS[arm]
    side_cap = side.capitalize()
    if mode == "constrained":
        system = _SYS_CONSTRAINED.format(side=side_cap)
        user = _board_text(board, side_cap) + "\n\nLegal moves: " + ", ".join(presented_sans) + \
            "\n\nRespond with exactly one of these moves."
    else:
        system = _SYS_FREE.format(side=side_cap)
        user = _board_text(board, side_cap) + "\n\nWhat is the best move for {}?".format(side_cap)
    try:
        text = call_openai(model, system, user) if provider == "openai" \
            else call_anthropic(model, system, user)
    except Exception as e:  # network / provider / key errors
        return None, "{}: {}".format(type(e).__name__, e)
    san = extract_san(text, board, menu=presented_sans if mode == "constrained" else None)
    return san, None
