# Chess × Jev — P0 validation gates

Prerequisites that must pass before any live evaluation is interpreted. All are
offline (no API spend) and run via:

```bash
conda run --no-capture-output -n LLM_RL python -u -m llm_rl_scripts.chess.jev.validate --all
```

| Gate | What it guarantees | Where |
|------|--------------------|-------|
| **A — engine config** | Opponent strength knobs are actually honoured, not silently ignored. `configure` no longer swallows errors; `verify_engine_config` fails closed if a requested UCI option is undeclared or out of range. The binary self-reports `UCI_Elo: spin[1320,3190]` — the 1320 floor, direct evidence that sub-1320 "Elo" anchors are impossible. | `stockfish_env.py` |
| **B — termination** | Checkmate / stalemate / insufficient material / repetition / 50-move / **move-cap truncation** / illegal / API error are each named and counted distinctly. A hit move-cap or a provider failure is reported as **truncated**, never silently counted as a chess draw. | `outcomes.py` |
| **C — reward accounting** | Terminal loss, illegal-move penalty and truncation are separated; score (W+D/2) is computed over *finished* games only. A deterministic mating-line fixture confirms env reward == `classify()`. | `outcomes.py`, `eval_jev.py` |
| **D — legality fail-closed** | The move Jev returns is asserted to be in the presented legal menu before it is applied; a malformed response or provider exception is recorded as `api_error` (truncated), never repaired or mapped to a legal move. | `eval_jev.py`, `position_eval.py` |
| **SAN↔UCI** | `SAN → parse → push` round-trips for promotion, castling, en passant and checkmate SAN. | `validate.py` |

**Status:** all gates PASS (2026-10-25, Stockfish 19, python-chess 1.11.2).

## Why this changes a prior number

The original full-game runner reported every `reward == 0` game — including games
that merely hit the move cap — as a "draw". Under the fixed accounting, a capped
game is **truncated**, reported separately. So the strength-ladder draw rates in
[`REPORT.md`](REPORT.md) (e.g. "half the winning positions become draws" vs the
random mover) conflate real draws with unfinished games and must be re-run under
`outcomes.py` before they are trusted. The endgame insufficient-material result
is unaffected: those games terminated on a genuine board rule, which the taxonomy
confirms.
