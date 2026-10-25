# Chess × Jev — mini-report

> **Status (2026-10-25):** superseded in part by follow-up work.
> - **The draw column below is an artifact.** Re-run under the P0 termination
>   accounting ([`results/ladder_20261025/LADDER_RESULTS.md`](results/ladder_20261025/LADDER_RESULTS.md)),
>   **all 28 games ended in checkmate or a move-cap truncation — zero genuine
>   draws.** The old "draws" were unfinished games counted as ½. So the 0.75
>   score vs random (3 wins + 3 *unfinished*) does not support a rating margin;
>   Jev loses every finished game to every engine setting (incl. depth-1/Skill-0).
> - **Primary study:** [`results/primary_20261025/PRIMARY_RESULTS.md`](results/primary_20261025/PRIMARY_RESULTS.md) —
>   on 200 fixed positions Jev beats a random mover (match@1 20.5% vs 3.0%) but
>   does **not** beat a cheap no-model heuristic and blunders more than it.
> - **P2 — is it Jev-specific?** [`results/p2_20261025/P2_RESULTS.md`](results/p2_20261025/P2_RESULTS.md):
>   no. On the same 200 positions Jev is statistically **indistinguishable from
>   `gpt-4o` constrained to the legal menu** and far below **`claude-opus-5-5`**
>   (best-move match 45% vs 20%). Legality is a wrapper property any model gets
>   (free generation is 18% illegal for gpt-4o, ~0% once constrained).
> - The Elo estimate below remains speculative; it rested on the artifact draw
>   rates and on non-calibrated weak opponents.

**Question.** Can the TypeSafe **Jev** model (a "System One" structured-decision
model: Choice / Score / Noul, no text generation) play LMRL-Gym's Chess task as a
zero-shot policy, and what playing strength (Elo) does it reach?

**Setup.** Jev plays White and picks each move from the position's **legal SAN
moves** via the Choice primitive — no RL training. Because every option is a
legal move, Jev *cannot* emit an illegal move. The natural baseline is the
paper's GPT-4 column (also zero-shot). Code: [`eval_jev.py`](eval_jev.py),
[`policy.py`](policy.py), [`simple_env.py`](simple_env.py) (pure python-chess,
random opponent), [`stockfish_env.py`](stockfish_env.py) (Stockfish opponent via
python-chess UCI — the repo's `ChessEnv` pulls in JaxSeq, so this bypasses it).
Model `jev-latest`; opponent Stockfish 19 at 0.1 s/move; agent moves capped as
noted. Reward: +1 win, −1 illegal-or-loss, 0 otherwise.

## The legality result (the thesis)

Across **every game in this report — 0.0% illegal moves, live.** The failure mode
that gives the paper's GPT-4 a raw −81.3 (normalized 0) and leaves even trained
models at 22–35% illegal in full games (Table 6) is **structurally impossible**
for Jev: Choice-from-legal can only return a legal move. On the paper's
summed-reward metric this even looks dominant — Jev's clean −1.0 per game (one
terminal loss, zero illegal penalties) beats GPT-4's −81.3 and ILQL's −20.5 —
but that is precisely because that metric is dominated by illegal-move penalties,
the one axis Jev can't lose on. **Legality is necessary, not sufficient.**

## Strength ladder (live)

Score = win + ½·draw. 0% illegal throughout.

| Opponent | Games | W / D / L | Score | Jev vs opp |
|----------|-------|-----------|-------|------------|
| Random mover | 6 | 0.50 / 0.50 / 0.00 | **0.75** | ~190 Elo above |
| Stockfish depth-1 | 6 | 0.00 / 0.50 / 0.50 | 0.25 | ~190 Elo below |
| Stockfish Skill-0 | 6 | 0.00 / 0.17 / 0.83 | 0.083 | ~420 Elo below |
| Stockfish UCI_Elo 1320 | 10 | 0.00 / 0.00 / 1.00 | 0.00 | ≥ ~180 below (upper bound) |

(Full-game vs SF-1320: avg 21.5 plies before mate, range 11–39.)

## Elo estimate: ~700–900 (rough)

From ΔElo = −400·log₁₀(S/(1−S)): Jev is **~190 Elo above a random mover**
(score 0.75, never loses to it) and **~190 below depth-1 Stockfish** (score
0.25) — the 50% crossover sits between those two. The only Elo-*calibrated*
anchor, UCI_Elo 1320 at 0/10, gives a firm **upper bound of ~1100–1140**
(rule-of-three). Anchoring the chain to a random mover (~500–700 on the human
scale) places Jev around **700–900**.

**Caveat (load-bearing):** the dominant uncertainty is not Jev's scores but the
missing Elo calibration of the sub-1320 opponents (random, depth-1, Skill-0).
More Jev games would not tighten this; only calibrated weak opponents would. The
one hard claim is the upper bound: **Jev is below ~1100**, and loses to every
genuine Stockfish setting tested.

## Playing profile

Consistent across all runs: Jev plays legal, clearly better-than-random moves and
reaches winning positions against a random opponent — but **half those become
draws** because it cannot convert mates, and it loses to even the weakest real
engine. That is a **novice** signature: plausible moves, no tactical calculation,
no endgame technique. It fits the overall thesis — Choice-from-legal removes the
illegal-move failure entirely, but a per-move selector with no search does not
play *strong* chess.

## Endgames: K+Q vs K — can Jev convert a won position?

10 games, a fresh random KQK position each game, randomly-defending lone king,
White (Jev) to convert:

| W / D / L | Score | Avg moves | Illegal |
|-----------|-------|-----------|---------|
| 0.10 / 0.90 / 0.00 | 0.55 | 3.5 | 0% |

Instrumented terminations show the draws are **insufficient material** — i.e. the
**queen is captured**, leaving K-vs-K. Mechanism, verbatim: Jev's move is a queen
check delivered from a square *adjacent to the bare king with the queen
undefended* (e.g. `1. Qe2+`, `1. Qg8+`, at low confidence 0.15–0.22); the king
simply captures it, ending the game as a draw — often on **move 1**.

For comparison, the paper's Table 7 endgame victory rates are BC 26%, ILQL 57%,
MC 69%, PPO-online 88% — all against a *Stockfish-1200* defender. Jev converts
only ~10% against an *easier* random defender, below even BC.

**Takeaway:** in a trivially won position, a lookahead-free selector throws away
the decisive piece because it cannot see the recapture. Legality is guaranteed;
capture-awareness and endgame technique are absent. Endgames does not rescue Jev
— it exposes the no-search limitation in its starkest form.

## Reproducibility

```bash
# Offline logic check (LLM_RL env for python-chess; no key, no engine):
conda run --no-capture-output -n LLM_RL python -u \
  -m llm_rl_scripts.chess.jev.eval_jev --self-test --episodes 3

# Live vs Stockfish (needs TYPESAFE_API_KEY and a Stockfish binary):
EP=/opt/homebrew/bin/stockfish
conda run --no-capture-output -n LLM_RL python -u -m llm_rl_scripts.chess.jev.eval_jev \
  --opponent stockfish --engine-path $EP --episodes 10 --max-moves 60      # UCI_Elo 1320
conda run --no-capture-output -n LLM_RL python -u -m llm_rl_scripts.chess.jev.eval_jev \
  --opponent stockfish --engine-path $EP --engine-skill 0 --episodes 6 --max-moves 30
conda run --no-capture-output -n LLM_RL python -u -m llm_rl_scripts.chess.jev.eval_jev \
  --opponent stockfish --engine-path $EP --engine-depth 1 --episodes 6 --max-moves 30
conda run --no-capture-output -n LLM_RL python -u -m llm_rl_scripts.chess.jev.eval_jev \
  --opponent random --episodes 6 --max-moves 30

# Endgames: K+Q vs K conversion (fresh random position per game):
conda run --no-capture-output -n LLM_RL python -u -m llm_rl_scripts.chess.jev.eval_jev \
  --endgame-pieces kQK --opponent random --episodes 10 --max-moves 40
```

Stockfish's `UCI_Elo` floors at 1320; weaker opponents use Skill Level / depth
limits (not Elo-calibrated). Each live agent move is one Jev API call. Numbers
above are from logged runs on 2026-10-25 with model `jev-latest`, Stockfish 19.
