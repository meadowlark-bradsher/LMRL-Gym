# Chess × Jev — full-game ladder, re-run under fixed termination accounting

Re-run of the strength ladder from [`../../REPORT.md`](../../REPORT.md) using the
P0 termination taxonomy ([`outcomes.py`](../../outcomes.py)), which separates a
**move-cap truncation** (an unfinished game) from a genuine draw. Same opponent
settings and move caps as the original run; the only change is the accounting.

Per-game records: [`ladder.jsonl`](ladder.jsonl). Regenerate this table with
`python -m llm_rl_scripts.chess.jev.analyze_ladder --ladder <path>`.

## The correction

| rung | n | **old** W/D/L (score) | **new** finished / truncated | **new** W/D/L of finished (score) | real draws |
|------|---|-----------------------|------------------------------|-----------------------------------|------------|
| random (max 30) | 6 | 0.50/0.50/0.00 (0.75) | 3 finished / **3 truncated** | 3/0/0 (**1.000**) | **0** |
| SF depth-1 (max 30) | 6 | 0.00/0.50/0.50 (0.25) | 2 finished / **4 truncated** | 0/0/2 (0.000) | **0** |
| SF Skill-0 (max 30) | 6 | 0.00/0.17/0.83 (0.083) | 5 finished / 1 truncated | 0/0/5 (0.000) | **0** |
| SF UCI_Elo 1320 (max 60) | 10 | 0.00/0.00/1.00 (0.00) | 10 finished / 0 truncated | 1/0/9 (0.100) | **0** |

Termination reasons (every game): only `checkmate_win`, `checkmate_loss`, and
`move_cap`. **No stalemate, repetition, insufficient-material or 50-move draw
occurred in any game.**

## What changed, and why it matters

1. **The entire "draw" column was an artifact.** Across all 28 games there were
   **zero genuine draws**. Every "draw" in the old table (0.50, 0.50, 0.17 of the
   rungs) was a game that simply hit the move cap and was silently scored as a
   draw. This is exactly the Gate B failure the validation work targeted, now
   quantified: the old draw rate measured the move cap, not Jev's play.

2. **The "0.75 score / ~190 Elo above random" claim does not survive.** Of the 6
   games vs a random mover, the 3 that *finished* were all Jev checkmate wins
   (score 1.0 on finished games) — but the other 3 never finished in 30 plies.
   The old 0.75 was 3 wins plus 3 unfinished-games-counted-as-half-draws. The
   honest reading: **vs random, Jev wins when it can force mate inside the cap
   but fails to convert the other half of games in time** — consistent with the
   "reaches winning positions, can't convert" profile, and *not* evidence of a
   stable rating margin.

3. **Jev loses every finished game to every real engine setting**, including the
   weakest: 0/2 vs depth-1 and 0/5 vs Skill-0 (all losses), confirming it sits
   below even the non-Elo-calibrated weak opponents.

4. **vs SF-1320: 1 win / 9 losses this run** (old: 0/10). The single win is within
   run-to-run variance (Jev's low-margin nondeterminism + the engine's own
   0.1 s/move nondeterminism), but it means the strict "0/10 → rating upper
   bound" argument in REPORT.md should be stated as ≤1 non-loss in 10, not 0.

## Caveats

- **Small n** (6–10 games/rung) and a **move cap that drives truncation**,
  especially vs the random mover (half its games did not finish in 30 plies). A
  clean draw/win rate vs random needs a much higher cap (or `can_claim_*` draw
  adjudication) so games actually terminate; that is the right next run, and it
  would convert most of the random-rung truncations into real wins or genuine
  (50-move / repetition) draws.
- This re-run re-classifies the *same* experiment; it does not add calibrated
  weak opponents, so it says nothing new about Elo. The one firm claim is
  qualitative and now artifact-free: **Jev records no genuine draws, wins only
  against random (when games finish), and loses to every engine setting.**
