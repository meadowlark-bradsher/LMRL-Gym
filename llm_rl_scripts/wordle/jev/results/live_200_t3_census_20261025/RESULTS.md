# Wordle × Jev — rigorous 3-guess benchmark (cluster-bootstrap CIs)

**Setup.** `jev-latest`, vocab `wordle_official_200.txt` (236 words), **census**
(every word is a target once) × **5 independent Jev rollouts** per target,
**3-guess budget**, seed 0 → 1,180 Jev games, 2,706 scored Jev decisions, 0 API
errors. Optimal arm = greedy exact-EIG restricted to the posterior (the repo's
`OptimalPolicy` semantics), same 5 repeats. 95% CIs are a cluster bootstrap
(B=10,000) that resamples **target words** (repeats within a word are
correlated), via `analyze_wordle.py`.

## Headline

| quantity | estimate | 95% CI |
|---|---|---|
| Jev solve rate | **0.741** | [0.696, 0.784] |
| Optimal solve rate | **0.936** | [0.912, 0.958] |
| **Gap (optimal − Jev)** | **0.196** | **[0.146, 0.247]** |
| Jev avg guesses (solved) | 2.712 | [2.655, 2.767] |
| match@1 (Jev turns) | 0.275 | [0.255, 0.296] |
| mean regret | 0.586 bits | [0.572, 0.601] |

**The gap's CI clears zero.** At a 3-guess budget the exact optimum solves ~20
points more words than Jev, and the interval [14.6, 24.7] is nowhere near 0.

## Reading

- Jev's per-turn selection is **chance-level**: match@1 = 0.275 and regret =
  0.586 bits match the offline *random* mock (0.246 / 0.712) far more than any
  competent selector, and are flat across every budget we've run (6/4/3 tries).
  The quantity that changes with the budget is the *cost* of that chance-level
  selection, not its quality.
- This confirms the pilot (50 episodes gave Jev 0.78 / optimal 0.94); the pilot
  was a mild up-draw of the same ~0.74 process. The finding survives the
  CI-and-repeats treatment the rest of the study uses.

## Honest caveats

- The "optimal" arm is greedy exact-EIG **restricted to still-possible words**
  (hard-mode-ish, myopic — no multi-step lookahead), matching the repo's
  `OptimalPolicy`. It is therefore *beatable on individual words*: Jev wins some
  targets outright (e.g. `group` 5/5 vs 1/5, `sooth` 5/5 vs 0/5) where a
  non-greedy guess happens to fit the tight budget. In aggregate it still
  dominates, and the per-turn **regret** is measured against this same greedy
  argmax, so it is the consistent reference.
- Single vocabulary (236 words, ≤ the 255 Choice cap, so Jev chooses on every
  turn — fully paired). The larger vocabularies are perf-gated (EIG is O(n²))
  and untested here.

## Reproduce

```
# this exact run (needs TYPESAFE_API_KEY):
conda run --no-capture-output -n LLM_RL python -u \
    -m llm_rl_scripts.wordle.jev.eval_jev --census --repeats 5 --max-guesses 3 \
    --seed 0 --vocab llm_rl_scripts/wordle/vocab/wordle_official_200.txt \
    --jsonl results/live_200_t3_census_20261025/games.jsonl

conda run --no-capture-output -n LLM_RL python -u \
    -m llm_rl_scripts.wordle.jev.analyze_wordle \
    results/live_200_t3_census_20261025/games.jsonl --bootstrap 10000
```
