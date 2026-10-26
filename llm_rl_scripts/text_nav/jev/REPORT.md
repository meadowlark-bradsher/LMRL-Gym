# Text-Nav × Jev — mini-report

> **Status: pre-program exploratory report (superseded in part).** This is the
> original mini-report from before the controlled experiment program. Its numbers
> (the 0/50 baseline and the first "narrative" runs — e.g. "milk taken 54×") come
> from early exploratory runs whose logs were **not committed** and that **predate
> the matched-seed harness**, so unlike the later experiments they are **not
> regenerable** from `results/`. Treat them as observations, not measurements.
> The narrative/salience claims here are **superseded** by the controlled,
> reproducible **Experiment B** ([EXP_B_RESULTS.md](EXP_B_RESULTS.md)), which
> re-establishes them with opportunity-conditioned rates and a selectivity test.
> For the full program and reproducible results, start at [SUMMARY.md](SUMMARY.md).

**Question.** Can the TypeSafe **Jev** model (a "System One" structured-decision
model: Choice / Score / Noul, no text generation) serve as a zero-shot policy on
LMRL-Gym's Text-Nav task, and what does its behavior reveal about the task?

**Setup.** Jev picks each move from the environment's **admissible commands** via
the Choice primitive — no RL training, no fine-tuning. The natural baseline is
therefore the paper's GPT-4 column (also zero-shot), *not* the RL-trained
columns. Code: [`eval_jev.py`](eval_jev.py), [`policy.py`](policy.py),
reusable client in [`LLM_RL/jev/`](../../../LLM_RL/jev/). Model `jev-latest`;
real env runs in the `LLM_RL` conda env (TextWorld 1.5.4).

## What the task actually is

Text-Nav presents itself as navigation ("your task for today is to close the
fridge"), but its reward model ([`env/make_game.py`](../env/make_game.py)) makes
it a **sparse-reward, latent-goal RL task**:

```python
Quest(win_events=[], fail_events=[eaten(food)])                       # eating the food => FAIL
Quest(win_events=[ in(food, fridge) ∧ closed(fridge) ], reward=1)     # +1, terminal, only here
```

Reward is **0 at every step** except the single terminal state where the
perishable is inside the fridge *and* the fridge is closed; eating it **fails**
the episode. There is no reward shaping. The true goal — *fetch the perishable
and put it away* — is never stated; the agent is meant to **discover** it from
the sparse signal. Credit assignment back from that lone +1 is precisely the
capability the benchmark measures (and why the RL baselines do well: ILQL 83.7 /
GPT-4 40.2 normalized on PO; dataset success 0.26).

*(Win-detection was independently verified: stepping each game's own winning
walkthrough through the runner flips `score=1, won=True` every time, so the 0%
results below are real, not a measurement bug.)*

## Results

All runs partially observed (PO), `--max-steps 20`. Success = win (`won`).

| Run | Mode | Episodes | Success | Key behavior |
|-----|------|----------|---------|--------------|
| A | Goal (objective "close fridge") | 50 | **0/50** | Opens then **closes** the fridge — takes the instruction literally; sometimes eats the fruit (the fail event) |
| B | Goal, objective pinned every turn | 50 | **0/50** | Same literal close; pinning the (misleading) objective didn't help |
| C | Narrative persona, food = "stale food" | 10 | **0/10** | **Eating suppressed (0 eats)**; engaged fridge but never took the table food |
| D | Narrative + food = "milk", instruction stripped | 10 | **0/10** | **Took the milk 54×** (vs ~0 before); **0 eats**; but **0 "insert milk into fridge"**, no milk-vs-fresh selectivity (54 vs 58) |

The narrative context (run D):

> *You live alone and have just gotten home after a long shift… You rushed out
> this morning and may have accidentally left something out that can spoil if it
> is not kept cold. You are watching your calorie intake and have already
> exceeded your goal for the day… You dislike letting things go to waste.*

Names no object or action from the task; the surfaced "close the fridge"
instruction was stripped so context stood alone.

## Interpretation

The refined narrative run (D) cleanly separates two layers:

1. **Salience — evoked.** Naming the item "milk" plus the perishable/"left out"
   framing made Jev actively **take the milk** (0 → 54 across episodes), and the
   calorie frame **reliably suppressed eating** (0 of the game's fail action, in
   both narrative runs). Situational context demonstrably reshapes which objects
   and immediate actions are salient.
2. **Plan composition — not evoked.** Jev never assembled *carry milk → kitchen →
   open → insert → close*: 0 inserts of milk into the fridge, no preference for
   the milk over the fresh-food distractor, and incoherent fridge fiddling
   (re-inserting the fruit, trying the chest, dropping the milk).

**Conclusion.** As a per-step selector with no reward signal, value function, or
cross-episode exploration, Jev is structurally mismatched to Text-Nav, whose
whole point is discovering a latent multi-step goal from sparse reward — so 0%
is a category mismatch, not a defect. Narrative context moves the
**recognition/reactive** axis (what matters, what to avoid) but **not** the
**multi-step planning** axis toward an unstated goal. That planning-from-reward
is exactly the RL capability the task is built to test.

**Where Jev should instead win:** tasks where the right move is a legal choice
from the *visible* state with a known objective — e.g. Chess (Choice over legal
SAN moves), where the paper's GPT-4 scores 0 purely from illegal moves, a
failure Jev's constrained Choice makes impossible. That is the next experiment.

## Reproducibility

```bash
# Offline logic check (no key, no network, no TextWorld):
python -m llm_rl_scripts.text_nav.jev.eval_jev --self-test

# Live baseline (goal mode), PO, 50 episodes (needs TYPESAFE_API_KEY):
conda run --no-capture-output -n LLM_RL python -u \
  -m llm_rl_scripts.text_nav.jev.eval_jev --episodes 50 --max-steps 20

# Narrative + milk experiment (run D), 10 episodes:
conda run --no-capture-output -n LLM_RL python -u \
  -m llm_rl_scripts.text_nav.jev.eval_jev --narrative --episodes 10 --max-steps 20
```

Use `conda run --no-capture-output ... python -u` — plain `conda run` buffers all
output until the process exits (looks hung). Each live move is one Jev API call.
Numbers above are from logged runs on 2026-10-25 with model `jev-latest`.
