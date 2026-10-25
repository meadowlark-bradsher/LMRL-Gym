# Chess × Jev — P2: is the move quality Jev-specific?

**Question.** P1 showed Jev plays legal but middling chess. Is that a property of
*Jev* (structured Choice), or just what any model gets once it's constrained to a
legal-move menu? P2 runs two general LLMs over the **same 200 positions** and the
**same frozen Stockfish grader**, in two modes:

- **constrained (arm B)** — shown the identical legal SAN menu, told to pick one.
- **free (arm C)** — asked for the best move with no menu; whatever SAN it emits
  is validated against the board (the original GPT-4-style interface).

Models (user's choice — both providers): **GPT-4-class `gpt-4o`** (OpenAI; the
paper's baseline lineage) and **`claude-opus-5-5`** (Anthropic; frontier tier).
Jev is arm A from P1. Scores reuse the frozen `eval_cache.json`, so only the LLM
calls were new. Per-game records: [`p2_scores.jsonl`](p2_scores.jsonl); combined
with P1 in [`combined_scores.jsonl`](combined_scores.jsonl); full table
[`P2_analysis.md`](P2_analysis.md).

## Results (200 positions/arm)

| arm | illegal | median cpL* | match@1 | blunder | missed mate |
|-----|---------|-------------|---------|---------|-------------|
| `claude_constrained` | 0.5% | **2.5** | **45.2%** | **15.6%** | 23.8% |
| `claude_free` | 3.5% | 3.0 | 43.5%¹ | 10.9%¹ | 28.6% |
| `gpt4o_free` | **18.0%** | 39.0¹ | 31.7%¹ | 32.9%¹ | 31.6% |
| `heuristic_legal` | 0.0% | 105.5 | 26.5% | 40.5% | 40.9% |
| `jev` | 0.0% | 425.5 | 20.5% | 55.5% | 54.5% |
| `gpt4o_constrained` | 1.0% | 422.5 | 17.2% | 54.5% | 59.1% |
| `uniform_legal` | 0.0% | 266.0 | 3.0% | 51.0% | 72.7% |

\*non-mate positions only. ¹ free-arm cpL/match/blunder are on a **censored**
sample — illegal moves have no score and drop out — so the free arms are read for
**legality**, and move quality is compared on the **constrained** arms.

**Paired vs Jev (same positions, 95% cluster-bootstrap CI; − = Jev better):**

| vs | match@1 diff | blunder diff | reading |
|----|--------------|--------------|---------|
| `gpt4o_constrained` | +0.035 [−0.053, +0.089] | +0.015 [−0.082, +0.078] | **indistinguishable from Jev** |
| `claude_constrained` | **−0.246** [−0.389, −0.178] | **+0.400** [+0.329, +0.469] | **Claude dominates Jev** |

## Answer: no, the move quality is not Jev-specific

1. **Legality is a wrapper property, now quantified across models.** Free SAN
   generation is illegal **18.0%** of the time for `gpt-4o` and **3.5%** for
   `claude-opus-5-5`; constraining to the legal menu drops both to ~0.5–1%, the
   same ~0% Jev gets by construction. The paper's GPT-4 illegal-move failure
   reproduces (gpt-4o free, 18%) and is removed by the legality wrapper **for any
   model** — Jev is not unique in reaching 0% illegal.

2. **Jev ≈ a GPT-4-class model on the same menu.** `jev` and `gpt4o_constrained`
   are statistically indistinguishable on every metric: match@1 20.5% vs 17.2%
   (paired CI spans 0), blunder 55.5% vs 54.5% (CI spans 0), median cpL 425 vs
   423. "Structured Choice over legal SAN" delivers legality plus roughly
   GPT-4o-level decision quality — no measurable decision-quality premium over a
   same-era general LLM handed the identical menu.

3. **A frontier model crushes all of them.** `claude_constrained` matches the
   engine's best move **45.2%** vs Jev's 20.5% (+24.6pp, CI clear of 0), blunders
   **15.6%** vs 55.5% (−40pp, CI clear of 0), median loss **2.5cp** vs 425. Given
   the same legal menu, the frontier LLM is in a different class — so the ceiling
   on "constrained-action move quality" is set by the model, not the wrapper.

**Synthesis of P1+P2.** The structured-Choice interface contributes exactly one
thing in chess: the legality guarantee, which is a property of the legal-action
wrapper and available to any model. On move quality Jev tracks a GPT-4-class
model and sits far below a frontier model. So Jev's value proposition here is
"guaranteed-legal, GPT-4o-class moves with a tiny structured call" — not stronger
chess decisions than an LLM would make on the same menu.

## Caveats (load-bearing)

- **Tier asymmetry is deliberate.** `gpt-4o` is GPT-4-class (paper lineage);
  `claude-opus-5-5` is frontier. Results are reported per model, not pooled.
- **A measurement bug was found and fixed mid-run.** `claude-opus-5-5` has
  thinking always on; an initial `max_tokens=64` was consumed by thinking,
  returning empty text and a spurious ~86% "illegal" rate. Raised to 2048 (effort
  low) and the claude arms were re-run; the gpt-4o arms were unaffected. This is
  the same class of silent artifact the P0 gates target — a token-budget bug, not
  a model result.
- **Free-arm quality is censored** (illegal moves excluded), so quality is judged
  on constrained arms only; free arms measure legality.
- **Single live run**, argmax; gpt-4o at temperature 0, claude at effort low. The
  frozen records make the reported statistics exact.
