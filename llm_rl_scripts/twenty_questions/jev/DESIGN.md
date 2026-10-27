# Twenty Questions × Jev — design

## The task

LMRL-Gym Twenty Questions (`llm_rl_scripts/twenty_questions/`). The guesser asks
up to 20 yes/no questions about a hidden object and wins by naming it. The object
is drawn from `DEFAULT_OBJECT_DICT` — **158 objects across 17 categories**. The
benchmark reward is −1 per question, with the winning guess free, so the return
is −(questions − 1) on a win and −20 on a loss.

The benchmark's own oracle is a flan-T5-XL checkpoint on a GCS bucket, which is
unavailable here (no credentials). We therefore run **J/J self-play**: Jev is
both the answerer (oracle) and the world-model whose answer-probabilities drive
belief updates — the same "pure Jev first" choice made for Car-Dealer.

## Where Jev plugs in

This is the BED sibling of Wordle, but it is **not** the degenerate/closed-form
case. In Wordle the likelihood `p(feedback | guess, word)` is deterministic, so
the posterior and EIG need no model. Here the useful questions are
natural-language attributes ("Is it alive?", "Is it bigger than a microwave?")
whose yes-probability has **no closed form** — it needs a world model. That is
exactly the slot the compatibility study reserved for Jev's Noul:

> Jev supplies the yes/no answer-probability = its Noul, which is exactly what
> BED's EIG + hypothesis filter consume.

The belief/planning loop is externalised into pure code (`bed.py`); the only
learned quantity is the per-(object, question) likelihood
`p(answer=yes | object, question)`.

### Primitive: Noul, not Choice

A live probe settled the primitive. Asked `p(yes | object, question)`:

| object | question | `nouls` (Noul) | `choice` over {yes,no} |
|---|---|---|---|
| Cat | Is it alive? | 0.95 | yes, p=1.00 |
| Cat | Is it a kind of animal? | 0.99 | yes, p=1.00 |
| Cat | Can it fly? | 0.04 | no, p=0.00 |
| Car | Is it alive? | 0.02 | no, p=0.00 |
| Car | Does it have a motor? | 0.96 | yes, p=1.00 |

The **Noul is graded**; a yes/no **Choice saturates to 0/1** and throws the
likelihood away. EIG and the Bayesian filter both need the graded quantity, so
the likelihood is read from `nouls`. (A finding in its own right: Choice carries
the decision, the Noul carries the calibrated probability.)

## The loop (`bed.py`)

Hypotheses = the 158 objects. Three kinds of question:

* **category** — "Is it a kind of {phrase}?" for each of the 17 categories.
  Closed-form likelihood (1 if the object's category matches, else 0).
* **attribute** — a bank of ~44 natural-language yes/no attributes. Likelihood =
  Jev's Noul (no closed form).
* **guess** — "Is it a(n) {object}?" for each object. Closed-form; winning the
  game requires a `yes` to one of these.

Each turn: score the candidate questions (all category + all attribute + guesses
for the top-k hypotheses) by **exact EIG** in bits,
`EIG(q) = H(post) − E_a[H(post|a)]`, where `p(yes)=Σ post(h)·p(yes|h,q)`; ask the
argmax; update the posterior by Bayes. Attribute likelihoods are clamped to
`[ε, 1−ε]` (ε=0.05) so one confidently-wrong Noul cannot irreversibly eliminate
the truth; closed-form category/guess likelihoods are applied exactly. The loop
commits to a guess once one hypothesis holds ≥ `confidence` (0.9) of the mass.

## Economy — one cached matrix

`p(yes | object, attribute)` depends only on (object, attribute), never on the
game history, so it is computed **once** for every pair and cached
(`matrix.py`, JSON with provenance). One `nouls` call per object returns the
yes-probability for the whole attribute bank. Every episode then costs only the
oracle: with `--oracle matrix` the oracle thresholds the same matrix (zero API,
self-consistent — isolates the planning loop); with `--oracle jev` the oracle
calls Jev live (crisp Choice over {yes,no}), which adds answer stochasticity and
can disagree with the matrix — the real J/J stress test.

## Arms and metrics

* **matrix oracle** (deterministic, self-consistent): solve rate (expected ≈1.0
  if the bank is discriminative within 20), avg questions, per category.
* **jev oracle** (live J/J): solve rate and avg questions; the gap to the matrix
  oracle is the cost of Jev's answer stochasticity / self-inconsistency.
* **selector role** (`--selector jev`, Wordle-parallel): Jev picks the question
  via Choice over the EIG-ranked shortlist; graded by match@1 to the EIG argmax
  and regret in bits.

## Reproduce

```
# offline self-test (no API)
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev --self-test

# one-time matrix build (live)
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev \
    --build-matrix --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json

# episodes
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev \
    --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json \
    --oracle matrix --episodes 158 --jsonl results/.../episodes.jsonl
```

Run live work with `conda run --no-capture-output -n LLM_RL python -u ...`.
