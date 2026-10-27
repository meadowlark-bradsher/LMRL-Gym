# Twenty Questions × Jev — results

Live `jev-latest`, 2026-10-26. BED loop in pure code (`bed.py`), likelihood from
Jev's cached Noul matrix (`matrix_jev.json`, 158 objects × 44 attributes). All
numbers regenerate from the committed `results/*/summary.json` and
`episodes.jsonl`; configs are in each `summary.json["config"]`.

## Headline: the same model is perfect as the likelihood and useless as the planner

| role Jev plays | who plans | solve | avg questions | selection match@1 |
|---|---|---|---:|---:|
| **Likelihood** (Noul) + matrix oracle | BED (EIG argmax) | **158/158 = 1.000** | **10.21** | — |
| **Likelihood** (Noul) + live J/J oracle | BED (EIG argmax) | **158/158 = 1.000** | **10.19** | — |
| **Selector** (Choice), no history | Jev | **0/158 = 0.000** | 20.0 (cap) | **0.048** |
| Selector (Choice) + no-repeat | Jev | 0/158 = 0.000 | 20.0 (cap) | 0.077 |
| Selector (Choice) + history | Jev | **137/158 = 0.867** | 14.12 | 0.202 |

Census = every one of the 158 objects played once.

## 1. Jev's Noul is a well-calibrated likelihood

`analyze_matrix.py` on the 14 attributes that name a category (ground truth is
known from the object dictionary): mean **AUC 0.977**, min 0.857, 13/14 ≥ 0.90.
The Noul separates in-category from out-of-category objects almost perfectly
("Is it an animal?" AUC 1.000, mean 0.99 in / 0.02 out). The one low value,
`write_draw` (AUC 0.857, mean-in 0.45), is a genuinely partial attribute — only
some office/art items are for writing — not a Jev error.

The live probe behind the design: asked `p(yes | object, question)`, `nouls`
returns graded values (Cat/alive 0.95, Cat/fly 0.04, Car/alive 0.02) where a
yes/no `choice` saturates to 0/1. EIG and the filter both need the graded
quantity, so the likelihood is read from the Noul, not from Choice.

## 2. With the loop externalised, BED solves every object in ~10 questions

Matrix oracle (deterministic, self-consistent): **158/158**, avg **10.21**
questions (min 4, median 10, max 18). The tail is within-category
disambiguation (Sports avg 12.4, Tools 14.5, Toys 15.0). Every object is found
inside the 20-question budget.

## 3. Live J/J costs ~nothing: Jev answers its own questions consistently

Live Jev oracle (answers each attribute question with a Choice over {yes,no}):
**158/158**, avg **10.19** — indistinguishable from the deterministic oracle.
Jev's live answers match its own Noul matrix on **390/393 = 0.992** of attribute
questions; the 3 disagreements caused 0 losses (the ε-floor recovers). So Jev's
world-model is coherent across the two ways it is queried.

## 4. As the planner, Jev collapses — and the mechanism is the thesis

When Jev picks the question (Choice over the EIG-ranked shortlist) instead of
supplying the likelihood, it solves **0/158**, hits the 20-question cap every
game, and matches the EIG argmax only **4.8%** of the time (mean regret 0.68
bits). The mechanism: it fixates on four generic "good 20-questions" questions —
"smaller than a microwave?" (chosen 931×), "man-made?" (738×), "indoors?"
(644×), "alive?" (520×) — and **re-asks them** (one game asked "smaller than a
microwave?" on turns 4, 5, 7, 9, 10, EIG decaying 0.50 → 0.06). It has no belief
state, so it cannot track what it already resolved or steer toward the remaining
uncertainty. This is the Car-Dealer / Text-Nav finding in a third task: Jev takes
the locally-salient option and does not compose a sequential strategy.

### Fairness controls (do history / no-repeat rescue the selector?)

The no-history selector literally cannot know what it already asked. Two
controls, echoing the Text-Nav goal/history decomposition, separate "it just
repeats itself" from "it has no belief state":

* **+no-repeat** — already-asked questions are dropped from the shortlist, so
  repetition is mechanically impossible. **Still 0/158** (match@1 0.077). Blocking
  the exact repeat does nothing: with no context, Jev just picks its
  next-most-generic favorite (a near-duplicate), and the posterior still stalls.
* **+history** — the running Q&A transcript is put in the Choice state. **Jumps to
  137/158 = 0.867**, avg 14.1 questions, match@1 0.202, regret 0.323 bits.

So the binding constraint is **the belief/history representation, not repetition**.
Given the transcript, Jev becomes a competent selector; blocking repeats without
context leaves it lost. And even with history it is far from optimal — match@1
0.20 (≈10× the 1/48 chance rate) — so it succeeds the way it does in Wordle: the
code-maintained posterior does the heavy lifting and a forgiving task tolerates
sub-optimal questions, not because Jev selects well. This is the Text-Nav history
effect and the Wordle "posterior does the work" effect in one experiment.

## Reproduce

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_jev --self-test
# matrix census (no API), jev census, selector arms: see each results dir's config
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_matrix \
    --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json
```
