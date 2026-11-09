# Open-world Twenty Questions — expert-onboarding study, pre-registration

**ENV-704 (Phase A — onboarding cost).** The open-world loop (§3.11 / ENV-702) admits a
novel object for free: the moment the expert is called, the object's full Noul row
appears in `H_known`, fully formed. A genuinely open world has to **pay to build the
row** — when the expert admits a novel species, someone must supply its
`p(yes | object, question)` values before the planner can use them. This study makes
admission cost something, models the elicited row as *partial*, and measures how much
onboarding effort the loop actually needs.

(Phase B — two-oracle nested VOI, a cheap Noul oracle plus an expensive human during a
game — is the named follow-on, not this study.)

## The onboarding model

When the expert admits a novel object `o` (an out-of-pool expert call), `o` enters
`H_known` with a row **elicited at effort `E`**:

* a **policy** picks `E` of the 44 attributes — `diagnostic` (o's most distinctive:
  largest |p−0.5| in the true row) or `random` — and sets them to their true values;
* the remaining `44 − E` attributes are set to **0.5** — uninformative: the planner does
  not know o's answer, so on those questions o neither discriminates nor is struck;
* category and identity (the guess question) are closed-form and **free** — naming the
  object supplies them.

The world stays truthful: the **oracle always answers from the true matrix**. Only the
planner's **belief** uses the partial row (a new `like_matrix` seam in
`run_openworld_episode`, default = the true matrix, so ENV-702 is unchanged). So the gap
between what o really is and what the planner has been told about o is exactly `44 − E`
attributes, chosen by the policy.

## Cost

* **Elicitation** costs one attribute-answer per attribute filled: `E` per admission.
* **An expert call** costs 1, whether it admits a new species or re-escalates one whose
  partial row was too thin to resolve a recurrence.
* **Total(E) = c_elicit · (E · admitted) + c_expert · expert_calls.** The tension: a
  richer row (higher `E`) costs more up front but resolves recurrences in fewer questions
  and fewer re-escalations; a thin row is cheap but the species may fail to resolve and
  pay the expert again.

## Metrics (over the stream, per `E` and policy)

* `admitted` — distinct species admitted.
* `recur_games` — in-pool games (a species already admitted when the game starts).
* `recur_resolution_rate` — recurrences resolved (won) ÷ `recur_games`.
* `recur_reescalation_rate` — recurrences that fired the expert *again* ÷ `recur_games`.
* `recur_mean_questions` — questions to resolve a recurrence.
* `onboarding_attrs = E · admitted`, `expert_calls`, and `total(E)` at a few cost ratios.
* (secondary) detector `auc` — partial rows look more like the catch-all, so they may
  blur the known/novel boundary.

## Pre-registered predictions

* **P1 — diminishing returns.** `recur_resolution_rate` rises steeply with `E` then
  plateaus, and `recur_mean_questions` falls then flattens: a **modest `E`** — roughly
  enough diagnostic attributes to separate o from its pool neighbours — captures most of
  the benefit, well below full (44). Full elicitation is wasteful.
* **P2 — order matters.** `diagnostic` elicitation reaches the plateau at a smaller `E`
  than `random`: eliciting the *right* (distinctive) attributes is cheaper than eliciting
  many. At matched `E`, diagnostic ≥ random on resolution.
* **P3 — interior cost optimum.** `total(E)` is U-shaped over a range of
  `c_elicit / c_expert`; the optimal `E*` is interior (`0 < E* < 44`) — neither "admit
  blind" nor "elicit everything" wins once both costs are counted.
* **Falsification.** If resolution is flat in `E` (onboarding buys nothing) or rises
  linearly with no plateau (no diminishing returns), P1 fails; if `diagnostic` and
  `random` coincide, P2 fails.

## Protocol

Same stream, matrix, and PY fit as §3.11 / ENV-702 (Zipf s=1.1, seed 7, 2000 games);
backend `soft` (the BED resolver, ENV-702's best in-pool, isolating the onboarding effect
from elimination dynamics). Sweep `E ∈ {0, 2, 4, 8, 16, 44}` × policy ∈ {diagnostic,
random}, on the clean oracle (onboarding, not noise, is the variable under test) plus a
10% check. `--self-test` asserts the partial-row semantics (an un-elicited attribute is
inert; a full row reproduces the free-admission baseline bit-for-bit) before any run.

Reproduce: `onboarding_openworld --self-test`, then `onboarding_openworld --run --out
<dir>` and `analyze_onboarding --dir <dir>`.
