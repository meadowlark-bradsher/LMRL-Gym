# Open-world Twenty Questions — retraction study, pre-registration

Written before the retraction arms are run. The §3.11 open-world loop (`openworld.py`)
uses one belief representation — a soft, ε-floored Bayesian posterior that never
hard-eliminates. This study asks a different question: **when is hard elimination
safe, and does making it reversible recover what it loses under a noisy answerer?**
That question is the measurable core of the CMBS retraction design (monotone
survivor set + an append-only op-log, now adding an un-eliminate op); this study is
CMBS-*semantics*-aligned but CMBS-*code*-independent — it runs on the 20Q harness
with its own belief backends, and the CMBS store becomes one more backend later.

## Scaffold (this step — no study run yet)

The belief representation is factored behind a `BeliefBackend` interface so the EIG
planner, the oracle, the catch-all, and the accumulation loop are all unchanged
across arms. A backend owns three things and nothing else:

* `distribution()` — the current belief over `H_known ∪ {UNKNOWN}`, which the EIG
  planner and the commit gate read;
* `decision(confidence)` — `ask`, `guess(name)`, or `expert`: each backend owns its
  own commit semantics (a distribution gate for the soft one, a survivor-count gate
  for the hard ones);
* `observe(question, answer)` — the belief update.

The planner (`rank_questions` / `expected_info_gain` over `distribution()`) and the
loop are shared, so any difference between arms is the belief update alone.

### Backends

1. **`soft`** — the §3.11 posterior: ε = 0.05 floored Bayesian filter, commit when
   the argmax ≥ confidence or a single hypothesis remains. **Behavior-preserving:**
   this backend must reproduce the committed clean-arm run bit-for-bit (146 distinct,
   144 expert, 0 false-positive, 1854 won, 2 wrong) — the refactor's regression gate.
2. **`hard`** — monotone hard elimination (the CMBS baseline stance): a known object
   is struck when an answer contradicts it, and never returns. Documented semantics,
   because hard elimination interacts with the catch-all in a specific way:
   * It can rule a hypothesis *out* but not *in* — the un-contradictable `UNKNOWN`
     never dies — so a known object is committed only when it is the **sole surviving
     known**; the expert fires when the surviving knowns empty, or when the PY
     `UNKNOWN` mass dominates. The distribution is uniform over survivors for the
     known mass, `UNKNOWN` carrying the PY mass (1.0 once the knowns are empty).
3. **`retract`** *(next step, CMBS-aligned — not in this scaffold)* — `hard` plus an
   append-only elimination log where a kill records the evidence that caused it, and
   an un-eliminate op reverses a kill when later evidence outweighs it. The exact
   trigger and envelope follow the CMBS spec; the scaffold leaves the extension point.

## Predictions (to be tested once `retract` lands and the study runs)

* **P1 (clean parity).** With a self-consistent oracle, `soft` and `hard` both solve
  the in-pool games — `hard` reaches the true object by elimination down to the sole
  survivor — so clean-oracle miss rates are comparable and both detectors fire on
  novelty. Hard elimination is not worse *when the answerer is trusted*.
* **P2 (hard breaks under noise).** As the attribute-flip rate rises (0 → 5 → 10%),
  `hard`'s missed-in-pool rate climbs **steeply** — a single flipped answer strikes
  the true object and monotonicity forbids recovery, so the game ends in a wrong
  guess or a false-positive expert call. `hard` degrades **worse than `soft`** at
  every positive noise level (soft's ε-floor only dents the truth).
* **P3 (retraction recovers most of the loss).** `retract` sits between `hard` and
  `soft`: reversing the eliminations whose evidence is outweighed recovers the bulk
  of `hard`'s noise-induced misses, approaching `soft`'s miss rate, **at the cost of**
  a bounded number of un-eliminate ops per game (reported, not hidden).
* **P4 (the audit trail is the point).** `retract` matches `soft`'s outcomes closely
  while keeping what `soft` lacks — an append-only, replayable log of every kill and
  reversal. The finding, if P3 holds, is that reversibility buys soft-posterior
  robustness without giving up hard elimination's auditability.

## Metrics (per backend × noise level, seed-replicated)

Missed-in-pool rate (in-pool game that ends wrong or fires a false-positive expert),
detector AUC, false-positive expert rate, mean questions, and — for `retract` — mean
un-eliminate ops per game. The stream, PY fit, and noise grid are exactly §3.11's, so
arms differ only in the belief backend.

## Falsification

P2 failing (hard no worse than soft under noise) would mean the matrix is
discriminative enough that elimination rarely strikes the truth — hard elimination
would then be safe as-is and retraction unnecessary on this task. P3 failing
(retraction does not recover the loss, or needs unbounded ops) would mean the
reversal trigger is wrong, not that reversibility is worthless.
