# ST-3 follow-up: the stream-K prior — pre-registration (draft)

**Status (2026-11-13):** built and gated (`STRESS_ST3_STREAMK_BUILD.md`); not tagged, not run.

## Question

ST-3 (`STRESS_ST3_RESULTS.md`, FINDINGS §3.14) found that every prior arm reads K from
|H_known|, the loop's own admissions. A backend that misses novel objects
therefore under-counts K and runs a lower prior than the stream's real missing mass:

* `soft`'s `true` prior ended at 0.049 at σ = 0.75, against a stream missing mass of
  about 0.28.
* `soft` flagged only 0.12 of novel games, and 0.10 of the generated children.

Two causes are confounded:

* **Self-censoring:** the prior is too low because K is too low.
* **Near-copy likelihood:** the parent of a child explains the answers so well that
  no reasonable prior moves the decision.

ST-3's P3 hints at the second: a factor-3 change in the prior moved soft's TPR by
0.005. But that change was between two *censored* priors (0.049 against 0.016).

This follow-up gives the gate the missing mass a correctly specified model would
carry. It keeps the generating (σ, θ), but takes K as the stream's distinct species
before game g, whatever the loop admitted. Everything else is held fixed.

## Design

* **The `stream` prior:** p_new = (θ + K_stream σ)/(θ + g), where K_stream is the
  number of distinct species in the stream before game g. It is 1.0 while
  |H_known| = 0, as in every ST-3 arm. It is never below the `true` arm's prior,
  because the loop only admits species the stream has shown.
* **Grid:** σ ∈ {0, 0.25, 0.5, 0.75} × flip ∈ {0, 0.05} × ST-3's stream and noise
  seeds × three arm-backends: primary `soft`, primary `modeswitch_guard` (published
  P0a) and norepeat `modeswitch_guard` (P0a with `no_repeat`). That is **108 jobs of
  5000 games**, with ST-3's budget, confidence, ε, K and real matrix.
* **Pairing:** each job `s<σ>_stream_<rest>` uses the same stream, objects, oracle
  seed and backend as ST-3's `s<σ>_true_<rest>`. Only the K inside the prior
  differs. Every comparison below is stream-K against ST-3's committed `true` cell
  (`results/stress_st3_20261113`); ST-3 is not re-run.
* **Why these arms:**
  * `soft` is the backend whose collapse is in question.
  * Norepeat P0a is where ST-3 found the prior trading escalations against wrong
    answers.
  * Published P0a catches every novel object at σ = 0.75, so its K already equals
    the stream's. That makes it the control.
  * `hard` equals P0a at flip 0 and is left out.

## Predictions

* **SK0 — mechanical gate.** At σ = 0 the PY prior is θ/(θ + g) whatever K is, so
  all 27 σ = 0 stream-K jobs play exactly the games of their ST-3 `true` pair.
* **SK1 — `soft`'s collapse is the near-copy likelihood, not self-censoring.**
  * Let gain = TPR(stream) − TPR(true) for primary `soft`, all games, flip 0.
  * Reading:
    * **likelihood** if the gain is ≤ 0.05 at both σ = 0.5 and σ = 0.75;
    * **self-censoring** if it is ≥ 0.15 at either;
    * **mixed** otherwise.
  * The registered expectation is *likelihood*. Arithmetic behind it: the stream
    prior moves the log-odds of UNKNOWN by about 2 nats at σ = 0.75. A child that
    matches its parent on ~18 of 20 answers leads the background by several nats
    more than that under ε = 0.05.
* **SK2 — the discrete gate moves along the escalate-vs-wrong knob, and stays
  unsafe.** For norepeat P0a at flip 0:
  * at σ ∈ {0.5, 0.75}, TPR(stream) ≥ TPR(true), FPR(stream) ≥ FPR(true) and
    wrong(stream) ≤ wrong(true);
  * at σ = 0.75, wrong(stream) ≥ 0.05. The correct missing mass does not make the
    discrete gate safe on a crowded pool.
* **SK3 — control.** For published P0a at σ = 0.75, flip 0, TPR, FPR and the
  autonomous share each move by at most 0.01.

## Scoring

`analyze_st3_streamk.py` implements these rules; the code is the reference when this
wording is ambiguous.

* **Cells and rates:**
  * A cell is (σ, flip, arm, backend), pooled over its stream seeds and, under
    noise, its noise seeds.
  * Novel game, TPR, FPR, wrong rate and autonomous share are as in `analyze_st3`,
    over all 5000 games.
  * Each stream-K cell is compared with the ST-3 `true` cell of the same key.
* **Verdicts:**
  * SK0, SK2 and SK3 are PASS only if every part holds.
  * SK1 reports its reading, and is PASS only on *likelihood*.
  * A missing cell or a missing base job counts as not holding.
* **Reported, not scored:**
  * every cell at both flips and every σ, by half;
  * TPR on generated children (`tpr_tail`) and the AUC change;
  * the prior in the last window;
  * admissions;
  * the per-job range of the TPR and wrong-rate changes. Pairing is by stream seed,
    and there are 3 per flip-0 cell, too few for an interval.
* **Figure:** `fig_streamk_vs_true.png`, with TPR, FPR and wrong rate against σ for
  stream-K and `true`, one panel per arm-backend. The caption names the analyzer and
  both run directories.

## What would falsify, and what it would mean

* **SK1 reads *self-censoring* or *mixed*:** a correctly specified prior recovers a
  material part of `soft`'s detection. The fix for `soft` on a growing stream is
  then an outside estimate of K, from a census or a stream count, and not only a
  change of backend. That would weaken FINDINGS §3.14's "the prior cannot be the fix
  for a detector that misses".
* **SK2's wrong floor fails** (wrong < 0.05 at σ = 0.75): with the true missing mass,
  P0a with `no_repeat` is nearly safe on the crowded pool. Self-censoring was then
  the main cost in ST-3's norepeat arm, and §3.14's "no arm is both autonomous and
  safe" needs that qualifier.
* **SK2's direction fails:** the discrete gate's prior response is not the monotone
  knob ST-3 reported. That would need a trajectory read.
* **SK3 fails:** K was not already correct under the published arm; check the
  admission record.

## Note on what stays fixed

The stream-K prior is an oracle: no deployed loop knows the stream's species count.
It is an upper bound on what any estimate of K can buy, not a proposal. The
answerer, the admitted rows (always correct), the generator, ρ_real and the build
order (real objects first) are all ST-3's.
