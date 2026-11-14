# ST-3 stream-K follow-up — results

**Scored against:** `STRESS_ST3_STREAMK_PREREG.md`, as tagged
`prereg-stress-st3-streamk-20261113` (commit b94802e). The build note is
`STRESS_ST3_STREAMK_BUILD.md`.

**Run:** `results/stress_st3_streamk_20261113`, 108 jobs × 5000 games. It ran as
jobq 57 on 6 workers under `taskpolicy -c utility` and took 16 min. Provenance is
clean: the registered commit, Python 3.9, `PYTHONHASHSEED=0`, BLAS at 1 thread.

**Pairing:** each job is compared with its ST-3 `true` job in
`results/stress_st3_20261113`. The pair shares the stream, the objects, the oracle
seed and the backend; only the K inside the prior differs. ST-3 was not re-run.

**Scoring:** `analyze_st3_streamk` writes `analysis.json` and
`fig_streamk_vs_true.png`. After the run, only the figure's y-label and caption
layout were changed; the scoring code is unchanged.

All numbers are deterministic replays of seeded jobs. "P0a" is `modeswitch_guard`.
The stream-K prior is an oracle: it gives the gate the stream's real missing mass,
which no deployed loop knows.

## Verdicts

| | prediction | verdict | in one line |
|---|---|---|---|
| SK0 | at σ = 0 the stream-K prior plays exactly the `true` games | **PASS** | 27/27 jobs identical |
| SK1 | `soft`'s collapse is the near-copy likelihood, not self-censoring | **PASS** (*likelihood*) | TPR gain +0.005 at σ = 0.5, −0.012 at σ = 0.75 |
| SK2 | norepeat P0a moves along the escalate-vs-wrong knob and stays unsafe | **FAIL** | the direction parts fail, by at most 0.02; the wrong floor holds (0.111 at σ = 0.75) |
| SK3 | control: published P0a unchanged at σ = 0.75 | **PASS** | every column identical |

## Headline

1. **Correcting the missing mass does not rescue `soft`.**
   * Under the stream-K prior, `soft`'s last-window prior at σ = 0.75 is 0.265,
     against 0.049 under ST-3's `true` prior. That is 5.4× higher, and correct.
   * Its TPR still goes 0.119 → 0.107, and its wrong rate 0.439 → 0.449. On the
     generated children the TPR goes 0.059 → 0.047.
   * At σ = 0.5 (0.060 against 0.030), the TPR goes 0.335 → 0.340.
   * The three stream seeds agree in sign at σ = 0.75: per-job TPR changes run
     −0.017 to −0.007.
   * So `soft`'s collapse on the endless tail is the near-copy likelihood: the parent
     of a generated child explains the answers better than the background. Too low a
     count of what it has missed is not the cause.
   * FINDINGS §3.14's "the prior cannot be the fix for a detector that misses" now
     holds even for the oracle prior.
2. **The correct prior also leaves P0a with `no_repeat` where it was.** All rows are
   flip 0:

   | σ | prior | last-window prior | TPR | FPR | wrong | won |
   |---:|---|---:|---:|---:|---:|---:|
   | 0.5 | `true` | 0.050 | 0.753 | 0.000 | 0.034 | 0.875 |
   | 0.5 | stream-K | 0.060 | 0.734 | 0.000 | 0.037 | 0.874 |
   | 0.75 | `true` | 0.203 | 0.716 | 0.068 | 0.115 | 0.578 |
   | 0.75 | stream-K | 0.265 | 0.721 | 0.068 | 0.111 | 0.581 |

   * Every change is ≤ 0.02, and inside the per-job ranges, which span zero.
   * ST-3's knob, `fixed` (≈ 0.04) against `true` (≈ 0.20), moved the wrong rate
     from 0.247 to 0.115. A further step to the correct ≈ 0.27 buys nothing.
   * P0a with `no_repeat` is still wrong in 0.111 of games at σ = 0.75. The cost is
     the survivor-against-UNKNOWN decision on near copies, not self-censoring.
3. **The higher, correct prior *lowers* `soft`'s AUC at σ = 0.75**: 0.744 → 0.641 at
   flip 0, and 0.685 → 0.604 at flip 0.05.
   * TPR and FPR barely move (FPR 0.002 → 0.002 at flip 0), so the ranking change
     sits in `p_unknown_max` scores that do not cross the gate.
   * A plausible reading, not checked: the censored prior fell over the stream while
     the stream-K prior stays high. That changes how in-pool and novel scores from
     different parts of the stream interleave in a pooled AUC.
   * Not scored. Like ST-2's P2, it argues for scoring TPR and wrong rates, not
     pooled AUC.

## Per prediction

### SK0: mechanical gate (PASS)

All 27 σ = 0 jobs, covering 3 arm-backends × 9 seed combinations, have per-game
columns identical to their ST-3 pair.

### SK1: `soft` (PASS, reading *likelihood*)

* **Gain = TPR(stream) − TPR(true)** for primary `soft`, flip 0, all games:
  * **+0.005** at σ = 0.5 (0.340 against 0.335);
  * **−0.012** at σ = 0.75 (0.107 against 0.119).
* Both gains are ≤ 0.05, which is the *likelihood* reading.

| σ | flip | TPR stream / true | wrong stream / true | children TPR stream / true | admissions stream / true |
|---:|---:|---|---|---|---|
| 0.25 | 0 | 0.787 / 0.794 | 0.012 / 0.012 | 0.523 / 0.536 | 675 / 677 |
| 0.5 | 0 | 0.340 / 0.335 | 0.099 / 0.101 | 0.163 / 0.159 | 756 / 755 |
| 0.75 | 0 | 0.107 / 0.119 | 0.449 / 0.439 | 0.047 / 0.059 | 799 / 879 |
| 0.75 | 0.05 | 0.135 / 0.126 | 0.445 / 0.450 | 0.075 / 0.066 | 1955 / 1844 |

### SK2: norepeat P0a (FAIL)

| part | holds? | stream vs true |
|---|---|---|
| σ = 0.5, TPR up | no | 0.734 vs 0.753 |
| σ = 0.5, wrong down | no | 0.037 vs 0.034 |
| σ = 0.5, FPR up | yes | 0.000 both |
| σ = 0.75, TPR up | yes | 0.721 vs 0.716 |
| σ = 0.75, wrong down | yes | 0.111 vs 0.115 |
| σ = 0.75, FPR up | no | stream slightly below `true`; both 0.068 to three decimals |
| σ = 0.75, wrong ≥ 0.05 | yes | 0.111 |

* The registered direction parts had no tolerance, and the changes are
  sampling-sized: per-job TPR changes run −0.048 to +0.003 at σ = 0.5, and −0.018 to
  +0.024 at σ = 0.75.
* The monotone knob ST-3 saw between `fixed` and `true` does not continue to the
  correct prior. Between ≈ 0.20 and ≈ 0.27 the gate is flat.
* At flip 0.05 the same arm moves slightly in the predicted direction at σ = 0.75:
  TPR 0.858 → 0.880, wrong 0.064 → 0.054. Reported, not scored.

### SK3: control (PASS)

Published P0a at σ = 0.75 has every per-game column identical to ST-3's at both
flips. It already admits every novel object, so |H_known| equals the stream's K.
At σ = 0.5 it moves slightly (FPR 0.429 → 0.442, won 0.505 → 0.493; reported).

## What this changes

* FINDINGS §3.14's main reading stands, and is now tested: on an endless tail of
  near copies, self-censoring is real, but it is not what breaks `soft` or what
  leaves P0a with `no_repeat` unsafe. Even with the true missing mass, `soft` is
  wrong in 0.45 of games at σ = 0.75, and P0a with `no_repeat` in 0.11.
* The lever is the decision on near copies: whether a known survivor or UNKNOWN
  wins at the end of a game. Better estimates of K, by census, by stream count or by
  an outside model, should not be expected to fix it.

## Not tested; next steps

The two other ST-3 follow-ups remain:

* the interleaved build, for P1's composition confound;
* AUC matched on composition, for P5.

A third direction now looks more relevant than either: a near-copy-aware end-of-game
decision. One version escalates when the leading known and UNKNOWN are within some
margin after the budget, instead of guessing the survivor.
