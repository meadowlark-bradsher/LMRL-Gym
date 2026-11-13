# ST-3 Endless tail — results

Scored against `STRESS_TAIL_PREREG.md` as tagged `prereg-stress-st3-20261113`
(commit 556df8e). Build note: `STRESS_ST3_BUILD.md`. Run directory:
`results/stress_st3_20261113`: 648 jobs × 5000 games, jobq 47, 6 workers under
`taskpolicy -c utility`, 122 min wall. Provenance is clean: registered commit, Python
3.9, `PYTHONHASHSEED=0`, BLAS pinned to 1 thread.

* **Registered scoring:** `analyze_st3` writes `analysis.json` and two figures. After
  the run, only the x-axis ticks of `fig_auc_by_pool_size.png` were changed, for
  legibility; the scoring code is unchanged.
* **Exploratory, not registered:** `st3_composition` writes `composition.json`, and
  is labelled as such below.

All numbers are deterministic replays of seeded jobs. A cell pools its 3 stream seeds,
and under noise its 3 × 2 stream × noise seeds. "P0a" is `modeswitch_guard`. At
flip 0, norepeat `hard` and norepeat `modeswitch_guard` are game-for-game identical;
the guard only matters under noise.

## Verdicts

| | prediction | verdict | in one line |
|---|---|---|---|
| P1 | expert load tracks the missing mass (σ ≤ 0.5) | **FAIL** | holds at σ = 0 and for the discrete gate under noise; detection falls in the second half for `soft` from σ = 0.25 and for every backend at σ = 0.5 |
| P2 | without the loop, the discrete gate ignores the prior | **FAIL** | holds at σ ≤ 0.25; at σ = 0.75 the prior trades escalations against wrong guesses |
| P2b | in the published arm, the prior reaches the gate through the loop | **PASS** | FPR 0.998 under `true` against 0.728 under `fixed` |
| P3 | the soft gate depends on the prior | **FAIL** | soft TPR is about the same under every prior (0.12 at σ = 0.75) |
| P4 | the online estimate is usable | **FAIL** | σ̂ is biased low, because it is fitted only on admissions the loop itself made |
| P5 | at matched pool size, σ does not change AUC | **FAIL** | AUC is monotone in σ, but within a pool-size bin it differs by up to 0.24 |

## Headline

1. **Missed novelty cuts its own evidence.** The loop learns about novelty only from
   its own expert calls. They set |H_known|, the K in every prior arm, `true`
   included, and they are the data the online σ̂ is fitted on. A backend that misses
   novel objects therefore under-counts K and sees a lower prior. That is why P4
   fails: σ̂ at game 2000 is 0.12 at σ = 0.5 and 0.00 at σ = 0.75 on the soft arm,
   0.37 and 0.57–0.67 on the P0a arm. It also confounds P3: on the soft arm the
   `true` prior ends at 0.049 at σ = 0.75, against a stream missing mass of 0.28.
   Species the loop never admitted come back as novel games: 1409 of the 3554 novel
   games in soft's second half at σ = 0.75.
2. **`soft` fails on near-duplicates, not on real novel objects.** Exploratory split
   (`composition.json`, true prior, flip 0):
   * Real objects arriving novel are flagged 0.99–1.00 of the time by every backend
     at every σ.
   * Tail children are the generator rows that copy a known parent with 23% of
     cells resampled. `soft` flags only 0.73 of them at σ = 0.25 (first half), and
     0.10 at σ = 0.75.
   * `soft` instead names the parent. Its wrong-answer rate is 0.44 of all games at
     σ = 0.75, against 0.115 for norepeat P0a.
   * The priors differ by a factor of about 3 on this arm and hardly move soft's TPR
     (P3), so the likelihood of the near-duplicate parent overwhelms the prior.
3. **P1's drift is partly by construction.** `build` puts the 158 real objects first,
   so the first half's novel games are mostly real objects and the second half's are
   all tail children. For σ ≥ 0.25 the TPR difference between halves mixes a change
   in *what* is novel with a change over time. Within tail children alone, the TPR
   still falls between halves: 1.00 → 0.95 for P0a at σ = 0.25, and 0.83 → 0.50 at
   σ = 0.5. As more siblings are admitted, a child more often survives elimination
   next to a known relative.
4. **No arm both escalates sparingly and stays safe at σ ≥ 0.5.** Shares of all
   games, true prior, flip 0:

   | σ | backend | won | expert | wrong |
   |---:|---|---:|---:|---:|
   | 0.5 | `soft` | 0.849 | 0.050 | 0.101 |
   | 0.5 | P0a, norepeat | 0.875 | 0.091 | 0.034 |
   | 0.5 | P0a, published (loop on) | 0.505 | 0.486 | 0.009 |
   | 0.75 | `soft` | 0.502 | 0.060 | 0.439 |
   | 0.75 | P0a, norepeat | 0.578 | 0.307 | 0.115 |
   | 0.75 | P0a, published (loop on) | 0.002 | 0.998 | 0.000 |

   The repeat loop keeps P0a safe at σ = 0.75 only by escalating every game
   (FPR 0.998). Removing it brings back autonomy at a wrong rate of 0.115. At σ ≤ 0.25
   every backend is fine. P0a: won 0.947, wrong 0.001 at σ = 0.25.
5. **For the discrete gate the prior is a knob between escalating and guessing wrong
   (P2).** At σ = 0.75, norepeat P0a:

   | prior | TPR | FPR | wrong |
   |---|---:|---:|---:|
   | `true` | 0.716 | 0.068 | 0.115 |
   | `fixed` | 0.435 | 0.000 | 0.247 |

   When a novel child's known parent is still in the live set at the end of a game,
   the gate weighs the UNKNOWN mass against the survivor. A higher prior escalates,
   and a lower one guesses the survivor. P2's premise was that the call fires only
   when the live set empties, and that is false once the pool is crowded. At σ ≤ 0.25
   P2 holds exactly (spreads ≤ 0.001). At σ = 0.5 the TPR spread is 0.022, just over
   the 0.02 tolerance.

## Per prediction

### P1: expert load against the stream's missing mass (FAIL)

True prior. `ratio` is (necessary-call rate h2/h1) ÷ (stream p_new h2/h1); a part
holds when |ratio − 1| ≤ 0.25 and |ΔTPR| ≤ 0.05.

| backend | flip | σ | TPR h1 → h2 | ratio | parts |
|---|---:|---:|---|---:|---|
| `soft` | 0, 0.05 | 0 | 1.000 → 1.000, 0.996 → 1.000 | 1.18 | pass |
| `soft` | 0 | 0.25 | 0.926 → 0.366 | 0.44 | fail, fail |
| `soft` | 0 | 0.5 | 0.549 → 0.039 | 0.12 | fail, fail |
| norepeat `hard` / P0a | 0, 0.05 | 0 | 1.000 → 1.000 | 1.18 | pass |
| norepeat `hard` / P0a | 0 | 0.25 | 1.000 → 0.946 | 0.80 | TPR fails by 0.004; ratio passes |
| norepeat `hard` / P0a | 0.05 | 0.25 | 1.000 → 0.956 | 0.80 | pass |
| norepeat `hard` / P0a | 0 | 0.5 | 0.897 → 0.495 | 0.70 | fail, fail |
| norepeat `hard` / P0a | 0.05 | 0.5 | 0.908 → 0.592 | 0.79 | TPR fails; ratio passes |

At flip 0.05, `soft` at σ = 0.25 and 0.5 is within 0.02 of its flip-0 row. At
σ = 0.75 (reported, not scored), the norepeat discrete TPR goes 0.80 → 0.60 with a
ratio of 0.78, and `soft` goes 0.21 → 0.02.

`fig_expert_load_vs_missing_mass.png` shows the windows: the discrete calls follow the
stream's p_new at σ ≤ 0.25 and fall below it at σ ≥ 0.5. `soft` falls below it from
σ = 0.25.

### P2: discrete gate against the prior, norepeat, flip 0 (FAIL)

Max − min across `true` / `online` / `fixed`:

| σ | TPR spread | FPR spread |
|---:|---:|---:|
| 0, 0.25 | ≤ 0.001 | 0 |
| 0.5 | 0.022 | 0 |
| 0.75 | 0.281 | 0.068 |

`hard` and P0a are identical here.

### P2b: the loop carries the prior, published arm, σ = 0.75, flip 0 (PASS)

* FPR is 0.998 under `true` and 0.728 under `fixed`, a difference of 0.27, for both
  `hard` and P0a.
* The pre-registration expected the fixed prior (about 0.07) to stay below the
  loop's 0.15 threshold. It did: 0.089 at window 9 and 0.068 at window 19. Yet the
  repeat rate under `fixed` is still 0.80 in the first half and 0.64 in the second.
* So at pools of this size the loop engages well below the 0.15 found in ST-0's
  snapshot study (pool of 158).

### P3: soft gate against the prior (FAIL)

* **(a)** At σ = 0.75, TPR is 0.119 under `true` and 0.114 under `fixed` (flip 0),
  and 0.126 against 0.121 at flip 0.05. Both fail.
* **(b)** At σ = 0, FPR is 0.000 under both priors at flip 0, which fails;
  0.001 > 0.001 holds at flip 0.05. `soft` raises almost no false escalations at
  σ = 0 under any prior.
* Caveat: see Headline 1. On this arm the `true` prior is computed from soft's own
  under-counted K, so the gap between priors that P3 meant to test (≈ 0.28 against
  ≈ 0.016) was only ≈ 0.05 against 0.016 in the run.

### P4: online fit (FAIL)

| σ | σ̂ at game 2000, `soft` | σ̂ at game 2000, P0a norepeat | soft TPR from game 2000, online vs true |
|---:|---|---|---|
| 0.25 | 0.283 / 0.267 | 0.300 / 0.300 (boundary, passes) | 0.375 vs 0.408 (fails), 0.401 vs 0.412 |
| 0.5 | 0.117 / 0.117 | 0.367 / 0.375 | 0.044 vs 0.041, 0.062 vs 0.058 |
| 0.75 | 0.000 / 0.000 | 0.567 / 0.667 | 0.014 vs 0.024, 0.025 vs 0.027 |

Pairs are flip 0 / flip 0.05. The soft TPR parts pass except at σ = 0.25, flip 0, but
only because soft's TPR is low under every prior.

How far the online prior falls behind depends on how much novelty the arm catches. In
the last window at σ = 0.75:

| arm | novel objects caught | online prior | `true` prior |
|---|---|---:|---:|
| published `hard` (loop on) | all | 0.263 | 0.265 |
| norepeat P0a | 0.72 | 0.141 | 0.203 |
| `soft` | 0.12 | 0.010 | 0.049 |

### P5: AUC against σ at matched pool size (FAIL)

* **Part (a) holds for all three backends.** AUC falls with σ: 1.000 / 0.990 / 0.842 /
  0.744 for `soft`, and 1.000 / 0.992 / 0.846 / 0.728 for norepeat `hard` and P0a.
* **Part (b) fails.**

  | backend | bin (\|H_known\|) | AUC at σ = 0.25 / 0.5 / 0.75 |
  |---|---|---|
  | `soft` | [100, 200) | 1.000 / 0.991 / 0.942 |
  | `soft` | [200, 400) | 0.991 / 0.835 / 0.753 |
  | discrete | [100, 200) | 1.000 / 1.000 / 0.976 |
  | discrete | [200, 400) | 0.975 / 0.835 / 0.874 |
  | discrete | [400, 800) | 0.751 (σ = 0.5) / 0.750 (σ = 0.75) |

  The [400, 800) bin is the one place where matched pools agree.
* **Not the pre-registration's suggested cause.** It pointed to background-marginal
  refreshes. The composition (Headline 2) points to something simpler: at higher σ, a
  bin's novel games are mostly tail children whose parent is already known, and at
  lower σ more of them are real objects. Pool size alone does not fix the share of
  novel games that have a known near-duplicate. This is untested here.
* **Comparison with ST-1** ("size, not packing"). In ST-1 the generated pool was
  admitted all at once. Here the novel class itself is drawn as relatives of the
  pool, so the two results do not contradict each other.
* `fig_auc_by_pool_size.png` shows the bins.

## What this changes

* P0a stays the recommended backend at σ ≤ 0.25: won 0.947, wrong 0.001, and the
  expert load follows the missing mass.
* At σ ≥ 0.5 none of the six arm-backends is both autonomous and safe. With the loop
  on, P0a buys safety by escalating everything; without it, wrong answers come back.
  Neither changing the prior nor fitting it online fixes this, because the failure is
  the survivor-versus-UNKNOWN decision on near-duplicates, and with self-censored
  K the prior cannot see the problem.
* `soft` should not be used on a growing stream: wrong 0.10 at σ = 0.5, 0.44 at 0.75.

## Not tested; possible next steps

* **Stream-K arm.** Feed the `true` prior the stream's K instead of |H_known|. This
  would separate self-censoring from near-duplicate likelihood in `soft`'s collapse
  and in P3.
* **Interleaved build.** Real objects and tail children mixed in random order, which
  removes the composition confound in P1.
* **Matched-composition AUC.** Bin by |H_known| *and* by whether the novel object's
  parent is known, which tests the reading of P5 given above.
