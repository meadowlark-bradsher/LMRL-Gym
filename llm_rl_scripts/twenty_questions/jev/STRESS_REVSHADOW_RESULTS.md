# Reverse shadow: a hard view beside a playing `soft` — results

* **Prereg:** `STRESS_REVSHADOW_PREREG.md`, tag `prereg-stress-revshadow-20261116` (af49b73).
* **Build:** `STRESS_REVSHADOW_BUILD.md`.
* **Jira:** EE-41 (ENV-706).
* **Run:** jobq 26, 2026-11-16. 219 jobs on 6 workers in 134 min. Output in `results/stress_revshadow_20261116`.
* **Scoring:** `analyze_revshadow --run-dir results/stress_revshadow_20261116` produced `analysis.json` and
  `fig_revshadow.png`.

## In one paragraph

**The question.** The shadow study ran `soft` silently beside P0a. This one turns that around: `soft` plays, and a
hard survivor set rides along on `soft`'s own questions. Can that hard view stop `soft`'s wrong answers (the
*veto*), or flag a novel object early (the *alarm*), well enough to justify building it as a played arm?

**The answer is no, by the pre-registered rule.** The arm is built only if RS1 and RS3 both pass, and both fail.
The veto works where the pool of known objects stays small:
* it catches 87% of `soft`'s wrong answers on the ST-3 tail;
* it catches nearly all of them on look-alikes (ST-2).

It fails where the pool has grown crowded:
* it catches 29% on the `always` tail;
* in 68% of those wrong games, two or more known objects still fit every answer, so the hard view cannot decide
  either.

On the ST-3 tail the veto brings `soft`'s wrong rate from 0.435 to 0.056. That misses the 0.05 bar by 0.006.

**So what.** A hard check on `soft`'s questions is a strong filter while the candidates are few. A played arm would
send about as many tail games to the expert as `always` does. Each of those adds a new object to the pool, so the
arm would likely end up in the crowded regime, where the check stops working. P0a with `no_repeat` under `always`
is still the only arm that is both autonomous and safe on the tail.

**Distance from the main question.** This is a side branch. It concerns the belief store behind a questioner, not
why the negotiating AI never asks. Its contribution is the design point for the hard gate: deferring to the expert
on a contradiction is nearly free in a fixed pool, but it cannot separate look-alikes once the pool crowds.

## Terms

* **`soft`**: the soft posterior with ε = 0.05. It plays every game here, exactly as committed.
* **Hard view**: a survivor set over the known objects, kept with `HardMonotone`'s rule. A known is *struck* when
  one of `soft`'s answers contradicts it, and it is never revived. It changes nothing in the game.
* **Leader (`end_top`)**: `soft`'s top known object at the 20-question budget end.
* **Veto (G-end)**: at the budget end, answer the leader only if the hard view has not struck it; otherwise call
  the expert.
* **Alarm (G-alarm)**: call the expert as soon as the hard set empties.
* **Tolerance-1**: the same gates, striking a known only on its second contradicting answer.
* **Wrong**: an unconfirmed answer that is not the secret, over all games. **Safe** means wrong ≤ 0.05.
* **FPR**: in-pool games (the secret is a known object) that end with the expert, over in-pool games.
  **Autonomous** means FPR ≤ 0.20.
* **Tail**: σ = 0.75, flip 0, where most secrets are novel near copies of known objects. It is scored on two
  brackets:
  * **ST-3 bracket**: `soft` under ST-3's `top` rule. Its wrong answers are never corrected, so novel objects
    rarely enter the pool.
  * **`always` bracket**: the end-game study's `always` rule. Every budget-end game goes to the expert, who names
    the object, so the pool grows.

  A played arm's admissions would fall between the two.
* **Noisy ST-2**: the look-alike test with 5% or 10% of attribute answers flipped, pooled over look-alike distance d.

## Verdicts

| | prediction | scope | measured | verdict |
|---|---|---|---|---|
| RS0 | mechanical gate | all 219 jobs | identical, re-derived, no flip-0 violation | **PASS** |
| RS1 | veto catches ≥ 0.5 of `soft`'s wrong answers, in every scope | ST-3 tail | 0.871 (n = 6,525) | pass |
| | | `always` tail | **0.293** (n = 6,462) | **fail** |
| | | ST-2, d ≤ 2 | 0.997 (n = 2,364) | pass |
| | | **overall** | | **FAIL** |
| RS2 | raw alarm fires before a win in ≥ 0.10 of noisy ST-2 games, at flip 0.10 | noisy ST-2 | 0.234 (0.127 at flip 0.05) | **PASS** |
| RS3 | veto: wrong ≤ 0.05 and FPR ≤ 0.20, on both brackets | ST-3 tail | wrong **0.056**, FPR 0.002 | fail |
| | | `always` tail | wrong **0.305**, FPR 0.007 | fail |
| | | **overall** | | **FAIL** (neither bracket) |
| RS4 | alarm fires in ≥ 0.6 of novel games | ST-3 tail | 0.873 (n = 7,397) | **PASS** |

**Follow-on.** The played arm (`soft` chooses the questions, a hard view gates the answer) is **not built**. RS1 and
RS3 fail, and RS3 does not hold on even one bracket.

### RS0 in detail

* All 219 jobs replayed identically to their committed records: 147 ST-2, 36 ST-3 and 36 end-game.
* The hard set re-derived exactly from a standalone `HardMonotone` in all 10,950 checked games.
* At flip 0, no in-pool secret was ever struck, across 148,485 games.
* The `committed` outcome reproduces the prereg's references:

  | bracket | rule | wrong, counting right-unconfirmed answers | FPR |
  |---|---|---|---|
  | ST-3 tail | `top` | 0.439 | 0.002 |
  | `always` tail | `always` | 0 | 0.245 |

## What happened

### The veto works while the pool is small (RS1)

The catch rate is the share of `soft`'s wrong answers whose leader the hard view had struck. Flip 0 (from the
analysis cells):

| σ | ST-3 bracket | `always` bracket |
|---|---|---|
| 0.25 | 1.000 | 0.947 |
| 0.50 | 0.889 | 0.512 |
| 0.75 | 0.871 | **0.293** |

**The two brackets play the same secret stream; they differ in pool size.**
* On the ST-3 tail the pool ends a 5,000-game stream at about 290 known objects.
* On the `always` tail it ends at about 1,500–1,860, because every escalated novel object is admitted.

**In the crowded pool, the hard view is undecided.** Count the knowns still alive in the hard view at the end of
`soft`'s wrong games:

| bracket | none alive | one alive | two or more alive |
|---|---|---|---|
| ST-3 tail | 5,686 | 738 | 101 |
| `always` tail | 1,801 | 267 | **4,394 (68%)** |

So the veto's failure is not `soft` asking poor questions. Twenty questions cannot separate that many close
neighbours, and the leader is one of several that fit every answer.

**On the ST-3 tail, the veto and the alarm are the same gate.** Their outcomes agree to every digit. Whenever the
leader was struck, the whole hard set was empty: `soft`'s leader was always a hard survivor if any existed. The 13%
the veto misses are novel objects that answer every one of `soft`'s questions the way a known object would.

### The veto on look-alikes: safe for a small FPR (RS1, reported cells)

On ST-2 the veto removes `soft`'s wrong answers almost completely:

| scope | wrong, base → veto | FPR, base → veto |
|---|---|---|
| ST-2, d ≤ 2 | 0.0225 → 0.00006 | 0.004 → 0.029 |
| noisy ST-2, flip 0.10 | 0.034 → 0.000006 | 0.012 → 0.047 |

**The price is every right-but-unconfirmed answer.** The veto gives up 99.6% of them at d ≤ 2 (1,372 games) and
99.96% at flip 0.10.
* Those answers turned out right even though an answer had contradicted the secret, so the hard view had struck it.
* At flip 0 this never happens (RS0(c)). Under noise it is nearly the whole class.
* The veto therefore behaves almost like "never answer unconfirmed", except where the leader was never contradicted.

### The raw alarm is too costly under noise (RS2)

In noisy ST-2, the alarm fires before `soft`'s winning guess in 23.4% of won in-pool games at flip 0.10, and 12.7%
at flip 0.05. That pushes FPR to 0.269 at flip 0.10, against the veto's 0.047. The tolerance-1 alarm is cheaper
(2.5% before a win), but it catches far less: its veto catches 0.113 on the ST-3 tail and 0.512 on ST-2 at d ≤ 2.

### The alarm does see novel objects (RS4)

On the ST-3 tail the alarm fires in 87.3% of novel games. That is above P0a's own 0.716 with `no_repeat` in the
end-game study, measured on a different arm's questions. It is a median of 5 questions before `soft`'s end
(quartiles 3–8). On the `always` tail it fires in only 47.8% (reported, not scored), for the same crowding reason.

### RS3: close on one bracket, far on the other

| bracket | wrong: base → veto | FPR: base → veto |
|---|---|---|
| ST-3 tail | 0.435 → **0.056** | 0.002 → 0.002 |
| `always` tail | 0.431 → **0.305** | 0.002 → 0.007 |

Both brackets stay well inside the FPR bar.
* **The ST-3 tail fails safety by 0.006.** The miss is the 13% of wrong answers the hard view never contradicted.
* **The `always` tail fails by a wide margin.**
* **At σ = 0.5 both would pass**, though it is not scored: wrong 0.011 and 0.042.

### Reported, not scored

* **`G-both`** (alarm, then veto) is identical to the veto on the ST-3 tail. On ST-2 it inherits the alarm's FPR
  (0.150 at d ≤ 2; 0.269 at flip 0.10). On the `always` tail it equals the veto.
* **Tolerance-1** is weaker throughout. Its veto's wrong rate is 0.386 on the ST-3 tail and 0.012 on noisy ST-2 at flip 0.10.
* **Per-cell numbers** by σ, d and flip are in `analysis.json` under `cells`.

## What this means

* **The prereg's reading of an RS1 failure holds only in part.** It said a failure would mean `soft`'s near-copy
  misses are invisible to a hard check on its own questions. That is true in the crowded pool. In the sparse pool,
  the hard check catches 87%, and on look-alikes nearly 100%.
* **A played arm would likely be in the crowded pool.** On the ST-3 tail the veto sends 44% of games to the expert,
  close to `always`'s 51%, and each escalated novel object is admitted.
  * So the `always` bracket is probably the closer guide to a played arm.
  * By it, the veto leaves about 0.30 of games wrong.
  * This is an inference from the two brackets; the played arm was not run.
* **P0a with `no_repeat` under `always` remains the only tail arm that is both autonomous and safe** (wrong 0, FPR
  0.099, end-game study). Its edge here is that it chooses questions to empty the hard set. `soft`'s questions do
  not, and in a crowded pool that difference decides.
* **On look-alikes in a fixed pool, a hard veto on `soft` is a cheap safety layer.** Wrong falls to about 0 for an
  FPR near 0.03–0.05. This supports the case that the hard gate's job is deferring to the expert. It is not only an
  audit trail.
* **The raw alarm should not run under noise without P0a's bounded reinstate.** That is the guard RS2 anticipated.

## Provenance

* Every number above is from `results/stress_revshadow_20261116/analysis.json`, written by `analyze_revshadow.py`.
  That covers the verdicts, `scopes` and `cells`, and the end-of-game survivor counts (`surv_end_in_wrong`).
* **Two numbers are not in `analysis.json`:**
  * **Pool sizes:** the last game's `k_before` in the σ = 0.75, flip-0 job records under `jobs/`, for the ST-3 and
    end-game brackets.
  * **The `always` rule's escalation share (51%):** its `committed` expert rate.
* **`fig_revshadow.png`:** RS1 catch rate by σ for each bracket, and RS3 wrong against FPR on the tail. Its caption
  names the analyzer and the run directory.
