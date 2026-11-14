# ST-3 end-game follow-up — results

**Scored against:** `STRESS_ST3_ENDGAME_PREREG.md`, tagged
`prereg-stress-st3-endgame-20261114` (commit e633513). The build note is
`STRESS_ST3_ENDGAME_BUILD.md`.

**Run:** `results/stress_st3_endgame_20261114`, 324 jobs × 5000 games.
* It ran as jobq 81 on 6 workers under `taskpolicy -c utility` and took 74 min.
* Provenance is clean: the registered commit with a clean tree, Python 3.9.21,
  `PYTHONHASHSEED=0`, BLAS at 1 thread.

**Pairing:** each job is compared with its ST-3 `true` job in
`results/stress_st3_20261113`. The two share the stream, the objects, the oracle seed
and the backend. Only the decision taken when the 20 questions run out differs. ST-3
was not re-run: its committed cells are the `top` rule.

**Scoring:**
* `analyze_st3_endgame` (unchanged since the tag) writes `analysis.json` and
  `fig_endgame_rules.png`.
* The split of won and FPR changes into *crowded* and *newly in-pool* games is
  exploratory, not registered. It comes from `st3_endgame_crowding`, which reads the
  same per-game columns and writes `crowding.json`.

**Terms:**
* "P0a" is `modeswitch_guard`. "Published P0a" is the primary arm; "P0a with
  `no_repeat`" is the norepeat arm.
* *Novel* means the secret is not yet in the known pool.
* *TPR* is the share of novel games sent to the expert.
* *FPR* is the share of in-pool games sent to the expert, i.e. unnecessary calls.
* *wrong* is the share of all games that end with an unconfirmed known answer.
* *won* is the share of all games ending in a confirmed correct guess.
* **Autonomous** means FPR ≤ 0.20; **safe** means wrong ≤ 0.05.
* Every number is a deterministic replay of seeded jobs. Rows are flip 0, σ = 0.75,
  unless marked.

## In plain terms

* **Question.** When the loop runs out of questions without a confirmed guess, ST-3
  answered with its best known guess unless UNKNOWN led. Every ST-3 wrong answer came
  from that one decision. If the loop calls the expert there more often, does some
  backend become both autonomous and safe on the hardest tail (σ = 0.75), and what do
  the extra admissions cost?
* **Finding.**
  * **Yes for P0a with `no_repeat`.** Calling the expert whenever the budget runs out
    takes its wrong rate from 0.115 to 0. Its unnecessary-call rate goes from 0.068 to
    0.099, and its won rate rises slightly, from 0.578 to 0.588. It is the first arm
    in the ST-3 line to be both autonomous and safe at σ = 0.75.
  * **No for `soft`.** The same rule makes it safe (wrong 0.439 → 0), but its
    unnecessary-call rate goes to 0.245. The extra admissions *crowd* it: 0.107 of
    all games are in-pool games it used to win that now run out of budget.
* **So what.** The lever named by the stream-K follow-up was the right one. With it,
  the survivor backend's safety on the endless tail is a decision rule at the end of
  the game. The near-copy weakness that remains is `soft`'s. Given the *same* final
  pool, `soft` sends 0.245 of in-pool games to the expert and P0a with `no_repeat`
  sends 0.099.
* **Distance from the main question** (why the AI never asks, and what fixes it).
  This is the 20Q loop, one step away from negotiation. Here "asking" means calling
  the expert. The result is another case of *rules decide the move*: a fixed rule for
  when to ask removes every unsafe answer at a small, measured cost, where the
  belief alone (ST-3, stream-K) did not.

## Verdicts

| | prediction | verdict | in one line |
|---|---|---|---|
| EG0 | mechanical gate | **PASS** | 324/324 jobs pair with ST-3 up to the first escalated budget-end game; `always` has wrong 0 and TPR 1 everywhere; published P0a at σ = 0.75 is identical under every rule (27/27) |
| EG1 | under `always`, both arms autonomous at σ = 0.5 and 0.75 | **FAIL** | P0a with `no_repeat` passes (FPR 0.012, 0.099); `soft` passes at σ = 0.5 (0.041) and fails at σ = 0.75 (0.245) |
| EG2 | P0a with `no_repeat`: `half` unsafe, `conf` safe and autonomous | **PASS** | `half` wrong 0.065; `conf` wrong 0, FPR 0.099 (at σ = 0.75 `conf` *is* `always`; see below) |
| EG3 | admission pays for `soft` (won gain ≥ +0.05) | **FAIL**, reading *neutral* | gain −0.009: +0.097 from newly in-pool repeats, −0.107 from crowded games |
| EG4 | `always` gives up ≤ 0.02 correct unconfirmed answers | **FAIL** | `soft` at σ = 0.75 gives up 0.022; the other three cells 0.001–0.012 |
| EG5 | the rules order wrong and FPR | **PASS** | all four cells monotone within the 0.01 tolerance |

## Headline

1. **P0a with `no_repeat` becomes autonomous and safe at σ = 0.75 under `conf` or
   `always`.**

   | rule | wrong | FPR | TPR | won | admitted per job |
   |---|---:|---:|---:|---:|---:|
   | `top` (ST-3) | 0.115 | 0.068 | 0.716 | 0.578 | 1317 |
   | `half` | 0.065 | 0.084 | 0.832 | 0.581 | 1506 |
   | `conf` | 0.000 | 0.099 | 1.000 | 0.588 | 1737 |
   | `always` | 0.000 | 0.099 | 1.000 | 0.588 | 1737 |

   * Per job, against the `top` pair, `always` changes:
     * wrong by −0.130 to −0.092;
     * FPR by +0.018 to +0.038;
     * won by +0.009 to +0.010.

     All three stream seeds agree in sign.
   * The won gain is real but small. The 0.021 of games that become in-pool win
     0.009, and 0.012 of games are rescued against 0.011 crowded.
   * Under noise (flip 0.05, reported, not scored) the FPR is 0.314 under `always`.
     As the prereg expected, the arm is not autonomous under noise, whatever the end
     rule: ST-3's `top` was already at 0.277.
2. **`soft` is made safe but not autonomous: the admissions crowd it.**

   | rule | wrong | FPR | TPR | won | in-pool share | won given in-pool |
   |---|---:|---:|---:|---:|---:|---:|
   | `top` (ST-3) | 0.439 | 0.002 | 0.119 | 0.502 | 0.507 | 0.990 |
   | `half` | 0.042 | 0.215 | 0.888 | 0.497 | 0.635 | 0.783 |
   | `conf` | 0.009 | 0.239 | 0.975 | 0.493 | 0.648 | 0.760 |
   | `always` | 0.000 | 0.245 | 1.000 | 0.493 | 0.653 | 0.755 |

   * Under `always`, `soft` admits every species it sees: 1737 per job against 293
     under `top`.
   * Pairing each game with its `top` twin (`crowding.json`):
     * **Newly in-pool, 0.146 of all games.** These are repeats of species only the
       new rule admitted. `soft` wins 0.097 of them, which is most of the "admission
       pays" effect the prereg expected.
     * **Crowded, 0.107 of all games.** In-pool under both rules and won under `top`,
       they now run out of budget and go to the expert. The admitted near copies make
       their parents' games and their siblings' games harder to finish in 20
       questions.
     * The two nearly cancel, giving won −0.009 (EG3 *neutral*).
     * In-pool games sent to the expert are 0.160 of all games, so FPR is 0.245 of an
       in-pool share of 0.653. Of that 0.160, 0.107 is crowded and 0.049 is newly
       in-pool games that are not won.
   * At the budget end `soft`'s belief is nearly flat: the leading known's mass has a
     median of 0.058 in in-pool games (0.9 quantile 0.12).
   * `half` comes closest: wrong 0.042 (safe), FPR 0.215 (0.015 over the line). No rule
     makes `soft` both at σ = 0.75.
   * At σ = 0.5 every new rule makes `soft` both autonomous and safe. Under `always`,
     FPR is 0.041 and wrong 0; crowding there is 0.031 of games.
3. **Same pool, different backend.**
   * Under `always`, every novel game is sent to the expert, and admitted rows are
     always correct. So all three arm-backends end with the same pool: every species
     the stream showed, 1737 per job at σ = 0.75.
   * On that common pool, the FPR is:
     * 0.245 for `soft`;
     * 0.099 for P0a with `no_repeat`;
     * 0.998 for published P0a.
   * The gap between `soft` and P0a with `no_repeat` is the cost of `soft`'s soft
     likelihood on near copies. The pool is the same, so it is not a cost of what was
     admitted. This agrees with stream-K's SK1 reading *likelihood*.

## Per prediction

### EG0: mechanical gate (PASS)

* (a, b) All 324 jobs match their ST-3 pair in every ST-3 column up to the first game
  the rule ends differently. That game is a budget-end `wrong` the rule escalated.
* (c) Under `always`, wrong is 0 and TPR is 1 in all 108 jobs, at both flips.
* (d) All 27 published-P0a jobs at σ = 0.75 are identical to ST-3 under every rule.
  There is no budget-end `wrong` to change: 0.99 of its games reach the budget end,
  and in none of them does a known lead.

### EG1: autonomous under `always` (FAIL)

| arm-backend | σ = 0.5 FPR | σ = 0.75 FPR |
|---|---:|---:|
| `soft` | 0.041 | **0.245** |
| P0a, `no_repeat` | 0.012 | 0.099 |

* The prereg's first-order estimate was FPR 0.010 for `soft` and 0.085 for P0a with
  `no_repeat` at σ = 0.75. That estimate escalates only the ST-3 in-pool `wrong`
  games.
* It named the risk: "the extra admissions crowd the pool until in-pool games run
  out of budget". That is what happened to `soft`, at 0.107 of games.
* P0a with `no_repeat` already admitted 1317 species per job under `top`, so its pool
  grew only from 1317 to 1737. `soft`'s grew sixfold.
* The falsification text applies to `soft` as written: "escalating at the end is not
  free". The lever for `soft` moves to the planner or the question budget. It does not
  move for P0a with `no_repeat`.

### EG2: which end rule P0a with `no_repeat` needs (PASS)

* `half`: wrong 0.065 > 0.05, so unsafe, as predicted. The prereg's first-order
  estimate was 0.076.
* `conf`: wrong 0, FPR 0.099. Safe and autonomous.
* **At σ = 0.75, `conf` and `always` are the same rule on this backend.**
  * On the survivor backends, `conf` escalates a sole survivor when the UNKNOWN mass
    u > 0.1. At every budget-end game in these runs u ≥ 0.233, at both flips.
  * So the `conf` and `always` jobs are identical, record for record. The
    prereg's worry for `conf` (sole survivors wrong early in the stream, while u is
    still < 0.1) cannot arise at σ = 0.75 under the `true` prior.
  * At σ = 0.5 the two differ: u ≤ 0.1 at 0.51 of budget-end games. There `conf` leaves
    wrong 0.011 and `always` 0, with FPR 0.008 against 0.012.

### EG3: does admission pay for `soft`? (FAIL, *neutral*)

* won(`always`) − won(`top`) = −0.009. The per-job range is −0.038 to +0.011.
* The neutral net hides two effects of about 0.1 each:
  * +0.097 from repeats of newly admitted species;
  * −0.107 from crowded in-pool games, plus 0.001 rescued.
* The prereg's basis was half right. The repeats did become winnable, at a 0.66 win
  rate given newly in-pool. The prereg missed that the near copies would take as
  many wins from their parents.

### EG4: right answers given up by `always` (FAIL)

| arm-backend | σ = 0.5 | σ = 0.75 |
|---|---:|---:|
| `soft` | 0.012 | **0.022** |
| P0a, `no_repeat` | 0.004 | 0.001 |

* These are budget-end games where the secret led and `always` escalated. `top` would
  have answered them correctly but unconfirmed.
* `soft` exceeds 0.02 by 0.002, at σ = 0.75 only.
* These correct answers came out of a nearly flat belief (in-pool median p_top
  0.058). `conf` gives up almost the same share, 0.0219, so no threshold rule that
  answers only when confident would keep them. They are correct by tie-break luck,
  not by confidence. Against wrong answers at 0.439 under `top`, the trade is lopsided.

### EG5: the rules order the outcomes (PASS)

* For `soft` and P0a with `no_repeat` at σ = 0.5 and 0.75, wrong falls and FPR rises
  from `top` to `half` to `conf` to `always`.
* No step breaks the order, even within the 0.01 tolerance.

## Reported, not scored

* **Flip 0.05, σ = 0.75**, under `always`:
  * `soft`: FPR 0.331, wrong 0;
  * P0a with `no_repeat`: FPR 0.314, wrong 0.

  Neither is autonomous. Under noise the in-pool games fail on flipped answers, as in
  ST-3.
* **Published P0a at σ = 0.5**, under `always`:
  * FPR 0.429 → 0.459;
  * won 0.505 → 0.481;
  * wrong 0.009 → 0.

  At σ ≤ 0.25, flip 0, the rules remove what little wrong there is (`soft` at
  σ = 0.25: 0.012 → 0) and move won and FPR by at most 0.004.
* **The prior in the last window** at σ = 0.75 rises with admissions. Under `always`
  it is 0.265 for every arm-backend, against 0.049 (`soft`) and 0.203 (P0a with
  `no_repeat`) under `top`. That is stream-K's oracle value, reached here from the
  loop's own admissions.
* **The figure** `fig_endgame_rules.png` plots wrong, FPR and won against the rule at
  σ = 0.5 and 0.75, one panel per arm-backend. Its caption names the analyzer and both
  run directories.

## What this changes

* FINDINGS §3.14 said that at σ = 0.75 no arm is both autonomous and safe. That now
  holds only under ST-3's end rule.
* P0a with `no_repeat` and the budget-end rule `always` (equal to `conf` at σ = 0.75)
  meets both definitions on the endless tail at flip 0: wrong 0, FPR 0.099. It does
  not meet them under noise.
* For `soft`, the end rule trades wrong answers for crowded in-pool games almost one
  for one. Its near-copy weakness is in the likelihood, as SK1 found, and a decision
  at the end of the game cannot fix it.

## Not tested; next steps

* **Planner.** A planner that spends its last questions on the leading survivor
  against UNKNOWN, or on parent against child, might win back `soft`'s crowded games.
  So might a larger budget only for games still ambiguous near the end. That is the
  lever EG1's failure points to.
* **Noise.** Under flip 0.05 no end rule helps the in-pool side. That is ST-2's
  territory (name confusion and noisy answers), not the end rule.
* **The `conf` threshold.** It only differs from `always` while u ≤ 0.1, so at
  σ = 0.75 it was not separately tested. A stream whose UNKNOWN mass starts low and
  rises would separate them.
