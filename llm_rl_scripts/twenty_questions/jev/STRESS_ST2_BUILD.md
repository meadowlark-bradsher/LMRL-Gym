# ST-2 pre-tag tooling — build note

The tooling `STRESS_LOOKALIKE_PREREG.md` lists under *Built before tagging*. Nothing
here plays a game on the real matrix. The self-test, smoke run and timing use the
synthetic matrix only. The look-alike census below reads the real matrix but plays
no games, so it is not an ST-2 outcome.

**Status (2026-11-12):** built, not tagged, not run.

Everything runs under `/opt/miniconda3/envs/LLM_RL/bin/python` (Python 3.9) with
`PYTHONHASHSEED=0`.

## What was built

| piece | where |
|---|---|
| ST-2 runner: `--plan`, `--self-test`, `--smoke` (synthetic matrix), `--run` (registration-checked, resumable per job) | `stress_st2.py` |
| nested look-alike sets with Zipf-weighted parents (`lookalike_plan`, `build`) | `stress_st2.py` |
| per-game trace: by-name guesses, whether the secret's parent was guessed, re-guesses, repeat questions | `stress_st2.GameTrace` |
| look-alike census per set: distance to parent, nearest other object, ties | `stress_st2.lookalike_census` |
| ST-2 analyzer: the prereg's *Scoring* rules, `analysis.json`, 2 figures with provenance captions | `analyze_st2.py` |

Nothing in the shared code changed. The runner calls `openworld.run_accumulation`
with its existing `oracle=` and `observer_factory=` hooks, and the G arm uses
`stress_oracles.lookalike_confusion` from ST-0b.

Self-test (synthetic matrix, about 45 s):

```
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_st2 --self-test
```

It checks: sets are seeded, share parents across d and are nested; each look-alike
sits at exactly d from its parent; real rows are untouched; parents follow the
stream's Zipf weights; the stream and PY fit are the same for every job; the G
oracle confuses only a guess of the secret's parent and only at d ≤ 2; every backend
plays a short job in both arms with no `false_confirm` under exact guesses.

## Grid

`stress_st2 --plan`: **732 jobs of 2500 games.** Exact arm 147 per backend (7 d × 3
look-alike seeds × (1 at flip 0 + 2 flips × 3 noise seeds)). G arm 36 per backend (2 d
× 3 look-alike seeds × 2 g × 3 noise seeds). Four backends.

## Look-alike census (real matrix, no games)

`census_job` over the 3 look-alike seeds × 7 distances. "Nearer" counts look-alikes
whose nearest other object (real or look-alike) is closer than their parent. "Twin"
counts look-alikes at binarized distance 0 from another object.

| seed | distinct parents | d = 1 nearer / twin | d = 2 nearer | d = 6 nearer | d = 12 nearer |
|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 20 / 20 | 7 | 23 | 78 |
| 2 | 37 | 26 / 26 | 2 | 27 | 76 |
| 3 | 41 | 12 / 12 | 0 | 21 | 80 |

Two things follow from this. Neither changes the design as written, but both are
worth deciding on before the tag.

* **Siblings make exact twins at d = 1.** Drawing 100 parents with replacement by
  Zipf weight (Decision 2) gives only 37–42 distinct parents, and the commonest
  parents get many look-alikes. Two d = 1 look-alikes of one parent that flip the
  same cell are identical rows. That is 12–26 of the 100 at d = 1, and 0–2 at d ≥ 2.
  Such a twin cannot be told from its admitted sibling by any attribute, only by a
  by-name guess. This is a real long-tail case, but it is not "a novel object at
  distance 1 from a known one". The alternative is to draw parents without
  replacement (still Zipf-weighted). That gives 100 distinct parents and no sibling
  twins. Common objects would then be only *more likely* to have a look-alike, not
  more likely to have several.
* **At large d the parent stops being the nearest neighbour.** The real matrix is
  crowded (98 of 158 objects have a neighbour within one attribute). By d = 8 about
  40 of 100 look-alikes, and by d = 12 about 78, are nearer some other object than
  their parent. So d measures distance to the parent, not to the nearest known. P2
  and P5 are stated in d, and the runner keeps it that way. The census is stored per
  run (`census.json`), so the analysis can also report results by nearest-neighbour
  distance as exploratory.

## Smoke run and timing

`--smoke` on the **synthetic** matrix: 20 full-length (2500-game) jobs, d ∈ {1, 6},
flip ∈ {0, 0.10}, G arm at d = 1, g = 0.20, one seed, all four backends. It ran
through `jobq` (job 5, 4 slots, `taskpolicy -b`) while the machine had 9 of 10 slots
busy. The analyzer read the output and scored a complete grid. Its verdicts are
about the synthetic matrix and mean nothing.

| backend | wall seconds per job (min–max over 5) |
|---|---:|
| `soft` | 1189 (1008–1401) |
| `hard` | 394 (387–414) |
| `modeswitch` | 398 (378–435) |
| `modeswitch_guard` | 398 (324–448) |

The workers ran at about 40% of a core each (background priority under load; total
5567 CPU-s over 3139 s wall on 4 workers). **Projected for the full grid:** 183
`soft` jobs and 549 discrete jobs, about 73,000 job-seconds at this rate. That is
**about 20 h on 6 slots under similar load**, roughly half of it `soft`. On an idle
machine it would be several times faster. Options if that is too long: drop
unguarded `modeswitch` (−25%), or run `soft` only at d ∈ {1, 2, 6, 12}.

### What the smoke run changed in the analysis

The first P5 rule compared a parent's in-pool games before and after its
look-alike's admission, adjusted by control games in the same windows. On the
smoke output it gave −3 to −4 questions for every discrete cell, the wrong sign. The
cause is the repeat loop. The stream's prior is above ≈ 0.15 for its first 274 games,
and in-pool games there run all 20 questions and end as escalations. Common parents
are admitted early, so their "before" games are mostly loop games. The rule now
uses only games from 274 on and adjusts each parent game by control games within
100 games of it. On the smoke's d = 6 cells (where P5 predicts no change) the
adjusted value is +0.25 (95% interval −2.7 to 3.2). That is now in the
prereg (Decision 9 and *Scoring*).

The smoke run also shows repeat questions persisting after game 274 at d = 1 on the
synthetic matrix (246 of 429 repeat games in the `hard` cell). The synthetic matrix
has many exact twin rows, so this may not carry over to the real matrix. ST-2 logs
repeats per game either way (Decision 6).

## Reproduce

```
# grid listing
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_st2 --plan

# smoke (synthetic matrix; 20 full-length jobs; output is not a result)
jobq -N 4 -L lmrl-20q-st2-smoke taskpolicy -b env PYTHONHASHSEED=0 PYTHONPATH=. \
    OPENBLAS_NUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_st2 --smoke --outdir <scratch> --workers 4
PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.analyze_st2 --run-dir <scratch>

# the registered run, after the tag
jobq -N 6 -L lmrl-20q-st2 taskpolicy -b env PYTHONHASHSEED=0 PYTHONPATH=. \
    OPENBLAS_NUM_THREADS=1 /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_st2 --run \
    --prereg-tag prereg-stress-st2-<date> --workers 6
```
