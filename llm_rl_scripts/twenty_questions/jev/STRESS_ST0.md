# ST-0 shared stress-test infrastructure — build note

The shared pieces `STRESS_DESIGN.md` asks for before any stress study runs. They
are built and gated. No stress study has been registered or run. Building them
turned up two problems in the existing open-world harness: a reproducibility
problem and a planner failure. Both affect how the drafted pre-registrations
should be set before they are frozen.

Run everything here under the `LLM_RL` conda env (Python 3.9) with a pinned
`PYTHONHASHSEED`. Both requirements are explained below.

## What was built

| piece | where | what it does |
|---|---|---|
| ST-0a pool generator | `stress_generator.py` | Parent-perturbation objects (`generate_objects`, `build_pool`), `lookalike(parent, d)`, and the ρ_real calibration (`--calibrate`) |
| ST-0b answer channels | `stress_oracles.py` | Sampled, sticky, bursty, guess-noise and look-alike-confusion answerers; cross-model answering by passing a second matrix; `FlipCounter` for the realised flip rate. The i.i.d. `noisy_matrix_oracle` is untouched. |
| ST-0c scoring | `openworld.run_openworld_episode` | A "yes" to a by-name guess is `won` only if the name is the secret; otherwise `false_confirm` |
| ST-0d snapshot mode | `stress_harness.run_snapshot` | Fixed `H_known`, held-out novelty, no admissions; the shared `scorecard` |
| ST-0e census per cell | `stress_census.census` | Called on any pool |
| loop hooks | `openworld.py` | `run_accumulation(oracle=...)` with a `begin_game` hook; `OWEpisode.n_live_final`; opt-in `no_repeat` (below) |
| gates | `stress_regression.py` | Committed and A/B regression gates |

Mechanics self-test (offline, about 15 s):

```
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_harness --self-test
```

ST-0f (a numpy EIG path) is not built yet. It is only needed for `soft` at
N ≥ 2000, and it lands with ST-1.

## Finding 1 — the committed runs are not bit-for-bit reproducible as recorded

The design note's regression gate ("must reproduce bit-for-bit") failed on the
**untouched** code before any change was made. There are two causes.

* **Interpreter.** The committed runs were made under Python 3.9 (`LLM_RL`).
  Under 3.12, `soft` drifts: the clean arm's mean questions per game is 8.715
  against the committed 8.7185. The likely cause is 3.12's compensated float
  `sum`, which moves posteriors by an ulp and occasionally flips an EIG tie. The
  same commit that produced the runs (885f03a) gives the same 8.715 under 3.12.
  Under 3.9 the clean arm and every `soft` cell reproduce exactly.
* **String-hash order.** `hard`, `retract`, `tombstone` and `modeswitch` keep
  their candidates in Python `set`s of names. Iteration order depends on
  `PYTHONHASHSEED`, which is random per process by default. It breaks ties among
  equally weighted survivors differently. The committed values for these
  backends are one draw that cannot be reproduced.

How much the hash order moves the retraction study's headline cells (10% flips,
mean over noise seeds 1–2, five hash seeds):

| metric | `hard`: committed / range over hash seeds | `modeswitch@3`: committed / range |
|---|---|---|
| in-pool failure rate | 0.1750 / 0.1750–0.1753 | 0.1491 / 0.1491–0.1491 |
| detector AUC | 0.9128 / 0.9127–0.9128 | 0.9261 / 0.9261–0.9261 |
| wrong guesses | 0 / 0–0 | 0 / 0–0 |
| false escalations | 324.5 / 324.5–325.0 | 276.5 / 276.5–276.5 |
| questions per game | 8.8193 / 8.809–8.877 | 9.4745 / 9.480–9.499 |

No conclusion moves. Only questions per game shifts, in the second decimal, and
the retraction study's cost ordering (`hard` < `modeswitch` < `soft`'s 9.77)
holds at every hash seed.

**What the gates check instead.**

* *committed*: the hash-independent part of the record (the clean arm's game
  records and every `soft` cell) must reproduce exactly under 3.9.
* *ab*: every cell must reproduce game by game between base `main` and the
  working tree under the same pinned `PYTHONHASHSEED`.

Stress runs call `require_hash_seed()` and refuse to start without a pinned
seed.

**Fixed since (2026-11-11).** The survivor-set backends now read survivors in
the fixed `known_hyps` order, and the retraction and audit grids were redrawn
(`OPENWORLD_RETRACTION_RESULTS.md`, *Redraw*). The table above describes the first
draw: `hard`'s questions/game is now 8.774, and `modeswitch@3`'s is 9.437. Both
are below the hash-seed range, because admission order acts as a popularity
tie-break (`tie_order_probe.py`). The committed gate now checks every retraction
cell, not only `soft`. `hash_spread.json` is a record of the old code.

## Finding 2 — the discrete backends loop on a repeated question when the novelty prior is large

`hard` and `modeswitch` keep the catch-all's mass fixed at its prior: they can
strike a known object, but they never re-weigh `UNKNOWN`. A question that
splits `UNKNOWN` from the knowns (often a category question, because
`UNKNOWN`'s category likelihood is the known-set fraction) therefore keeps its
expected information gain however often it is answered. The open-world loop
has no repeat guard, so above a threshold the planner asks the same question
every turn until the budget runs out. The game then ends as an expert call,
because `UNKNOWN` tops a uniform survivor split.

Closed real pool, 150 in-pool games per prior, exact oracle:

| catch-all prior | `hard` won | `modeswitch` won | `soft` won | `hard` games with a repeated question |
|---:|---:|---:|---:|---:|
| 0.02 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.05 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.10 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.15 | 0.433 | 0.433 | 1.000 | 0.567 |
| 0.20 | 0.073 | 0.073 | 1.000 | 0.927 |
| 0.25 | 0.040 | 0.040 | 1.000 | 0.960 |

With `no_repeat=True` every backend wins every game at every prior in this table.

**It is already in the committed record.** On the §3.11 clean stream, `hard`
re-asks a question in 14 games, all early in the stream (prior 0.276–0.657).
12 of them are in-pool games that end as expert calls. That accounts exactly
for `hard`'s 12 fewer clean wins than `soft` (1842 vs 1854). So the retraction
study's clean-oracle "false escalations" for `hard` and `modeswitch` are this
loop, not a property of hard elimination. The effect on the cited conclusions
is small, because the PY prior falls below 0.15 early in a 2000-game stream.
But it is a real failure of the discrete backends whenever novelty is common.

**What was added.** An opt-in `no_repeat` flag on `run_openworld_episode`,
`run_accumulation` and `run_snapshot`, mirroring `bed.run_episode`'s flag. It
defaults to off, so every committed result is unchanged; the A/B gate checks
this. A gate-forced guess (`modeswitch`'s reinstate test) is never filtered.

## Calibration — ρ_real

All-generated pools with the real category composition, 20 generator seeds per ρ.
The target is the real matrix's `nn_within_1` = 98 of 158.

| ρ | nn_within_1 (mean over seeds) | distinct rows | ambiguous cells |
|---:|---:|---:|---:|
| 0.05 | 125.8 | 108.1 | 1043 |
| 0.10 | 119.1 | 117.6 | 1039 |
| 0.20 | 103.5 | 127.6 | 1031 |
| 0.225 | 99.0 | 131.5 | 1041 |
| 0.25 | 96.3 | 132.5 | 1045 |
| 0.30 | 93.8 | 135.6 | 1036 |
| 0.50 | 81.8 | 139.5 | 1033 |
| 1.00 | 74.6 | 140.9 | 1032 |
| real matrix | 98 | 140 | 1035 |

* **ρ_real = 0.2345**, by interpolating the seed means. The standard error per
  point is 1.1–1.8, so the curve is within two standard errors of the target
  from about ρ = 0.20 to 0.28. Treat ρ_real as ≈ 0.23 ± 0.04.
* **Matched on near neighbours, the generator makes more exact twins.** At
  ρ_real a generated pool has about 131 distinct rows against the real 140.
  Ambiguous cells match the real matrix at every ρ, because generated values are
  drawn from real ones.
* **There is a crowding floor.** Even at ρ = 1, where every cell is resampled
  from its category, 74.6 of 158 objects sit within one attribute of a
  neighbour. Generated pools can be made much more crowded than the real matrix
  but only modestly less. ST-1's drafted level 2·ρ_real ≈ 0.47 (about 82) is
  already close to the floor.

## Consequences for the drafted pre-registrations

To settle before each is registered:

1. **ST-1 snapshot prior.** The draft sets `p_unknown_prior` to the novelty
   share, 200/1200 = 0.167, which is inside the collapse zone. Either run ST-1
   with `no_repeat=True`, or use a prior ≤ 0.10, or treat the loop as one of the
   stressors. The choice changes what ST-1 measures, so it is a registration
   decision. **Decided 2026-11-11** (revised `STRESS_CROWDING_PREREG.md`): a
   1000 + 100 game mix (calibrated prior 0.091) for the published architecture,
   plus a `no_repeat` arm for `hard` and `modeswitch`, with the repeat rate
   logged per cell.
2. **ST-3 heavy tails.** At σ = 0.75 the PY prior stays high for a long stretch
   of the stream. Without a repeat guard, the discrete backends' expert load and
   their prior-insensitivity prediction (ST-3 P2) would mostly measure this
   loop.
3. **ST-1's crowding axis.** The 2·ρ_real level sits near the floor. Replacing
   it with ρ = 1 would make the least-crowded level the floor itself. **Decided
   2026-11-11:** levels {0 (twins), ρ_real/3, ρ_real, 1 (floor)}.
4. **All studies.** Pin `PYTHONHASHSEED` and use the `LLM_RL` env. Record both in
   every summary.

## Reproduce

```
# gates (~10 min on 12 cores)
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_regression

# rho_real calibration -> results/stress_calibration_20261111/calibration.json
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_generator --calibrate --seeds 20 \
    --rhos 0.05,0.1,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.35,0.4,0.5,0.7,1.0 \
    --out llm_rl_scripts/twenty_questions/jev/results/stress_calibration_20261111/calibration.json

# repeat-question loop -> results/stress_st0_diagnostics_20261111/repeat_probe.json
PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_harness --repeat-probe \
    --out llm_rl_scripts/twenty_questions/jev/results/stress_st0_diagnostics_20261111/repeat_probe.json

# hash-order spread -> results/stress_st0_diagnostics_20261111/hash_spread.json
PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
    -m llm_rl_scripts.twenty_questions.jev.stress_harness --hash-spread \
    --out llm_rl_scripts/twenty_questions/jev/results/stress_st0_diagnostics_20261111/hash_spread.json
```

## Gate results

Run on 2026-11-11: Python 3.9.21 (`LLM_RL`), `PYTHONHASHSEED=0`, A/B reference
`origin/main` at 9126ebd. **REGRESSION PASS, 52 of 52.**

* **committed, 6/6 exact.** All 2000 game records of the §3.11 clean arm, field
  by field (outcome, questions, `p_unknown_max` to the last bit, fire turn), and
  every per-seed metric of the five `soft` retraction runs (noise 0, 0.05 ×2,
  0.10 ×2).
* **ab, 46/46 game-by-game identical.** The clean arm, plus every retraction
  cell × noise seed: `soft`, `hard`, `retract@1/2/4`, `tombstone@1/3` and
  `modeswitch@1/3` at 0, 5% and 10% flips.

Both mechanics self-tests pass: `stress_harness --self-test` and the existing
`eval_openworld --self-test`.
