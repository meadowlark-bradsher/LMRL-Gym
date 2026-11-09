# CD-NOUL-2 Phase 0 gates from the pilot

Rendered by `gates_noul2.py render` from `results/noul2_pilot_20261108/gates/` (pilot `results/noul2_pilot_20261108`, 864 episodes, 0 error stubs; brief hash `611ad7ac429a`, realizer hash `0694a14552e1`; rendered at commit `ce15f01`). Thresholds are `DESIGN_NOUL2.md` §6. Pilot outcome contrasts were not read; gate 6 is the one contrast §6 names.

## Verdicts

| gate | quantity | threshold | value | verdict |
|---|---|---|---|---|
| 1 opponent protocol, `candid` | well-formed share / defaulted share over 842 moves | ≥ 0.99 / ≤ 0.01 | 1.0000 / 0.0000 | PASS |
| 1 opponent protocol, `guarded` | well-formed share / defaulted share over 895 moves | ≥ 0.99 / ≤ 0.01 | 1.0000 / 0.0000 | PASS |
| 1 opponent protocol, `adversarial` | well-formed share / defaulted share over 880 moves | ≥ 0.99 / ≤ 0.01 | 1.0000 / 0.0000 | PASS |
| 2 never-lies audit | contradictions in 60 blind-labelled answers per brief | 0 candid, ≤ 2 guarded; adversarial misleads ≥ 20 % | — | PENDING: the pilot arms never ask, so no opponent answers exist; needs the answer probe |
| 3 guarded is guarded | `regex` accuracy on guarded answers | < 0.9 | — | PENDING (answer probe) |
| 4 kernel fit, `candid` | held-out gain over T1, nats per move (842 moves, 3 folds) | ≥ 0.05 to adopt | -0.0035 | KEEP T1 |
| 4 kernel fit, `guarded` | held-out gain over T1, nats per move (895 moves, 3 folds) | ≥ 0.05 to adopt | 0.0503 | ADOPT FIT |
| 4 kernel fit, `adversarial` | held-out gain over T1, nats per move (880 moves, 3 folds) | ≥ 0.05 to adopt | 0.0891 | ADOPT FIT |
| 5 realizer fidelity, buyer | blind recovery / added commitments in 200 realized moves | ≥ 0.99 / 0 | 1.0000 / 0 | PASS |
| 5 realizer fidelity, seller | blind recovery / added commitments in 200 realized moves | ≥ 0.99 / 0 | 1.0000 / 0 | PASS |
| 6 clairvoyant gap | Δ_clair = plan_reveal − plan_noask, Y, `guarded`, focal opens (36 clusters) | ≥ δ = 0.02 | 0.2233 [0.1651, 0.2879] | PASS |
| 7 parity | Stage 2 runner reproduces Stage 1 `plan_ask-oracle-0` with scripted T1 + templates + `oracle` | exact | — | PASS (`test_arms_noul2.py::test_parity_with_stage1`, offline) |
| 8 calibration constants | Platt map and a_q vectors from the frozen Stage 1 matrices | committed | — | PASS (`results/noul_bank_20261108/calibration_noul2.json`, `calibrate_noul.py`) |
| 9 no confident-wrong reads on candid | confident-wrong reads under Platt + clamp 0.95 on blind-audited candid answers | 0 | — | PENDING (answer probe) |

## Gate 1 — opponent protocol, per brief

| brief | episodes | opponent moves | well-formed | retried | defaulted | stop end_turn | realizer calls ok / fallback | opponent asked (episodes) | tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| candid | 288 | 842 | 842 | 0 | 0 | 842 | 751 / 0 of 751 | 45 | 1303461 / 496896 |
| guarded | 288 | 895 | 895 | 0 | 0 | 895 | 791 / 0 of 791 | 24 | 1394752 / 516313 |
| adversarial | 288 | 880 | 880 | 0 | 0 | 880 | 777 / 0 of 777 | 18 | 1358592 / 506485 |

## Gate 4 — kernel fit per brief (`kernel_fit.heldout_report`, episode-level held-out split)

| brief | moves | episodes | mean LL fitted (all) | mean LL T1 (all) | T1 zero-probability moves | held-out gain | adopt | fitted α₀ | β | p_counter | p_hold | accept_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| candid | 842 | 288 | -7.4000 | -7.4688 | 441 | -0.0035 | no | 4/5 | 0.024 | 1.000 | 1.000 | 0.104 |
| guarded | 895 | 288 | -7.2521 | -7.3704 | 462 | 0.0503 | yes | 7/10 | 0.024 | 0.999 | 0.500 | 0.281 |
| adversarial | 880 | 288 | -6.7806 | -6.8997 | 422 | 0.0891 | yes | 7/10 | 0.024 | 0.999 | 0.500 | 0.273 |

A T1 zero-probability move is a logged opponent move the reference kernel gives probability 0 (it is floored at 1e-06 in the likelihood). Per-fold detail: `gates/kernel_fit_<brief>.json`.

Support of the T1 family over the live opponent's moves (moves outside the kernel's support, by kind of the logged move; T1 never asks, so every opponent question is outside it):

| brief | kind | moves | outside T1 support | outside fitted support |
|---|---|---|---|---|
| candid | ACCEPT | 188 | 3 | 3 |
| candid | ASK | 45 | 45 | 45 |
| candid | PROPOSE | 562 | 393 | 388 |
| candid | WALK | 47 | 0 | 0 |
| guarded | ACCEPT | 194 | 0 | 0 |
| guarded | ASK | 24 | 24 | 24 |
| guarded | PROPOSE | 623 | 438 | 429 |
| guarded | WALK | 54 | 0 | 0 |
| adversarial | ACCEPT | 202 | 0 | 0 |
| adversarial | ASK | 18 | 18 | 18 |
| adversarial | PROPOSE | 615 | 404 | 395 |
| adversarial | WALK | 45 | 0 | 0 |

Reading, `candid`: 393 of the opponent's 562 proposals (0.70) are deals the T1 family cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of 0.024 makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from accept_noise (0.104) and p_hold (1.000). The T1 family is misspecified in support for this opponent; the verdict column applies the registered rule as written.

Reading, `guarded`: 438 of the opponent's 623 proposals (0.70) are deals the T1 family cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of 0.024 makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from accept_noise (0.281) and p_hold (0.500). The T1 family is misspecified in support for this opponent; the verdict column applies the registered rule as written.

Reading, `adversarial`: 404 of the opponent's 615 proposals (0.66) are deals the T1 family cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of 0.024 makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from accept_noise (0.273) and p_hold (0.500). The T1 family is misspecified in support for this opponent; the verdict column applies the registered rule as written.

## Gate 5 — realizer fidelity (blind judge over the packet; `gates/realizer_packet.jsonl`, key `realizer_key.json`, labels `realizer_labels.jsonl`)

| role | judged | recovered | recovery | added commitments | judge unparsed | auto round-trip ok | fallbacks |
|---|---|---|---|---|---|---|---|
| buyer | 200 | 200 | 1.0000 | 0 | 0 | 200 | 0 |
| seller | 200 | 200 | 1.0000 | 0 | 0 | 200 | 0 |

| move kind | judged | recovered | added |
|---|---|---|---|
| ACCEPT | 25 | 25 | 0 |
| PROPOSE | 209 | 209 | 0 |
| REJECT | 166 | 166 | 0 |

All realized moves in the pilot: 2319 (auto round-trip ok 2319, fallbacks 0); replay skipped episodes: 0.

Judge: `claude-sonnet-5-5 ×400`.

## Gate 6 — clairvoyant gap

Δ_clair = `plan_reveal − plan_noask` in Y = net gain / (U_max − r), `guarded`, focal opens: paired over 36 (buyer ranking, seller ranking) clusters from 144 episodes, percentile cluster bootstrap B = 5000 seed 0.

| quantity | value |
|---|---|
| Δ_clair point | 0.2233 |
| 95 % CI | [0.1651, 0.2879] |
| δ | 0.02 |
| exact Δ_full under the planner's assumed kernel (Y, same cells; `ceilings.json`) | 0.1019 |

## Pending

Gates 2, 3 and 9 need opponent answers to the bank's questions under each brief (the answer probe: `LLMOpponent.answer` under planted rankings, about 60 answers per brief, blind-labelled with a sealed key as Stage 1's `audit.py`). The pilot arms `plan_noask` and `plan_reveal` never ask, so the pilot holds none.
