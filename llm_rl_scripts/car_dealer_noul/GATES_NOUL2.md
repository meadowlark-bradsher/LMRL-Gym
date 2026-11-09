# CD-NOUL-2 Phase 0 gates from the pilot

Rendered by `gates_noul2.py render` from `results/noul2_pilot_20261108/gates/` (pilot `results/noul2_pilot_20261108`, 864 episodes, 0 error stubs; brief hash `611ad7ac429a`, realizer hash `0694a14552e1`; rendered at commit `2f70e7d`). Thresholds are `DESIGN_NOUL2.md` §6. Pilot outcome contrasts were not read; gate 6 is the one contrast §6 names.

## Verdicts

| gate | quantity | threshold | value | verdict |
|---|---|---|---|---|
| 1 opponent protocol, `candid` | well-formed share / defaulted share over 842 moves | ≥ 0.99 / ≤ 0.01 | 1.0000 / 0.0000 | PASS |
| 1 opponent protocol, `guarded` | well-formed share / defaulted share over 895 moves | ≥ 0.99 / ≤ 0.01 | 1.0000 / 0.0000 | PASS |
| 1 opponent protocol, `adversarial` | well-formed share / defaulted share over 880 moves | ≥ 0.99 / ≤ 0.01 | 1.0000 / 0.0000 | PASS |
| 2 never-lies audit | contradictions in blind-labelled probe answers: candid 0 of 60, guarded 0 of 60; adversarial misleads 0.57 of 60 | 0 candid, ≤ 2 guarded; adversarial ≥ 0.2 | — | candid PASS, guarded PASS, adversarial MISLEADS |
| 3 guarded is guarded | `regex` accuracy on the 108 guarded probe answers | < 0.9 | 0.4722 | PASS |
| 4 kernel fit (widened family, §3.3a), `candid` | held-out gain over T1, nats per move (842 moves, 3 folds); T1-family fit -0.0035 (superseded) | ≥ 0.05 to adopt | 4.7684 | ADOPT FIT |
| 4 kernel fit (widened family, §3.3a), `guarded` | held-out gain over T1, nats per move (895 moves, 3 folds); T1-family fit 0.0503 (superseded) | ≥ 0.05 to adopt | 4.8544 | ADOPT FIT |
| 4 kernel fit (widened family, §3.3a), `adversarial` | held-out gain over T1, nats per move (880 moves, 3 folds); T1-family fit 0.0891 (superseded) | ≥ 0.05 to adopt | 4.4474 | ADOPT FIT |
| 5 realizer fidelity, buyer | blind recovery / added commitments in 200 realized moves | ≥ 0.99 / 0 | 1.0000 / 0 | PASS |
| 5 realizer fidelity, seller | blind recovery / added commitments in 200 realized moves | ≥ 0.99 / 0 | 1.0000 / 0 | PASS |
| 6 clairvoyant gap | Δ_clair = plan_reveal − plan_noask, Y, `guarded`, focal opens (36 clusters) | ≥ δ = 0.02 | 0.2233 [0.1651, 0.2879] | PASS |
| 7 parity | Stage 2 runner reproduces Stage 1 `plan_ask-oracle-0` with scripted T1 + templates + `oracle` | exact | — | PASS (`test_arms_noul2.py::test_parity_with_stage1`, offline) |
| 8 calibration constants | Platt map and a_q vectors from the frozen Stage 1 matrices | committed | — | PASS (`results/noul_bank_20261108/calibration_noul2.json`, `calibrate_noul.py`) |
| 9 no confident-wrong reads on candid | confident-wrong `noulcal` reads (platt, clamp 0.95) on the audited candid answers; guarded reported | 0 | candid 0 of 60 (bank 0 of 108); guarded bank 1 of 108 | PASS |

## Gate 1 — opponent protocol, per brief

| brief | episodes | opponent moves | well-formed | retried | defaulted | stop end_turn | realizer calls ok / fallback | opponent asked (episodes) | tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| candid | 288 | 842 | 842 | 0 | 0 | 842 | 751 / 0 of 751 | 45 | 1303461 / 496896 |
| guarded | 288 | 895 | 895 | 0 | 0 | 895 | 791 / 0 of 791 | 24 | 1394752 / 516313 |
| adversarial | 288 | 880 | 880 | 0 | 0 | 880 | 777 / 0 of 777 | 18 | 1358592 / 506485 |

## Gate 4 — widened kernel family (`DESIGN_NOUL2.md` §3.3a; `gates_noul2 kernel-wide`, `gates/kernel_wide.json`)

The family adds `eps_propose` (weight on a uniform draw over all legal proposals) and `p_ask` (a type-independent question rate) to the T1 form; both are fitted with the T1 parameters, the rule is unchanged (held-out gain over the registered T1 ≥ 0.05 nats per move).

| brief | held-out gain over T1 | adopt | eps_propose | p_ask | α₀ | β | p_counter | p_hold | accept_noise | mean LL widened / T1-family fit / T1 (all moves) |
|---|---|---|---|---|---|---|---|---|---|---|
| candid | 4.7684 | yes | 0.709 | 0.069 | 4/5 | 0.024 | 1.000 | 1.000 | 0.000 | -2.5430 / -7.4000 / -7.4688 |
| guarded | 4.8544 | yes | 0.747 | 0.036 | 4/5 | 0.024 | 1.000 | 1.000 | 0.045 | -2.4180 / -7.2521 / -7.3704 |
| adversarial | 4.4474 | yes | 0.696 | 0.028 | 4/5 | 0.024 | 1.000 | 1.000 | 0.088 | -2.3447 / -6.7806 / -6.8997 |

The held-out gain is dominated by T1's floor (1e-06) on the moves outside its support. Floor-free check — mean log-likelihood over only the moves T1 itself supports (all-moves fits, descriptive):

| brief | moves T1 supports | T1 | T1-family fit | widened fit |
|---|---|---|---|---|
| candid | 401 | -0.4891 | -0.4748 | -1.0523 |
| guarded | 433 | -0.4937 | -0.4870 | -1.0284 |
| adversarial | 458 | -0.5275 | -0.5229 | -1.0314 |

Split count (`split_breakdown`): of the moves a kernel gives zero probability at the opponent's true ranking, those zero under every ranking (no type evidence; the η = 0 filter keeps its belief) and those positive under some other ranking (a split: the filter sets the true ranking's belief to 0 for the rest of the episode):

| brief | kernel | zero at true ranking | zero under every ranking | split | split by kind | episodes with a split |
|---|---|---|---|---|---|---|
| candid | T1 | 441 | 376 | 65 | ACCEPT 3, PROPOSE 62 | 63 of 288 |
| candid | T1-family fit | 436 | 375 | 61 | ACCEPT 3, PROPOSE 58 | 59 of 288 |
| candid | widened fit | 3 | 0 | 3 | ACCEPT 3 | 3 of 288 |
| guarded | T1 | 462 | 377 | 85 | PROPOSE 85 | 79 of 288 |
| guarded | T1-family fit | 453 | 377 | 76 | PROPOSE 76 | 70 of 288 |
| guarded | widened fit | 0 | 0 | 0 | — | 0 of 288 |
| adversarial | T1 | 422 | 346 | 76 | PROPOSE 76 | 71 of 288 |
| adversarial | T1-family fit | 413 | 346 | 67 | PROPOSE 67 | 62 of 288 |
| adversarial | widened fit | 0 | 0 | 0 | — | 0 of 288 |

## Gate 4 — kernel fit per brief, T1 family (superseded by §3.3a; kept as the record) (`kernel_fit.heldout_report`, episode-level held-out split)

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

Reading, `candid`: 393 of the opponent's 562 proposals (0.70) are deals the T1 family cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of 0.024 makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from accept_noise (0.104) and p_hold (1.000). The T1 family is misspecified in support for this opponent; DESIGN_NOUL2 §3.3a widens it (above).

Reading, `guarded`: 438 of the opponent's 623 proposals (0.70) are deals the T1 family cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of 0.024 makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from accept_noise (0.281) and p_hold (0.500). The T1 family is misspecified in support for this opponent; DESIGN_NOUL2 §3.3a widens it (above).

Reading, `adversarial`: 404 of the opponent's 615 proposals (0.66) are deals the T1 family cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of 0.024 makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from accept_noise (0.273) and p_hold (0.500). The T1 family is misspecified in support for this opponent; DESIGN_NOUL2 §3.3a widens it (above).

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

## Gates 2, 3, 9 — the answer probe (`answer_probe.py`; `results/noul2_probe_20261108`)

The pilot arms never ask, so the opponent's answers come from a probe: every bank question asked as the focal's opening move to the live opponent under each brief, role and planted ranking (3 × 2 × 6 × 9 answers), every raw reply stored. Blind packet of 180 answers balanced on the truthful answer, sealed key, labeller: Claude Fable 5.1 subagent, blind (audit_view.txt only). Limitation: no prior exchange, so this probes the briefs' answering, not answers under negotiating pressure.

| brief | answers | audited | labels yes / no / unclear | contradictions | agree with truth | `regex` acc. (bank) | `regex` unparsed | retried / defaulted |
|---|---|---|---|---|---|---|---|---|
| candid | 108 | 60 | 30 / 30 / 0 | 0 | 1.00 | 1.0000 | 0 | 0 / 0 |
| guarded | 108 | 60 | 13 / 16 / 31 | 0 | 0.48 | 0.4722 | 57 | 0 / 0 |
| adversarial | 108 | 60 | 32 / 28 / 0 | 34 | 0.43 | 0.3796 | 0 | 0 / 0 |

| brief | Noul reads | map (T, b, clamp) | confident-wrong, audited | confident-wrong, bank | confident-wrong, raw bank | mean read error |
|---|---|---|---|---|---|---|
| candid | 108 | platt (3.621, 1.973, 0.95) | 0 | 0 | 0 | 0.0500 |
| guarded | 108 | platt (3.621, 1.973, 0.95) | 0 | 1 | 0 | 0.3266 |
| adversarial | 108 | platt (3.621, 1.973, 0.95) | 34 | 67 | 38 | 0.6102 |
