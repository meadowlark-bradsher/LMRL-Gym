# Pre-registration: CD-NOUL-1 Stage 1 — likelihood from the model, choice from the planner

**Drafted:** 2026-11-08. **Status: FROZEN 2026-11-08**, tag `prereg-noul1-20261108` on the commit that
adds this file; no confirmatory episode was played before that tag; the confirmatory run is made from
the tagged commit (`run_noul.py --phase confirmatory --tag prereg-noul1-20261108` refuses to run otherwise). **Game:** `live-car-dealer-v0.1`,
the CD-VOI-1 primary cell — easy (r = 2/5), H = 4, c_q = 0; opponent = the registered T1 kernel
(α₀ = 4/5, β = 8, p_counter = 1, p_hold = 1, shortlist 3), truthful answers; the planner-complete shortlist
menu (`DESIGN_LIVE.md` §5). **Design:** [`DESIGN_NOUL.md`](DESIGN_NOUL.md). **Models:** `jev-latest`
(Noul decoders and the `jev_agent` arm; HTTP backend, `TYPESAFE_API_KEY`) and `claude-sonnet-5-5`
(verbalizer bank, `sonnet_answer` decoder, `sonnet_agent` arm; Anthropic SDK).

## Question

Every live Car Dealer study so far gave the focal the opponent's answers as typed `yes` / `no`. The
20-Questions program's working architecture — the LLM supplies only p(utterance | hypothesis), code
holds the belief and chooses — has one step that only exists once the opponent speaks: turning a
free-text reply into a per-type likelihood. **Does the exact planner, fed that likelihood by a model,
recover the value of inquiry in a natural-language Car Dealer, and does it beat the LLM-as-agent on
identical menus?** CD-VOI-1's Jev captured −10 % of the exact ceiling (P1 −0.010 Y against Δ\*_menu
0.102); R2's reward-only learner found no asking at all.

## What is frozen before the tag (Phase 0, already committed)

| input | file | sha256 (first 16) |
|---|---|---|
| verbalizer bank, 4,320 utterances (2 regimes × 2 roles × 9 questions × 6 rankings × K = 20) | `results/noul_bank_20261108/bank.jsonl` | `5cc43213cd444fcc` |
| blind audit labels (gate 1: 0 contradictions in both regimes) | `results/noul_bank_20261108/audit_labels.jsonl` | `def0315566e0b695` |
| Noul decoder matrix (two answer Nouls + six type Nouls per item) | `results/noul_bank_20261108/matrix_noul.jsonl` | `3e29647b3f413ec3` |
| Sonnet judge matrix (P(YES) as JSON per item) | `results/noul_bank_20261108/matrix_sonnet.jsonl` | `fa06ca4b8523105a` |
| decoder prompts (`decode.context_state`, `noul_questions`, `SONNET_SYSTEM`) | `decode.prompt_hash()` | `aea24a833e5c` |

The regex decoder's cue lists (`decode.py`) were written before the bank existed and have not changed.
`decoder_calibration.json` (Phase 0, descriptive) is cited from the file and was not used to alter any
decoder, prompt or threshold. The verbalizer in `plain` conveys only the answer to the question asked;
in `oblique` it may let its priorities show but never states a falsehood (audited).

## Arms (`arms_noul.parse_arm`), regimes, cells

| arm | chooser | belief updated on answers by | planner & filter hedge a | regimes |
|---|---|---|---|---|
| `plan_ask-oracle-{0,0.2}` | exact planner, ASK allowed | the true yes/no (ceiling for binary decoders) | 0 / 0.2 | plain, oblique |
| `plan_ask-regex-{0,0.2}` | exact planner | keyword parser; unparsed → uninformative | 0 / 0.2 | plain, oblique |
| `plan_ask-noul_answer-{0,0.2}` | exact planner | Jev Nouls, p_yes = y/(y+n) | 0 / 0.2 | plain, oblique |
| `plan_ask-noul_type-{0,0.2}` | exact planner | Jev Nouls, six-type vector | 0 / 0.2 | plain, oblique |
| `plan_ask-sonnet_answer-{0,0.2}` | exact planner | Sonnet judge p_yes | 0 / 0.2 | plain, oblique |
| `plan_noask` | exact planner, ASK forbidden | — (never sees text) | 0 | none |
| `jev_agent` | `jev-latest` Choice over the lettered menu, utterances in the state text | menu scaffolding: `noul_answer` at a = 0.2 | 0.2 (menu) | plain, oblique |
| `sonnet_agent` | `claude-sonnet-5-5` letter choice, identical text | menu scaffolding: `noul_answer` at a = 0.2 | 0.2 (menu) | plain, oblique |

The hedge a is CD-MIS-2's assumed answer flip rate, applied identically in the planner's ASK branches
and the filter's answer update (`filters_mis.FilterSpec(answer_noise=a)`); the belief update is exact
Bayes with no background floor (`arms_noul.ETA = 0`; see `DESIGN_NOUL.md` §7c for why). Typed opponent
moves update every arm's belief through the matched filter. Agents see the Phase 1 plain treatment: own
utility of each deal, no tags, no EIG, no P(accept), no posterior.

Cells: 2 focal roles × 2 openers × 6 focal types × 6 opponent types = 144, **K = 5 repeats**, **seed
20261112** (`run_noul.PHASES["confirmatory"]`). The opponent's move stream is seeded by
(seed, role, opener, i, j, k) and the bank variant served for a question by (seed, cell, repeat,
question), so every arm in a cell hears the same words when it asks the same question. Text arms run in
both regimes (720 episodes per arm per regime), `plan_noask` once. Planner arms are offline and
deterministic given the matrices; the two agent arms spend model calls: about 2,900 Jev Choice calls
(Phase 1 averaged 2.06 decisions per episode) and about 4,000 Sonnet calls (≈ $15); wall time is
dominated by the Sonnet agent, roughly 2–3 h sequential.

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python3 -u -m llm_rl_scripts.car_dealer_noul.run_noul --phase confirmatory \
    --tag prereg-noul1-20261108 --bank results/noul_bank_20261108 --out results/noul_stage1_<date> \
    --jev-backend http --sonnet-backend anthropic
```

## Outcome, clusters, verdicts

Y = focal normalized net gain = net / (U_max − r), as in Phase 1. Contrasts are paired over the 36
(buyer ranking, seller ranking) type-pair clusters with a percentile cluster bootstrap, B = 5000, seed 0
(`analyze_noul.py`, reusing `analyze_phase1.paired_bootstrap`). Ceilings are exact; the captured fraction
divides the bootstrap draws by the exact mean Δ\*_menu of the same cell set in Y units. **Primary cell set:
focal-opens** (the ask-versus-offer decision is the focal's first move there; the opponent-opens ceiling
sits below δ — CD-VOI-1's reasoning). **Primary hedge: a = 0.2** (`verdicts_noul.PRIMARY_HEDGE`): it is the
value CD-MIS-2 registered and found to recover the evasive-answer loss, and the setting an agent would run
with without knowing whether its counterpart answers plainly; its cost in the plain regime is what P1's
captured fraction and P5 measure. δ = 0.02 Y. Vocabulary in `verdicts_noul.py` (CI excludes 0 on the
predicted side → confirmed; the other side → opposite; else not supported).

Rules fixed in `verdicts_noul.py`:

| | quantity (focal-opens unless stated) | prediction |
|---|---|---|
| **P1 (primary)** | Δ Y, `plan_ask-noul_answer-0.2` (plain) − `plan_noask` | > 0 **and** captured fraction of Δ\*_menu (point) ≥ 0.5 |
| P2 | Δ Y, `plan_ask-noul_answer-0.2` − `plan_ask-regex-0.2`, oblique | > 0 (the model likelihood earns its keep where the text is indirect) |
| P3 | Δ Y, `plan_ask-noul_answer-0.2` − `plan_ask-regex-0.2`, plain | CI within ±δ of 0 (a parser suffices for direct answers) |
| P4 | Δ Y, `plan_ask-noul_type-0.2` − `plan_ask-noul_answer-0.2`, oblique | ≥ 0 (CI not entirely below 0; exploratory) |
| P5 | Δ Y, `plan_ask-<d>-0.2` − `plan_ask-<d>-0`, oblique, d ∈ {noul_answer, regex} | ≥ 0 (the fixed hedge is not harmful under decoding error) |
| P6 | selectivity P(ask \| ASK optimal) − P(ask \| legal, not optimal), `sonnet_agent`, each regime | upper CI bound < 0.2 (asks non-selectively; ASK-optimality under the a = 0 oracle at the exact belief) |
| P7 | Δ Y, `plan_ask-noul_answer-0.2` − `sonnet_agent` and − `jev_agent`, each regime | > 0 (the architecture beats the LLM-as-agent on identical menus) |
| P8 | Δ in-episode answer log-loss (bits), `noul_answer-0.2` − `regex-0.2`, oblique, all cells | < 0 |
| descriptive | the same contrasts at a = 0; `sonnet_answer` as decoder vs regex and vs oracle; `vs_oracle` for every decoder; questions per episode; walk rates; decoder diagnostics (P(true type) after each decode, zeroed-true-type episodes); agent positions, well-formed share, tokens | reported, no rule |

One primary, no multiplicity correction on the secondaries.

## Sanity gates (fail → fix the pipeline, do not read results)

- No error stubs; every (arm, regime) key has 720 episodes; agents' well-formed share ≥ 0.99.
- `plan_ask-oracle-0` (plain) − `plan_noask`, all cells, reproduces the exact Δ\*_menu in Y units within its CI
  (the oracle decoder with no hedge *is* CD-VOI-1's `exact_ask`; `test_arms_noul.py` replays it decision for
  decision on the Phase 1 seed).
- `plan_ask-oracle-0` is text-invariant: oblique − plain = 0 exactly.

## Predictions, stated plainly

- **P1 positive; the captured-fraction threshold is the real test.** In `plain` every decoder reads the
  answer, so the belief after a decode is the exact belief up to the hedge; what the hedge costs is fewer
  questions (the 0.2 planner values a question at 60 % of its truthful information). If the architecture
  captures less than half of Δ\*_menu with perfect decoding, the hedge is too expensive for this game and
  the reading is "robustness bought at the price of inquiry" — a result, not a failure of the pipeline.
- **P2 confirmed, P3 confirmed.** Phase 0 calibration (descriptive) found the parser reading 12.5 % of
  oblique replies and the Noul 71 %, with no confident errors; in plain all three are perfect. The open
  question is whether a better belief turns into better *offers* within four opportunities.
- **P4 open.** The type reading was weak in Phase 0 (posterior on the true type 0.24 vs 0.17 uniform); the
  verbalizer's leaked emphasis may or may not be worth more than the binary answer.
- **P5 confirmed for regex, plausible for noul.** The hedge exists to blunt confident misreads; the parser
  makes them (54 in the bank), the Noul did not.
- **P6 confirmed.** The persona study's Sonnet agent asked persistently; Jev never asked; neither tracks
  ASK-optimality. The registered quantity is the difference, so a model that asks everywhere and one that
  never asks both confirm P6.
- **P7 confirmed in both regimes.** The honest alternative: Sonnet 5.5 reading indirect text and choosing
  from a planner-complete menu closes the gap on its own — then the program's "the LLM is a poor planner"
  reading needs revising for current models, and that is worth knowing.
- **P8 confirmed** — it is nearly a restatement of the Phase 0 calibration in episode weights.

## Disclosures made at registration

- Phase 0 ran an offline smoke of the full arm set with mock agents (pilot seed 20261111, one repeat) in the
  scratchpad; its episode counts, ask rates per arm and decode records were read as plumbing checks, its
  outcome contrasts were not, and it is not part of the study. The ask-rate reading is what surfaced the
  hedge's cost and is disclosed in `DESIGN_NOUL.md` §7c; the primary hedge was then chosen on the grounds
  above, before any outcome was seen.
- Live smokes before the freeze: 24 verbalizer items and 10 decoder items (both models), to check prompt
  compliance; not kept in `results/`.
- The agents' menus are built from a belief decoded by `noul_answer` at a = 0.2, not from the oracle
  decoding (which would hand the agent the answer's meaning through SAFE / PLAN) and not from no decoding
  (which would handicap the agent's menu). The agent-versus-planner contrast therefore isolates the chooser
  on identical menus; it is not a test of an unscaffolded agent.
- `jev-latest` and `claude-sonnet-5-5` are pinned by name only; the decoder matrices are frozen files, so
  decoder drift is impossible inside the study, but the two agent arms are live and any later replication
  may see different agents.
- Noul's two-way reading leans NO on deflecting replies (mean p_yes 0.42 on the 26 human-unclear audit
  items, Phase 0); it is reported, not corrected.

## Analysis (fixed in advance)

```bash
PYTHONHASHSEED=0 PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.analyze_noul --dir results/noul_stage1_<date> --B 5000 --seed 0
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.verdicts_noul --analysis results/noul_stage1_<date>/analysis.json
PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.render_noul_note --dir results/noul_stage1_<date> --out llm_rl_scripts/car_dealer_noul/RESULTS_NOUL.md
```

Verdicts applied by code and frozen before reading. No rule changes after data; any pipeline correction
after the run is disclosed in a post-run section of this file. A regeneration test pins the frozen
results, as in the persona study.

## Not registered here

Stage 2 (a live LLM opponent with a strategic brief, the realizer, a Noul over the opponent's offers, the
clairvoyant ceiling) is a separate registration written after this read-out. Any second hedge value, any
decoder prompt change, and any use of the `oblique` regime's leaked type information beyond P4 are
exploratory if reported.

## Limits accepted in advance

One regime of the game and one kernel; the opponent never lies and never adapts to the text; the type
space is the six rankings (binning is free, as in 20Q); the question bank is fixed and the focal's own
language is templated; the plain-regime bank has low lexical diversity (the model converges on a few
phrasings); 36 type pairs are the whole type space, so cluster intervals have a precision ceiling; the
Sonnet judge, the verbalizer and the Sonnet agent are the same model family (the `sonnet_answer` decoder
is included so a Noul-versus-Sonnet gap is read against a same-family judge).
