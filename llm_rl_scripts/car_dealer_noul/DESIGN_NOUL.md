# CD-NOUL-1 — likelihood from the model, choice from the planner (plan, 2026-11-08)

**Status:** plan for review. Nothing built, nothing registered, no model call made.
**Question:** does the Twenty-Questions architecture — an LLM supplying only the likelihood
of what the counterpart *says*, an exact planner choosing every move — recover the value of
inquiry in a *natural-language* Car Dealer that the LLM-as-agent left on the table?

## 1. Why this is the next study

Every live Car Dealer study so far (CD-VOI-1, CD-MIS-1/2, CD-SEL-1, the DRL rungs) has the
opponent speak in **typed moves**: `ASK top:price` is answered with the literal string `"yes"`.
The planner's belief update is therefore exact and free; the only question asked of a model was
*which move to pick*, and the answer was that neither Jev (0/2555 ASK-optimal nodes) nor a
reward-only learner (R2, 0/2747) discovers the value of asking, while any planner handed a value
oracle recovers it exactly (R1, R4).

In 20Q the split that worked was: Jev's **Noul** returns p(answer | hypothesis, question) once
per (object, attribute) pair; BED code holds the posterior and picks the question (158/158).
The one step of that recipe untested in Car Dealer is the one that only exists once the opponent
speaks: turning a free-text utterance into a per-type likelihood. CD-NOUL-1 isolates that step
against an exact ceiling, and compares the resulting architecture with the LLM-as-agent on the
same conversations.

What transfers from 20Q unchanged: LLM as likelihood; code as belief + selector; the Noul
matrix precomputed once and reused across arms (`twenty_questions/jev/matrix.py`).
What has to change (established by this program):

| 20Q | Car Dealer | Evidence |
|---|---|---|
| selector = EIG | selector = exact value planner over ask / offer / accept / walk | elicit toy (EIG ≠ VOI), R4 (value content, entropy potential fails) |
| truthful oracle answerer | strategic responder; planner and filter assume a **fixed hedge** on answers | CD-MIS-2 (`C_ans0.2` recovers the evasive loss; filters/detectors inert) |
| no realizer needed | an LLM must phrase moves and read replies; it is handed the **action**, never the posterior | CD-SEL-1 (shown P(accept) → SAFE 94 %, P1 −0.058) |

## 2. The game and the planner (reused, untouched)

`llm_rl_scripts/car_dealer_live/`: env `live-car-dealer-v0.1`, the registered CD-VOI-1 cell —
H = 4, c_q = 0, regime easy (r = 2/5), kernel T1 α₀ = 4/5, β = 8, p_counter = 1, p_hold = 1,
shortlist 3; the planner-complete shortlist menu (`menu_live.py`); the exact oracle (`oracle.py`)
with the hedged answer model of `filters_mis.FilterSpec(answer_noise=a)`; 144 cells
(2 roles × 2 openers × 6 focal × 6 opponent types), focal-opens cells primary, Y = net/(1 − r).
Nothing in `car_dealer_live` changes; CD-NOUL-1 plugs a **decoder** into the ASK branch of the
filter and leaves the DP alone.

The types are the six strict rankings over price / warranty / delivery, the questions the nine
TOP / ABOVE / BOTTOM items of `car_dealer_binned/questions.py`. The hypothesis space comes
for free here exactly as it did in 20Q; the open-type-space problem is a named limit (§8).

## 3. Two stages; register and run Stage 1 first

### Stage 1 — verbalized opponent (the text layer alone)

The opponent is the **registered T1 kernel**: its moves and its truthful yes/no answers are
drawn exactly as in CD-VOI-1, so `exact_ask` (O_true) remains the exact ceiling. The only
addition: each answer is rendered into a free-text seller/buyer utterance by a **verbalizer**
(`claude-sonnet-5-5`, persona = the opponent's ranking) under two registered text regimes:

- `plain` — answer the question directly, in character, one or two sentences;
- `oblique` — in character, may answer indirectly, deflect, bundle in other preferences or
  sales talk; **must never state a falsehood about its own priorities** (the truthful-answer
  invariant that keeps the ceiling exact). Evasive-but-truthful is the regime CD-MIS-2 found
  harmful; here it is the regime where a text decoder can earn its keep.

Verbalizations depend only on (regime, answering role, question, opponent ranking), never on
the history, so they are precomputed as a **bank**: 2 regimes × 2 roles × 9 questions × 6
rankings × K = 20 variants = 4,320 utterances (one Sonnet call each, ~$15), frozen with
provenance (`verbalize.py`). Episodes sample a variant by seed. The same variant is served to
every arm in a cell (common random numbers: arm contrasts are paired on identical conversations).

In `plain` the verbalizer is told to convey the answer to the question asked and *nothing else*
about its ranking, so the binary-answer ceiling `exact_ask` is an upper bound for every decoder.
In `oblique` it may let its priorities show, so a type-level decoder can in principle exceed the
binary ceiling there; `oracle` is then the ceiling for *binary-answer* decoders only, and any
excess is reported as what the text carries beyond the yes/no.

**Decoders** (utterance → per-type likelihood vector, consumed by `filters_mis.update`):

| id | what supplies the likelihood | calls |
|---|---|---|
| `oracle` | the true yes/no (ceiling for decoding) = CD-VOI-1 `exact_ask` | 0 |
| `regex` | a keyword yes/no parser; unparseable → uninformative (0.5, 0.5) | 0 |
| `noul_answer` | Jev `nouls`, one question: "The speaker's truthful answer to Q is YES." → p_yes; likelihood = p_yes·L_yes + (1−p_yes)·L_no under the bank's known answer structure | 1 per bank item |
| `noul_type` | Jev `nouls`, six questions: "This speaker's priorities are a > b > c." → six scores, normalised row-wise then floored (`FilterSpec` background floor) | 1 per bank item |
| `sonnet_answer` | the same 2-way judgment from `claude-sonnet-5-5` as a probability in a structured reply (is the Noul specifically needed, or any LLM likelihood?) | 1 per bank item |

Every decoder is applied to every bank item **once**; the result is a frozen likelihood matrix
(the 20Q `matrix_jev.json` pattern). All planner arms then run **offline at scripted speed**,
exact-on-logs; only the agent arms below spend model calls per decision.

**Arms** (all on the identical menu and conversations):

- `plan_ask[d]` for each decoder d — exact DP with ASK, hedge a = 0.2 in both planner and
  filter (CD-MIS-2's registered recovery), belief updated by decoder d on text and by the
  matched filter on typed moves (moves stay typed in Stage 1);
- `plan_ask[d, a=0]` — same with no hedge (does the fixed hedge still help when the "noise"
  is decoding error rather than lying?);
- `plan_noask` — exact DP with ASK forbidden (no text, no decoder; CD-VOI-1 `exact_noask`);
- `jev_agent` — CD-VOI-1 `jev_ask_plain` with the typed answer in the state text **replaced by
  the utterance** (the status quo, re-run on the same conversations);
- `sonnet_agent` — `claude-sonnet-5-5` choosing from the same lettered menu given the same
  state text and utterances (the persona study's agent protocol; closes "is it just Jev?").

Agent arms see the own-utility column only, as in Phase 1 (no EIG, no P(accept), no posterior).

### Stage 2 — live LLM opponent (follow-on registration, after Stage 1 reads out)

`claude-sonnet-5-5` plays the opponent with a planted ranking and a strategic brief, choosing
its moves from the discrete action set with a decision token (the persona buyer's protocol) and
answering in free text, allowed to evade. There is then no exact kernel and no O_true; the
ceiling becomes the **clairvoyant** planner (true type revealed, as R3's O_true), and the
contrast is between architectures at equal information: `plan_ask[noul_*]` vs `regex` vs
`sonnet_agent` vs `jev_agent`, plus a Noul over the opponent's *offers* (the second likelihood an
LLM can supply; MIS-1 says the kernel's action likelihoods are already robust, so this is
expected small). Stage 2 adds the **realizer** (LLM phrases the planner's chosen move to the
LLM opponent; handed the action only). Its registration is written after Stage 1, using Stage 1's
decoder calibration as a prior — which is why it is not registered now.

## 4. Estimands and predictions (Stage 1, to be registered)

Units: Y = net gain / (1 − r), focal-opens cells primary, paired by cell, cluster bootstrap on
the (buyer ranking, seller ranking) pair, B = 5000, as in CD-VOI-1. δ = 0.02 as registered there.
Ceiling Δ*_menu = `exact_ask − exact_noask` is exact (no CI); "captured fraction" = contrast / Δ*.

| id | estimand | prediction |
|---|---|---|
| P1 (primary) | `plan_ask[noul_answer] − plan_noask`, regime `plain` | > 0, CI excludes 0; captured fraction ≥ 0.5. (Jev in CD-VOI-1: −0.010, i.e. −10 % of 0.102.) |
| P2 | `plan_ask[noul_answer] − plan_ask[regex]`, regime `oblique` | > 0 (the LLM likelihood earns its keep only when the text is indirect) |
| P3 | same contrast, regime `plain` | within δ of 0 (a parser suffices for direct answers) |
| P4 | `plan_ask[noul_type] − plan_ask[noul_answer]`, `oblique` | ≥ 0 (text carries more than the yes/no; exploratory, no sign required) |
| P5 | `plan_ask[d, a=0.2] − plan_ask[d, a=0]` for d ∈ {noul_answer, regex}, `oblique` | ≥ 0 (the fixed hedge is not harmful under decoding error) |
| P6 | ask selectivity of `sonnet_agent`: P(ask \| optimal) − P(ask \| not optimal) | < 0.2 (asks non-selectively; the persona agent asked persistently, Jev never — direction left open, magnitude registered) |
| P7 | `plan_ask[noul_answer] − sonnet_agent` and `− jev_agent`, both regimes | > 0 in both |
| P8 | decoding log-loss (bits per answer) `noul_answer` < `regex`, `oblique` | yes; and `oracle` captured fraction = 1 by construction (gate, not prediction) |

Mechanism diagnostics logged per decision as in Phase 1 (exact Q\* of every option, ASK-optimal
flag, regrets), plus per-ASK: decoder likelihood vector, posterior mass on the true type before
and after, log-loss, whether the decoded argmax answer matched truth.

Verdict rules (`verdicts_noul.py`): HELD / SOFT / REFUTED per prediction with the Phase 1
conventions (CI vs 0 and vs δ); the study's headline is the P1 captured fraction and the P2 − P3
difference (the decoder's value is *conditional on the text regime*, which is the claim).

## 5. Gates before the registration tag (Phase 0 of this study)

1. **Verbalizer never lies.** A blind hand-audit of 60 bank items per regime (ids hidden,
   truth hidden) labelled yes / no / unclear; registration requires 0 contradictions of the
   truthful answer in `plain` and ≤ 2 in `oblique` (those items are regenerated, the count is
   reported). The oblique regime must also *be* oblique: `regex` accuracy on the audit sample
   < 0.9, otherwise the regime is re-prompted (recorded as a Phase 0 iteration).
2. **Decoder plumbing.** Mock Jev / mock Sonnet backends; a likelihood matrix built through
   the client equals the direct computation; `oracle` decoder reproduces CD-VOI-1 `exact_ask`
   values per cell to Monte Carlo tolerance on the same seeds.
3. **Ceiling unchanged.** `exact_ask` / `exact_noask` root values equal the frozen
   `results/live_phase1_jev_h4_cq0_20261029/ceilings.json` exactly (the game did not move).
4. **Common random numbers.** Every arm in a cell receives the identical variant sequence
   (asserted in the runner).
5. **Hedge fixed before any data.** a = 0.2 from CD-MIS-2, written into the registration; a
   decoder-calibrated â is a named secondary, not the primary.

No gate uses the confirmatory seeds; no decoder rule changes after the tag.

## 6. Build plan (`llm_rl_scripts/car_dealer_noul/`)

| file | does | reuses |
|---|---|---|
| `verbalize.py` | bank builder: prompts per regime, `claude-sonnet-5-5`, resumable JSONL, provenance | persona `run.Client` pattern |
| `decode.py` | the five decoders → likelihood vectors; Noul/Sonnet matrices built once and cached as JSON with model + bank hash | `twenty_questions/jev/matrix.py`, `filters_mis.FilterSpec` |
| `arms_noul.py` | `plan_ask[d, a]`, `plan_noask`, text-state agent arms | `oracle.Oracle`, `menu_live`, `jev_policy_live.JevLivePolicy` (subclassed, state text takes an utterance) |
| `sonnet_agent.py` | Sonnet choosing from the lettered menu; decision token parse; malformed → logged, re-asked once | persona `buyer.parse_decision` |
| `run_noul.py` | `--phase {bank,matrix,pilot,confirmatory}`, tag check, seeds `(seed, cell, repeat)`, CRN | `run_phase1.py`, persona `run.head_has_tag` |
| `analyze_noul.py`, `verdicts_noul.py`, `render_noul_note.py` | contrasts, captured fractions, decoder diagnostics, rendered note (no typed numbers) | `analyze_phase1.py`, `render_phase1_note.py` |
| `test_noul.py` | the §5 gates + regeneration tests on frozen results | |
| `PREREGISTRATION_NOUL.md` | §3–§5 as registered, tag `prereg-noul-<date>` | |

Scale and cost (Stage 1): bank 2,160 Sonnet calls (~$5); decoder matrices 2,160 Noul calls ×
2 decoders (Jev) + 2,160 Sonnet judgments (~$3); planner arms free; `jev_agent` ≈ CD-VOI-1 scale
(5 repeats × 144 cells × 2 regimes ≈ 1,440 episodes, ~4 decisions each); `sonnet_agent` the same
(~5,800 calls, ~$25). Total well under $50; wall time dominated by the agent arms (~2 h, keep-awake).

## 7. Sequence

1. Phase 0: build; bank; blind verbalizer audit (gate 1); matrices; gates 2–5; pilot on
   non-confirmatory seeds (plumbing only, no reading of contrasts) → write the registration.
2. Tag; confirmatory run (planner arms offline, then the two agent arms).
3. Analyze once; render note; regeneration test; PR against fork main.
4. Read-out decides Stage 2's registration (live LLM opponent, realizer, Noul over offers).

## 7a. Phase 0 decisions (2026-11-08, before any bank item was generated)

| Decision | Choice | Why |
|---|---|---|
| Bank keyed by answering role | yes (buyer and seller voices), K = 20 per cell → 4,320 | the live game's opponent is either role; a seller and a buyer phrase "price matters most" differently |
| Model sees the truthful answer | yes, marked "for your reference only" | it is the only way to hold the never-false invariant; the audit checks it held |
| Thinking | default (adaptive, no parameter), `max_tokens` 600, one retry at 2,000 if no text | the persona study lost 5 stories to thinking consuming the budget |
| Audit sample | 60 per regime, balanced 30 yes / 30 no, seed 20261108, labels yes / no / unclear | `audit.py`; thresholds: 0 contradictions in `plain`, ≤ 2 in `oblique`, regex bank accuracy < 0.9 in `oblique` |
| Regex decoder | earliest yes / no cue wins; unparsed → uninformative (0.5) | a reasonable parser, not a tuned one; its cue lists are frozen before the bank exists |

**Decoder matrices (2026-11-08).** `decode.py build` applied the Noul decoders (one Jev call per
item carrying the two answer Nouls and the six type Nouls; `p_yes = y/(y+n)`, type vector scaled
to max 1) and the Sonnet 2-way judge (P(YES) as JSON) to every bank item once;
`results/noul_bank_20261108/matrix_{noul,sonnet}.jsonl` with provenance (model, prompt hash).
`decode.py calibration` wrote `decoder_calibration.json`: accuracy, uninformative share,
confident-wrong share and log-loss per regime and decoder, plus the decoders' behaviour on the
blind-audited items split by the reader's label. It is descriptive — no decoder rule, prompt or
threshold was changed after reading it — and the registration will cite it from the file.

**Arms and runner (2026-11-08).** `arms_noul.py` (arms, `TextBank`, `NoulCell`, `variant_plan`,
`play_episode`), `agents_noul.py` (`TextJevPolicy`, `SonnetAgent`), `run_noul.py` (phases, resume by
episode key, tag check, ceilings). Decisions made while building, all before any registration:

| Decision | Choice | Why |
|---|---|---|
| Filter floor | **none** (exact Bayes, `ETA = 0`) | CD-MIS's 1e-3 floor left mass on ruled-out types and made the a = 0 planner ask once more than Phase 1's `exact_ask`; the parity gate (`plan_ask-oracle-0` ≡ `exact_ask`, `plan_noask` ≡ `exact_noask`, decision for decision on the frozen Phase 1 seeds) requires exact Bayes. A confident wrong decode at hedge 0 can then zero the true type and the belief stays put — that arm's honest failure mode, recorded per episode. |
| Agents' menu belief | updated on answers by `noul_answer` at the registered hedge (`AGENT_MENU_DECODER`) | so `sonnet_agent` / `jev_agent` and `plan_ask-noul_answer-0.2` face identical menus and the contrast isolates the chooser; a menu built from the oracle decoding would hand the agent the answer's meaning through the SAFE / PLAN candidates |
| Evaluator's reference | `ask_optimal_true` and `decision_regret_true` under the a = 0 oracle at the exact matched belief | one reference for every arm's selectivity and regret, as in CD-MIS |
| CRN | opponent stream = Phase 1's `(seed, role, opener, i, j, k)`; bank variant = `(seed, cell, repeat, question)` | arm-independent conversations; Phase 1 parity |
| Agent letter parsing | first standalone capital letter naming an option; retry once; else first option, flagged | mirrors the Jev policy's malformed-Choice fallback |

Open for the registration: **the primary hedge.** Under a 20 % assumed flip rate the planner values
questions less and asks at a minority of the nodes where the exact planner would; in the `plain`
regime, where decoding is near-perfect, that is pure cost, while in `oblique` it is insurance against
confident misreads. The pre-registration must name a = 0.2 or a = 0 as primary (the other is P5)
before the confirmatory seed is used; the smoke run that showed the trade-off was offline, in the
scratchpad, with mock agents, and its contrasts were not read.

**Analysis pipeline (2026-11-08).** `analyze_noul.py` (unit = (arm, regime); `plan_noask` paired with
every regime; Phase 1's 36-type-pair cluster bootstrap and captured fraction reused from
`analyze_phase1`; contrast families `ask_value`, `decoder`, `vs_oracle`, `type_vs_answer`, `hedge`,
`chooser`, `text_invariance`; the same pairs in questions per episode and, for decoder pairs, in
in-episode answer log-loss; selectivity against `ask_optimal_true`; decoder and agent diagnostics),
`verdicts_noul.py` (P1–P8 as in §4, gates: pipeline, `plan_ask-oracle-0 − plan_noask` reproduces
Δ\*_menu, oracle decoder text-invariant; `PRIMARY_HEDGE` is the registration's one open constant),
`render_noul_note.py` (note from the two JSON files, no typed numbers). Tests run the pipeline on
synthetic records with planted effects and a schema-only pass over the offline smoke run, whose
contrasts stay unread.

## 8. Limits named now

- The type space is the six rankings — binning is free, as in 20Q. A user's real counterpart
  has an open type space; discovering *which* bins matter for value is Stage 3, not this study.
- Stage 1's opponent does not lie and does not adapt its moves to the text. The strategic
  responder is Stage 2.
- The question bank is fixed and templated; the focal's own language is not under test here.
- `claude-sonnet-5-5` is both verbalizer and (in two arms) judge/agent: the `sonnet_answer`
  decoder is included so that any Noul-vs-Sonnet gap is read against a same-family judge, and
  the bank is frozen before any decoder runs.
