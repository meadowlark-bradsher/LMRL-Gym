# CD-NOUL-2 — the live LLM opponent (Stage 2 plan, 2026-11-08, DRAFT — not registered; §3.1a/§3.1b added after the offline Phase 0 calibration)

Follow-on to CD-NOUL-1 Stage 1 (`DESIGN_NOUL.md`, `PREREGISTRATION_NOUL.md` tag `prereg-noul1-20261108`,
`RESULTS_NOUL.md`, FINDINGS §3.12). Stage 1 kept the registered T1 kernel as the opponent and verbalized only
its yes/no answers, so the exact ceiling survived. Stage 2 replaces the opponent with `claude-sonnet-5-5`
playing a planted type under a strategic brief, adds the realizer that phrases the planner's moves, reads
the opponent's offers as a second likelihood, and carries the two fixes Stage 1's read-out asks for. This
note fixes the design; the registration is written after its Phase 0 gates pass and is a separate document.

## 1. What Stage 1 settled, and what it left open

Numbers are Stage 1's frozen results (`results/noul_stage1_20261108/{analysis,verdicts}.json`,
`results/noul_bank_20261108/decoder_calibration.json`), Y units, focal-opens cells, 95 % cluster CIs.

Settled:

- **The seat matters more than the model.** On identical menus the exact planner fed by Jev's `Noul`
  likelihood beats Jev-as-chooser by +0.159 [+0.130, +0.192] and Sonnet-as-chooser by +0.160 [+0.130,
  +0.191] in `plain`; both agents ask at ≤ 6 of 703 ASK-optimal decisions and finish about a full ceiling
  *below* the planner that never asks. Stage 2 keeps both chooser arms as the comparison the architecture
  is measured against.
- **On guarded text a model decoder is necessary** (Noul − parser +0.045 [+0.029, +0.062]; log-loss
  −1.47 bits) and **the parser is harmful under an exact filter** (51 episodes with the true type zeroed
  at a = 0). Stage 2 keeps `regex` as the control and never runs a confident-wrong-capable decoder
  without a hedge.
- **Type-reading pays when priorities leak** (`noul_type − noul_answer` +0.028 [+0.014, +0.043] in
  `oblique`). A live opponent's offers and phrasing leak priorities continuously, which is the premise of
  the offer likelihood below.

Open, and addressed here:

- **The plain-text calibration gap (P3).** Noul reads an unambiguous yes at ≈ 0.9 (0.197 bits log-loss
  on the plain bank; the Sonnet judge 0.085). Under the exact filter that residual costs −0.027 [−0.044,
  −0.012] against the parser on clear text and turns 0.040 of clear negotiations into walk-outs at a = 0.
  §3.1 adds a calibration layer fit on Stage 1's frozen bank.
- **The fixed hedge cuts both ways (P5 and its mirror).** The CD-MIS-2 flip rate a = 0.2 bought +0.033
  [+0.009, +0.058] on guarded text and cost −0.028 [−0.047, −0.010] on clear text with the same decoder,
  and the *oracle* planner at a = 0.2 captures only 0.76 of the ceiling because the hedged kernel asks at
  0.425 of the ASK-optimal nodes. §3.2 replaces the one global number with a per-question flip rate
  derived from the decoder's own measured uncertainty, and lets the filter use each reply's read directly.
- **The opponent never lied and never adapted.** Stage 1's ceiling was exact because the opponent was the
  kernel. §2 says what the ceiling becomes when it is not.

## 2. What changes

### 2.1 The opponent: `claude-sonnet-5-5` with a planted ranking and a strategic brief

The opponent is given its role (buyer or seller), its planted strict ranking over price / warranty /
delivery, its reservation r = 2/5 in utility terms, the horizon, the public history so far, and a brief.
At its turn it sees the legal moves of `protocol_live.legal_actions` lettered (ACCEPT / REJECT / HOLD /
PROPOSE deal / ASK question / WALK) and must emit a decision token and an utterance:

```
DECISION: <letter>
SAY: <one to three sentences in its own words>
```

When asked a question it emits `SAY:` only. The parser is the persona buyer's (`car_dealer_persona/buyer.py`
`parse_decision` pattern): a reply without a parseable token is retried once with a larger budget, then
recorded malformed and the move defaults to the kernel's sampled move (counted; gate in §6). Moves are
therefore **typed** — the game engine, the menu and the evaluator are unchanged — and only the opponent's
*words* are free text. Three briefs, each a frozen prompt whose hash goes into the registration:

| brief | answers | moves | role in the design |
|---|---|---|---|
| `candid` | truthful, may be indirect (Stage 1's `plain`–`oblique` range arises naturally) | its own | positive control for the decoders; the calibration fix is tested here |
| `guarded` | may deflect, hedge, decline, answer a different question; **never states a falsehood** | its own | **primary** — the strategic responder Stage 1 deferred |
| `adversarial` | may mislead about its priorities | its own | secondary, three arms only (§4): the case the hedge was built for |

The `never states a falsehood` clause is audited blind, as Stage 1's verbalizer was (§6). A lying opponent
is kept to the secondary brief because type inference from testimony is the premise under test; an
opponent that lies at will is a different study (named in §10).

### 2.2 The ceiling: clairvoyant under the assumed kernel

There is no exact opponent model, so there is no exact ceiling. Two references replace it, both empirical
and both played against the same live opponent:

- **`plan_noask`** — the planner with ASK forbidden (as in Stage 1).
- **`plan_reveal`** — the planner with the opponent's true ranking given (belief = point mass; asking is
  worthless and never chosen). This is R3's `O_true` idea (`car_dealer_drl/README.md`) with the planner's
  own assumed kernel kept: it is *the value of perfect type information to this planner against this
  opponent*, which bounds what any decoder can deliver by inference.

Δ_clair = `plan_reveal − plan_noask` is the Stage 2 ceiling; captured fraction = (`plan_ask[d] −
plan_noask`) / Δ_clair, with both arms' per-cluster means paired as in Stage 1. Δ_clair is itself a gate
(§6): if knowing the type is worth less than δ against the live opponent there is no inquiry value to
recover and the study is void for its primary.

### 2.3 The planner keeps an assumed kernel; Phase 0 fits it

The exact planner needs p(opponent move | type). Stage 2 keeps the T1 kernel form (`opponent.py`:
aspiration falling to r, softmax(β·U) proposals over a shortlist, accept iff U ≥ aspiration, hold/concede)
and asks only which **parameters** fit the LLM opponent. Phase 0 plays the opponent against scripted focal
probes (the Stage 1 `plan_noask` and `plan_ask-oracle-0` move sequences, 2 roles × 6 types × 20
episodes per brief) and fits `KernelParams` (α₀, β, p_counter, p_hold, accept_noise) by maximum likelihood
of the opponent's observed moves; the registered T1 parameters are the alternative. The fitted kernel is
adopted for a brief only if it beats T1 in held-out move log-likelihood (§6), and the registration pins
both parameter sets. CD-MIS-1 says the planner is robust to kernel misspecification of every kind it swept
except evasive answers, so the primary is read under the fitted kernel and `plan_ask[noul_answer]` under
T1 is a secondary arm — the one that says whether the fit mattered.

### 2.4 The realizer

The planner's chosen move is phrased for the LLM opponent by `claude-sonnet-5-5` from the **action and
the public history only** — never the belief, never a probability, never the planner's values (CD-SEL-1's
lesson in the other direction: the model is handed the action). A PROPOSE is phrased as an offer of that
deal; an ASK as the bank question in natural words; ACCEPT / REJECT / HOLD / WALK as a sentence. The
realized text must **round-trip**: a parser (the Stage 1 `regex` cue lists for yes/no are not enough here;
the realizer's output is checked by a structured re-read through the same decision-token parser the
opponent uses, i.e. the realizer is asked to emit `DECISION:`/`SAY:` for its own move) must recover the
same action in ≥ 0.99 of a pilot's moves, and the `SAY:` must not add a commitment (no second deal named,
no concession offered). The `template` arm (Stage 1's templated question text and a templated offer line)
is the control that says whether phrasing moves the opponent at all.

### 2.5 The second likelihood: `noul_offer`

Every opponent move arrives with words. Stage 1 read only answers. `noul_offer` asks Jev's `nouls`, on the
opponent's PROPOSE / REJECT / HOLD utterance plus the typed move, the six statements "the counterpart's
priorities are a > b > c" (the `noul_type` prompt of `decode.py` with the move in the context), scales the
vector to max 1 and multiplies it into the filter update **alongside** the kernel's move likelihood — the
utterance is treated as a second observation, conditionally independent of the move given the type, at
tempering τ = 1. CD-MIS-1 found the kernel's action likelihoods already robust, so the prediction is a
small non-negative gain; the arm exists because an offer is the one utterance a counterpart cannot refuse
to make.

## 3. The two fixes from Stage 1's read-out

### 3.1 Calibration layer for the answer decoder (`noul_answer+cal`)

A monotone map g: p_yes → p'_yes fit by maximum likelihood on Stage 1's frozen Noul matrix
(`results/noul_bank_20261108/matrix_noul.jsonl`, 4,320 items with known truth): one-parameter temperature
on the log-odds, fit separately per Stage 1 regime and then pooled — the pooled map is the one registered,
and the per-regime maps are reported to show how much the right temperature depends on how guarded the
text is. The fit uses only Stage 1 data, which is already collected and frozen, so it is a pre-registration
constant, not a tuned quantity; Stage 2's live text is its test. Prediction: on `candid`, `noul_answer+cal`
closes to within δ of `sonnet_answer` (whose raw log-loss 0.085 bits Stage 1 found good enough to lose only
−0.002 to the oracle on clear text). The same temperature is applied to `sonnet_answer` as a secondary
(it should be near 1).

### 3.1a Phase 0 finding (2026-11-08, offline, from the frozen Stage 1 matrices)

`calibrate_noul.py` was run on the 4,320-item Stage 1 bank (`results/noul_bank_20261108/calibration_noul2.json`;
numbers below are read from it). A single temperature does what §3.1 asked on clear text and something
unwanted on guarded text:

| decoder | map | T | b | acc. oblique | log-loss plain | log-loss oblique | confident-wrong oblique (of 2,160) | E[read error] plain | E[read error] oblique |
|---|---|---|---|---|---|---|---|---|---|
| `noul` | raw | 1.00 | +0.00 | 0.712 | 0.197 | 0.691 | 0 | 0.125 | 0.289 |
| `noul` | pooled | 2.07 | +0.00 | 0.712 | 0.041 | 0.678 | 18 | 0.027 | 0.183 |
| `noul` | safe | 1.17 | +0.00 | 0.712 | 0.150 | 0.675 | 0 | 0.096 | 0.265 |
| `noul` | platt | 3.62 | +1.97 | 0.815 | 0.003 | 0.520 | 17 | 0.002 | 0.169 |
| `sonnet` | raw | 1.00 | +0.00 | 0.830 | 0.085 | 0.584 | 0 | 0.057 | 0.300 |
| `sonnet` | pooled | 3.93 | +0.00 | 0.830 | 0.001 | 0.395 | 18 | 0.000 | 0.133 |
| `sonnet` | safe | 2.21 | +0.00 | 0.830 | 0.004 | 0.435 | 0 | 0.002 | 0.194 |
| `sonnet` | platt | 5.61 | +1.33 | 0.900 | 0.001 | 0.323 | 13 | 0.000 | 0.100 |

- **On `plain` every map helps and none can hurt**: the raw decoders are right on all 2,160 items, so the
  plain-only fit runs to the search ceiling (T = 50) and the pooled T already takes Noul from 0.197 to 0.041
  bits.
- **On `oblique` the pooled temperature creates confident-wrong reads where the raw decoders had none** —
  18 for Noul and 18 for the judge, overlapping on only 3 items. They are not
  misleading texts: inspected, they are truthful indirect replies in which the speaker talks up its top
  attribute and brushes off the one the question named (e.g. asked "price over delivery?", a
  warranty-first buyer says it cares about coverage, not speed), and Noul reads the absence of the named
  attribute as a NO lean (the Stage 1 "NO-lean on deflections", p_yes ≈ 0.15–0.25 against a truth of YES). A
  temperature amplifies that bias; it cannot remove it.
- **The Platt fit (slope and intercept) can.** Noul's intercept comes out at b ≈ +2.0 logits toward YES —
  the size of the lean — and lifts oblique accuracy from 0.712 to 0.815 (the judge
  0.830 → 0.900) and oblique log-loss from 0.691 to 0.520 bits, but still leaves
  17 confident-wrong reads (judge 13). These fits are in-sample on the Stage 1 bank (two parameters on 4,320
  items; over-fitting is not the worry — the verbalizer's distribution is, which is why Stage 2's live text
  is the test).
- **`safe`** is the largest pure temperature with zero confident-wrong reads on the oblique bank: T = 1.17 for
  Noul buys little on clear text (0.197 → 0.150 bits); T = 2.21 for the judge buys almost everything
  (0.085 → 0.004).

What this changes in the design (decisions for the registration, listed with a recommendation):

1. **The calibration map is per brief**, as the a_q vectors already were: `candid` text is truthful and
   near-direct, where a sharp map is safe by the never-lies gate, so `candid` takes the Platt map; `guarded`
   takes the map the registration picks from the table above. Recommendation: **Platt with a read clamp**
   (below) for both, because a filter that cannot see a confident-wrong read coming is better protected by
   bounding every read than by blunting all of them.
2. **A read clamp** `p' ∈ [1 − c, c]`, c = 0.95, applied after the map in the `adaptive` filter: the
   boundary analogue of CD-MIS-1's background floor, at the read rather than the belief. It bounds the damage
   of a confident-wrong read at 19:1 odds against the true type (recoverable from the next move's kernel
   likelihood; Stage 1's zeroed-truth episodes came from reads of exactly 1.0) at the cost of capping a
   correct read's evidence at the same odds. c is a registered constant; c = 1 is "no clamp" and is the
   natural secondary.
3. **A new gate 9 (Phase 0 pilot):** on the `candid` pilot's blind-audited answers, the chosen map produces
   0 confident-wrong reads; on `guarded`, the count is reported against the bank's expectation
   (17/2,160 ≈ 0.8 % for Noul under Platt).
4. The §4 arm `plan_ask-noul-adaptive` (raw Noul) stays, so the value of the map is measured, and a
   `plan_ask-noulcal-adaptive-noclamp` secondary (c = 1) measures the clamp.

`decode.calibrated_p_yes` is the pure-temperature read; `calibrate_noul.read(p, T, b)` the Platt read; the
registration pins (T, b, c) per decoder × brief and the a_q vectors computed under the same map.

### 3.1b Decision (2026-11-08, NEG-9)

Taken as recommended, and binding for the registration unless the Phase 0 pilot's gate 9 fails:

- **Map:** the Platt map (slope and intercept fit pooled on the frozen Stage 1 Noul matrix,
  `calibration_noul2.json` → `decoders.noul.candidates.platt`) for both `candid` and `guarded` (and for
  `adversarial` where run). `MapSpec(name="platt")`.
- **Clamp:** every adaptive read is clipped to [0.05, 0.95] (c = 0.95) before it enters the filter.
- **Secondary:** `plan_ask-noulcal-adaptive-noclamp` (c = 1), `guarded` only, so the clamp's cost and
  protection are measured rather than assumed.
- **Gate 9** (added to §6): on the `candid` pilot's blind-audited answers the Platt-mapped, clamped read
  produces 0 confident-wrong reads (p' > 0.9 against a truthful NO, or < 0.1 against a truthful YES); on
  `guarded` the count is reported against the bank's expectation. If gate 9 fails on `candid`, the fallback
  is the `safe` temperature for `candid` and the decision is re-taken before the tag.
- The a_q vectors of §3.2 are computed under the same Platt map (`candidates.platt.a_q`), per brief via
  the regime map `candid → plain`, `guarded` / `adversarial` → `oblique`.

### 3.2 Uncertainty-scaled hedge (`a = adaptive`)

Stage 1 used one flip rate a in two places: the planner's ASK branch (the hedged kernel's
`answer_noise`) and the filter's update (`FilterSpec(answer_noise=a)`). The adaptive scheme separates
them:

- **Filter:** the read p'_yes (calibrated) is used directly, `answer_vector(q, p'_yes, noise = 0)` — the
  decoder's own uncertainty *is* the hedge, reply by reply; a deflection decodes near 0.5 and moves the
  belief little, a clear yes moves it almost fully.
- **Planner:** the ASK branch for question q uses a per-question flip rate a_q equal to the decoder's
  expected read error on that question, E[min(p', 1 − p')], estimated from the Stage 1 bank for the
  regime nearest the brief (`plain` for `candid`, `oblique` for `guarded` and `adversarial`) and frozen in
  the registration (one 9-vector per decoder × brief). This needs the hedged kernel to accept a
  per-question noise map: the planner's hedge enters through `LiveConfig.answer_noise` (Stage 1's
  `NoulCell.hedged` builds the hedged kernel from it) and the filter's through `FilterSpec.answer_noise`;
  both gain an optional `answer_noise_by_q` whose default reproduces the global behaviour exactly (parity
  test, as in Stage 1's `test_arms_noul.py`).

Prediction: `adaptive ≥ fixed 0.2` on both `candid` and `guarded` (the fixed hedge lost on clear text and
under-asked everywhere), and `adaptive ≥ fixed 0` on `guarded` (the exact filter with soft reads made
asking negative in `oblique` at a = 0, −0.061 [−0.092, −0.030] over all cells). The `adversarial` brief is
where this scheme can fail — a confident lie decodes as a confident truth and the filter has no floor —
and that is why `adversarial` is kept, as a secondary, with the fixed hedge beside it.

## 4. Arms

Planner arms share the shortlist menu, the fitted kernel (§2.3) and the realizer; chooser arms see the
same menus with their menu belief decoded by `noul_answer+cal` at `adaptive` (so, as in Stage 1, the
chooser is isolated).

| arm | decoder(s) | hedge | briefs | role |
|---|---|---|---|---|
| `plan_ask-noulcal-adaptive` | `noul_answer+cal` | adaptive | candid, guarded, adversarial | **primary** |
| `plan_ask-noulcal-0.2` | `noul_answer+cal` | fixed 0.2 | candid, guarded, adversarial | the Stage 1 hedge |
| `plan_ask-noul-adaptive` | `noul_answer` (raw) | adaptive | candid, guarded | isolates the calibration layer |
| `plan_ask-sonnet-adaptive` | `sonnet_answer` | adaptive | candid, guarded | same-family judge |
| `plan_ask-regex-0.2` | `regex` | fixed 0.2 | candid, guarded | parser control (never at a = 0) |
| `plan_ask-noulcal+offer-adaptive` | `noul_answer+cal` × `noul_offer` | adaptive | candid, guarded | the second likelihood |
| `plan_ask-noulcal-adaptive-T1` | `noul_answer+cal` | adaptive | guarded | registered T1 kernel instead of the fit |
| `plan_ask-noulcal-adaptive-template` | `noul_answer+cal` | adaptive | guarded | templated text instead of the realizer |
| `plan_ask-noulcal-adaptive-noclamp` | `noul_answer+cal` | adaptive, c = 1 | guarded | the read clamp's effect (§3.1b) |
| `plan_noask` | — | — | all | reference |
| `plan_reveal` | — | — | all | clairvoyant ceiling |
| `sonnet_agent` | menu belief only | — | candid, guarded | LLM-as-chooser |
| `jev_agent` | menu belief only | — | candid, guarded | Jev-as-chooser |

Cells as Stage 1: 2 roles × 2 openers × 6 focal × 6 opponent types = 144 per (arm, brief); focal-opens
primary. Common random numbers cannot reach a sampled LLM opponent, so paired contrasts rely on the
cluster structure (36 type pairs) and on K; the opponent is called at default sampling and every call's
text is stored.

## 5. Estimands and predictions (to be registered after §6)

Y = net gain / (U_max − r); δ = 0.02; CAPTURED_MIN 0.5; cluster bootstrap B = 5000 over the 36 type pairs.

| rule | quantity (focal opens unless stated) | prediction |
|---|---|---|
| **Q1 primary** | `plan_ask-noulcal-adaptive − plan_noask`, `guarded` | > 0 and captured ≥ 0.5 of Δ_clair |
| Q2 | `plan_ask-noulcal-adaptive − sonnet_agent` and `− jev_agent`, `guarded` and `candid` | > 0 (the chooser result replicates against a live opponent) |
| Q3 | `adaptive − 0.2` for `noul_answer+cal`, `candid` and `guarded` | ≥ 0 on both |
| Q4 | `noul_answer+cal − sonnet_answer` at `adaptive`, `candid` | within δ (the plain-text gap is closed) |
| Q5 | `noul_answer+cal − noul_answer` at `adaptive`, `candid` | > 0 (the calibration layer is what closes it) |
| Q6 | `noul_answer+cal − regex`, `guarded` | > 0 (Stage 1's P2 against a strategic responder) |
| Q7 | `+offer − without`, `guarded` | ≥ 0 (expected small; exploratory) |
| Q8 | `fitted kernel − T1`, `guarded` | ≥ −δ (the fit does not hurt; whether it helps is reported) |
| Q9 | `realizer − template`, `guarded` | ≥ −δ (phrasing does not hurt) |
| Q10 | `adaptive − 0.2` for `noul_answer+cal`, `adversarial` | reported, no sign predicted — the hedge's designed case |
| Q11 | sonnet_agent selectivity P(ask \| optimal) − P(ask \| not optimal), both briefs | < 0.2 |
| Q12 | in-episode answer log-loss, `noul_answer+cal − regex`, `guarded`, all cells | < 0 |

"ASK-optimal" for Q11 and for the selectivity tables is defined under the fitted kernel at the exact
matched belief with the true ranking's answer — the Stage 1 evaluator with the assumed kernel swapped in;
it is a reference, not a truth, and is labelled so.

## 6. Gates before the registration tag (Phase 0 of Stage 2)

1. **Opponent protocol.** Decision-token well-formed ≥ 0.99 over the pilot's opponent moves per brief;
   malformed defaults counted and ≤ 1 %.
2. **Never-lies audit** (`candid`, `guarded`). From the pilot, 60 answers per brief, blind-labelled against
   the planted ranking with the sealed key, as Stage 1's `audit.py`: 0 contradictions in `candid`, ≤ 2 in
   `guarded`. The `adversarial` brief is audited for the opposite — that it does in fact mislead in ≥ 20 %
   of answers — else it is dropped as inert.
3. **Guarded is guarded.** The `regex` parser's accuracy on the `guarded` pilot answers < 0.9 (Stage 1's
   oblique gate; Stage 1 measured 0.125).
4. **Kernel fit.** Fitted `KernelParams` beats T1 in held-out move log-likelihood on the pilot by ≥ 0.05
   nats per move, else T1 is kept for that brief; both pinned.
5. **Realizer fidelity.** Round-trip action recovery ≥ 0.99 and 0 added commitments in 200 realized moves
   per role, judged blind.
6. **Clairvoyant gap.** Δ_clair = `plan_reveal − plan_noask` ≥ δ on the pilot for `guarded` (focal opens);
   else the primary is void and Stage 2 reports the chooser contrasts only.
7. **Parity.** With a scripted T1 opponent, templated text and the `oracle` decoder, the Stage 2 runner
   reproduces Stage 1's `plan_ask-oracle-0` decisions exactly; the per-question noise map with all
   entries equal reproduces the global hedge exactly.
8. **Calibration constants.** The Platt map of §3.1b and the a_q vectors of §3.2 are computed from the
   frozen Stage 1 matrices by a committed script and their values are in the registration.
9. **No confident-wrong reads on candid text.** On the `candid` pilot's blind-audited answers the chosen map
   (Platt, clamp 0.95) produces 0 confident-wrong reads; on `guarded` the count is reported against the
   bank's expectation (§3.1b).

Pilot outcome contrasts are not read; only the gates above are.

## 7. Build plan (`llm_rl_scripts/car_dealer_noul/`, new files; nothing in `car_dealer_live` changes
behaviour)

- `opponent_llm.py` — the three briefs, state rendering, lettered legal moves, decision-token parse with
  one retry, default-to-kernel fallback, per-call logging; `MockOpponent` for tests.
- `realizer.py` — action + history → `DECISION:`/`SAY:` text; `template` fallback; round-trip check.
- `kernel_fit.py` — maximum-likelihood `KernelParams` from logged opponent moves; held-out score.
- `calibrate_noul.py` — temperature fit on `matrix_noul.jsonl`; a_q vectors per decoder × regime; writes
  `results/noul_bank_20261108/calibration_noul2.json`.
- `decode.py` — `noul_offer` (reuses `noul_questions` type statements with the move in context); the
  calibrated wrapper; live decoding (no matrix — the text is new each episode).
- `protocol_live.LiveConfig` / `oracle.py` / `filters_mis.FilterSpec` — optional `answer_noise_by_q` (default =
  global; parity test). Additive only: every existing call path is unchanged.
- `arms_noul2.py`, `run_noul2.py` — arms of §4, briefs, resumable runner (episode key = arm, brief, cell,
  repeat), worker pool over cells (the opponent, realizer and judge are all network calls), tag check.
- `analyze_noul2.py`, `verdicts_noul2.py`, `render_noul2_note.py`, tests incl. planted effects and a
  regeneration test once the run is frozen.

## 8. Budget (from Stage 1's measured usage)

Stage 1's `sonnet_agent` spent 940,323 input and 453,302 output tokens on 1,131 decisions (720 episodes,
`plain`), i.e. about $6.4 per 720 episodes at $2 / $10 per MTok, in ~90 minutes sequential. A Stage 2
episode makes 2–4 opponent calls, 2–4 realizer calls and 1–3 decoder calls (Jev ~0.1 s, free of Sonnet
cost; the Sonnet judge arm adds one call per answer). Estimate ≈ $12–15 and ~2 h per 720 episodes
sequential; the §4 table is 27 (arm, brief) pairs → 19,440 episodes at K = 5 ≈ $325–405 and ~54 h
sequential, ~7 h with 8 cell-parallel workers. If that is over budget the registered cut is K = 3 (432
episodes per pair, 11,664 episodes, ≈ $195–245) with the chooser arms kept at K = 5; the cut is decided
before the tag.

## 9. Sequence

1. Phase 0: briefs and realizer prompts frozen → pilot (2 roles × 6 × 6 types × 1 repeat × `plan_noask`
   and `plan_reveal`, per brief) → gates 1–6 → kernel fit → calibration constants (gate 8) → parity
   (gate 7).
2. `PREREGISTRATION_NOUL2.md`: rules Q1–Q12, arms, K, seed, hashes of briefs, prompts, calibration
   constants and fitted kernels; FROZEN + tag `prereg-noul2-<date>`.
3. Confirmatory run from the tag; single analysis; verdicts by code; note rendered from JSON;
   regeneration test; PR.

## 10. Limits named now

- The opponent's **moves are typed**; only its words are free. A fully free-text counterpart (offers
  phrased in money and terms, parsed into deals) is the realizer's mirror image and is not here.
- The focal's answers to the opponent's questions are truthful and templated (the focal's own candour is
  not under test).
- The ceiling is clairvoyant **under the assumed kernel**: a planner with a better opponent model could
  exceed it. Δ_clair bounds inference, not planning.
- The type space is still the six rankings. An open type space — discovering which dimensions of a
  counterpart matter — is Stage 3.
- One opponent model (`claude-sonnet-5-5`) realizes, judges and plays the chooser arm, as in Stage 1; the
  briefs are prompts, and prompt sensitivity is a named secondary (one paraphrase of `guarded`, if budget
  allows) rather than a swept factor.
- No common random numbers through a sampled opponent: intervals are wider than Stage 1's at equal K.
- The `adversarial` brief tests lying only as a secondary; a study of deception detection would need its
  own ceiling and its own registration.
