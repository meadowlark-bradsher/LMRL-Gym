# ST-6 Salience — pre-registration

**Status: DRAFT, not registered.** Stage 0 is frozen when tagged
`prereg-stress-st6s0-<date>`. Stages 1–2 are registered separately, after Stage 0's
gate, because their error model is built from Stage 0's measurements.
Shared infrastructure, scorecard and mechanisms: `STRESS_DESIGN.md`.

## Question

In every study so far, how often an object appears and how well the model knows it
are independent. In deployment they are not. The rare tail is where the model knows
least, and the model is in three places in the loop:

* it built the Noul row the planner believes (`matrix.py`)
* it answers the questions
* it fills the row of every object the expert admits

**Salience** is how readily an object comes to the model's mind. This study measures
it, tests whether the model's knowledge degrades with it, and if so feeds
progressively lower-salience secrets through the architecture.

## Stage 0 — measurement (API)

### Object universe

Within the existing 17 categories (`CATEGORY_PHRASE`):

* the 158 benchmark objects
* every distinct name produced by the fluency sampling below
* a **deep tail**: up to 40 names per category taken from a taxonomy (WordNet
  hyponyms, or Wikidata subclass members) that the fluency sampling never produced.
  These have salience 0 by construction.

Names are canonicalised: lowercased, singularised, and merged with the benchmark's
`variants`. Ambiguous merges are listed and resolved by hand before registration.

### Salience

* **Fluency salience s_f(o):** for each category, 200 samples at temperature 1.0 of
  "Name one kind of {phrase}." s_f(o) is the fraction of samples naming o.
* **Log-probability salience** (if the Jev client exposes token log-probabilities): the
  mean log-probability of o's name given the same prompt. It is recorded alongside s_f;
  s_f is primary.
* **World-frequency anchor:** Wikipedia pageviews over a fixed 12-month window. It
  separates "the model rarely thinks of it" from "it is rare in the world". The
  objects where those disagree are reported separately.

Objects are binned into salience deciles. The deep tail is its own bin.

### Knowledge quality, per object

* **q1 — self-consistency:** the row is elicited k = 3 times with `matrix.py`'s
  prompt; the mean binarized disagreement between repeats.
* **q2 — cross-model disagreement:** the binarized disagreement with a second model's
  row. This is the same second model as ST-4's A1, built with the same code.
* **q3 — ambiguity share:** the fraction of cells with 0.2 < p < 0.8.
* **q4 — error against ground truth**, on a slice. These are the attributes of the 44
  that Wikidata can decide for animals and plants (to be listed at registration).
  Optionally, a hand-labelled set of 40 objects stratified by salience.
* **q5 — the guess channel**, measured with the live answerer prompt:
  * *false-no:* when the secret is o, the answer to "Is it {o}?" is not yes
  * *false-yes:* when the secret is o, the answer is yes to the name of o's nearest
    known neighbour (binarized Hamming)

  Each is asked 3 times per object.

### Cost (estimate, recorded against actuals)

* fluency sampling: 17 × 200 calls
* rows: about 3 × |universe| Jev calls, with |universe| expected at 600–900
* second model: |universe| calls
* q5: about 6 × |universe| calls

### Stage 0 predictions

* **P0.1 — knowledge degrades with salience.** Spearman correlation between s_f and
  q2 ≤ −0.3. The bottom-decile mean q2 is ≥ 2× the top decile's. The deep tail's is
  higher still.
* **P0.2 — so does consistency, more weakly.** The same direction holds for q1 and q3,
  with Spearman ≤ −0.2.
* **P0.3 — the tail is where the model is consistently wrong.** On the q4 slice,
  confident errors (|p − 0.5| > 0.3 and on the wrong side of ground truth) are ≥ 2×
  as frequent in the bottom half of salience as in the top half. These are the errors
  q1 cannot see.
* **P0.4 — the guess channel is not exact in the tail.** The false-no rate on the true
  name is ≥ 0.02 in the deep-tail bin. The false-yes rate to the nearest neighbour's
  name is higher in the bottom salience half than in the top half.

### Gate

Stages 1–2 proceed only if **P0.1 holds**. If it fails, the result is written up as
"no salience gradient in the model's attribute knowledge at this resolution" and the
study stops. That is a useful negative: it would mean the §3.11 caveat about oracle
calibration is not concentrated in the tail.

## Stage 1 — offline simulation (registered after the gate)

Its error model comes from Stage 0. The planned shape:

* **Belief** = Jev's first row sample. **Answerer** = Jev's second row sample (the
  realistic deployment: the same model answers on another call). In a second arm,
  the answerer is the second model.
* **Guess channel** = q5's measured false-no and false-yes rates, per salience bin.
* **Rows on admission** = the LLM-elicited row, not the true one. This is the
  realistic expert path.
* **Three stream arms over the same universe:**
  * *coupled:* Zipf rank = salience rank, so common objects are well known
  * *decoupled:* the same per-object errors, with Zipf ranks shuffled
  * *matched-mean i.i.d.:* A0 flips at the coupled arm's game-weighted mean rate
* Backends `soft`, `hard`, `modeswitch`; shared scorecard with `false_confirm`; 3
  seeds.

Predictions planned for Stage 1 (to be fixed at its registration):

* **Coupling is kinder on average and harsher in the tail.** Overall in-pool failure
  is lower in the coupled arm than in matched-mean i.i.d., because the head is drawn
  most and known best. But in the bottom salience bins, autonomous resolution falls
  below §3.11's size-biased curve (0.375 in its rarest bin) by ≥ 0.1.
* **The tail is where `false_confirm` lives.** Any `false_confirm` under the measured
  guess channel is concentrated in the bottom salience half (≥ 75% of them).
* **The rescue fades in the tail.** `modeswitch`'s recovery over `hard` is smaller in
  the bottom salience half, because tail errors are sticky (P0.3) and stack into more
  than one killer.

## Stage 2 — live confirmation (registered after Stage 1)

About 300 live games with Jev answering, secrets stratified across salience bins, under
`hard` and `modeswitch`. It checks that Stage 1's per-bin failure rates match live play
within their seed spread. If they don't, the offline error model is wrong, and Stage 1's
conclusions are withdrawn, not patched.

## Relation to the other studies

* ST-6 supplies measured rates for the placeholders in ST-2's G arm and ST-4's G arm.
  If Stage 0 lands before those studies are registered, their grids are set from it.
* ST-4's A1 and ST-6's q2 share the second model's matrix and its code.
* A low-salience object that resembles a common one is the joint worst case of ST-2
  and ST-6. Stage 1 reports it as a slice: deep-tail objects whose nearest known
  neighbour is in the top salience decile.
