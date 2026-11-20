# More than one cause — pre-registration (Flywheel experiment 2, FLYW-7)

**Status (2026-11-19): REGISTERED** at tag `prereg-flyw2-compound-20261119` (commit 008e6da, the build commit on
branch `claude/flyw2-compound-prereg`; user: "Tag at S = 5"). PR #104 had merged only the first draft commit, so
the tag is on the branch commit, not the merge. Thresholds, seed counts and rules are as written below.
The open choices are decided (§Decisions; the recommended option on each). The runner, analyzer and blind seed
tool are built (`FLYW2_BUILD.md`), and the CX0 gate passes (jobq 77). The 5-seed smoke (jobq 78) was read only by
the blind seed tool, which sets **S = 5** (§Seeds). Apart from the disclosed noise-free calibration (§Basis), no
outcome has been read. The smoke's noisy games were played, and only their within-unit variances were computed.

Jira: epic FLYW-2. This is FLYW-7 (the two-secret oracle). FLYW-8 (fixes) and FLYW-9 (the two-fault composition
study in fault-graph) are decided from it. Design source: sre-flywheel README, "Experiment 2: more than one cause",
and gap 10 ("exhausted means ask the expert" is guaranteed only without noise).

## The question in plain words

Today's 20q loop (P0a) assumes the incident has exactly one cause. It eliminates candidates until one is left,
and when none are left it decides the cause is new and calls the expert. Real incidents can have two causes at
once.

**When two causes are present, how does single-cause elimination fail, and what does the failure cost?** The
README names two failures:

1. **False exhaustion.** No single candidate fits the answers, the survivor set empties, and the loop calls the
   expert for a "new" cause that is really two known ones. In the flywheel, that expert answer would be admitted
   as a new failure mode, so the graph learns a species that does not exist.
2. **A confident partial answer.** The loop names one of the two causes and stops. The other cause is left in
   place.

**What the answer decides.**

* **The production recommendation.** It says hard gate first, and treats an empty survivor set as the signal to
  call the expert. If compound causes routinely empty the survivor set, that signal is not clean and the
  recommendation needs revising before anything is built on it.
* **Whether FLYW-8's fixes are worth building** (pairs as hypotheses, explain-away, noisy-OR, linked causes).
  If single-cause elimination fails cheaply and safely, the fixes are not worth their cost.
* **Gap 10.** When the survivor set does empty, can the answers seen so far tell the three causes apart: a new
  cause, a noise error, or a compound?

**What the noise-free calibration already shows (disclosed, §Basis).** With exact answers, compounds almost never
empty the survivor set (1.6% of pairs). P0a names one of the two causes in 98.5% of pairs and raises no alarm in
97.6%. So the failure at flip 0 is the confident partial answer, not false exhaustion. If the incident is checked
after the fix, the partial answer costs P0a about one expert call more than a single-cause incident (+20.0 at the
primary costs). The scored tests ask whether this still holds once answers are noisy.

**Distance from the main question.** This is a side branch, like FLYW-1. The main question is why the negotiating
AI never asks and what fixes it. FLYW-7 belongs to the SRE flywheel: it tests whether the belief store's "one
cause" assumption survives two causes.

## Terms

| term | meaning |
|---|---|
| **secret / cause** | One of the 158 objects of the 20q matrix |
| **known pool** | The 141 objects P0a can name. The other 17 are **novels**: 10% of each category, rounded half up, never a category's last member, drawn with a fixed seed (`holdout_novels`) |
| **game kinds** | **single**: one known secret. **novel**: one novel secret. **KK**: a pair of known secrets (the compound games scored). **KN**: a known and a novel secret (reported) |
| **OR oracle** | The answer to a question is yes if *either* secret's exact answer is yes. Attributes use the matrix at threshold 0.5; category and by-name questions use the exact closed form |
| **flip** | The probability that an attribute answer is flipped. Category and by-name answers are never flipped |
| **action** | A by-name guess, FLYW-1's convention: in SRE, applying the remediation for that cause. A guess answered yes is a **right action**; answered no, a **wrong action** |
| **probe** | A question or an action |
| **verification** | After a right action, the loop sees whether the incident has cleared. It clears when every cause has been acted on. Exact and free |
| **named one / both / neither** | How many of the game's secrets were confirmed by a right action |
| **exhaustion** | P0a's live set empties on an answer, whether or not a reinstate then revives a candidate. **Exhaustion escalation**: the game ends with an expert call because the set is empty after the reinstate budget |
| **false exhaustion** | An exhaustion escalation in a game whose secrets are all known (single or KK) |
| **catch-all fires** | The soft posterior, run in shadow on the same answers with the Pitman-Yor catch-all prior, puts ≥ 0.9 on the catch-all at some turn |
| **partial answer** | A KK game that names one secret and stops. **Silent partial**: also no exhaustion at any turn and no catch-all firing, so nothing flagged it |
| **pair share w** | The share of incidents that have two causes |

## Design

### The arm: today's loop (S)

S is P0a exactly as in FLYW-1's arm A: `run_openworld_episode` with backend `modeswitch_guard`, reinstate budget
K = 3, `no_repeat` on, the graded matrix likelihood (ε = 0.05), confidence 0.9, 20 probes, `always` end rule. Its
code is unchanged. Only the oracle is new (`KeyedSetOracle`, `flyw2_compound.py`): it answers for the set of
secrets and ignores the single `true_h` the episode hands it.

* **Catch-all prior.** p_U = 0.10 in every game, for P0a and for the soft shadow. It is close to the novel share
  among single-cause incidents (17 / 158 = 0.108). P0a's decisions depend on it only through the EIG ranking.
* **A right action ends P0a's game**, as a correct guess does today. In a KK game, verification then shows the
  incident has not cleared, and the loop calls the expert: S has no way to continue. So every KK game under S
  ends with one expert call, after a partial answer, an exhaustion or the budget end.
* **"Names both" is 0 for S by construction.** It is measured so that FLYW-8's arms, which can continue, are
  scored on the same measures.
* **The soft shadow** (`SoftPosterior`, same likelihood, same p_U) observes every answer P0a observes. It never
  acts.

### Games

* **Every game of every kind**, at each flip and noise seed: 141 singles, 17 novels, all 9,870 KK pairs, and all
  2,397 KN pairs. Each pair is played once per seed. Pairs are not sampled.
* **Flips** 0, 0.05 and 0.10. Flip 0 is deterministic and was run once as the calibration. The scored tests are
  at 0.05 and 0.10.
* **Keyed paired noise**, as in FLYW-1, under a fresh namespace. Whether the answer to attribute question q is
  flipped in game g under seed k is a hash of (`flyw2`, k, g, q). The game key is the sorted secret names, so a
  pair is one game whichever member is listed first. No FLYW-1 noise stream is reused.
* **Scope: this pool.** As in FLYW-1, the 158 objects and all their pairs are the whole population, not a
  sample, and the only randomness is the answer noise. Conclusions hold for this pool.

### Pair-share levels

w ∈ {0, 0.1, 0.25, 0.5}. S does not know w, and its play does not depend on it. So each level is a reweighting of
the per-kind results: an incident is a KK pair with probability w; otherwise a novel with probability ν = 17/158,
or else a single. w enters in two places:

* **incident-level tables** (reported): cost per incident, exhaustion escalations per incident, and how much of
  the "new cause" alarm is really compounds;
* **the gap-10 classifier's prior** (TW2 uses w = 0.25; the others are reported). FLYW-8's pairs-as-hypotheses
  arm will also use w as its pair prior.

### Measures

Per game: kind, flip, seed, secrets, how it ended (`named` / `exhausted` / `budget`), the names confirmed, probes,
wrong actions, reinstates, flips applied, the turns at which the live set emptied, the live-set size after each
answer, the soft shadow's catch-all mass after each answer and the turn it first reached 0.9, the soft top pick
at the end, and every step (kind, key, answer).

Per kind and flip:

* named one / both / neither;
* exhaustion at any turn; exhaustion escalation; budget-end escalation;
* catch-all fires;
* partial and silent partial (KK);
* wrong actions, probes and expert calls;
* **total cost** (below).

**Total cost** per game = probes + c_wrong × wrong actions + c_expert × expert calls. A game that names every
secret calls no expert; any other game calls one. **Primary costs: c_expert = 10, c_wrong = 50**, FLYW-1's primary
ratio 0.2, so the numbers compare with FLYW-1 and its cost follow-up. Reported: (10, 20), (10, 200), (30, 150),
(100, 500).

**The compound penalty (primary).** For each KK pair {a, b} and seed: the pair game's cost minus the mean of a's
and b's single-game costs at the same flip and seed. Averaged over all 9,870 pairs. It is the extra cost a
two-cause incident brings over a one-cause incident with one of the same causes. Under S with verification, it
is one expert call plus whatever extra probes and wrong actions the second cause causes. It is also an upper
bound on what any fix can save per compound incident, if the fix makes a compound incident no more expensive than
a single one.

**Why not "the pair against two separate incidents".** That comparison is −125.9 at flip 0 (§Basis): a pair game
is far *cheaper* than two single games, because S hands the second cause to an expert who costs 10, while an
autonomous single game costs 146 (P0a guesses 2.7 wrong names per game, at 50 each). It measures the cost scale,
not the compounding, so it is reported only.

**Unverified sensitivity.** If the incident is not checked after a fix, a partial answer is a silent miss. It is
then charged c_wrong in place of the expert call. Reported.

**Two other readings, reported.** They test whether the primary result depends on FLYW-1's identity-level action
convention:

* **cat17 decisions.** The category is the remediation, scored as FLYW-1 scores A: cut at the first guess in
  either secret's category.
* **Guesses as diagnostics.** A by-name guess is an exact check (one probe), not a remediation, so there are no
  wrong actions.

## Basis: the disclosed noise-free calibration

`flyw2_calibration.py`, run 2026-11-19 as jobq 72 (1 core, 243 s, `PYTHONHASHSEED=0`, BLAS pinned to 1 thread),
output `results/flyw2_calibration_20261119/` (`summary.json` SHA-256 `27cf9a38…babae8`, and `games.json.gz`). It
played S on every game at flip 0, where games are deterministic:

* 141 singles, 17 novels, 9,870 KK pairs, 2,397 KN pairs;
* no noisy cell (flip 0.05 or 0.10) was played;
* an earlier run of the same script (jobq 71) gave identical numbers before two reported readings were added to
  it.

With exact answers an in-pool single is never struck, so "error" escalations do not exist at flip 0. The flip-0
cells of every test below are therefore known. They are bases, not tests.

**Per kind, flip 0** (cost at c_expert 10, c_wrong 50, verified):

| kind | n | named 0 / 1 / 2 | exhaustion esc. | any exhaustion | catch-all fires | silent partial | probes | wrong | cost |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single | 141 | 0 / 1.000 / – | 0 | 0 | 0 | – | 9.77 | 2.72 | 145.9 |
| novel | 17 | 1.000 / 0 / – | 1.000 | 1.000 | 0.471 | – | 14.82 | 8.76 | 463.1 |
| **KK** | 9,870 | 0.016 / **0.985** / 0 | **0.016** | 0.024 | **0** | **0.976** | 9.69 | 2.93 | 166.0 |
| KN | 2,397 | 0.492 / 0.508 / 0 | 0.492 | 0.510 | 0.241 | 0.490 | 11.96 | 5.73 | 308.5 |

**The compound penalty at flip 0** (KK cost − mean of its members' single costs):

| reading | penalty | made of |
|---|---:|---|
| **primary** (10, 50), verified | **+20.0** | expert call +10; wrong actions +0.20 (× 50 = +10.1); probes −0.07 |
| unverified (silent miss = 50) | +59.4 | |
| cat17 decisions | +9.2 | the expert call on the 94% of pairs that span two categories |
| guesses as diagnostics | +9.9 | the expert call |
| (10, 20) / (10, 200) / (30, 150) / (100, 500) | +14.0 / +50.4 / +60.3 / +201.1 | |

**How the answers combine.** d_min is the number of questions on which the better-fitting member is contradicted
by the OR answers, if every question were asked. d_min = 0 means one member's yes-set contains the other's
("masked").

| stratum | pairs | exhaustion esc. | partial |
|---|---:|---:|---:|
| d_min 0 | 178 | 0 | 1.000 |
| d_min 1 | 310 | 0 | 1.000 |
| d_min 2 | 578 | 0.009 | 0.991 |
| d_min 3–4 | 2,739 | 0.010 | 0.991 |
| d_min ≥ 5 | 6,065 | 0.020 | 0.980 |
| same category | 557 | 0 | 1.000 |
| cross category | 9,313 | 0.016 | 0.984 |

Spearman of d_min against any exhaustion is 0.11. The overlap barely matters at flip 0. A likely reason, not
tested: P0a starts guessing names early (at probe 7.1 on average in FLYW-1's calibration), so it rarely reaches
the questions on which the pair contradicts its better-fitting member. Once one category question is answered
yes, the other category questions have no information value among the live objects and are not asked.

**Gap 10 at flip 0.** 153 KK games and 17 novel games ended in an exhaustion escalation. On their answers the
log Bayes factor of "compound" against "novel" separates them with AUC 0.938. But the three-way posterior's top
pick, at w = 0.1 or 0.25, calls all 153 compounds and 16 of the 17 novels "single + error". Only at w = 0.5 does
it call the compounds compound, and then it calls 9 novels compound too. With ε = 0.05, one contradicted answer
on a known object is a cheap explanation, and P0a's short question trail rarely holds more than one.

**The alarm mix at flip 0.** At w = 0.25, compounds are 4.6% of exhaustion escalations (1.6% at w = 0.1, 12.6% at
w = 0.5); the rest are novels. Under noise, flip-struck singles will add "error" escalations to this mix.

**The cost scale, disclosed.** At the primary costs an autonomous single game costs 145.9 and an expert call 10.
So calling the expert at turn 0 would be cheaper than P0a in every kind of game. This comes from FLYW-1's
convention, that each by-name guess is a remediation, combined with P0a's habit of guessing names. It is why the
penalty, not the raw cost, is the primary measure (decided: §Decisions, 1).

## Predictions

All scored tests are at flips 0.05 and 0.10, on KK pairs against singles at the same flip, pooled over the run's
seeds. A test scored "at each flip" must hold at both.

* **CX0: mechanical gate.** Every part must hold before any noisy run.
  * (a) A single-secret game through `KeyedSetOracle` at flip 0 reproduces `run_openworld_episode` with
    `matrix_oracle`, step for step, for all 158 objects as the secret.
  * (b) The runner's flip-0 output reproduces the calibration's `summary.json` exactly.
  * (c) No game at any flip confirms a name that is not one of its secrets. Guesses are exact, so this holds by
    construction; the runner asserts it.
  * (d) The keyed noise is a pure function of (seed, game, question), and the same pair gets the same key in
    either order.
* **CX1: compounds do not routinely empty the survivor set** (decides the production recommendation).
  * (a) At each noisy flip, the **excess false-exhaustion rate**, KK's exhaustion-escalation rate minus the
    singles', is **< 0.10**. Basis 0.016 − 0 = 0.016 at flip 0.
  * (b) At each noisy flip, KK's catch-all firing rate minus the singles' is **< 0.10**. Basis 0 − 0.
  * "Routinely" is set at one compound in ten: above it, at w = 0.25, compounds would make up a sizeable share
    of the new-cause alarms the flywheel admits from. The share is reported.
* **CX2: compounding costs about one expert call (primary, total cost).** Pooled over the two noisy flips, the
  compound penalty at the primary costs, with verification, is **≤ c_expert + 0.5 × c_wrong = 35**. That is, a
  two-cause incident costs today's loop no more than one expert call plus half a wrong action more than a
  one-cause incident. Basis +20.0. The bar is stated in cost units so that it does not depend on the basis.
* **CX3: the failure is the confident partial answer.** At each noisy flip, KK games:
  * (a) name one secret in **≥ 0.80** of pairs; basis 0.985;
  * (b) end as a silent partial in **≥ 0.70**; basis 0.976.
* **TW1: compound and novel exhaustions are separable.** Over the exhaustion escalations of KK and novel games,
  pooled over the noisy flips, the AUC of log BF(compound : novel) is **≥ 0.85**. Basis 0.938 (153 against 17
  games).
* **TW2: the three-way test works on P0a's own answers.** Over all exhaustion escalations of singles ("error"),
  novels and KK pairs, pooled over the noisy flips, the three-way posterior's top pick at w = 0.25 (prior
  single 0.669, compound 0.25, novel 0.081) has a macro-averaged recall of **≥ 0.60** (chance is 0.33). **The
  basis points to a fail**: at flip 0 it recalled 0 of 153 compounds and 1 of 17 novels. It is scored because
  gap 10 asks exactly this, and a fail is informative: on P0a's own question trail, the three explanations of an
  empty survivor set cannot be told apart without extra probes.

**Reported, not scored:**

* the d_min strata at every flip, and the Spearman of d_min against exhaustion and against the penalty;
* every cost pair, the unverified sensitivity, cat17 decisions and guesses-as-diagnostics, for the penalty and
  per kind;
* KN pairs: how often the known member is named and how often they escalate;
* the pair against two separate incidents;
* incident-level tables at w = 0, 0.1, 0.25 and 0.5: cost, exhaustion escalations, silent partials, and the
  compound, error and novel shares of exhaustion escalations;
* TW2's confusion matrix at every w, the pairwise AUCs (error against novel, error against compound), and TW2
  with the classifier's ε set to the true flip;
* the soft shadow's top pick against the secrets, by kind;
* turns to the first exhaustion, and reinstates per kind.

## Decision rule and kill criterion

The analyzer to be built (`analyze_flyw2.py`) implements these rules; where this wording is ambiguous, its code is
the reference. A result within 2 SE of its bar is **inconclusive**, and **an inconclusive test counts as not
passed** (FLYW-1 Addendum 1).

1. **CX0 fails:** stop. Fix the build before any noisy run.
2. **CX1 not passed** (either part, either flip): compounds empty the survivor set routinely under noise. **The
   hard-gate-first recommendation needs revising:** an empty survivor set no longer means "new cause". FLYW-8
   must first build a compound-aware escalation (a discriminating probe before the expert call, or pairs as
   hypotheses at the gate), and the flywheel's admission step must not admit from exhaustion alone.
3. **CX1 passes:** the recommendation stands on this count. Then:
   * **CX2 and CX3 pass: kill criterion.** Single-cause elimination fails cheaply and safely, provided the
     incident is checked after each fix. **FLYW-8's modelling fixes (pairs as hypotheses, noisy-OR, linked
     causes) are not built.** The fix carried forward is the check itself: the production recommendation gains
     "verify after every remediation; if the incident has not cleared, escalate." The unverified penalty says
     what skipping the check costs. FLYW-9 is then needed only to learn the likelihood, not to rescue elimination.
   * **CX2 not passed:** compounding costs more than an expert call plus half a wrong action. FLYW-8 builds its
     cheapest arm first, explain-away (verify and continue), then pairs as hypotheses, scored against S on this
     penalty.
   * **CX3 not passed, with CX2 passing:** the failure is not mainly the silent partial. The report says what
     replaced it (exhaustion below CX1's bar, or the budget end), and FLYW-8 targets that.
4. **TW1 and TW2** decide only gap 10's text in the README and whether FLYW-8 includes a discriminating probe.
   They do not change the decisions above. If TW2 fails, gap 10 is answered "not on P0a's own answers": separating
   novel, error and compound needs probes chosen for it.

## Seeds (blind rule, as in FLYW-1)

The quantity each test measures is a mean over this fixed population (9,870 pairs, 141 singles, 17 novels), so the
only randomness is the answer noise. Its standard error is estimated within units: within pairs for KK quantities
and within secrets for singles. For each unit, take the variance across its seeds and pool over the units
(one-way ANOVA within-unit mean square). The SE of a mean over N units is √(Σ σᵢ²) ÷ (N √S). The penalty's unit is
the pair, with its members' single games from the same seed.

* **The rule.** For CX1(a), CX1(b), CX2 and CX3, S = max(5, ⌈5 × (SE₅ ÷ (margin ÷ 3))²⌉). Here margin is the
  distance from the flip-0 basis to the bar: 0.084, 0.10, 15.0, and 0.185 / 0.276. SE₅ comes from a 5-seed smoke
  at flips 0.05 and 0.10.
* **The run** uses the largest S over the tests, capped at **4 hours on 6 workers**. If the cap binds, the
  inconclusive band applies. TW1 and TW2 get no seed rule of their own; their SE is reported by a bootstrap over
  seeds.
* **Blind.** `flyw2_seed_rule.py`, to be built, reads the smoke and prints only variances, standard errors and
  seed counts, never a rate, a cost or a difference. The smoke's games are not otherwise read.
* **Cost estimate.** At flip 0 a full set of games takes 243 s on one core. The smoke measured 429 core-seconds
  per noisy (flip, seed), so the 4-hour cap allows 100 seeds per flip on 6 workers.

**The seed count (smoke jobq 78, 2026-11-19, `results/flyw2_smoke_20261119/seed_rule.json`).**

| test | flip | SE₅ | margin | S |
|---|---|---:|---:|---:|
| CX1(a) | 0.05 / 0.10 | 0.0105 / 0.0145 | 0.084 | 5 / 5 |
| CX1(b) | 0.05 / 0.10 | 0 / 0 | 0.10 | 5 / 5 |
| CX2 | pooled | 0.379 | 15.0 | 5 |
| CX3(a) | 0.05 / 0.10 | 0.0012 / 0.0016 | 0.185 | 5 / 5 |
| CX3(b) | 0.05 / 0.10 | 0.0013 / 0.0017 | 0.276 | 5 / 5 |

* **Every test needs only the rule's floor, so the run uses S = 5 seeds** per noisy flip. That is about 12 minutes
  on 6 workers. The cap does not bind.
* **Disclosed:** CX1(b)'s within-unit variance is exactly 0 at both flips. For every pair and every single, the
  catch-all-firing indicator was the same on all five seeds. The tool computes no rate, so whether it was always
  0 or always 1 for a given unit was not read. The smoke's seeds 1–5 are the run's seeds 1–5, under the same keyed
  noise.
* **TW1 and TW2 have no seed rule** (as drafted). At S = 5 they rest on about 85 novel games per flip, so their
  bootstrap SE over 5 seeds will be coarse. Running more seeds would cost about 2.4 minutes of wall time per extra
  seed. That would be a change to the rule, and so the user's call before the tag.

## FLYW-8: the fixes (listed, not in this prereg)

FLYW-7 measures the failure only. The fixes are FLYW-8, pre-registered separately if the decision rule calls for
them. Each is scored on this prereg's measures and against S on the compound penalty.

| arm | what it does | build cost | note |
|---|---|---|---|
| **EA: explain-away (verify and continue)** | After a right action that does not clear the incident, condition on the named cause: under hypothesis h, an answer's likelihood is that of OR(named, h). Rebuild P0a's live set over the remaining pool from all answers so far, and continue on the remaining budget | Small: a likelihood wrapper and a replay. Same speed as S | The first fix to build. Measures "names both" |
| **PH: pairs as hypotheses** | P0a over 141 singles + 9,870 pairs, prior w spread over the pairs. A pair is struck when its OR prediction is contradicted. An action names one member | Medium: 71× the hypotheses. EIG needs the vectorised planner (`stress_fastplan`) | Space grows quadratically; also the gate-side fix for CX1 |
| **NOR: noisy-OR likelihood** | The soft belief with a noisy-OR likelihood over pairs; P0a's kill rule applied to it | Medium | Matters if FLYW-9 finds symptoms compose as a union; here the oracle is an exact OR |
| **LC: linked causes** (optional) | Causes as nodes with "can co-occur" edges (synthetic in 20q, e.g. by category); only linked pairs are hypotheses; eliminated by propagation | Large | The constraint-graph form of the meet ("Bitmask or graph"); nothing in 20q measures it yet |
| **DP: discriminating probe** (added from the calibration) | At an exhaustion, before the expert call, ask the questions that best separate the best single, the best pair and the catch-all | Small | The active version of the gap-10 test; motivated by TW2's expected fail |

**Why none is folded in here.** EA and DP are cheap enough, but each adds an arm with its own bars to a draft whose
calibration already changes the expected picture: the failure looks like a cheap partial answer, not
exhaustion. The kill criterion may close FLYW-8's modelling arms outright, so they wait for the result. If the user
prefers, EA can be added to this prereg as a reported arm at little cost. The user chose not to (§Decisions, 4).

## What stays fixed, and what this is not

* **The belief is P0a, unchanged.** The soft posterior is a shadow only (the shadow studies found its top pick
  unreliable under noise).
* **The OR is exact and the oracle is the self-consistent matrix.** Real symptoms may not combine as a union. That
  is FLYW-9's question in fault-graph. No live Jev and no API calls.
* **Causes are passive.** A right action does not remove its cause's symptoms from later answers; S stops at it
  anyway. EA, in FLYW-8, is where that choice starts to matter.
* **Identity-level actions** by FLYW-1's convention; class-level (cat17) decisions are reported only.
* **Two causes at most.** Three or more is not tested.
* **The bitmask form of the meet**, as every 20q result so far.
* **Compute** goes through `jobq`, with `-N` equal to the workers and every BLAS thread variable set to 1 per
  worker, under `taskpolicy -c utility`, as in FLYW-1.

## Decisions taken 2026-11-19 (user, in session; Jira FLYW-7)

The user reviewed the draft's open choices and took the recommended option on each of the first four. The fifth
was not raised and stays as drafted. Nothing in the design changed.

1. **Cost scale: FLYW-1's convention is primary.** Each by-name guess is a remediation, with c_expert 10 and
   c_wrong 50, for comparability with FLYW-1 and its cost follow-up. The cat17 and diagnostic readings are
   reported. The turn-0 escalation problem (§Basis) is why the scored quantity is the penalty, which is close to
   one expert call in all three readings at flip 0 (+20.0, +9.2, +9.9).
2. **Verification: verified is primary.** A partial answer costs one expert call. The unverified reading (+59.4 at
   flip 0) is reported as a sensitivity.
3. **The bars: kept as drafted.** CX1 < 0.10, CX2 ≤ 35, CX3 ≥ 0.80 and ≥ 0.70, TW1 ≥ 0.85, TW2 ≥ 0.60.
4. **Explain-away: left to FLYW-8.** FLYW-7 stays a measurement of the failure only.
5. **Novel hold-out and p_U = 0.10: as drafted** (seed `flyw2-novels`; the 17 novels are listed in
   `summary.json`).

## Before registering

1. ~~The user decides the open choices~~ (done, §Decisions); this draft merges.
2. ~~Build `flyw2_run.py`, `analyze_flyw2.py` and `flyw2_seed_rule.py`, with a build note~~ (done:
   `FLYW2_BUILD.md`).
3. ~~Run the CX0 gate~~ (PASS, jobq 77).
4. ~~Run the 5-seed smoke and the blind seed tool~~ (S = 5, jobq 78; §Seeds).
5. ~~The draft merges. Tag `prereg-flyw2-compound-<date>`~~ (tag `prereg-flyw2-compound-20261119`), then queue
   `flyw2_run --run --seeds 5` through `jobq -N 6`.
