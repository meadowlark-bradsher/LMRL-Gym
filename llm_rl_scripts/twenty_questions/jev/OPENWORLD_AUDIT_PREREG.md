# Open-world Twenty Questions — auditability study, pre-registration

**ENV-703.** The retraction study (ENV-702, `OPENWORLD_RETRACTION_RESULTS.md`) scored
five belief backends on five axes — in-pool failure, detector AUC, wrong-guesses,
escalations, questions/game — and `soft` led or tied on the ones a graded posterior is
good at. Its own closing section ("What the scorecard leaves out") names the axis it
never measured: the discrete backends (`hard`/`retract`/`tombstone`/`modeswitch`) exist
for a **replayable, append-only audit trail** — the ability to reconstruct and verify
*why* each hypothesis was eliminated or revived, op by op (the CMBS premise). This study
makes that axis measurable and scores every backend on it, so the ENV-702 scorecard can
be read without the implicit "just use `soft`."

This is **CMBS-semantics-aligned, CMBS-code-independent** (the same stance as ENV-702):
the backends already encode the mask-meet/reinstate semantics, and we score the op-log
they induce. Wiring the real CMBS v0.2.0 package in (its hash-chained op-log) is the
production-validation follow-on, gated on a v0.2.0 install; it is *not* in this study.

## The canonical op-log

An external **auditor** observes each game non-invasively: after every answer it reads
the backend's public survivor set (`survivors` / `live`) and reconstructs an ordered,
append-only op-log by diffing. No backend is modified; the loop gains one optional
`observer` hook (default off, so the ENV-702 runs are byte-for-byte unchanged).

Each record is `(seq, turn, op, hypothesis, cause, prev_hash, hash)`:

* `op ∈ {ELIMINATE, REINSTATE}` — a known left or re-entered the survivor set this turn.
* `cause` — for ELIMINATE, the `(qid, answer)` that contradicted the hypothesis this
  turn (the strike happens on the turn's question, so the cause is local and exact);
  for REINSTATE, the backend's revive reason.
* `hash = H(prev_hash ‖ seq ‖ turn ‖ op ‖ hypothesis ‖ cause)` — a chain over the log,
  so any altered or dropped record is detectable (append-only tamper-evidence).

**`soft` has no survivor set** (no `survivors` attribute — it never eliminates). Its only
audit trail is the dense posterior: a full weight vector every turn. So `soft` induces
**zero** op records; its "log" is `turns × |support|` floats, and there is no discrete
decision to point a *why* query at. That asymmetry is the finding, not an oversight.

## Metrics (per game, averaged over the stream)

1. **Replay cost.**
   * `ops_per_game` — length of the sparse op-log (0 for `soft`).
   * `log_cells` — storage to replay the final committed state: `ops_per_game` for the
     discrete backends; `Σ_turn |support|` dense floats for `soft`.
   * `replay_exact` — reconstruct the final survivor set by applying the op-log to the
     initial known set **without the likelihood model**, and compare to the backend's
     actual final survivors. Discrete: must be exact. `soft`: impossible (no ops).

2. **Op-log verifiability.**
   * `verifiable_frac` — fraction of ELIMINATE ops whose precondition re-checks: an
     independent recompute of `p_obs(hypothesis, cause_q, cause_ans)` against the
     committed matrix is `< kill`. `soft`: undefined over an empty op-log → 0.
   * `reinstate_local` — is a REINSTATE's reason local and checkable from the record
     (structured: the hypothesis had ≤ bound distinct killers) or only recoverable by
     replaying a global scalar race (`retract`'s log-likelihood margin)? A structural
     property: `tombstone`/`modeswitch` True, `retract` False.
   * `tamper_evident` — a one-record edit breaks the hash chain (demonstrated in
     `--self-test`). True for any non-empty discrete log; `soft` has no chain.

3. **Answer-the-why.**
   * `why_frac` — of knowns eliminated at game end, the fraction whose log yields a
     concrete contradicting `(qid, answer)`. `hard`/`retract`/`tombstone`/`modeswitch`:
     the ELIMINATE record carries it. `soft`: 0 — a known is never eliminated, its mass
     merely decays, so "why was X eliminated?" has no event to answer.

## Pre-registered predictions

* **P1 — sparse, exact replay.** Every discrete backend yields a sparse op-log
  (`ops_per_game ≪ log_cells` of the `soft` dense alternative) that replays the final
  survivor set exactly (`replay_exact = 1.0`); `soft` yields none (`replay_exact = 0`,
  dense storage only).
* **P2 — verifiable and tamper-evident.** Discrete `verifiable_frac = 1.000` (every
  strike is a real contradiction); the hash chain catches a planted edit; `soft` scores
  0 (nothing to verify).
* **P3 — why, and the structured/scalar split.** `why_frac = 1.000` for the discrete
  backends' eliminations and 0 for `soft`; `reinstate_local` is True for
  `tombstone`/`modeswitch` and False for `retract` — the same structured-vs-scalar
  distinction ENV-702 §2 drew, now on the audit axis.
* **P4 — the reframe.** On a composite auditability score `soft ≈ 0` despite leading
  every ENV-702 axis, so "just use `soft`" does not survive scoring auditability; and
  among the auditable backends `modeswitch` keeps the ENV-702 resolution **and** is
  fully auditable (P1–P3), so it is the backend to reach for when a replayable belief
  log is a requirement.
* **Falsification.** If a discrete backend's `replay_exact < 1.0` or
  `verifiable_frac < 1.0` the op-log is not a faithful audit trail and the "auditable"
  claim fails for it; if `soft` admitted a sparse replayable log the asymmetry would
  collapse.

## Protocol

Same stream, matrix, PY fit, and noise grid as §3.11 / ENV-702 (Zipf s=1.1, seed 7,
2000 games; attribute-flip 0.0/0.05/0.10). Backends and knobs as in ENV-702
(`retract@{1,2,4}`, `tombstone@{1,3}`, `modeswitch@{1,3}`). Per-game op-logs are
reconstructed by the auditor; metrics are averaged per cell. `--self-test` proves
replay-exactness, precondition verification, the planted-tamper catch, and the
`soft`-has-no-op-log asymmetry before any grid runs.

Reproduce: `audit_openworld --self-test`, then `audit_openworld --run --out <dir>` and
`analyze_audit --dir <dir>`.
