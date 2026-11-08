# Open-world Twenty Questions — auditability study, results

**ENV-703.** The retraction study (ENV-702) measured the five backends on the axes a
graded posterior is good at and found `soft` led or tied on them, closing with the
caveat that it never scored the axis the discrete backends exist for: a replayable,
append-only, verifiable op-log that can answer *why* each hypothesis was eliminated or
revived. This study scores that axis. An external auditor rides the open-world loop
through a non-invasive `observer` hook and reconstructs a canonical op-log purely from
each backend's public survivor-set transitions — no backend is modified, and with the
hook off the ENV-702 runs are byte-for-byte unchanged (`OPENWORLD_AUDIT_PREREG.md`).

CMBS-semantics-aligned, CMBS-code-independent (the ENV-702 stance): the backends already
encode the mask-meet/reinstate semantics, so we score the op-log they induce. Wiring in
the real CMBS v0.2.0 package (its own hash-chained log) is the production-validation
follow-on, gated on a v0.2.0 install; it is not in this study.

## Headline: the axis ENV-702 omitted is the one that separates `soft` from the rest

Mean over 2 seed replicates at 10% attribute-flip; run `openworld_audit_20261108_233741`.

| backend | discrete | ops/game | log cells/game | replay-exact | verifiable | answer-why | reinstate-local |
|---|:--:|---:|---:|:--:|:--:|:--:|:--:|
| soft | no | 0.0 | 1103.6 | — | — | — | — |
| hard | yes | 107.4 | 107.4 | 1.000 | 1.000 | 1.000 | — |
| retract @4 | yes | 93.9 | 93.9 | 1.000 | 1.000 | 1.000 | **0.000** |
| tombstone @3 | yes | 28.0 | 28.0 | 1.000 | 1.000 | 1.000 | **1.000** |
| modeswitch @3 | yes | 108.7 | 108.7 | 1.000 | 1.000 | 1.000 | **1.000** |

Read against the ENV-702 scorecard, where `soft` had the best in-pool failure (0.039)
and AUC (0.996): here `soft` scores **nothing**. It keeps no survivor set, so it induces
no op-log — there is no discrete eliminate/revive to replay, verify, or explain. Its only
trace is the dense posterior: a full weight vector every turn (~1104 cells/game), which
is both larger than any discrete op-log and *not* replayable as a sequence of decisions.
Every discrete backend, by contrast, scores 1.000 on all three audit properties.

## 1. Replay (P1) — a sparse op-log that reconstructs the committed state

Each discrete backend's op-log replays its final survivor set **exactly** (`replay-exact
= 1.000`) by applying ELIMINATE/REINSTATE records to the initial known set, with no
likelihood model and no matrix — the definition of a replayable audit trail. The log is
sparse: tens to ~110 op records per game (`tombstone` is lowest at 28 because a known
leaves the live set once, on its first contradiction, and later contradictions while
dead add killers without a new survivor-set op). `soft` has `replay-exact = —`: with no
discrete ops, the only thing to "replay" is the per-turn float vector, which reconstructs
weights, not decisions.

The ops/game figure is a replay-*cost* proxy, not a quality ranking among the discrete
backends — it tracks how each backend's strike/reinstate dynamics move the survivor set,
which ENV-702 already characterised. The auditability claim is the `replay-exact = 1.000`
column, which holds for all of them and for none of `soft`.

## 2. Verifiability (P2) — every strike re-checks, and the log is tamper-evident

`verifiable = 1.000` for every discrete backend: an independent recompute of each
ELIMINATE's precondition — does the recorded `(question, answer)` genuinely contradict
the hypothesis under a freshly built likelihood from the committed matrix — passes for
every op. A SHA-256 hash chain links the records, so a one-record edit (shown in
`--self-test`: flipping a logged hypothesis name) breaks the chain and is caught. `soft`
has no op to verify. This is the property a replayable belief log is *for* — an auditor
can confirm, op by op, that every elimination was warranted — and it is exactly what the
ε-floored posterior cannot offer.

## 3. Answer-why, and the structured/scalar split (P3)

`answer-why = 1.000` for every discrete backend's eliminations: each ELIMINATE record
names the concrete `(qid, answer)` that struck the hypothesis, so "why was X eliminated?"
is answered from the log alone. `soft` scores `—`: a known is never eliminated, its mass
merely decays, so the question has no event to point at.

The **reinstate-local** column is the ENV-702 §2 distinction, now on the audit axis.
`tombstone` and `modeswitch` score 1.000 — a revive's reason is local and checkable from
the record (the hypothesis had ≤ bound distinct killers, listed in the op). `retract`
scores **0.000** — its revive is justified by a *global* scalar race (cumulative
log-likelihood within a margin), so "why was X revived?" is answerable only by replaying
the whole race, not from a local record. `retract` is auditable for its eliminations but
its reinstatements are not locally explainable; the structured backends are both.

## 4. The reframe (P4)

ENV-702 read naively says "just use `soft`." Scored on auditability, `soft` is at the
bottom of every column — it buys the measured axes by giving up the replayable,
verifiable, explainable belief log the discrete backends exist for. So the honest reading
of the two studies together is:

* If a replayable belief log is **not** a requirement, `soft` wins ENV-702's axes (and
  pays only ~1 extra question/game).
* If it **is** a requirement, `soft` is disqualified here, and among the auditable
  backends `modeswitch` is the one to reach for: it carried the ENV-702 resolution
  (recovers in-pool loss, keeps the detector, zero wrong guesses) **and** scores 1.000 on
  all three audit properties with a locally-explainable reinstate — the only backend that
  is both the ENV-702 resolution and fully auditable.

`retract` and `tombstone` remain cautionary: ENV-702 showed they blind the detector and
manufacture wrong guesses, and ENV-703 adds that `retract`'s reinstatements are not even
locally explainable. Auditability does not rescue a backend that is wrong; it is the axis
on which the *right* backend (`modeswitch`) and the posterior (`soft`) finally separate.

## Caveats

* This scores the op-log each backend *induces*, reconstructed by an external observer
  from public survivor transitions. It is faithful to the CMBS semantics but is not the
  CMBS v0.2.0 package's own hash-chained log; validating against that (its two hashes,
  per-hypothesis support sets) is the gated follow-on, not done here.
* `log cells/game` counts op records for the discrete backends and dense posterior cells
  for `soft` — a structural density comparison (sparse events vs a full vector per turn),
  not a byte-for-byte storage benchmark; the load-bearing result is the replay/verify/why
  columns, not the cell counts.
* Same stream, matrix, PY fit, and noise grid as §3.11 / ENV-702; the auditor adds only
  observation. `--self-test` proves replay-exactness, precondition verification, the
  planted-tamper catch, and the `soft`-has-no-op-log asymmetry before any grid runs.

## Reproduce

Self-test then grid:

```
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.audit_openworld --self-test
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.audit_openworld --run \
    --games 2000 --backends soft,hard,retract,tombstone,modeswitch \
    --levels 0.0,0.05,0.10 --noise-seeds 2 --margins 4 --tolerances 3
PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_audit \
    --dir llm_rl_scripts/twenty_questions/jev/results/openworld_audit_20261108_233741
```

The figure `fig_auditability.png` shows the three audit properties per backend (soft at
0, the discrete backends at 1.0) and the sparse-op-log vs dense-trace storage asymmetry.
