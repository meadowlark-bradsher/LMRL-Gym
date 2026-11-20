# Moving the 20Q work into its own elimination repo — plan

**Status (2026-11-20): DECIDED.** D1–D6 in §2 take the recommended option. User, 2026-11-20: "Go with
your recommended options for D1–D6". The name `meadowlark-bradsher/elimination` was checked and is free.
Nothing has been moved or created yet. The new repo's own license is **deferred** (user, 2026-11-20:
"lets leave the license out for now"). The repo starts private with no LICENSE file, and one is added
before it is shared or made public.

**Roadmap.** This plan is enabler **E1** on the `sre-flywheel` ROADMAP. The sre-flywheel session is the
integrator, and this plan's owner owns E1. §8 lists what the roadmap adds.

## The question in plain words

The Twenty Questions work in this directory has stopped being about LMRL-Gym. It is a general method
for **elimination**: keep a set of candidate explanations, ask the question that rules out the most,
drop candidates the answers contradict, notice when the true answer is probably none of them, and call
an expert at the right time. The SRE flywheel (`sre-flywheel` README, component 1) already treats it
that way.

**Where should it live, and how do we move it without carrying LMRL-Gym's license into the new repo?**

## Answer

A **new repo** with a domain-neutral core and 20Q as its first domain. The move is done so that **no
file in the new repo, in any commit, contains LMRL-Gym content**. Then the LMRL-Gym MIT notice does not
apply to it, because MIT binds only "copies or substantial portions" of the licensed work.

This is possible because of an audit done for this plan (§1): every line of code here is yours. The one
piece of upstream content the code needs is the 158-object list, and the new repo can *read* that from a
local LMRL-Gym checkout without *copying* it.

Not legal advice. A list of common nouns may not be protectable at all. The plan takes the strict line
anyway because it costs little, except for the last phase (§5, Phase 3).

---

## 1. What the audit found

| Item | Finding | Consequence |
|---|---|---|
| LMRL-Gym license | MIT, `Copyright (c) 2022 Charlie Snell`, no NOTICE file | Only upstream *content* triggers it |
| Authorship of `twenty_questions/jev/` and `LLM_RL/jev/` | All 79 commits touching these paths are yours (first one 2026-10-25) | Code and docs can move with history |
| Code (62 `.py` files, 17.7k lines) | Upstream symbols used: **one**, `DEFAULT_OBJECT_DICT` (`bed.py:26`). Everything else is your own: `ATTRIBUTE_BANK`, `CATEGORY_PHRASE` and the question text | Replace that one import with a loader (§3) |
| Jev client `LLM_RL/jev/client.py` (229 lines) | Yours. It talks to TypeSafe (`TYPESAFE_API_KEY`), not to anything in LMRL-Gym | Copy it. LMRL-Gym keeps its own copy for Wordle and Car Dealer |
| Files outside `results/` that list the objects | A scan for the 201 LMRL surface forms (158 objects plus spelling variants) flags **only `matrix_jev.json`** (all 158). No other code or doc names 10 or more | `matrix_jev.json` stays in LMRL-Gym and is read the same way as the object list |
| `results/` (373 MB, about 44.7k files) | Game logs name the objects everywhere | Stays in LMRL-Gym as the archive. Analyzers already take result directories as CLI arguments (`--base-dir`, `--maps`, `--part2`), so nothing is hard-coded |
| Your 128-item set (`belief-trajectory/data/items.txt`, vendored into `20q-env`) | **All 128 items are LMRL surface forms.** It is a subset of the LMRL list. `20q-env/data/DATA_PROVENANCE.md` names belief-trajectory as the source, not LMRL-Gym | Switching to it does **not** get away from LMRL-derived data. This affects 20q-env and belief-trajectory too (§6) |
| Registered studies | 11 `prereg-*` tags for this work in the fork: ST-1, ST-2, ST-3, ST-3 stream-K, ST-3 end-game, shadow, reverse shadow, FLYW-1 decision, FLYW-1 addendum 1, FLYW-1 cost, FLYW-2 compound | Tags stay where they are. The new repo cites them by tag and SHA, and never re-tags |
| Links into this path from elsewhere | FINDINGS.md cites `twenty_questions/jev/` 5 times. `sre-flywheel` README links here once | Add pointers. Don't rewrite history |
| Tests | No pytest suite. The safety net is the gate scripts (`stress_regression`, ST-0, `stress_guard_gate`, `flyw1_cost_gate`, `stress_fastguess_gate`) | These gates are the acceptance tests for the move |

## 2. Decisions

**Decided 2026-11-20: every row takes the "Recommended" option.** The alternatives are kept for the
record.

| # | Decision | Decided (was: recommended) | Alternatives not taken |
|---|---|---|---|
| D1 | Repo name | `elimination` | `elim`, `bed-elimination`, or a name of yours |
| D2 | How the 20Q domain gets the 158 objects and `matrix_jev.json` | **Bring your own checkout.** A loader reads them from a local LMRL-Gym at the archive tag, given by `LMRL_GYM_PATH`, and checks a SHA-256 recorded in the new repo. A hash is not a copy | (b) Copy them and add the MIT notice. You asked not to. (c) Switch now to a new item set. That breaks byte-identical regression, so it is Phase 3 instead |
| D3 | History | **Carry it** with `git filter-repo`, keeping only your paths and dropping `results/` and `matrix_jev.json` from every commit. The license gate (G1) checks every blob in the history | Fresh start with one import commit pointing at the archive SHA. Simpler, but `git blame` and the prereg trail no longer reach the code |
| D4 | Hard-elimination path | **Keep the jev copy** during the move. Decide on CMBS in Phase 2 | Switch to CMBS's `MaskMeetTombstoneReducer` now. That mixes a behaviour change into the move |
| D5 | Visibility | **Private** (like 20q-env, sre-flywheel) | Public. Then D2 matters more, since others clone it |
| D6 | New results | **Not as loose files in git.** A run manifest in git, payloads in LFS or outside storage | Keep the LMRL-Gym convention (committed result dirs) |

## 3. Target layout

```
elimination/                     # D1
  (no LICENSE yet)               # deferred; never the LMRL-Gym notice
  README.md                      # what elimination is; links to the archive and the 11 prereg tags
  elimination/
    core/                        # Phase 2: domain-neutral
      hypotheses.py              #   candidate set, open-world catch-all slot
      likelihood.py              #   p(yes | hypothesis, question): matrix, exact mask, or live model
      planner.py                 #   EIG choice (from bed.py, stress_fastplan.py)
      backends.py                #   hard / soft / modeswitch / guarded = P0a (from openworld_backends.py)
      openworld.py               #   Pitman-Yor novelty, expert admission
      escalation.py              #   when to ask the expert; cost model (from flyw1_decision.py)
    clients/jev/                 # from LLM_RL/jev/ (client.py + __init__.py)
    domains/
      lmrl20q/                   # D2 loader: reads objects + matrix from $LMRL_GYM_PATH; ships no names
      twentyq_env/               # later: adapter onto 20q-env (after §6)
    twentyq/                     # Phase 1: the current flat package, moved as is
  studies/                       # Phase 2: stress/, openworld/, flyw/ — runners, analyzers, prereg docs
  tools/license_scan.py          # gate G1
```

Phase 1 moves the flat package to `elimination/twentyq/` and changes nothing else. Phase 2 lifts the core
out of it.

## 4. Acceptance gates

These must pass before Phase 1 counts as done. Every Phase 2 PR must keep G2–G4 passing.

- **G1: license-clean.** `tools/license_scan.py` walks **every blob in every commit** (`git rev-list --all
  --objects`). It fails on any blob containing 10 or more of the 201 LMRL surface forms, on the string
  `Copyright (c) 2022 Charlie Snell`, or on any path that came from LMRL-Gym outside the kept paths. It
  also lists commit authors, which should be you only. The scanner keeps the hash of the name list, not
  the list itself. It reads the list from `$LMRL_GYM_PATH` at scan time.
- **G2: byte-identical regression.** `stress_regression` and ST-0 reproduce the archive's outputs byte
  for byte. Use the same `PYTHONHASHSEED` and BLAS thread pins as the original runs, through `jobq` per
  CLAUDE.md.
- **G3: registered studies re-score.** Run `analyze_st3_endgame` and `analyze_flyw1_cost` against the
  archived result dirs (`--base-dir`, `--run`, `--maps` under `$LMRL_GYM_PATH/.../results/`). Their output
  must match the committed `results/stress_st3_endgame_20261114/analysis.json` and
  `results/flyw1_cost_analysis_20261119.json` exactly.
- **G4: no hidden upstream import.** With LMRL-Gym off `sys.path`, every module imports. Only
  `domains/lmrl20q` touches `$LMRL_GYM_PATH`, and it fails with a clear message when that is unset.

## 5. Phases

### Phase 0: decide and freeze (in LMRL-Gym)

1. Settle D1–D6.
2. **Precondition (roadmap):** FLYW-7's rule, "verify after every remediation; if the incident has not
   cleared, escalate" (FINDINGS §3.16), is merged into `PRODUCTION_RECOMMENDATIONS.md`. Don't tag before
   that.
   Then tag fork `main` as `twentyq-archive-<date>` and record the SHA. That is the archive tag the loader and
   all citations use.
3. Compute the SHA-256 of the canonicalized `DEFAULT_OBJECT_DICT` and of `matrix_jev.json` at that tag.

### Phase 1: license-clean move (a pure move, no behaviour change)

1. Work in a fresh clone, never this checkout. Run `git filter-repo` with:
   - `--path llm_rl_scripts/twenty_questions/jev/ --path LLM_RL/jev/`
   - `--path-glob '*/results/*' --path llm_rl_scripts/twenty_questions/jev/matrix_jev.json --invert-paths`
     (a second pass)
   - `--path-rename llm_rl_scripts/twenty_questions/jev/:elimination/twentyq/
     --path-rename LLM_RL/jev/:elimination/clients/jev/`

   `LLM_RL/jev/` holds only `client.py` and `__init__.py` today, and the `__init__` re-exports from
   `LLM_RL.jev.client`, so step 4's rewrite covers it.
2. Run G1 on the filtered history **before** pushing anywhere. If a blob fails, add it to the
   invert-paths list and rerun.
3. Create the private repo (D5) and push. No LICENSE file yet (deferred, see Status).
4. In one commit:
   - replace `from llm_rl_scripts.twenty_questions.env.data import DEFAULT_OBJECT_DICT` with
     `domains/lmrl20q.load_object_dict()`, hash-checked
   - point `load_matrix`'s default at the loader
   - rewrite `llm_rl_scripts.twenty_questions.jev.` → `elimination.twentyq.` and
     `LLM_RL.jev.client` → `elimination.clients.jev.client` in code and in the `python -m` lines of the docs
   - add `pyproject.toml` with numpy, scipy and matplotlib only
5. Run G2, G3 and G4.
6. Add pointers back:
   - a `MOVED.md` in this directory naming the new repo and the commit (the directory stays frozen;
     LMRL-Gym history is not rewritten)
   - a FINDINGS §8 provenance row
   - the new text for the `sre-flywheel` README component-1 row, **sent to the sre-flywheel
     (integrator) session**, which makes the edit. That README has one writer.

### Phase 2: generalize (one PR per lift, each passing G2–G4)

Lift the core in this order: `bed.py` → `core/hypotheses, likelihood, planner`;
`openworld_backends.py` → `core/backends` (P0a stays named and unguarded `modeswitch` stays for
reproducibility); `openworld.py` → `core/openworld`; `flyw1_decision.py` → `core/escalation`.

Rename "object" and "question" to "hypothesis" and "test" only inside `core/`. Studies keep their
registered wording.

Move studies into `studies/` with their prereg, build and results docs. The prereg docs keep citing
their LMRL-Gym tags.

Then make the CMBS choice that D4 deferred (the jev reducer or CMBS's `MaskMeetTombstoneReducer`).
Make it with the roadmap's O8 (bitmask or graph) and cmbs ADR-001 in view.

### Phase 3: a default domain with no LMRL ancestry (optional; the only expensive phase)

Needed only if you want the new repo to run end to end without an LMRL-Gym checkout, for example once it
is public.

1. Build a new item set by a documented process that never reads the LMRL list. Overlap with common
   nouns like "Banana" is expected and fine: being derived from something is about the process, not the
   overlap.
2. Collect a new Jev matrix: 44 attributes × N items through TypeSafe. Estimate the cost before
   collecting.
3. Re-baseline. Old numbers stay cited from the archive. New numbers are new results, not
   reproductions.

## 6. Out of scope, but found here

`20q-env` and `belief-trajectory` carry an LMRL-derived item list (`items.txt`, 128 of the LMRL surface
forms) without an LMRL-Gym notice, and 20q-env's provenance file doesn't mention LMRL. If the new repo
will depend on 20q-env (`domains/twentyq_env`), that has to be settled first: add the notice there, or
rebuild its item set the Phase 3 way. That is a separate change in those repos.

## 7. What stays untouched

- LMRL-Gym history, tags and `results/`. Nothing is rewritten or deleted.
- Car Dealer, Split Deal, Wordle and the shared `FINDINGS.md`.
- The running SD3-E job, which lives in another worktree and branch.

## 8. Roadmap integration (sre-flywheel E1)

These come from the sre-flywheel integrator session (2026-11-20) and don't change any decision in §2.

- **Timing.** Phases 0–1 are *Now*. Phase 2 is *Next*: it is the first bet of the roadmap's O4, whose
  export from fault-graph targets `core/`. Phase 3 is *Later* and optional.
- **Freeze.** From the archive tag on, no new study starts in `twenty_questions/jev/`. Later studies
  (FLYW-8 arm DP, a FLYW-9 likelihood study) start in the new repo once Phase 1's gates pass. The freeze
  covers this directory only, not `LLM_RL/jev/`, whose client Wordle and Car Dealer still use.
- **Before the tag:** the FLYW-7 remediation rule goes into `PRODUCTION_RECOMMENDATIONS.md` (Phase 0
  step 2).
- **D4:** the CMBS choice is made in Phase 2 with O8 and cmbs ADR-001 in view.
- **Phase 1 step 6:** the sre-flywheel README row goes through the integrator.
- **§6** (item-list provenance in 20q-env and belief-trajectory) is listed under E1 as unassigned.
