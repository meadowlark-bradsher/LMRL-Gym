"""Reverse shadow: a hard survivor set beside a playing `soft`
(``STRESS_REVSHADOW_PREREG.md``; Jira EE-41, ENV-706).

Committed `soft` games are replayed through the loop's ``observer`` hook. A passive
hard view rides along: a known is struck when an answer's likelihood under it is
below 0.5 (``HardMonotone``'s rule) and is never revived. It updates on every answer
the game produces, including by-name guesses answered "no". It never touches the
game: each replayed job must match its committed record in every per-game column
and the scorecard.

Per game it records the hard set's size at the end, the first turn it empties (the
*alarm*), and, at the budget end, whether the hard view had struck `soft`'s leading
known (the *veto*). A tolerance-1 view (struck on a second contradicting answer) is
kept beside it for the reported variant.

    # self-test (synthetic matrix only)
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_revshadow --self-test
    # the grid, as registered
    ... stress_revshadow --plan
    # replay committed jobs with the observer on and require identity (real matrix; prints timing only)
    ... stress_revshadow --regress st2:exact_d1_la1_f0.1_s1_g0_soft
    # synthetic smoke: fresh games with the observer, for plumbing and timing
    ... stress_revshadow --smoke --workers 4 --outdir <dir>
    # the registered run
    ... stress_revshadow --run --prereg-tag <tag> --workers 6 --outdir <dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

from .stress_shadow import SOURCES, _ids, _load, _module
from .stress_st1 import pin_blas, provenance

PREREG = "llm_rl_scripts/twenty_questions/jev/STRESS_REVSHADOW_PREREG.md"
RESULTS = "llm_rl_scripts/twenty_questions/jev/results"
BUDGET = 20
EPS = 0.05
KILL = 0.5
CHECK_GAMES = 50          # games per job whose hard set is re-derived (RS0 b)
RH_COLS = ("rh_n_obs", "rh_alarm", "rh_alarm1", "rh_surv_end", "rh_surv1_end",
           "rh_end_top", "rh_end_struck", "rh_end_contra", "rh_secret_contra",
           "rh_u_leads")


# --------------------------------------------------------------------------- #
# The grid                                                                     #
# --------------------------------------------------------------------------- #

def plan() -> List[dict]:
    """The registered 219 jobs, each a committed `soft` job to replay:

    * ST-2, exact answerer (no ``G_`` name confusion), `soft`: 147;
    * ST-3, ``true`` prior, `soft`: 36;
    * ST-3 end-game, `soft` under ``always``: 36.

    Ordered slowest first (ST-2 and the end-game jobs), so the pool drains evenly."""
    jobs = []
    for jid in _ids("st2"):
        if jid.startswith("exact_") and jid.endswith("_soft"):
            jobs.append({"src": "st2", "job_id": jid})
    for jid in _ids("st3"):
        if "_true_" in jid and jid.endswith("_soft"):
            jobs.append({"src": "st3", "job_id": jid})
    for jid in _ids("end"):
        if jid.endswith("_soft_endalways"):
            jobs.append({"src": "end", "job_id": jid})
    for j in jobs:
        j["id"] = "{}__{}".format(j["src"], j["job_id"])
    order = {"st2": 0, "end": 1, "st3": 2}
    return sorted(jobs, key=lambda j: (order[j["src"]], j["id"]))


# --------------------------------------------------------------------------- #
# The hard observer                                                            #
# --------------------------------------------------------------------------- #

class HardObserver:
    """Wraps a job's own per-game trace; keeps a passive hard survivor set.

    ``cnt[n]`` counts the answers that contradicted known ``n``, tracked while it is
    below 2. The hard set S is ``cnt == 0`` (``HardMonotone``); the tolerance-1 set
    S1 is ``cnt < 2``."""

    def __init__(self, known, trace, idx: Dict[str, int], sink: List[dict], mtx,
                 check: List[int]):
        self.known = known
        self.names = [h.name for h in known]
        self.trace = trace
        self.idx = idx
        self.sink = sink
        self.mtx = mtx
        self.check = check          # [games left to re-derive, failures]
        self.cnt: Dict[str, int] = {n: 0 for n in self.names}
        self.alive1 = list(self.names)
        self.n_s = len(self.names)
        self.n_s1 = len(self.names)
        # An empty H_known is an empty hard set from the start (turn 0).
        self.alarm = 0 if not self.names else -1
        self.alarm1 = 0 if not self.names else -1
        self.steps: List[tuple] = []
        self.like = None
        self.by_name = None

    def __call__(self, turn, q, ans, bk, surv_before) -> None:
        self.trace(turn, q, ans, bk, surv_before)
        if not self.names:
            return
        if self.like is None:
            self.like, self.by_name = bk.like, bk.by_name
        self.steps.append((q, ans))
        like, by_name, cnt = self.like, self.by_name, self.cnt
        keep = []
        for n in self.alive1:           # knowns with fewer than 2 contradictions
            c = cnt[n]
            p_yes = like(q, by_name[n])
            if (p_yes if ans else 1.0 - p_yes) < KILL:
                cnt[n] = c + 1
                if c == 0:
                    self.n_s -= 1
                    keep.append(n)
                else:
                    self.n_s1 -= 1
            else:
                keep.append(n)
        self.alive1 = keep
        if self.n_s == 0 and self.alarm < 0:
            self.alarm = turn
        if self.n_s1 == 0 and self.alarm1 < 0:
            self.alarm1 = turn

    def survivors(self) -> set:
        return {n for n in self.names if self.cnt[n] == 0}

    def _contra(self, name: str) -> int:
        """Answers in this game that contradicted ``name`` (every answer, not capped)."""
        h = self.by_name[name]
        k = 0
        for q, ans in self.steps:
            p_yes = self.like(q, h)
            if (p_yes if ans else 1.0 - p_yes) < KILL:
                k += 1
        return k

    def _rederive(self, ep) -> bool:
        """RS0(b): a standalone ``HardMonotone``, with an independently built
        likelihood and question bank, fed the game's own steps (every answer before
        the decision; a final 'yes' guess is not an update), ends with the same
        survivor set and empties on the same turn."""
        from .bed import GUESS
        from .openworld import (UNKNOWN, UNKNOWN_HYP, build_openworld_bank,
                                make_openworld_likelihood)
        from .openworld_backends import HardMonotone
        all_hyps = list(self.known) + [UNKNOWN_HYP]
        by_name = {h.name: h for h in all_hyps}
        like = make_openworld_likelihood(self.mtx, self.known, eps=EPS)
        bank = build_openworld_bank(self.known)
        by_qid = {q.qid: q for qs in bank.values() for q in qs}
        hm = HardMonotone(self.known, by_name, like, UNKNOWN, 0.0, eps=EPS, kill=KILL)
        steps = list(ep.steps)
        if steps and steps[-1].kind == GUESS and steps[-1].answer_yes:
            steps = steps[:-1]
        alarm = 0 if not self.names else -1
        for s in steps:
            hm.observe(by_qid[s.qid], s.answer_yes)
            if not hm.survivors and alarm < 0:
                alarm = s.turn
        return (len(steps) == len(self.steps) and hm.survivors == self.survivors()
                and alarm == self.alarm)

    def finish(self, ep) -> None:
        self.trace.finish(ep)
        if self.names and self.check[0] > 0:
            self.check[0] -= 1
            if not self._rederive(ep):
                self.check[1] += 1
        top, struck, contra = -1, -1, -1
        if ep.end_budget and ep.end_top is not None:
            top = self.idx[ep.end_top]
            struck = int(self.cnt[ep.end_top] > 0)
            contra = self._contra(ep.end_top)
        sec = self._contra(ep.true_name) if (ep.in_pool and self.steps) else (
            0 if ep.in_pool else -1)
        self.sink.append({
            "rh_n_obs": len(self.steps), "rh_alarm": self.alarm, "rh_alarm1": self.alarm1,
            "rh_surv_end": self.n_s, "rh_surv1_end": self.n_s1,
            "rh_end_top": top, "rh_end_struck": struck, "rh_end_contra": contra,
            "rh_secret_contra": sec,
            "rh_u_leads": int(ep.end_u_leads) if ep.end_budget else -1})


def hard_hook(sink: List[dict], check: List[int], names: Optional[list] = None):
    """The ``shadow`` argument of ``stress_st2.play`` / ``stress_st3.play``. With
    ``names``, the job's object names are appended to it."""
    def shadow(objects, mtx):
        if names is not None:
            names.extend(h.name for h in objects)
        idx = {h.name: i for i, h in enumerate(objects)}
        return lambda known, trace: HardObserver(known, trace, idx, sink, mtx, check)
    return shadow


# --------------------------------------------------------------------------- #
# Replaying one job                                                            #
# --------------------------------------------------------------------------- #

KEEP = ("obj", "in_pool", "outcome", "n_questions", "p_unknown_max", "fire_turn",
        "k_before")


def replay(src: str, job: dict, matrix_mode: str, committed: Optional[dict] = None
           ) -> Tuple[dict, float]:
    """Play ``job`` with the hard observer on. With ``committed``, check identity."""
    m = _module(src)
    sink: List[dict] = []
    check = [CHECK_GAMES, 0]
    names: list = []
    t0 = time.time()
    data, sc = m.play(job, matrix_mode, shadow=hard_hook(sink, check, names))
    secs = time.time() - t0
    g = data["games"]
    if len(sink) != len(g["obj"]):
        raise AssertionError("one observer row per game")
    identical = None
    if committed is not None:
        identical = g == committed["games"] and sc == committed["scorecard"]
    cols = {k: [r[k] for r in sink] for k in RH_COLS}
    for k in KEEP:
        cols[k] = g[k]
    out = {"src": src, "job": job, "identical": identical, "games": cols,
           "rederived": CHECK_GAMES - check[0], "rederive_failures": check[1],
           "seconds": secs}
    if src == "st2":
        pof, at = data["meta"]["parent_of"], {n: i for i, n in enumerate(names)}
        out["parent_idx"] = [at[pof[n]] if n in pof else -1 for n in names]
    return out, secs


def replay_job(args) -> Tuple[str, Optional[bool], float]:
    j, outdir, matrix_mode = args
    path = os.path.join(outdir, "jobs", j["id"] + ".json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        return j["id"], rec["identical"], rec["seconds"]
    if "job" in j:                              # smoke: a fresh synthetic job
        rec, secs = replay(j["src"], j["job"], matrix_mode)
    else:
        committed = _load(j["src"], j["job_id"])
        rec, secs = replay(j["src"], committed["job"], matrix_mode, committed)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return j["id"], rec["identical"], secs


# --------------------------------------------------------------------------- #
# Run                                                                          #
# --------------------------------------------------------------------------- #

def run(outdir: str, jobs: List[dict], matrix_mode: str, config: dict,
        workers: int) -> None:
    from .stress_st3 import _segment
    os.makedirs(os.path.join(outdir, "jobs"), exist_ok=True)
    cfg_path = os.path.join(outdir, "config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    _segment(outdir, provenance(), workers)
    t0 = time.time()
    results = {}
    with Pool(workers) as pool:
        for i, (jid, same, secs) in enumerate(pool.imap_unordered(
                replay_job, [(j, outdir, matrix_mode) for j in jobs]), 1):
            results[jid] = {"identical": same, "seconds": secs}
            if same is False:
                print("  NOT IDENTICAL: {}".format(jid), flush=True)
            if i % 25 == 0 or i == len(jobs):
                print("  {}/{} jobs  ({:.0f} min elapsed)".format(
                    i, len(jobs), (time.time() - t0) / 60), flush=True)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"schema": "twenty_questions_stress_revshadow_v1", "config": config,
                   "jobs": [dict(j, **results[j["id"]]) for j in jobs]}, f, indent=1,
                  sort_keys=True)
    print("wrote {}".format(outdir))


def regress(ids: Sequence[str]) -> None:
    """Replay committed jobs with the observer attached; require identity. Prints only
    identity and timing; the observer's measurements are discarded."""
    for s in ids:
        src, jid = s.split(":", 1)
        committed = _load(src, jid)
        rec, secs = replay(src, committed["job"], "real", committed)
        ok = rec["identical"] and rec["rederive_failures"] == 0
        print("  {:4s} {:52s} {} ({:.0f} s)".format(
            src, jid, "identical" if ok else "DIFFERS", secs), flush=True)
        if not ok:
            raise SystemExit("regression: {} {}".format(src, jid))
    print("[regress] {} committed jobs replay identically with the observer on".format(len(ids)))


def smoke_jobs(games_st3: int = 5000, games_st2: int = 2500) -> List[dict]:
    """Synthetic-matrix `soft` jobs shaped like the registered grid's extremes."""
    from . import stress_st2 as s2
    from . import stress_st3 as s3
    jobs = []
    for j in s2.jobs_for(d_levels=[1, 12], flips=[0.0, 0.10], backends=["soft"], g_rates=[]):
        if j["la_seed"] == 1 and j["noise_seed"] in (0, 1):
            jobs.append({"src": "st2", "job": dict(j, games=games_st2)})
    for j in s3.jobs_for(sigmas=[0.0, 0.75], priors=["true"], flips=[0.0, 0.05],
                         stream_seeds=[1]):
        if j["backend"] == "soft" and j["noise_seed"] in (0, 1):
            jobs.append({"src": "st3", "job": dict(j, games=games_st3)})
    for j in s3.endgame_jobs(rules=["always"], sigmas=[0.75], flips=[0.0], stream_seeds=[1]):
        if j["backend"] == "soft":
            jobs.append({"src": "end", "job": dict(j, games=games_st3)})
    for j in jobs:
        j["id"] = "{}__{}".format(j["src"], j["job"]["job_id"])
    return jobs


# --------------------------------------------------------------------------- #
# Self-test (synthetic matrix only)                                            #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    from . import stress_st2 as s2
    from . import stress_st3 as s3
    # 1. The observer never changes a game, and its hard set re-derives exactly.
    for label, j3 in (
            ("ST-3 sigma 0.75 flip 0.05", [j for j in s3.jobs_for(
                sigmas=[0.75], priors=["true"], flips=[0.05], stream_seeds=[1])
                if j["backend"] == "soft"][0]),
            ("end-game always sigma 0.75 flip 0", [j for j in s3.endgame_jobs(
                rules=["always"], sigmas=[0.75], flips=[0.0], stream_seeds=[1])
                if j["backend"] == "soft"][0])):
        j3 = dict(j3, games=400)
        plain, sc = s3.play(j3, "synthetic")
        sink, check = [], [10 ** 9, 0]
        with_h, sc2 = s3.play(j3, "synthetic", shadow=hard_hook(sink, check))
        assert plain["games"] == with_h["games"] and sc == sc2, \
            "observer changed a game ({})".format(label)
        assert check[1] == 0, "{} games fail the re-derivation ({})".format(check[1], label)
        g = plain["games"]
        # The observer's budget-end record agrees with the committed end columns
        # (end-game jobs carry them) and with the outcome (every budget end).
        if "end_budget" in g:
            for r, eb, ul in zip(sink, g["end_budget"], g["end_u_leads"]):
                assert (r["rh_u_leads"] >= 0) == bool(eb)
                assert r["rh_u_leads"] < 0 or r["rh_u_leads"] == ul
        for r, out, nq in zip(sink, g["outcome"], g["n_questions"]):
            if out == 2:                        # wrong: always a budget-end answer
                assert r["rh_u_leads"] == 0 and r["rh_end_top"] >= 0 and nq == BUDGET
        if j3["flip"] == 0.0:
            bad = [i for i, (r, ip) in enumerate(zip(sink, g["in_pool"]))
                   if ip and (r["rh_secret_contra"] != 0 or r["rh_alarm"] >= 0)]
            assert not bad, "secret struck under exact answers at games {}".format(bad[:5])
        n_end = sum(1 for r in sink if r["rh_end_top"] >= 0)
        n_struck = sum(1 for r in sink if r["rh_end_struck"] == 1)
        n_alarm = sum(1 for r in sink if r["rh_alarm"] >= 0)
        print("[self-test] {}: 400 games identical with the observer on; every hard set "
              "re-derives; {} budget ends ({} with end_top struck), {} alarms".format(
                  label, n_end, n_struck, n_alarm))
    # 2. ST-2, exact and noisy answers.
    for flip in (0.0, 0.10):
        j2 = [j for j in s2.jobs_for(d_levels=[1], flips=[flip], backends=["soft"],
                                     g_rates=[])
              if j["la_seed"] == 1][0]
        j2 = dict(j2, games=600)
        plain2, sc_a = s2.play(j2, "synthetic")
        sink3, check3 = [], [10 ** 9, 0]
        w2, sc_b = s2.play(j2, "synthetic", shadow=hard_hook(sink3, check3))
        assert plain2["games"] == w2["games"] and sc_a == sc_b, "observer changed an ST-2 game"
        assert check3[1] == 0
        gg = plain2["games"]
        if flip == 0.0:
            assert all(not ip or (r["rh_secret_contra"] == 0 and r["rh_alarm"] < 0)
                       for r, ip in zip(sink3, gg["in_pool"])), "true object struck"
        # tolerance-1 is never stricter than tolerance 0
        assert all(r["rh_alarm1"] < 0 or (r["rh_alarm"] >= 0 and r["rh_alarm1"] >= r["rh_alarm"])
                   for r in sink3)
        assert all(r["rh_surv1_end"] >= r["rh_surv_end"] for r in sink3)
        print("[self-test] ST-2 synthetic d=1 flip {}: identical, re-derived; {} alarms, "
              "{} budget ends with end_top struck".format(
                  flip, sum(1 for r in sink3 if r["rh_alarm"] >= 0),
                  sum(1 for r in sink3 if r["rh_end_struck"] == 1)))
    # 3. The grid.
    jobs = plan()
    by: Dict[str, int] = {}
    for j in jobs:
        by[j["src"]] = by.get(j["src"], 0) + 1
    assert by == {"st2": 147, "st3": 36, "end": 36}, by
    print("[self-test] grid: {} jobs {}".format(len(jobs), by))
    print("[self-test] PASS")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    from .stress_harness import require_hash_seed
    from .stress_st3 import check_prereg
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--regress", nargs="+", metavar="SRC:JOB_ID")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--prereg-tag")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir")
    a = ap.parse_args()
    hs = require_hash_seed()
    blas = pin_blas(1)
    if a.plan:
        jobs = plan()
        for j in jobs:
            print(j["id"])
        print("{} jobs".format(len(jobs)))
    elif a.self_test:
        self_test()
    elif a.regress:
        regress(a.regress)
    elif a.smoke or a.run:
        if a.run:
            if not a.prereg_tag:
                raise SystemExit("--run needs --prereg-tag")
            reg = check_prereg(a.prereg_tag, PREREG)
            jobs, mode = plan(), "real"
        else:
            reg, jobs, mode = {}, smoke_jobs(), "synthetic"
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = a.outdir or os.path.join(RESULTS, "stress_revshadow_{}".format(stamp))
        config = {"mode": "run" if a.run else "smoke", "matrix_mode": mode,
                  "created": stamp, "pythonhashseed": hs, "blas": blas,
                  "check_games": CHECK_GAMES, "eps": EPS, "kill": KILL, "budget": BUDGET,
                  "sources": SOURCES, "n_jobs": len(jobs), **reg}
        run(outdir, jobs, mode, config, a.workers)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
