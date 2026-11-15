"""Shadow-mode `soft` beside P0a (``STRESS_SHADOW_PREREG.md``; Jira EE-40, ENV-705).

A passive ``SoftPosterior`` rides committed P0a games through the loop's ``observer``
hook. It is updated on every answer the game produces, with P0a's own likelihood,
known pool and novelty prior, and it never touches the game: each replayed job must
match its committed record in every per-game column and the scorecard.

Per game it records where the two views disagree. A turn *disagrees* when `soft`'s
leading known has been struck by the hard view. It also records `soft`'s leader at
P0a's decision, the near-copy reference for novel games, and, at each P0a revival,
the revived object and `soft`'s leader after the same answer.

    # self-test (synthetic matrix only, a few minutes)
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_shadow --self-test
    # the grid, as registered
    ... stress_shadow --plan
    # replay committed jobs with the shadow on and require identity (real matrix; prints timing only)
    ... stress_shadow --regress st2:exact_d1_la1_f0.1_s1_g0_modeswitch_guard
    # synthetic smoke: fresh games with the shadow, for plumbing and timing
    ... stress_shadow --smoke --workers 4 --outdir <dir>
    # the registered run
    ... stress_shadow --run --prereg-tag <tag> --workers 6 --outdir <dir>
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

import numpy as np

from .stress_st1 import pin_blas, provenance

PREREG = "llm_rl_scripts/twenty_questions/jev/STRESS_SHADOW_PREREG.md"
RESULTS = "llm_rl_scripts/twenty_questions/jev/results"
SOURCES = {
    "st2": os.path.join(RESULTS, "stress_st2_20261112"),
    "st3": os.path.join(RESULTS, "stress_st3_20261113"),
    "end": os.path.join(RESULTS, "stress_st3_endgame_20261114"),
}
BUDGET = 20
EPS = 0.05
CHECK_GAMES = 50          # games per job whose shadow posterior is re-derived (SH0 b)
SHADOW_COLS = ("sh_n_obs", "sh_disagree", "sh_first_disagree", "sh_leader_first",
               "sh_leader_dec", "sh_struck_dec", "sh_p_leader_dec", "sh_p_unknown_dec",
               "sh_turns_left_dec", "sh_ref", "sh_revivals", "sh_guesses")


# --------------------------------------------------------------------------- #
# The grid                                                                     #
# --------------------------------------------------------------------------- #

def _ids(src: str) -> List[str]:
    d = os.path.join(SOURCES[src], "jobs")
    return sorted(f[:-len(".json.gz")] for f in os.listdir(d) if f.endswith(".json.gz"))


def plan() -> List[dict]:
    """The registered 255 jobs, each a committed P0a job to replay:

    * ST-2, exact answerer (no ``G_`` name confusion), P0a: 147;
    * ST-3, ``true`` prior, published P0a and P0a with ``no_repeat``: 72;
    * ST-3 end-game, P0a with ``no_repeat`` under ``always``: 36.

    Ordered slowest first (large tails), so the pool drains evenly."""
    jobs = []
    for jid in _ids("st2"):
        if jid.startswith("exact_") and jid.endswith("_modeswitch_guard"):
            jobs.append({"src": "st2", "job_id": jid})
    for jid in _ids("st3"):
        if "_true_" in jid and jid.endswith("_modeswitch_guard"):
            jobs.append({"src": "st3", "job_id": jid})
    for jid in _ids("end"):
        if jid.endswith("_norepeat_modeswitch_guard_endalways"):
            jobs.append({"src": "end", "job_id": jid})
    for j in jobs:
        j["id"] = "{}__{}".format(j["src"], j["job_id"])
    return sorted(jobs, key=lambda j: (j["src"] == "st2", -_sigma(j), j["id"]))


def _sigma(j: dict) -> int:
    return int(j["job_id"][1:3]) if j["src"] != "st2" else 0


# --------------------------------------------------------------------------- #
# The shadow observer                                                          #
# --------------------------------------------------------------------------- #

def _signs(objects, mtx) -> np.ndarray:
    """objects x attributes: +1 above 0.5, -1 below, 0 at 0.5."""
    attrs = sorted(mtx)
    vals = np.array([[mtx[a].get(h.name, 0.5) for a in attrs] for h in objects])
    return np.sign(vals - 0.5).astype(np.int8)


class ShadowObserver:
    """Wraps a job's own per-game trace; keeps a passive ``SoftPosterior``."""

    def __init__(self, known, trace, idx: Dict[str, int], signs: np.ndarray,
                 sink: List[dict], mtx, check: List[int]):
        self.known = known
        self.names = [h.name for h in known]
        self.trace = trace
        self.idx = idx
        self.signs = signs
        self.sink = sink
        self.mtx = mtx
        self.check = check          # [games left to re-derive, failures]
        self.sp = None
        self.U = None
        self.n_obs = 0
        self.disagree = 0
        self.first = -1
        self.leader_first = -1
        self.dec = (-1, -1, -1.0, -1.0)
        self.revs: List[List[int]] = []
        self.prev_rev = 0

    def __call__(self, turn, q, ans, bk, surv_before) -> None:
        from .openworld_backends import SoftPosterior
        self.trace(turn, q, ans, bk, surv_before)
        if not self.names:
            return
        if self.sp is None:
            self.U = bk.U
            self.sp = SoftPosterior(self.known, bk.by_name, bk.like, bk.U, bk.u_prior,
                                    eps=EPS)
            self.u0 = bk.u_prior
        self.sp.observe(q, ans)
        self.n_obs += 1
        post = self.sp.post
        best, bp = None, 0.0
        for n in self.names:            # argmax over knowns, ties in known order
            if post[n] > bp:
                best, bp = n, post[n]
        # Category and by-name answers are exact for `soft` too, so every known can
        # reach mass 0. Then there is no leader, and the turn cannot disagree.
        surv = bk.survivors
        struck = best is not None and best not in surv
        if struck:
            self.disagree += 1
            if self.first < 0:
                self.first, self.leader_first = turn, self.idx[best]
        nrev = getattr(bk, "n_revives", 0)
        if nrev > self.prev_rev:
            # A revival happens only when the live set empties, and revives one known,
            # so the live set after it is that known alone.
            if nrev != self.prev_rev + 1 or len(surv) != 1:
                raise AssertionError("unexpected revival shape")
            self.revs.append([self.idx[next(iter(surv))],
                              -1 if best is None else self.idx[best]])
            self.prev_rev = nrev
        self.dec = ((-1, -1, 0.0, post[self.U]) if best is None
                    else (self.idx[best], int(struck), bp, post[self.U]))

    def _rederive(self, ep) -> bool:
        """SH0(b): a standalone SoftPosterior fed the game's own steps (every answer
        before the decision; a final 'yes' guess is not an update) equals the shadow."""
        from .bed import GUESS
        from .openworld import (UNKNOWN, UNKNOWN_HYP, build_openworld_bank,
                                make_openworld_likelihood)
        from .openworld_backends import SoftPosterior
        all_hyps = list(self.known) + [UNKNOWN_HYP]
        by_name = {h.name: h for h in all_hyps}
        like = make_openworld_likelihood(self.mtx, self.known, eps=EPS)
        bank = build_openworld_bank(self.known)
        by_qid = {q.qid: q for qs in bank.values() for q in qs}
        sp = SoftPosterior(self.known, by_name, like, UNKNOWN, self.u0, eps=EPS)
        steps = list(ep.steps)
        if steps and steps[-1].kind == GUESS and steps[-1].answer_yes:
            steps = steps[:-1]
        for s in steps:
            sp.observe(by_qid[s.qid], s.answer_yes)
        return len(steps) == self.n_obs and sp.post == self.sp.post

    def finish(self, ep) -> None:
        self.trace.finish(ep)
        secret = ep.true_name
        ref = -1
        if self.names and secret not in set(self.names):
            kidx = np.fromiter((self.idx[n] for n in self.names), dtype=np.int64,
                               count=len(self.names))
            s = self.signs[self.idx[secret]]
            dist = (self.signs[kidx] * s == -1).sum(axis=1)
            ref = int(kidx[int(np.argmin(dist))])      # first minimum = known order
        if self.sp is not None and self.check[0] > 0:
            self.check[0] -= 1
            if not self._rederive(ep):
                self.check[1] += 1
        from .bed import GUESS
        guesses = []                    # every by-name guess: [turn, object]
        for st in ep.steps:
            if st.kind == GUESS:
                nm = st.qid.split(":", 1)[1] if ":" in st.qid else st.qid
                guesses.append([st.turn, self.idx[nm]])
        lead, struck, pl, pu = self.dec
        self.sink.append({
            "sh_n_obs": self.n_obs, "sh_disagree": self.disagree,
            "sh_first_disagree": self.first, "sh_leader_first": self.leader_first,
            "sh_leader_dec": lead, "sh_struck_dec": struck, "sh_p_leader_dec": pl,
            "sh_p_unknown_dec": pu, "sh_turns_left_dec": BUDGET - ep.n_questions,
            "sh_ref": ref, "sh_revivals": self.revs, "sh_guesses": guesses})


def shadow_hook(sink: List[dict], check: List[int], names: Optional[list] = None):
    """The ``shadow`` argument of ``stress_st2.play`` / ``stress_st3.play``. With
    ``names``, the job's object names are appended to it."""
    def shadow(objects, mtx):
        if names is not None:
            names.extend(h.name for h in objects)
        idx = {h.name: i for i, h in enumerate(objects)}
        signs = _signs(objects, mtx)
        return lambda known, trace: ShadowObserver(known, trace, idx, signs, sink, mtx,
                                                   check)
    return shadow


# --------------------------------------------------------------------------- #
# Replaying one job                                                            #
# --------------------------------------------------------------------------- #

def _module(src: str):
    if src == "st2":
        from . import stress_st2 as m
    else:
        from . import stress_st3 as m
    return m


def _load(src: str, job_id: str) -> dict:
    with gzip.open(os.path.join(SOURCES[src], "jobs", job_id + ".json.gz"), "rt") as f:
        return json.load(f)


def replay(src: str, job: dict, matrix_mode: str, committed: Optional[dict] = None
           ) -> Tuple[dict, float]:
    """Play ``job`` with the shadow on. With ``committed``, check identity."""
    m = _module(src)
    sink: List[dict] = []
    check = [CHECK_GAMES, 0]
    names: list = []
    t0 = time.time()
    data, sc = m.play(job, matrix_mode, shadow=shadow_hook(sink, check, names))
    secs = time.time() - t0
    g = data["games"]
    if len(sink) != len(g["obj"]):
        raise AssertionError("one shadow row per game")
    identical = None
    if committed is not None:
        identical = g == committed["games"] and sc == committed["scorecard"]
    cols = {k: [r[k] for r in sink] for k in SHADOW_COLS}
    for k in ("obj", "in_pool", "outcome", "n_questions", "n_retractions"):
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
        json.dump({"schema": "twenty_questions_stress_shadow_v1", "config": config,
                   "jobs": [dict(j, **results[j["id"]]) for j in jobs]}, f, indent=1,
                  sort_keys=True)
    print("wrote {}".format(outdir))


def regress(ids: Sequence[str]) -> None:
    """Replay committed jobs with the shadow attached; require identity. Prints only
    identity and timing; the shadow's measurements are discarded."""
    for s in ids:
        src, jid = s.split(":", 1)
        committed = _load(src, jid)
        rec, secs = replay(src, committed["job"], "real", committed)
        ok = rec["identical"] and rec["rederive_failures"] == 0
        print("  {:4s} {:52s} {} ({:.0f} s)".format(
            src, jid, "identical" if ok else "DIFFERS", secs), flush=True)
        if not ok:
            raise SystemExit("regression: {} {}".format(src, jid))
    print("[regress] {} committed jobs replay identically with the shadow on".format(len(ids)))


def smoke_jobs(games_st3: int = 5000, games_st2: int = 2500) -> List[dict]:
    """Synthetic-matrix jobs shaped like the registered grid's extremes."""
    from . import stress_st2 as s2
    from . import stress_st3 as s3
    jobs = []
    for j in s2.jobs_for(d_levels=[1, 12], flips=[0.0, 0.10], backends=["modeswitch_guard"],
                         g_rates=[]):
        if j["la_seed"] == 1 and j["noise_seed"] in (0, 1):
            jobs.append({"src": "st2", "job": dict(j, games=games_st2)})
    for j in s3.jobs_for(sigmas=[0.0, 0.75], priors=["true"], flips=[0.0, 0.05],
                         stream_seeds=[1]):
        if j["backend"] == "modeswitch_guard" and j["noise_seed"] in (0, 1):
            jobs.append({"src": "st3", "job": dict(j, games=games_st3)})
    for j in s3.endgame_jobs(rules=["always"], sigmas=[0.75], flips=[0.0], stream_seeds=[1]):
        if j["no_repeat"] and j["backend"] == "modeswitch_guard":
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
    # 1. The shadow never changes a game, and its posterior re-derives exactly.
    j3 = [j for j in s3.endgame_jobs(rules=["always"], sigmas=[0.75], flips=[0.05],
                                     stream_seeds=[1])
          if j["no_repeat"] and j["backend"] == "modeswitch_guard"][0]
    j3 = dict(j3, games=400)
    plain, sc = s3.play(j3, "synthetic")
    sink, check = [], [10 ** 9, 0]
    with_sh, sc2 = s3.play(j3, "synthetic", shadow=shadow_hook(sink, check))
    assert plain["games"] == with_sh["games"] and sc == sc2, "shadow changed an ST-3 game"
    assert check[1] == 0, "{} ST-3 games fail the re-derivation".format(check[1])
    g = plain["games"]
    revs = sum(len(r["sh_revivals"]) for r in sink)
    assert all(len(r["sh_revivals"]) == n for r, n in zip(sink, g["n_retractions"])), \
        "revival records must match n_retractions"
    print("[self-test] ST-3 synthetic: 400 games identical with the shadow on; "
          "every posterior re-derives; {} revivals recorded".format(revs))
    # The prior the shadow starts from is the committed per-game prior.
    # (checked indirectly: re-derivation uses the shadow's u0; here compare u0 to the column)
    sink2, check2 = [], [0, 0]
    priors_seen: List[float] = []

    def spy(objects, mtx):
        make = shadow_hook(sink2, check2)(objects, mtx)

        def wrap(known, trace):
            o = make(known, trace)
            fin = o.finish

            def finish(ep):
                priors_seen.append(getattr(o, "u0", None))
                fin(ep)
            o.finish = finish
            return o
        return wrap
    s3.play(dict(j3, games=150), "synthetic", shadow=spy)
    col = plain["games"]["prior"][:150]
    bad = [i for i, (a, b) in enumerate(zip(priors_seen, col)) if a is not None and a != b]
    assert not bad, "shadow prior differs from the committed prior at games {}".format(bad[:5])
    print("[self-test] the shadow's starting prior equals the job's prior column (150 games)")
    # 2. ST-2, exact answers: the true object is never struck, so no in-pool game has a
    #    disagreeing turn whose leader is the secret (SH0 c); revivals are recorded.
    for flip in (0.0, 0.10):
        j2 = [j for j in s2.jobs_for(d_levels=[1], flips=[flip],
                                     backends=["modeswitch_guard"], g_rates=[])
              if j["la_seed"] == 1][0]
        j2 = dict(j2, games=600)
        plain2, sc_a = s2.play(j2, "synthetic")
        sink3, check3 = [], [10 ** 9, 0]
        w2, sc_b = s2.play(j2, "synthetic", shadow=shadow_hook(sink3, check3))
        assert plain2["games"] == w2["games"] and sc_a == sc_b, "shadow changed an ST-2 game"
        assert check3[1] == 0
        gg = plain2["games"]
        hit = sum(1 for r, o, ip in zip(sink3, gg["obj"], gg["in_pool"])
                  if ip and r["sh_first_disagree"] >= 0 and r["sh_leader_first"] == o)
        dis = sum(1 for r in sink3 if r["sh_first_disagree"] >= 0)
        if flip == 0.0:
            assert hit == 0, "true object struck under exact answers"
        print("[self-test] ST-2 synthetic d=1 flip {}: identical, re-derived; {} games "
              "disagree, {} in-pool with the secret leading at first disagreement".format(
                  flip, dis, hit))
    # 3. The grid.
    jobs = plan()
    by = {}
    for j in jobs:
        by[j["src"]] = by.get(j["src"], 0) + 1
    assert by == {"st2": 147, "st3": 72, "end": 36}, by
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
        outdir = a.outdir or os.path.join(RESULTS, "stress_shadow_{}".format(stamp))
        config = {"mode": "run" if a.run else "smoke", "matrix_mode": mode,
                  "created": stamp, "pythonhashseed": hs, "blas": blas,
                  "check_games": CHECK_GAMES, "eps": EPS, "budget": BUDGET,
                  "sources": SOURCES, "n_jobs": len(jobs), **reg}
        run(outdir, jobs, mode, config, a.workers)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
