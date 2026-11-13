"""ST-3 runner: the endless tail (``STRESS_TAIL_PREREG.md``).

    # list the grid and its job counts (no games played):
    PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
        -m llm_rl_scripts.twenty_questions.jev.stress_st3 --plan

    # the streams' own missing mass, per sigma and stream seed (no games played):
    ... stress_st3 --stream-census --out <file.json>

    # mechanics self-test on the *synthetic* matrix (no ST-3 outcome is produced):
    ... stress_st3 --self-test

    # plumbing check + timing on the synthetic matrix:
    ... stress_st3 --smoke --outdir <dir>

    # the registered run (refuses unless --prereg-tag is an ancestor of HEAD and
    # the pre-registration is unchanged since it):
    ... stress_st3 --run --prereg-tag prereg-stress-st3-<date> --workers 6

Accumulation mode (``openworld.run_accumulation``) on a Pitman-Yor stream with
an unbounded supply of species. Game n opens a new species with probability
(theta + K sigma) / (theta + n), K the distinct species so far; otherwise it
repeats seen species j with weight (count_j - sigma). The first 158 species are
the real objects in a seeded random order, the rest come from the ST-0a
generator at rho_real. The stream depends only on (sigma, stream seed), so every
prior, backend and flip rate plays the same stream.

The planner is the numpy path (``stress_fastplan.FastPlanner``, ``tie_sorted``
rule), rebuilt each time ``H_known`` grows: the pool reaches ~1750 objects at
sigma = 0.75, which the Python path cannot play in useful time. Every job
writes ``jobs/<id>.json.gz`` (scorecard, per-game columns, the census of
``H_known`` every 250 games and, for the online prior, the fitted (sigma,
theta) path), so an interrupted run resumes by skipping finished jobs.

All seeds derive from fixed integers -- never ``hash()``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

from .stress_st1 import BLAS_VARS, _git, pin_blas, provenance

SIGMAS = [0.0, 0.25, 0.5, 0.75]
THETA = 20.0
PRIORS = ["true", "online", "fixed"]
FIXED_SIGMA, FIXED_THETA = 0.2, 20.0       # the §3.11 fit
REFIT_EVERY = 100
FLIPS = [0.0, 0.05]
STREAM_SEEDS = [1, 2, 3]
NOISE_SEEDS = [1, 2]
ARMS = [("primary", "soft", False), ("primary", "hard", False),
        ("primary", "modeswitch", False), ("primary", "modeswitch_guard", False),
        ("norepeat", "hard", True), ("norepeat", "modeswitch_guard", True)]
GAMES = 5000
WINDOW = 250
RHO_REAL = 0.2345
N_REAL = 158
BUDGET = 20
CONFIDENCE = 0.9
EPS = 0.05
K = 3
PREREG = "llm_rl_scripts/twenty_questions/jev/STRESS_TAIL_PREREG.md"
MATRIX = "llm_rl_scripts/twenty_questions/jev/matrix_jev.json"
OUTCOME_CODE = {"won": 0, "expert": 1, "wrong": 2, "false_confirm": 3}


def _sid(sigma: float) -> int:
    return int(round(sigma * 100))


# --------------------------------------------------------------------------- #
# Grid                                                                         #
# --------------------------------------------------------------------------- #

def jobs_for(sigmas=SIGMAS, priors=PRIORS, flips=FLIPS, stream_seeds=STREAM_SEEDS,
             noise_seeds=NOISE_SEEDS, arms=ARMS) -> List[dict]:
    """Every sigma x prior x flip x stream seed (x noise seed under noise; flip 0
    is deterministic, so it has one) x arm."""
    jobs = []
    for sg in sigmas:
        for pr in priors:
            for flip in flips:
                for ss in stream_seeds:
                    for ns in ([0] if flip == 0.0 else noise_seeds):
                        for arm, bk, nr in arms:
                            jid = "s{:02d}_{}_f{:g}_ss{}_ns{}_{}_{}".format(
                                _sid(sg), pr, flip, ss, ns, arm, bk)
                            jobs.append({
                                "job_id": jid, "sigma": sg, "theta": THETA,
                                "prior": pr, "flip": flip, "stream_seed": ss,
                                "noise_seed": ns, "arm": arm, "backend": bk,
                                "no_repeat": nr,
                                "oracle_seed": 5_000_000 + 10_000 * _sid(sg)
                                + 100 * ss + ns})
    # Heavy tails grow the largest pools; start them first for load balance.
    jobs.sort(key=lambda j: (-j["sigma"], j["backend"] != "soft", j["job_id"]))
    return jobs


# --------------------------------------------------------------------------- #
# The stream                                                                   #
# --------------------------------------------------------------------------- #

def py_stream(sigma: float, theta: float, games: int, seed: int) -> List[int]:
    """A Pitman-Yor (Chinese-restaurant) stream of species ids, numbered in order
    of first appearance (game g opens species K with probability
    (theta + K sigma) / (theta + g))."""
    rng = random.Random(seed)
    counts: List[int] = []
    out: List[int] = []
    for g in range(games):
        k = len(counts)
        if rng.random() * (theta + g) < theta + k * sigma:
            counts.append(1)
            out.append(k)
            continue
        r = rng.random() * (g - k * sigma)
        acc = 0.0
        pick = k - 1
        for j, c in enumerate(counts):
            acc += c - sigma
            if acc > r:
                pick = j
                break
        counts[pick] += 1
        out.append(pick)
    return out


def stream_seeds(sigma: float, ss: int) -> Dict[str, int]:
    base = 4_000_000 + 1000 * _sid(sigma) + ss
    return {"stream": base, "order": base + 7, "generate": base + 50}


def build(sigma: float, ss: int, games: int, matrix_mode: str):
    """(objects, matrix incl. generated rows, stream) for one (sigma, stream seed);
    deterministic. ``objects[i]`` is species i."""
    from .bed import load_hypotheses
    from .matrix import load_matrix, synthetic_matrix
    from .stress_generator import generate_objects, merge

    seeds = stream_seeds(sigma, ss)
    stream = py_stream(sigma, THETA, games, seeds["stream"])
    n_species = max(stream) + 1
    objs = load_hypotheses()
    base = (synthetic_matrix(objs) if matrix_mode == "synthetic"
            else load_matrix(MATRIX)["matrix"])
    real = list(objs)
    random.Random(seeds["order"]).shuffle(real)
    n_gen = max(0, n_species - N_REAL)
    gen, rows = generate_objects(objs, base, n_gen, RHO_REAL, seeds["generate"],
                                 prefix="tail")
    objects = (real + gen)[:n_species]
    return objects, merge(base, rows), stream


def stream_census(sigma: float, ss: int, games: int = GAMES,
                  window: int = WINDOW) -> dict:
    """The stream's own novelty, independent of any loop: per window, the
    realized new-species rate and the mean true p_new = (theta + K sigma) /
    (theta + g) entering each game (K = distinct species so far)."""
    seeds = stream_seeds(sigma, ss)
    stream = py_stream(sigma, THETA, games, seeds["stream"])
    half = games // 2
    seen = 0
    new_rate, pnew, per_game = [], [], []
    w_new, w_p = 0, 0.0
    above = 0
    species_at_half = 0
    for g, s in enumerate(stream):
        if g == half:
            species_at_half = seen
        p = (THETA + seen * sigma) / (THETA + g)
        per_game.append(p)
        above += p > 0.15
        w_p += p
        if s == seen:           # ids are numbered by first appearance
            w_new += 1
            seen += 1
        if (g + 1) % window == 0:
            new_rate.append(w_new / window)
            pnew.append(w_p / window)
            w_new, w_p = 0, 0.0
    return {"sigma": sigma, "stream_seed": ss, "species": seen,
            "species_at_half": species_at_half,
            "window_new_rate": new_rate, "window_true_pnew": pnew,
            "true_pnew_first_half": sum(per_game[:half]) / half,
            "true_pnew_second_half": sum(per_game[half:]) / (games - half),
            "games_prior_above_0.15": above}


# --------------------------------------------------------------------------- #
# Priors                                                                       #
# --------------------------------------------------------------------------- #

class Prior:
    """``prior_fn`` for ``run_accumulation``. ``true``: the generating (sigma,
    theta); ``fixed``: the §3.11 fit; ``online``: the fixed values until game
    REFIT_EVERY, then the PY MLE refitted every REFIT_EVERY games on the loop's
    own history -- a game is "new" iff it ended in an admission, so K in the fit
    is |H_known|, the same K the gate reads. Logs the prior of every game."""

    def __init__(self, kind: str, sigma: float, theta: float):
        self.kind = kind
        if kind == "true":
            self.sigma, self.theta = sigma, theta
        elif kind in ("fixed", "online"):
            self.sigma, self.theta = FIXED_SIGMA, FIXED_THETA
        else:
            raise ValueError(kind)
        self.priors: List[float] = []
        self.fits: List[Tuple[int, float, float]] = []

    def __call__(self, g: int, k_before: int, records) -> float:
        from .openworld import fit_pitman_yor, py_p_new
        if self.kind == "online" and g > 0 and g % REFIT_EVERY == 0:
            flags = [r.expert_called and not r.in_pool for r in records]
            self.sigma, self.theta, _ = fit_pitman_yor(flags)
            self.fits.append((g, self.sigma, self.theta))
        p = py_p_new(k_before, g, self.sigma, self.theta)
        self.priors.append(1.0 if k_before == 0 else p)
        return p


# --------------------------------------------------------------------------- #
# Playing a job                                                                #
# --------------------------------------------------------------------------- #

class Trace:
    """Per-game ``observer`` for ``run_accumulation``: repeated questions and the
    live-set size at the end of each finished game."""

    def __init__(self, sink: List[dict]):
        self.sink = sink

    def __call__(self, turn, q, ans, bk, surv_before) -> None:
        pass

    def finish(self, ep) -> None:
        from .bed import GUESS
        others = [s.qid for s in ep.steps if s.kind != GUESS]
        self.sink.append({"n_repeat_questions": len(others) - len(set(others)),
                          "n_live_final": ep.n_live_final})


def make_oracle(job: dict, matrix):
    from .bed import matrix_oracle
    from .openworld import noisy_matrix_oracle
    return (matrix_oracle(matrix) if job["flip"] == 0.0
            else noisy_matrix_oracle(matrix, job["flip"], seed=job["oracle_seed"]))


def planner_factory(matrix, python: bool = False):
    """A FastPlanner for the current H_known (None while it is empty). With
    ``python``, the Python reference ranked by the same tie rule (parity only)."""
    from .stress_fastplan import FastPlanner, SnapPythonPlanner

    def make(known):
        if not known:
            return None
        fast = FastPlanner(known, matrix, eps=EPS)
        return SnapPythonPlanner(fast) if python else fast
    return make


def known_census(objects, matrix, recs, window: int = WINDOW) -> List[dict]:
    """The census of H_known entering each window (admission order)."""
    from .stress_census import census
    names: List[str] = []
    out = []
    known = set()
    for g, r in enumerate(recs):
        if g % window == 0:
            if names:
                c = census(matrix, names)
                out.append({"game": g, "n_known": len(names),
                            "distinct_rows": c["distinct_rows"],
                            "nn_within_1": c["nn_within_1"]})
            else:
                out.append({"game": g, "n_known": 0, "distinct_rows": 0, "nn_within_1": 0})
        if r.expert_called and not r.in_pool and r.true_name not in known:
            known.add(r.true_name)
            names.append(r.true_name)
    return out


def play(job: dict, matrix_mode: str, python_planner: bool = False) -> Tuple[dict, dict]:
    """(per-game columns + job metadata, scorecard) for one job."""
    from .openworld import run_accumulation
    from .stress_harness import scorecard

    games = job.get("games", GAMES)
    objects, mtx, stream = build(job["sigma"], job["stream_seed"], games, matrix_mode)
    prior = Prior(job["prior"], job["sigma"], job["theta"])
    traces: List[dict] = []
    recs = run_accumulation(objects, mtx, stream, prior.sigma, prior.theta,
                            max_questions=BUDGET, confidence=CONFIDENCE, eps=EPS,
                            backend=job["backend"], tolerance=K,
                            oracle=make_oracle(job, mtx), no_repeat=job["no_repeat"],
                            observer_factory=lambda known: Trace(traces),
                            prior_fn=prior,
                            planner_factory=planner_factory(mtx, python_planner))
    assert len(traces) == len(recs) == len(prior.priors) == games
    cols: Dict[str, list] = {k: [] for k in (
        "obj", "in_pool", "is_new", "outcome", "n_questions", "p_unknown_max",
        "fire_turn", "k_before", "prior", "n_retractions", "n_repeat_questions",
        "n_live_final")}
    for g, (r, t) in enumerate(zip(recs, traces)):
        cols["obj"].append(stream[g])
        cols["in_pool"].append(int(r.in_pool))
        cols["is_new"].append(int(r.is_new))
        cols["outcome"].append(OUTCOME_CODE[r.outcome])
        cols["n_questions"].append(r.n_questions)
        cols["p_unknown_max"].append(r.p_unknown_max)
        cols["fire_turn"].append(-1 if r.fire_turn is None else r.fire_turn)
        cols["k_before"].append(r.k_before)
        cols["prior"].append(prior.priors[g])
        cols["n_retractions"].append(r.n_retractions)
        cols["n_repeat_questions"].append(t["n_repeat_questions"])
        cols["n_live_final"].append(t["n_live_final"])
    sc = scorecard(recs, confidence=CONFIDENCE)
    sc["species"] = max(stream) + 1
    sc["admitted"] = sum(1 for r in recs if r.expert_called and not r.in_pool)
    meta = {"species": max(stream) + 1, "n_generated": max(0, max(stream) + 1 - N_REAL),
            "online_fits": prior.fits,
            "census": known_census(objects, mtx, recs)}
    return {"meta": meta, "games": cols}, sc


def play_job(args) -> Tuple[str, dict, float]:
    job, matrix_mode, outdir = args
    path = os.path.join(outdir, "jobs", job["job_id"] + ".json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        return job["job_id"], rec["scorecard"], rec["seconds"]
    t0 = time.time()
    data, sc = play(job, matrix_mode)
    secs = time.time() - t0
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(dict(data, job=job, scorecard=sc, seconds=secs), f)
    os.replace(tmp, path)
    return job["job_id"], sc, secs


# --------------------------------------------------------------------------- #
# Provenance and run                                                           #
# --------------------------------------------------------------------------- #

def check_prereg(tag: str) -> dict:
    """The tag must exist, be an ancestor of HEAD, and the pre-registration must
    be unchanged since it."""
    import subprocess
    tag_commit = _git("rev-list", "-n", "1", tag)
    subprocess.run(["git", "merge-base", "--is-ancestor", tag_commit, "HEAD"], check=True)
    if _git("diff", tag_commit, "HEAD", "--", PREREG):
        raise SystemExit("{} changed since {}: a registered design is frozen".format(PREREG, tag))
    return {"prereg_tag": tag, "prereg_commit": tag_commit}


def _segment(outdir: str, prov: dict, workers: int) -> None:
    jobdir = os.path.join(outdir, "jobs")
    done = sorted(f[:-len(".json.gz")] for f in os.listdir(jobdir) if f.endswith(".json.gz"))
    path = os.path.join(outdir, "segments.json")
    segs = json.load(open(path)) if os.path.exists(path) else []
    segs.append({"started": _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                 "provenance": prov, "workers": workers,
                 "blas": {v: os.environ.get(v) for v in BLAS_VARS},
                 "jobs_done_at_start": done})
    with open(path, "w") as f:
        json.dump(segs, f, indent=1)


def _census_args(jobs, games):
    return sorted({(j["sigma"], j["stream_seed"], j.get("games", games)) for j in jobs})


def _stream_census_job(args):
    sg, ss, games = args
    return "s{:02d}_ss{}".format(_sid(sg), ss), stream_census(sg, ss, games)


def run(outdir: str, jobs: List[dict], matrix_mode: str, config: dict,
        workers: int) -> str:
    os.makedirs(os.path.join(outdir, "jobs"), exist_ok=True)
    cfg_path = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_path):
        config = json.load(open(cfg_path))      # the first start's config is the run's
    else:
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    _segment(outdir, provenance(), workers)
    with Pool(workers) as pool:
        streams = dict(pool.map(_stream_census_job, _census_args(jobs, GAMES)))
        with open(os.path.join(outdir, "stream_census.json"), "w") as f:
            json.dump(streams, f, indent=1, sort_keys=True)
        results = {}
        t0 = time.time()
        for i, (jid, sc, secs) in enumerate(pool.imap_unordered(
                play_job, [(j, matrix_mode, outdir) for j in jobs]), 1):
            results[jid] = {"scorecard": sc, "seconds": secs}
            if i % 25 == 0 or i == len(jobs):
                print("  {}/{} jobs  ({:.0f} min elapsed)".format(
                    i, len(jobs), (time.time() - t0) / 60), flush=True)
    segments = json.load(open(os.path.join(outdir, "segments.json")))
    summary = {"schema": "twenty_questions_stress_st3_v1", "config": config,
               "segments": segments, "stream_census": streams,
               "jobs": [dict(j, **results[j["job_id"]]) for j in jobs]}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print("wrote {}".format(outdir))
    return outdir


# --------------------------------------------------------------------------- #
# Self-test (synthetic matrix only)                                            #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    from .bed import load_hypotheses
    from .openworld import py_p_new

    # The stream is seeded, numbered by first appearance, and has PY growth.
    a = py_stream(0.5, THETA, 3000, 11)
    assert a == py_stream(0.5, THETA, 3000, 11), "stream not deterministic"
    assert all(s <= max(a[:i], default=-1) + 1 for i, s in enumerate(a)), \
        "species not numbered in order of first appearance"
    ks = [max(py_stream(0.0, THETA, 5000, s)) + 1 for s in range(1, 21)]
    expect = sum(THETA / (THETA + g) for g in range(5000))
    assert abs(sum(ks) / len(ks) - expect) < 0.1 * expect, (sum(ks) / len(ks), expect)
    k75 = [max(py_stream(0.75, THETA, 5000, s)) + 1 for s in range(1, 6)]
    k25 = [max(py_stream(0.25, THETA, 5000, s)) + 1 for s in range(1, 6)]
    assert min(k75) > max(k25) > expect, "species count not increasing in sigma"
    print("[self-test] stream OK (seeded, first-appearance ids, sigma=0 mean K {:.0f} "
          "vs {:.0f} expected; sigma=0.75 K {}-{})".format(
              sum(ks) / len(ks), expect, min(k75), max(k75)))

    # Species 0..157 are the real objects in a seeded order; then generated rows.
    objs = load_hypotheses()
    o1, m1, s1 = build(0.75, 1, 2000, "synthetic")
    o2, m2, s2 = build(0.75, 1, 2000, "synthetic")
    assert [h.name for h in o1] == [h.name for h in o2] and s1 == s2
    assert sorted(h.name for h in o1[:N_REAL]) == sorted(h.name for h in objs)
    assert [h.name for h in o1[:N_REAL]] != [h.name for h in objs], "real order not shuffled"
    assert len(o1) == max(s1) + 1 and all(h.name.startswith("tail_") for h in o1[N_REAL:])
    assert all(m1[aid][h.name] == m2[aid][h.name] for aid in m1 for h in o1)
    o3, _, _ = build(0.75, 2, 2000, "synthetic")
    assert [h.name for h in o3[:N_REAL]] != [h.name for h in o1[:N_REAL]]
    print("[self-test] objects OK (real objects first in a seeded order, "
          "{} generated, deterministic)".format(len(o1) - N_REAL))

    # Priors: true/fixed are the PY formula; online refits on the loop's history
    # only, at multiples of REFIT_EVERY.
    class R:
        def __init__(self, new):
            self.expert_called, self.in_pool = new, not new
    p = Prior("online", 0.5, THETA)
    hist = [R(i % 3 == 0) for i in range(350)]
    vals = [p(g, sum(1 for r in hist[:g] if r.expert_called), hist[:g]) for g in range(350)]
    assert [f[0] for f in p.fits] == [100, 200, 300], p.fits
    assert abs(vals[50] - py_p_new(17, 50, FIXED_SIGMA, FIXED_THETA)) < 1e-12
    t = Prior("true", 0.5, THETA)
    assert abs(t(40, 10, []) - py_p_new(10, 40, 0.5, THETA)) < 1e-12
    print("[self-test] priors OK (online fits at {}, sigma_hat {:.2f})".format(
        [f[0] for f in p.fits], p.fits[-1][1]))

    # The numpy path plays the same games as the Python reference (same tie rule),
    # and the planner_factory hook is a no-op on everything but speed.
    for bk, nr in (("soft", False), ("hard", False), ("modeswitch_guard", True)):
        job = {"job_id": "t", "sigma": 0.5, "theta": THETA, "prior": "online",
               "flip": 0.05, "stream_seed": 1, "noise_seed": 1, "arm": "x",
               "backend": bk, "no_repeat": nr, "oracle_seed": 9, "games": 600}
        fast, _ = play(job, "synthetic")
        ref, _ = play(job, "synthetic", python_planner=True)
        keys = ("outcome", "n_questions", "fire_turn", "k_before")
        same = sum(all(fast["games"][k][g] == ref["games"][k][g] for k in keys)
                   for g in range(600))
        assert same == 600, "{}: numpy path differs from Python in {} games".format(
            bk, 600 - same)
        diff = max(abs(x - y) for x, y in zip(fast["games"]["p_unknown_max"],
                                               ref["games"]["p_unknown_max"]))
        assert diff < 1e-9, diff
        print("  parity {:16s} no_repeat={}: 600/600 games identical, "
              "max |d p_unknown_max| {:.1e}".format(bk, nr, diff))
    print("[self-test] ST-3 runner OK (PYTHONHASHSEED={})".format(
        os.environ.get("PYTHONHASHSEED")))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    from .stress_harness import require_hash_seed
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--stream-census", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--prereg-tag", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--blas-threads", type=int, default=1)
    args = ap.parse_args()
    pin_blas(args.blas_threads)
    hs = require_hash_seed()

    if args.plan:
        jobs = jobs_for()
        by: Dict[Tuple[str, str], int] = {}
        for j in jobs:
            by[(j["arm"], j["backend"])] = by.get((j["arm"], j["backend"]), 0) + 1
        print("{} jobs of {} games".format(len(jobs), GAMES))
        for k in sorted(by):
            print("  {:8s} {:16s} {}".format(k[0], k[1], by[k]))
        return

    if args.stream_census:
        out = {"s{:02d}_ss{}".format(_sid(sg), ss): stream_census(sg, ss)
               for sg in SIGMAS for ss in STREAM_SEEDS}
        for k, v in out.items():
            print("  {}: species {:5d} (half {:5d})  true p_new halves {:.3f} -> {:.3f}  "
                  "games with prior > 0.15: {}".format(
                      k, v["species"], v["species_at_half"], v["true_pnew_first_half"],
                      v["true_pnew_second_half"], v["games_prior_above_0.15"]))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"theta": THETA, "games": GAMES, "window": WINDOW,
                           "streams": out, "provenance": provenance()}, f, indent=1)
            print("wrote {}".format(args.out))
        return

    if args.self_test:
        self_test()
        return

    if args.smoke:
        if not args.outdir:
            raise SystemExit("--smoke needs --outdir (smoke output is not a result)")
        jobs = jobs_for(sigmas=[0.0, 0.75], priors=["true", "online"], flips=[0.0, 0.05],
                        stream_seeds=[1], noise_seeds=[1])
        config = {"mode": "smoke", "matrix": "synthetic", "games": GAMES,
                  "provenance": provenance()}
        run(args.outdir, jobs, "synthetic", config, args.workers)
        return

    if args.prereg_tag is None:
        raise SystemExit("--run needs --prereg-tag")
    if hs != "0" or sys.version_info[:2] != (3, 9):
        raise SystemExit("--run is registered under PYTHONHASHSEED=0 and Python 3.9 (LLM_RL)")
    reg = check_prereg(args.prereg_tag)
    jobs = jobs_for()
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("llm_rl_scripts/twenty_questions/jev/results",
                                         "stress_st3_{}".format(stamp))
    config = {"mode": "run", "matrix": os.path.basename(MATRIX), "sigmas": SIGMAS,
              "theta": THETA, "priors": PRIORS,
              "fixed": [FIXED_SIGMA, FIXED_THETA], "refit_every": REFIT_EVERY,
              "flips": FLIPS, "stream_seeds": STREAM_SEEDS, "noise_seeds": NOISE_SEEDS,
              "arms": ARMS, "games": GAMES, "window": WINDOW, "rho_real": RHO_REAL,
              "budget": BUDGET, "confidence": CONFIDENCE, "eps": EPS, "tolerance_K": K,
              "registration": reg, "provenance": provenance(), "created": stamp}
    run(outdir, jobs, "real", config, args.workers)


if __name__ == "__main__":
    main()
