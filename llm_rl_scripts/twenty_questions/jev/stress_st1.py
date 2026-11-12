"""ST-1 runner: crowding x scale (``STRESS_CROWDING_PREREG.md``).

    # list the grid and its job counts (no games played):
    PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
        -m llm_rl_scripts.twenty_questions.jev.stress_st1 --plan

    # plumbing check on the *synthetic* matrix (no ST-1 outcome is produced):
    ... stress_st1 --smoke --outdir /tmp/st1_smoke

    # the registered run (refuses unless --prereg-tag is an ancestor of HEAD and
    # the pre-registration is unchanged since it):
    ... stress_st1 --run --prereg-tag prereg-stress-st1-<date> --soft-max-n 5000

Snapshot mode (``stress_harness.run_snapshot``) with the numpy planner
(``stress_fastplan.FastPlanner``). Every job writes ``jobs/<id>.json.gz`` (its
scorecard plus compact per-game columns), so an interrupted run resumes by
skipping finished jobs. The census of each pool is computed once
(``census/<pool>.json``). ``summary.json`` carries the config, provenance (git
commit, Python, PYTHONHASHSEED, prereg tag) and every job's scorecard;
``analyze_st1`` scores the predictions from it.

All seeds derive from fixed integers -- never ``hash()``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

RHO_REAL = 0.2345
LEVELS = [("twins", 0.0), ("crowded", 0.078), ("real", RHO_REAL), ("floor", 1.0)]
SIZES = [158, 500, 1000, 2000, 5000]
FLIPS = [0.0, 0.05, 0.10]
REPLICATES = [1, 2, 3]       # generator seeds (N > 158) / novelty+order seeds (N = 158)
NOISE_SEEDS = [1, 2, 3]
ARMS = [("primary", "soft", False), ("primary", "hard", False),
        ("primary", "modeswitch", False), ("norepeat", "hard", True),
        ("norepeat", "modeswitch", True)]
N_IN_POOL = 1000
N_NOVEL = 100
BUDGET = 20
CONFIDENCE = 0.9
EPS = 0.05
K = 3
PREREG = "llm_rl_scripts/twenty_questions/jev/STRESS_CROWDING_PREREG.md"
MATRIX = "llm_rl_scripts/twenty_questions/jev/matrix_jev.json"
OUTCOME_CODE = {"won": 0, "expert": 1, "wrong": 2, "false_confirm": 3}


# --------------------------------------------------------------------------- #
# Grid                                                                         #
# --------------------------------------------------------------------------- #

def pool_specs(sizes=SIZES, levels=LEVELS, replicates=REPLICATES) -> List[dict]:
    """One spec per pool. N = 158 is the real matrix (no level); its replicates
    vary only the novelty set and the secret order."""
    specs = []
    for n in sizes:
        lvls = [("realmatrix", None)] if n == 158 else levels
        for li, (lname, rho) in enumerate(lvls):
            for rep in replicates:
                seed = 7_000_000 + 10_000 * n + 100 * li + rep
                specs.append({"pool_id": "N{}_{}_r{}".format(n, lname, rep), "n": n,
                              "level": lname, "rho": rho, "replicate": rep,
                              "pool_seed": seed, "novel_seed": seed + 50,
                              "order_seed": seed + 7})
    return specs


def jobs_for(specs: List[dict], flips=FLIPS, noise_seeds=NOISE_SEEDS, arms=ARMS,
             soft_max_n: int = 5000) -> List[dict]:
    jobs = []
    for spec in specs:
        for flip in flips:
            for ns in ([0] if flip == 0.0 else noise_seeds):
                for arm, backend, no_repeat in arms:
                    if backend == "soft" and spec["n"] > soft_max_n:
                        continue
                    jid = "{}_f{:g}_s{}_{}_{}".format(spec["pool_id"], flip, ns, arm, backend)
                    jobs.append(dict(spec, job_id=jid, flip=flip, noise_seed=ns, arm=arm,
                                     backend=backend, no_repeat=no_repeat,
                                     oracle_seed=spec["pool_seed"] * 10 + ns))
    # Longest first, for load balance: soft at large N dominates.
    jobs.sort(key=lambda j: (-(j["n"] * (8 if j["backend"] == "soft" else 1)), j["job_id"]))
    return jobs


# --------------------------------------------------------------------------- #
# Building a pool                                                              #
# --------------------------------------------------------------------------- #

def build(spec: dict, matrix_mode: str):
    """(pool, matrix incl. novel rows, novel) for a spec; deterministic."""
    from .bed import load_hypotheses
    from .matrix import load_matrix, synthetic_matrix
    from .stress_generator import build_pool, generate_objects, merge

    objs = load_hypotheses()
    base = (synthetic_matrix(objs) if matrix_mode == "synthetic"
            else load_matrix(MATRIX)["matrix"])
    if spec["rho"] is None:
        pool, pm = list(objs), base
        novel_rho = RHO_REAL
    else:
        pool, pm = build_pool(objs, base, spec["n"], spec["rho"], spec["pool_seed"])
        novel_rho = spec["rho"]
    novel, rows = generate_objects(objs, base, spec.get("n_novel", N_NOVEL), novel_rho,
                                   spec["novel_seed"], prefix="nov")
    return pool, merge(pm, rows), novel


def census_job(args) -> Tuple[str, dict]:
    spec, matrix_mode, outdir = args
    from .stress_census import census
    path = os.path.join(outdir, "census", spec["pool_id"] + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return spec["pool_id"], json.load(f)
    pool, mtx, _ = build(spec, matrix_mode)
    c = census(mtx, [h.name for h in pool], categories={h.name: h.category for h in pool})
    c.pop("nn_hamming_hist", None)
    c.pop("nn_within_category_hist", None)
    with open(path, "w") as f:
        json.dump(c, f, sort_keys=True)
    return spec["pool_id"], c


def play_job(args) -> Tuple[str, dict, float]:
    job, matrix_mode, outdir = args
    path = os.path.join(outdir, "jobs", job["job_id"] + ".json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        return job["job_id"], rec["scorecard"], rec["seconds"]
    from .bed import matrix_oracle
    from .openworld import noisy_matrix_oracle
    from .stress_fastplan import FastPlanner
    from .stress_harness import run_snapshot, scorecard

    t0 = time.time()
    pool, mtx, novel = build(job, matrix_mode)
    oracle = (matrix_oracle(mtx) if job["flip"] == 0.0
              else noisy_matrix_oracle(mtx, job["flip"], seed=job["oracle_seed"]))
    planner = FastPlanner(pool, mtx, eps=EPS)
    n_in = job.get("n_in_pool", N_IN_POOL)
    recs = run_snapshot(pool, mtx, novel, n_in_pool=n_in, oracle=oracle,
                        backend=job["backend"], seed=job["order_seed"],
                        p_unknown_prior=len(novel) / (n_in + len(novel)),
                        max_questions=BUDGET, confidence=CONFIDENCE, eps=EPS,
                        tolerance=K, no_repeat=job["no_repeat"], planner=planner)
    sc = scorecard(recs, confidence=CONFIDENCE)
    secs = time.time() - t0
    cols = {
        "in_pool": [int(r.in_pool) for r in recs],
        "outcome": [OUTCOME_CODE[r.outcome] for r in recs],
        "n_questions": [r.n_questions for r in recs],
        "n_guesses": [r.n_guesses for r in recs],
        "n_repeat_questions": [r.n_repeat_questions for r in recs],
        "p_unknown_max": [r.p_unknown_max for r in recs],
        "n_live_final": [r.n_live_final for r in recs],
        "struck_by_flip": [None if r.truth_struck_by_flip is None else int(r.truth_struck_by_flip)
                           for r in recs],
        "emptied": [None if r.emptied_after_strike is None else int(r.emptied_after_strike)
                    for r in recs],
    }
    with gzip.open(path, "wt") as f:
        json.dump({"job": job, "scorecard": sc, "seconds": secs, "games": cols}, f)
    return job["job_id"], sc, secs


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def check_prereg(tag: str) -> dict:
    """The tag must exist, be an ancestor of HEAD, and the pre-registration must
    be unchanged since it."""
    tag_commit = _git("rev-list", "-n", "1", tag)
    subprocess.run(["git", "merge-base", "--is-ancestor", tag_commit, "HEAD"], check=True)
    diff = _git("diff", tag_commit, "HEAD", "--", PREREG)
    if diff:
        raise SystemExit("{} changed since {}: a registered design is frozen".format(PREREG, tag))
    return {"prereg_tag": tag, "prereg_commit": tag_commit}


def provenance() -> dict:
    """What a cross-machine comparison needs: the code, the interpreter, and the
    numeric stack whose summation order the tie rule guards against."""
    import platform
    import numpy as np
    import scipy
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas = "{} {}".format(blas.get("name"), blas.get("version"))
    except Exception:  # numpy < 1.25 has no mode="dicts"
        blas = "unknown"
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "git_dirty_files": [ln.strip() for ln in _git(
                "status", "--porcelain", "--untracked-files=no").splitlines()],
            "python": sys.version.split()[0], "executable": sys.executable,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "numpy": np.__version__, "scipy": scipy.__version__, "blas": blas,
            "platform": platform.platform(), "machine": platform.machine(),
            "blas_threads": {v: os.environ.get(v) for v in BLAS_VARS}}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS")


def pin_blas(threads: int) -> Dict[str, str]:
    """Set the BLAS thread count for the worker processes (spawned after this, so
    they inherit it). Unpinned, OpenBLAS starts one thread per core in *every*
    worker: 12 workers x 12 threads oversubscribe the machine, and the idle
    threads' spinning shows up as kernel time."""
    for v in BLAS_VARS:
        os.environ[v] = str(threads)
    return {v: os.environ[v] for v in BLAS_VARS}


def _segment(outdir: str, prov: dict, workers: int) -> None:
    """Append this start to ``segments.json``. ``config.json`` is written once, by
    the first start; every start (the first included) records its code commit,
    worker count, BLAS threads and the jobs already finished, so a resumed run
    says which jobs each code version produced."""
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


def run(outdir: str, specs: List[dict], jobs: List[dict], matrix_mode: str,
        config: dict, workers: int) -> str:
    os.makedirs(os.path.join(outdir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "census"), exist_ok=True)
    cfg_path = os.path.join(outdir, "config.json")
    if os.path.exists(cfg_path):
        config = json.load(open(cfg_path))      # the first start's config is the run's
    else:
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    _segment(outdir, provenance(), workers)
    with Pool(workers) as pool:
        census = dict(pool.map(census_job, [(s, matrix_mode, outdir) for s in specs]))
        print("census: {} pools".format(len(census)))
        results = {}
        t0 = time.time()
        for i, (jid, sc, secs) in enumerate(pool.imap_unordered(
                play_job, [(j, matrix_mode, outdir) for j in jobs]), 1):
            results[jid] = {"scorecard": sc, "seconds": secs}
            if i % 25 == 0 or i == len(jobs):
                print("  {}/{} jobs  ({:.0f} min elapsed)".format(
                    i, len(jobs), (time.time() - t0) / 60), flush=True)
    seg_path = os.path.join(outdir, "segments.json")
    segments = json.load(open(seg_path)) if os.path.exists(seg_path) else []
    summary = {"schema": "twenty_questions_stress_st1_v1", "config": config,
               "segments": segments,
               "pools": specs, "census": census,
               "jobs": [dict(j, **results[j["job_id"]]) for j in jobs]}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print("wrote {}".format(outdir))
    return outdir


def main() -> None:
    from .stress_harness import require_hash_seed
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--prereg-tag", default=None)
    ap.add_argument("--soft-max-n", type=int, choices=[2000, 5000], default=None,
                    help="the pre-registered runtime-cap decision; required for --run")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS threads per worker (default 1; see pin_blas)")
    args = ap.parse_args()
    pin_blas(args.blas_threads)
    hs = require_hash_seed()

    if args.plan:
        for cap in (5000, 2000):
            jobs = jobs_for(pool_specs(), soft_max_n=cap)
            print("soft-max-n {}: {} pools, {} jobs ({} soft)".format(
                cap, len(pool_specs()), len(jobs), sum(j["backend"] == "soft" for j in jobs)))
        return

    if args.smoke:
        if not args.outdir:
            raise SystemExit("--smoke needs --outdir (smoke output is not a result)")
        specs = pool_specs(sizes=[158, 300], replicates=[1])
        for s in specs:
            s.update(n_novel=10, n_in_pool=30)
        jobs = jobs_for(specs, flips=[0.0, 0.10], noise_seeds=[1])
        for j in jobs:
            j.update(n_novel=10, n_in_pool=30)
        config = {"mode": "smoke", "matrix": "synthetic", "provenance": provenance()}
        run(args.outdir, specs, jobs, "synthetic", config, args.workers)
        return

    if args.prereg_tag is None or args.soft_max_n is None:
        raise SystemExit("--run needs --prereg-tag and --soft-max-n (the runtime-cap decision)")
    if hs != "0" or sys.version_info[:2] != (3, 9):
        raise SystemExit("--run is registered under PYTHONHASHSEED=0 and Python 3.9 (LLM_RL)")
    reg = check_prereg(args.prereg_tag)
    specs = pool_specs()
    jobs = jobs_for(specs, soft_max_n=args.soft_max_n)
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("llm_rl_scripts/twenty_questions/jev/results",
                                         "stress_st1_{}".format(stamp))
    config = {"mode": "run", "matrix": os.path.basename(MATRIX), "levels": LEVELS,
              "sizes": SIZES, "flips": FLIPS, "replicates": REPLICATES,
              "noise_seeds": NOISE_SEEDS, "arms": ARMS, "n_in_pool": N_IN_POOL,
              "n_novel": N_NOVEL, "budget": BUDGET, "confidence": CONFIDENCE, "eps": EPS,
              "tolerance_K": K, "soft_max_n": args.soft_max_n, "registration": reg,
              "provenance": provenance(), "created": stamp}
    run(outdir, specs, jobs, "real", config, args.workers)


if __name__ == "__main__":
    main()
