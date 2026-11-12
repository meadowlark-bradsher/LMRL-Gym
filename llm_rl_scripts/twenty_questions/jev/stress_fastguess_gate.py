"""Gates for the closed-form guess EIG (``FastPlanner.eig_guesses``), a speed-only
change made mid-run to ST-1. See ``STRESS_ST1_BUILD.md`` (Mid-run change).

    # replay of finished ST-1 grid jobs + old-vs-new A/B on cells with no finished job:
    OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH=. python -m \\
        llm_rl_scripts.twenty_questions.jev.stress_fastguess_gate --gate OUTDIR [--workers 6]
    # closed form vs dense rows on 3000 random posteriors (edge cases included):
    ... stress_fastguess_gate --unit OUTDIR
    # per-game timing on one off-grid pool per N (indicative; load-dependent):
    ... stress_fastguess_gate --probe OUTDIR [--dense]

replay: every r1 job of each level present in the grid store (all 7 flip x noise-seed
        combinations) plus the r2 noiseless jobs, replayed with the closed form; the
        per-game columns and the scorecard must equal the stored record exactly.
ab:     19 off-grid N = 5000 cells (twins, which has the most equal-mass ties; hard,
        modeswitch and both no_repeat arms at every level), 220 games each at the
        0.091 prior, played with ``rank`` (closed form) and ``rank_dense`` (the code
        the grid's first segment ran); the games must be identical.

Each task runs in a fresh process (maxtasksperchild=1) and the ``rank_dense`` swap
is undone after each old-side task, so no task can inherit another's code path.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from multiprocessing import Pool

GRID = "llm_rl_scripts/twenty_questions/jev/results/stress_st1_20261111/jobs"


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True)


def _replay(args):
    path, out = args
    from . import stress_fastplan as F
    from . import stress_st1 as S
    assert F.FastPlanner.rank is not F.FastPlanner.rank_dense
    rec = json.load(gzip.open(path, "rt"))
    jid, sc, secs = S.play_job((rec["job"], "real", os.path.join(out, "replay")))
    new = json.load(gzip.open(os.path.join(out, "replay", "jobs", jid + ".json.gz"), "rt"))
    return ("replay", jid, _canon(new["games"]) == _canon(rec["games"]),
            _canon(new["scorecard"]) == _canon(rec["scorecard"]), rec["seconds"], secs)


def _ab(args):
    job, variant, out = args
    from . import stress_fastplan as F
    from . import stress_st1 as S
    orig = F.FastPlanner.rank
    if variant == "old":
        F.FastPlanner.rank = F.FastPlanner.rank_dense
    try:
        jid, sc, secs = S.play_job((job, "real", os.path.join(out, "ab_" + variant)))
    finally:
        F.FastPlanner.rank = orig
    return ("ab_" + variant, jid, secs)


def ab_jobs():
    from . import stress_st1 as S
    lv = dict(S.LEVELS)
    cells = [("twins", "soft", False, 0.05), ("twins", "hard", False, 0.0),
             ("twins", "modeswitch", False, 0.0)]
    for lvl in ("twins", "crowded", "real", "floor"):
        for b, nr in (("hard", False), ("modeswitch", False), ("hard", True),
                      ("modeswitch", True)):
            cells.append((lvl, b, nr, 0.05))
    jobs = []
    for i, (lvl, b, nr, flip) in enumerate(cells):
        seed = 9_300_000 + 1000 * i
        jobs.append({"pool_id": "ab", "n": 5000, "level": lvl, "rho": lv[lvl],
                     "replicate": 0, "pool_seed": seed, "novel_seed": seed + 50,
                     "order_seed": seed + 7,
                     "job_id": "ab_{}_{}_{}_f{:g}".format(lvl, b, int(nr), flip),
                     "flip": flip, "noise_seed": 1, "arm": "x", "backend": b,
                     "no_repeat": nr, "oracle_seed": seed * 10 + 1,
                     "n_in_pool": 200, "n_novel": 20})
    return jobs


def gate(out: str, workers: int) -> dict:
    for d in ("replay", "ab_old", "ab_new"):
        os.makedirs(os.path.join(out, d, "jobs"), exist_ok=True)
    jobs = ab_jobs()
    paths = sorted(p for p in glob.glob(GRID + "/*.json.gz")
                   if "_r1_" in p or ("_r2_" in p and "_f0_" in p))
    tasks_ab = [(j, v, out) for v in ("old", "new") for j in jobs]
    res = []
    with Pool(workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_ab, tasks_ab):
            res.append(r); print(r, flush=True)
        for r in pool.imap_unordered(_replay, [(p, out) for p in paths]):
            res.append(r); print(r, flush=True)
    ab_ok = []
    for j in jobs:
        a = json.load(gzip.open(os.path.join(out, "ab_old", "jobs", j["job_id"] + ".json.gz"), "rt"))
        b = json.load(gzip.open(os.path.join(out, "ab_new", "jobs", j["job_id"] + ".json.gz"), "rt"))
        ab_ok.append((j["job_id"], len(a["games"]["outcome"]),
                      _canon(a["games"]) == _canon(b["games"]),
                      _canon(a["scorecard"]) == _canon(b["scorecard"])))
    rp = [r for r in res if r[0] == "replay"]
    secs = {r[1] + ":" + r[0]: r[2] for r in res if r[0].startswith("ab")}
    summary = {
        "replay_jobs": len(rp),
        "replay_games_identical": sum(r[2] for r in rp),
        "replay_scorecard_identical": sum(r[3] for r in rp),
        "replay": [{"job_id": r[1], "games_identical": r[2], "scorecard_identical": r[3],
                    "seconds_stored": r[4], "seconds_closed_form": r[5]} for r in rp],
        "ab_cells": len(ab_ok), "ab_games": sum(x[1] for x in ab_ok),
        "ab_games_identical_cells": sum(x[2] for x in ab_ok),
        "ab_scorecard_identical_cells": sum(x[3] for x in ab_ok),
        "ab": [{"job_id": x[0], "games": x[1], "games_identical": x[2],
                "scorecard_identical": x[3],
                "seconds_dense": secs.get(x[0] + ":ab_old"),
                "seconds_closed_form": secs.get(x[0] + ":ab_new")} for x in ab_ok],
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "workers": workers,
    }
    summary["pass"] = (summary["replay_games_identical"] == len(rp)
                       and summary["replay_scorecard_identical"] == len(rp)
                       and summary["ab_games_identical_cells"] == len(ab_ok))
    with open(os.path.join(out, "gate.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("GATE", "PASS" if summary["pass"] else "FAIL")
    return summary


def probe(out: str, dense: bool = False) -> list:
    """Seconds per game, 44 games (40 + 4: the 0.091 prior) per cell, one off-grid
    real-level pool per N, flip 0.05. Includes the pool build."""
    from . import stress_fastplan as F
    from . import stress_st1 as S
    if dense:
        F.FastPlanner.rank = F.FastPlanner.rank_dense
    os.makedirs(os.path.join(out, "jobs"), exist_ok=True)
    cells = [(5000, "soft", False), (5000, "hard", False), (5000, "modeswitch", False),
             (5000, "hard", True), (1000, "soft", False), (1000, "hard", False),
             (1000, "modeswitch", False)]
    res = []
    for n, b, nr in cells:
        seed = 9_100_000 + n
        job = {"pool_id": "probe", "n": n, "level": "real", "rho": S.RHO_REAL,
               "replicate": 0, "pool_seed": seed, "novel_seed": seed + 50,
               "order_seed": seed + 7,
               "job_id": "probe_N{}_{}_{}_{}".format(n, b, int(nr), "dense" if dense else "cf"),
               "flip": 0.05, "noise_seed": 1, "arm": "x", "backend": b, "no_repeat": nr,
               "oracle_seed": seed * 10 + 1, "n_in_pool": 40, "n_novel": 4}
        _, sc, secs = S.play_job((job, "real", out))
        r = {"n": n, "backend": b, "no_repeat": nr, "games": sc["games"], "seconds": secs,
             "s_per_game": secs / sc["games"], "mean_questions": sc["mean_questions"]}
        res.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(out, "probe_{}.json".format("dense" if dense else "closed_form")), "w") as f:
        json.dump(res, f, indent=1)
    return res


def unit(out: str, trials: int = 3000) -> dict:
    """max |eig_rows(guess rows) - eig_guesses| over random posteriors on the real
    158: sparse/peaked (Dirichlet 0.05), one dominant mass, zero-mass entries,
    uniform (survivor sets), UNKNOWN present or absent, guess keys absent from the
    posterior."""
    import numpy as np
    from .bed import GUESS, Question, load_hypotheses
    from .matrix import load_matrix
    from .openworld import UNKNOWN
    from .stress_fastplan import FastPlanner
    from .stress_st1 import EPS, MATRIX
    hyps = load_hypotheses()
    fp = FastPlanner(hyps, load_matrix(MATRIX)["matrix"], eps=EPS)
    allnames = [h.name for h in hyps]
    rng = np.random.default_rng(0)
    worst, cases = 0.0, 0
    for t in range(trials):
        names = list(rng.choice(allnames, size=rng.integers(2, len(allnames) + 1),
                                replace=False))
        if rng.random() < 0.8:
            names.append(UNKNOWN)
        kind = t % 5
        p = rng.dirichlet(np.ones(len(names)) * (0.05 if kind == 1 else 1.0))
        if kind == 2:
            p[rng.integers(len(p))] += 50
            p /= p.sum()
        if kind == 3:
            p[rng.random(len(p)) < 0.3] = 0
            p = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
        if kind == 4:
            p = np.ones(len(p)) / len(p)
        keys = list(rng.choice(allnames, size=40, replace=False))
        qs = [Question(GUESS, k, "x") for k in keys]
        d = fp.eig_rows(fp.rows(qs, names), p) - fp.eig_guesses(keys, names, p)
        worst = max(worst, float(np.max(np.abs(d))))
        cases += len(keys)
    res = {"trials": trials, "guesses": cases, "max_abs_diff": worst}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "unit.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", metavar="OUTDIR")
    g.add_argument("--probe", metavar="OUTDIR")
    g.add_argument("--unit", metavar="OUTDIR")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dense", action="store_true", help="--probe the pre-change ranking")
    a = ap.parse_args()
    if a.gate:
        gate(a.gate, a.workers)
    elif a.unit:
        unit(a.unit)
    else:
        probe(a.probe, a.dense)


if __name__ == "__main__":
    main()
