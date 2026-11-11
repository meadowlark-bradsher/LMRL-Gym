"""Regression gate: changes to the open-world loop must not move any committed result.

    # both gates (~10 min on 12 cores); run under the LLM_RL env (Python 3.9):
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_regression

    # one gate only:
    ... stress_regression --committed-only
    ... stress_regression --ab-only --ab-ref origin/main

Two gates, both offline:

* **committed** -- the committed record must reproduce exactly: every game
  record of the §3.11 clean arm (``results/openworld_20261108_161009/records.json``)
  and every per-seed metric of every cell of the retraction grid
  (``results/openworld_retraction_20261108_213525``), all five backends. Needs
  Python 3.9 (the ``LLM_RL`` env); 3.12's compensated float ``sum`` moves
  ``soft`` by an ulp. The survivor-set backends iterate in ``known_hyps`` order,
  so the result must not depend on ``PYTHONHASHSEED``; run the gate under two
  seeds to check that. (Before that fix their committed values were one draw
  over Python's string-hash order; the grid was redrawn under the fixed order.)
  ``--committed-soft-only`` restricts the grid to the ``soft`` cells.
* **ab** -- every cell (clean arm plus all 27 retraction cells, every noise
  seed) is played twice, by the code at ``--ab-ref`` and by the working tree,
  under the same ``PYTHONHASHSEED``, and every game record must match. Against
  a ref older than the fixed-order backends it fails on the survivor-set cells
  by design.

Exit status is non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from typing import Dict, List, Tuple

from .analyze_openworld import auc
from .bed import load_hypotheses
from .matrix import load_matrix
from .openworld import GameRecord, fit_pitman_yor, make_stream, run_accumulation

RESULTS = "llm_rl_scripts/twenty_questions/jev/results"
CLEAN_DIR = os.path.join(RESULTS, "openworld_20261108_161009")
RETRACTION_DIR = os.path.join(RESULTS, "openworld_retraction_20261108_213525")
MATRIX = "llm_rl_scripts/twenty_questions/jev/matrix_jev.json"


def _setup(games: int, zipf_s: float, seed: int):
    objs = load_hypotheses()
    matrix = load_matrix(MATRIX)["matrix"]
    stream, index_rank = make_stream(objs, games, zipf_s, seed)
    seen: set = set()
    new_flags: List[bool] = []
    for idx in stream:
        new_flags.append(objs[idx].name not in seen)
        seen.add(objs[idx].name)
    sigma, theta, _ = fit_pitman_yor(new_flags)
    return objs, matrix, stream, index_rank, sigma, theta


def _retraction_metrics(records: List[GameRecord]) -> Dict[str, float]:
    # Identical to eval_openworld.run_retraction_grid's metrics().
    in_pool = [r for r in records if r.in_pool]
    ipf = sum(1 for r in in_pool if r.outcome != "won")
    labels = [0 if r.in_pool else 1 for r in records]
    scores = [r.p_unknown_max for r in records]
    return {
        "in_pool": len(in_pool),
        "in_pool_fail": ipf,
        "in_pool_fail_rate": ipf / len(in_pool) if in_pool else float("nan"),
        "auc": auc(scores, labels),
        "false_positive_calls": sum(1 for r in records if r.false_positive),
        "won": sum(1 for r in records if r.outcome == "won"),
        "wrong": sum(1 for r in records if r.outcome == "wrong"),
        "expert_calls": sum(1 for r in records if r.expert_called),
        "mean_retractions": sum(r.n_retractions for r in records) / len(records),
        "mean_questions": sum(r.n_questions for r in records) / len(records),
    }


def _jobs(soft_only: bool = False) -> List[Tuple[str, dict]]:
    """(label, run kwargs) for the clean arm and every retraction cell x seed."""
    with open(os.path.join(CLEAN_DIR, "summary.json")) as f:
        ccfg = json.load(f)["config"]
    with open(os.path.join(RETRACTION_DIR, "summary.json")) as f:
        payload = json.load(f)
    rcfg = payload["config"]
    jobs = [("clean", dict(games=ccfg["games"], zipf_s=ccfg["zipf_s"], seed=ccfg["seed"],
                           confidence=ccfg["confidence"], eps=ccfg["eps"],
                           oracle_noise=0.0, noise_seed=0, backend="soft"))]
    for ci, cell in enumerate(payload["cells"]):
        if soft_only and cell["backend"] != "soft":
            continue
        for ns in range(1, cell["n_noise_seeds"] + 1):
            kw = dict(games=rcfg["games"], zipf_s=rcfg["zipf_s"], seed=rcfg["seed"],
                      confidence=rcfg["confidence"], eps=rcfg["eps"],
                      oracle_noise=cell["oracle_noise"], noise_seed=ns,
                      backend=cell["backend"], cell_index=ci)
            knob = ""
            if cell["margin"] is not None:
                kw["margin"] = cell["margin"]
                knob = "@{:g}".format(cell["margin"])
            if cell["tolerance"] is not None:
                kw["tolerance"] = cell["tolerance"]
                knob = "@{:g}".format(cell["tolerance"])
            jobs.append(("{}{} noise={} seed={}".format(
                cell["backend"], knob, cell["oracle_noise"], ns), kw))
    return jobs


def _play(job: Tuple[str, dict]) -> Tuple[str, List[dict]]:
    label, kw = job
    kw = dict(kw)
    kw.pop("cell_index", None)
    objs, matrix, stream, index_rank, sigma, theta = _setup(
        kw.pop("games"), kw.pop("zipf_s"), kw.pop("seed"))
    records = run_accumulation(objs, matrix, stream, sigma, theta, **kw)
    rank_of = {h.name: index_rank[i] for i, h in enumerate(objs)}
    for r in records:
        r.rank = rank_of[r.true_name]
    return label, [dataclasses.asdict(r) for r in records]


def gate_committed(workers: int, soft_only: bool = False) -> List[str]:
    if sys.version_info[:2] != (3, 9):
        print("  warning: Python {}.{} -- the committed runs were made under 3.9 "
              "(LLM_RL env); soft will drift by an ulp".format(*sys.version_info[:2]))
    with open(os.path.join(CLEAN_DIR, "records.json")) as f:
        clean_committed = json.load(f)
    with open(os.path.join(RETRACTION_DIR, "summary.json")) as f:
        cells = json.load(f)["cells"]
    jobs = _jobs(soft_only=soft_only)
    with Pool(workers) as pool:
        results = pool.map(_play, jobs)
    errs: List[str] = []
    for (label, kw), (_, recs) in zip(jobs, results):
        bad: List[str] = []
        if label == "clean":
            if len(recs) != len(clean_committed):
                bad.append("{} games vs committed {}".format(len(recs), len(clean_committed)))
            for g, (a, b) in enumerate(zip(recs, clean_committed)):
                for k, v in b.items():
                    if a.get(k) != v:
                        bad.append("game {} {}: {!r} vs {!r}".format(g, k, a.get(k), v))
        else:
            committed = cells[kw["cell_index"]]["per_seed"][kw["noise_seed"] - 1]
            fresh = _retraction_metrics([GameRecord(**r) for r in recs])
            bad = ["{}: {!r} vs {!r}".format(k, fresh.get(k), v)
                   for k, v in committed.items() if fresh.get(k) != v]
        print("  {:40s} {}".format(label, "OK" if not bad else "MISMATCH"))
        errs.extend("committed {} {}".format(label, b) for b in bad[:10])
    return errs


def dump(path: str, workers: int) -> None:
    jobs = _jobs()
    with Pool(workers) as pool:
        results = pool.map(_play, jobs)
    with open(path, "w") as f:
        json.dump({"hashseed": os.environ.get("PYTHONHASHSEED"),
                   "cells": {label: recs for label, recs in results}}, f)


def gate_ab(ref: str, workers: int) -> List[str]:
    hs = os.environ.get("PYTHONHASHSEED")
    if hs is None or hs == "random":
        return ["ab: set a fixed PYTHONHASHSEED; the survivor-set backends depend on it"]
    tmp = tempfile.mkdtemp(prefix="st0_ab_")
    try:
        base = os.path.join(tmp, "base")
        os.makedirs(base)
        archive = subprocess.run(["git", "archive", ref, "llm_rl_scripts"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", base], input=archive, check=True)
        # The base ref predates this file; run the same dumper against its package.
        shutil.copy(__file__, os.path.join(
            base, "llm_rl_scripts/twenty_questions/jev/stress_regression.py"))
        outs = {}
        for tag, path in (("base", base), ("head", ".")):
            out = os.path.join(tmp, "{}.json".format(tag))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([path, "."]))
            subprocess.run([sys.executable, "-m",
                            "llm_rl_scripts.twenty_questions.jev.stress_regression",
                            "--dump", out, "--workers", str(workers)],
                           check=True, env=env)
            with open(out) as f:
                outs[tag] = json.load(f)["cells"]
        errs: List[str] = []
        for label, base_recs in outs["base"].items():
            head_recs = outs["head"].get(label)
            bad = 0
            if head_recs is None or len(head_recs) != len(base_recs):
                bad = 1
            else:
                for a, b in zip(head_recs, base_recs):
                    bad += any(a.get(k) != v for k, v in b.items())
            print("  {:40s} {}".format(label, "OK" if not bad else
                                       "MISMATCH ({} games)".format(bad)))
            if bad:
                errs.append("ab {}: {} games differ".format(label, bad))
        return errs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed-only", action="store_true")
    ap.add_argument("--committed-soft-only", action="store_true")
    ap.add_argument("--ab-only", action="store_true")
    ap.add_argument("--ab-ref", default="origin/main")
    ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    if args.dump:
        dump(args.dump, args.workers)
        return
    errs: List[str] = []
    if not args.ab_only:
        print("[gate committed] clean arm + {} cells vs the committed record, "
              "PYTHONHASHSEED={}".format("soft" if args.committed_soft_only else "all",
                                         os.environ.get("PYTHONHASHSEED")))
        errs += gate_committed(args.workers, soft_only=args.committed_soft_only)
    if not args.committed_only:
        print("[gate ab] every cell, {} vs working tree, PYTHONHASHSEED={}".format(
            args.ab_ref, os.environ.get("PYTHONHASHSEED")))
        errs += gate_ab(args.ab_ref, args.workers)
    for e in errs[:40]:
        print("  " + e)
    print("REGRESSION {}".format("PASS" if not errs else "FAIL ({} problems)".format(len(errs))))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
