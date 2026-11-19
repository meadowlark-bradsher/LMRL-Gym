"""FLYW-7 runner: arm S against the two-secret OR oracle, the CX0 gate and the smoke.
Design: ``FLYW2_COMPOUND_PREREG.md``; build notes: ``FLYW2_BUILD.md``.

    # CX0 build gate (plays flip 0 on the workers, then checks it against the calibration):
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw2_run \
        --gate --workers 6 --out llm_rl_scripts/twenty_questions/jev/results/flyw2_gate_<date>

    # smoke (5 seeds at flips 0.05 and 0.10; read only by flyw2_seed_rule):
    ... flyw2_run --smoke --workers 6 --out .../flyw2_smoke_<date>

    # the registered run (flip 0 once, seeds 1..S at each noisy flip; resumable):
    ... flyw2_run --run --seeds S --workers 6 --out .../flyw2_run_<date>

A job is one block of games at one (flip, seed): the singles and novels, or a chunk of
the KK or KN pairs. Each job writes ``jobs/<job_id>.json.gz`` when it finishes, so a
stopped run resumes by rerunning the same command. Pin BLAS threads to 1 per worker and
submit through ``jobq -N <workers>``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from multiprocessing import Pool
from typing import Dict, List, Sequence

from .bed import load_hypotheses, matrix_oracle
from .flyw1_decision import keyed_uniform as flyw1_keyed_uniform
from .flyw2_compound import (P0A, P_UNKNOWN, KeyedSetOracle, all_questions, game_key,
                             holdout_novels, keyed_uniform, pairs_of, play_s)
from .openworld import run_openworld_episode
from .stress_harness import require_hash_seed

HERE = os.path.dirname(__file__)
MATRIX = os.path.join(HERE, "matrix_jev.json")
CALIB = os.path.join(HERE, "results", "flyw2_calibration_20261119", "games.json.gz")
NOISY = (0.05, 0.10)
KK_CHUNKS = 10
KN_CHUNKS = 3

_CACHE: dict = {}


def load_inputs():
    if "inputs" not in _CACHE:
        with open(MATRIX) as f:
            matrix = json.load(f)["matrix"]
        hyps = load_hypotheses()
        novels = holdout_novels(hyps)
        known = [h for h in hyps if h.name not in novels]
        nov_h = [h for h in hyps if h.name in novels]
        _CACHE["inputs"] = (matrix, hyps, known, nov_h)
    return _CACHE["inputs"]


def game_list(block: str, chunk: int):
    """(kind, truths) for one block, in the calibration's order."""
    _, _, known, nov_h = load_inputs()
    if block == "singles":
        return [("single", [h]) for h in known] + [("novel", [h]) for h in nov_h]
    if block == "KK":
        pairs = [list(p) for p in pairs_of(known)]
        return [("KK", p) for i, p in enumerate(pairs) if i % KK_CHUNKS == chunk]
    if block == "KN":
        pairs = [[a, b] for a in known for b in nov_h]
        return [("KN", p) for i, p in enumerate(pairs) if i % KN_CHUNKS == chunk]
    raise ValueError(block)


def jobs_for(flip: float, seeds: Sequence[int]) -> List[dict]:
    out = []
    for s in seeds:
        blocks = ([("singles", 0)] + [("KK", c) for c in range(KK_CHUNKS)]
                  + [("KN", c) for c in range(KN_CHUNKS)])
        for b, c in blocks:
            out.append({"flip": flip, "seed": s, "block": b, "chunk": c,
                        "job_id": "f{}_s{}_{}_{}".format(flip, s, b, c)})
    return out


def play_block(job: dict) -> List[dict]:
    matrix, _, known, _ = load_inputs()
    oracle = KeyedSetOracle(matrix, job["flip"], job["seed"])
    out = []
    for kind, truths in game_list(job["block"], job["chunk"]):
        r = play_s(truths, known, matrix, oracle)
        r["kind"] = kind
        out.append(r)
    return out


def play_job(args) -> tuple:
    job, outdir = args
    path = os.path.join(outdir, "jobs", job["job_id"] + ".json.gz")
    if os.path.exists(path):
        return job["job_id"], 0.0, True
    t0 = time.time()
    games = play_block(job)
    secs = time.time() - t0
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump({"job": job, "secs": round(secs, 2), "games": games}, f)
    os.replace(tmp, path)
    return job["job_id"], secs, False


def run(outdir: str, jobs: List[dict], workers: int, config: dict) -> None:
    os.makedirs(os.path.join(outdir, "jobs"), exist_ok=True)
    cfg_path = os.path.join(outdir, "config.json")
    prev = []
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prev = json.load(f).get("segments", [])
    config = dict(config, segments=prev + [{"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "n_jobs": len(jobs), "workers": workers}])
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=1)
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for _jid, _secs, _skipped in pool.imap_unordered(
                play_job, [(j, outdir) for j in jobs]):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print("  {}/{} jobs ({:.1f} min)".format(done, len(jobs),
                                                          (time.time() - t0) / 60),
                      flush=True)


def load_games(outdir: str, flips: Sequence[float] = None) -> List[dict]:
    """Every game in a run directory, each tagged with its flip and seed."""
    out = []
    jdir = os.path.join(outdir, "jobs")
    for fn in sorted(os.listdir(jdir)):
        if not fn.endswith(".json.gz"):
            continue
        with gzip.open(os.path.join(jdir, fn), "rt") as f:
            d = json.load(f)
        if flips is not None and d["job"]["flip"] not in flips:
            continue
        for g in d["games"]:
            g["flip"], g["seed"] = d["job"]["flip"], d["job"]["seed"]
            out.append(g)
    return out


# --------------------------------------------------------------------------- #
# CX0 build gate                                                               #
# --------------------------------------------------------------------------- #

def gate(outdir: str, workers: int) -> dict:
    matrix, hyps, known, _ = load_inputs()
    res: Dict[str, object] = {}

    # (a) single-secret games through the set oracle are P0a's games, step for step.
    base = matrix_oracle(matrix)
    oracle = KeyedSetOracle(matrix, 0.0, 0)
    mism = []
    for h in hyps:
        r = play_s([h], known, matrix, oracle)
        ep = run_openworld_episode(h, known, matrix, P_UNKNOWN, base,
                                   max_questions=P0A["max_probes"],
                                   confidence=P0A["confidence"], eps=P0A["eps"],
                                   backend="modeswitch_guard", tolerance=P0A["tolerance"],
                                   no_repeat=True, end_rule="always")
        ref = [[s.kind[0], s.qid.split(":", 1)[1], bool(s.answer_yes)] for s in ep.steps]
        if ref != r["steps"]:
            mism.append(h.name)
    res["cx0a_mismatches"] = mism
    res["cx0a"] = not mism

    # (b) the runner's flip 0 reproduces the calibration's records exactly.
    run(outdir, jobs_for(0.0, [0]), workers, {"mode": "gate"})
    mine = {(g["kind"], tuple(g["truths"])): g for g in load_games(outdir, [0.0])}
    with gzip.open(CALIB, "rt") as f:
        calib = {(g["kind"], tuple(g["truths"])): g for g in json.load(f)}
    strip = lambda g: {k: v for k, v in g.items() if k not in ("flip", "seed")}
    diff = [k for k in calib if k not in mine or strip(mine[k]) != calib[k]]
    res["cx0b_games"] = len(calib)
    res["cx0b_extra"] = len(set(mine) - set(calib))
    res["cx0b_mismatches"] = [list(k) for k in diff[:20]]
    res["cx0b"] = not diff and len(mine) == len(calib)

    # (c) no confirmed name outside the secrets (asserted in play_s; checked here too).
    bad = [k for k, g in mine.items() if not set(g["named"]) <= set(g["truths"])]
    res["cx0c"] = not bad

    # (d) keyed noise is pure and order-free, and not FLYW-1's stream.
    u1 = keyed_uniform(3, game_key(["Cup", "Piano"]), "attr:x")
    u2 = keyed_uniform(3, game_key(["Piano", "Cup"]), "attr:x")
    o = KeyedSetOracle(matrix, 0.5, 7)
    a, b = known[0], known[1]
    qs = all_questions(known)
    o.begin([a, b]); first = [o(q) for q in qs]
    o.begin([b, a]); second = [o(q) for q in qs]
    res["cx0d"] = (u1 == u2 and first == second
                   and keyed_uniform(3, "Cup#0", "x") != flyw1_keyed_uniform(3, "Cup#0", "x"))
    res["gate_pass"] = all(res[k] for k in ("cx0a", "cx0b", "cx0c", "cx0d"))
    with open(os.path.join(outdir, "gate.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hs = require_hash_seed()
    cfg = {"python_hash_seed": hs, "settings": dict(P0A, p_unknown=P_UNKNOWN)}
    if args.gate:
        res = gate(args.out, args.workers)
        print(json.dumps(res, indent=1))
        if not res["gate_pass"]:
            raise SystemExit("CX0 gate FAILED")
        return
    if args.smoke:
        jobs = [j for f in NOISY for j in jobs_for(f, range(1, 6))]
        run(args.out, jobs, args.workers, dict(cfg, mode="smoke", seeds=5))
        print("smoke done: read it only with flyw2_seed_rule")
        return
    jobs = jobs_for(0.0, [0]) + [j for f in NOISY
                                 for j in jobs_for(f, range(1, args.seeds + 1))]
    run(args.out, jobs, args.workers, dict(cfg, mode="run", seeds=args.seeds))


if __name__ == "__main__":
    main()
