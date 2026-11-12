"""Gates for the ``modeswitch`` budget guard (``modeswitch_guard``). The criteria are
fixed in ``STRESS_REINSTATE_GUARD.md`` before any of this is run.

    PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m \\
        llm_rl_scripts.twenty_questions.jev.stress_guard_gate --out OUTDIR [--workers 6]

Resumable: finished guarded jobs are skipped.

grid        every ``modeswitch`` job of the ST-1 store (both arms, all 714), replayed
            whole under ``modeswitch_guard`` with the stored job seeds. Whole jobs,
            because the noisy answerer draws from one random stream across a job's
            games; the guard never changes which questions are asked, so the stream
            is consumed identically.
compare     per game, guarded vs stored: every column must be equal except in games
            the guard is allowed to change -- stored ``wrong`` at the budget with one
            live known -- and those may change only to ``expert`` with an empty live
            set (``p_unknown_max`` may rise). Guarded ST-1 cell metrics follow.
trace       flip-0 jobs only (exact answerer, so single games replay alone): every
            stored single-live ``wrong`` game is replayed under both backends; the
            question/answer sequences must match, and the outcome must change exactly
            when the unguarded game reinstated on its final answer.
retraction  the published retraction grid (stream mode) with ``modeswitch`` and
            ``modeswitch_guard`` at K in {1, 3}: every per-seed metric of the guarded
            cells must equal the unguarded cells, which must equal the committed run.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
from multiprocessing import Pool

RUN = "llm_rl_scripts/twenty_questions/jev/results/stress_st1_20261111"
RETRACTION = ("llm_rl_scripts/twenty_questions/jev/results/"
              "openworld_retraction_20261108_213525/summary.json")
WON, EXPERT, WRONG = 0, 1, 2
ALLOWED = {"outcome", "n_live_final", "p_unknown_max"}


def _load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _guard_job(args):
    path, out = args
    from . import stress_st1 as S
    rec = _load(path)
    job = dict(rec["job"], backend="modeswitch_guard")
    jid, _, secs = S.play_job((job, "real", out))
    return jid, secs


def grid(out: str, workers: int) -> None:
    os.makedirs(os.path.join(out, "jobs"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RUN, "jobs", "*_modeswitch.json.gz")),
                   key=lambda p: -_load(p)["seconds"])
    with Pool(workers) as pool:
        for i, (jid, secs) in enumerate(pool.imap_unordered(
                _guard_job, [(p, out) for p in paths]), 1):
            if i % 25 == 0 or i == len(paths):
                print("  {}/{} guarded jobs".format(i, len(paths)), flush=True)


def compare(out: str) -> dict:
    from .analyze_st1 import cell_metrics
    census = json.load(open(os.path.join(RUN, "summary.json")))["census"]
    acc = {"unguarded": {}, "guarded": {}}
    pools = {}
    tot = dict(jobs=0, games=0, changed=0, changed_in_pool=0, changed_novel=0,
               disallowed=0, n158_changed=0, candidates=0)
    for path in sorted(glob.glob(os.path.join(RUN, "jobs", "*_modeswitch.json.gz"))):
        old = _load(path)
        new = _load(os.path.join(out, "jobs", os.path.basename(path)))
        j = old["job"]
        go, gn = old["games"], new["games"]
        tot["jobs"] += 1
        for i in range(len(go["outcome"])):
            tot["games"] += 1
            cand = (go["outcome"][i] == WRONG and go["n_live_final"][i] == 1
                    and go["n_questions"][i] == 20)
            tot["candidates"] += cand
            diff = {k for k in go if go[k][i] != gn[k][i]}
            if not diff:
                continue
            ok = (cand and diff <= ALLOWED and gn["outcome"][i] == EXPERT
                  and gn["n_live_final"][i] == 0)
            tot["disallowed"] += not ok
            tot["changed"] += 1
            tot["changed_in_pool" if go["in_pool"][i] else "changed_novel"] += 1
            tot["n158_changed"] += j["n"] == 158
        key = "{}|{}|{}|{}|{}".format(j["n"], j["level"], j["flip"], j["arm"], "modeswitch")
        for name, cols in (("unguarded", go), ("guarded", gn)):
            d = acc[name].setdefault(key, {})
            for k, v in cols.items():
                d.setdefault(k, []).extend(v)
        pools.setdefault(key, set()).add(j["pool_id"])
    cells = {}
    for key in sorted(pools):
        cs = [census[p] for p in sorted(pools[key])]
        cells[key] = {name: {m: cell_metrics(acc[name][key], cs)[m]
                             for m in ("won", "safe", "wrong", "auc", "fail")}
                      for name in acc}
    res = {"totals": tot, "cells": cells}
    with open(os.path.join(out, "compare.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("compare", json.dumps(tot))
    return res


def _trace_job(path):
    from . import stress_st1 as S
    from .bed import matrix_oracle
    from .openworld import run_openworld_episode
    from .stress_fastplan import FastPlanner
    rec = _load(path)
    job, g = rec["job"], rec["games"]
    idx = [i for i, o in enumerate(g["outcome"]) if o == WRONG and g["n_live_final"][i] == 1]
    r = dict(games=len(idx), reproduced=0, steps_identical=0, final_revive=0,
             changed=0, changed_iff_final_revive=0)
    if not idx:
        return r
    pool, mtx, novel = S.build(job, "real")
    rng = random.Random(job["order_seed"])
    n_in = job.get("n_in_pool", S.N_IN_POOL)
    secrets = [rng.choice(pool) for _ in range(n_in)] + list(novel)
    rng.shuffle(secrets)
    fp = FastPlanner(pool, mtx, eps=S.EPS)
    oracle = matrix_oracle(mtx)
    for i in idx:
        eps_ = {}
        for b in ("modeswitch", "modeswitch_guard"):
            revs = []
            ep = run_openworld_episode(
                secrets[i], pool, mtx, len(novel) / len(secrets), oracle,
                max_questions=S.BUDGET, confidence=S.CONFIDENCE, eps=S.EPS, backend=b,
                tolerance=S.K, no_repeat=job["no_repeat"], planner=fp,
                observer=lambda t, q, a, bk, bef: revs.append(bk.n_revives))
            eps_[b] = (ep, len(revs) >= 2 and revs[-1] > revs[-2])
        (a, fa), (b2, _) = eps_["modeswitch"], eps_["modeswitch_guard"]
        r["reproduced"] += (a.outcome == "wrong" and a.n_live_final == 1
                            and a.p_unknown_max == g["p_unknown_max"][i])
        r["steps_identical"] += ([(s.qid, s.answer_yes) for s in a.steps]
                                 == [(s.qid, s.answer_yes) for s in b2.steps])
        r["final_revive"] += fa
        ch = a.outcome != b2.outcome
        r["changed"] += ch
        r["changed_iff_final_revive"] += ch == fa
    return r


def trace(out: str, workers: int) -> dict:
    paths = sorted(glob.glob(os.path.join(RUN, "jobs", "*_f0_*_modeswitch.json.gz")))
    with Pool(workers) as pool:
        rows = pool.map(_trace_job, paths)
    tot = {k: sum(r[k] for r in rows) for k in rows[0]}
    with open(os.path.join(out, "trace.json"), "w") as f:
        json.dump(tot, f, indent=1)
    print("trace", tot)
    return tot


def retraction(out: str) -> dict:
    from .eval_openworld import run_retraction_grid
    from .stress_st1 import MATRIX
    committed = json.load(open(RETRACTION))
    cfg = committed["config"]
    path = run_retraction_grid(MATRIX, cfg["games"], cfg["zipf_s"], cfg["seed"],
                               cfg["confidence"], cfg["eps"],
                               ["modeswitch", "modeswitch_guard"], cfg["levels"],
                               cfg["n_noise_seeds"], cfg["margins"], [1, 3],
                               os.path.join(out, "retraction"))
    new = json.load(open(os.path.join(path, "summary.json")))

    def rows(cells, bk):
        return {(c["tolerance"], c["oracle_noise"]): c["per_seed"]
                for c in cells if c["backend"] == bk}
    old_ms = rows(committed["cells"], "modeswitch")
    new_ms, new_g = rows(new["cells"], "modeswitch"), rows(new["cells"], "modeswitch_guard")
    keys = sorted(new_g)
    res = {"cells": len(keys),
           "guard_equals_unguarded": sum(new_g[k] == new_ms[k] for k in keys),
           "unguarded_equals_committed": sum(new_ms[k] == old_ms.get(k) for k in keys),
           "run": os.path.relpath(path)}
    with open(os.path.join(out, "retraction.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("retraction", res)
    return res


def main() -> None:
    from .stress_harness import require_hash_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if require_hash_seed() != "0" or sys.version_info[:2] != (3, 9):
        raise SystemExit("run under the LLM_RL env (Python 3.9) with PYTHONHASHSEED=0")
    os.makedirs(a.out, exist_ok=True)
    rt = retraction(a.out)
    tr = trace(a.out, a.workers)
    grid(a.out, a.workers)
    cp = compare(a.out)["totals"]
    gates = {
        "G1_unguarded_reproduces_committed_retraction": rt["unguarded_equals_committed"] == rt["cells"],
        "G1_trace_reproduces_stored": tr["reproduced"] == tr["games"],
        "G2_trace_steps_identical": tr["steps_identical"] == tr["games"],
        "G2_trace_changed_iff_final_revive": tr["changed_iff_final_revive"] == tr["games"],
        "G2_grid_only_allowed_changes": cp["disallowed"] == 0,
        "G3_retraction_guard_equals_unguarded": rt["guard_equals_unguarded"] == rt["cells"],
        "G3_n158_unchanged": cp["n158_changed"] == 0,
    }
    gates["pass"] = all(gates.values())
    with open(os.path.join(a.out, "gates.json"), "w") as f:
        json.dump(gates, f, indent=1)
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
