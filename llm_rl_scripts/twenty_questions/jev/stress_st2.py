"""ST-2 runner: the look-alike tail (``STRESS_LOOKALIKE_PREREG.md``).

    # list the grid and its job counts (no games played):
    PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \
        -m llm_rl_scripts.twenty_questions.jev.stress_st2 --plan

    # mechanics self-test on the *synthetic* matrix (no ST-2 outcome is produced):
    ... stress_st2 --self-test

    # plumbing check + timing on the synthetic matrix:
    ... stress_st2 --smoke --outdir /tmp/st2_smoke

    # the registered run (refuses unless --prereg-tag is an ancestor of HEAD and
    # the pre-registration is unchanged since it):
    ... stress_st2 --run --prereg-tag prereg-stress-st2-<date> --workers 6

Accumulation mode (``openworld.run_accumulation``) on the Python planner path,
which is the path that produced §3.11: 2500 games over the 158 real objects
plus 100 look-alikes. A look-alike set is fixed by its seed: the 100 parents
and, for each look-alike, the order in which its parent's flippable cells are
flipped. The look-alike at distance d flips the first d cells of that order, so
the sets at different d share parents and are nested (common random numbers
for the d axis). Every job writes ``jobs/<id>.json.gz`` (scorecard plus compact
per-game columns), so an interrupted run resumes by skipping finished jobs.
``summary.json`` carries the config, provenance and every job's scorecard;
``analyze_st2`` scores the predictions from the per-game columns.

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

D_LEVELS = [1, 2, 3, 4, 6, 8, 12]
FLIPS = [0.0, 0.05, 0.10]
BACKENDS = ["soft", "hard", "modeswitch", "modeswitch_guard"]
G_RATES = [0.05, 0.20]
G_MAX_D = 2                  # the G arm's confusable look-alikes: d <= 2
LA_SEEDS = [1, 2, 3]
NOISE_SEEDS = [1, 2, 3]
N_LOOKALIKES = 100
GAMES = 2500
ZIPF_S = 1.1
STREAM_SEED = 7
BUDGET = 20
CONFIDENCE = 0.9
EPS = 0.05
K = 3
PREREG = "llm_rl_scripts/twenty_questions/jev/STRESS_LOOKALIKE_PREREG.md"
MATRIX = "llm_rl_scripts/twenty_questions/jev/matrix_jev.json"
OUTCOME_CODE = {"won": 0, "expert": 1, "wrong": 2, "false_confirm": 3}


# --------------------------------------------------------------------------- #
# Grid                                                                         #
# --------------------------------------------------------------------------- #

def jobs_for(d_levels=D_LEVELS, flips=FLIPS, backends=BACKENDS, g_rates=G_RATES,
             la_seeds=LA_SEEDS, noise_seeds=NOISE_SEEDS) -> List[dict]:
    """Exact arm: every d x flip x backend x look-alike seed (x noise seed under
    noise; flip 0 is deterministic, so it has one). G arm: flip 0, d <= G_MAX_D
    (beyond it no look-alike is confusable and the cell equals the exact one),
    every g, the noise seed driving the confusion draws."""
    jobs = []

    def add(arm, d, la, flip, ns, g, bk):
        jid = "{}_d{}_la{}_f{:g}_s{}_g{:g}_{}".format(arm, d, la, flip, ns, g, bk)
        jobs.append({"job_id": jid, "arm": arm, "d": d, "la_seed": la, "flip": flip,
                     "noise_seed": ns, "g": g, "backend": bk,
                     "oracle_seed": 2_000_000 + 10_000 * la + 100 * d + ns,
                     "confusion_seed": 3_000_000 + 10_000 * la + 100 * d + ns})

    for d in d_levels:
        for la in la_seeds:
            for flip in flips:
                for ns in ([0] if flip == 0.0 else noise_seeds):
                    for bk in backends:
                        add("exact", d, la, flip, ns, 0.0, bk)
            for g in g_rates:
                if d > G_MAX_D:
                    continue
                for ns in noise_seeds:
                    for bk in backends:
                        add("G", d, la, 0.0, ns, g, bk)
    # soft is ~3x slower; start it first for load balance.
    jobs.sort(key=lambda j: (j["backend"] != "soft", j["job_id"]))
    return jobs


# --------------------------------------------------------------------------- #
# Building the object set                                                      #
# --------------------------------------------------------------------------- #

def lookalike_plan(objs, matrix, la_seed: int, n: int = N_LOOKALIKES,
                   s: float = ZIPF_S, stream_seed: int = STREAM_SEED) -> List[dict]:
    """The look-alike set for one seed, independent of d. ``n`` distinct parents
    are drawn *without* replacement from the real objects, one at a time, each
    draw weighted by the remaining objects' Zipf weights *in the stream the games
    are played on* (``make_stream`` over the 158 + n objects, seed
    ``stream_seed``; the rank map depends only on the object count and the seed).
    So common objects are likelier to have a look-alike, and no object has two:
    no two look-alikes are siblings, and none can be another's exact twin. Each
    look-alike carries a seeded random order of its parent's flippable cells
    (cells at exactly 0.5 cannot cross and are excluded)."""
    from .openworld import make_stream, zipf_weights
    from .stress_generator import AIDS

    _, index_rank = make_stream(list(range(len(objs) + n)), 0, s, stream_seed)
    zw = zipf_weights(len(objs) + n, s)
    weights = [zw[index_rank[i] - 1] for i in range(len(objs))]
    rng = random.Random(1_000_000 + la_seed)
    if n > len(objs):
        raise ValueError("{} distinct parents from {} objects".format(n, len(objs)))
    remaining = list(range(len(objs)))
    plan = []
    for i in range(n):
        j = rng.choices(range(len(remaining)), weights=[weights[r] for r in remaining],
                        k=1)[0]
        parent = remaining.pop(j)
        flippable = [a for a in AIDS if matrix[a][objs[parent].name] != 0.5]
        rng.shuffle(flippable)
        plan.append({"name": "la{:03d}".format(i), "parent": objs[parent].name,
                     "parent_index": parent, "order": flippable})
    return plan


def build(job: dict, matrix_mode: str):
    """(objects, matrix incl. look-alike rows, parent_of) for a job; deterministic.
    Objects are the 158 real ones in ``load_hypotheses`` order, then la000..la099."""
    from .bed import Hypothesis, load_hypotheses
    from .matrix import load_matrix, synthetic_matrix
    from .stress_generator import merge

    objs = load_hypotheses()
    base = (synthetic_matrix(objs) if matrix_mode == "synthetic"
            else load_matrix(MATRIX)["matrix"])
    by_name = {h.name: h for h in objs}
    plan = lookalike_plan(objs, base, job["la_seed"], n=job.get("n_lookalikes", N_LOOKALIKES))
    rows, las, parent_of = {}, [], {}
    for p in plan:
        par = by_name[p["parent"]]
        row = {a: base[a][par.name] for a in base}
        for a in p["order"][:job["d"]]:
            row[a] = 1.0 - row[a]
        las.append(Hypothesis(name=p["name"], variants=(p["name"],), category=par.category))
        rows[p["name"]] = row
        parent_of[p["name"]] = par.name
    return objs + las, merge(base, rows), parent_of


def lookalike_census(objects, matrix, parent_of: Dict[str, str]) -> dict:
    """Per look-alike: binarized distance to its parent, to its nearest other
    object (real or look-alike), and whether that nearest object is the parent."""
    from .stress_oracles import hamming
    out = []
    names = [h.name for h in objects]
    for la, par in parent_of.items():
        others = [(hamming(matrix, la, n), n) for n in names if n != la]
        nn_d = min(d for d, _ in others)
        nn = sorted(n for d, n in others if d == nn_d)
        out.append({"name": la, "parent": par, "d_parent": hamming(matrix, la, par),
                    "nn_distance": nn_d, "nn_is_parent_only": nn == [par],
                    "n_at_nn": len(nn)})
    return {"lookalikes": out,
            "n_parent_not_nearest": sum(1 for r in out if r["nn_distance"] < r["d_parent"]),
            "n_ties_with_parent": sum(1 for r in out if r["nn_distance"] == r["d_parent"]
                                      and not r["nn_is_parent_only"]),
            "n_distinct_parents": len(set(parent_of.values()))}


# --------------------------------------------------------------------------- #
# Playing a job                                                                #
# --------------------------------------------------------------------------- #

class GameTrace:
    """Per-game ``observer`` for ``run_accumulation``: records the by-name guesses
    and repeated questions of each finished game (``finish`` gets the episode)."""

    def __init__(self, sink: List[dict]):
        self.sink = sink

    def __call__(self, turn, q, ans, bk, surv_before) -> None:
        pass

    def finish(self, ep) -> None:
        from .bed import GUESS
        guesses = [s.qid for s in ep.steps if s.kind == GUESS]
        names = [s.qid.split(":", 1)[1] if ":" in s.qid else s.qid for s in ep.steps
                 if s.kind == GUESS]
        others = [s.qid for s in ep.steps if s.kind != GUESS]
        self.sink.append({"guess_names": names,
                          "n_repeat_guesses": len(guesses) - len(set(guesses)),
                          "n_repeat_questions": len(others) - len(set(others)),
                          "n_live_final": ep.n_live_final})


def _guess_name(qid: str, names: set) -> str:
    """The object named by a guess qid (``bank`` qids are ``guess:<name>``)."""
    if qid in names:
        return qid
    return qid.split(":", 1)[1] if ":" in qid else qid


def make_oracle(job: dict, matrix, parent_of: Dict[str, str]):
    from .bed import matrix_oracle
    from .openworld import noisy_matrix_oracle
    from .stress_oracles import lookalike_confusion
    oracle = (matrix_oracle(matrix) if job["flip"] == 0.0
              else noisy_matrix_oracle(matrix, job["flip"], seed=job["oracle_seed"]))
    if job["g"] > 0.0:
        confusable = parent_of if job["d"] <= G_MAX_D else {}
        oracle = lookalike_confusion(oracle, confusable, job["g"], seed=job["confusion_seed"])
    return oracle


def stream_and_prior(n_objects: int, games: int):
    """The stream (object indices) and its PY (sigma, theta), fitted on the
    stream's own first-appearance flags as §3.11 did. Depends only on the object
    count, so it is the same for every job."""
    from .openworld import fit_pitman_yor, make_stream
    stream, index_rank = make_stream(list(range(n_objects)), games, ZIPF_S, STREAM_SEED)
    seen, flags = set(), []
    for i in stream:
        flags.append(i not in seen)
        seen.add(i)
    sigma, theta, _ = fit_pitman_yor(flags)
    return stream, index_rank, sigma, theta


def play(job: dict, matrix_mode: str) -> Tuple[dict, dict]:
    """(per-game columns, scorecard) for one job."""
    from .openworld import run_accumulation
    from .stress_harness import scorecard

    objects, mtx, parent_of = build(job, matrix_mode)
    games = job.get("games", GAMES)
    stream, index_rank, sigma, theta = stream_and_prior(len(objects), games)
    traces: List[dict] = []
    recs = run_accumulation(objects, mtx, stream, sigma, theta, max_questions=BUDGET,
                            confidence=CONFIDENCE, eps=EPS, backend=job["backend"],
                            tolerance=K, oracle=make_oracle(job, mtx, parent_of),
                            observer_factory=lambda known: GameTrace(traces))
    assert len(traces) == len(recs)
    names = {h.name for h in objects}
    idx = {h.name: i for i, h in enumerate(objects)}
    cols: Dict[str, list] = {k: [] for k in (
        "obj", "in_pool", "is_new", "outcome", "n_questions", "p_unknown_max",
        "fire_turn", "k_before", "n_retractions", "n_guesses", "n_repeat_guesses",
        "n_repeat_questions", "n_live_final", "parent_guessed", "n_parent_guesses",
        "self_guessed")}
    for r, t in zip(recs, traces):
        gn = [_guess_name(q, names) for q in t["guess_names"]]
        par = parent_of.get(r.true_name)
        cols["obj"].append(idx[r.true_name])
        cols["in_pool"].append(int(r.in_pool))
        cols["is_new"].append(int(r.is_new))
        cols["outcome"].append(OUTCOME_CODE[r.outcome])
        cols["n_questions"].append(r.n_questions)
        cols["p_unknown_max"].append(r.p_unknown_max)
        cols["fire_turn"].append(-1 if r.fire_turn is None else r.fire_turn)
        cols["k_before"].append(r.k_before)
        cols["n_retractions"].append(r.n_retractions)
        cols["n_guesses"].append(len(gn))
        cols["n_repeat_guesses"].append(t["n_repeat_guesses"])
        cols["n_repeat_questions"].append(t["n_repeat_questions"])
        cols["n_live_final"].append(t["n_live_final"])
        cols["parent_guessed"].append(None if par is None else int(par in gn))
        cols["n_parent_guesses"].append(None if par is None else gn.count(par))
        cols["self_guessed"].append(int(r.true_name in gn))
    sc = scorecard(recs, confidence=CONFIDENCE)
    la = [i for i, o in enumerate(cols["obj"]) if o >= 158]
    sc["lookalike_games"] = len(la)
    sc["lookalike_false_confirm"] = sum(cols["outcome"][i] == 3 for i in la)
    sc["lookalike_wrong"] = sum(cols["outcome"][i] == 2 for i in la)
    sc["admitted_lookalikes"] = sum(1 for i in la if cols["outcome"][i] == 1
                                    and not cols["in_pool"][i])
    meta = {"sigma": sigma, "theta": theta, "n_objects": len(objects),
            "parent_of": parent_of,
            "rank": [index_rank[i] for i in range(len(objects))]}
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


def census_job(args) -> Tuple[str, dict]:
    la_seed, d, matrix_mode = args
    objects, mtx, parent_of = build({"la_seed": la_seed, "d": d}, matrix_mode)
    return "la{}_d{}".format(la_seed, d), lookalike_census(objects, mtx, parent_of)


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
    sets = sorted({(j["la_seed"], j["d"]) for j in jobs})
    with Pool(workers) as pool:
        census = dict(pool.map(census_job, [(la, d, matrix_mode) for la, d in sets]))
        with open(os.path.join(outdir, "census.json"), "w") as f:
            json.dump(census, f, indent=1, sort_keys=True)
        print("census: {} look-alike sets".format(len(census)), flush=True)
        results = {}
        t0 = time.time()
        for i, (jid, sc, secs) in enumerate(pool.imap_unordered(
                play_job, [(j, matrix_mode, outdir) for j in jobs]), 1):
            results[jid] = {"scorecard": sc, "seconds": secs}
            if i % 25 == 0 or i == len(jobs):
                print("  {}/{} jobs  ({:.0f} min elapsed)".format(
                    i, len(jobs), (time.time() - t0) / 60), flush=True)
    segments = json.load(open(os.path.join(outdir, "segments.json")))
    summary = {"schema": "twenty_questions_stress_st2_v1", "config": config,
               "segments": segments, "census": {k: {kk: vv for kk, vv in v.items()
                                                    if kk != "lookalikes"}
                                                for k, v in census.items()},
               "jobs": [dict(j, **results[j["job_id"]]) for j in jobs]}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print("wrote {}".format(outdir))
    return outdir


# --------------------------------------------------------------------------- #
# Self-test (synthetic matrix only)                                            #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    from .bed import GUESS, Question, load_hypotheses, matrix_oracle
    from .matrix import synthetic_matrix
    from .stress_oracles import hamming

    objs = load_hypotheses()
    base = synthetic_matrix(objs)
    # Sets are seeded, share parents across d, and sit at exactly d.
    j = {"la_seed": 1, "d": 1}
    o1, m1, p1 = build(dict(j, d=1), "synthetic")
    o6, m6, p6 = build(dict(j, d=6), "synthetic")
    oA, mA, pA = build(dict(j, d=6), "synthetic")
    assert p1 == p6 == pA and [h.name for h in o6] == [h.name for h in oA]
    assert all(m6[a][n] == mA[a][n] for a in m6 for n in m6[a]), "build not deterministic"
    for la, par in p6.items():
        assert hamming(m6, la, par) == 6 and hamming(m1, la, par) == 1
        flipped1 = {a for a in m1 if m1[a][la] != base[a][par]}
        flipped6 = {a for a in m6 if m6[a][la] != base[a][par]}
        assert flipped1 < flipped6, "d-sets are not nested"
    assert len(o6) == 258 and all(m6[a][h.name] == base[a][h.name] for a in base for h in objs)
    o2, _, p2 = build({"la_seed": 2, "d": 1}, "synthetic")
    assert p2 != p1, "look-alike seeds give the same parents"
    # Parents follow the stream's Zipf weights: the commonest objects get most.
    from collections import Counter
    _, index_rank, _, _ = stream_and_prior(258, 10)
    plan = [lookalike_plan(objs, base, s) for s in range(1, 21)]
    assert all(len({p["parent_index"] for p in pl}) == 100 for pl in plan), \
        "parents not distinct"
    cnt = Counter(p["parent_index"] for pl in plan for p in pl)
    by_rank = sorted(range(158), key=lambda i: index_rank[i])
    top, bottom = by_rank[:20], by_rank[-20:]
    assert sum(cnt[i] for i in top) > sum(cnt[i] for i in bottom) + 100, \
        "parents not Zipf-weighted"
    print("[self-test] look-alike sets OK (seeded, nested in d, exact distance, "
          "real rows untouched, distinct Zipf-weighted parents)")

    # The stream is the same for every job and is seed 7's rank map.
    s1 = stream_and_prior(258, 300)
    s2 = stream_and_prior(258, 300)
    assert s1[0] == s2[0] and s1[2:] == s2[2:]

    # G oracle: confusion only at d <= 2, only on a guess of the secret's parent.
    la = o1[158]
    par = p1[la.name]
    gq = Question(GUESS, par, "")
    yes_g = make_oracle({"flip": 0.0, "g": 1.0, "d": 1, "oracle_seed": 1,
                         "confusion_seed": 1}, m1, p1)
    assert yes_g(gq, la) is True
    no_g = make_oracle({"flip": 0.0, "g": 1.0, "d": 3, "oracle_seed": 1,
                        "confusion_seed": 1}, m1, p1)
    assert no_g(gq, la) is False
    assert yes_g(Question(GUESS, objs[0].name if objs[0].name != par else objs[1].name, ""),
                 la) is False
    print("[self-test] G oracle OK (confusion at d <= 2 only, parent guess only)")

    # A short job of each kind plays, writes the columns and scores look-alikes.
    for bk in ("soft", "hard", "modeswitch", "modeswitch_guard"):
        for arm, g in (("exact", 0.0), ("G", 0.2)):
            job = {"job_id": "t", "arm": arm, "d": 1, "la_seed": 1, "flip": 0.0,
                   "noise_seed": 1, "g": g, "backend": bk, "oracle_seed": 5,
                   "confusion_seed": 6, "games": 400}
            data, sc = play(job, "synthetic")
            cols = data["games"]
            assert len(cols["obj"]) == 400
            assert all((o >= 158) == (pg is not None)
                       for o, pg in zip(cols["obj"], cols["parent_guessed"]))
            if g == 0.0:
                assert sc["lookalike_false_confirm"] == 0, "exact oracle false_confirm"
            print("  {:16s} {:5s}: look-alike games {:3d}  admitted {:3d}  "
                  "false_confirm {:3d}  wrong {:3d}  parent guessed {:.2f}".format(
                      bk, arm, sc["lookalike_games"], sc["admitted_lookalikes"],
                      sc["lookalike_false_confirm"], sc["lookalike_wrong"],
                      sum(x for x in cols["parent_guessed"] if x is not None)
                      / max(1, sc["lookalike_games"])))
    print("[self-test] ST-2 runner OK (PYTHONHASHSEED={})".format(
        os.environ.get("PYTHONHASHSEED")))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    from .stress_harness import require_hash_seed
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--prereg-tag", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--blas-threads", type=int, default=1)
    args = ap.parse_args()
    pin_blas(args.blas_threads)
    hs = require_hash_seed()

    if args.plan:
        jobs = jobs_for()
        by = {}
        for j in jobs:
            by[(j["arm"], j["backend"])] = by.get((j["arm"], j["backend"]), 0) + 1
        print("{} jobs of {} games".format(len(jobs), GAMES))
        for k in sorted(by):
            print("  {:5s} {:16s} {}".format(k[0], k[1], by[k]))
        return

    if args.self_test:
        self_test()
        return

    if args.smoke:
        if not args.outdir:
            raise SystemExit("--smoke needs --outdir (smoke output is not a result)")
        jobs = jobs_for(d_levels=[1, 6], flips=[0.0, 0.10], la_seeds=[1],
                        noise_seeds=[1], g_rates=[0.2])
        for j in jobs:
            j["games"] = GAMES          # full length, for timing
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
                                         "stress_st2_{}".format(stamp))
    config = {"mode": "run", "matrix": os.path.basename(MATRIX), "d_levels": D_LEVELS,
              "flips": FLIPS, "backends": BACKENDS, "g_rates": G_RATES,
              "g_max_d": G_MAX_D, "la_seeds": LA_SEEDS, "noise_seeds": NOISE_SEEDS,
              "n_lookalikes": N_LOOKALIKES, "games": GAMES, "zipf_s": ZIPF_S,
              "stream_seed": STREAM_SEED, "budget": BUDGET, "confidence": CONFIDENCE,
              "eps": EPS, "tolerance_K": K, "registration": reg,
              "provenance": provenance(), "created": stamp}
    run(outdir, jobs, "real", config, args.workers)


if __name__ == "__main__":
    main()
