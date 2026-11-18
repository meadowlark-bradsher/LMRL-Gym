"""FLYW-1 runner: Part 1 (arms A-D, FLYW-4), Part 2 (escalation triggers, FLYW-5), the
DE0 build gate and the smoke run. Design: ``FLYW1_DECISION_PREREG.md``; build notes:
``FLYW1_BUILD.md``.

    # DE0 build gate (single core, a few minutes):
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_run \
        --gate --out llm_rl_scripts/twenty_questions/jev/results/flyw1_gate_<date>

    # smoke (projects cost, measures the paired SE):
    ... flyw1_run --smoke --workers 6 --out .../flyw1_smoke_<date>

    # full runs (resumable: finished jobs are skipped):
    ... flyw1_run --part 1 --workers 6 --seeds 5 --out .../flyw1_part1_<date>
    ... flyw1_run --part 2 --workers 6 --seeds 5 --out .../flyw1_part2_<date>

Every job writes ``jobs/<job_id>.json.gz`` when it finishes, so a stopped run resumes
by rerunning the same command. Pin BLAS threads to 1 per worker and submit through
``jobq -N <workers>``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

from .bed import Hypothesis, load_hypotheses
from .class_maps import GENERATOR_VERSION, identity_tree, load_table, measures
from .flyw1_decision import (P0A, ClassGame, KeyedOracle, holdouts, keyed_uniform,
                             kill_rule_twins, play_a, score_a_record,
                             tloss_continuation)
from .openworld import run_openworld_episode
from .stress_harness import require_hash_seed

HERE = os.path.dirname(__file__)
MATRIX = os.path.join(HERE, "matrix_jev.json")
MAPS = os.path.join(HERE, "results", "flyw1_class_maps_20261118")
CALIB = os.path.join(HERE, "results", "flyw1_a_calibration_20261118", "a_games.json")
FLIPS = (0.0, 0.05, 0.10)
RATIOS = (0.5, 0.2, 0.05)
PU_SCALES = (1.0, 0.5, 2.0)
PART1_ARMS = ("B", "C", "D", "Cbin")      # Cbin is reported only
PART2_MAPS = ("cat17", "k4_bal_align", "k4_skew_align", "k8_bal_align", "k8_skew_align",
              "k4_bal_rand_s1", "k4_skew_rand_s1", "k8_bal_rand_s1", "k8_skew_rand_s1")

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #

def load_inputs(maps_dir: str = MAPS):
    if "inputs" in _CACHE:
        return _CACHE["inputs"]
    with open(MATRIX) as f:
        matrix = json.load(f)["matrix"]
    hyps = load_hypotheses()
    with open(os.path.join(maps_dir, "index.json")) as f:
        index = json.load(f)
    maps = {}
    for row in index["maps"]:
        with open(os.path.join(maps_dir, "maps", row["map_id"] + ".json")) as f:
            mp = json.load(f)
        # DE0(d): the registered assignment and generator version.
        if mp["assignment_sha256_12"] != row["assignment_sha256_12"] \
                or mp["version"] != GENERATOR_VERSION:
            raise SystemExit("map {} does not match its registered hash/version"
                             .format(row["map_id"]))
        maps[row["map_id"]] = mp
    _CACHE["inputs"] = (matrix, hyps, maps)
    return _CACHE["inputs"]


def by_names(hyps: Sequence[Hypothesis], names: Sequence[str]) -> List[Hypothesis]:
    d = {h.name: h for h in hyps}
    return [d[n] for n in names]


# --------------------------------------------------------------------------- #
# Jobs                                                                         #
# --------------------------------------------------------------------------- #

def seeds_for(flip: float, n_seeds: int) -> List[int]:
    return [0] if flip == 0.0 else list(range(1, n_seeds + 1))


def part1_jobs(n_seeds: int, map_ids: Optional[Sequence[str]] = None,
               flips: Sequence[float] = FLIPS) -> List[dict]:
    _, _, maps = load_inputs()
    ids = list(map_ids) if map_ids else list(maps)
    jobs = []
    for flip in flips:
        for s in seeds_for(flip, n_seeds):
            jobs.append({"part": 1, "arm": "A", "map_id": None, "flip": flip, "seed": s,
                         "job_id": "p1_A_f{}_s{}".format(flip, s)})
            for m in ids:
                for arm in PART1_ARMS:
                    jobs.append({"part": 1, "arm": arm, "map_id": m, "flip": flip,
                                 "seed": s,
                                 "job_id": "p1_{}_{}_f{}_s{}".format(arm, m, flip, s)})
            # Reported only: B on the identity map (every object its own class).
            jobs.append({"part": 1, "arm": "B", "map_id": "identity", "flip": flip,
                         "seed": s, "job_id": "p1_B_identity_f{}_s{}".format(flip, s)})
    return jobs


def rotations(map_ids: Sequence[str] = PART2_MAPS) -> List[dict]:
    _, hyps, maps = load_inputs()
    objs = [h.name for h in hyps]
    out = []
    for m in map_ids:
        out.extend(holdouts(m, maps[m]["classes"], objs, maps[m]["k"]))
    return out


def part2_jobs(n_seeds: int, in_pool_reps: int,
               rot_ids: Optional[Sequence[str]] = None,
               flips: Sequence[float] = FLIPS) -> List[dict]:
    jobs = []
    for r in rotations():
        if rot_ids and r["rotation"] not in rot_ids:
            continue
        for flip in flips:
            for s in seeds_for(flip, n_seeds):
                for arm in ("A", "B", "D"):
                    jobs.append({"part": 2, "arm": arm, "rotation": r["rotation"],
                                 "map_id": r["map_id"], "flip": flip, "seed": s,
                                 "in_pool_reps": in_pool_reps,
                                 "job_id": "p2_{}_{}_f{}_s{}".format(arm, r["rotation"],
                                                                     flip, s)})
    return jobs


# --------------------------------------------------------------------------- #
# Playing                                                                      #
# --------------------------------------------------------------------------- #

def play_part1(job: dict) -> List[dict]:
    matrix, hyps, maps = load_inputs()
    oracle = KeyedOracle(matrix, job["flip"], job["seed"])
    out = []
    for h in hyps:
        oracle.begin(h.name + "#0")
        if job["arm"] == "A":
            out.append(play_a(h, hyps, matrix, oracle, 0.0))
        else:
            cls = ({x.name: i for i, x in enumerate(hyps)} if job["map_id"] == "identity"
                   else maps[job["map_id"]]["classes"])
            g = ClassGame(job["arm"], h, hyps, matrix, cls, oracle, 0.0).run()
            rec = g.record()
            if rec["outcome"] == "trigger":
                rec["outcome"] = "expert"
            out.append(rec)
    return out


def part2_games(rot: dict, reps: int, flip: float):
    """(secret name, kind, rep, weight). At flip 0 the in-pool repeats are identical
    (deterministic), so each is played once with weight ``reps``."""
    games = []
    for n in rot["known"]:
        if flip == 0.0:
            games.append((n, "in_pool", 0, reps))
        else:
            games.extend((n, "in_pool", r, 1) for r in range(reps))
    games.extend((n, "same", 0, 1) for n in rot["same_class_novels"])
    games.extend((n, "new", 0, 1) for n in rot["new_class_novels"])
    return games


def play_part2(job: dict) -> List[dict]:
    matrix, hyps, maps = load_inputs()
    rot = next(r for r in rotations([job["map_id"]]) if r["rotation"] == job["rotation"])
    cls = maps[job["map_id"]]["classes"]
    known = by_names(hyps, rot["known"])
    n_novel = len(rot["same_class_novels"]) + len(rot["new_class_novels"])
    p_u = n_novel / (n_novel + job["in_pool_reps"] * len(known))
    oracle = KeyedOracle(matrix, job["flip"], job["seed"])
    h_of = {h.name: h for h in hyps}
    out = []
    for name, kind, rep, weight in part2_games(rot, job["in_pool_reps"], job["flip"]):
        oracle.begin("{}#{}".format(name, rep))
        h = h_of[name]
        if job["arm"] == "A":
            rec = play_a(h, known, matrix, oracle, p_u)
        else:
            g = ClassGame(job["arm"], h, known, matrix, cls, oracle, p_u).run()
            rec = g.record()
            if g.outcome == "trigger":
                tl = {}
                for ratio in RATIOS:
                    for sc in PU_SCALES:
                        tl["mean_r{}_pu{}".format(ratio, sc)] = tloss_continuation(
                            g, ratio, min(p_u * sc, 0.99), "mean")
                    tl["max_r{}_pu1.0".format(ratio)] = tloss_continuation(
                        g, ratio, p_u, "max")
                rec["tloss"] = tl
        rec.update({"kind": kind, "rep": rep, "weight": weight, "p_u": round(p_u, 6)})
        out.append(rec)
    return out


def play_job(args) -> tuple:
    job, outdir = args
    path = os.path.join(outdir, "jobs", job["job_id"] + ".json.gz")
    if os.path.exists(path):
        return job["job_id"], 0.0, True
    t0 = time.time()
    games = play_part1(job) if job["part"] == 1 else play_part2(job)
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
        for jid, secs, skipped in pool.imap_unordered(
                play_job, [(j, outdir) for j in jobs]):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print("  {}/{} jobs ({:.1f} min)".format(done, len(jobs),
                                                          (time.time() - t0) / 60),
                      flush=True)


# --------------------------------------------------------------------------- #
# DE0 build gate                                                               #
# --------------------------------------------------------------------------- #

def gate(outdir: str) -> dict:
    matrix, hyps, maps = load_inputs()
    res: Dict[str, object] = {}

    # Keyed noise is order-free and paired: the same (seed, game, q) gives the same
    # flip whatever was asked before.
    o1, o2 = KeyedOracle(matrix, 0.3, 7), KeyedOracle(matrix, 0.3, 7)
    from .openworld import build_openworld_bank
    qs = build_openworld_bank(hyps)["attribute"]
    o1.begin("Apple#0"), o2.begin("Apple#0")
    a1 = {q.qid: o1(q, hyps[1]) for q in qs}
    a2 = {q.qid: o2(q, hyps[1]) for q in reversed(qs)}
    res["keyed_order_free"] = a1 == a2
    rate = sum(keyed_uniform(3, "x#{}".format(i), "q") < 0.1 for i in range(20000)) / 20000
    res["keyed_flip_rate_at_0.1"] = round(rate, 4)

    # (a) A at flip 0 reproduces the calibration run; at flip 0.10 the runner's A
    # equals a direct run_openworld_episode with a fresh keyed oracle.
    with open(CALIB) as f:
        calib = {g["secret"]: g for g in json.load(f)}
    a_recs = play_part1({"part": 1, "arm": "A", "map_id": None, "flip": 0.0, "seed": 0})
    mism = [r["secret"] for r in a_recs
            if [[k[0], key, a] for k, key, a in calib[r["secret"]]["steps"]] != r["steps"]
            or calib[r["secret"]]["outcome"] != r["outcome"]]
    res["de0a_flip0_matches_calibration"] = not mism
    res["de0a_flip0_mismatches"] = mism[:10]
    direct_ok = True
    for h in hyps[:30]:
        o = KeyedOracle(matrix, 0.10, 1)
        o.begin(h.name + "#0")
        r = play_a(h, hyps, matrix, o, 0.0)
        o2 = KeyedOracle(matrix, 0.10, 1)
        o2.begin(h.name + "#0")
        ep = run_openworld_episode(h, hyps, matrix, 0.0, o2, max_questions=20,
                                   confidence=0.9, eps=0.05, backend="modeswitch_guard",
                                   tolerance=3, no_repeat=True, end_rule="always")
        if [[s.kind[0], s.qid.split(":", 1)[1], s.answer_yes] for s in ep.steps] \
                != r["steps"]:
            direct_ok = False
    res["de0a_noisy_matches_direct"] = direct_ok

    # (b) one-class map: B, C, D act at the first probe and are right.
    one = {h.name: 0 for h in hyps}
    bad_b = []
    for arm in ("B", "C", "D"):
        for h in hyps:
            o = KeyedOracle(matrix, 0.0, 0)
            o.begin(h.name + "#0")
            g = ClassGame(arm, h, hyps, matrix, one, o, 0.0).run()
            if not (g.outcome == "right" and g.probes == 1):
                bad_b.append((arm, h.name))
    res["de0b_one_class"] = not bad_b

    # (c) flip 0: B, C, D take wrong actions only on unresolvable objects. Full check
    # in the analyzer over every flip-0 job; here on one map per construction.
    objs, qids, table, _ = load_table()
    twins = kill_rule_twins(matrix, hyps)
    sample = ["cat17", "k2_bal_rand_s1", "k4_skew_mix15_s1", "k8_bal_mix05_s2",
              "k16_skew_align", "k16_bal_rand_s2"]
    bad_c = []
    t0 = time.time()
    for m in sample:
        cls = maps[m]["classes"]
        unres = {o for o in objs if any(cls[t] != cls[o] for t in twins[o])}
        for arm in ("B", "C", "D"):
            for h in hyps:
                o = KeyedOracle(matrix, 0.0, 0)
                o.begin(h.name + "#0")
                g = ClassGame(arm, h, hyps, matrix, cls, o, 0.0).run()
                if g.wrong and h.name not in unres:
                    bad_c.append((m, arm, h.name, g.outcome, g.wrong))
    res["de0c_sample_maps"] = sample
    res["de0c_flip0_wrong_only_unresolvable"] = not bad_c
    res["de0c_violations"] = bad_c[:20]
    res["de0c_secs_per_game"] = round((time.time() - t0) / (len(sample) * 3 * len(hyps)), 4)

    # (d) every map matched its registered hash and generator version (load_inputs).
    res["de0d_maps_registered"] = len(maps) == 57

    # Part 2 rotations: every class keeps a known member; alignment of each pool.
    rots = rotations()
    rot_rows = []
    for r in rots:
        cls = maps[r["map_id"]]["classes"]
        kept = {cls[o] for o in r["known"]}
        others = set(cls.values()) - {r["held_class"]}
        known = r["known"]
        _, kpaths = identity_tree(known, qids, table)
        meas = measures(known, qids, table, kpaths, {o: cls[o] for o in known})
        rot_rows.append({"rotation": r["rotation"], "held_class": r["held_class"],
                         "n_new": len(r["new_class_novels"]),
                         "n_same": len(r["same_class_novels"]), "n_known": len(known),
                         "every_class_kept": others <= kept,
                         "alignment_bintree_reduced_pool": meas["alignment_bintree"]})
    res["part2_n_rotations"] = len(rots)
    res["part2_every_class_kept"] = all(r["every_class_kept"] for r in rot_rows)
    res["part2_rotations"] = rot_rows

    res["PASS"] = bool(res["keyed_order_free"] and res["de0a_flip0_matches_calibration"]
                       and res["de0a_noisy_matches_direct"] and res["de0b_one_class"]
                       and res["de0c_flip0_wrong_only_unresolvable"]
                       and res["de0d_maps_registered"] and res["part2_n_rotations"] == 31
                       and res["part2_every_class_kept"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gate.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


# --------------------------------------------------------------------------- #
# Smoke                                                                        #
# --------------------------------------------------------------------------- #

SMOKE_MAPS = ("cat17", "k4_bal_align", "k8_skew_mix05_s1", "k4_bal_mix15_s1",
              "k8_bal_rand_s1")


def smoke(outdir: str, workers: int, n_seeds: int = 2) -> dict:
    """Part 1: one map per construction, flips 0.05/0.10, seeds 1..n_seeds (2 in the
    first smoke; 5 for the seed rule, ``flyw1_seed_rule.py``). Part 2: two rotations,
    flip 0.10, seed 1. Projects the full cost."""
    p1 = part1_jobs(n_seeds, SMOKE_MAPS, flips=(0.05, 0.10))
    rots = [r["rotation"] for r in rotations()]
    p2 = part2_jobs(1, 4, [rots[0], rots[-1]], flips=(0.10,))
    run(outdir, p1 + p2, workers, {"mode": "smoke"})
    secs: Dict[str, List[float]] = {"p1A": [], "p1": [], "p2": []}
    jobs = {}
    for j in p1 + p2:
        with gzip.open(os.path.join(outdir, "jobs", j["job_id"] + ".json.gz"), "rt") as f:
            d = json.load(f)
        jobs[j["job_id"]] = d
        key = "p1A" if (j["part"] == 1 and j["arm"] == "A") else "p1" if j["part"] == 1 else "p2"
        secs[key].append(d["secs"])
    # Paired per-game differences A - B (probes, wrong actions) per construction.
    _, _, maps = load_inputs()
    paired = {}
    for m in SMOKE_MAPS:
        cls = maps[m]["classes"]
        dp, dw = [], []
        for flip in (0.05, 0.10):
            for s in range(1, n_seeds + 1):
                a = jobs["p1_A_f{}_s{}".format(flip, s)]["games"]
                b = jobs["p1_B_{}_f{}_s{}".format(m, flip, s)]["games"]
                for ra, rb in zip(a, b):
                    sa = score_a_record(ra, cls)
                    dp.append(sa["probes"] - rb["probes"])
                    dw.append(sa["wrong"] - rb["wrong"])
        paired[m] = {"construction": maps[m]["alignment_construction"],
                     "sd_probes": round(statistics.pstdev(dp), 4),
                     "sd_wrong": round(statistics.pstdev(dw), 4)}
        # No paired means are computed or written: the build stays blind to B vs A.
    # Projection.
    n_maps_by_c = {"align": 8, "mix05": 16, "mix15": 16, "rand": 16, "category": 1}
    seeds_needed = {}
    for m, p in paired.items():
        n_per_seed = 158 * n_maps_by_c[p["construction"]]   # games per flip per seed, pooled
        for meas, margin in (("sd_probes", 0.1), ("sd_wrong", 0.02)):
            need = (p[meas] / (margin / 3.0)) ** 2 / n_per_seed
            seeds_needed["{}:{}".format(m, meas)] = round(need, 2)
    mean_p1 = statistics.mean(secs["p1"])
    mean_p1a = statistics.mean(secs["p1A"])
    proj = {}
    na = len(PART1_ARMS)
    for n_seeds in (5, 10, 20):
        n_cells = 1 + 2 * n_seeds
        cpu = (57 * na + 1) * n_cells * mean_p1 + n_cells * mean_p1a
        proj["part1_seeds{}_hours_on_6".format(n_seeds)] = round(cpu / 6 / 3600, 2)
        trim = (33 * na + 1) * n_cells * mean_p1 + n_cells * mean_p1a
        proj["part1_trimmed_seeds{}_hours_on_6".format(n_seeds)] = round(trim / 6 / 3600, 2)
    # Part 2: per-game seconds by arm, times games per job.
    p2_rate = {}
    for j in p2:
        d = jobs[j["job_id"]]
        p2_rate.setdefault(j["arm"], []).append(d["secs"] / sum(1 for _ in d["games"]))
    all_rots = rotations()
    games_reps = {reps: sum(len(r["known"]) * reps + len(r["same_class_novels"])
                            + len(r["new_class_novels"]) for r in all_rots)
                  for reps in (4, 2)}
    games_flip0 = sum(len(r["known"]) + len(r["same_class_novels"])
                      + len(r["new_class_novels"]) for r in all_rots)
    for reps in (4, 2):
        for n_seeds in (5,):
            per_arm = {a: statistics.mean(v) for a, v in p2_rate.items()}
            cpu = sum(per_arm.values()) * (games_flip0 + 2 * n_seeds * games_reps[reps])
            proj["part2_reps{}_seeds{}_hours_on_6".format(reps, n_seeds)] = round(cpu / 6 / 3600, 2)
    out = {"secs_per_job": {k: round(statistics.mean(v), 2) for k, v in secs.items() if v},
           "paired": paired, "seeds_needed_for_se_le_margin_over_3": seeds_needed,
           "projection": proj}
    with open(os.path.join(outdir, "smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, choices=(1, 2))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--in-pool-reps", type=int, default=4)
    ap.add_argument("--smoke-seeds", action="store_true",
                    help="smoke: use --seeds (the seed-rule smoke uses --seeds 5)")
    ap.add_argument("--trim", action="store_true",
                    help="Part 1: seed-1 maps plus align and cat17 (the prereg's trim)")
    a = ap.parse_args()
    hs = require_hash_seed()
    if a.gate:
        res = gate(a.out)
        print(json.dumps({k: v for k, v in res.items() if k != "part2_rotations"}, indent=1))
        sys.exit(0 if res["PASS"] else 1)
    if a.smoke:
        out = smoke(a.out, a.workers, a.seeds if a.smoke_seeds else 2)
        # The smoke's own summary includes paired means; print only cost and spread.
        print(json.dumps({k: out[k] for k in ("secs_per_job", "projection")}, indent=1))
        return
    if a.part == 1:
        _, _, maps = load_inputs()
        ids = [m for m, mp in maps.items() if not a.trim or mp["seed"] in (None, 1)]
        jobs = part1_jobs(a.seeds, ids)
    elif a.part == 2:
        jobs = part2_jobs(a.seeds, a.in_pool_reps)
    else:
        ap.error("--part, --gate or --smoke")
    run(a.out, jobs, a.workers, {"part": a.part, "seeds": a.seeds, "trim": a.trim,
                                 "in_pool_reps": a.in_pool_reps, "pythonhashseed": hs,
                                 "p0a": P0A, "flips": FLIPS, "ratios": RATIOS,
                                 "pu_scales": PU_SCALES})


if __name__ == "__main__":
    main()
