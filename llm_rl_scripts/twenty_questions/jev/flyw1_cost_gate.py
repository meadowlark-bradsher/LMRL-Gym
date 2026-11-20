"""Build gate and projection for the FLYW-1 total-cost follow-up (``FLYW1_COST_PREREG.md``).

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_cost_gate \
        --out llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_gate_<date>

No fresh map is generated and no fresh noise seed is played: the gate uses only FLYW-1's
generator seeds (1, 2) and FLYW-1's flip-0 games.

1. **Maps.** ``class_maps.build_all(seeds=(1,))`` and ``(seeds=(2,))``, written with
   ``write_all`` (the ``--seeds`` path), reproduce every FLYW-1 map file byte for byte,
   including ``cat17`` and the ``align`` maps, which therefore do not depend on the seeds.
2. **Flip 0.** A, B and D on ``align`` and ``cat17`` at flip 0, run through
   ``part1_jobs(arms=("A","B","D"), maps_dir=<the regenerated set>)`` and ``play_part1``,
   reproduce FLYW-1 Part 1's games exactly (every record field).
3. **Jobs.** ``part1_jobs`` with no new options still lists exactly FLYW-1 Part 1's 30,590
   jobs. The cost run's jobs (A, B, D; 57 maps; seeds 1001-1020 at flips 0.05 and 0.10,
   plus flip 0) carry ``maps_dir``; no noisy job shares an id or a noise seed with FLYW-1
   (Part 1 used 1-66, the DE1 extension up to 787). Flip-0 jobs are deterministic replays
   and keep FLYW-1's ids, in the run's own directory.
4. **Projection.** The cost run's hours from FLYW-1 Part 1's per-job ``secs``; a fresh
   seed-3/4 map borrows the timings of the FLYW-1 map of the same K, profile and
   construction. FLYW-1's own maps stand in for the fresh set (same ids up to the seed, same
   count), so no fresh map is needed.
"""

from __future__ import annotations

import argparse
import filecmp
import glob
import gzip
import json
import os
import shutil
import sys

from . import class_maps
from .flyw1_run import (HERE, MAPS, load_inputs, part1_jobs, play_part1,
                        project_extension, resolve_maps)
from .stress_harness import require_hash_seed

PART1 = os.path.join(HERE, "results", "flyw1_part1_20261118")
COST_SEED_START, COST_SEEDS = 1001, 20


def check_maps(tmp: str) -> dict:
    out = {}
    for s in (1, 2):
        d = os.path.join(tmp, "maps_seed{}".format(s))
        maps, prov = class_maps.build_all(seeds=(s,))
        class_maps.write_all(d, maps, prov, (s,))
        names = sorted(os.listdir(os.path.join(d, "maps")))
        diff = [n for n in names
                if not filecmp.cmp(os.path.join(d, "maps", n), os.path.join(MAPS, "maps", n),
                                   shallow=False)]
        out["seed{}".format(s)] = {"maps": len(names), "byte_identical": len(names) - len(diff),
                                   "differ": diff}
    out["PASS"] = all(v["differ"] == [] and v["maps"] == 33 for v in out.values())
    return out


def check_flip0(maps_dir_abs: str) -> dict:
    rel = os.path.relpath(maps_dir_abs, HERE)
    _, _, maps = load_inputs(resolve_maps(rel))
    ids = sorted(m for m, mp in maps.items()
                 if mp["alignment_construction"] in ("align", "category"))
    jobs = part1_jobs(1, ids, flips=(0.0,), arms=("A", "B", "D"), maps_dir=rel)
    res, ok = {}, True
    for j in jobs:
        with gzip.open(os.path.join(PART1, "jobs", j["job_id"] + ".json.gz"), "rt") as f:
            ref = json.load(f)["games"]
        mine = json.loads(json.dumps(play_part1(j)))
        same = mine == ref
        ok &= same
        res[j["job_id"]] = same
    return {"jobs": len(jobs), "maps": ids, "identical": sum(res.values()),
            "per_job": res, "PASS": ok and len(jobs) == 1 + 2 * len(ids)}


def check_jobs(cost_jobs) -> dict:
    flyw1 = {os.path.basename(p)[:-len(".json.gz")]
             for p in glob.glob(os.path.join(PART1, "jobs", "*.json.gz"))}
    default = {j["job_id"] for j in part1_jobs(66)}
    ids = [j["job_id"] for j in cost_jobs]
    # Flip-0 jobs are deterministic replays and keep FLYW-1's ids (in their own output
    # directory); a noisy job must never share an id or a noise seed with FLYW-1.
    noisy = {j["job_id"] for j in cost_jobs if j["flip"] != 0.0}
    seeds = sorted({j["seed"] for j in cost_jobs})
    return {"default_part1_jobs_unchanged": default == flyw1, "flyw1_part1_jobs": len(flyw1),
            "cost_jobs": len(ids), "cost_jobs_unique": len(set(ids)) == len(ids),
            "noisy_overlap_with_flyw1_ids": len(noisy & flyw1),
            "flip0_ids_shared_with_flyw1": len((set(ids) - noisy) & flyw1),
            "cost_seeds": [seeds[0], seeds[1], seeds[-1]],
            "all_carry_maps_dir": all("maps_dir" in j for j in cost_jobs),
            "arms": sorted({j["arm"] for j in cost_jobs}),
            "PASS": (default == flyw1 and len(set(ids)) == len(ids)
                     and not noisy & flyw1 and seeds[0] == 0
                     and seeds[1] == COST_SEED_START
                     and seeds[-1] == COST_SEED_START + COST_SEEDS - 1
                     and all("maps_dir" in j for j in cost_jobs))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6, help="for the projection only")
    a = ap.parse_args()
    hs = require_hash_seed()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp_maps")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    res = {"pythonhashseed": hs}
    res["maps"] = check_maps(tmp)
    res["flip0"] = check_flip0(os.path.join(tmp, "maps_seed1"))
    # Stand-in for the fresh set: FLYW-1's own map ids (seed 1/2 in place of 3/4).
    stand_in = os.path.relpath(MAPS, HERE)
    cost_jobs = part1_jobs(COST_SEEDS, arms=("A", "B", "D"), seed_start=COST_SEED_START,
                           maps_dir=stand_in)
    res["jobs"] = check_jobs(cost_jobs)
    res["projection"] = project_extension(cost_jobs, PART1, a.workers)
    res["projection"]["note"] = ("FLYW-1 Part 1 per-job secs, up to 20 timings per "
                                 "(arm, map, flip); wall = CPU / workers (efficiency 1.0)")
    res["PASS"] = res["maps"]["PASS"] and res["flip0"]["PASS"] and res["jobs"]["PASS"]
    shutil.rmtree(tmp)
    with open(os.path.join(a.out, "gate.json"), "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({k: (v if k != "flip0" else {x: v[x] for x in v if x != "per_job"})
                      for k, v in res.items()}, indent=1))
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
