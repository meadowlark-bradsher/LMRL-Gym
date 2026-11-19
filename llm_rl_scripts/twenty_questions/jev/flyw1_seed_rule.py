"""FLYW-1 seed rule (user decision 2026-11-18): how many noise seeds the full run needs.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_seed_rule \
        --smoke llm_rl_scripts/twenty_questions/jev/results/flyw1_smoke_20261118

**What is estimated.** A cell's quantity is the mean of an arm difference (e.g. B - A)
over the fixed pool of 158 secrets. The secrets are the whole pool, not a sample, so
the only randomness is the answer noise. Its standard error is estimated from the
within-secret spread across noise seeds, a one-way ANOVA within-secret mean square:

1. per secret i, the sample variance s_i^2 of the paired difference across its S0 = 5
   smoke seeds;
2. pooled over the secrets: MS_w = mean_i s_i^2, with 158 (S0 - 1) degrees of freedom;
3. for a pooled test over M maps (158 M games per seed),
   SE(S) = sqrt(MS_w / (158 M S)), assuming each map of a construction has the
   within-secret variance of the smoke's map for that construction.

**Rule.** For each pooled test, S = max(5, ceil(5 (SE_5 / (margin / 3))^2)). The run
uses the largest S over the tests, capped by the 8-hour budget on 6 workers. If the cap
binds, the analyzer reports SE and treats a result within 2 SE of the margin as
inconclusive.

**Blind.** Only variances, standard errors and seed counts are computed and printed;
the paired differences' means are never formed. Flip-0 cells are deterministic (SE 0).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
from typing import Dict, List

from .flyw1_decision import score_a_record
from .flyw1_run import SMOKE_MAPS, load_inputs

S0 = 5
NOISY = (0.05, 0.1)
N_MAPS = {"align": 8, "mix05": 16, "mix15": 16, "rand": 16, "category": 1}
N_MAPS_TRIM = {"align": 8, "mix05": 8, "mix15": 8, "rand": 8, "category": 1}
HOURS_5 = 0.66          # Part 1 at 5 seeds, 6 workers (first smoke, FLYW1_BUILD.md)
CAP_HOURS = 8.0


def hours(s: int) -> float:
    """Part 1 cost scales with the number of cells per map: 1 (flip 0) + 2 S."""
    return HOURS_5 * (1 + 2 * s) / (1 + 2 * 5)


def load(smoke: str, job_id: str) -> List[dict]:
    with gzip.open(os.path.join(smoke, "jobs", job_id + ".json.gz"), "rt") as f:
        return json.load(f)["games"]


def ms_within(smoke: str, m: str, flip: float, x: str, y: str, field: str) -> float:
    """Pooled within-secret variance of (x - y) on map m. 'A' is scored on the map."""
    _, _, maps = load_inputs()
    cls = maps[m]["classes"]
    per: Dict[str, List[float]] = {}
    for s in range(1, S0 + 1):
        def vals(arm):
            if arm == "A":
                return {g["secret"]: score_a_record(g, cls)[field]
                        for g in load(smoke, "p1_A_f{}_s{}".format(flip, s))}
            return {g["secret"]: g[field]
                    for g in load(smoke, "p1_{}_{}_f{}_s{}".format(arm, m, flip, s))}
        vx, vy = vals(x), vals(y)
        for k in vx:
            per.setdefault(k, []).append(vx[k] - vy[k])
    assert all(len(v) == S0 for v in per.values())
    return statistics.mean(statistics.variance(v) for v in per.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", required=True)
    a = ap.parse_args()
    _, _, maps = load_inputs()
    constr = {maps[m]["alignment_construction"]: m for m in SMOKE_MAPS}
    tests = []
    # (name, construction or "all", flips, x, y, field, margin)
    for c in N_MAPS:
        for f in NOISY:
            tests.append(("DE1(a)", c, f, "B", "A", "probes", 0.1))
            tests.append(("DE1(c)", c, f, "B", "A", "wrong", 0.02))
    tests.append(("DE1(b)", "align", 0.05, "A", "B", "probes", 0.5))
    tests.append(("DE1(b)", "mix05", 0.05, "A", "B", "probes", 0.3))
    tests.append(("DE3(b)", "rand", 0.1, "C", "B", "wrong", 0.02))
    tests.append(("DE5:C", "all", 0.05, "B", "C", "probes", 0.5))
    tests.append(("DE5:D", "all", 0.05, "B", "D", "probes", 0.5))
    rows = []
    for name, c, f, x, y, field, margin in tests:
        if c == "all":
            msw = statistics.mean(ms_within(a.smoke, m, f, x, y, field)
                                  for m in SMOKE_MAPS)
            n_full, n_trim = 57, 33
        else:
            msw = ms_within(a.smoke, constr[c], f, x, y, field)
            n_full, n_trim = N_MAPS[c], N_MAPS_TRIM[c]
        row = {"test": name, "unit": c, "flip": f, "diff": "{} - {}".format(x, y),
               "field": field, "margin": margin, "ms_within": round(msw, 5),
               "df": 158 * (S0 - 1)}
        for tag, n in (("full", n_full), ("trim", n_trim)):
            se5 = math.sqrt(msw / (158 * n * S0))
            s_need = max(5, math.ceil(5 * (se5 / (margin / 3.0)) ** 2))
            row["se5_" + tag] = round(se5, 5)
            row["seeds_" + tag] = s_need
        rows.append(row)
    s_full = max(r["seeds_full"] for r in rows)
    s_cap = max(s for s in range(5, 2000) if hours(s) <= CAP_HOURS)
    out = {"rule": "S = max(5, ceil(5 (SE5 / (margin/3))^2)) per pooled test; run max S, "
                   "capped at {} h on 6 workers".format(CAP_HOURS),
           "smoke": a.smoke, "smoke_seeds": S0, "tests": rows,
           "seeds_needed_full": s_full, "seeds_cap": s_cap,
           "cap_binds": s_full > s_cap, "seeds_to_run": min(s_full, s_cap),
           "part1_hours_on_6": round(hours(min(s_full, s_cap)), 2),
           "note": "variances and standard errors only; no paired means are formed"}
    with open(os.path.join(a.smoke, "seed_rule.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print("{:7s} {:9s} flip {:<5} {:6s} ms_w={:.4f} se5={:.4f} margin/3={:.4f} -> S={}"
              .format(r["test"], r["unit"], r["flip"], r["field"], r["ms_within"],
                      r["se5_full"], r["margin"] / 3, r["seeds_full"]))
    print(json.dumps({k: out[k] for k in ("seeds_needed_full", "seeds_cap", "cap_binds",
                                          "seeds_to_run", "part1_hours_on_6")}))


if __name__ == "__main__":
    main()
