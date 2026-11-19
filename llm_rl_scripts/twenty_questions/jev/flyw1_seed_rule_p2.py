"""FLYW-1 Addendum 1, item 3: Part 2's own seed count for ET1-ET3, computed blind.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw1_seed_rule_p2 \
        --part2 llm_rl_scripts/twenty_questions/jev/results/flyw1_part2_20261118

**What is estimated.** The same within-secret rule as Part 1 (``flyw1_seed_rule.py``),
applied to Part 2's own 5 seeds. A *unit* is one game: (map, rotation, secret, repeat,
kind). It is the same set of games under every seed, and the noise is the only
randomness. For each unit, take the sample variance s_u^2 of its per-game value across
the 5 seeds. The SE of a pooled rate over the units U at S seeds is
sqrt(sum_u w_u^2 s_u^2) / (W sqrt S), with W = sum_u w_u. Flip 0 is deterministic and is
left out, as the tests pool flips 0.05 and 0.10 only.

**Per-game values, at ratio 0.2 with the true p_U (``mean_r0.2_pu1.0``), per arm B and D:**

* ET3 (`rand` maps, in-pool and same-class games): T-loss expert minus T-mass expert.
  Bar: |mean| <= 0.02, margin 0.02.
* ET2 (every map on its own, new-class novels): T-loss acting at the trigger point.
  Bar: <= 0.10, margin 0.10 (absolute bar, as DE5).
* ET1 (`align` maps and `cat17`, in-pool and same-class games): the bar is a ratio,
  R = T-loss rate / T-mass rate <= 0.7, so its margin is 0.3 on the ratio scale (the
  distance from no effect, 1, to the bar, as DE1(b)'s margin is its bar). By the delta
  method at the bar, SE(R) = SE(T-loss - 0.7 T-mass) / (T-mass rate). This needs
  T-mass's pooled spurious rate. It is formed inside the tool and is **not printed**;
  only SE(R) and S are.

**Rule.** S = max(5, ceil(5 (SE_5 / (margin / 3))^2)) per test.

**Cost.** Projected from the jobs' own ``secs``: the extension runs arms B and D on
seeds 6..S at both noisy flips. Wall time = CPU seconds / 6 workers x the efficiency the
registered run achieved (its measured wall time against its CPU seconds / 6).

**Blind.** Only variances, standard errors, seed counts and hours are printed or
written. No rate, no difference of rates and no outcome count is printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from .analyze_flyw1 import load_jobs, load_maps, p2_outcome

S0 = 5
NOISY = (0.05, 0.1)
TRIG = "mean_r0.2_pu1.0"
WALL_MIN_REGISTERED = 30.1      # jobq 64, measured
BUDGET_HOURS = 4.0
SPUR = ("in_pool", "same")


def seed_count(se5: float, margin: float) -> int:
    return max(5, math.ceil(5 * (se5 / (margin / 3.0)) ** 2))


def pooled_se(units: Dict[Tuple, Dict[int, Tuple[float, float]]], s: int) -> float:
    """units: key -> {seed: (weight, value)}. SE of the weighted pooled mean at s seeds."""
    num, w_tot = 0.0, 0.0
    for per in units.values():
        assert len(per) == S0, "unit missing a seed"
        ws = {w for w, _ in per.values()}
        assert len(ws) == 1, "weight differs across seeds"
        w = ws.pop()
        v = [x for _, x in per.values()]
        m = sum(v) / S0
        var = sum((x - m) ** 2 for x in v) / (S0 - 1)
        num += w * w * var
        w_tot += w
    return math.sqrt(num) / (w_tot * math.sqrt(s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part2", required=True)
    a = ap.parse_args()
    maps = load_maps()
    jobs = load_jobs(a.part2)
    constr = {m: maps[m]["alignment_construction"] for m in maps}

    # (test, arm[, map]) -> unit -> seed -> (w, value)
    et1_loss = defaultdict(lambda: defaultdict(dict))   # value: loss - 0.7 mass
    et1_mass_w = defaultdict(float)
    et1_mass_n = defaultdict(float)
    et3 = defaultdict(lambda: defaultdict(dict))
    et2 = defaultdict(lambda: defaultdict(dict))
    secs_bd_noisy = 0.0
    secs_all = 0.0
    for j in jobs:
        jb = j["job"]
        secs_all += j["secs"]
        if jb["arm"] not in ("B", "D") or jb["flip"] not in NOISY:
            continue
        secs_bd_noisy += j["secs"]
        cls = maps[jb["map_id"]]["classes"]
        c = constr[jb["map_id"]]
        for g in j["games"]:
            unit = (jb["map_id"], jb["rotation"], jb["flip"], g["secret"], g["rep"], g["kind"])
            om = p2_outcome(g, "mass", cls)
            ol = p2_outcome(g, TRIG, cls)
            w, s = g["weight"], jb["seed"]
            if g["kind"] in SPUR:
                if c in ("align", "category"):
                    et1_loss[jb["arm"]][unit][s] = (w, float(ol["expert"]) - 0.7 * float(om["expert"]))
                    et1_mass_w[jb["arm"]] += w
                    et1_mass_n[jb["arm"]] += w * float(om["expert"])
                elif c == "rand":
                    et3[jb["arm"]][unit][s] = (w, float(ol["expert"]) - float(om["expert"]))
            else:
                et2[(jb["arm"], jb["map_id"])][unit][s] = (w, float(ol["acted_at_trigger"]))

    rows: List[dict] = []
    for arm in ("B", "D"):
        tm = et1_mass_n[arm] / et1_mass_w[arm]          # internal only, never printed
        se5 = pooled_se(et1_loss[arm], S0) / tm if tm > 0 else float("inf")
        rows.append({"test": "ET1", "arm": arm, "unit": "align+cat17", "scale": "ratio",
                     "margin": 0.3, "n_units": len(et1_loss[arm]),
                     "se5": round(se5, 5), "seeds": seed_count(se5, 0.3)})
        se5 = pooled_se(et3[arm], S0)
        rows.append({"test": "ET3", "arm": arm, "unit": "rand", "scale": "difference",
                     "margin": 0.02, "n_units": len(et3[arm]),
                     "se5": round(se5, 5), "seeds": seed_count(se5, 0.02)})
        for (a2, m), units in sorted(et2.items()):
            if a2 != arm:
                continue
            se5 = pooled_se(units, S0)
            rows.append({"test": "ET2", "arm": arm, "unit": m, "scale": "rate",
                         "margin": 0.1, "n_units": len(units),
                         "se5": round(se5, 5), "seeds": seed_count(se5, 0.1)})

    eff = WALL_MIN_REGISTERED * 60 / (secs_all / 6.0)
    per_seed_h = secs_bd_noisy / S0 / 6.0 * eff / 3600.0
    reg_h = WALL_MIN_REGISTERED / 60.0
    s_need = max(r["seeds"] for r in rows)
    s_cap = max(s for s in range(S0, 5000) if reg_h + (s - S0) * per_seed_h <= BUDGET_HOURS)
    out = {"rule": "S = max(5, ceil(5 (SE5 / (margin/3))^2)) per test (Addendum 1, item 3)",
           "part2": a.part2, "smoke_seeds": S0, "trigger": TRIG, "flips": list(NOISY),
           "tests": rows, "seeds_needed": s_need,
           "extension_hours_on_6": round((s_need - S0) * per_seed_h, 2),
           "part2_total_hours_on_6": round(reg_h + (s_need - S0) * per_seed_h, 2),
           "budget_hours": BUDGET_HOURS, "seeds_cap_in_budget": s_cap,
           "fits_budget": s_need <= s_cap,
           "hours_per_extra_seed_BD": round(per_seed_h, 4), "wall_efficiency": round(eff, 3),
           "note": "variances, SEs, seed counts and hours only; ET1's T-mass level is formed "
                   "internally to scale its ratio SE and is not printed"}
    with open(os.path.join(a.part2, "seed_rule_p2.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for r in rows:
        print("{:4s} {} {:16s} {:10s} n={:5d} se5={:.4f} margin/3={:.4f} -> S={}".format(
            r["test"], r["arm"], r["unit"], r["scale"], r["n_units"], r["se5"],
            r["margin"] / 3, r["seeds"]))
    print(json.dumps({k: out[k] for k in ("seeds_needed", "extension_hours_on_6",
                                          "part2_total_hours_on_6", "seeds_cap_in_budget",
                                          "fits_budget", "hours_per_extra_seed_BD")}))


if __name__ == "__main__":
    main()
