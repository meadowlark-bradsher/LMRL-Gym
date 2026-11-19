"""FLYW-7 blind seed rule (``FLYW2_COMPOUND_PREREG.md`` §Seeds).

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.flyw2_seed_rule \
        --smoke llm_rl_scripts/twenty_questions/jev/results/flyw2_smoke_<date>

Reads the 5-seed smoke and writes ``seed_rule.json`` beside it. **Blind:** it forms and
prints only within-unit variances, standard errors, seed counts and timings. No rate,
cost or difference between kinds is written or printed; the means it subtracts inside
each unit are discarded.

For each test the per-unit quantity is formed per seed:

* CX1(a) / CX1(b): the exhaustion-escalation / catch-all-firing indicator, per KK pair
  and per single; the test is a difference of the two pooled means, so its SE is
  √(SE_KK² + SE_single²);
* CX2: the penalty per KK pair (pair cost − mean of its members' single costs at the same
  seed), at c_expert 10, c_wrong 50, verified; pooled over the two flips, so
  SE = √(SE_0.05² + SE_0.10²) ÷ 2;
* CX3(a) / CX3(b): the partial / silent-partial indicator per KK pair.

Within-unit variance: for each unit the sample variance across its seeds, pooled over
units (one-way ANOVA within mean square). SE of a mean over N units at S seeds =
√(Σ σᵢ²) ÷ (N √S). S = max(5, ⌈5 × (SE₅ ÷ (margin ÷ 3))²⌉).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

from .flyw2_compound import score
from .flyw2_run import NOISY, load_games

MARGINS = {"CX1(a)": 0.084, "CX1(b)": 0.10, "CX2": 15.0, "CX3(a)": 0.185, "CX3(b)": 0.276}
CAP_HOURS = 4.0
WORKERS = 6


def within_var(values: Dict[object, List[float]]) -> List[float]:
    """Per-unit sample variance across seeds (the unit's mean is not kept)."""
    out = []
    for v in values.values():
        if len(v) < 2:
            continue
        m = sum(v) / len(v)
        out.append(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return out


def se_of_mean(vars_: List[float], s: int) -> float:
    n = len(vars_)
    return math.sqrt(sum(vars_)) / (n * math.sqrt(s)) if n else float("nan")


def seeds_needed(se5: float, margin: float) -> int:
    return max(5, math.ceil(5 * (se5 / (margin / 3)) ** 2))


def silent_partial(g) -> float:
    return float(len(g["named"]) == 1 and not g["exhaust_turns"]
                 and g["soft_fire_turn"] is None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    args = ap.parse_args()
    games = load_games(args.smoke, NOISY)
    seeds = sorted({g["seed"] for g in games})
    S5 = len(seeds)
    tests = []
    pen_se = {}
    for flip in NOISY:
        gs = [g for g in games if g["flip"] == flip]
        single = {(g["truths"][0], g["seed"]): g for g in gs if g["kind"] == "single"}
        units = defaultdict(lambda: defaultdict(list))
        for g in gs:
            if g["kind"] == "single":
                u = g["truths"][0]
                units["ex_single"][u].append(float(g["end"] == "exhausted"))
                units["fire_single"][u].append(float(g["soft_fire_turn"] is not None))
            elif g["kind"] == "KK":
                u = tuple(g["truths"])
                units["ex_KK"][u].append(float(g["end"] == "exhausted"))
                units["fire_KK"][u].append(float(g["soft_fire_turn"] is not None))
                units["partial"][u].append(float(len(g["named"]) == 1))
                units["silent"][u].append(silent_partial(g))
                a, b = (single[(n, g["seed"])] for n in g["truths"])
                units["penalty"][u].append(
                    score(g, 10, 50)["cost"]
                    - (score(a, 10, 50)["cost"] + score(b, 10, 50)["cost"]) / 2)
        se = {k: se_of_mean(within_var(v), S5) for k, v in units.items()}
        for test, s in (("CX1(a)", math.hypot(se["ex_KK"], se["ex_single"])),
                        ("CX1(b)", math.hypot(se["fire_KK"], se["fire_single"])),
                        ("CX3(a)", se["partial"]), ("CX3(b)", se["silent"])):
            tests.append({"test": test, "flip": flip, "se5": round(s, 6),
                          "margin": MARGINS[test],
                          "seeds": seeds_needed(s, MARGINS[test])})
        pen_se[flip] = se["penalty"]
    s = math.sqrt(sum(v ** 2 for v in pen_se.values())) / 2
    tests.append({"test": "CX2", "flip": "pooled", "se5": round(s, 6),
                  "margin": MARGINS["CX2"], "seeds": seeds_needed(s, MARGINS["CX2"])})

    # Timing: core-seconds per (flip, seed) from the smoke's own jobs.
    import gzip
    secs = defaultdict(float)
    jdir = os.path.join(args.smoke, "jobs")
    for fn in os.listdir(jdir):
        if fn.endswith(".json.gz"):
            with gzip.open(os.path.join(jdir, fn), "rt") as f:
                d = json.load(f)
            secs[(d["job"]["flip"], d["job"]["seed"])] += d["secs"]
    per_fs = sum(secs.values()) / len(secs)
    cap = int(CAP_HOURS * 3600 * WORKERS / (per_fs * len(NOISY)))
    need = max(t["seeds"] for t in tests)
    chosen = min(need, cap)
    out = {"smoke_seeds": S5, "tests": tests, "seeds_needed": need,
           "core_seconds_per_flip_seed": round(per_fs, 1),
           "cap_seeds_at_{}h_{}w".format(CAP_HOURS, WORKERS): cap,
           "seeds_run": chosen, "cap_binds": need > cap,
           "projected_wall_hours": round(chosen * len(NOISY) * per_fs / WORKERS / 3600, 2)}
    with open(os.path.join(args.smoke, "seed_rule.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
