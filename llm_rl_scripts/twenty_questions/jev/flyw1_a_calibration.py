"""FLYW-1 pre-registration calibration: arm A (the control) at flip 0.

    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.flyw1_a_calibration \
        --maps llm_rl_scripts/twenty_questions/jev/results/flyw1_class_maps_20261118 \
        --out llm_rl_scripts/twenty_questions/jev/results/flyw1_a_calibration_20261118

**What this is.** The noise-free bases in ``FLYW1_DECISION_PREREG.md`` were first
taken from the identity tree, which assumes A narrows to one object before it guesses.
P0a can guess earlier, and the prereg scores A on its *first by-name guess in the
secret's class*. So A's cost is measured instead of modelled: the 158 games of the
control arm at flip 0, which are deterministic. Only arm A is played; nothing is learned
about B, C or D. The prereg discloses this run.

**A's settings** are the prereg's: P0a (``modeswitch_guard``, K = 3), ``no_repeat``,
closed world (catch-all prior 0), confidence 0.9, eps 0.05, 20 probes, ``always`` end
rule, matrix oracle at flip 0.

**Per map**, from the same 158 games: A's probes to decide (questions and guesses up to
and including its first guess in the secret's class; all probes if it never makes one),
wrong actions (earlier guesses in other classes), A-strict probes (the whole game) and
the share decided. The bases for B, C and D are the class maps' noise-free ``*_cost``
measures (questions + actions, failed actions on unresolvable objects included).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List, Sequence

from .bed import GUESS, load_hypotheses, matrix_oracle
from .openworld import run_openworld_episode
from .stress_harness import require_hash_seed

MATRIX = os.path.join(os.path.dirname(__file__), "matrix_jev.json")
SETTINGS = dict(p_unknown_prior=0.0, max_questions=20, confidence=0.9, eps=0.05,
                backend="modeswitch_guard", tolerance=3, no_repeat=True, end_rule="always")


def play_a(matrix: Dict[str, Dict[str, float]]) -> List[dict]:
    hyps = load_hypotheses()
    oracle = matrix_oracle(matrix)
    games = []
    for h in hyps:
        ep = run_openworld_episode(h, hyps, matrix, SETTINGS["p_unknown_prior"], oracle,
                                   max_questions=SETTINGS["max_questions"],
                                   confidence=SETTINGS["confidence"], eps=SETTINGS["eps"],
                                   backend=SETTINGS["backend"],
                                   tolerance=SETTINGS["tolerance"],
                                   no_repeat=SETTINGS["no_repeat"],
                                   end_rule=SETTINGS["end_rule"])
        games.append({"secret": h.name, "outcome": ep.outcome, "n_probes": len(ep.steps),
                      "steps": [[s.kind, s.qid.split(":", 1)[1], s.answer_yes]
                                for s in ep.steps]})
    return games


def score_a(games: Sequence[dict], cls: Dict[str, int]) -> dict:
    probes, wrong, decided, strict = [], [], 0, []
    for g in games:
        target = cls[g["secret"]]
        n_wrong, done = 0, None
        for i, (kind, key, _ans) in enumerate(g["steps"], start=1):
            if kind != GUESS:
                continue
            if cls[key] == target:
                done = i
                break
            n_wrong += 1
        probes.append(done if done is not None else g["n_probes"])
        wrong.append(n_wrong)
        decided += done is not None
        strict.append(g["n_probes"])
    n = len(games)
    return {"a_probes": round(statistics.mean(probes), 4),
            "a_wrong_per_game": round(sum(wrong) / n, 4),
            "a_decided": round(decided / n, 4),
            "a_strict_probes": round(statistics.mean(strict), 4)}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def bases(rows: Sequence[dict]) -> dict:
    """The prereg's noise-free bases over a set of maps (``rows`` = index rows with
    the A columns merged in)."""
    def by(c):
        return [r for r in rows if r["alignment_construction"] == c]

    def mean(v):
        return round(statistics.mean(v), 4) if v else None

    out = {"n_maps": len(rows)}
    out["de1a_min_a_minus_b"] = round(min(r["a_probes"] - r["pure_cost"] for r in rows), 4)
    out["de1a_maps_b_above_a_plus_0.1"] = [r["map_id"] for r in rows
                                           if r["pure_cost"] > r["a_probes"] + 0.1]
    for c in ("align", "mix05", "mix15", "rand", "category"):
        v = by(c)
        if v:
            out["saving_b_" + c] = mean([r["a_probes"] - r["pure_cost"] for r in v])
            out["min_saving_b_" + c] = round(min(r["a_probes"] - r["pure_cost"] for r in v), 4)
    syn = [r for r in rows if r["family"] == "synthetic"] if "family" in rows[0] else \
        [r for r in rows if r["alignment_construction"] != "category"]
    out["de2_spearman_alignment_vs_saving"] = round(
        spearman([r["alignment"] for r in syn], [r["a_probes"] - r["pure_cost"] for r in syn]), 4)
    rand = by("rand")
    out["de3a_c_minus_b_rand"] = mean([r["class_cost"] - r["pure_cost"] for r in rand])
    out["de3a_min_c_minus_b_rand"] = round(min(r["class_cost"] - r["pure_cost"] for r in rand), 4)
    cross = by("mix15") + rand
    out["de4a_b_minus_d_cross"] = mean([r["pure_cost"] - r["ec2_cost"] for r in cross])
    out["de4a_b_minus_d_align"] = mean([r["pure_cost"] - r["ec2_cost"] for r in by("align")])
    out["de4b_rand_maps_d_le_c"] = sum(r["ec2_cost"] <= r["class_cost"] for r in rand)
    best = [min(r["pure_cost"], r["class_cost"], r["ec2_cost"]) for r in rows]
    out["de5_mean_b_minus_best"] = mean([r["pure_cost"] - b for r, b in zip(rows, best)])
    gb = sum(r["a_probes"] - r["pure_cost"] for r in rows)
    gbest = sum(r["a_probes"] - b for r, b in zip(rows, best))
    out["de5_ratio_reported"] = round(gb / gbest, 4) if gbest else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--maps", required=True, help="class-map result dir (index.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrix", default=MATRIX)
    a = ap.parse_args()
    hash_seed = require_hash_seed()
    with open(a.matrix) as f:
        matrix = json.load(f)["matrix"]
    with open(os.path.join(a.maps, "index.json")) as f:
        index = json.load(f)
    games = play_a(matrix)
    rows = []
    for r in index["maps"]:
        with open(os.path.join(a.maps, "maps", r["map_id"] + ".json")) as f:
            mp = json.load(f)
        assert mp["assignment_sha256_12"] == r["assignment_sha256_12"]
        rows.append(dict(r, family=mp["family"], **score_a(games, mp["classes"])))
    trimmed = [r for r in rows if r["seed"] in (None, 1)]
    os.makedirs(a.out, exist_ok=True)
    summary = {"what": "FLYW-1 prereg calibration: arm A only, flip 0, 158 games",
               "settings": SETTINGS, "pythonhashseed": hash_seed,
               "maps_dir": a.maps, "maps_generator_version": index["generator_version"],
               "a_outcomes": {o: sum(g["outcome"] == o for g in games)
                              for o in sorted({g["outcome"] for g in games})},
               "a_strict_mean_probes": round(statistics.mean(g["n_probes"] for g in games), 4),
               "bases_full": bases(rows), "bases_trimmed": bases(trimmed),
               "per_map": [{k: r[k] for k in ("map_id", "alignment", "a_probes",
                                              "a_wrong_per_game", "a_decided",
                                              "a_strict_probes", "pure_cost", "class_cost",
                                              "ec2_cost")} for r in rows]}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    with open(os.path.join(a.out, "a_games.json"), "w") as f:
        json.dump(games, f)
        f.write("\n")
    print(json.dumps({k: summary[k] for k in ("a_outcomes", "a_strict_mean_probes",
                                              "bases_full", "bases_trimmed")}, indent=1))


if __name__ == "__main__":
    main()
