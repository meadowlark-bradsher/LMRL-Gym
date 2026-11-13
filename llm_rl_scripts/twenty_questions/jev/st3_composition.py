"""ST-3 exploratory composition split (NOT a registered analysis).

Splits each cell's novel games (true object not in H_known) by

* origin: a real object (species index < N_REAL; ``build`` puts the 158 real objects
  first) or a generator tail child;
* appearance: the species' first appearance in the stream, or a repeat of a species
  the loop never admitted (it only admits on an expert call).

Writes ``composition.json`` next to ``analysis.json``.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.st3_composition --run-dir <run_dir>
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

N_REAL = 158
EXPERT = 1
HALF = 2500


def _r(n, d):
    return n / d if d else None


def compose(run_dir: str) -> dict:
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (cell, half) -> group -> [n, flagged]
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            rec = json.load(f)
        j, g = rec["job"], rec["games"]
        cell = (j["sigma"], j["prior"], j["flip"], j["arm"], j["backend"])
        for i in range(len(g["obj"])):
            if g["in_pool"][i]:
                continue
            half = "first_half" if i < HALF else "second_half"
            flagged = g["outcome"][i] == EXPERT
            groups = ("real" if g["obj"][i] < N_REAL else "tail",
                      "first_appearance" if g["is_new"][i] else "repeat_unadmitted")
            for key in (half, "all"):
                for grp in groups:
                    a = acc[cell + (key,)][grp]
                    a[0] += 1
                    a[1] += flagged
    rows = []
    for k in sorted(acc):
        row = {"sigma": k[0], "prior": k[1], "flip": k[2], "arm": k[3], "backend": k[4],
               "span": k[5]}
        for grp, (n, fl) in sorted(acc[k].items()):
            row[grp] = {"novel_games": n, "tpr": _r(fl, n)}
        rows.append(row)
    out = {"schema": "twenty_questions_stress_st3_composition_v1", "run_dir": run_dir,
           "registered": False, "n_real": N_REAL, "rows": rows}
    with open(os.path.join(run_dir, "composition.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    out = compose(ap.parse_args().run_dir)
    for r in out["rows"]:
        if r["prior"] == "true" and r["flip"] == 0.0 and r["span"] != "all":
            cells = []
            for grp in ("real", "tail", "first_appearance", "repeat_unadmitted"):
                x = r.get(grp)
                if x:
                    cells.append("{} n={} tpr={:.3f}".format(grp, x["novel_games"], x["tpr"]))
            print(r["sigma"], r["arm"], r["backend"], r["span"], " | ".join(cells))


if __name__ == "__main__":
    main()
