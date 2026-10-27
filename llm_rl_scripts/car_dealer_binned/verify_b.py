"""Verify that the frozen Condition B analyses and the rendered note regenerate.

For each frozen result directory: re-run ``analyze_condition_b`` on its logs
into a temporary file and compare with the committed ``analysis.json``; then
re-render ``SCRIPTED_RESULTS.md`` from the committed JSON and compare with the
committed note. With ``--sweep`` the horizon sweep is recomputed too (~3 min)
and compared with the frozen ``sweep.json``.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.verify_b
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

DIRS = [f"results/binned_b_scripted_{n}_20261026" for n in ("r2_easy", "r6_easy", "r2_hard")]
NOTE = "llm_rl_scripts/car_dealer_binned/SCRIPTED_RESULTS.md"
SWEEP = "results/binned_b_sweep_20261026/sweep.json"


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args(argv)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for d in DIRS:
            out = os.path.join(tmp, os.path.basename(d) + ".json")
            run([sys.executable, "-m", "llm_rl_scripts.car_dealer_binned.analyze_condition_b", d, "--seed", "0", "--B", "5000", "--json", out])
            a = json.load(open(out))
            b = json.load(open(os.path.join(d, "analysis.json")))
            a.pop("source", None)
            b.pop("source", None)
            same = a == b
            ok &= same
            print(("PASS" if same else "FAIL"), d, "analysis.json regenerates" if same else "analysis.json DIFFERS")
        note = os.path.join(tmp, "note.md")
        run([sys.executable, "-m", "llm_rl_scripts.car_dealer_binned.render_scripted_note", "--out", note])
        same = open(note).read() == open(NOTE).read()
        ok &= same
        print(("PASS" if same else "FAIL"), NOTE, "renders identically" if same else "DIFFERS from a fresh render")
        if args.sweep:
            sw = os.path.join(tmp, "sweep.json")
            run([sys.executable, "-m", "llm_rl_scripts.car_dealer_binned.sweep_horizon", "--out", sw, "--random-repeats", "20", "--seed", "0"])
            a = json.load(open(sw)); b = json.load(open(SWEEP))
            a.pop("elapsed_s", None); b.pop("elapsed_s", None)
            same = a == b
            ok &= same
            print(("PASS" if same else "FAIL"), SWEEP, "regenerates" if same else "DIFFERS")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
