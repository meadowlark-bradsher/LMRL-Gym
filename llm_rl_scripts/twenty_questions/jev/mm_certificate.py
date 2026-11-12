"""Missing mass: point estimate vs anytime-valid betting certificate, multi-seed.

For each Zipf stream seed and each (backend, flip-rate) cell, play the open-world
accumulation stream (``openworld.run_accumulation``) and save one row per game,
with the TRUE missing mass ``theta`` outside ``H_known`` entering that game. Then
compare two ways of declaring "theta <= eps":

* **point estimate** -- stop the first game the estimate is <= eps. Estimates: the
  loop's live Pitman-Yor ``p_new`` (k = |H_known|, n = games) and Good-Turing N1/n
  over the identities the loop actually learned.
* **certificate** -- test-by-betting. ``Y_t = 1`` iff game t's secret is not in
  ``H_known``, scored from the expert-corrected outcome: a win or an in-pool expert
  call (the expert names a known object) is 0; an admission is 1; a game that ends
  unresolved (``wrong`` / ``false_confirm``) is scored 1, which is conservative
  because wealth is decreasing in Y. Wealth ``K_t = prod (1 + lam_s (eps - Y_s))``
  with ``lam_s`` predictable in ``[0, CAP/(1-eps)]``. ``theta`` never increases (H_known
  only grows), so under H0: theta_t >= eps every earlier factor has conditional mean
  <= 1, K is a nonnegative supermartingale, and Ville gives
  P(stop with theta > eps) <= alpha. Stop when K >= 1/alpha. The stake is the Kelly
  bet under a forecast p: lam = (eps - p) / (eps (1 - eps)), clipped -- the forecast
  sets the speed, never the validity.

A false stop is a stop at a game whose true theta is > eps. The headline is the
false-stop rate over seeds, point estimate vs certificate, per eps.

Run (deterministic; resumable -- finished cells are skipped)::

    PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python -m \
        llm_rl_scripts.twenty_questions.jev.mm_certificate --run \
        --outdir llm_rl_scripts/twenty_questions/jev/results/mm_certificate_20261112
    ... --analyze --outdir <same>
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics as st
import sys
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

MATRIX = os.path.join(os.path.dirname(__file__), "matrix_jev.json")
GAMES, ZIPF_S = 2000, 1.1
CELLS = [("soft", 0.0), ("hard", 0.05), ("hard", 0.10),
         ("modeswitch", 0.05), ("modeswitch", 0.10)]
TOLERANCE = 3                       # modeswitch K=3, the resolved config (PR #43)
ALPHA = 0.05
EPS = [0.20, 0.10, 0.05, 0.03, 0.02]
CAP = 0.5                           # lam <= CAP/(1-eps): at most half the wealth on Y=1
KEEP = ("game", "true_name", "in_pool", "is_new", "outcome", "n_questions",
        "k_before", "expert_called", "false_positive")


def _cell_path(outdir: str, backend: str, noise: float, seed: int) -> str:
    return os.path.join(outdir, "records",
                        "{}_f{:03d}_s{:03d}.json.gz".format(backend, int(round(noise * 100)), seed))


def run_cell(task: Tuple[str, str, float, int]) -> str:
    outdir, backend, noise, seed = task
    path = _cell_path(outdir, backend, noise, seed)
    if os.path.exists(path):
        return path
    from .bed import load_hypotheses
    from .eval_openworld import _load_matrix
    from .openworld import fit_pitman_yor, make_stream, run_accumulation, zipf_weights

    objs = load_hypotheses()
    matrix = _load_matrix(MATRIX)
    stream, index_rank = make_stream(objs, GAMES, ZIPF_S, seed)
    seen, flags = set(), []
    for idx in stream:                      # §3.11 protocol: PY fit on the realised
        flags.append(objs[idx].name not in seen)   # stream (hindsight; flagged)
        seen.add(objs[idx].name)
    sigma, theta_py, _ = fit_pitman_yor(flags)
    recs = run_accumulation(objs, matrix, stream, sigma, theta_py,
                            oracle_noise=noise, noise_seed=seed if noise > 0 else 0,
                            backend=backend, tolerance=TOLERANCE)
    zw = zipf_weights(len(objs), ZIPF_S)
    z = sum(zw)
    w = {objs[i].name: zw[index_rank[i] - 1] / z for i in range(len(objs))}
    known_mass = 0.0
    rows = []
    for r in recs:
        row = {k: getattr(r, k) for k in KEEP}
        row["theta"] = max(0.0, 1.0 - known_mass)   # true mass outside H_known, entering
        rows.append(row)
        if r.expert_called and not r.in_pool:
            known_mass += w[r.true_name]
    out = {"backend": backend, "oracle_noise": noise, "zipf_seed": seed,
           "noise_seed": seed if noise > 0 else 0, "games": GAMES, "zipf_s": ZIPF_S,
           "tolerance": TOLERANCE if backend == "modeswitch" else None,
           "py_fit": {"sigma": sigma, "theta": theta_py},
           "python": sys.version.split()[0],
           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "records": rows}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #

def series(cell: dict) -> List[dict]:
    """Per game: true theta, observed Y, and forecasts built only from the past."""
    from .openworld import py_p_new
    sg, th = cell["py_fit"]["sigma"], cell["py_fit"]["theta"]
    counts: Dict[str, int] = {}
    ysum = 0
    out = []
    for g, r in enumerate(cell["records"]):
        admitted = r["expert_called"] and not r["in_pool"]
        resolved = r["outcome"] == "won" or (r["expert_called"] and r["in_pool"])
        y = 0 if resolved else 1          # admission -> 1; unresolved -> 1 (conservative)
        n1 = sum(1 for c in counts.values() if c == 1)
        k = r["k_before"]
        out.append({
            "g": g, "theta": r["theta"], "y": y,
            "gt": n1 / g if g else 1.0,
            "py": py_p_new(k, g, sg, th) if k else 1.0,
            "rate": (ysum + 1) / (g + 2),
        })
        ysum += y
        if resolved or admitted:          # identities the loop actually learned
            counts[r["true_name"]] = counts.get(r["true_name"], 0) + 1
    return out


def point_stop(T: List[dict], key: str, eps: float) -> Optional[int]:
    return next((t["g"] for t in T if t["g"] >= 1 and t[key] <= eps), None)


def cert_stop(T: List[dict], key: str, eps: float) -> Optional[int]:
    lam_max = CAP / (1 - eps)
    logk, thr = 0.0, math.log(1 / ALPHA)
    for t in T:
        p = t[key]
        lam = 0.0 if p >= eps else min((eps - p) / (eps * (1 - eps)), lam_max)
        logk += math.log1p(lam * (eps - t["y"]))
        if logk >= thr:
            return t["g"] + 1             # certified entering the next game
    return None


def _theta_at(T: List[dict], g: Optional[int]) -> Optional[float]:
    if g is None or g >= len(T):
        return None
    return T[g]["theta"]


RULES = [("point", "py"), ("point", "gt"), ("cert", "py"), ("cert", "gt"), ("cert", "rate")]


def analyze(outdir: str) -> dict:
    by_cell: Dict[Tuple[str, float], List[dict]] = {}
    for fn in sorted(os.listdir(os.path.join(outdir, "records"))):
        if not fn.endswith(".json.gz"):
            continue
        with gzip.open(os.path.join(outdir, "records", fn), "rt") as f:
            c = json.load(f)
        by_cell.setdefault((c["backend"], c["oracle_noise"]), []).append(c)

    summary = {"config": {"games": GAMES, "zipf_s": ZIPF_S, "alpha": ALPHA, "eps": EPS,
                          "stake_cap": CAP, "tolerance": TOLERANCE,
                          "cells": [list(c) for c in CELLS]}, "cells": []}
    for (bk, nz) in [c for c in CELLS if c in by_cell]:
        cells = by_cell[(bk, nz)]
        Ts = [series(c) for c in cells]
        row = {"backend": bk, "oracle_noise": nz, "n_seeds": len(cells),
               "seeds": [c["zipf_seed"] for c in cells],
               "false_positive_calls_mean": st.mean(
                   sum(r["false_positive"] for r in c["records"]) for c in cells),
               "unresolved_mean": st.mean(
                   sum(r["outcome"] in ("wrong", "false_confirm") for r in c["records"])
                   for c in cells),
               "final_theta_mean": st.mean(T[-1]["theta"] for T in Ts),
               "by_eps": {}}
        for eps in EPS:
            truth = [next((t["g"] for t in T if t["theta"] <= eps), None) for T in Ts]
            e = {"truth_reached": sum(x is not None for x in truth),
                 "truth_game_median": (st.median([x for x in truth if x is not None])
                                       if any(x is not None for x in truth) else None)}
            for kind, key in RULES:
                stops = [(point_stop if kind == "point" else cert_stop)(T, key, eps) for T in Ts]
                stopped = [(g, T, tr) for g, T, tr in zip(stops, Ts, truth)
                           if g is not None and g < len(T)]
                false = sum(1 for g, T, _ in stopped if _theta_at(T, g) > eps)
                lags = [g - tr for g, T, tr in stopped if tr is not None]
                e["{}_{}".format(kind, key)] = {
                    "stopped": len(stopped), "false_stops": false,
                    "false_stop_rate": false / len(cells),
                    "false_given_stopped": false / len(stopped) if stopped else None,
                    "lag_median": st.median(lags) if lags else None,
                }
            row["by_eps"][str(eps)] = e
        summary["cells"].append(row)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _print(summary)
    return summary


def _print(s: dict) -> None:
    print("alpha={}  stake cap {}/(1-eps)  false-stop rate = seeds stopping at true theta > eps / all seeds"
          .format(ALPHA, CAP))
    for row in s["cells"]:
        print("\n== {} @ {:.0%} flips  ({} seeds; false escalations {:.1f}/stream, unresolved {:.1f}, "
              "final theta {:.4f})".format(row["backend"], row["oracle_noise"], row["n_seeds"],
                                           row["false_positive_calls_mean"], row["unresolved_mean"],
                                           row["final_theta_mean"]))
        print("  eps   | truth reached | point PY     point GT     | cert PY-stake  cert GT-stake  cert rate-stake")
        for eps, e in row["by_eps"].items():
            def f(k):
                d = e[k]
                return "{:>2}/{:<2}({:>2} st)".format(d["false_stops"], row["n_seeds"], d["stopped"])
            print("  {:<5} | {:>3} med {:>6} | {}  {} | {}  {}  {}".format(
                eps, e["truth_reached"], str(e["truth_game_median"]),
                f("point_py"), f("point_gt"), f("cert_py"), f("cert_gt"), f("cert_rate")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    if args.run:
        if os.environ.get("PYTHONHASHSEED") in (None, "random"):
            raise SystemExit("set PYTHONHASHSEED=0 (survivor backends iterate name sets)")
        seeds = range(args.seed_start, args.seed_start + args.seeds)
        tasks = [(args.outdir, bk, nz, s) for s in seeds for bk, nz in CELLS]
        tasks = [t for t in tasks if not os.path.exists(_cell_path(*t))]
        print("{} cells to run on {} workers".format(len(tasks), args.workers), flush=True)
        with Pool(args.workers) as pool:
            for i, p in enumerate(pool.imap_unordered(run_cell, tasks), 1):
                print("[{}/{}] {}".format(i, len(tasks), os.path.basename(p)), flush=True)
    if args.analyze:
        analyze(args.outdir)


if __name__ == "__main__":
    main()
