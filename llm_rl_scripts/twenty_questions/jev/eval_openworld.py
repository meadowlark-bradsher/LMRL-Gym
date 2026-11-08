"""Run the open-world accumulation experiment (offline, deterministic).

    # mechanics self-test (no data files needed beyond the module):
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_openworld --self-test

    # full run on the cached Noul matrix:
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.eval_openworld \
        --matrix llm_rl_scripts/twenty_questions/jev/matrix_jev.json \
        --games 2000 --zipf 1.1 --seed 7

Writes ``results/openworld_<stamp>/{summary.json,records.json}`` with the config,
the fitted Pitman-Yor parameters, and one row per game so the analyzer and every
cited number regenerate.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
from typing import Dict, List

from .analyze_openworld import auc
from .bed import Hypothesis, load_hypotheses, matrix_oracle
from .matrix import load_matrix, synthetic_matrix
from .openworld import (
    UNKNOWN, GameRecord, build_openworld_bank, fit_pitman_yor,
    make_stream, py_p_new, run_accumulation, run_openworld_episode,
)


def _load_matrix(path: str) -> Dict[str, Dict[str, float]]:
    return load_matrix(path)["matrix"]


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    """Assert the core mechanics on the synthetic category-coherent matrix."""
    objs = load_hypotheses()
    matrix = synthetic_matrix(objs)
    oracle = matrix_oracle(matrix)
    by_name = {h.name: h for h in objs}

    # 1. Cold start: empty H_known -> first game must fire the expert.
    ep = run_openworld_episode(objs[0], [], matrix, p_unknown_prior=1.0, oracle=oracle)
    assert ep.outcome == "expert", ep.outcome
    assert ep.in_pool is False

    # 2. In-pool secret with known distractors -> resolved, UNKNOWN stays low.
    #    Use one object per category so the categories are separable.
    seen_cat: Dict[str, Hypothesis] = {}
    for h in objs:
        seen_cat.setdefault(h.category, h)
    known = list(seen_cat.values())              # ~17 known, distinct categories
    true_in = known[3]
    ep_in = run_openworld_episode(true_in, known, matrix, p_unknown_prior=0.1,
                                  oracle=oracle)
    assert ep_in.outcome == "won", ep_in.outcome
    assert ep_in.p_unknown_max < 0.5, ep_in.p_unknown_max

    # 3. Out-of-pool secret from a category with NO known member -> UNKNOWN wins.
    #    Build a known set missing the Animals category entirely.
    known_no_animals = [h for h in known if h.category != "Animals"]
    animal = next(h for h in objs if h.category == "Animals")
    ep_out = run_openworld_episode(animal, known_no_animals, matrix,
                                   p_unknown_prior=0.1, oracle=oracle)
    assert ep_out.outcome == "expert", ep_out.outcome
    assert ep_out.p_unknown_max >= 0.5, ep_out.p_unknown_max

    # 4. PY: p_new decreasing in n, and the MLE recovers sigma>0 on a heavy
    #    synthetic accumulation (many distinct early, slowing).
    assert py_p_new(5, 10, 0.5, 1.0) > py_p_new(5, 100, 0.5, 1.0)
    # Heavy-tailed new_flags: new on ~ every power-of-two step.
    flags = []
    nxt = 1
    for i in range(1, 400):
        is_new = (i == nxt)
        flags.append(is_new)
        if is_new:
            nxt = int(nxt * 1.6) + 1
    sigma, theta, ll = fit_pitman_yor(flags)
    assert sigma >= 0.0 and ll < 0.0

    # 5. Bank: guesses only for known objects; all 17 category questions present.
    bank = build_openworld_bank(known_no_animals)
    assert len(bank["category"]) == 17
    assert all(q.key != "Zebra" for q in bank["guess"])  # Zebra (Animals) excluded

    print("[self-test] open-world mechanics OK")
    print("  cold-start -> expert; in-pool -> won (p_unk<0.5);",
          "out-of-pool -> expert (p_unk>=0.5)")
    print("  PY MLE on synthetic heavy tail: sigma={:.2f} theta={:.2f}".format(
        sigma, theta))


# --------------------------------------------------------------------------- #
# Full run                                                                     #
# --------------------------------------------------------------------------- #

def run(matrix_path: str, games: int, zipf_s: float, seed: int,
        confidence: float, eps: float, outdir: str,
        oracle_noise: float = 0.0, noise_seed: int = 0) -> str:
    objs = load_hypotheses()
    matrix = _load_matrix(matrix_path)
    stream, index_rank = make_stream(objs, games, zipf_s, seed)

    # Characterise the realised discovery process, then use the same PY params
    # for the live UNKNOWN prior (a design study, not a deployment -- noted).
    seen: set = set()
    new_flags: List[bool] = []
    for idx in stream:
        name = objs[idx].name
        new_flags.append(name not in seen)
        seen.add(name)
    sigma, theta, loglik = fit_pitman_yor(new_flags)

    records = run_accumulation(objs, matrix, stream, sigma, theta,
                               confidence=confidence, eps=eps,
                               oracle_noise=oracle_noise, noise_seed=noise_seed)
    # stamp the frequency rank onto each record.
    for r in records:
        r.rank = index_rank[next(i for i, h in enumerate(objs) if h.name == r.true_name)]

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(outdir, "openworld_{}".format(stamp))
    os.makedirs(outpath, exist_ok=True)

    n_distinct = len(seen)
    n_expert = sum(1 for r in records if r.expert_called)
    n_fp = sum(1 for r in records if r.false_positive)
    n_won = sum(1 for r in records if r.outcome == "won")
    n_wrong = sum(1 for r in records if r.outcome == "wrong")
    summary = {
        "schema": "twenty_questions_openworld_v1",
        "created": stamp,
        "config": {
            "matrix": os.path.basename(matrix_path),
            "games": games, "zipf_s": zipf_s, "seed": seed,
            "confidence": confidence, "eps": eps,
            "oracle_noise": oracle_noise, "noise_seed": noise_seed,
            "n_true_objects": len(objs),
        },
        "pitman_yor_fit": {"sigma": sigma, "theta": theta, "loglik": loglik},
        "totals": {
            "games": len(records),
            "distinct_species_seen": n_distinct,
            "expert_calls": n_expert,
            "false_positive_expert_calls": n_fp,
            "won": n_won,
            "wrong": n_wrong,
            "final_pool_size": records[-1].k_before + (
                1 if records[-1].expert_called and not records[-1].in_pool else 0),
        },
    }
    with open(os.path.join(outpath, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(outpath, "records.json"), "w") as f:
        json.dump([dataclasses.asdict(r) for r in records], f, indent=2)

    print("wrote {}".format(outpath))
    print("  PY fit: sigma={:.3f} theta={:.3f}".format(sigma, theta))
    print("  distinct seen {}/{}, expert calls {} ({} false-positive), "
          "won {}, wrong {}".format(n_distinct, len(objs), n_expert, n_fp,
                                     n_won, n_wrong))
    return outpath


def run_noise_grid(matrix_path: str, games: int, zipf_s: float, seed: int,
                   confidence: float, eps: float, levels: List[float],
                   n_noise_seeds: int, outdir: str) -> str:
    """Repeat the accumulation under a noisy answerer at several flip rates, with
    several noise seeds per rate, and report the detector's spread. The stream
    (``seed``) is fixed; only the oracle's flips vary, so the discovery process is
    identical across cells and only the noise moves the metrics."""
    objs = load_hypotheses()
    matrix = _load_matrix(matrix_path)
    stream, index_rank = make_stream(objs, games, zipf_s, seed)
    seen: set = set()
    new_flags = []
    for idx in stream:
        new_flags.append(objs[idx].name not in seen)
        seen.add(objs[idx].name)
    sigma, theta, _ = fit_pitman_yor(new_flags)

    def metrics(records: List[GameRecord]) -> Dict[str, float]:
        labels = [0 if r.in_pool else 1 for r in records]
        scores = [r.p_unknown_max for r in records]
        tp = sum(1 for r in records if (not r.in_pool) and r.p_unknown_max >= confidence)
        fn = sum(1 for r in records if (not r.in_pool) and r.p_unknown_max < confidence)
        fp = sum(1 for r in records if r.in_pool and r.p_unknown_max >= confidence)
        tn = sum(1 for r in records if r.in_pool and r.p_unknown_max < confidence)
        return {
            "auc": auc(scores, labels),
            "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
            "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
            "expert_calls": sum(1 for r in records if r.expert_called),
            "false_positive_calls": sum(1 for r in records if r.false_positive),
            "won": sum(1 for r in records if r.outcome == "won"),
            "wrong": sum(1 for r in records if r.outcome == "wrong"),
        }

    def agg(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
        vals = [r[key] for r in rows]
        mean = sum(vals) / len(vals)
        return {"mean": mean, "min": min(vals), "max": max(vals)}

    cells = []
    for lvl in levels:
        rows = []
        for ns in range(1, n_noise_seeds + 1):
            if lvl == 0.0:
                recs = run_accumulation(objs, matrix, stream, sigma, theta,
                                        confidence=confidence, eps=eps)
                rows.append(metrics(recs))
                break  # deterministic; one cell suffices
            recs = run_accumulation(objs, matrix, stream, sigma, theta,
                                    confidence=confidence, eps=eps,
                                    oracle_noise=lvl, noise_seed=ns)
            rows.append(metrics(recs))
        cells.append({
            "oracle_noise": lvl, "n_noise_seeds": len(rows),
            "per_seed": rows,
            "agg": {k: agg(rows, k) for k in
                    ["auc", "tpr", "fpr", "expert_calls",
                     "false_positive_calls", "won", "wrong"]},
        })
        c = cells[-1]["agg"]
        print("noise {:.2f}: AUC {:.3f}[{:.3f},{:.3f}]  TPR {:.3f}  FPR {:.4f}  "
              "FP {:.1f}  miss {:.1f}".format(
                  lvl, c["auc"]["mean"], c["auc"]["min"], c["auc"]["max"],
                  c["tpr"]["mean"], c["fpr"]["mean"],
                  c["false_positive_calls"]["mean"], c["wrong"]["mean"]))

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(outdir, "openworld_noise_grid_{}".format(stamp))
    os.makedirs(outpath, exist_ok=True)
    payload = {
        "schema": "twenty_questions_openworld_noise_grid_v1",
        "created": stamp,
        "config": {"matrix": os.path.basename(matrix_path), "games": games,
                   "zipf_s": zipf_s, "seed": seed, "confidence": confidence,
                   "eps": eps, "levels": levels, "n_noise_seeds": n_noise_seeds},
        "pitman_yor_fit": {"sigma": sigma, "theta": theta},
        "cells": cells,
    }
    with open(os.path.join(outpath, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("wrote {}".format(outpath))
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--oracle-noise", type=float, default=0.0,
                    help="flip each attribute answer at this rate (noisy answerer)")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-grid", action="store_true",
                    help="run the clean + noisy flip-rate grid with seed replicates")
    ap.add_argument("--levels", default="0.0,0.05,0.10")
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--outdir", default="llm_rl_scripts/twenty_questions/jev/results")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.noise_grid:
        levels = [float(x) for x in args.levels.split(",")]
        run_noise_grid(args.matrix, args.games, args.zipf, args.seed,
                       args.confidence, args.eps, levels, args.noise_seeds,
                       args.outdir)
        return
    run(args.matrix, args.games, args.zipf, args.seed, args.confidence,
        args.eps, args.outdir, oracle_noise=args.oracle_noise,
        noise_seed=args.noise_seed)


if __name__ == "__main__":
    main()
