"""Score the ST-3 stream-K follow-up (``STRESS_ST3_STREAMK_PREREG.md``, *Scoring*).

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_st3_streamk \
        --run-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_streamk_<stamp> \
        --base-dir llm_rl_scripts/twenty_questions/jev/results/stress_st3_20261113

Every stream-K job ``s<σ>_stream_<rest>`` pairs with the ST-3 job
``s<σ>_true_<rest>`` in the base run: the same stream, the same objects, the same
oracle seed and backend. Only K in the prior differs (the stream's distinct
species so far, against |H_known|). Writes ``analysis.json`` and one figure into
the run directory. Populations and rates are ``analyze_st3``'s; the code is the
reference when the prereg's wording is ambiguous.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, List

from .analyze_st3 import EXPERT, WRONG, Pool_, rates

N_REAL = 158
ARMS = [("primary", "soft"), ("primary", "modeswitch_guard"), ("norepeat", "modeswitch_guard")]
SIGMAS = [0.0, 0.25, 0.5, 0.75]
FLIPS = [0.0, 0.05]
# SK1: soft's TPR gain from the stream prior, sigma in {0.5, 0.75}, flip 0.
SK1_SMALL = 0.05          # every gain <= this: the near-copy likelihood is the cause
SK1_LARGE = 0.15          # any gain >= this: self-censoring is a material cause
# SK2: norepeat P0a under the stream prior, sigma in {0.5, 0.75}, flip 0.
SK2_WRONG_FLOOR = 0.05    # wrong rate at sigma = 0.75 stays >= this
# SK3: published P0a at sigma = 0.75 (it already catches every novel object).
SK3_TOL = 0.01


def _load(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def tail_tpr(cols: dict, idx: List[int]) -> dict:
    """TPR on novel games whose secret is a generated child (species >= N_REAL)."""
    nov = [i for i in idx if not cols["in_pool"][i] and cols["obj"][i] >= N_REAL]
    hit = sum(1 for i in nov if cols["outcome"][i] == EXPERT)
    return {"novel_tail": len(nov), "tpr_tail": hit / len(nov) if nov else float("nan")}


def metrics(p: Pool_) -> dict:
    allg = list(range(len(p.g)))
    out = {"all": rates(p.cols, allg),
           "first_half": rates(p.cols, p.idx(lambda g, i: g < 2500)),
           "second_half": rates(p.cols, p.idx(lambda g, i: g >= 2500))}
    out["all"].update(tail_tpr(p.cols, allg))
    last = p.idx(lambda g, i: g >= 4750)
    out["mean_prior_last_window"] = (sum(p.cols["prior"][i] for i in last) / len(last)
                                     if last else float("nan"))
    out["admitted"] = sum(1 for i in allg if not p.cols["in_pool"][i]
                          and p.cols["outcome"][i] == EXPERT)
    return out


def _job_delta(a: dict, b: dict) -> Dict[str, float]:
    """Per-job (stream - true) TPR and wrong rate."""
    ra, rb = (rates(r["games"], list(range(len(r["games"]["obj"])))) for r in (a, b))
    return {"tpr": ra["tpr"] - rb["tpr"], "wrong": ra["wrong_rate"] - rb["wrong_rate"]}


def analyze(run_dir: str, base_dir: str, make_figures: bool = True) -> dict:
    stream_pools: Dict[tuple, Pool_] = defaultdict(Pool_)
    true_pools: Dict[tuple, Pool_] = defaultdict(Pool_)
    deltas: Dict[tuple, List[dict]] = defaultdict(list)
    identical0, n0, n_jobs, missing = 0, 0, 0, []
    for path in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        rec = _load(path)
        j = rec["job"]
        n_jobs += 1
        base_id = j["job_id"].replace("_stream_", "_true_", 1)
        bpath = os.path.join(base_dir, "jobs", base_id + ".json.gz")
        if not os.path.exists(bpath):
            missing.append(base_id)
            continue
        base = _load(bpath)
        assert base["job"]["oracle_seed"] == j["oracle_seed"], base_id
        assert base["games"]["obj"] == rec["games"]["obj"], base_id
        key = (j["sigma"], j["flip"], j["arm"], j["backend"])
        stream_pools[key].add(rec)
        true_pools[key].add(base)
        deltas[key].append(_job_delta(rec, base))
        if j["sigma"] == 0.0:
            n0 += 1
            identical0 += rec["games"] == base["games"]

    cells = []
    C: Dict[tuple, dict] = {}
    for key in sorted(stream_pools):
        s, t = metrics(stream_pools[key]), metrics(true_pools[key])
        d = deltas[key]
        row = {"sigma": key[0], "flip": key[1], "arm": key[2], "backend": key[3],
               "stream": s, "true": t,
               "delta_all": {m: s["all"][m] - t["all"][m]
                             for m in ("tpr", "fpr", "wrong_rate", "autonomous",
                                       "auc", "tpr_tail")},
               "job_delta_range": {m: [min(x[m] for x in d), max(x[m] for x in d)]
                                   for m in ("tpr", "wrong")},
               "n_jobs": len(d)}
        cells.append(row)
        C[key] = row

    complete = n_jobs == 108 and not missing
    v = score(C, identical0, n0)
    out = {"schema": "twenty_questions_stress_st3_streamk_analysis_v1",
           "run_dir": run_dir, "base_dir": base_dir, "n_jobs": n_jobs,
           "missing_base_jobs": missing, "complete_grid": complete,
           "sigma0_identical_jobs": [identical0, n0], "cells": cells,
           "verdicts": v if complete else {},
           "verdicts_partial_grid": {} if complete else v}
    if make_figures:
        out["figures"] = [os.path.basename(figure(C, run_dir, base_dir))]
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def score(C: Dict[tuple, dict], identical0: int, n0: int) -> dict:
    def c(sg, fl, arm, bk):
        return C.get((sg, fl, arm, bk))

    v: Dict[str, dict] = {}
    # SK0 (mechanical gate): at sigma = 0 the PY prior ignores K.
    v["SK0"] = {"verdict": "PASS" if n0 == 27 and identical0 == n0 else "FAIL",
                "identical": identical0, "jobs": n0}

    # SK1: classify soft's TPR gain; registered expectation "likelihood".
    gains = {}
    for sg in (0.5, 0.75):
        m = c(sg, 0.0, "primary", "soft")
        gains["s{:g}".format(sg)] = m["delta_all"]["tpr"] if m else float("nan")
    gs = list(gains.values())
    if any(g != g for g in gs):
        reading = "incomplete"
    elif all(g <= SK1_SMALL for g in gs):
        reading = "likelihood"
    elif any(g >= SK1_LARGE for g in gs):
        reading = "self-censoring"
    else:
        reading = "mixed"
    v["SK1"] = {"verdict": "PASS" if reading == "likelihood" else "FAIL",
                "reading": reading, "tpr_gain": gains}

    # SK2: norepeat P0a moves along the escalate-vs-wrong knob, and stays unsafe.
    parts = {}
    for sg in (0.5, 0.75):
        m = c(sg, 0.0, "norepeat", "modeswitch_guard")
        if m is None:
            parts["s{:g}".format(sg)] = False
            continue
        d = m["delta_all"]
        parts["s{:g}_tpr_up".format(sg)] = d["tpr"] >= 0
        parts["s{:g}_fpr_up".format(sg)] = d["fpr"] >= 0
        parts["s{:g}_wrong_down".format(sg)] = d["wrong_rate"] <= 0
    m = c(0.75, 0.0, "norepeat", "modeswitch_guard")
    parts["s0.75_wrong_floor"] = bool(m) and m["stream"]["all"]["wrong_rate"] >= SK2_WRONG_FLOOR
    v["SK2"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts}

    # SK3 (control): published P0a at sigma = 0.75 already admits every novel object.
    m = c(0.75, 0.0, "primary", "modeswitch_guard")
    parts = {}
    for k in ("tpr", "fpr", "autonomous"):
        parts[k] = bool(m) and abs(m["delta_all"][k]) <= SK3_TOL
    v["SK3"] = {"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts}
    return v


def figure(C: Dict[tuple, dict], run_dir: str, base_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ARMS), figsize=(4 * len(ARMS), 3.2), sharey=True)
    for ax, (arm, bk) in zip(axes, ARMS):
        for prior, ls in (("stream", "-"), ("true", "--")):
            xs = [sg for sg in SIGMAS if (sg, 0.0, arm, bk) in C]
            for metric, col, lab in (("tpr", "C0", "TPR"), ("wrong_rate", "C3", "wrong"),
                                     ("fpr", "C2", "FPR")):
                ys = [C[(sg, 0.0, arm, bk)][prior]["all"][metric] for sg in xs]
                ax.plot(xs, ys, ls, color=col, marker="o", ms=3,
                        label="{} ({})".format(lab, prior))
        ax.set_title("{} ({})".format(bk, arm), fontsize=8)
        ax.set_xlabel("σ")
        ax.set_xticks(SIGMAS)
    axes[0].set_ylabel("rate (all games; TPR of novel, FPR of in-pool)")
    axes[0].legend(fontsize=6)
    fig.text(0.01, 0.005, "ST-3 stream-K, flip 0, pooled over stream seeds; solid = stream-K "
             "prior, dashed = ST-3 true prior (K = |H_known|). analyze_st3_streamk, {} vs {}".format(
                 run_dir, base_dir), fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(run_dir, "fig_streamk_vs_true.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    a = ap.parse_args()
    out = analyze(a.run_dir, a.base_dir)
    print("{} jobs, complete grid: {}, sigma=0 identical {}/{}".format(
        out["n_jobs"], out["complete_grid"], *out["sigma0_identical_jobs"]))
    for k, x in (out["verdicts"] or out["verdicts_partial_grid"]).items():
        print("  {:5s} {}{}".format(k, x["verdict"],
                                     "  ({})".format(x["reading"]) if "reading" in x else ""))
    print("wrote {}".format(os.path.join(a.run_dir, "analysis.json")))


if __name__ == "__main__":
    main()
