"""How much the survivor tie-break order moves the discrete backends (diagnostic).

The survivor-set backends break ties among equally weighted survivors in a fixed
order (``openworld_backends._in_order``: the ``known_hyps`` order, i.e. admission
order). That order is not neutral: objects are admitted roughly in discovery order,
which on a Zipf stream is roughly popularity order, so ties favour likely secrets.
This probe replays the retraction study's 10%-flip cell for ``hard`` and
``modeswitch@3`` under three orders -- admission (the committed one), alphabetical,
and reversed admission -- and reports the scorecard for each.

    PYTHONHASHSEED=0 PYTHONPATH=. /opt/miniconda3/envs/LLM_RL/bin/python \\
        -m llm_rl_scripts.twenty_questions.jev.tie_order_probe \\
        --out llm_rl_scripts/twenty_questions/jev/results/openworld_tie_order_20261111/probe.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

from . import openworld_backends as ob
from .openworld import run_accumulation
from .stress_regression import _retraction_metrics, _setup

ORDERS = ("admission", "alpha", "reversed")
_admission = ob._in_order


def _play(job):
    order, backend, noise_seed = job
    if order == "alpha":
        ob._in_order = lambda names, m: sorted(n for n in names if n in m)
    elif order == "reversed":
        ob._in_order = lambda names, m: [n for n in reversed(names) if n in m]
    else:
        ob._in_order = _admission
    objs, matrix, stream, _, sigma, theta = _setup(2000, 1.1, 7)
    kw = {"tolerance": 3} if backend == "modeswitch" else {}
    recs = run_accumulation(objs, matrix, stream, sigma, theta, confidence=0.9,
                            eps=0.05, oracle_noise=0.1, noise_seed=noise_seed,
                            backend=backend, **kw)
    return job, _retraction_metrics(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    jobs = list(itertools.product(ORDERS, ["hard", "modeswitch"], [1, 2]))
    with Pool(args.workers) as pool:
        out = pool.map(_play, jobs)
    keys = ("mean_questions", "in_pool_fail_rate", "auc", "wrong", "false_positive_calls")
    agg = {}
    for (order, backend, _), m in out:
        agg.setdefault("{} {}".format(backend, order), []).append(m)
    summary = {k: {m: sum(r[m] for r in rows) / len(rows) for m in keys}
               for k, rows in agg.items()}
    for k, v in summary.items():
        print("{:22s} ".format(k) + "  ".join("{}={:.4f}".format(m, v[m]) for m in keys))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"runtime": {"python": sys.version.split()[0],
                               "pythonhashseed": os.environ.get("PYTHONHASHSEED")},
                   "cell": {"games": 2000, "zipf_s": 1.1, "seed": 7, "oracle_noise": 0.1,
                            "noise_seeds": [1, 2], "modeswitch_tolerance": 3},
                   "mean_over_noise_seeds": summary,
                   "per_seed": [[list(j), m] for j, m in out]}, f, indent=2, sort_keys=True)
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
