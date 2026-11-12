"""ST-1 post-hoc analyses. NOT part of the pre-registered scoring (``analyze_st1``).

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \\
        llm_rl_scripts.twenty_questions.jev.analyze_st1_posthoc --dir RUN_DIR [--trace]

1. ``by_n``: in-pool won / safe / wrong and pooled novelty AUC against pool size N,
   one line per crowding level. The registered figures put crowding on the x axis,
   but in this grid crowding rises with N (a fixed 44-attribute space fills up), so
   N is the variable that separates the cells. Writes ``posthoc_by_n.json`` and
   ``fig_st1_posthoc_by_n.png``.
2. ``--trace``: the trajectory read that ``STRESS_CROWDING_PREREG.md`` (*What would
   falsify*) requires when P2 fails on `wrong` games with < 2 live knowns at turn 20.
   Replays every such flip-0 game of the discrete backends (games are deterministic
   given the job record) and classifies its last turn:
     * ``narrowed_on_turn20``: two or more knowns were live before question 20 and its
       answer left one -- budget exhaustion, counted one question late;
     * ``guess_refuted_turn20``: question 20 was a by-name guess that came back "no";
       ``revived`` marks games in which ``modeswitch`` reinstated hypotheses.
   Writes ``posthoc_trace.json``.
3. ``replicate_ranges``: min and max over each cell's 3 replicate pools of the in-pool
   won / safe / wrong rates. Writes ``posthoc_replicates.json``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random

LEVELS = ("twins", "crowded", "real", "floor")
SIZES = (158, 500, 1000, 2000, 5000)
WRONG = 2


def _cell(cells, n, level, flip, arm, backend):
    return cells.get("{}|{}|{}|{}|{}".format(
        n, "realmatrix" if n == 158 else level, flip, arm, backend))


def by_n(run_dir: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = json.load(open(os.path.join(run_dir, "analysis.json")))["cells"]
    out = {}
    panels = [("primary", "modeswitch"), ("norepeat", "modeswitch"), ("primary", "soft")]
    flips = ("0.0", "0.1")
    fig, axes = plt.subplots(len(flips) + 1, len(panels), figsize=(14, 11), sharex=True,
                             sharey="row")
    colors = {"won": "tab:green", "safe": "tab:blue", "wrong": "tab:red"}
    styles = dict(zip(LEVELS, ("-", "--", "-.", ":")))
    for col, (arm, b) in enumerate(panels):
        for row, flip in enumerate(flips):
            ax = axes[row][col]
            for lvl in LEVELS:
                pts = [(n, _cell(cells, n, lvl, flip, arm, b)) for n in SIZES]
                pts = [(n, c) for n, c in pts if c is not None]
                for m in ("won", "safe", "wrong"):
                    ax.plot([n for n, _ in pts], [c[m] for _, c in pts], styles[lvl],
                            color=colors[m], marker="o", ms=3,
                            label="{} ({})".format(m, lvl) if col == 0 and row == 0 else None)
                out["{}|{}|{}|{}".format(arm, b, flip, lvl)] = {
                    str(n): {k: c[k] for k in ("won", "safe", "wrong", "auc", "crowding",
                                               "repeat_rate")} for n, c in pts}
            ax.set_title("{} ({}), flips {:.0%}".format(b, arm, float(flip)), fontsize=10)
            ax.set_xscale("log")
            if col == 0:
                ax.set_ylabel("share of in-pool games")
        ax = axes[-1][col]
        for lvl in LEVELS:
            pts = [(n, _cell(cells, n, lvl, "0.0", arm, b)) for n in SIZES]
            pts = [(n, c) for n, c in pts if c is not None]
            ax.plot([n for n, _ in pts], [c["auc"] for _, c in pts], styles[lvl],
                    color="black", marker="o", ms=3, label=lvl)
        ax.axhline(0.5, color="grey", lw=0.6)
        ax.set_title("novelty AUC, {} ({}), flips 0%".format(b, arm), fontsize=10)
        ax.set_xlabel("pool size N (log)")
        if col == 0:
            ax.set_ylabel("pooled AUC")
    axes[0][0].legend(fontsize=7, ncol=2, loc="center right")
    axes[-1][0].legend(fontsize=7, title="crowding level", title_fontsize=7)
    fig.text(0.01, 0.005,
             "Source: analyze_st1_posthoc on {} (post-hoc, not registered).\n"
             "Each point is one cell pooled over its replicate pools and noise seeds; "
             "line style = crowding level (N = 158 is the real matrix).".format(run_dir),
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(run_dir, "fig_st1_posthoc_by_n.png"), dpi=110)
    with open(os.path.join(run_dir, "posthoc_by_n.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def replicate_ranges(run_dir: str) -> dict:
    """Min and max over a cell's 3 replicate pools (noise seeds pooled within a
    pool) of the in-pool won / safe / wrong rates: the spread to quote with a
    cell's pooled value. Writes ``posthoc_replicates.json``."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))
    for path in glob.glob(os.path.join(run_dir, "jobs", "*.json.gz")):
        rec = json.load(gzip.open(path, "rt"))
        j, g = rec["job"], rec["games"]
        a = acc["{}|{}|{}|{}|{}".format(j["n"], j["level"], j["flip"], j["arm"],
                                        j["backend"])][j["replicate"]]
        for i, o in enumerate(g["outcome"]):
            if g["in_pool"][i]:
                a[0] += 1
                a[1] += o == 0
                a[2] += o == 1
                a[3] += o == WRONG
    out = {}
    for key in sorted(acc):
        reps = [(v[1] / v[0], v[2] / v[0], v[3] / v[0]) for v in acc[key].values()]
        out[key] = {m: [min(r[k] for r in reps), max(r[k] for r in reps)]
                    for k, m in enumerate(("won", "safe", "wrong"))}
        out[key]["replicates"] = len(reps)
    with open(os.path.join(run_dir, "posthoc_replicates.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def trace(run_dir: str) -> dict:
    from .bed import GUESS, matrix_oracle
    from .openworld import run_openworld_episode
    from .stress_fastplan import FastPlanner
    from . import stress_st1 as S

    res = {}
    for arm in ("primary", "norepeat"):
        for b in ("hard", "modeswitch"):
            counts = collections.Counter()
            pattern = os.path.join(run_dir, "jobs", "*_f0_*_{}_{}.json.gz".format(arm, b))
            for path in sorted(glob.glob(pattern)):
                rec = json.load(gzip.open(path, "rt"))
                j, g = rec["job"], rec["games"]
                idx = [i for i, o in enumerate(g["outcome"])
                       if o == WRONG and g["n_live_final"][i] < 2]
                for i, o in enumerate(g["outcome"]):
                    if o == WRONG:
                        counts["wrong_" + ("in_pool" if g["in_pool"][i] else "novel")] += 1
                if not idx:
                    continue
                pool, mtx, novel = S.build(j, "real")
                rng = random.Random(j["order_seed"])
                n_in = j.get("n_in_pool", S.N_IN_POOL)
                secrets = [rng.choice(pool) for _ in range(n_in)] + list(novel)
                rng.shuffle(secrets)
                fp = FastPlanner(pool, mtx, eps=S.EPS)
                for i in idx:
                    last = []

                    def obs(turn, q, ans, bk, before):
                        last[:] = [q.kind, len(before), len(bk.survivors),
                                   getattr(bk, "n_revives", 0)]
                    ep = run_openworld_episode(
                        secrets[i], pool, mtx, len(novel) / len(secrets), matrix_oracle(mtx),
                        max_questions=S.BUDGET, confidence=S.CONFIDENCE, eps=S.EPS,
                        backend=b, tolerance=S.K, no_repeat=j["no_repeat"], observer=obs,
                        planner=fp)
                    assert ep.outcome == "wrong" and ep.n_live_final == g["n_live_final"][i]
                    kind, before, after, revives = last
                    cls = ("narrowed_on_turn20" if before >= 2 and after == 1
                           else "guess_refuted_turn20" if kind == GUESS else "other")
                    counts["{}|{}|{}".format("in_pool" if ep.in_pool else "novel", cls,
                                             "revived" if revives else "no_revive")] += 1
            res["{}/{}".format(arm, b)] = dict(sorted(counts.items()))
    with open(os.path.join(run_dir, "posthoc_trace.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    by_n(a.dir)
    print("wrote posthoc_by_n.json, fig_st1_posthoc_by_n.png")
    replicate_ranges(a.dir)
    print("wrote posthoc_replicates.json")
    if a.trace:
        print(json.dumps(trace(a.dir), indent=1))


if __name__ == "__main__":
    main()
