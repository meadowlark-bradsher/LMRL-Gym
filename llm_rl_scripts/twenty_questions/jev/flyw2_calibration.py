"""FLYW-7 pre-registration calibration: arm S (today's P0a) at flip 0, every game.

    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.flyw2_calibration \
        --out llm_rl_scripts/twenty_questions/jev/results/flyw2_calibration_20261119

**What this is.** The bases for ``FLYW2_COMPOUND_PREREG.md``. At flip 0 every game is
deterministic, so each kind of game is played once per secret or pair:

* 141 in-pool singles, 17 novels (held out by ``holdout_novels``),
* all 9,870 pairs of known objects (KK, the compound games scored),
* all 2,397 known + novel pairs (KN, reported only).

Only flip 0 is played. With exact answers an in-pool single can never be struck, so
"error" escalations do not exist here; the noisy cells, which carry the test, are not
played. The prereg discloses this run.

Writes ``games.json.gz`` (every record) and ``summary.json`` (the bases).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import time
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from .bed import load_hypotheses, matrix_oracle
from .flyw2_compound import (P0A, P_UNKNOWN, KeyedSetOracle, ThreeWay, all_questions,
                             holdout_novels, pair_overlap, pairs_of, play_s, score,
                             score_class, score_diagnostic)
from .stress_harness import require_hash_seed

HERE = os.path.dirname(__file__)
MATRIX = os.path.join(HERE, "matrix_jev.json")
COSTS = [(10, 50), (10, 20), (10, 200), (30, 150), (100, 500)]   # (c_expert, c_wrong)
PRIMARY = (10, 50)
SHARES = (0.1, 0.25, 0.5)
_CAT = {h.name: h.category for h in load_hypotheses()}   # cat17 class map (reported)


def play_all(matrix, hyps):
    novels = holdout_novels(hyps)
    known = [h for h in hyps if h.name not in novels]
    nov_h = [h for h in hyps if h.name in novels]
    oracle = KeyedSetOracle(matrix, 0.0, 0)
    recs: List[dict] = []
    for h in known:
        r = play_s([h], known, matrix, oracle); r["kind"] = "single"; recs.append(r)
    for h in nov_h:
        r = play_s([h], known, matrix, oracle); r["kind"] = "novel"; recs.append(r)
    for a, b in pairs_of(known):
        r = play_s([a, b], known, matrix, oracle); r["kind"] = "KK"; recs.append(r)
    for a in known:
        for b in nov_h:
            r = play_s([a, b], known, matrix, oracle); r["kind"] = "KN"; recs.append(r)
    return known, nov_h, recs


def mean(v):
    v = list(v)
    return round(statistics.mean(v), 4) if v else None


def summarize_kind(rs: Sequence[dict]) -> dict:
    n = len(rs)
    ends = Counter(r["end"] for r in rs)
    named = Counter(min(len(r["named"]), 2) for r in rs)
    out = {"n": n,
           "end": {k: round(v / n, 4) for k, v in sorted(ends.items())},
           "named_none_one_both": [round(named.get(i, 0) / n, 4) for i in range(3)],
           "probes": mean(r["n_probes"] for r in rs),
           "wrong": mean(r["wrong"] for r in rs),
           "any_exhaustion": round(sum(bool(r["exhaust_turns"]) for r in rs) / n, 4),
           "reinstates": mean(r["reinstates"] for r in rs),
           "soft_fires": round(sum(r["soft_fire_turn"] is not None for r in rs) / n, 4),
           "soft_pu_final": mean(r["soft_pu_final"] for r in rs),
           "soft_top_is_truth": round(sum(r["soft_top_final"] in r["truths"]
                                          for r in rs) / n, 4)}
    for ce, cw in COSTS:
        out["cost_{}_{}".format(ce, cw)] = mean(score(r, ce, cw)["cost"] for r in rs)
    out["cost_10_50_unverified"] = mean(score(r, 10, 50, verified=False)["cost"]
                                        for r in rs)
    cat = _CAT
    cs = [score_class(r, cat, 10, 50) for r in rs]
    out["cat17_cost_10_50"] = mean(c["cost"] for c in cs)
    out["cat17_wrong"] = mean(c["wrong"] for c in cs)
    out["cat17_right"] = round(sum(c["right"] for c in cs) / n, 4)
    out["diagnostic_cost_10"] = mean(score_diagnostic(r, 10) for r in rs)
    # A partial answer that raised no alarm: no exhaustion, soft catch-all never fired.
    part = [r for r in rs if 0 < len(r["named"]) < len(r["truths"])]
    out["silent_partial"] = round(sum(not r["exhaust_turns"]
                                      and r["soft_fire_turn"] is None
                                      for r in part) / n, 4)
    return out


def spearman(x, y) -> float:
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # average ties
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hs = require_hash_seed()
    os.makedirs(args.out, exist_ok=True)
    with open(MATRIX) as f:
        matrix = json.load(f)["matrix"]
    hyps = load_hypotheses()
    t0 = time.time()
    known, nov_h, recs = play_all(matrix, hyps)
    secs = time.time() - t0
    with gzip.open(os.path.join(args.out, "games.json.gz"), "wt") as f:
        json.dump(recs, f)

    by_kind = {k: [r for r in recs if r["kind"] == k]
               for k in ("single", "novel", "KK", "KN")}
    summ: Dict[str, object] = {
        "settings": dict(P0A, p_unknown=P_UNKNOWN, backend="modeswitch_guard",
                         no_repeat=True, end_rule="always", flip=0.0),
        "python_hash_seed": hs, "seconds": round(secs, 1),
        "novels": [h.name for h in nov_h], "n_known": len(known),
        "kinds": {k: summarize_kind(v) for k, v in by_kind.items()}}

    # Compound excess over two separate incidents, per KK pair.
    single_cost = {r["truths"][0]: r for r in by_kind["single"]}
    qs = all_questions(known)
    base = matrix_oracle(matrix)
    h_of = {h.name: h for h in known}
    pair_rows = []
    for r in by_kind["KK"]:
        a, b = r["truths"]
        ov = pair_overlap(h_of[a], h_of[b], qs, base)
        row = {"pair": [a, b], **ov, "end": r["end"],
               "exhausted": r["end"] == "exhausted",
               "any_exhaustion": bool(r["exhaust_turns"]),
               "partial": len(r["named"]) == 1}
        for ce, cw in COSTS:
            c = score(r, ce, cw)["cost"]
            sep = (score(single_cost[a], ce, cw)["cost"]
                   + score(single_cost[b], ce, cw)["cost"])
            row["excess_{}_{}".format(ce, cw)] = c - sep
            row["vs_mean_member_{}_{}".format(ce, cw)] = c - sep / 2
        pair_rows.append(row)
    ex = {}
    for ce, cw in COSTS:
        k = "excess_{}_{}".format(ce, cw)
        v = np.array([p[k] for p in pair_rows], float)
        km = "vs_mean_member_{}_{}".format(ce, cw)
        vm = np.array([p[km] for p in pair_rows], float)
        ex["{}_{}".format(ce, cw)] = {
            "excess_over_two_incidents": round(float(v.mean()), 3),
            "excess_p10_p50_p90": [round(float(x), 1) for x in np.percentile(v, [10, 50, 90])],
            "share_pairs_excess_gt0": round(float((v > 0).mean()), 4),
            "excess_over_mean_member": round(float(vm.mean()), 3)}
    summ["KK_excess"] = ex
    # Compound penalty (the prereg's primary form): KK cost minus the mean of its two
    # members' single-game costs, under each reading.
    def penalty(f):
        return round(statistics.mean(
            f(r) - (f(single_cost[r["truths"][0]]) + f(single_cost[r["truths"][1]])) / 2
            for r in by_kind["KK"]), 3)
    summ["KK_penalty"] = {
        "primary_10_50_verified": penalty(lambda r: score(r, 10, 50)["cost"]),
        "unverified_10_50": penalty(lambda r: score(r, 10, 50, verified=False)["cost"]),
        "cat17_10_50": penalty(lambda r: score_class(r, _CAT, 10, 50)["cost"]),
        "diagnostic_10": penalty(lambda r: score_diagnostic(r, 10)),
        "wrong_actions": penalty(lambda r: r["wrong"]),
        "probes": penalty(lambda r: r["n_probes"])}
    for ce, cw in COSTS:
        summ["KK_penalty"]["verified_{}_{}".format(ce, cw)] = penalty(
            lambda r, ce=ce, cw=cw: score(r, ce, cw)["cost"])

    # Strata by how the answers combine.
    def stratum(p):
        d = p["d_min"]
        return "d0" if d == 0 else "d1" if d == 1 else "d2" if d == 2 else \
            "d3-4" if d <= 4 else "d5+"
    strata = {}
    for key in ("d0", "d1", "d2", "d3-4", "d5+"):
        rows = [p for p in pair_rows if stratum(p) == key]
        if not rows:
            continue
        strata[key] = {"n": len(rows),
                       "exhausted": mean(p["exhausted"] for p in rows),
                       "any_exhaustion": mean(p["any_exhaustion"] for p in rows),
                       "partial": mean(p["partial"] for p in rows),
                       "excess_10_50": mean(p["excess_10_50"] for p in rows)}
    for same in (True, False):
        rows = [p for p in pair_rows if p["same_category"] == same]
        strata["same_category" if same else "cross_category"] = {
            "n": len(rows), "exhausted": mean(p["exhausted"] for p in rows),
            "any_exhaustion": mean(p["any_exhaustion"] for p in rows),
            "partial": mean(p["partial"] for p in rows),
            "excess_10_50": mean(p["excess_10_50"] for p in rows)}
    summ["KK_strata"] = strata
    summ["KK_spearman_dmin_vs_any_exhaustion"] = round(spearman(
        [p["d_min"] for p in pair_rows], [p["any_exhaustion"] for p in pair_rows]), 4)
    summ["KK_spearman_dmin_vs_excess_10_50"] = round(spearman(
        [p["d_min"] for p in pair_rows], [p["excess_10_50"] for p in pair_rows]), 4)
    summ["KK_dmin_hist"] = dict(sorted(Counter(p["d_min"] for p in pair_rows).items()))

    # Gap 10 at flip 0: every escalated game (exhaustion or budget end), classified.
    tw = ThreeWay(known, matrix)
    nu = len(nov_h) / (len(nov_h) + len(known))
    esc = [r for r in recs if r["end"] in ("exhausted", "budget")
           and r["kind"] in ("single", "novel", "KK")]
    truth_lab = {"single": "single", "novel": "novel", "KK": "compound"}
    ev = [(truth_lab[r["kind"]], r["end"], tw.log_evidence(r["steps"])) for r in esc]
    three = {"n_escalations": {"{}|{}".format(t, e): c for (t, e), c in
                               sorted(Counter((t, e) for t, e, _ in ev).items())},
             "nu": round(nu, 4)}
    for w in SHARES:
        conf = Counter()
        for t, e, lev in ev:
            post = ThreeWay.posterior(lev, w, nu)
            conf[(t, max(post, key=post.get))] += 1
        three["confusion_w{}".format(w)] = {"{}->{}".format(a, b): c
                                            for (a, b), c in sorted(conf.items())}
    sc = [lev["compound"] - lev["novel"] for t, _, lev in ev if t == "compound"]
    sn = [lev["compound"] - lev["novel"] for t, _, lev in ev if t == "novel"]
    three["auc_compound_vs_novel_logBF"] = round(auc(sc, sn), 4)
    summ["three_way_flip0"] = three

    # How much of the "novel" alarm is compounds, by pair share (reported).
    rate = {k: summ["kinds"][k]["end"].get("exhausted", 0.0) for k in ("single", "novel", "KK")}
    alarm = {}
    for w in (0.0,) + SHARES:
        comp = w * rate["KK"]
        nov = (1 - w) * nu * rate["novel"]
        err = (1 - w) * (1 - nu) * rate["single"]
        tot = comp + nov + err
        alarm[str(w)] = {"exhaustion_per_incident": round(tot, 4),
                         "share_compound": round(comp / tot, 4) if tot else None,
                         "share_novel": round(nov / tot, 4) if tot else None}
    summ["alarm_mix_flip0"] = alarm

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
