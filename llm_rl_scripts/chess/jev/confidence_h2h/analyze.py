"""Analyze the confidence head-to-head: calibration, discrimination, gate utility.

Reads `raw/<condition>.jsonl` and the FROZEN grader, scores the pre-registered events
(acceptable = cp_lost<=50; best = cp_lost==0), and computes per-condition metrics with
paired bootstrap (B=5000, seed 0) over positions for every j_noul-vs-baseline difference.
Writes `results.json` (every number + CI) and `results.md`. Numbers are never retyped
into the doc; the FINDINGS.md patch reads this JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import chess

from llm_rl_scripts.chess.jev.confidence_h2h import common

CONDS = ["j_noul", "l_logprob", "l_verbal_gpt", "l_verbal_claude"]
BASELINES = ["l_logprob", "l_verbal_gpt", "l_verbal_claude"]
TAUS = [50, 70, 80, 90]
B = 5000


def load_raw(d: str, cond: str) -> Dict[str, dict]:
    path = os.path.join(d, cond + ".jsonl")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["position_id"]] = r
    return out


def auroc(pairs: List[Tuple[float, int]]) -> Optional[float]:
    """pairs = (score, label in {0,1}); rank-based AUROC with tie handling."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    ordered = sorted((s for s, _ in pairs))
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j] == ordered[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[ordered[i]] = avg
        i = j
    rsum = sum(ranks[s] for s in pos)
    return (rsum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def ece(triples: List[Tuple[float, int]], bins: int = 10) -> Optional[float]:
    """Equal-mass ECE over (p, event) triples."""
    if not triples:
        return None
    xs = sorted(triples)
    n = len(xs)
    total = 0.0
    for b in range(bins):
        lo = b * n // bins
        hi = (b + 1) * n // bins
        chunk = xs[lo:hi]
        if not chunk:
            continue
        mp = statistics.fmean(p for p, _ in chunk)
        me = statistics.fmean(e for _, e in chunk)
        total += len(chunk) / n * abs(mp - me)
    return total


def brier(triples: List[Tuple[float, int]]) -> Optional[float]:
    return statistics.fmean((p - e) ** 2 for p, e in triples) if triples else None


def quintile_lift(tops: List[Tuple[float, int]]) -> Optional[float]:
    """top-quintile acceptable-rate minus mean of bottom four quintiles."""
    if len(tops) < 5:
        return None
    xs = sorted(tops)
    n = len(xs)
    q = [xs[i * n // 5:(i + 1) * n // 5] for i in range(5)]
    rate = [statistics.fmean(y for _, y in grp) if grp else 0.0 for grp in q]
    return rate[4] - statistics.fmean(rate[:4])


def selective(tops: List[Tuple[float, int]], tau: int) -> Optional[float]:
    """acceptable-rate among positions with top-prob >= the tau-th percentile."""
    if not tops:
        return None
    xs = sorted(s for s, _ in tops)
    cut = xs[min(len(xs) - 1, int(tau / 100.0 * len(xs)))]
    kept = [y for s, y in tops if s >= cut]
    return statistics.fmean(kept) if kept else None


def build(raw: Dict[str, dict], grader, event_key: str):
    """Per-condition: triples [(p,event,pos)] and tops [(pos,top_prob,top_event)]."""
    evf = common.is_acceptable if event_key == "acceptable" else common.is_best
    triples, tops = [], []
    for pid, r in raw.items():
        if r.get("error") or not r.get("probs"):
            continue
        entry = grader[r["fen"]]
        for uci, p in r["probs"].items():
            ev = evf(entry, uci)
            if ev is not None:
                triples.append((pid, float(p), int(ev)))
        ch = r.get("chosen_uci")
        if ch is not None and ch in r["probs"]:
            ev = evf(entry, ch)
            if ev is not None:
                tops.append((pid, float(r["probs"][ch]), int(ev)))
    return triples, tops


def metrics_from(triples, tops):
    return {
        "ece": ece([(p, e) for _, p, e in triples]),
        "brier": brier([(p, e) for _, p, e in triples]),
        "auroc": auroc([(s, y) for _, s, y in tops]),
        "quintile_lift": quintile_lift([(s, y) for _, s, y in tops]),
        "selective": {str(t): selective([(s, y) for _, s, y in tops], t) for t in TAUS},
        "n_triples": len(triples), "n_positions": len(tops),
        "base_rate_top": (statistics.fmean(y for _, _, y in tops) if tops else None),
    }


def paired_bootstrap(a_tr, a_top, b_tr, b_top, metric_fn, seed=0):
    """Resample positions; return (point_diff, ci_lo, ci_hi) for metric A - B."""
    def by_pos(items):
        d = defaultdict(list)
        for it in items:
            d[it[0]].append(it)
        return d
    atr, atop = by_pos(a_tr), by_pos(a_top)
    btr, btop = by_pos(b_tr), by_pos(b_top)
    pids = sorted(set(atop) & set(btop))
    va0 = metric_fn([t for p in pids for t in atr[p]], [t for p in pids for t in atop[p]])
    vb0 = metric_fn([t for p in pids for t in btr[p]], [t for p in pids for t in btop[p]])
    if va0 is None or vb0 is None:
        return None, None, None
    point = va0 - vb0
    rng = random.Random(seed)
    diffs = []
    for _ in range(B):
        samp = [rng.choice(pids) for _ in pids]
        at = [t for p in samp for t in atr[p]]
        ao = [t for p in samp for t in atop[p]]
        bt = [t for p in samp for t in btr[p]]
        bo = [t for p in samp for t in btop[p]]
        va, vb = metric_fn(at, ao), metric_fn(bt, bo)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    if not diffs:
        return point, None, None
    diffs.sort()
    return point, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def _ece_fn(tr, top):
    return ece([(p, e) for _, p, e in tr])


def _auroc_fn(tr, top):
    return auroc([(s, y) for _, s, y in top])


def _sel_fn(tau):
    return lambda tr, top: selective([(s, y) for _, s, y in top], tau)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default=os.path.join(common.HERE, "raw"))
    p.add_argument("--out-dir", default=common.HERE)
    args = p.parse_args()

    grader = common.load_grader()
    raws = {c: load_raw(args.raw_dir, c) for c in CONDS}
    positions = common.load_positions()

    results = {"grader_hash": common.grader_hash(), "n_positions": len(positions),
               "B": B, "conditions": {}, "comparisons": {}, "cost_latency": {},
               "coverage": {}, "near_tie": {}}

    # near-tie stratum (grader-only, condition-independent).
    nt = 0
    for pos in positions:
        entry = grader[pos["fen"]]
        ucis = [m.uci() for m in chess.Board(pos["fen"]).legal_moves]
        if common.n_acceptable(entry, ucis) >= 2:
            nt += 1
    results["near_tie"] = {"positions_ge2_acceptable": nt, "of": len(positions)}

    built = {}
    for ev in ("acceptable", "best"):
        results["conditions"][ev] = {}
        built[ev] = {}
        for c in CONDS:
            tr, top = build(raws[c], grader, ev)
            built[ev][c] = (tr, top)
            results["conditions"][ev][c] = metrics_from(tr, top)

    # Paired j_noul vs each baseline on the primary (acceptable) + secondary (best).
    for ev in ("acceptable", "best"):
        results["comparisons"][ev] = {}
        atr, atop = built[ev]["j_noul"]
        for base in BASELINES:
            btr, btop = built[ev][base]
            comp = {}
            comp["ece"] = paired_bootstrap(atr, atop, btr, btop, _ece_fn)
            comp["auroc"] = paired_bootstrap(atr, atop, btr, btop, _auroc_fn)
            for tau in TAUS:
                comp["selective_%d" % tau] = paired_bootstrap(atr, atop, btr, btop, _sel_fn(tau))
            results["comparisons"][ev]["j_noul_vs_" + base] = {
                k: {"point": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in comp.items()}

    # Cost / latency / coverage.
    for c in CONDS:
        rs = [r for r in raws[c].values() if not r.get("error")]
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        cost = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        results["cost_latency"][c] = {
            "mean_latency_s": statistics.fmean(lat) if lat else None,
            "mean_cost_usd": statistics.fmean(cost) if cost else None,
            "total_cost_usd": sum(cost) if cost else None, "n_calls_logged": len(rs)}
    cov = [r["logprob_coverage"] for r in raws["l_logprob"].values()
           if r.get("logprob_coverage") is not None]
    results["coverage"]["l_logprob_mean"] = statistics.fmean(cov) if cov else None
    # Jev within-position Noul SD (mean over options, mean over positions).
    sds = []
    for r in raws["j_noul"].values():
        if r.get("noul_sd"):
            sds.append(statistics.fmean(r["noul_sd"].values()))
    results["jev_noul_within_sd_mean"] = statistics.fmean(sds) if sds else None

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_md(results, os.path.join(args.out_dir, "results.md"))
    print("wrote results.json + results.md")
    return 0


def _ci(d):
    if d.get("point") is None or d.get("ci_lo") is None:
        return "  -  "
    return "{:+.4f} [{:+.4f}, {:+.4f}]".format(d["point"], d["ci_lo"], d["ci_hi"])


def write_md(R, path):
    L = ["# Confidence head-to-head — results\n",
         "Grader hash `{}` · {} positions · B={} · near-tie (≥2 acceptable) {}/{}\n".format(
             R["grader_hash"], R["n_positions"], R["B"],
             R["near_tie"]["positions_ge2_acceptable"], R["near_tie"]["of"])]
    for ev in ("acceptable", "best"):
        L.append("\n## Event: {} (cp_lost {})\n".format(ev, "<=50" if ev == "acceptable" else "==0"))
        L.append("| condition | n_triples | n_pos | ECE | Brier | AUROC | quintile lift | sel@80 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            m = R["conditions"][ev][c]
            def f(x): return "{:.3f}".format(x) if isinstance(x, float) else "-"
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                c, m["n_triples"], m["n_positions"], f(m["ece"]), f(m["brier"]),
                f(m["auroc"]), f(m["quintile_lift"]), f(m["selective"]["80"])))
        L.append("\n**Paired j_noul − baseline (95% CI); ECE neg = Noul better; AUROC/sel pos = Noul better:**\n")
        L.append("| vs | ΔECE | ΔAUROC | Δsel@80 |")
        L.append("|---|---|---|---|")
        for base in BASELINES:
            comp = R["comparisons"][ev]["j_noul_vs_" + base]
            L.append("| {} | {} | {} | {} |".format(base, _ci(comp["ece"]), _ci(comp["auroc"]), _ci(comp["selective_80"])))
    L.append("\n## Cost / latency / coverage\n")
    L.append("| condition | mean latency s | mean cost $ | total $ |")
    L.append("|---|---|---|---|")
    for c in CONDS:
        cl = R["cost_latency"][c]
        def g(x, n=4): return "{:.{}f}".format(x, n) if isinstance(x, float) else "-"
        L.append("| {} | {} | {} | {} |".format(c, g(cl["mean_latency_s"]), g(cl["mean_cost_usd"], 5), g(cl["total_cost_usd"])))
    L.append("\nl_logprob mean coverage: {:.3f}  ·  Jev within-position Noul SD (mean): {:.4f}\n".format(
        R["coverage"]["l_logprob_mean"] or 0.0, R.get("jev_noul_within_sd_mean") or 0.0))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
