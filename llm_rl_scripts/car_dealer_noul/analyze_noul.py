"""Analyze one CD-NOUL-1 Stage 1 result dir (``run_noul.py`` output) into ``analysis.json``.

Reads ``runs.jsonl``, ``summary.json`` and ``ceilings.json``; skips error stubs. The unit of
analysis is an (arm, regime) pair — ``plan_noask`` has the single regime ``none`` and is paired
with every text regime. Clusters for every paired contrast are the 36 (buyer ranking, seller
ranking) type pairs (``analyze_phase1.cluster_of``); the paired difference is over cluster means
with a percentile cluster bootstrap; ceilings are exact and the captured fraction divides the
bootstrap draws by the exact mean ceiling of the same cell set, in Y units.

Contrasts (design §4), each for every hedge and every regime where defined:
  ask_value      plan_ask-<d>-<a> (regime)  −  plan_noask            [+ captured fraction of Δ*_menu, Δ*_full]
  decoder        plan_ask-<d>-<a>           −  plan_ask-regex-<a>    (the parser control)
  type_vs_answer plan_ask-noul_type-<a>     −  plan_ask-noul_answer-<a>
  hedge          plan_ask-<d>-0.2           −  plan_ask-<d>-0
  chooser        plan_ask-noul_answer-<a>   −  sonnet_agent / jev_agent
  text_invariance plan_ask-oracle-0 (oblique) − plan_ask-oracle-0 (plain)   [gate: must be 0]
Also per (arm, regime): outcome stats, selectivity against the evaluator's reference
(``ask_optimal_true``), in-episode decoder diagnostics (answer log-loss, posterior on the true
type after each decode, zeroed-true-type episodes) and agent diagnostics (well-formed share,
positions, retries, tokens).

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.analyze_noul --dir results/<dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from fractions import Fraction
from statistics import mean
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_live.analyze_phase1 import CELL_SETS, CLUSTER_DEFINITION, _clean, ceiling_stats, cluster_of, in_set, paired_bootstrap, y

from . import arms_noul as an
from .verbalize import REGIMES

Key = Tuple[str, str]   # (arm, regime)


def load_runs(path: str) -> Tuple[List[dict], int]:
    recs, stubs = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                stubs += 1
                continue
            recs.append(r)
    return recs, stubs


def key_of(r: dict) -> Key:
    return (r["arm"], r["regime"])


def cluster_means(recs: List[dict], cell_set: str, fn=y) -> Dict[tuple, float]:
    by = defaultdict(list)
    for r in recs:
        if in_set(r, cell_set):
            by[cluster_of(r)].append(fn(r))
    return {k: mean(v) for k, v in by.items()}


def _cluster_ci(recs: List[dict], fn, B: int, seed: int):
    by = defaultdict(list)
    for r in recs:
        by[cluster_of(r)].append(fn(r))
    vals = [mean(v) for v in by.values()]
    if len(vals) < 2 or not B:
        return None
    rng = random.Random(seed)
    n = len(vals)
    draws = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [draws[int(0.025 * B)], draws[min(B - 1, int(0.975 * B))]]


# ---- per (arm, regime) --------------------------------------------------------------------------

def answer_logloss(d: dict) -> Optional[float]:
    """-log2 of the decoder's probability on the truthful answer (answer decoders only)."""
    dec = d.get("decode")
    if not dec or "p_yes" not in dec["read"]:
        return None
    p = dec["read"]["p_yes"] if dec["truthful"] == "yes" else 1.0 - dec["read"]["p_yes"]
    return -math.log2(min(max(p, 1e-6), 1.0 - 1e-6))


def decode_stats(recs: List[dict]) -> dict:
    asks = [d for r in recs for d in r["decisions"] if d.get("decode")]
    ll = [answer_logloss(d) for d in asks]
    ll = [x for x in ll if x is not None]
    post = [d["decode"]["p_true_after"] for d in asks]
    exact = [d["decode"]["p_true_exact_after"] for d in asks]
    zeroed = sum(1 for r in recs if r["final_agent_belief"]["p_true"] < 1e-9)
    conf_wrong = 0
    for d in asks:
        p = d["decode"]["read"].get("p_yes")
        if p is not None and abs(p - 0.5) >= 0.4 and ((p > 0.5) != (d["decode"]["truthful"] == "yes")):
            conf_wrong += 1
    return {"n_decodes": len(asks), "mean_answer_logloss_bits": mean(ll) if ll else None,
            "mean_p_true_after": mean(post) if post else None, "mean_p_true_exact_after": mean(exact) if exact else None,
            "confident_wrong_decodes": conf_wrong, "episodes_true_type_zeroed": zeroed,
            "mean_final_p_true": mean(r["final_agent_belief"]["p_true"] for r in recs) if recs else None,
            "mean_final_logloss_true_bits": mean(r["final_agent_belief"]["logloss_true_bits"] for r in recs) if recs else None}


def agent_stats(recs: List[dict]) -> Optional[dict]:
    infos = [d["agent"] for r in recs for d in r["decisions"] if d.get("agent")]
    if not infos:
        return None
    pos = defaultdict(int)
    for i in infos:
        pos[str(i["chosen_position"])] += 1
    usage = [i["usage"] for i in infos if i.get("usage")]
    return {"decisions": len(infos), "well_formed_share": sum(1 for i in infos if i.get("well_formed")) / len(infos),
            "retried": sum(1 for i in infos if i.get("retried")), "positions": dict(sorted(pos.items(), key=lambda kv: int(kv[0]))),
            "mean_choice_entropy_bits": mean(i["choice_entropy_bits"] for i in infos if i.get("choice_entropy_bits") is not None)
            if any(i.get("choice_entropy_bits") is not None for i in infos) else None,
            "input_tokens": sum(u["input_tokens"] for u in usage) if usage else None,
            "output_tokens": sum(u["output_tokens"] for u in usage) if usage else None}


def per_key_stats(recs: List[dict], B: int, seed: int) -> dict:
    if not recs:
        return {"n": 0}
    n_dec = sum(r["n_decisions"] for r in recs)
    return {"n": len(recs), "n_clusters": len({cluster_of(r) for r in recs}),
            "mean_normalized_net_gain": mean(y(r) for r in recs), "ci95_normalized_net_gain": _cluster_ci(recs, y, B, seed),
            "mean_net_gain": mean(r["net_gain"]["float"] for r in recs), "agreement_rate": mean(1.0 if r["agreement"] else 0.0 for r in recs),
            "walk_rate": mean(1.0 if r["terminal"]["reason"] == "walk" else 0.0 for r in recs),
            "mean_questions_asked": mean(r["questions_asked"] for r in recs), "ci95_questions_asked": _cluster_ci(recs, lambda r: r["questions_asked"], B, seed),
            "mean_proposals_made": mean(r["proposals_made"] for r in recs),
            "mean_total_decision_regret_true": mean(r["total_decision_regret_true"] for r in recs),
            "zero_regret_decision_share_true": (sum(r["zero_regret_decisions_true"] for r in recs) / n_dec) if n_dec else None,
            "n_decisions": n_dec, "decode": decode_stats(recs), "agent": agent_stats(recs)}


def selectivity(recs: List[dict], B: int, seed: int) -> dict:
    by = defaultdict(lambda: [0, 0, 0, 0])  # cluster -> [asked|opt, n|opt, asked|notopt, n|notopt]
    for r in recs:
        c = by[cluster_of(r)]
        for d in r["decisions"]:
            if not d["ask_legal"]:
                continue
            if d["ask_optimal_true"]:
                c[1] += 1; c[0] += d["chosen_kind"] == "ASK"
            else:
                c[3] += 1; c[2] += d["chosen_kind"] == "ASK"
    tot = [sum(c[i] for c in by.values()) for i in range(4)]
    p_opt = tot[0] / tot[1] if tot[1] else None
    p_not = tot[2] / tot[3] if tot[3] else None
    out = {"asked_when_optimal": tot[0], "decisions_ask_optimal": tot[1], "p_ask_given_optimal": p_opt,
           "asked_when_not_optimal": tot[2], "decisions_ask_legal_not_optimal": tot[3], "p_ask_given_not_optimal": p_not,
           "difference": (p_opt - p_not) if (p_opt is not None and p_not is not None) else None, "n_clusters": len(by)}
    if out["difference"] is not None and B:
        rng = random.Random(seed)
        cl = list(by.values()); n = len(cl); draws = []
        for _ in range(B):
            s = [0, 0, 0, 0]
            for _ in range(n):
                c = cl[rng.randrange(n)]
                for i in range(4):
                    s[i] += c[i]
            if s[1] and s[3]:
                draws.append(s[0] / s[1] - s[2] / s[3])
        if draws:
            draws.sort()
            out["difference_ci95"] = [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]]
    return out


# ---- contrasts ----------------------------------------------------------------------------------

def contrast_pairs(keys: List[Key]) -> List[Tuple[str, Key, Key, bool]]:
    """(name, A, B, is_ask_pair) for every pair the design names that the run contains."""
    have = set(keys)
    regimes = [r for r in REGIMES if any(k[1] == r for k in keys)]
    pairs: List[Tuple[str, Key, Key, bool]] = []
    noask = (an.PLAN_NOASK, an.NO_TEXT_REGIME)
    for a in an.HEDGES:
        for d in an.DECODERS:
            arm = an.arm_name(d, a)
            for rg in regimes:
                if (arm, rg) in have and noask in have:
                    pairs.append((f"ask_value|{arm}|{rg}", (arm, rg), noask, True))
                ctrl = (an.arm_name("regex", a), rg)
                if d != "regex" and (arm, rg) in have and ctrl in have:
                    pairs.append((f"decoder|{arm}|{rg}", (arm, rg), ctrl, False))
                orc = (an.arm_name("oracle", a), rg)
                if d != "oracle" and (arm, rg) in have and orc in have:
                    pairs.append((f"vs_oracle|{arm}|{rg}", (arm, rg), orc, False))
        for rg in regimes:
            t, s = (an.arm_name("noul_type", a), rg), (an.arm_name("noul_answer", a), rg)
            if t in have and s in have:
                pairs.append((f"type_vs_answer|{a}|{rg}", t, s, False))
            for agent in an.AGENT_ARMS:
                p, g = (an.arm_name("noul_answer", a), rg), (agent, rg)
                if p in have and g in have:
                    pairs.append((f"chooser|{agent}|{a}|{rg}", p, g, False))
            if (agent_key := (an.AGENT_ARMS[0], rg)) in have and noask in have:
                pass
    for d in an.DECODERS:
        for rg in regimes:
            h, l = (an.arm_name(d, Fraction(1, 5)), rg), (an.arm_name(d, Fraction(0)), rg)
            if h in have and l in have:
                pairs.append((f"hedge|{d}|{rg}", h, l, False))
    for agent in an.AGENT_ARMS:
        for rg in regimes:
            if (agent, rg) in have and noask in have:
                pairs.append((f"ask_value|{agent}|{rg}", (agent, rg), noask, True))
    orc0 = an.arm_name("oracle", Fraction(0))
    if (orc0, "oblique") in have and (orc0, "plain") in have:
        pairs.append((f"text_invariance|{orc0}", (orc0, "oblique"), (orc0, "plain"), False))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-contrasts", action="store_true", help="print the contrast estimates (off by default: a pilot's contrasts are not to be read)")
    args = ap.parse_args(argv)
    analysis = analyze(args.dir, args.B, args.seed)
    out = os.path.join(args.dir, "analysis.json")
    with open(out, "w") as f:
        json.dump(_clean(analysis), f, indent=1)
    print(f"wrote {out}: {analysis['episodes']} episodes, {analysis['error_stubs']} stubs, {len(analysis['keys'])} (arm, regime) keys, "
          f"{len(analysis['contrasts'])} contrasts")
    if args.print_contrasts:
        for name, c in analysis["contrasts"].items():
            a = c["all"]
            if a["point"] is not None:
                print(f"  {name:48s} all: {a['point']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] n={a['n_clusters']}")
    return 0


def analyze(d: str, B: int = 5000, seed: int = 0) -> dict:
    summary = json.load(open(os.path.join(d, "summary.json")))
    ceil_path = os.path.join(d, "ceilings.json")
    ceil = json.load(open(ceil_path)) if os.path.exists(ceil_path) else {"cells": {}}
    recs, stubs = load_runs(os.path.join(d, "runs.jsonl"))
    by: Dict[Key, List[dict]] = defaultdict(list)
    for r in recs:
        by[key_of(r)].append(r)
    keys = sorted(by)
    r_res = float(Fraction(summary["config"]["reservation"]))
    umax = sorted({round(r["u_max"]["float"], 12) for r in recs}) if recs else [1.0]
    assert len(umax) == 1, f"u_max differs across records: {umax}"
    y_scale = umax[0] - r_res
    ceilings = ceiling_stats(ceil, y_scale)

    per_key = {f"{a}|{rg}": {cs: per_key_stats([r for r in by[(a, rg)] if in_set(r, cs)], B, seed) for cs in CELL_SETS} for a, rg in keys}
    contrasts, contrasts_questions, contrasts_logloss = {}, {}, {}
    for name, A, Bk, is_ask in contrast_pairs(keys):
        contrasts[name] = {}
        contrasts_questions[name] = {}
        for cs in CELL_SETS:
            cm = ceilings.get(cs, {})
            ceil_for = {"delta_menu": cm.get("mean_delta_menu_Y"), "delta_full": cm.get("mean_delta_full_Y")} if is_ask and cm.get("n_cells") else None
            contrasts[name][cs] = paired_bootstrap(cluster_means(by[A], cs), cluster_means(by[Bk], cs), B, seed, ceil_for)
            contrasts_questions[name][cs] = paired_bootstrap(cluster_means(by[A], cs, lambda r: r["questions_asked"]),
                                                             cluster_means(by[Bk], cs, lambda r: r["questions_asked"]), B, seed)
        if name.startswith("decoder|"):   # in-episode decoding log-loss of the decoder vs the parser, per episode mean
            def ep_ll(r):
                xs = [answer_logloss(dd) for dd in r["decisions"]]
                xs = [x for x in xs if x is not None]
                return mean(xs) if xs else None
            def cm_ll(recs_, cs):
                byc = defaultdict(list)
                for r in recs_:
                    v = ep_ll(r)
                    if v is not None and in_set(r, cs):
                        byc[cluster_of(r)].append(v)
                return {k: mean(v) for k, v in byc.items()}
            contrasts_logloss[name] = {cs: paired_bootstrap(cm_ll(by[A], cs), cm_ll(by[Bk], cs), B, seed) for cs in CELL_SETS}
    sel = {f"{a}|{rg}": selectivity(by[(a, rg)], B, seed) for a, rg in keys if an.parse_arm(a).asks}
    return {
        "env": summary.get("env"), "design": summary.get("design"), "config": summary["config"], "kernel": summary["kernel"],
        "bank": summary.get("bank"), "agent_menu_decoder": summary.get("agent_menu_decoder"), "registered_hedge": summary.get("registered_hedge"),
        "arms": summary["arms"], "regimes": summary.get("regimes"), "keys": [f"{a}|{rg}" for a, rg in keys],
        "episodes": len(recs), "error_stubs": stubs, "repeats": summary.get("repeats"), "seed": summary.get("seed"),
        "elapsed_s": summary.get("elapsed_s"), "bootstrap": {"B": B, "seed": seed}, "cluster_definition": CLUSTER_DEFINITION,
        "y_scale": y_scale, "ceilings": ceilings, "per_key": per_key, "contrasts": contrasts,
        "contrasts_questions": contrasts_questions, "contrasts_answer_logloss": contrasts_logloss, "selectivity": sel,
    }


if __name__ == "__main__":
    sys.exit(main())
