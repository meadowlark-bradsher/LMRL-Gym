"""FLYW-7 analyzer: scores ``FLYW2_COMPOUND_PREREG.md`` on a run directory.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.analyze_flyw2 \
        --run .../results/flyw2_run_<date> --gate .../results/flyw2_gate_<date>/gate.json \
        --seed-rule .../results/flyw2_smoke_<date>/seed_rule.json --out .../flyw2_analysis_<date>.json

Where the prereg's wording is ambiguous, this code is the reference.

* **Verdicts.** A result within 2 SE of its bar is INCONCLUSIVE, and an inconclusive test
  counts as not passed. SEs are within-unit (pairs for KK quantities, secrets for singles),
  as in the seed rule; TW1 and TW2 use a bootstrap over seeds (200 resamples, fixed RNG).
* **Rates** pool every unit and seed at a flip. "At each flip" tests must pass at 0.05 and
  0.10; CX2 pools the two flips' means equally.
* **Exhaustion escalation** = a game whose ``end`` is ``exhausted``.
* **TW1/TW2 events** = exhaustion escalations of singles ("single"), novels ("novel") and
  KK pairs ("compound") at the noisy flips.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import rankdata

from .flyw2_calibration import COSTS, _CAT, spearman, summarize_kind
from .flyw2_compound import (ThreeWay, all_questions, pair_overlap, score, score_class,
                             score_diagnostic)
from .flyw2_run import NOISY, load_games, load_inputs
from .flyw2_seed_rule import se_of_mean, within_var
from .bed import matrix_oracle
from .stress_harness import require_hash_seed

BARS = {"CX1(a)": ("<", 0.10), "CX1(b)": ("<", 0.10), "CX2": ("<=", 35.0),
        "CX3(a)": (">=", 0.80), "CX3(b)": (">=", 0.70), "TW1": (">=", 0.85),
        "TW2": (">=", 0.60)}
TW2_W = 0.25
SHARES = (0.0, 0.1, 0.25, 0.5)
LABEL = {"single": "single", "novel": "novel", "KK": "compound"}
N_BOOT = 200


def verdict(test: str, est: float, se: float) -> str:
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return "NO DATA"
    op, bar = BARS[test]
    if se is not None and not math.isnan(se) and abs(est - bar) <= 2 * se:
        return "INCONCLUSIVE"
    ok = {"<": est < bar, "<=": est <= bar, ">=": est >= bar}[op]
    return "PASS" if ok else "FAIL"


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def silent(g) -> bool:
    return (len(g["named"]) == 1 and not g["exhaust_turns"]
            and g["soft_fire_turn"] is None)


class Run:
    def __init__(self, games: List[dict]):
        self.games = games
        self.flips = sorted({g["flip"] for g in games})
        self.by = defaultdict(list)                     # (flip, kind) -> games
        self.single = {}                                # (flip, seed, name) -> game
        for g in games:
            self.by[(g["flip"], g["kind"])].append(g)
            if g["kind"] == "single":
                self.single[(g["flip"], g["seed"], g["truths"][0])] = g
        _, _, known, nov_h = load_inputs()
        self.known, self.nov_h = known, nov_h
        self.nu = len(nov_h) / (len(nov_h) + len(known))

    def seeds(self, flip):
        return sorted({g["seed"] for g in self.games if g["flip"] == flip})

    def members(self, g):
        return [self.single[(g["flip"], g["seed"], n)] for n in g["truths"]]

    def penalty(self, g, f) -> float:
        a, b = self.members(g)
        return f(g) - (f(a) + f(b)) / 2

    # mean and within-unit SE of a per-unit quantity
    def stat(self, flip, kind, f, unit=lambda g: tuple(g["truths"])):
        gs = self.by[(flip, kind)]
        if not gs:
            return float("nan"), float("nan"), 0
        vals = defaultdict(list)
        for g in gs:
            vals[unit(g)].append(float(f(g)))
        allv = [x for v in vals.values() for x in v]
        s = len(self.seeds(flip))
        se = se_of_mean(within_var(vals), s) if s > 1 else 0.0
        return statistics.mean(allv), se, len(allv)


def primary_cost(g):
    return score(g, 10, 50)["cost"]


def scored(run: Run) -> dict:
    res = {}
    for flip in NOISY:
        if not run.by[(flip, "KK")]:
            continue
        for test, f in (("CX1(a)", lambda g: g["end"] == "exhausted"),
                        ("CX1(b)", lambda g: g["soft_fire_turn"] is not None)):
            mk, sk, _ = run.stat(flip, "KK", f)
            ms, ss, _ = run.stat(flip, "single", f)
            est, se = mk - ms, math.hypot(sk, ss)
            res["{} @{}".format(test, flip)] = {"KK": round(mk, 5), "single": round(ms, 5),
                                                "estimate": round(est, 5), "se": round(se, 5),
                                                "verdict": verdict(test, est, se)}
        for test, f in (("CX3(a)", lambda g: len(g["named"]) == 1),
                        ("CX3(b)", silent)):
            m, se, _ = run.stat(flip, "KK", f)
            res["{} @{}".format(test, flip)] = {"estimate": round(m, 5), "se": round(se, 5),
                                                "verdict": verdict(test, m, se)}
    pens = [run.stat(fl, "KK", lambda g: run.penalty(g, primary_cost))
            for fl in NOISY if run.by[(fl, "KK")]]
    if len(pens) == len(NOISY):
        est = statistics.mean(p[0] for p in pens)
        se = math.sqrt(sum(p[1] ** 2 for p in pens)) / len(pens)
        res["CX2 pooled"] = {"per_flip": [round(p[0], 3) for p in pens],
                             "estimate": round(est, 3), "se": round(se, 4),
                             "verdict": verdict("CX2", est, se)}
    return res


def three_way(run: Run, flips: Sequence[float], eps_of=lambda flip: 0.05) -> dict:
    matrix = load_inputs()[0]
    tws = {}
    events = []      # (flip, seed, label, evidence)
    for flip in flips:
        eps = eps_of(flip)
        if eps not in tws:
            tws[eps] = ThreeWay(run.known, matrix, eps=eps)
        for kind in ("single", "novel", "KK"):
            for g in run.by[(flip, kind)]:
                if g["end"] == "exhausted":
                    events.append((flip, g["seed"], LABEL[kind],
                                   tws[eps].log_evidence(g["steps"])))
    return events


def tw_scores(events, nu: float, rng_seed: int = 7) -> dict:
    def tw1(evs):
        pos = [e["compound"] - e["novel"] for _, _, lab, e in evs if lab == "compound"]
        neg = [e["compound"] - e["novel"] for _, _, lab, e in evs if lab == "novel"]
        return auc(pos, neg)

    def tw2(evs, w=TW2_W):
        hit, tot = Counter(), Counter()
        for _, _, lab, e in evs:
            post = ThreeWay.posterior(e, w, nu)
            tot[lab] += 1
            hit[lab] += max(post, key=post.get) == lab
        labs = [l for l in ("single", "novel", "compound") if tot[l]]
        return (sum(hit[l] / tot[l] for l in labs) / 3 if labs else float("nan"),
                {l: [hit[l], tot[l]] for l in labs})

    def pair_auc(evs, a, b):
        pos = [e[a] - e[b] for _, _, lab, e in evs if lab == a]
        neg = [e[a] - e[b] for _, _, lab, e in evs if lab == b]
        return auc(pos, neg)

    est1 = tw1(events)
    est2, recall = tw2(events)
    seeds = sorted({s for _, s, _, _ in events})
    by_seed = defaultdict(list)
    for ev in events:
        by_seed[ev[1]].append(ev)
    rng = np.random.default_rng(rng_seed)
    b1, b2 = [], []
    if len(seeds) > 1:
        for _ in range(N_BOOT):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            evs = [ev for s in pick for ev in by_seed[s]]
            b1.append(tw1(evs))
            b2.append(tw2(evs)[0])
    se1 = float(np.nanstd(b1, ddof=1)) if b1 else float("nan")
    se2 = float(np.nanstd(b2, ddof=1)) if b2 else float("nan")
    conf = {}
    for w in (0.1, 0.25, 0.5):
        c = Counter()
        for _, _, lab, e in events:
            post = ThreeWay.posterior(e, w, nu)
            c["{}->{}".format(lab, max(post, key=post.get))] += 1
        conf["w{}".format(w)] = dict(sorted(c.items()))
    return {"n_events": dict(Counter(lab for _, _, lab, _ in events)),
            "TW1": {"estimate": round(est1, 4), "se": round(se1, 4),
                    "verdict": verdict("TW1", est1, se1)},
            "TW2": {"estimate": round(est2, 4), "se": round(se2, 4), "recall": recall,
                    "verdict": verdict("TW2", est2, se2)},
            "confusion": conf,
            "auc_single_vs_novel": round(pair_auc(events, "single", "novel"), 4),
            "auc_compound_vs_single": round(pair_auc(events, "compound", "single"), 4)}


def reported(run: Run) -> dict:
    out = {}
    matrix, _, known, _ = load_inputs()
    qs = all_questions(known)
    base = matrix_oracle(matrix)
    h_of = {h.name: h for h in known}
    ov = {}
    for flip in run.flips:
        kinds = {k: summarize_kind(run.by[(flip, k)])
                 for k in ("single", "novel", "KK", "KN") if run.by[(flip, k)]}
        kk = run.by[(flip, "KK")]
        rep = {"kinds": kinds, "seeds": len(run.seeds(flip))}
        if kk:
            pen = {"primary_10_50_verified": lambda g: score(g, 10, 50)["cost"],
                   "unverified_10_50": lambda g: score(g, 10, 50, verified=False)["cost"],
                   "cat17_10_50": lambda g: score_class(g, _CAT, 10, 50)["cost"],
                   "diagnostic_10": lambda g: score_diagnostic(g, 10),
                   "wrong_actions": lambda g: g["wrong"], "probes": lambda g: g["n_probes"]}
            for ce, cw in COSTS:
                pen["verified_{}_{}".format(ce, cw)] = (
                    lambda g, ce=ce, cw=cw: score(g, ce, cw)["cost"])
            rep["penalty"] = {k: round(statistics.mean(run.penalty(g, f) for g in kk), 3)
                              for k, f in pen.items()}
            rep["vs_two_incidents_10_50"] = round(statistics.mean(
                primary_cost(g) - sum(primary_cost(m) for m in run.members(g)) for g in kk), 3)
            # strata by how the answers combine
            for g in kk:
                k = tuple(g["truths"])
                if k not in ov:
                    ov[k] = pair_overlap(h_of[k[0]], h_of[k[1]], qs, base)
            strata = defaultdict(list)
            for g in kk:
                d = ov[tuple(g["truths"])]["d_min"]
                strata["d0" if d == 0 else "d1" if d == 1 else "d2" if d == 2 else
                       "d3-4" if d <= 4 else "d5+"].append(g)
                strata["same_category" if ov[tuple(g["truths"])]["same_category"]
                       else "cross_category"].append(g)
            rep["strata"] = {k: {"n": len(v),
                                 "exhausted": round(statistics.mean(g["end"] == "exhausted"
                                                                    for g in v), 4),
                                 "partial": round(statistics.mean(len(g["named"]) == 1
                                                                  for g in v), 4),
                                 "penalty_10_50": round(statistics.mean(
                                     run.penalty(g, primary_cost) for g in v), 3)}
                             for k, v in sorted(strata.items())}
            dm = [ov[tuple(g["truths"])]["d_min"] for g in kk]
            rep["spearman_dmin_vs_exhausted"] = round(spearman(
                dm, [g["end"] == "exhausted" for g in kk]), 4)
            rep["spearman_dmin_vs_penalty"] = round(spearman(
                dm, [run.penalty(g, primary_cost) for g in kk]), 4)
            rep["first_exhaustion_turn"] = {
                k: (round(statistics.mean(g["exhaust_turns"][0] for g in run.by[(flip, k)]
                                          if g["exhaust_turns"]), 2)
                    if any(g["exhaust_turns"] for g in run.by[(flip, k)]) else None)
                for k in kinds}
            # incident-level mixtures by pair share
            ex = {k: kinds[k]["end"].get("exhausted", 0.0) for k in ("single", "novel", "KK")}
            cost = {k: kinds[k]["cost_10_50"] for k in ("single", "novel", "KK")}
            sil = kinds["KK"]["silent_partial"]
            mix = {}
            for w in SHARES:
                pk, pn, ps = w, (1 - w) * run.nu, (1 - w) * (1 - run.nu)
                tot = pk * ex["KK"] + pn * ex["novel"] + ps * ex["single"]
                mix[str(w)] = {
                    "cost_per_incident": round(pk * cost["KK"] + pn * cost["novel"]
                                               + ps * cost["single"], 3),
                    "exhaustion_esc_per_incident": round(tot, 4),
                    "silent_partial_per_incident": round(pk * sil, 4),
                    "share_of_exhaustion_esc": {
                        "compound": round(pk * ex["KK"] / tot, 4) if tot else None,
                        "error": round(ps * ex["single"] / tot, 4) if tot else None,
                        "novel": round(pn * ex["novel"] / tot, 4) if tot else None}}
            rep["by_pair_share"] = mix
        out[str(flip)] = rep
    return out


def decision(sc: dict, tw: dict) -> List[str]:
    v = {k: x["verdict"] for k, x in sc.items()}
    lines = []
    cx1 = [v.get("CX1({}) @{}".format(p, f)) for p in "ab" for f in NOISY]
    cx3 = [v.get("CX3({}) @{}".format(p, f)) for p in "ab" for f in NOISY]
    cx2 = v.get("CX2 pooled")
    if None in cx1 or None in cx3 or cx2 is None:
        return ["incomplete run: not every scored test has data"]
    if not all(x == "PASS" for x in cx1):
        lines.append("CX1 not passed: the hard-gate-first recommendation needs revising; "
                     "FLYW-8 starts with a compound-aware escalation, and the flywheel must "
                     "not admit from exhaustion alone.")
    else:
        lines.append("CX1 passes: the recommendation stands on exhaustion.")
        if cx2 == "PASS" and all(x == "PASS" for x in cx3):
            lines.append("KILL: CX2 and CX3 pass. FLYW-8's modelling fixes are not built; "
                         "carry 'verify after every remediation; escalate if not cleared'.")
        elif cx2 != "PASS":
            lines.append("CX2 not passed: FLYW-8 builds explain-away first, then pairs as "
                         "hypotheses, scored on the penalty.")
        else:
            lines.append("CX3 not passed with CX2 passing: the failure is not mainly the "
                         "silent partial; FLYW-8 targets what replaced it.")
    if tw:
        lines.append("Gap 10: TW1 {}, TW2 {}.".format(tw["TW1"]["verdict"],
                                                      tw["TW2"]["verdict"]))
        if tw["TW2"]["verdict"] != "PASS":
            lines.append("TW2 not passed: separating novel, error and compound needs probes "
                         "chosen for it (FLYW-8 arm DP).")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gate")
    ap.add_argument("--seed-rule")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    require_hash_seed()
    run = Run(load_games(args.run))
    bad = [g for g in run.games if not set(g["named"]) <= set(g["truths"])]
    out = {"run": args.run, "flips": run.flips,
           "seeds": {str(f): len(run.seeds(f)) for f in run.flips},
           "cx0c_in_run": not bad}
    if args.gate:
        with open(args.gate) as f:
            out["gate_pass"] = json.load(f)["gate_pass"]
    if args.seed_rule:
        with open(args.seed_rule) as f:
            sr = json.load(f)
        out["seed_rule"] = {k: sr[k] for k in ("seeds_needed", "seeds_run", "cap_binds")}
    out["scored"] = scored(run)
    noisy = [f for f in NOISY if f in run.flips]
    tw = None
    if noisy:
        ev = three_way(run, noisy)
        tw = tw_scores(ev, run.nu)
        tw_eps = tw_scores(three_way(run, noisy, eps_of=lambda f: f), run.nu)
        out["three_way"] = tw
        out["three_way_eps_true_flip"] = {k: tw_eps[k] for k in ("TW1", "TW2", "confusion")}
    if 0.0 in run.flips:
        out["three_way_flip0"] = tw_scores(three_way(run, [0.0]), run.nu)
    out["decision"] = decision(out["scored"], tw)
    out["reported"] = reported(run)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in out if k != "reported"}, indent=1))


if __name__ == "__main__":
    main()
