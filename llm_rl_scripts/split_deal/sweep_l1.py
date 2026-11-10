"""L1 words-value sweep of Split Deal and gate SD-W (`DESIGN_SPLIT.md` §5, §7). Offline, exact, no API.

For every instance, regime (easy, hard) and prior over B's type (independent uniform, the primary; the
value-pair constraints, a sensitivity prior), and every A type:

* V_words(Q) for Q = 0..6 at ask costs 0, 1/100, 3/100 Y, with V_none = V_words(0) and V_reveal;
* the exact policy played against every B type: questions asked, walks, agreements, and the L0
  metrics E (joint efficiency) and C_A (A's surplus share) of the outcome;
* at cost 0, the greedy policies (EIG-greedy T4, value-greedy) at each Q;
* at cost 0, the re-offer variant with H = 4 moves and hazard h after a rejection, with and without asks.

Values are averaged over the joint prior of (A type, B type) within an instance (under the independent
prior, A types weigh equally), then over the nine instances with equal weight. Y units throughout.

Gate SD-W, as registered in the note before this ran: W = D_words(3) = V_words(3) - V_none, easy, ask
cost 0, independent prior. W >= 1/10 go; 2/25 <= W < 1/10 conditional; W < 2/25 no-go. Hard is the one
fallback, used only if easy is no-go and hard is not.

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l1 run --out results/split_l1_20261110/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l1 render results/split_l1_20261110/sweep.json \
        --note llm_rl_scripts/split_deal/SWEEP_SPLIT_L1.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from typing import Dict, List, Optional

from .belief import Cell, Tables
from .frontier import frontier_table
from .game import REGIMES, TOTAL, Instance, canonical_instances, outside_option
from .spec_numbers import pair_constraints

SWEEP_VERSION = "split-deal-l1-v0.1"
Q_MAX = 6
GATE_Q = 3
COSTS = (Fraction(0), Fraction(1, 100), Fraction(3, 100))
HAZARDS = (Fraction(0), Fraction(1, 10), Fraction(1, 4))
REOFFER_H = 4
PRIORS = ("independent", "constrained")
GO, FLOOR, SCARCITY = Fraction(1, 10), Fraction(2, 25), Fraction(1, 25)
RULES = ("eig", "value")
ZERO = Fraction(0)


# -- one (instance, regime, prior) -------------------------------------------------------

def _prior_mask(tab: Tables, va, prior: str) -> int:
    if prior == "independent":
        return tab.full
    return sum(1 << i for i, vb in enumerate(tab.types) if pair_constraints(va, vb))


def _s_star(tab: Tables, va, vb) -> Optional[Fraction]:
    best = None
    for a in tab.allocs:
        ua, ub = tab.inst.utility(va, a), tab.inst.utility(vb, tab.inst.rest(a))
        if ua >= tab.d and ub >= tab.d:
            s = ua + ub - 2 * tab.d
            best = s if best is None or s > best else best
    return best


def _blank_stats():
    return {"pairs": 0, "asks": 0, "walks": 0, "agree": 0, "e_n": 0, "e_sum": ZERO, "c_n": 0, "c_sum": ZERO}


def solve_group(args) -> dict:
    inst_counts, regime, prior = args
    inst = Instance(tuple(inst_counts))
    tab = Tables(inst, regime)
    nb = len(tab.types)
    s_star = [[_s_star(tab, va, vb) for vb in tab.types] for va in tab.types]

    weight = 0
    v_words = {c: [ZERO] * (Q_MAX + 1) for c in COSTS}
    stats = {c: [_blank_stats() for _ in range(Q_MAX + 1)] for c in COSTS}
    v_reveal = ZERO
    v_greedy = {r: [ZERO] * (Q_MAX + 1) for r in RULES}
    v_reoffer = {(h, asks): ZERO for h in HAZARDS for asks in (False, True)}

    for ia, va in enumerate(tab.types):
        mask = _prior_mask(tab, va, prior)
        n = mask.bit_count()
        if n == 0:
            continue
        weight += n
        bs = [b for b in range(nb) if (mask >> b) & 1]
        for c in COSTS:
            cell = Cell(tab, va, mask, c)
            for q in range(Q_MAX + 1):
                v_words[c][q] += n * cell.words(mask, q)[0]
                st = stats[c][q]
                for b in bs:
                    asks, agreed, walked = cell.rollout(b, q, "exact")
                    st["pairs"] += 1
                    st["asks"] += asks
                    st["walks"] += walked
                    st["agree"] += agreed is not None
                    ss = s_star[ia][b]
                    if ss:
                        st["e_n"] += 1
                        if agreed is not None:
                            a = tab.allocs[agreed]
                            st["e_sum"] += (inst.utility(va, a) + inst.utility(tab.types[b], inst.rest(a)) - 2 * tab.d) / ss
                    if agreed is not None:
                        a = tab.allocs[agreed]
                        ua, ub = inst.utility(va, a), inst.utility(tab.types[b], inst.rest(a))
                        s = ua + ub - 2 * tab.d
                        if s > 0:
                            st["c_n"] += 1
                            st["c_sum"] += (ua - tab.d) / s
            if c == 0:
                v_reveal += n * cell.reveal()
                for r in RULES:
                    for q in range(Q_MAX + 1):
                        v_greedy[r][q] += n * cell.greedy(mask, q, r)
                for h in HAZARDS:
                    for asks in (False, True):
                        v_reoffer[(h, asks)] += n * cell.reoffer(mask, REOFFER_H, h, asks)

    def mean(x):
        return x / weight

    out = {"instance": inst.key, "regime": regime, "prior": prior, "n_types": nb, "pairs": weight,
           "v_reveal": mean(v_reveal),
           "v_words": {str(c): [mean(v) for v in v_words[c]] for c in COSTS},
           "v_greedy": {r: [mean(v) for v in v_greedy[r]] for r in RULES},
           "v_reoffer": {str(h): {"offers": mean(v_reoffer[(h, False)]), "asks_and_offers": mean(v_reoffer[(h, True)])}
                         for h in HAZARDS},
           "policy": {str(c): [_finish_stats(s) for s in stats[c]] for c in COSTS}}
    return out


def _finish_stats(s: dict) -> dict:
    p = s["pairs"]
    return {"asks_per_episode": Fraction(s["asks"], p), "walk_share": Fraction(s["walks"], p),
            "agree_share": Fraction(s["agree"], p),
            "mean_E": s["e_sum"] / s["e_n"] if s["e_n"] else None, "pairs_with_E": s["e_n"],
            "mean_C_A": s["c_sum"] / s["c_n"] if s["c_n"] else None, "agreements_with_C_A": s["c_n"]}


# -- pooling, gate, JSON ------------------------------------------------------------------

def _pool(groups: List[dict]) -> dict:
    """Equal-weight mean over instances of every value (and policy statistic) in the groups."""
    k = len(groups)

    def avg(get):
        vals = [get(g) for g in groups]
        if any(v is None for v in vals):
            return None
        return sum(vals, ZERO) / k

    out = {"v_reveal": avg(lambda g: g["v_reveal"]),
           "v_words": {c: [avg(lambda g, c=c, q=q: g["v_words"][c][q]) for q in range(Q_MAX + 1)]
                       for c in groups[0]["v_words"]},
           "v_greedy": {r: [avg(lambda g, r=r, q=q: g["v_greedy"][r][q]) for q in range(Q_MAX + 1)] for r in RULES},
           "v_reoffer": {h: {kk: avg(lambda g, h=h, kk=kk: g["v_reoffer"][h][kk]) for kk in ("offers", "asks_and_offers")}
                         for h in groups[0]["v_reoffer"]},
           "policy": {c: [{f: avg(lambda g, c=c, q=q, f=f: g["policy"][c][q][f])
                           for f in ("asks_per_episode", "walk_share", "agree_share", "mean_E", "mean_C_A")}
                          for q in range(Q_MAX + 1)] for c in groups[0]["policy"]}}
    return out


def d_words(block: dict, cost: str = "0", q: int = GATE_Q) -> Fraction:
    return block["v_words"][cost][q] - block["v_words"][cost][0]


def tier(w: Fraction) -> str:
    return "go" if w >= GO else "conditional" if w >= FLOOR else "no-go"


def gate(pooled: Dict[str, dict]) -> dict:
    """``pooled`` maps regime -> pooled block for the independent prior."""
    w = {r: d_words(pooled[r]) for r in REGIMES}
    tiers = {r: tier(w[r]) for r in REGIMES}
    primary, fallback = "easy", False
    if tiers["easy"] == "no-go" and tiers["hard"] != "no-go":
        primary, fallback = "hard", True
    verdict = tiers[primary]
    re0 = pooled[primary]["v_reoffer"]["0"]
    d_re = re0["asks_and_offers"] - re0["offers"]
    return {"W": {r: float(w[r]) for r in REGIMES}, "W_exact": {r: str(w[r]) for r in REGIMES}, "tiers": tiers,
            "primary_regime": primary, "fallback_used": fallback, "verdict": verdict,
            "bars": {"go": str(GO), "floor": str(FLOOR), "scarcity": str(SCARCITY)},
            "reoffer_d_words_h0": float(d_re), "reoffer_d_words_h0_exact": str(d_re),
            "scarcity_conditional": verdict != "no-go" and d_re < SCARCITY}


def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs: List[float], ys: List[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def _jsonable(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def compute(instances: Optional[List[Instance]] = None, workers: int = 1) -> dict:
    instances = instances or canonical_instances()
    jobs = [(list(i.counts), r, p) for i, r, p in itertools.product(instances, REGIMES, PRIORS)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            groups = list(ex.map(solve_group, jobs))
    else:
        groups = [solve_group(j) for j in jobs]
    pooled = {p: {r: _pool([g for g in groups if g["regime"] == r and g["prior"] == p]) for r in REGIMES}
              for p in PRIORS}
    easy_indep = [g for g in groups if g["regime"] == "easy" and g["prior"] == "independent"]
    type_count = {"n_types": [g["n_types"] for g in easy_indep],
                  "W": [float(d_words(g)) for g in easy_indep],
                  "spearman": spearman([g["n_types"] for g in easy_indep], [float(d_words(g)) for g in easy_indep])}
    frontier = {r: frontier_table(instances, outside_option(r), TOTAL - outside_option(r)) for r in REGIMES}
    return {"sweep": SWEEP_VERSION,
            "config": {"q_max": Q_MAX, "gate_q": GATE_Q, "costs": [str(c) for c in COSTS],
                       "hazards": [str(h) for h in HAZARDS], "reoffer_moves": REOFFER_H, "priors": list(PRIORS),
                       "regimes": {r: str(v) for r, v in REGIMES.items()}, "instances": [i.key for i in instances]},
            "gate": gate(pooled["independent"]),
            "type_count": type_count,
            "pooled": _jsonable(pooled),
            "groups": _jsonable(groups),
            "frontier": frontier}


def _commit() -> str:
    """HEAD, marked +dirty when the package's code differs from it (new or modified .py files)."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        here = os.path.dirname(os.path.abspath(__file__))
        status = subprocess.check_output(["git", "status", "--porcelain", "--", here], text=True)
        dirty = any(line.rstrip().endswith(".py") for line in status.splitlines())
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def run(args) -> int:
    t0 = time.time()
    doc = compute(workers=args.workers)
    doc["commit"] = _commit()
    doc["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    g = doc["gate"]
    print(f"wrote {args.out} ({doc['seconds']} s); W easy {g['W']['easy']:.4f} ({g['tiers']['easy']}), "
          f"W hard {g['W']['hard']:.4f} ({g['tiers']['hard']}) -> verdict {g['verdict']}"
          f"{' (hard fallback)' if g['fallback_used'] else ''}")
    return 0


# -- note -----------------------------------------------------------------------------------

def _f(x, nd=4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


VERDICT_TEXT = {
    "go": "a live study on Split Deal is worth designing (its registration still needs its own power "
          "calculation, NEG-33)",
    "conditional": "offline work continues; live spend needs a power calculation showing an expected 95 % "
                   "half-width below W / 2 (NEG-33)",
    "no-go": "in this configuration; the game is to be revised as a new spec version with its own bar stated "
             "before its sweep",
}


def render(doc: dict, path: str) -> str:
    g, cfg = doc["gate"], doc["config"]
    P = doc["pooled"]
    L = [f"# Split Deal — L1 words-value sweep and gate SD-W",
         "",
         f"_Rendered by `sweep_l1.py render` from `{path}` (sweep `{doc['sweep']}`, commit `{doc['commit']}`, "
         f"{doc['seconds']} s). Exact on the model: no sampling, no API; every number is formatted from the JSON. "
         "Gate and configuration as registered in `DESIGN_SPLIT.md` §7 (spec v0.2) before this ran._",
         "",
         "Protocol 4.1: A asks up to Q bank questions, answered truthfully, then makes one binding offer or walks; "
         "B accepts iff its utility is at least d. Values in Y (net gain over d as a share of 10 − d), averaged "
         "over the joint prior of the two types within an instance, then over the nine instances with equal "
         "weight. `D_words(Q) = V_words(Q) − V_none`; `D_rest = V_reveal − V_words(Q)`.",
         "",
         "## Gate SD-W",
         "",
         "| regime | W = D_words(3) | exact | tier |",
         "|---|---|---|---|"]
    for r in ("easy", "hard"):
        L.append(f"| {r} | {g['W'][r]:.4f} | {g['W_exact'][r]} | {g['tiers'][r]} |")
    L += ["",
          f"Bars: go at W ≥ {float(Fraction(g['bars']['go'])):.2f} Y, conditional from "
          f"{float(Fraction(g['bars']['floor'])):.2f} Y. Primary regime: **{g['primary_regime']}**"
          f"{' (the pre-named fallback; easy was a no-go)' if g['fallback_used'] else ''}.",
          "",
          f"**Verdict: {g['verdict']}** — {VERDICT_TEXT[g['verdict']]}.",
          "",
          f"Re-offer check (H = {cfg['reoffer_moves']}, h = 0, {g['primary_regime']}): asks-and-offers minus "
          f"offers-only = {g['reoffer_d_words_h0']:.4f} Y against the {float(Fraction(g['bars']['scarcity'])):.2f} Y "
          "line. " + ("Below it, and the gate did not fail: the words matter *because* offers are scarce, so the "
                      "live protocol must keep the single-offer rule and any write-up must say the result is "
                      "conditional on it." if g["scarcity_conditional"] else
                      ("At or above it: the words keep their value even when offers may be repeated."
                       if g["verdict"] != "no-go" else "Not applicable: the gate failed.")),
          ""]

    for prior in ("independent", "constrained"):
        title = "independent uniform prior (primary)" if prior == "independent" else \
            "value-pair constraints (sensitivity prior; never gates)"
        L += [f"## By Q, ask cost 0 — {title}", ""]
        for r in ("easy", "hard"):
            b = P[prior][r]
            v, pol = b["v_words"]["0"], b["policy"]["0"]
            L += [f"**{r}** (V_reveal = {b['v_reveal']:.4f})", "",
                  "| Q | V_words | D_words | D_rest | questions / ep | walk | agree | mean E | mean C_A |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for q in range(cfg["q_max"] + 1):
                st = pol[q]
                L.append(f"| {q} | {v[q]:.4f} | {v[q] - v[0]:.4f} | {b['v_reveal'] - v[q]:.4f} | "
                         f"{st['asks_per_episode']:.2f} | {_pct(st['walk_share'])} | {_pct(st['agree_share'])} | "
                         f"{_f(st['mean_E'], 3)} | {_f(st['mean_C_A'], 3)} |")
            L.append("")
    L += ["`walk` and `agree` are shares of episodes under the exact policy; `mean E` is joint efficiency "
          "S(o)/S* over pairs with S* > 0 (no deal counts 0) and `mean C_A` A's share of the surplus over "
          "agreements, both from the L0 oracle (`frontier.py`). Questions per episode: ties go to not asking.",
          "",
          "## Ask cost (independent prior)",
          "",
          "| regime | cost (Y) | D_words(3) | questions / ep at Q = 3 | D_words(6) |",
          "|---|---|---|---|---|"]
    for r in ("easy", "hard"):
        b = P["independent"][r]
        for c in cfg["costs"]:
            v = b["v_words"][c]
            L.append(f"| {r} | {c} | {v[3] - v[0]:.4f} | {b['policy'][c][3]['asks_per_episode']:.2f} | "
                     f"{v[6] - v[0]:.4f} |")
    L += ["",
          "## Greedy gap, ask cost 0 (independent prior)",
          "",
          "EIG-greedy (T4) asks the question with the most expected information about B's type; value-greedy the "
          "one with the largest one-step gain in offer value. Both ask while a question splits the posterior and "
          "then make the optimal offer. The gap is exact (T5) minus greedy.",
          "",
          "| regime | Q | exact | EIG-greedy | value-greedy | gap EIG | gap value |",
          "|---|---|---|---|---|---|---|"]
    for r in ("easy", "hard"):
        b = P["independent"][r]
        for q in range(1, cfg["q_max"] + 1):
            ex, ge, gv = b["v_words"]["0"][q], b["v_greedy"]["eig"][q], b["v_greedy"]["value"][q]
            L.append(f"| {r} | {q} | {ex:.4f} | {ge:.4f} | {gv:.4f} | {ex - ge:.4f} | {ex - gv:.4f} |")
    L += ["",
          f"## Re-offer variant (H = {cfg['reoffer_moves']}, ask cost 0)",
          "",
          "Each move is an ask, a binding offer or a walk; a rejected offer continues with probability 1 − h. "
          "`D` is asks-and-offers minus offers-only, the words value when rejections are tests.",
          "",
          "| prior | regime | h | offers only | asks and offers | D |",
          "|---|---|---|---|---|---|"]
    for prior in ("independent", "constrained"):
        for r in ("easy", "hard"):
            for h in cfg["hazards"]:
                x = P[prior][r]["v_reoffer"][h]
                L.append(f"| {prior} | {r} | {h} | {x['offers']:.4f} | {x['asks_and_offers']:.4f} | "
                         f"{x['asks_and_offers'] - x['offers']:.4f} |")
    tc = doc["type_count"]
    L += ["",
          "## Words value against type count (easy, ask cost 0, independent prior)",
          "",
          "| instance | types | W = D_words(3) | W hard | D_rest(3) easy |",
          "|---|---|---|---|---|"]
    groups = {(x["instance"], x["regime"], x["prior"]): x for x in doc["groups"]}
    order = sorted(cfg["instances"], key=lambda k: -groups[(k, "easy", "independent")]["n_types"])
    for k in order:
        e, h = groups[(k, "easy", "independent")], groups[(k, "hard", "independent")]
        we = e["v_words"]["0"][3] - e["v_words"]["0"][0]
        wh = h["v_words"]["0"][3] - h["v_words"]["0"][0]
        L.append(f"| {k} | {e['n_types']} | {we:.4f} | {wh:.4f} | {e['v_reveal'] - e['v_words']['0'][3]:.4f} |")
    rho = tc["spearman"]
    L += ["", f"Spearman rank correlation of W with type count over the nine instances: {rho:+.2f}. "
          + ("W does not rise with type count, so by the reading registered in §7 type count was not what held "
             "the words back." if rho <= 0 else
             "W rises with type count (descriptive; nine instances)."),
          "",
          "## L0 frontier by instance",
          "",
          "Per pair of types with an acceptable allocation: S* in units of 10 − d (so a joint value; 2 is the most), "
          "the size of the Pareto set among acceptable allocations, the number of Nash points, and A's share of the "
          "surplus at the Nash point (the lower one if tied) over pairs with S* > 0.",
          "",
          "| regime | instance | pairs with a deal | mean S* | mean Pareto size | mean Nash points | Nash share C_A |",
          "|---|---|---|---|---|---|---|"]
    for r in ("easy", "hard"):
        for x in doc["frontier"][r]:
            L.append(f"| {r} | {x['instance']} | {x['pairs_with_deal'] / x['n_pairs']:.2f} | "
                     f"{_f(x['mean_s_star_given_deal'], 3)} | {_f(x['mean_pareto_given_deal'], 2)} | "
                     f"{_f(x['mean_nash_points_given_deal'], 2)} | {_f(x['mean_nash_share_a_given_positive'], 3)} |")
    return "\n".join(L) + "\n"


def render_cmd(args) -> int:
    with open(args.json) as f:
        doc = json.load(f)
    with open(args.note, "w") as f:
        f.write(render(doc, args.json))
    print(f"wrote {args.note}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.set_defaults(func=run)
    n = sub.add_parser("render")
    n.add_argument("json")
    n.add_argument("--note", required=True)
    n.set_defaults(func=render_cmd)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
