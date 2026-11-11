"""L1 words-value sweep of Split Deal v0.3 and gate SD3-W (`DESIGN_SPLIT_V03.md` §5). Offline, exact, no API.

The v0.2 sweep (``sweep_l1.py``) on the v0.3 tables (``private_d.PrivateTables``): a type is (values,
level), d(lo) = 4, d(hi) = 11/2, and the bank has 13 questions with HIGH. The same sweep runs on the
tables without HIGH as the ungated control (decision 3). For every instance, bank (with HIGH, without),
prior over B's type and every A type:

* V_words(Q) for Q = 0..6 at ask costs 0, 1/100, 3/100 Y, V_none = V_words(0), and V_reveal;
* the exact policy played against every B type: questions asked (and whether HIGH was among them),
  walks, agreements, joint efficiency E and A's surplus share C_A of the outcome;
* at cost 0, EIG-greedy and value-greedy at each Q, and the re-offer variant (H = 4, hazard h).

Operational reading of gate SD3-W, fixed here before the sweep ran:

* W = D_words(3) = V_words(3) - V_none, protocol 4.1, ask cost 0, the 13-question bank (HIGH counts
  toward the three), independent uniform prior over B's type. Y is net gain over A's own d as a share of
  10 - A's own d (decision 2). Within an instance, values average over A's type and B's type uniformly
  (so over both sides' levels uniformly); then the nine instances weigh equally.
* Tiers as SD-W: go at W >= 1/10, conditional from 2/25 (live spend then needs a power calculation),
  no-go below. No fallback: levels replace v0.2's regimes. On a no-go the one-sided words study stays on
  v0.2, and v0.3 is used only for gate SD3-D (decision 4).
* Scarcity check: H = 4, h = 0, asks-and-offers minus offers-only below 1/25 Y means the words value
  depends on the single-offer rule. It is computed on the same pooled block as W.
* Diagnostics, deciding nothing: W per level pair (A's level / B's level), from the exact policy's
  realized gains against each B type (whose mean over B types is V_words, tested); every pooled value
  also on the common scale (U - own d) / 10 (decision 2); the control bank without HIGH; the value-pair
  constraint prior of v0.2 (a sensitivity prior there, carried over unchanged; it constrains the values
  part of a type only).

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l1_v03 run --out results/split_v03_l1_20261110/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l1_v03 render results/split_v03_l1_20261110/sweep.json \\
        --note llm_rl_scripts/split_deal/SWEEP_SPLIT_V03_L1.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from typing import List, Optional

from .belief import Cell
from .game import TOTAL, Instance, canonical_instances
from .private_d import HIGH, LEVELS, PrivateTables
from .spec_numbers import pair_constraints
from .sweep_l1 import COSTS, FLOOR, GATE_Q, GO, HAZARDS, Q_MAX, REOFFER_H, RULES, SCARCITY, _commit, _jsonable, \
    spearman, tier

SWEEP_VERSION = "split-deal-v03-l1-v0.1"
PRIORS = ("independent", "constrained")
BANKS = ("high", "no_high")                       # with HIGH (gated), without (the control)
PAIRS = [f"{a}/{b}" for a in LEVELS for b in LEVELS]   # A's level / B's level
ZERO = Fraction(0)


def _prior_mask(tab: PrivateTables, ta, prior: str) -> int:
    if prior == "independent":
        return tab.full
    return sum(1 << i for i, tb in enumerate(tab.types) if pair_constraints(ta[0], tb[0]))


def _blank_stats():
    return {"pairs": 0, "asks": 0, "high": 0, "walks": 0, "agree": 0, "e_n": 0, "e_sum": ZERO, "c_n": 0,
            "c_sum": ZERO}


def play(cell: Cell, tab: PrivateTables, b: int, q: int):
    """The exact policy against B type ``b``: (questions asked, allocation agreed or None, walked).

    The same moves as ``Cell.rollout(b, q, "exact")``, keeping the questions."""
    S, asked = cell.prior, []
    while q > 0:
        j = cell.words(S, q)[1]
        if j is None:
            break
        S = S & tab.yes[j] if (tab.yes[j] >> b) & 1 else S & ~tab.yes[j]
        q -= 1
        asked.append(j)
    a = cell.final(S)[1]
    if a is None:
        return asked, None, True
    return asked, (a if (tab.acc[a] >> b) & 1 else None), False


def solve_group(args) -> dict:
    counts, bank, prior = args
    inst = Instance(tuple(counts))
    tab = PrivateTables(inst, high_question=(bank == "high"))
    j_high = tab.questions.index(HIGH) if bank == "high" else None
    nb = len(tab.types)
    s_star = [[tab.s_star(ta, tb) for tb in tab.types] for ta in tab.types]

    weight = 0
    v_words = {c: [ZERO] * (Q_MAX + 1) for c in COSTS}
    v_words_common = [ZERO] * (Q_MAX + 1)
    stats = {c: [_blank_stats() for _ in range(Q_MAX + 1)] for c in COSTS}
    v_reveal = ZERO
    v_greedy = {r: [ZERO] * (Q_MAX + 1) for r in RULES}
    v_reoffer = {(h, asks): ZERO for h in HAZARDS for asks in (False, True)}
    by_pair = {p: {"n": 0, "none": ZERO, "words": ZERO, "none_c": ZERO, "words_c": ZERO} for p in PAIRS}

    for ia, ta in enumerate(tab.types):
        mask = _prior_mask(tab, ta, prior)
        n = mask.bit_count()
        if n == 0:
            continue
        weight += n
        common = tab.scale(ta) / TOTAL               # Y on the own scale -> Y on the common scale of 10
        bs = [b for b in range(nb) if (mask >> b) & 1]
        for c in COSTS:
            cell = Cell(tab, ta, mask, c)
            for q in range(Q_MAX + 1):
                v = cell.words(mask, q)[0]
                v_words[c][q] += n * v
                if c == 0:
                    v_words_common[q] += n * v * common
                st = stats[c][q]
                realized = {}
                for b in bs:
                    asked, agreed, walked = play(cell, tab, b, q)
                    st["pairs"] += 1
                    st["asks"] += len(asked)
                    st["high"] += j_high in asked
                    st["walks"] += walked
                    st["agree"] += agreed is not None
                    realized[b] = cell.gain[agreed] if agreed is not None else ZERO
                    ss = s_star[ia][b]
                    tb = tab.types[b]
                    if agreed is not None:
                        a = tab.allocs[agreed]
                        ua, ub = inst.utility(ta[0], a), inst.utility(tb[0], inst.rest(a))
                        s = ua + ub - tab.d(ta) - tab.d(tb)
                        if s > 0:
                            st["c_n"] += 1
                            st["c_sum"] += (ua - tab.d(ta)) / s
                    if ss:
                        st["e_n"] += 1
                        if agreed is not None:
                            st["e_sum"] += s / ss
                if c == 0 and q in (0, GATE_Q):
                    key = "none" if q == 0 else "words"
                    for b in bs:
                        p = by_pair[f"{ta[1]}/{tab.types[b][1]}"]
                        p[key] += realized[b]
                        p[key + "_c"] += realized[b] * common
                        if q == 0:
                            p["n"] += 1
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

    pair_w = {p: {"W": (x["words"] - x["none"]) / x["n"], "W_common": (x["words_c"] - x["none_c"]) / x["n"],
                  "V_none": x["none"] / x["n"], "pairs": x["n"]} for p, x in by_pair.items() if x["n"]}
    return {"instance": inst.key, "bank": bank, "prior": prior, "n_types": nb, "pairs": weight,
            "v_reveal": mean(v_reveal),
            "v_words": {str(c): [mean(v) for v in v_words[c]] for c in COSTS},
            "v_words_common": [mean(v) for v in v_words_common],
            "v_greedy": {r: [mean(v) for v in v_greedy[r]] for r in RULES},
            "v_reoffer": {str(h): {"offers": mean(v_reoffer[(h, False)]),
                                   "asks_and_offers": mean(v_reoffer[(h, True)])} for h in HAZARDS},
            "by_level_pair": pair_w,
            "policy": {str(c): [_finish_stats(s) for s in stats[c]] for c in COSTS}}


def _finish_stats(s: dict) -> dict:
    p = s["pairs"]
    return {"asks_per_episode": Fraction(s["asks"], p), "high_share": Fraction(s["high"], p),
            "walk_share": Fraction(s["walks"], p), "agree_share": Fraction(s["agree"], p),
            "mean_E": s["e_sum"] / s["e_n"] if s["e_n"] else None, "pairs_with_E": s["e_n"],
            "mean_C_A": s["c_sum"] / s["c_n"] if s["c_n"] else None, "agreements_with_C_A": s["c_n"]}


# -- pooling, gate, JSON ------------------------------------------------------------------

POLICY_FIELDS = ("asks_per_episode", "high_share", "walk_share", "agree_share", "mean_E", "mean_C_A")


def _pool(groups: List[dict]) -> dict:
    """Equal-weight mean over instances of every value (and policy statistic) in the groups."""
    k = len(groups)

    def avg(get):
        vals = [get(g) for g in groups]
        if any(v is None for v in vals):
            return None
        return sum(vals, ZERO) / k

    pairs = [p for p in PAIRS if all(p in g["by_level_pair"] for g in groups)]
    return {"v_reveal": avg(lambda g: g["v_reveal"]),
            "v_words": {c: [avg(lambda g, c=c, q=q: g["v_words"][c][q]) for q in range(Q_MAX + 1)]
                        for c in groups[0]["v_words"]},
            "v_words_common": [avg(lambda g, q=q: g["v_words_common"][q]) for q in range(Q_MAX + 1)],
            "v_greedy": {r: [avg(lambda g, r=r, q=q: g["v_greedy"][r][q]) for q in range(Q_MAX + 1)] for r in RULES},
            "v_reoffer": {h: {kk: avg(lambda g, h=h, kk=kk: g["v_reoffer"][h][kk]) for kk in ("offers", "asks_and_offers")}
                          for h in groups[0]["v_reoffer"]},
            "by_level_pair": {p: {f: avg(lambda g, p=p, f=f: g["by_level_pair"][p][f]) for f in ("W", "W_common", "V_none")}
                              for p in pairs},
            "policy": {c: [{f: avg(lambda g, c=c, q=q, f=f: g["policy"][c][q][f]) for f in POLICY_FIELDS}
                           for q in range(Q_MAX + 1)] for c in groups[0]["policy"]}}


def d_words(block: dict, cost: str = "0", q: int = GATE_Q) -> Fraction:
    return block["v_words"][cost][q] - block["v_words"][cost][0]


def gate(block: dict, control: dict) -> dict:
    """``block``: pooled, independent prior, with HIGH. ``control``: the same without HIGH (ungated)."""
    w = d_words(block)
    verdict = tier(w)
    re0 = block["v_reoffer"]["0"]
    d_re = re0["asks_and_offers"] - re0["offers"]
    w_common = block["v_words_common"][GATE_Q] - block["v_words_common"][0]
    return {"W": float(w), "W_exact": str(w), "verdict": verdict,
            "W_common": float(w_common), "W_control": float(d_words(control)),
            "bars": {"go": str(GO), "floor": str(FLOOR), "scarcity": str(SCARCITY)},
            "reoffer_d_words_h0": float(d_re), "reoffer_d_words_h0_exact": str(d_re),
            "scarcity_conditional": verdict != "no-go" and d_re < SCARCITY}


def compute(instances: Optional[List[Instance]] = None, workers: int = 1) -> dict:
    instances = instances or canonical_instances()
    jobs = [(list(i.counts), bk, p) for i, bk, p in itertools.product(instances, BANKS, PRIORS)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            groups = list(ex.map(solve_group, jobs))
    else:
        groups = [solve_group(j) for j in jobs]
    pooled = {bk: {p: _pool([g for g in groups if g["bank"] == bk and g["prior"] == p]) for p in PRIORS}
              for bk in BANKS}
    main = [g for g in groups if g["bank"] == "high" and g["prior"] == "independent"]
    type_count = {"n_types": [g["n_types"] for g in main], "W": [float(d_words(g)) for g in main],
                  "spearman": (spearman([g["n_types"] for g in main], [float(d_words(g)) for g in main])
                               if len({g["n_types"] for g in main}) > 1 else None)}
    return {"sweep": SWEEP_VERSION,
            "config": {"q_max": Q_MAX, "gate_q": GATE_Q, "costs": [str(c) for c in COSTS],
                       "hazards": [str(h) for h in HAZARDS], "reoffer_moves": REOFFER_H, "priors": list(PRIORS),
                       "banks": list(BANKS), "levels": {k: str(v) for k, v in LEVELS.items()},
                       "level_pairs": PAIRS, "instances": [i.key for i in instances]},
            "gate": gate(pooled["high"]["independent"], pooled["no_high"]["independent"]),
            "type_count": type_count,
            "pooled": _jsonable(pooled),
            "groups": _jsonable(groups)}


def run(args) -> int:
    t0 = time.time()
    doc = compute(workers=args.workers)
    doc["commit"] = _commit()
    doc["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    g = doc["gate"]
    print(f"wrote {args.out} ({doc['seconds']} s); W {g['W']:.4f} -> {g['verdict']}; "
          f"scarcity D {g['reoffer_d_words_h0']:.4f}; control W {g['W_control']:.4f}")
    return 0


# -- note -----------------------------------------------------------------------------------

def _f(x, nd=4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


VERDICT_TEXT = {
    "go": "v0.3 keeps a words value worth a live study (its registration still needs its own power "
          "calculation, NEG-33)",
    "conditional": "offline work continues; live spend needs a power calculation showing an expected 95 % "
                   "half-width below W / 2 (NEG-33)",
    "no-go": "by the registered consequence, the one-sided words study stays on v0.2, which passed SD-W, and v0.3 "
             "is used only for gate SD3-D",
}
PAIR_LABEL = {"lo/lo": "A lo, B lo", "lo/hi": "A lo, B hi", "hi/lo": "A hi, B lo", "hi/hi": "A hi, B hi"}


def render(doc: dict, path: str) -> str:
    g, cfg, P = doc["gate"], doc["config"], doc["pooled"]
    main = P["high"]["independent"]
    L = ["# Split Deal v0.3 — L1 words-value sweep and gate SD3-W", "",
         f"_Rendered by `sweep_l1_v03.py render` from `{path}` (sweep `{doc['sweep']}`, commit `{doc['commit']}`, "
         f"{doc['seconds']} s). Exact on the model: no sampling, no API; every number is formatted from the JSON. "
         "Gate as registered in `DESIGN_SPLIT_V03.md` §5 before this ran; its operational reading is fixed in the "
         "docstring of `sweep_l1_v03.py`._", "",
         "Protocol 4.1 on the v0.3 game: each side's type is its item values and a private outside-option level "
         "(lo: d = 4, hi: d = 5.5, ½ each). A asks up to Q of the 13 bank questions (HIGH included), answered "
         "truthfully, then makes one binding offer or walks; B accepts iff its utility is at least its own d. Y is "
         "net gain over one's own d as a share of 10 − one's own d, averaged over both types uniformly within an "
         "instance, then over the nine instances with equal weight.", "",
         "## Gate SD3-W", "",
         f"W = D_words(3) = **{g['W']:.4f}** Y (exact {g['W_exact']}).", "",
         f"Bars: go at W ≥ {float(Fraction(g['bars']['go'])):.2f} Y, conditional from "
         f"{float(Fraction(g['bars']['floor'])):.2f} Y.", "",
         f"**Verdict: {g['verdict']}** — {VERDICT_TEXT[g['verdict']]}.", "",
         f"Re-offer check (H = {cfg['reoffer_moves']}, h = 0): asks-and-offers minus offers-only = "
         f"{g['reoffer_d_words_h0']:.4f} Y against the {float(Fraction(g['bars']['scarcity'])):.2f} Y line. "
         + ("Below it, and the gate did not fail: as in v0.2, the words matter *because* offers are scarce, so any "
            "live protocol keeps the single-offer rule and any write-up says the result is conditional on it."
            if g["scarcity_conditional"] else
            ("At or above it: the words keep their value even when offers may be repeated."
             if g["verdict"] != "no-go" else "Not applicable: the gate failed.")), "",
         "Diagnostics, deciding nothing: on the common scale (U − own d)/10, W = "
         f"{g['W_common']:.4f}; in the control bank without HIGH, W = {g['W_control']:.4f}. v0.2 gave 0.1659 (easy) "
         "and 0.1063 (hard) (`SWEEP_SPLIT_L1.md`).", "",
         "## W by level pair (diagnostic)", "",
         "From the exact policy's realized gains against each B type, Q = 3 minus Q = 0, ask cost 0, independent "
         "prior.", "",
         "| level pair | V_none | W (own scale) | W (common scale) | W without HIGH |", "|---|---|---|---|---|"]
    ctrl = P["no_high"]["independent"]["by_level_pair"]
    for p in cfg["level_pairs"]:
        x = main["by_level_pair"].get(p)
        if x is None:
            continue
        L.append(f"| {PAIR_LABEL[p]} | {x['V_none']:.4f} | {x['W']:.4f} | {x['W_common']:.4f} | "
                 f"{_f(ctrl.get(p, {}).get('W'))} |")
    L.append("")
    for bank, btitle in (("high", "13-question bank"), ("no_high", "control, 12 questions without HIGH")):
        for prior in ("independent", "constrained"):
            title = "independent uniform prior" if prior == "independent" else \
                "value-pair constraints (sensitivity prior; never gates)"
            b = P[bank][prior]
            v, vc, pol = b["v_words"]["0"], b["v_words_common"], b["policy"]["0"]
            L += [f"## By Q, ask cost 0 — {btitle}, {title}", "", f"V_reveal = {b['v_reveal']:.4f}.", "",
                  "| Q | V_words | D_words | D_words common scale | D_rest | questions / ep | asks HIGH | walk | "
                  "agree | mean E | mean C_A |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for q in range(cfg["q_max"] + 1):
                st = pol[q]
                L.append(f"| {q} | {v[q]:.4f} | {v[q] - v[0]:.4f} | {vc[q] - vc[0]:.4f} | {b['v_reveal'] - v[q]:.4f} | "
                         f"{st['asks_per_episode']:.2f} | "
                         f"{_pct(st['high_share']) if bank == 'high' else '—'} | {_pct(st['walk_share'])} | "
                         f"{_pct(st['agree_share'])} | {_f(st['mean_E'], 3)} | {_f(st['mean_C_A'], 3)} |")
            L.append("")
    L += ["`asks HIGH` is the share of episodes in which the exact policy asks HIGH. `walk` and `agree` are shares of "
          "episodes; `mean E` is joint efficiency S(o)/S* over pairs with S* > 0 (no deal counts 0) and `mean C_A` A's "
          "share of the surplus over agreements. Ties go to not asking.", "",
          "## Ask cost (13-question bank, independent prior)", "",
          "| cost (Y) | D_words(3) | questions / ep at Q = 3 | D_words(6) |", "|---|---|---|---|"]
    for c in cfg["costs"]:
        v = main["v_words"][c]
        L.append(f"| {c} | {v[3] - v[0]:.4f} | {main['policy'][c][3]['asks_per_episode']:.2f} | {v[6] - v[0]:.4f} |")
    L += ["", "## Greedy gap, ask cost 0 (13-question bank, independent prior)", "",
          "| Q | exact | EIG-greedy | value-greedy | gap EIG | gap value |", "|---|---|---|---|---|---|"]
    for q in range(1, cfg["q_max"] + 1):
        ex, ge, gv = main["v_words"]["0"][q], main["v_greedy"]["eig"][q], main["v_greedy"]["value"][q]
        L.append(f"| {q} | {ex:.4f} | {ge:.4f} | {gv:.4f} | {ex - ge:.4f} | {ex - gv:.4f} |")
    L += ["", f"## Re-offer variant (H = {cfg['reoffer_moves']}, ask cost 0)", "",
          "| bank | prior | h | offers only | asks and offers | D |", "|---|---|---|---|---|---|"]
    for bank in ("high", "no_high"):
        for prior in ("independent", "constrained"):
            for h in cfg["hazards"]:
                x = P[bank][prior]["v_reoffer"][h]
                L.append(f"| {bank} | {prior} | {h} | {x['offers']:.4f} | {x['asks_and_offers']:.4f} | "
                         f"{x['asks_and_offers'] - x['offers']:.4f} |")
    groups = {(x["instance"], x["bank"], x["prior"]): x for x in doc["groups"]}
    L += ["", "## By instance (ask cost 0, independent prior)", "",
          "| instance | types | W | W without HIGH | D_rest(3) | asks HIGH at Q = 3 |", "|---|---|---|---|---|---|"]
    order = sorted(cfg["instances"], key=lambda k: -groups[(k, "high", "independent")]["n_types"])
    for k in order:
        e, c = groups[(k, "high", "independent")], groups[(k, "no_high", "independent")]
        we = e["v_words"]["0"][3] - e["v_words"]["0"][0]
        wc = c["v_words"]["0"][3] - c["v_words"]["0"][0]
        L.append(f"| {k} | {e['n_types']} | {we:.4f} | {wc:.4f} | {e['v_reveal'] - e['v_words']['0'][3]:.4f} | "
                 f"{_pct(e['policy']['0'][3]['high_share'])} |")
    rho = doc["type_count"]["spearman"]
    if rho is not None:
        L += ["", f"Spearman rank correlation of W with type count over the nine instances: {rho:+.2f} (descriptive)."]
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
