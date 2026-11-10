"""L2 sweep of Split Deal and gate SD-D (`DESIGN_SPLIT.md` §5, §7). Offline, exact, no API.

For every instance, regime and pi, and every pair of types (independent uniform prior), plays A1
against B0 (``twosided.py``) under T0-T3 and the four cells of the 2 x 2 disclosure game, and records
each side's Y split by who proposed, agreement, and the L0 joint efficiency E of the outcome. Values
are averaged over pairs within an instance, then over the nine instances with equal weight.

Gate SD-D scores the four predictions registered in the note (operational reading fixed in the
docstring of ``twosided.py`` before this ran):

    D1  E(T2) >= E(T1) >= E(T0)                        easy, pi = 1/2 (hard reported)
    D2  Y_A(T2) < Y_A(T1), = (1 - pi) x A's lost rent    easy, pi = 1/2 (hard reported)
    D3  the 2 x 2 disclosure game is a strict prisoner's dilemma for both players, in some regime
    D4  in T3, A1 answers some questions and declines others, depending on its type   easy, pi = 1/2

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l2 run --out results/split_l2_20261110/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l2 render results/split_l2_20261110/sweep.json \
        --note llm_rl_scripts/split_deal/SWEEP_SPLIT_L2.md
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
from typing import Dict, List, Optional

from .belief import Tables
from .game import REGIMES, Instance, canonical_instances
from .sweep_l1 import _commit, _s_star
from .twosided import GAME, STRUCTURES, Program, Shared

SWEEP_VERSION = "split-deal-l2-v0.1"
Q = 3
PIS = (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4))
PRIMARY_PI = Fraction(1, 2)
ARMS = list(STRUCTURES) + [f"game:{a}/{b}" for a, b in GAME]
ZERO = Fraction(0)
FIELDS = ("YA", "YB", "YA_prop", "YA_resp", "YB_prop", "YB_resp", "agree_a", "agree_b", "E")


def _structure(arm: str):
    if arm.startswith("game:"):
        a, b = arm[5:].split("/")
        return GAME[(a, b)]
    return STRUCTURES[arm]


def solve_group(args) -> dict:
    counts, regime, pi_s = args
    pi = Fraction(pi_s)
    inst = Instance(tuple(counts))
    tab = Tables(inst, regime)
    sh = Shared(tab)
    n = len(tab.types)
    s_star = [[_s_star(tab, va, vb) for vb in tab.types] for va in tab.types]
    acc = {arm: {f: ZERO for f in FIELDS} for arm in ARMS}
    root = {arm: ZERO for arm in ARMS}
    e_pairs = 0
    exact_game = {cell: {"YA": ZERO, "YB": ZERO} for cell in GAME}
    decisions = []                     # T3: (A type index, question, answered, strict)
    for ia in range(n):
        for arm in ARMS:
            prog = Program(sh, ia, _structure(arm), Q, pi)
            root[arm] += prog.root()
            a = acc[arm]
            for b in range(n):
                out = prog.rollout(b)
                ka, kb = out["a_deal"], out["b_deal"]
                ya_p = prog.me.gain[ka] if ka is not None else ZERO
                yb_p = sh.seat[b].gain[sh.mirror[ka]] if ka is not None else ZERO
                ya_r = prog.me.gain[kb] if kb is not None else ZERO
                yb_r = sh.seat[b].gain[sh.mirror[kb]] if kb is not None else ZERO
                a["YA_prop"] += ya_p
                a["YB_resp"] += yb_p
                a["YA_resp"] += ya_r
                a["YB_prop"] += yb_r
                a["YA"] += pi * ya_p + (1 - pi) * ya_r
                a["YB"] += pi * yb_p + (1 - pi) * yb_r
                a["agree_a"] += ka is not None
                a["agree_b"] += kb is not None
                ss = s_star[ia][b]
                if ss:
                    if arm == ARMS[0]:
                        e_pairs += 1
                    e = ZERO
                    for k, w in ((ka, pi), (kb, 1 - pi)):
                        if k is not None:
                            sa = inst.utility(tab.types[ia], tab.allocs[k])
                            sb = inst.utility(tab.types[b], inst.rest(tab.allocs[k]))
                            e += w * (sa + sb - 2 * tab.d) / ss
                    a["E"] += e
                if arm == "T3":
                    decisions += [(ia, j, ans, strict) for j, ans, strict in out["decisions"]]
        # the 2 x 2 game with B asking by the exact program (decomposed: no coupling without reciprocity)
        me = sh.seat[ia]
        for (a_pol, b_pol) in GAME:
            ya = yb = ZERO
            for b in range(n):
                ka = me.rollout(b, Q, "exact")[1] if b_pol == "all" else me.final(tab.full)[1]
                if ka is not None and not (tab.acc[ka] >> b) & 1:
                    ka = None
                kb_seat = sh.seat[b].rollout(ia, Q, "exact")[1] if a_pol == "all" else sh.seat[b].final(tab.full)[1]
                kb = None
                if kb_seat is not None and (tab.acc[kb_seat] >> ia) & 1:
                    kb = sh.mirror[kb_seat]
                ya += pi * (me.gain[ka] if ka is not None else 0) + (1 - pi) * (me.gain[kb] if kb is not None else 0)
                yb += pi * (sh.seat[b].gain[sh.mirror[ka]] if ka is not None else 0) + \
                    (1 - pi) * (sh.seat[b].gain[kb_seat] if kb is not None else 0)
            exact_game[(a_pol, b_pol)]["YA"] += ya
            exact_game[(a_pol, b_pol)]["YB"] += yb
    pairs = n * n
    arms = {}
    for arm in ARMS:
        d = {f: acc[arm][f] / pairs for f in FIELDS if f != "E"}
        d["E"] = acc[arm]["E"] / e_pairs
        d["root"] = root[arm] / n
        arms[arm] = d
    game_exact = {f"{a}/{b}": {k: v / pairs for k, v in exact_game[(a, b)].items()} for a, b in GAME}
    return {"instance": inst.key, "regime": regime, "pi": pi_s, "n_types": n, "pairs": pairs, "pairs_with_E": e_pairs,
            "arms": arms, "game_exact_b": game_exact,
            "t3_decisions": [[ia, j, ans, strict] for ia, j, ans, strict in decisions]}


# -- pooling and the gate ------------------------------------------------------------------

def _pool(groups: List[dict]) -> dict:
    k = len(groups)
    arms = {arm: {f: sum((g["arms"][arm][f] for g in groups), ZERO) / k for f in groups[0]["arms"][arm]}
            for arm in ARMS}
    game_exact = {c: {f: sum((g["game_exact_b"][c][f] for g in groups), ZERO) / k for f in ("YA", "YB")}
                  for c in groups[0]["game_exact_b"]}
    return {"arms": arms, "game_exact_b": game_exact}


def _pd(m: Dict[str, Dict[str, Fraction]]) -> dict:
    """m maps 'a/b' (A's answering / B's answering) to {'YA', 'YB'}. Strict PD orderings for each player."""
    a = {"temptation": m["none/all"]["YA"], "reward": m["all/all"]["YA"], "punishment": m["none/none"]["YA"],
         "sucker": m["all/none"]["YA"]}
    b = {"temptation": m["all/none"]["YB"], "reward": m["all/all"]["YB"], "punishment": m["none/none"]["YB"],
         "sucker": m["none/all"]["YB"]}

    def holds(p):
        return p["temptation"] > p["reward"] > p["punishment"] > p["sucker"]

    return {"A": {k: float(v) for k, v in a.items()}, "B": {k: float(v) for k, v in b.items()},
            "A_holds": holds(a), "B_holds": holds(b), "holds": holds(a) and holds(b),
            "withholding_dominant_A": a["temptation"] > a["reward"] and a["punishment"] > a["sucker"],
            "withholding_dominant_B": b["temptation"] > b["reward"] and b["punishment"] > b["sucker"],
            "mutual_disclosure_beats_mutual_withholding": a["reward"] > a["punishment"] and b["reward"] > b["punishment"]}


def _game_block(arms: dict) -> dict:
    return {f"{a}/{b}": {"YA": arms[f"game:{a}/{b}"]["YA"], "YB": arms[f"game:{a}/{b}"]["YB"]} for a, b in GAME}


def _d4(groups: List[dict]) -> dict:
    answered = declined_strict = declined_tie = 0
    per_q: Dict[tuple, Dict[str, set]] = {}
    for g in groups:
        for ia, j, ans, strict in g["t3_decisions"]:
            slot = per_q.setdefault((g["instance"], j), {"answer": set(), "decline": set()})
            if ans:
                answered += 1
                slot["answer"].add(ia)
            else:
                declined_strict += strict
                declined_tie += not strict
                slot["decline"].add(ia)
    mixed = sorted(f"{inst}:q{j}" for (inst, j), s in per_q.items() if s["answer"] and s["decline"])
    total = answered + declined_strict + declined_tie
    return {"decisions": total, "answered": answered, "declined_strict": declined_strict, "declined_tie": declined_tie,
            "questions_seen": len(per_q), "questions_mixed": len(mixed), "mixed": mixed,
            "holds": answered > 0 and (declined_strict + declined_tie) > 0 and len(mixed) > 0}


def gate(pooled: dict, groups: List[dict]) -> dict:
    """``pooled[regime][pi]`` blocks (Fractions); ``groups`` the raw groups."""
    p = str(PRIMARY_PI)
    out = {}
    for r in REGIMES:
        arms = pooled[r][p]["arms"]
        E = {t: arms[t]["E"] for t in ("T0", "T1", "T2", "T3")}
        d1 = E["T2"] >= E["T1"] >= E["T0"]
        ya1, ya2 = arms["T1"]["YA"], arms["T2"]["YA"]
        rent = arms["T1"]["YA_resp"] - arms["T2"]["YA_resp"]
        identity = (ya1 - ya2) == (1 - PRIMARY_PI) * rent and arms["T1"]["YA_prop"] == arms["T2"]["YA_prop"]
        out[r] = {"D1": {"E": {t: float(v) for t, v in E.items()}, "holds": d1},
                  "D2": {"YA_T1": float(ya1), "YA_T2": float(ya2), "lost_rent": float(rent),
                         "identity_exact": identity, "holds": ya2 < ya1},
                  "D3": _pd(_game_block(arms)),
                  "D3_exact_b": _pd(pooled[r][p]["game_exact_b"]),
                  "D4": _d4([g for g in groups if g["regime"] == r and g["pi"] == p])}
    verdict = {"D1": out["easy"]["D1"]["holds"], "D2": out["easy"]["D2"]["holds"],
               "D3": any(out[r]["D3"]["holds"] for r in REGIMES), "D4": out["easy"]["D4"]["holds"]}
    return {"by_regime": out, "verdict": verdict, "primary": {"pi": p, "q": Q, "regime_for_D1_D2_D4": "easy"}}


def _jsonable(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def compute(instances: Optional[List[Instance]] = None, workers: int = 1) -> dict:
    instances = instances or canonical_instances()
    jobs = [(list(i.counts), r, str(p)) for i, r, p in itertools.product(instances, REGIMES, PIS)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            groups = list(ex.map(solve_group, jobs))
    else:
        groups = [solve_group(j) for j in jobs]
    pooled = {r: {str(p): _pool([g for g in groups if g["regime"] == r and g["pi"] == str(p)]) for p in PIS}
              for r in REGIMES}
    doc = {"sweep": SWEEP_VERSION,
           "config": {"q": Q, "pis": [str(p) for p in PIS], "primary_pi": str(PRIMARY_PI), "arms": ARMS,
                      "regimes": {r: str(v) for r, v in REGIMES.items()}, "instances": [i.key for i in instances],
                      "prior": "independent"},
           "gate": gate(pooled, groups),
           "pooled": _jsonable(pooled),
           "groups": [_jsonable({k: v for k, v in g.items() if k != "t3_decisions"}) for g in groups]}
    return doc


def run(args) -> int:
    t0 = time.time()
    doc = compute(workers=args.workers)
    doc["commit"] = _commit()
    doc["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    v = doc["gate"]["verdict"]
    print(f"wrote {args.out} ({doc['seconds']} s); " + ", ".join(f"{k} {'holds' if x else 'fails'}" for k, x in v.items()))
    return 0


# -- note --------------------------------------------------------------------------------------

ARM_LABEL = {"T0": "T0 no queries", "T1": "T1 private (A asks)", "T2": "T2 public (both ask, both answer)",
             "T3": "T3 reciprocal (A chooses)"}


def render(doc: dict, path: str) -> str:
    g, cfg, P = doc["gate"], doc["config"], doc["pooled"]
    v = g["verdict"]
    yes = lambda b: "**holds**" if b else "**fails**"   # noqa: E731
    L = ["# Split Deal — L2 two-sided program and gate SD-D", "",
         f"_Rendered by `sweep_l2.py render` from `{path}` (sweep `{doc['sweep']}`, commit `{doc['commit']}`, "
         f"{doc['seconds']} s). Exact on the model: no sampling, no API; every number is formatted from the JSON. "
         "Predictions as registered in `DESIGN_SPLIT.md` §7; their operational reading (Q, prior, how D3 and D4 "
         "are scored, tie-breaks) was fixed in the docstring of `twosided.py` before the sweep ran._", "",
         f"Protocol 4.2 with Q = {cfg['q']} rounds (A then B), then the proposer is drawn (A with probability π) and "
         "makes one binding offer. B is the rule counterpart B0 (asks by expected information gain, draws no "
         "inference from a decline, offers to maximise its own expected gain); A is A1, the exact best response to "
         "B0. Independent uniform prior. Y in units of 10 − d for each side; E is joint efficiency from the L0 oracle.",
         "", "## Gate SD-D (π = 1/2)", "",
         "| prediction | reading | verdict |", "|---|---|---|",
         f"| D1 | E(T2) ≥ E(T1) ≥ E(T0), easy | {yes(v['D1'])} |",
         f"| D2 | Y_A(T2) < Y_A(T1), easy | {yes(v['D2'])} |",
         f"| D3 | strict prisoner's dilemma for both players, in some regime | {yes(v['D3'])} |",
         f"| D4 | in T3, A1 answers some questions and declines others, by type, easy | {yes(v['D4'])} |", ""]
    for r in ("easy", "hard"):
        x = g["by_regime"][r]
        E = x["D1"]["E"]
        L += [f"### {r}", "",
              f"- **D1** {yes(x['D1']['holds'])}: E(T0) {E['T0']:.4f}, E(T1) {E['T1']:.4f}, E(T2) {E['T2']:.4f} "
              f"(E(T3) {E['T3']:.4f}).",
              f"- **D2** {yes(x['D2']['holds'])}: Y_A(T1) {x['D2']['YA_T1']:.4f}, Y_A(T2) {x['D2']['YA_T2']:.4f}; "
              f"A's lost information rent as responder {x['D2']['lost_rent']:.4f}, and the difference equals (1 − π) "
              f"times it {'exactly' if x['D2']['identity_exact'] else 'NOT exactly'}."]
        arms = P[r][str(Fraction(1, 2))]["arms"]
        if x["D2"]["lost_rent"] < 0:
            L.append(f"  The rent is negative: being known *raises* A's value as responder. An informed B0 makes offers "
                     f"A accepts more often (agreement when B proposes {100 * arms['T1']['agree_b']:.1f} % in T1, "
                     f"{100 * arms['T2']['agree_b']:.1f} % in T2), and that outweighs what B0 extracts.")
        for key, label in (("D3", "B0 asks by EIG, the registered reading"),
                           ("D3_exact_b", "B asks by the exact program")):
            d = x[key]
            L.append(f"- **{'D3' if key == 'D3' else 'D3, second reading'}** ({label}) {yes(d['holds'])}: "
                     f"A {'is' if d['A_holds'] else 'is not'} in a dilemma, B {'is' if d['B_holds'] else 'is not'}; "
                     f"withholding dominant for A {d['withholding_dominant_A']}, for B {d['withholding_dominant_B']}; "
                     f"mutual disclosure beats mutual withholding for both {d['mutual_disclosure_beats_mutual_withholding']}.")
        d4 = x["D4"]
        L += [f"- **D4** {yes(d4['holds'])}: {d4['decisions']} on-path decisions at B's questions; answered "
              f"{d4['answered']}, declined {d4['declined_strict']} strictly and {d4['declined_tie']} on a tie; "
              f"{d4['questions_mixed']} of {d4['questions_seen']} (instance, question) pairs are answered by some A "
              "types and declined by others. Choosing what to answer (T3) against always answering (T2): "
              f"Y_A {arms['T3']['YA'] - arms['T2']['YA']:+.4f}, Y_B {arms['T3']['YB'] - arms['T2']['YB']:+.4f}, "
              f"E {arms['T3']['E'] - arms['T2']['E']:+.4f}.", ""]
    if not v["D3"]:
        t = P["easy"][str(Fraction(1, 2))]["arms"]
        L += ["**Registered consequence of D3 failing in both regimes** (§7): \"this game has no negotiator's dilemma "
              "under its rules, and the two-way motivation would need a different mechanism (for example a private "
              "outside option) before a live study is worth designing.\"", "",
              "One qualification, from D4 rather than D3: the dilemma that fails is the all-or-nothing one. Question "
              "by question, A1 does withhold selectively, and in easy that is worth "
              f"{t['T3']['YA'] - t['T2']['YA']:+.4f} Y to A and {t['T3']['YB'] - t['T2']['YB']:+.4f} Y to B against "
              "always answering.", ""]
    L += ["## The 2 × 2 disclosure game (π = 1/2)", "",
          "Rows are A's answering, columns B's; each cell is (Y_A, Y_B). Temptation = withhold against a "
          "discloser, reward = both disclose, punishment = both withhold, sucker = disclose to a withholder.", ""]
    for r in ("easy", "hard"):
        for key, label in (("D3", "B0 asks by EIG"), ("D3_exact_b", "B asks by the exact program")):
            d = g["by_regime"][r][key]
            L += [f"**{r}, {label}**", "", "| | temptation | reward | punishment | sucker |", "|---|---|---|---|---|",
                  f"| A | {d['A']['temptation']:.4f} | {d['A']['reward']:.4f} | {d['A']['punishment']:.4f} | "
                  f"{d['A']['sucker']:.4f} |",
                  f"| B | {d['B']['temptation']:.4f} | {d['B']['reward']:.4f} | {d['B']['punishment']:.4f} | "
                  f"{d['B']['sucker']:.4f} |", ""]
    L += ["## Structures by π", "",
          "Y_A and Y_B are each side's expected Y; `prop` and `resp` are a side's Y when it proposes and when it "
          "responds; agreement shares by who proposed.", "",
          "| regime | π | structure | Y_A | Y_B | Y_A prop | Y_A resp | Y_B prop | Y_B resp | agree, A proposes | "
          "agree, B proposes | E |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ("easy", "hard"):
        for p in cfg["pis"]:
            for arm in ("T0", "T1", "T2", "T3"):
                a = P[r][p]["arms"][arm]
                L.append(f"| {r} | {p} | {ARM_LABEL[arm]} | {a['YA']:.4f} | {a['YB']:.4f} | {a['YA_prop']:.4f} | "
                         f"{a['YA_resp']:.4f} | {a['YB_prop']:.4f} | {a['YB_resp']:.4f} | {100 * a['agree_a']:.1f} % | "
                         f"{100 * a['agree_b']:.1f} % | {a['E']:.4f} |")
    L += ["", "## D3 by π (B0 reading)", "", "| regime | π | A in dilemma | B in dilemma |", "|---|---|---|---|"]
    for r in ("easy", "hard"):
        for p in cfg["pis"]:
            d = _pd({c: {"YA": P[r][p]["arms"][f"game:{c}"]["YA"], "YB": P[r][p]["arms"][f"game:{c}"]["YB"]}
                     for c in ("all/all", "all/none", "none/all", "none/none")})
            L.append(f"| {r} | {p} | {d['A_holds']} | {d['B_holds']} |")
    L += ["", "Not built here: B2 (a counterpart that reads A's declines and question choices) and the CFR "
          "exploitability check on the smallest instance (§5)."]
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
