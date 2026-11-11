"""L2 sweep of Split Deal v0.3 and gate SD3-D (`DESIGN_SPLIT_V03.md` §5). Offline, exact, no API.

The v0.2 L2 sweep (``sweep_l2.py``) on the v0.3 tables (``private_d.PrivateTables``), with the bank
that includes HIGH (gated) and without it (the ungated control, decision 3). For every instance, bank
and pi, and every pair of types (independent uniform prior), plays A1 against B0 (``twosided.py``,
unchanged) under T0-T3 and the four cells of the 2 x 2 disclosure game, and records each side's Y split
by who proposed and by A's level, agreement, and the joint efficiency E of the outcome.

Operational reading of gate SD3-D, fixed here before the sweep ran. Shared by every prediction:
Q = 3, pi = 1/2 (1/4 and 3/4 reported), the 13-question bank, B0 the counterpart (decision 5), the
independent uniform prior (as in v0.2: under the value-pair prior B0's questions would carry
information about B's type). Y on each side's own scale (decision 2); pooled means weigh the nine
instances equally and, within an instance, A's type and B's type uniformly. Common-scale means,
(U - own d) / 10, are reported and decide nothing.

    D1'    E(T2) >= E(T1) >= E(T0), pooled.
    D2'    Y_A(T2) < Y_A(T1), pooled.
    D2'-lo the same with A's type restricted to the lo level (B's type still uniform); the hi-level
           sign is reported, not predicted.
    D3'    the 2 x 2 game (each side answers everything, HIGH included, or nothing) is a strict
           prisoner's dilemma for both players on pooled means, with B0 asking by EIG: for each player,
           withhold-vs-disclose > both disclose > both withhold > disclose-vs-withhold. The second
           reading, B asking by the exact program, is reported beside it.
    D4'    over A1's on-path decisions in T3 at which B0 asks HIGH (each (A type, B type) pair's episode
           once, counts pooled over instances, as v0.2's D4): the share answered is higher for hi-level
           A types than for lo-level ones, AND lo-level A types decline more than half of theirs.
           Answering counts only when strictly better; ties go to declining. Also reported split by the
           round in which B0 asks HIGH on its T2 path (the instance's column in §3).
    D5'    over the same T3 decisions, the share answered is higher for value questions than for HIGH.

The registered consequences (§5) are rendered with whichever of D2'-lo, D3', D4' they depend on.
The control runs the same sweep with no prediction; D4' and D5' do not apply to it.

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l2_v03 run --out results/split_v03_l2_20261110/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_l2_v03 render results/split_v03_l2_20261110/sweep.json \\
        --note llm_rl_scripts/split_deal/SWEEP_SPLIT_V03_L2.md
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

from .game import TOTAL, Instance, canonical_instances
from .private_d import HIGH, LEVELS, PrivateTables
from .spec_numbers_v03 import high_rounds
from .sweep_l1 import _commit
from .sweep_l2 import ARMS, PIS, PRIMARY_PI, Q, _jsonable, _pd, _structure
from .twosided import GAME, Program, Shared

SWEEP_VERSION = "split-deal-v03-l2-v0.1"
BANKS = ("high", "no_high")
LEVEL_KEYS = tuple(LEVELS)                       # ("lo", "hi")
ZERO = Fraction(0)
FIELDS = ("YA", "YB", "YA_prop", "YA_resp", "YB_prop", "YB_resp", "agree_a", "agree_b", "YA_c", "YB_c")


def solve_group(args) -> dict:
    counts, bank, pi_s = args
    pi = Fraction(pi_s)
    inst = Instance(tuple(counts))
    tab = PrivateTables(inst, high_question=(bank == "high"))
    sh = Shared(tab)
    n = len(tab.types)
    j_high = tab.questions.index(HIGH) if bank == "high" else None
    s_star = [[tab.s_star(ta, tb) for tb in tab.types] for ta in tab.types]
    common = [tab.scale(t) / TOTAL for t in tab.types]
    # per A level: sums of each field over (A type of that level, B type) pairs; E sums and counts
    acc = {arm: {lv: {f: ZERO for f in FIELDS} for lv in LEVEL_KEYS} for arm in ARMS}
    e_sum = {arm: {lv: ZERO for lv in LEVEL_KEYS} for arm in ARMS}
    e_n = {lv: 0 for lv in LEVEL_KEYS}
    root = {arm: {lv: ZERO for lv in LEVEL_KEYS} for arm in ARMS}
    exact_game = {cell: {lv: {"YA": ZERO, "YB": ZERO} for lv in LEVEL_KEYS} for cell in GAME}
    decisions = []                     # T3: (A type index, A level, question, is HIGH, answered, strict, round)
    for ia, ta in enumerate(tab.types):
        lv = ta[1]
        for arm in ARMS:
            prog = Program(sh, ia, _structure(arm), Q, pi)
            root[arm][lv] += prog.root()
            a = acc[arm][lv]
            for b in range(n):
                out = prog.rollout(b)
                ka, kb = out["a_deal"], out["b_deal"]
                ya_p = prog.me.gain[ka] if ka is not None else ZERO
                yb_p = sh.seat[b].gain[sh.mirror[ka]] if ka is not None else ZERO
                ya_r = prog.me.gain[kb] if kb is not None else ZERO
                yb_r = sh.seat[b].gain[sh.mirror[kb]] if kb is not None else ZERO
                ya = pi * ya_p + (1 - pi) * ya_r
                yb = pi * yb_p + (1 - pi) * yb_r
                a["YA_prop"] += ya_p
                a["YB_resp"] += yb_p
                a["YA_resp"] += ya_r
                a["YB_prop"] += yb_r
                a["YA"] += ya
                a["YB"] += yb
                a["YA_c"] += ya * common[ia]
                a["YB_c"] += yb * common[b]
                a["agree_a"] += ka is not None
                a["agree_b"] += kb is not None
                ss = s_star[ia][b]
                if ss:
                    if arm == ARMS[0]:
                        e_n[lv] += 1
                    e = ZERO
                    tb = tab.types[b]
                    for k, w in ((ka, pi), (kb, 1 - pi)):
                        if k is not None:
                            sa = inst.utility(ta[0], tab.allocs[k])
                            sb = inst.utility(tb[0], inst.rest(tab.allocs[k]))
                            e += w * (sa + sb - tab.d(ta) - tab.d(tb)) / ss
                    e_sum[arm][lv] += e
                if arm == "T3":
                    decisions += [(ia, lv, j, j == j_high, ans, strict, rd)
                                  for (j, ans, strict), rd in zip(out["decisions"], out["decision_rounds"])]
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
            exact_game[(a_pol, b_pol)][lv]["YA"] += ya
            exact_game[(a_pol, b_pol)][lv]["YB"] += yb
    n_lv = n // len(LEVEL_KEYS)                    # A types per level
    arms = {}
    for arm in ARMS:
        d = {}
        for lv in LEVEL_KEYS:
            x = {f: acc[arm][lv][f] / (n_lv * n) for f in FIELDS}
            x["E"] = e_sum[arm][lv] / e_n[lv] if e_n[lv] else None
            x["root"] = root[arm][lv] / n_lv
            d[lv] = x
        x = {f: sum((acc[arm][lv][f] for lv in LEVEL_KEYS), ZERO) / (n * n) for f in FIELDS}
        x["E"] = sum(e_sum[arm].values(), ZERO) / sum(e_n.values())
        x["root"] = sum(root[arm].values(), ZERO) / n
        d["all"] = x
        arms[arm] = d
    game_exact = {}
    for a_pol, b_pol in GAME:
        c = exact_game[(a_pol, b_pol)]
        game_exact[f"{a_pol}/{b_pol}"] = {
            "all": {k: sum((c[lv][k] for lv in LEVEL_KEYS), ZERO) / (n * n) for k in ("YA", "YB")},
            **{lv: {k: c[lv][k] / (n_lv * n) for k in ("YA", "YB")} for lv in LEVEL_KEYS}}
    return {"instance": inst.key, "bank": bank, "pi": pi_s, "n_types": n, "pairs": n * n,
            "high_round_t2": high_rounds(tab) if bank == "high" else None,
            "arms": arms, "game_exact_b": game_exact,
            "t3_decisions": [list(x) for x in decisions]}


# -- pooling and the gate ------------------------------------------------------------------

SCOPES = ("all",) + LEVEL_KEYS


def _pool(groups: List[dict]) -> dict:
    k = len(groups)
    arms = {arm: {s: {f: sum((g["arms"][arm][s][f] for g in groups), ZERO) / k for f in groups[0]["arms"][arm][s]}
                  for s in SCOPES} for arm in ARMS}
    game_exact = {c: {s: {f: sum((g["game_exact_b"][c][s][f] for g in groups), ZERO) / k for f in ("YA", "YB")}
                      for s in SCOPES} for c in groups[0]["game_exact_b"]}
    return {"arms": arms, "game_exact_b": game_exact}


def _game_block(arms: dict, scope: str = "all", suffix: str = "") -> dict:
    return {f"{a}/{b}": {"YA": arms[f"game:{a}/{b}"][scope]["YA" + suffix],
                         "YB": arms[f"game:{a}/{b}"][scope]["YB" + suffix]} for a, b in GAME}


def _exact_block(game_exact: dict, scope: str = "all") -> dict:
    return {c: game_exact[c][scope] for c in game_exact}


def _rate(ans: int, n: int) -> Optional[Fraction]:
    return Fraction(ans, n) if n else None


def _answer_counts(rows) -> dict:
    n = len(rows)
    ans = sum(1 for r in rows if r[4])
    ties = sum(1 for r in rows if not r[4] and not r[5])
    return {"decisions": n, "answered": ans, "declined_tie": ties, "answer_rate": _rate(ans, n)}


def _d45(groups: List[dict]) -> dict:
    rows = [(g, r) for g in groups for r in g["t3_decisions"]]
    high = [r for _, r in rows if r[3]]
    value = [r for _, r in rows if not r[3]]
    by_level = {lv: _answer_counts([r for r in high if r[1] == lv]) for lv in LEVEL_KEYS}
    lo, hi = by_level["lo"], by_level["hi"]
    d4 = (lo["decisions"] > 0 and hi["decisions"] > 0 and hi["answer_rate"] > lo["answer_rate"]
          and Fraction(lo["decisions"] - lo["answered"], lo["decisions"]) > Fraction(1, 2))
    split = {}
    for g, r in rows:
        if not r[3]:
            continue
        rounds = g["high_round_t2"]
        key = next(iter(rounds)) if len(rounds) == 1 else "mixed"
        split.setdefault(key, []).append(r)
    by_round = {k: {lv: _answer_counts([r for r in v if r[1] == lv]) for lv in LEVEL_KEYS}
                for k, v in sorted(split.items())}
    on_path_round = {}
    for r in high:
        on_path_round.setdefault(str(r[6]), []).append(r)
    by_on_path = {k: {lv: _answer_counts([r for r in v if r[1] == lv]) for lv in LEVEL_KEYS}
                  for k, v in sorted(on_path_round.items())}
    per_instance = {}
    for g in groups:
        h = [r for r in g["t3_decisions"] if r[3]]
        per_instance[g["instance"]] = {lv: _answer_counts([r for r in h if r[1] == lv]) for lv in LEVEL_KEYS}
    vq, hq = _answer_counts(value), _answer_counts(high)
    d5 = vq["decisions"] > 0 and hq["decisions"] > 0 and vq["answer_rate"] > hq["answer_rate"]
    return {"D4": {"by_level": by_level, "holds": d4,
                   "lo_decline_share": (Fraction(lo["decisions"] - lo["answered"], lo["decisions"])
                                        if lo["decisions"] else None),
                   "by_t2_round": by_round, "by_on_path_round": by_on_path, "per_instance": per_instance},
            "D5": {"value": vq, "high": hq, "holds": d5}}


def gate(pooled: dict, groups: List[dict]) -> dict:
    p = str(PRIMARY_PI)
    arms = pooled["high"][p]["arms"]
    E = {t: arms[t]["all"]["E"] for t in ("T0", "T1", "T2", "T3")}

    def d2(scope):
        ya1, ya2 = arms["T1"][scope]["YA"], arms["T2"][scope]["YA"]
        rent = arms["T1"][scope]["YA_resp"] - arms["T2"][scope]["YA_resp"]
        identity = (ya1 - ya2) == (1 - PRIMARY_PI) * rent and arms["T1"][scope]["YA_prop"] == arms["T2"][scope]["YA_prop"]
        return {"YA_T1": float(ya1), "YA_T2": float(ya2), "lost_rent": float(rent), "identity_exact": identity,
                "holds": ya2 < ya1}

    d45 = _d45([g for g in groups if g["bank"] == "high" and g["pi"] == p])
    out = {"D1": {"E": {t: float(v) for t, v in E.items()}, "holds": E["T2"] >= E["T1"] >= E["T0"]},
           "D2": d2("all"), "D2_lo": d2("lo"), "D2_hi": d2("hi"),
           "D3": _pd(_game_block(arms)),
           "D3_exact_b": _pd(_exact_block(pooled["high"][p]["game_exact_b"])),
           "D3_common": _pd(_game_block(arms, suffix="_c")),
           "D4": _jsonable(d45["D4"]), "D5": _jsonable(d45["D5"])}
    carms = pooled["no_high"][p]["arms"]
    cE = {t: carms[t]["all"]["E"] for t in ("T0", "T1", "T2", "T3")}
    control = {"E": {t: float(v) for t, v in cE.items()},
               "YA_T1": float(carms["T1"]["all"]["YA"]), "YA_T2": float(carms["T2"]["all"]["YA"]),
               "YA_T1_lo": float(carms["T1"]["lo"]["YA"]), "YA_T2_lo": float(carms["T2"]["lo"]["YA"]),
               "D3": _pd(_game_block(carms)),
               "D3_exact_b": _pd(_exact_block(pooled["no_high"][p]["game_exact_b"]))}
    verdict = {"D1'": out["D1"]["holds"], "D2'": out["D2"]["holds"], "D2'-lo": out["D2_lo"]["holds"],
               "D3'": out["D3"]["holds"], "D4'": out["D4"]["holds"], "D5'": out["D5"]["holds"]}
    return {"readings": out, "control": control, "verdict": verdict, "primary": {"pi": p, "q": Q, "bank": "high"}}


def compute(instances: Optional[List[Instance]] = None, workers: int = 1) -> dict:
    instances = instances or canonical_instances()
    jobs = [(list(i.counts), bk, str(p)) for i, bk, p in itertools.product(instances, BANKS, PIS)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            groups = list(ex.map(solve_group, jobs))
    else:
        groups = [solve_group(j) for j in jobs]
    pooled = {bk: {str(p): _pool([g for g in groups if g["bank"] == bk and g["pi"] == str(p)]) for p in PIS}
              for bk in BANKS}
    return {"sweep": SWEEP_VERSION,
            "config": {"q": Q, "pis": [str(p) for p in PIS], "primary_pi": str(PRIMARY_PI), "arms": ARMS,
                       "banks": list(BANKS), "levels": {k: str(v) for k, v in LEVELS.items()},
                       "instances": [i.key for i in instances], "prior": "independent"},
            "gate": gate(pooled, groups),
            "pooled": _jsonable(pooled),
            "groups": [_jsonable({k: v for k, v in g.items() if k != "t3_decisions"}) for g in groups]}


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


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def _pd_line(d: dict) -> str:
    return (f"A {'is' if d['A_holds'] else 'is not'} in a dilemma, B {'is' if d['B_holds'] else 'is not'}; "
            f"withholding dominant for A {d['withholding_dominant_A']}, for B {d['withholding_dominant_B']}; "
            f"mutual disclosure beats mutual withholding for both {d['mutual_disclosure_beats_mutual_withholding']}")


def _pd_table(d: dict) -> List[str]:
    return ["| | temptation | reward | punishment | sucker |", "|---|---|---|---|---|",
            f"| A | {d['A']['temptation']:.4f} | {d['A']['reward']:.4f} | {d['A']['punishment']:.4f} | "
            f"{d['A']['sucker']:.4f} |",
            f"| B | {d['B']['temptation']:.4f} | {d['B']['reward']:.4f} | {d['B']['punishment']:.4f} | "
            f"{d['B']['sucker']:.4f} |"]


def _counts_cell(c: dict) -> str:
    return f"{c['answered']} / {c['decisions']} ({_pct(c['answer_rate'])})" if c["decisions"] else "—"


def render(doc: dict, path: str) -> str:
    g, cfg, P = doc["gate"], doc["config"], doc["pooled"]
    v, x = g["verdict"], g["readings"]
    V = {k.replace("'", "").replace("-", ""): b for k, b in v.items()}     # D1' -> D1, D2'-lo -> D2lo
    yes = lambda b: "**holds**" if b else "**fails**"   # noqa: E731
    half = str(Fraction(1, 2))
    arms = P["high"][half]["arms"]
    L = ["# Split Deal v0.3 — L2 two-sided program and gate SD3-D", "",
         f"_Rendered by `sweep_l2_v03.py render` from `{path}` (sweep `{doc['sweep']}`, commit `{doc['commit']}`, "
         f"{doc['seconds']} s). Exact on the model: no sampling, no API; every number is formatted from the JSON. "
         "Predictions as registered in `DESIGN_SPLIT_V03.md` §5; their operational reading was fixed in the "
         "docstring of `sweep_l2_v03.py` before the sweep ran._", "",
         f"Protocol 4.2 on the v0.3 game, Q = {cfg['q']} rounds (A then B), then the proposer is drawn (A with "
         "probability π) and makes one binding offer. Each side's type is its values and a private level (lo: d = 4, "
         "hi: d = 5.5). B is B0 (asks by expected information gain over the 13 questions, draws no inference from a "
         "decline, offers to maximise its own expected gain); A is A1, the exact best response to B0. Independent "
         "uniform prior. Y on each side's own scale; E is joint efficiency.", "",
         "## Gate SD3-D (π = 1/2)", "",
         "| prediction | reading | verdict |", "|---|---|---|",
         f"| D1′ | E(T2) ≥ E(T1) ≥ E(T0) | {yes(V['D1'])} |",
         f"| D2′ | Y_A(T2) < Y_A(T1): the information rent is positive | {yes(V['D2'])} |",
         f"| D2′-lo | the same for a lo-level A | {yes(V['D2lo'])} |",
         f"| D3′ | the 2 × 2 game is a strict prisoner's dilemma for both players (B0 reading) | {yes(V['D3'])} |",
         f"| D4′ | in T3 at B0's HIGH: hi-level A answers more often than lo-level A, and lo-level A declines most | "
         f"{yes(V['D4'])} |",
         f"| D5′ | in T3, A1 answers value questions at a higher rate than HIGH | {yes(V['D5'])} |", ""]
    E = x["D1"]["E"]
    L += [f"- **D1′** {yes(x['D1']['holds'])}: E(T0) {E['T0']:.4f}, E(T1) {E['T1']:.4f}, E(T2) {E['T2']:.4f} "
          f"(E(T3) {E['T3']:.4f})."]
    for key, label in (("D2", "D2′"), ("D2_lo", "D2′-lo"), ("D2_hi", "hi-level A (reported, not predicted)")):
        d = x[key]
        L.append(f"- **{label}** {yes(d['holds']) if key != 'D2_hi' else ''}: Y_A(T1) {d['YA_T1']:.4f}, "
                 f"Y_A(T2) {d['YA_T2']:.4f}; A's information rent as responder {d['lost_rent']:+.4f}"
                 f"{'' if d['identity_exact'] else ' (the (1 − π) identity does NOT hold exactly)'}.")
    for key, label in (("D3", "D3′ (B0 asks by EIG, the registered reading)"),
                       ("D3_exact_b", "D3′, second reading (B asks by the exact program)"),
                       ("D3_common", "D3′ on the common scale (reported)")):
        L.append(f"- **{label}** {yes(x[key]['holds'])}: {_pd_line(x[key])}.")
    d4, d5 = x["D4"], x["D5"]
    lo, hi = d4["by_level"]["lo"], d4["by_level"]["hi"]
    L += [f"- **D4′** {yes(d4['holds'])}: at B0's HIGH in T3, hi-level A answers {_counts_cell(hi)}, lo-level A "
          f"answers {_counts_cell(lo)}; lo-level A declines {_pct(d4['lo_decline_share'])} "
          f"({lo['declined_tie']} of its declines on a tie).",
          f"- **D5′** {yes(d5['holds'])}: A1 answers value questions {_counts_cell(d5['value'])} and HIGH "
          f"{_counts_cell(d5['high'])}.",
          f"- Choosing what to answer (T3) against always answering (T2): Y_A "
          f"{arms['T3']['all']['YA'] - arms['T2']['all']['YA']:+.4f}, Y_B "
          f"{arms['T3']['all']['YB'] - arms['T2']['all']['YB']:+.4f}, E "
          f"{arms['T3']['all']['E'] - arms['T2']['all']['E']:+.4f}.", "",
          "## Registered consequences", ""]
    cons = []
    if not v["D2'-lo"]:
        cons.append("**D2′-lo fails.** Registered: \"a private d does not make disclosure costly against B0, and the "
                    "next rung is B2 before any change to the game.\"")
    if not v["D3'"]:
        cons.append("**D3′ fails.** Registered: \"Making the reservation private is not enough against B0, which reads "
                    "nothing into a decline, and the next rung is B2 before any other change to the game.\"")
    if v["D2'"] and not v["D3'"]:
        d = x["D3"]
        which = []
        if not d["mutual_disclosure_beats_mutual_withholding"]:
            which.append("mutual disclosure does not beat mutual withholding for both")
        if not (d["withholding_dominant_A"] and d["withholding_dominant_B"]):
            which.append("withholding is not dominant for "
                         + " and ".join(p for p in ("A", "B") if not d[f"withholding_dominant_{p}"]))
        cons.append("**D2′ holds but D3′ fails.** Registered: \"Disclosure costs, but mutual disclosure does not beat "
                    "mutual withholding, or withholding is not dominant. The 2 × 2 cell values say which.\" Here: "
                    + "; ".join(which) + ".")
    if v["D4'"]:
        cons.append("**D4′ holds.** Registered: \"Against B0, a low-level A hides by declining at no cost. That finding "
                    "is specific to B0. A counterpart that reads declines (B2) would remove it.\"")
    L += (cons or ["None of the registered conditions is met."]) + [""]
    L += ["## D4′ by round", "",
          "Answered / decisions at B0's HIGH in T3, by A's level. First by the round in which B0 asks HIGH on its "
          "T2 path (the instance's column in `DESIGN_SPLIT_V03.md` §3), then by the round of the decision on the "
          "actual T3 path.", "",
          "| split | round | hi-level A | lo-level A |", "|---|---|---|---|"]
    for key, label in (("by_t2_round", "T2-path round"), ("by_on_path_round", "on-path round")):
        for rd, c in d4[key].items():
            L.append(f"| {label} | {rd} | {_counts_cell(c['hi'])} | {_counts_cell(c['lo'])} |")
    L += ["", "| instance | hi-level A | lo-level A |", "|---|---|---|"]
    for inst, c in d4["per_instance"].items():
        L.append(f"| {inst} | {_counts_cell(c['hi'])} | {_counts_cell(c['lo'])} |")
    L += ["", "## The 2 × 2 disclosure game (π = 1/2)", "",
          "Temptation = withhold against a discloser, reward = both disclose, punishment = both withhold, sucker = "
          "disclose to a withholder.", ""]
    for key, label in (("D3", "B0 asks by EIG"), ("D3_exact_b", "B asks by the exact program"),
                       ("D3_common", "B0 asks by EIG, common scale")):
        L += [f"**{label}**", ""] + _pd_table(x[key]) + [""]
    c = g["control"]
    L += ["## Control: the same game without HIGH (no prediction)", "",
          f"E(T0) {c['E']['T0']:.4f}, E(T1) {c['E']['T1']:.4f}, E(T2) {c['E']['T2']:.4f}, E(T3) {c['E']['T3']:.4f}. "
          f"Y_A(T1) {c['YA_T1']:.4f}, Y_A(T2) {c['YA_T2']:.4f}; lo-level A: Y_A(T1) {c['YA_T1_lo']:.4f}, "
          f"Y_A(T2) {c['YA_T2_lo']:.4f}.", "",
          f"- 2 × 2, B0 asks by EIG: {_pd_line(c['D3'])}.",
          f"- 2 × 2, B asks by the exact program: {_pd_line(c['D3_exact_b'])}.", ""]
    L += _pd_table(c["D3"]) + [""]
    L += ["## Structures by π and A's level", "",
          "Y_A and Y_B are each side's expected Y (own scale; `c` the common scale); `prop` and `resp` a side's Y when "
          "it proposes and when it responds; agreement shares by who proposed.", "",
          "| bank | π | structure | A level | Y_A | Y_B | Y_A c | Y_B c | Y_A prop | Y_A resp | Y_B prop | Y_B resp | "
          "agree, A proposes | agree, B proposes | E |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for bank in cfg["banks"]:
        for p in cfg["pis"]:
            for arm in ("T0", "T1", "T2", "T3"):
                for s in SCOPES:
                    a = P[bank][p]["arms"][arm][s]
                    L.append(f"| {bank} | {p} | {ARM_LABEL[arm]} | {s} | {a['YA']:.4f} | {a['YB']:.4f} | "
                             f"{a['YA_c']:.4f} | {a['YB_c']:.4f} | {a['YA_prop']:.4f} | {a['YA_resp']:.4f} | "
                             f"{a['YB_prop']:.4f} | {a['YB_resp']:.4f} | {100 * a['agree_a']:.1f} % | "
                             f"{100 * a['agree_b']:.1f} % | {a['E']:.4f} |")
    L += ["", "## D3′ by π (B0 reading)", "", "| bank | π | A in dilemma | B in dilemma |", "|---|---|---|---|"]
    for bank in cfg["banks"]:
        for p in cfg["pis"]:
            d = _pd(_game_block(P[bank][p]["arms"]))
            L.append(f"| {bank} | {p} | {d['A_holds']} | {d['B_holds']} |")
    L += ["", "Not built here: B2 (designed after these sweeps, its predictions registered before it runs; decision 7) "
          "and the CFR exploitability check."]
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
