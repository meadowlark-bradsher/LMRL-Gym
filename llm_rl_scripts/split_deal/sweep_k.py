"""Gate SD3-K (`DESIGN_SPLIT_B2.md` §6): the level-k readers on every instance. Offline, exact, no API.

The per-instance computation, the seconds-only pilot of decision 8 and its extension (§9), the gate
reading and the note.

One job is (instance, configuration, arm). Configurations (decision 6):

    primary   13-question bank, A1 as built, pi = 1/2: T0-T3 and the four 2 x 2 cells; pairings
              (A1, B0), (A1, B2), (A3, B2), and (A1, B2-d) on T2, T3 and the 2 x 2
    ties      A1' (ties to answering), otherwise primary:      T2, T3, the 2 x 2
    no_high   the 12-question bank, otherwise primary:         T2, T3, the 2 x 2
    pi_1/4, pi_3/4                                            T2, T3, the 2 x 2

Every job plays every (A type, B type) pair once per pairing; Q = 3. The worker reduces its outcomes to
exact sums per A level (``summarize``): each side's Y (own scale and the common scale (U - own d)/10),
split by who proposed, agreement shares, joint efficiency E, and A's answers at B's questions (HIGH and
value questions apart). It also checks that B2's and A3's computed values are what their rollouts realize.

Operational reading of gate SD3-K, fixed here before the gate ran. Pooled means weigh the nine instances
equally and, within an instance, A's type and B's type uniformly; "lo" restricts A's type to the lo level.
Primary configuration unless stated; Y on the own scale.

    K1  T3: Y_B(A1, B2) > Y_B(A1, B0).
    K2  Y_A^lo(T3; A1, B2) < Y_A^lo(T2; A1, B2).
    K3  T3, (A3, B2), over every episode's decisions at B2's HIGH (counts pooled over instances): a lo-level
        A3 declines more than half of its decisions, and a hi-level A3 answers a larger share than a lo-level
        one. A decline includes A3's ties (A3 breaks ties as the A1 it is built on). Not testable if either
        level has fewer than 100 such decisions.
    K4  [Y_A^lo(T3) - Y_A^lo(T2)] for (A3, B2) < the same for (A1, B0). The (A1, B0) terms are recomputed
        here and checked against the stored SD3-D values (results/split_v03_l2_20261110/sweep.json).
    K5  the 2 x 2 cells with (A1, B2) are not a strict prisoner's dilemma for both players (D3' ordering,
        sweep_l2._pd), on pooled Y_A and Y_B.

Each sensitivity (A1', no-HIGH bank, pi = 1/4, 3/4) is read the same way on its own runs, and the common
scale on the primary runs; none of them gates. The consequence for "K3's verdict differs under A1'"
compares the primary K3 with the ties configuration's K3. Reported beside: (A1, B2-d) for the
decomposition of K1, how often B asks HIGH, the misread share of (A3, B2), and D1'-D3' with B2.

Run plan (§9): every job in a fresh process, largest instances first, 8 workers.

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_k pilot
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_k probe     # the extension (DESIGN_SPLIT_B2.md §9)
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_k run --out results/split_v03_k_20261110/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_k render results/split_v03_k_20261110/sweep.json \
        --note llm_rl_scripts/split_deal/SWEEP_SPLIT_K.md
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from typing import Dict, List, Optional

from .game import TOTAL, Instance, canonical_instances
from .private_d import HIGH, LEVELS, PrivateTables
from .readers import A1Model, A3, B2, rollout_a1_b0, rollout_a1_b2, rollout_a1_b2d, rollout_a3_b2
from .sweep_l1 import _commit
from .sweep_l2 import Q, _jsonable, _pd
from .twosided import GAME, STRUCTURES, Shared

SWEEP_VERSION = "split-deal-v03-k-v0.1"
SD3D = "results/split_v03_l2_20261110/sweep.json"
SUPPORT = 100
WORKERS = 8

GAME_ARMS = [f"game:{a}/{b}" for a, b in GAME]
LEAN = ["T2", "T3"] + GAME_ARMS
CONFIGS = {
    "primary": {"bank": "high", "ties": False, "pi": "1/2", "arms": ["T0", "T1", "T2", "T3"] + GAME_ARMS},
    "ties": {"bank": "high", "ties": True, "pi": "1/2", "arms": LEAN},
    "no_high": {"bank": "no_high", "ties": False, "pi": "1/2", "arms": LEAN},
    "pi_1/4": {"bank": "high", "ties": False, "pi": "1/4", "arms": LEAN},
    "pi_3/4": {"bank": "high", "ties": False, "pi": "3/4", "arms": LEAN},
}


def _structure(arm: str):
    if arm.startswith("game:"):
        a, b = arm[5:].split("/")
        return GAME[(a, b)]
    return STRUCTURES[arm]


def _outcome(o: dict) -> list:
    return [o["a_deal"], o["b_deal"], [list(d) for d in o["decisions"]], o["misread"]]


def solve_arm(args, clock: dict = None) -> dict:
    """All pairings of one (instance, configuration, arm): per pairing, the outcome of every (A, B) pair.

    ``clock``, if given, receives the seconds spent on each pairing."""
    counts, config, arm = args
    clock = {} if clock is None else clock
    tick = [time.time()]

    def lap(name):
        now = time.time()
        clock[name] = now - tick[0]
        tick[0] = now

    cfg = CONFIGS[config]
    tab = PrivateTables(Instance(tuple(counts)), high_question=(cfg["bank"] == "high"))
    sh = Shared(tab)
    n = len(tab.types)
    st = _structure(arm)
    m = A1Model(sh, st, Q, Fraction(cfg["pi"]), answer_ties=cfg["ties"])
    b2s = {b: B2(sh, m, b) for b in range(n)}
    out = {"instance": list(counts), "config": config, "arm": arm, "pairings": {}}
    pairs = out["pairings"]
    pairs["A1/B0"] = [[_outcome(rollout_a1_b0(sh, m, ia, b)) for b in range(n)] for ia in range(n)]
    lap("A1/B0")
    pairs["A1/B2"] = [[_outcome(rollout_a1_b2(sh, m, ia, b2s[b])) for b in range(n)] for ia in range(n)]
    lap("A1/B2")
    if arm in LEAN:
        pairs["A1/B2d"] = [[_outcome(rollout_a1_b2d(sh, m, ia, b)) for b in range(n)] for ia in range(n)]
        lap("A1/B2d")
    a3_rows, a3_root = [], []
    for ia in range(n):
        a3 = A3(sh, m, b2s, ia)
        a3_root.append(a3.root())
        a3_rows.append([_outcome(rollout_a3_b2(sh, a3, b)) for b in range(n)])
    pairs["A3/B2"] = a3_rows
    lap("A3/B2")
    out["b2_root"] = [b2s[b].root() for b in range(n)]
    out["a3_root"] = a3_root
    return out


def jobs(instances: List[Instance]) -> list:
    return [(list(i.counts), c, arm) for i in instances for c, cfg in CONFIGS.items() for arm in cfg["arms"]]


# -- reducing one job to exact sums ------------------------------------------------------------

LVS = tuple(LEVELS)                                    # ("lo", "hi")
SCOPES = ("all",) + LVS
FIELDS = ("YA", "YB", "YA_c", "YB_c", "YA_prop", "YA_resp", "YB_prop", "YB_resp", "agree_a", "agree_b")
ZERO = Fraction(0)


def summarize(out: dict) -> dict:
    """Exact means per A level (and over both) of one job's outcomes, with A's answers counted."""
    cfg = CONFIGS[out["config"]]
    pi = Fraction(cfg["pi"])
    tab = PrivateTables(Instance(tuple(out["instance"])), high_question=(cfg["bank"] == "high"))
    sh = Shared(tab)
    n = len(tab.types)
    j_high = tab.questions.index(HIGH) if cfg["bank"] == "high" else None
    s_star = [[tab.s_star(ta, tb) for tb in tab.types] for ta in tab.types]
    common = [tab.scale(t) / TOTAL for t in tab.types]
    res = {"instance": out["instance"], "config": out["config"], "arm": out["arm"], "n_types": n, "pairings": {},
           "checks": {}}
    for name, rows in out["pairings"].items():
        acc = {lv: {f: ZERO for f in FIELDS} for lv in LVS}
        e = {lv: [ZERO, 0] for lv in LVS}
        cnt = {lv: {"high": [0, 0], "value": [0, 0], "asked_high": 0, "misread": 0} for lv in LVS}
        b_total = [ZERO] * n
        a_total = [ZERO] * n
        for ia, row in enumerate(rows):
            lv = tab.types[ia][1]
            ga = sh.seat[ia].gain
            for b, (ka, kb, decisions, misread) in enumerate(row):
                gb = sh.seat[b].gain
                ya_p = ga[ka] if ka is not None else ZERO
                yb_p = gb[sh.mirror[ka]] if ka is not None else ZERO
                ya_r = ga[kb] if kb is not None else ZERO
                yb_r = gb[sh.mirror[kb]] if kb is not None else ZERO
                ya, yb = pi * ya_p + (1 - pi) * ya_r, pi * yb_p + (1 - pi) * yb_r
                a = acc[lv]
                a["YA"] += ya
                a["YB"] += yb
                a["YA_c"] += ya * common[ia]
                a["YB_c"] += yb * common[b]
                a["YA_prop"] += ya_p
                a["YA_resp"] += ya_r
                a["YB_prop"] += yb_r
                a["YB_resp"] += yb_p
                a["agree_a"] += ka is not None
                a["agree_b"] += kb is not None
                a_total[ia] += ya
                b_total[b] += yb
                ss = s_star[ia][b]
                if ss:
                    e[lv][1] += 1
                    ta, tb = tab.types[ia], tab.types[b]
                    for k, w in ((ka, pi), (kb, 1 - pi)):
                        if k is not None:
                            sa = tab.inst.utility(ta[0], tab.allocs[k])
                            sb = tab.inst.utility(tb[0], tab.inst.rest(tab.allocs[k]))
                            e[lv][0] += w * (sa + sb - tab.d(ta) - tab.d(tb)) / ss
                c = cnt[lv]
                c["misread"] += bool(misread)
                c["asked_high"] += any(d[0] == j_high for d in decisions)
                for j, ans, _ in decisions:
                    slot = c["high" if j == j_high else "value"]
                    slot[0] += bool(ans)
                    slot[1] += 1
        n_lv = n // len(LVS)
        block = {}
        for lv in LVS:
            x = {f: acc[lv][f] / (n_lv * n) for f in FIELDS}
            x["E"] = e[lv][0] / e[lv][1] if e[lv][1] else None
            block[lv] = x
        x = {f: sum((acc[lv][f] for lv in LVS), ZERO) / (n * n) for f in FIELDS}
        x["E"] = sum(e[lv][0] for lv in LVS) / sum(e[lv][1] for lv in LVS)
        block["all"] = x
        block["counts"] = cnt
        res["pairings"][name] = block
        if name == "A1/B2":
            res["checks"]["b2_roots_realized"] = all(b_total[b] / n == out["b2_root"][b] for b in range(n))
        if name == "A3/B2":
            res["checks"]["a3_roots_realized"] = all(a_total[ia] / n == out["a3_root"][ia] for ia in range(n))
    return res


def run_job(args) -> dict:
    return summarize(solve_arm(args))


# -- pooling and the gate ------------------------------------------------------------------------

def _pool(groups: List[dict]) -> dict:
    """Equal-weight means over instances, per pairing and scope; decision counts are summed."""
    k = len(groups)
    out = {}
    for name in groups[0]["pairings"]:
        block = {}
        for sc in SCOPES:
            block[sc] = {f: sum((g["pairings"][name][sc][f] for g in groups), ZERO) / k
                         for f in groups[0]["pairings"][name][sc]}
        counts = {}
        for lv in LVS:
            c = {"high": [0, 0], "value": [0, 0], "asked_high": 0, "misread": 0}
            for g in groups:
                gc = g["pairings"][name]["counts"][lv]
                for key in ("high", "value"):
                    c[key] = [c[key][0] + gc[key][0], c[key][1] + gc[key][1]]
                c["asked_high"] += gc["asked_high"]
                c["misread"] += gc["misread"]
            counts[lv] = c
        block["counts"] = counts
        out[name] = block
    return out


def pooled_by(groups: List[dict]) -> dict:
    out: Dict[str, Dict[str, dict]] = {}
    for config, cfg in CONFIGS.items():
        out[config] = {arm: _pool([g for g in groups if g["config"] == config and g["arm"] == arm])
                       for arm in cfg["arms"] if any(g["config"] == config and g["arm"] == arm for g in groups)}
    return out


def _k3(counts: dict) -> dict:
    lo, hi = counts["lo"]["high"], counts["hi"]["high"]
    testable = lo[1] >= SUPPORT and hi[1] >= SUPPORT
    lo_decline = Fraction(lo[1] - lo[0], lo[1]) if lo[1] else None
    hi_rate = Fraction(hi[0], hi[1]) if hi[1] else None
    lo_rate = Fraction(lo[0], lo[1]) if lo[1] else None
    holds = testable and lo_decline > Fraction(1, 2) and hi_rate > lo_rate
    return {"testable": testable, "holds": holds, "lo_decisions": lo[1], "hi_decisions": hi[1],
            "lo_decline_share": lo_decline, "hi_answer_rate": hi_rate, "lo_answer_rate": lo_rate,
            "verdict": ("not testable" if not testable else "holds" if holds else "fails")}


def readings(P: dict, suffix: str = "") -> dict:
    """K1-K5 for one configuration's pooled blocks ``P[arm][pairing][scope][field]``; ``suffix`` "_c" reads the
    common scale (K3 has no scale)."""
    YA, YB = "YA" + suffix, "YB" + suffix
    out = {}
    t2, t3 = P.get("T2"), P.get("T3")
    if t3:
        out["K1"] = {"B2": t3["A1/B2"]["all"][YB], "B0": t3["A1/B0"]["all"][YB],
                     "B2d": t3["A1/B2d"]["all"][YB] if "A1/B2d" in t3 else None}
        out["K1"]["holds"] = out["K1"]["B2"] > out["K1"]["B0"]
        out["K3"] = _k3(t3["A3/B2"]["counts"])
    if t2 and t3:
        k2 = {"T3": t3["A1/B2"]["lo"][YA], "T2": t2["A1/B2"]["lo"][YA]}
        k2["holds"] = k2["T3"] < k2["T2"]
        out["K2"] = k2
        gain_a3 = t3["A3/B2"]["lo"][YA] - t2["A3/B2"]["lo"][YA]
        gain_b0 = t3["A1/B0"]["lo"][YA] - t2["A1/B0"]["lo"][YA]
        out["K4"] = {"A3/B2": gain_a3, "A1/B0": gain_b0, "holds": gain_a3 < gain_b0}
    if all(arm in P for arm in GAME_ARMS):
        for name in ("A1/B2", "A3/B2", "A1/B0"):
            m = {arm[5:]: {"YA": P[arm][name]["all"][YA], "YB": P[arm][name]["all"][YB]} for arm in GAME_ARMS}
            out["D3:" + name] = _pd(m)
        out["K5"] = {"pd": out["D3:A1/B2"], "holds": not out["D3:A1/B2"]["holds"]}
    return out


def gate(pooled: dict, sd3d: Optional[dict]) -> dict:
    prim = pooled["primary"]
    k = readings(prim)
    sens = {c: readings(pooled[c]) for c in CONFIGS if c != "primary" and c in pooled}
    common = readings(prim, "_c")
    verdict = {"K1": k["K1"]["holds"], "K2": k["K2"]["holds"], "K3": k["K3"]["verdict"], "K4": k["K4"]["holds"],
               "K5": k["K5"]["holds"]}
    k3_ties = sens.get("ties", {}).get("K3", {}).get("verdict")
    d = {}
    for name in ("A1/B2", "A3/B2", "A1/B0"):
        E = {t: prim[t][name]["all"]["E"] for t in ("T0", "T1", "T2", "T3") if t in prim}
        d[name] = {"E": E, "D1": E["T2"] >= E["T1"] >= E["T0"],
                   "YA_T1": prim["T1"][name]["all"]["YA"], "YA_T2": prim["T2"][name]["all"]["YA"],
                   "YA_T1_lo": prim["T1"][name]["lo"]["YA"], "YA_T2_lo": prim["T2"][name]["lo"]["YA"]}
    return {"readings": k, "verdict": verdict, "k3_ties": k3_ties,
            "k3_differs_under_ties": k3_ties is not None and k3_ties != verdict["K3"],
            "sensitivities": sens, "common_scale": common, "d_with_b2": d,
            "sd3d_match": _sd3d_match(prim, sd3d)}


def _sd3d_match(prim: dict, sd3d: Optional[dict]) -> Optional[bool]:
    """The recomputed (A1, B0) Y_A and Y_B in T0-T3 against the stored SD3-D pooled values, per scope."""
    if sd3d is None:
        return None
    stored = sd3d["pooled"]["high"]["1/2"]["arms"]
    for arm in ("T0", "T1", "T2", "T3"):
        if arm not in prim:
            continue
        for sc in SCOPES:
            for f in ("YA", "YB"):
                if abs(float(prim[arm]["A1/B0"][sc][f]) - stored[arm][sc][f]) > 1e-12:
                    return False
    return True


def compute(instances: Optional[List[Instance]] = None, workers: int = 1, sd3d_path: Optional[str] = SD3D) -> dict:
    instances = instances or canonical_instances()
    order = sorted(instances, key=lambda i: -len(PrivateTables(i).types))      # largest instances first
    todo = jobs(order)
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            groups = list(ex.map(run_job, todo))
    else:
        groups = [run_job(j) for j in todo]
    groups.sort(key=lambda g: ([i.key for i in instances].index("-".join(map(str, g["instance"]))),
                               list(CONFIGS).index(g["config"]), CONFIGS[g["config"]]["arms"].index(g["arm"])))
    pooled = pooled_by(groups)
    sd3d = None
    if sd3d_path and len(instances) == len(canonical_instances()) and os.path.exists(sd3d_path):
        with open(sd3d_path) as f:
            sd3d = json.load(f)
    return {"sweep": SWEEP_VERSION,
            "config": {"q": Q, "configs": CONFIGS, "instances": [i.key for i in instances], "support": SUPPORT,
                       "prior": "independent"},
            "gate": _jsonable(gate(pooled, sd3d)),
            "checks": {"b2_roots_realized": all(g["checks"]["b2_roots_realized"] for g in groups),
                       "a3_roots_realized": all(g["checks"]["a3_roots_realized"] for g in groups)},
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
    v = doc["gate"]["verdict"]
    print(f"wrote {args.out} ({doc['seconds']} s); " + ", ".join(
        f"{k} {x if isinstance(x, str) else ('holds' if x else 'fails')}" for k, x in v.items()))
    return 0


# -- note ------------------------------------------------------------------------------------------

def _f(x, nd=4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def _yes(b) -> str:
    if isinstance(b, str):
        return f"**{b}**"
    return "**holds**" if b else "**fails**"


def _pd_line(d: dict) -> str:
    return (f"A {'is' if d['A_holds'] else 'is not'} in a dilemma, B {'is' if d['B_holds'] else 'is not'}; "
            f"withholding dominant for A {d['withholding_dominant_A']}, for B {d['withholding_dominant_B']}; "
            f"mutual disclosure beats mutual withholding for both {d['mutual_disclosure_beats_mutual_withholding']}")


def _k_lines(k: dict) -> List[str]:
    L = []
    if "K1" in k:
        x = k["K1"]
        L.append(f"- **K1** {_yes(x['holds'])}: T3, Y_B(A1, B2) {x['B2']:.4f} against Y_B(A1, B0) {x['B0']:.4f}"
                 + (f"; with B2-d {x['B2d']:.4f}." if x.get("B2d") is not None else "."))
    if "K2" in k:
        x = k["K2"]
        L.append(f"- **K2** {_yes(x['holds'])}: a lo-level A1 against B2, Y_A^lo(T3) {x['T3']:.4f} against "
                 f"Y_A^lo(T2) {x['T2']:.4f}.")
    if "K3" in k:
        x = k["K3"]
        L.append(f"- **K3** {_yes(x['verdict'])}: at B2's HIGH in T3, a lo-level A3 declines "
                 f"{_pct(x['lo_decline_share'])} of {x['lo_decisions']} decisions; answer rates hi "
                 f"{_pct(x['hi_answer_rate'])} ({x['hi_decisions']} decisions), lo {_pct(x['lo_answer_rate'])}.")
    if "K4" in k:
        x = k["K4"]
        L.append(f"- **K4** {_yes(x['holds'])}: a lo-level A's gain from choosing over answering everything, "
                 f"Y_A^lo(T3) − Y_A^lo(T2): {x['A3/B2']:+.4f} for (A3, B2) against {x['A1/B0']:+.4f} for (A1, B0).")
    if "K5" in k:
        L.append(f"- **K5** {_yes(k['K5']['holds'])}: the 2 × 2 with (A1, B2): {_pd_line(k['K5']['pd'])}.")
    return L


CONFIG_LABEL = {"ties": "A1′ (ties to answering)", "no_high": "the 12-question bank (no HIGH)",
                "pi_1/4": "π = 1/4", "pi_3/4": "π = 3/4"}
PAIRING_LABEL = {"A1/B0": "(A1, B0)", "A1/B2": "(A1, B2)", "A1/B2d": "(A1, B2-d)", "A3/B2": "(A3, B2)"}


def render(doc: dict, path: str) -> str:
    g, cfg = doc["gate"], doc["config"]
    P = doc["pooled"]
    k, v = g["readings"], g["verdict"]
    L = ["# Split Deal v0.3 — B2, A3 and gate SD3-K", "",
         f"_Rendered by `sweep_k.py render` from `{path}` (sweep `{doc['sweep']}`, commit `{doc['commit']}`, "
         f"{doc['seconds']} s). Exact on the model: no sampling, no API; every number is formatted from the JSON. "
         "Predictions as registered in `DESIGN_SPLIT_B2.md` §6; their operational reading is fixed in the docstring "
         "of `sweep_k.py`._", "",
         "B2 is the exact best response to A1 (which best-responds to B0); A3 is the exact best response to B2. "
         "Protocol 4.2 on the v0.3 game, Q = 3; primary configuration: 13-question bank, A1 as built (ties to "
         "declining), π = 1/2. Independent uniform prior; Y on each side's own scale.", "",
         "## Gate SD3-K", "",
         "| prediction | reading | verdict |", "|---|---|---|",
         f"| K1 | B2 reads A1 profitably (T3) | {_yes(v['K1'])} |",
         f"| K2 | against B2, a lo-level A1 does worse in T3 than in T2 | {_yes(v['K2'])} |",
         f"| K3 | a lo-level A3 still declines most of B2's HIGH, and hi answers more than lo | {_yes(v['K3'])} |",
         f"| K4 | hiding is worth less to a lo-level A against B2 than against B0 | {_yes(v['K4'])} |",
         f"| K5 | the 2 × 2 against B2 is still not a strict dilemma | {_yes(v['K5'])} |", ""]
    L += _k_lines(k) + [""]
    chk = doc["checks"]
    L += [f"Checks: B2's computed values are what its rollouts realize: {chk['b2_roots_realized']}; A3's: "
          f"{chk['a3_roots_realized']}. The recomputed (A1, B0) values match the stored SD3-D ones (`{SD3D}`): "
          f"{g['sd3d_match']}.", "", "## Registered consequences", ""]
    cons = []
    if v["K2"] and v["K3"] == "holds":
        cons.append("**K2 and K3 hold.** Registered: “Hiding is exploitable by a reader, yet it persists against B2, "
                    "because B2 reads A through a model A3 has outgrown. Only an equilibrium can say whether it lasts, "
                    "so the CFR check on the smallest instance comes next, before any live design.”")
    if v["K3"] == "fails":
        cons.append("**K3 fails.** Registered: “Disclosure already unravels at level 3. Split Deal's selective hiding "
                    "was an artifact of a counterpart that does not read declines, and the two-sided live design "
                    "carries no hiding treatment.”")
    if v["K3"] == "not testable":
        cons.append("**K3 is not testable** under the support rule: B2 asked HIGH too seldom for either level to "
                    f"reach {cfg['support']} decisions.")
    if not v["K5"]:
        cons.append("**K5 fails.** Registered: “A strict dilemma appears only against a counterpart that reads A. Any "
                    "two-sided live study must then give its counterpart a reading rule of B2's kind, and NEG-33 "
                    "inherits that constraint.”")
    if g["k3_differs_under_ties"]:
        cons.append(f"**K3's verdict differs under A1′** ({g['k3_ties']} there). Registered: “Whether hiding survives "
                    "a reader then depends on how the read player breaks ties it is indifferent about, and the "
                    "write-up says so wherever K3 is cited.”")
    if v["K5"]:
        cons.append("**K5 holds.** Registered: “The all-or-nothing dilemma is absent in Split Deal against every "
                    "counterpart tried (v0.2 with B0, v0.3 with B0, v0.3 with B2). The two-way motivation then rests "
                    "on selective disclosure alone, and the write-up says so.”")
    for c in cons or ["None of the registered conditions is met."]:
        L += [c, ""]
    L += ["## Sensitivities and the common scale (reported, not gating)", ""]
    for c, kk in g["sensitivities"].items():
        L += [f"**{CONFIG_LABEL[c]}**", ""] + _k_lines(kk) + [""]
    L += ["**Common scale, (U − own d)/10, primary runs**", ""] + [x for x in _k_lines(g["common_scale"])
                                                                 if not x.startswith("- **K3")] + [""]
    L += ["## With B2: D1′–D3′ readings and the 2 × 2 (primary)", "",
          "| pairing | E(T0) | E(T1) | E(T2) | E(T3) | D1′ | Y_A(T1) | Y_A(T2) | Y_A^lo(T1) | Y_A^lo(T2) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in g["d_with_b2"].items():
        E = x["E"]
        L.append(f"| {PAIRING_LABEL[name]} | {E['T0']:.4f} | {E['T1']:.4f} | {E['T2']:.4f} | {E['T3']:.4f} | "
                 f"{x['D1']} | {x['YA_T1']:.4f} | {x['YA_T2']:.4f} | {x['YA_T1_lo']:.4f} | {x['YA_T2_lo']:.4f} |")
    L.append("")
    for name in ("A1/B2", "A3/B2", "A1/B0"):
        d = k["D3:" + name]
        L += [f"**2 × 2, {PAIRING_LABEL[name]}**: {_pd_line(d)}.", "",
              "| | temptation | reward | punishment | sucker |", "|---|---|---|---|---|",
              f"| A | {d['A']['temptation']:.4f} | {d['A']['reward']:.4f} | {d['A']['punishment']:.4f} | "
              f"{d['A']['sucker']:.4f} |",
              f"| B | {d['B']['temptation']:.4f} | {d['B']['reward']:.4f} | {d['B']['punishment']:.4f} | "
              f"{d['B']['sucker']:.4f} |", ""]
    L += ["## By structure and pairing (primary)", "",
          "Answers: answered / decisions at B's questions, HIGH and value questions apart. `asks HIGH`: episodes in "
          "which B asks HIGH. `misread`: episodes in which A's true type leaves B's belief.", "",
          "| structure | pairing | A level | Y_A | Y_B | Y_A resp | Y_B resp | agree, A proposes | agree, B proposes | "
          "E | HIGH answered | value answered | asks HIGH | misread |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in CONFIGS["primary"]["arms"]:
        for name, blk in P["primary"][arm].items():
            for sc in SCOPES:
                x = blk[sc]
                if sc == "all":
                    hi_c = val_c = ask_c = mis_c = "—"
                else:
                    c = blk["counts"][sc]
                    hi_c = f"{c['high'][0]} / {c['high'][1]}"
                    val_c = f"{c['value'][0]} / {c['value'][1]}"
                    ask_c, mis_c = str(c["asked_high"]), str(c["misread"])
                L.append(f"| {arm} | {PAIRING_LABEL[name]} | {sc} | {x['YA']:.4f} | {x['YB']:.4f} | "
                         f"{x['YA_resp']:.4f} | {x['YB_resp']:.4f} | {100 * x['agree_a']:.1f} % | "
                         f"{100 * x['agree_b']:.1f} % | {_f(x['E'])} | {hi_c} | {val_c} | {ask_c} | {mis_c} |")
    L += ["", "Not run here: the CFR check on the smallest instance (§8 of `DESIGN_SPLIT_B2.md`)."]
    return "\n".join(L) + "\n"


def render_cmd(args) -> int:
    with open(args.json) as f:
        doc = json.load(f)
    with open(args.note, "w") as f:
        f.write(render(doc, args.json))
    print(f"wrote {args.note}")
    return 0


def pilot(args) -> int:
    """Decision 8: time every job on 2-2-3 and print seconds only; the outcomes are discarded unread."""
    inst = Instance((2, 2, 3))
    total = 0.0
    for job in jobs([inst]):
        t0 = time.time()
        solve_arm(job)
        dt = time.time() - t0
        total += dt
        print(f"{job[1]:8s} {job[2]:14s} {dt:7.1f} s", flush=True)
    print(f"total {total:.1f} s for {len(jobs([inst]))} jobs on 2-2-3 ({len(PrivateTables(inst).types)} types)")
    return 0


def _peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30      # bytes on macOS


PROBE_LIMIT_GIB = 12.0


def probe(args) -> int:
    """The pilot's extension: the primary T3 job on 1-1-3, seconds and peak memory only; outcomes discarded."""
    t0 = time.time()
    clock: dict = {}

    def watchdog():
        last = time.time()
        while True:
            time.sleep(5)
            peak = _peak_gib()
            if peak > PROBE_LIMIT_GIB:
                print(f"STOPPED at {time.time() - t0:.0f} s: peak memory {peak:.1f} GiB > {PROBE_LIMIT_GIB} GiB; "
                      f"pairings finished: {', '.join(f'{k} {v:.0f} s' for k, v in clock.items()) or 'none'}",
                      flush=True)
                os._exit(3)
            if time.time() - last >= 300:
                last = time.time()
                print(f"... {time.time() - t0:.0f} s, peak {peak:.2f} GiB", flush=True)

    threading.Thread(target=watchdog, daemon=True).start()
    solve_arm(([1, 1, 3], "primary", "T3"), clock)
    for k, v in clock.items():
        print(f"{k:8s} {v:9.1f} s", flush=True)
    print(f"total {time.time() - t0:.1f} s, peak memory {_peak_gib():.2f} GiB "
          f"(primary T3 on 1-1-3, {len(PrivateTables(Instance((1, 1, 3))).types)} types)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot").set_defaults(func=pilot)
    sub.add_parser("probe").set_defaults(func=probe)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=WORKERS)
    r.set_defaults(func=run)
    nt = sub.add_parser("render")
    nt.add_argument("json")
    nt.add_argument("--note", required=True)
    nt.set_defaults(func=render_cmd)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
