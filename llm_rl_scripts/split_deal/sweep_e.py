"""Gate SD3-E (`DESIGN_SPLIT_CFR.md` §5): equilibria of game G by CFR+, certified by exact best responses, set
against the ladder on G. Offline, no API; floating point (``cfr.py``).

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_e run --out results/split_v03_e_20261111/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.sweep_e render results/split_v03_e_20261111/sweep.json \\
        --note llm_rl_scripts/split_deal/SWEEP_SPLIT_E.md

Run plan (§10, fixed 2026-11-11). Five distinct structures: T2 (the same structure as the 2 x 2 cell all/all), T3,
and the cells all/none, none/all and none/none. Each is solved from two starts: uniform, and warm at the ladder
profile (A3, B2) of the same structure (decision 5). A solve runs CFR+ with NashConv checked every ``EVERY``
iterations and stops at the first checkpoint with NashConv <= ``EPS_BAR``. Within ``BUDGET`` iterations it is
certified in the budget. Otherwise the same run continues to ``MAX_ITERATIONS`` (decision 4's four times the
iterations), then DCFR runs from the same start up to ``MAX_ITERATIONS``, and failing that the solve is recorded as
not certified, with the last DCFR average reported but never read by the gate. One more job per structure computes
the ladder on G: A1, A3, B0 and B2 as profiles, their exact values, the checks, their exploitability and the bridges.

Operational reading of §5, fixed here before any SD3-E value is computed:

* Means are over the 324 type pairs, uniform; "lo"/"hi" is A's level; Y is own-scale. A profile's values and
  behaviour are read from its outcome distribution per type pair (``outcomes``), and its values agree with
  ``Game.value`` (checked).
* Every certified equilibrium found counts. A prediction is read on each certified equilibrium of the structures it
  uses: for E4 every (T3, T2) pair, for E5 every combination of the four cells. One verdict across them is the
  verdict. Several verdicts, "not testable" included, give "equilibrium-dependent". No certified equilibrium in a
  structure the prediction needs gives "not certified".
* E1: NashConv(A3, B2) in T3 > 0.01 Y. It uses no equilibrium, so it is always read.
* E2: Y_B(sigma*_T3) > Y_B(A3, B2) in T3.
* E3, on sigma*_T3, at A's answering decisions at B's HIGH. A decision's mass is the probability of reaching it,
  over type pairs from the prior. Pooled over decisions:
  - hi answers with probability (hi answering mass / hi asked mass) >= 0.95;
  - P(lo | decline) (lo declining mass / all declining mass) >= 0.9. If no type declines HIGH on path, the
    clause is read as met: the level is fully disclosed on path.

  Support: the probability that B asks HIGH, given A's level, must be at least 0.05 for both levels (HIGH can be
  asked at most once on a path), else not testable. The lo decline rate is reported by round, not read.
* E4: Y_A^lo(sigma*_T3) - Y_A^lo(sigma*_T2) < Y_A^lo(A3, B2; T3) - Y_A^lo(A3, B2; T2), the ladder terms on G.
* E5: the strict prisoner's-dilemma ordering of D3' and K5 (``sweep_l2._pd``) on the cells' equilibrium values,
  where all/all is T2's equilibrium. E5 holds iff the ordering does not hold for both players.
* Checks, recorded beside the gate:
  - the ladder profiles' values on the tree equal the readers' exact rollouts;
  - the best responses to B0, A1 and B2 have the values of A1, B2 and A3 (§4), within ``cfr.TOL``;
  - a best response is never worse than the profile it deviates from.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from typing import Dict, List, Optional

from . import cfr  # noqa: F401  (sets the BLAS thread limit before numpy is used here)
import numpy as np                                                                  # noqa: E402

from .game import Instance                                                          # noqa: E402
from .private_d import HIGH                                                         # noqa: E402
from .readers import A1Model, A3, B2, rollout_a1_b0, rollout_a1_b2, rollout_a3_b2   # noqa: E402
from .sweep_l1 import _commit                                                       # noqa: E402
from .sweep_l2 import _pd                                                           # noqa: E402
from .twosided import GAME, STRUCTURES, Shared                                      # noqa: E402

SWEEP_VERSION = "split-deal-v03-e-v0.1"
SD3K = "results/split_v03_k_20261110/sweep.json"
EVERY, BUDGET, MAX_ITERATIONS, EPS_BAR = 500, 8000, 32000, cfr.EPS_BAR
WORKERS = 6
E1_BAR, E3_HI, E3_POST, E3_SUPPORT = 0.01, 0.95, 0.9, 0.05
STRUCTS = {"T2": STRUCTURES["T2"], "T3": STRUCTURES["T3"], "game:all/none": GAME[("all", "none")],
           "game:none/all": GAME[("none", "all")], "game:none/none": GAME[("none", "none")]}
CELL = {"all/all": "T2", "all/none": "game:all/none", "none/all": "game:none/all", "none/none": "game:none/none"}
STARTS = ("uniform", "warm")
LADDER = ("A1/B0", "A1/B2", "A3/B2")
SCOPES = ("all", "lo", "hi")
G_CONFIG = {"instance": list(cfr.G_INSTANCE), "q": cfr.G_Q, "bank": "merged", "pi": str(cfr.G_PI), "every": EVERY,
            "budget": BUDGET, "max_iterations": MAX_ITERATIONS, "eps_bar": EPS_BAR}


def _game(config: dict, structure: str):
    tab = cfr.g_tables(Instance(tuple(config["instance"])), config["bank"])
    return cfr.split_game(tab, STRUCTS[structure], config["q"], Fraction(config["pi"]))


# -- what a profile does ---------------------------------------------------------------------------

def outcomes(g: cfr.Game, prof) -> dict:
    """Per type pair: Y for both sides (own and common scale), agreement and E, from the profile's distribution
    over final-stage outcomes; then means over all pairs and by A's level."""
    t, ot, tab = g.tree, g.ot, g.tab
    n, m, pi = t.n[0], g.m, g.pi
    L = t.offer_nodes
    RA, RB = g.reach(0, prof[0][0])[L], g.reach(1, prof[1][0])[L]
    PA, PB = np.zeros((n, m, n)), np.zeros((n, m, n))          # [ta, offer, tb]: A proposes / B proposes
    for ty, idx in g.by_type[0]:
        W = RA[t.off_leaf[0][idx], ty][:, None] * prof[0][1][idx]
        np.add.at(PA[ty], ot["KEEP"][0][ty], W.T @ RB[t.off_leaf[0][idx]])
    for ty, idx in g.by_type[1]:
        W = RB[t.off_leaf[1][idx], ty][:, None] * prof[1][1][idx]
        X = np.zeros((m, n))
        np.add.at(X, ot["KEEP"][1][ty], W.T @ RA[t.off_leaf[1][idx]])
        PB[:, :, ty] = X.T
    accA = PA * ot["ACC_B"][None, :, :]                             # A's offer accepted by B
    accB = PB * ot["ACC_A"].T[:, :, None]                           # B's offer accepted by A
    GA, GB = ot["GA"], ot["GB"]
    ya_p = np.einsum("akb,ak->ab", accA, GA)
    yb_r = np.einsum("akb,bk->ab", accA, GB)
    ya_r = np.einsum("akb,ak->ab", accB, GA)
    yb_p = np.einsum("akb,bk->ab", accB, GB)
    YA, YB = pi * ya_p + (1 - pi) * ya_r, pi * yb_p + (1 - pi) * yb_r
    scale = np.array([float(tab.scale(ty)) for ty in tab.types])
    S = (GA * scale[:, None])[:, :, None] + (GB * scale[:, None]).T[None, :, :]      # surplus in utility units
    s_star = np.array([[float(tab.s_star(ta, tb) or 0) for tb in tab.types] for ta in tab.types])
    surplus = pi * np.einsum("akb,akb->ab", accA, S) + (1 - pi) * np.einsum("akb,akb->ab", accB, S)
    pos = s_star > 0
    E = np.where(pos, surplus / np.where(pos, s_star, 1.0), 0.0)
    pair = {"YA": YA, "YB": YB, "YA_c": YA * scale[:, None] / 10, "YB_c": YB * scale[None, :] / 10,
            "agree_a": accA.sum(axis=1), "agree_b": accB.sum(axis=1)}
    lv = np.array([ty[1] for ty in tab.types])
    out = {}
    for sc in SCOPES:
        rows = np.ones(n, bool) if sc == "all" else lv == sc
        o = {k: float(v[rows].mean()) for k, v in pair.items()}
        o["E"] = float(E[rows][pos[rows]].mean()) if pos[rows].any() else None
        out[sc] = o
    return out


def high_stats(g: cfr.Game, prof) -> Optional[dict]:
    """A's answering at B's HIGH (structures where A chooses): masses by A's level and round, from the prior."""
    t, tab = g.tree, g.tab
    if g.structure.a_answers != "choose" or HIGH not in tab.questions:
        return None
    jh = tab.questions.index(HIGH)
    n = t.n[0]
    RA, RB = g.reach(0, prof[0][0]), g.reach(1, prof[1][0])
    lv = np.array([ty[1] for ty in tab.types])
    rounds = {}
    for z in t.dec_nodes[0]:
        s = t.state[z]
        if s[0] != "ANS" or s[6] != jh:
            continue
        rd = g.q - s[1] + 1
        mass = RA[z] * g.p[0] * float(g.p[1] @ RB[z])                     # per A type
        e_ans = t.edge_of[0][next(c for c in range(t.first[z], t.first[z] + t.nch[z]) if t.label[c] == "answer")]
        ans = mass * prof[0][0][e_ans]
        r = rounds.setdefault(rd, {lvl: {"asked": 0.0, "answered": 0.0} for lvl in ("lo", "hi")})
        for lvl in ("lo", "hi"):
            r[lvl]["asked"] += float(mass[lv == lvl].sum())
            r[lvl]["answered"] += float(ans[lv == lvl].sum())
    tot = {lvl: {k: sum(r[lvl][k] for r in rounds.values()) for k in ("asked", "answered")} for lvl in ("lo", "hi")}
    prior = {lvl: float(g.p[0][lv == lvl].sum()) for lvl in ("lo", "hi")}
    dec = {lvl: tot[lvl]["asked"] - tot[lvl]["answered"] for lvl in ("lo", "hi")}
    return {"rounds": {str(k): v for k, v in sorted(rounds.items())}, "total": tot,
            "p_asked": {lvl: tot[lvl]["asked"] / prior[lvl] for lvl in ("lo", "hi")},
            "answer_rate": {lvl: (tot[lvl]["answered"] / tot[lvl]["asked"]) if tot[lvl]["asked"] else None
                            for lvl in ("lo", "hi")},
            "decline_mass": dec["lo"] + dec["hi"],
            "p_lo_given_decline": (dec["lo"] / (dec["lo"] + dec["hi"])) if dec["lo"] + dec["hi"] > 1e-15 else None}


def describe(g: cfr.Game, prof) -> dict:
    """Values, behaviour and exploitability of one profile."""
    nc = g.nashconv(prof)
    br_a, br_b = nc["br_policy"]
    return {"outcomes": outcomes(g, prof), "high": high_stats(g, prof),
            "value": list(nc["value"]), "br_value": list(nc["br_value"]), "eps": list(nc["eps"]),
            "nashconv": nc["nashconv"],
            "b_at_a_br": g.value((br_a, prof[1]))[1],               # B's value when A best-responds
            "a_at_b_br": g.value((prof[0], br_b))[0]}               # A's value when B best-responds


# -- jobs ------------------------------------------------------------------------------------------

def _exact(sh, m, b2s, a3s, n, pi, pairing):
    ya = yb = Fraction(0)
    for ia in range(n):
        for b in range(n):
            o = (rollout_a1_b0(sh, m, ia, b) if pairing == "A1/B0" else
                 rollout_a1_b2(sh, m, ia, b2s[b]) if pairing == "A1/B2" else rollout_a3_b2(sh, a3s[ia], b))
            ga, gb = sh.seat[ia].gain, sh.seat[b].gain
            ka, kb = o["a_deal"], o["b_deal"]
            ya += pi * (ga[ka] if ka is not None else 0) + (1 - pi) * (ga[kb] if kb is not None else 0)
            yb += pi * (gb[sh.mirror[ka]] if ka is not None else 0) + (1 - pi) * (gb[sh.mirror[kb]] if kb is not None else 0)
    return float(ya / (n * n)), float(yb / (n * n))


def ladder_job(config: dict, structure: str) -> dict:
    """The ladder on G for one structure: profiles, exact values, the checks, exploitability, bridges."""
    t0, c0 = time.time(), time.process_time()
    g = _game(config, structure)
    lad = cfr.Ladder(g)
    P = {nm: lad.profile(nm) for nm in ("A1", "A3", "B0", "B2")}
    exact = {pr: [float(x) for x in lad.exact(pr)] for pr in LADDER}
    prof = {pr: (P[pr[:2]], P[pr[3:]]) for pr in LADDER}
    desc = {pr: describe(g, prof[pr]) for pr in LADDER}
    tree_vs_exact = max(abs(a - b) for pr in LADDER for a, b in zip(desc[pr]["value"], exact[pr]))
    repro = {"BR(B0) = A1": abs(desc["A1/B0"]["br_value"][0] - exact["A1/B0"][0]),
             "BR(A1) = B2": abs(desc["A1/B2"]["br_value"][1] - exact["A1/B2"][1]),
             "BR(B2) = A3": abs(desc["A3/B2"]["br_value"][0] - exact["A3/B2"][0])}
    # bridge: the ladder with the full bank on the same instance and Q (the duplicate channel of section 3)
    full = cfr.g_tables(Instance(tuple(config["instance"])), "full")
    sh = Shared(full)
    n = len(full.types)
    m = A1Model(sh, STRUCTS[structure], config["q"], Fraction(config["pi"]))
    b2s = {b: B2(sh, m, b) for b in range(n)}
    a3s = {a: A3(sh, m, b2s, a) for a in range(n)}
    full_bank = {pr: list(_exact(sh, m, b2s, a3s, n, Fraction(config["pi"]), pr)) for pr in LADDER}
    return {"kind": "ladder", "structure": structure, "exact": exact, "profiles": desc, "full_bank": full_bank,
            "checks": {"tree_vs_exact": tree_vs_exact, "reproductions": repro,
                       "min_eps": min(min(d["eps"]) for d in desc.values())},
            "tree": {"nodes": g.tree.N, "offer_leaves": len(g.tree.offer_nodes), "menus": list(g.M)},
            "seconds": round(time.time() - t0, 1), "cpu": round(time.process_time() - c0, 1)}


def _run(g, variant, warm, config, log):
    every, top = config["every"], config["max_iterations"]
    found = {}

    def check(it, avg):
        nc = g.nashconv(avg)
        log.append({"variant": variant, "iteration": it, "nashconv": nc["nashconv"], "eps": list(nc["eps"])})
        found["avg"], found["it"], found["nc"] = avg, it, nc["nashconv"]
        return nc["nashconv"] <= config["eps_bar"]

    avg = cfr.solve(g, top, variant=variant, warm=warm, checkpoints=tuple(range(every, top + 1, every)),
                    on_checkpoint=check)
    return avg, found["it"], found["nc"] <= config["eps_bar"]


def solve_job(config: dict, structure: str, start: str) -> dict:
    """One equilibrium solve of one structure from one start, per the run plan."""
    t0, c0 = time.time(), time.process_time()
    g = _game(config, structure)
    warm = None
    if start == "warm":
        lad = cfr.Ladder(g)
        warm = cfr.as_regrets((lad.profile("A3"), lad.profile("B2")))
    log: List[dict] = []
    avg, it, ok = _run(g, "cfr+", warm, config, log)
    variant = "cfr+"
    stage = ("budget" if it <= config["budget"] else "4x iterations") if ok else None
    if not ok:
        avg, it, ok = _run(g, "dcfr", warm, config, log)
        variant = "dcfr"
        stage = "dcfr" if ok else "not certified"
    return {"kind": "solve", "structure": structure, "start": start, "certified": ok, "stage": stage,
            "variant": variant, "iterations": it, "log": log, "profile": describe(g, avg),
            "seconds": round(time.time() - t0, 1), "cpu": round(time.process_time() - c0, 1)}


def run_job(args) -> dict:
    kind, config, structure, start = args
    return ladder_job(config, structure) if kind == "ladder" else solve_job(config, structure, start)


def jobs(config: dict) -> list:
    order = ("T3", "T2", "game:none/all", "game:all/none", "game:none/none")      # largest first
    return ([("solve", config, s, st) for s in order for st in STARTS] +
            [("ladder", config, s, None) for s in order])


def compute(config: dict = None, workers: int = 1, sd3k_path: Optional[str] = SD3K) -> dict:
    config = dict(config or G_CONFIG)
    todo = jobs(config)
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            results = list(ex.map(run_job, todo))
    else:
        results = [run_job(j) for j in todo]
    ladder = {r["structure"]: r for r in results if r["kind"] == "ladder"}
    solves = {}
    for r in results:
        if r["kind"] == "solve":
            solves.setdefault(r["structure"], {})[r["start"]] = r
    sd3k = None
    if sd3k_path and config["instance"] == list(cfr.G_INSTANCE) and os.path.exists(sd3k_path):
        with open(sd3k_path) as f:
            stored = json.load(f)
        sd3k = {gr["arm"]: {pr: {sc: {k: gr["pairings"][pr][sc][k] for k in ("YA", "YB")} for sc in SCOPES}
                            for pr in LADDER}
                for gr in stored["groups"] if gr["instance"] == config["instance"] and gr["config"] == "primary"}
    doc = {"sweep": SWEEP_VERSION, "config": config, "ladder": ladder, "solves": solves, "sd3k_q3_full": sd3k}
    doc["gate"] = gate(doc)
    doc["checks"] = checks(doc)
    return doc


# -- the gate --------------------------------------------------------------------------------------

def _verdict(vs: List[str]) -> str:
    if not vs:
        return "not certified"
    return vs[0] if len(set(vs)) == 1 else "equilibrium-dependent"


def _eqs(doc, structure) -> list:
    """(start, profile) of every certified equilibrium of a structure, in STARTS order."""
    runs = doc["solves"].get(structure, {})
    return [(st, runs[st]["profile"]) for st in STARTS if st in runs and runs[st]["certified"]]


def _hb(b: bool) -> str:
    return "holds" if b else "fails"


def e3_reading(h: Optional[dict]) -> dict:
    if h is None:
        return {"verdict": "not testable", "why": "no answering decisions at HIGH"}
    if min(h["p_asked"].values()) < E3_SUPPORT:
        return {"verdict": "not testable", "why": "support", "p_asked": h["p_asked"]}
    hi_ok = h["answer_rate"]["hi"] is not None and h["answer_rate"]["hi"] >= E3_HI
    post = h["p_lo_given_decline"]
    post_ok = True if post is None else post >= E3_POST
    return {"verdict": _hb(hi_ok and post_ok), "hi_answer_rate": h["answer_rate"]["hi"],
            "p_lo_given_decline": post, "no_declines": post is None, "p_asked": h["p_asked"]}


def gate(doc: dict) -> dict:
    lad = doc["ladder"]
    a3b2_t3 = lad["T3"]["profiles"]["A3/B2"]
    a3b2_t2 = lad["T2"]["profiles"]["A3/B2"]
    out = {}
    nc = a3b2_t3["nashconv"]
    out["E1"] = {"verdict": _hb(nc > E1_BAR), "nashconv": nc}
    eq3, eq2 = _eqs(doc, "T3"), _eqs(doc, "T2")
    yb_l = a3b2_t3["outcomes"]["all"]["YB"]
    e2 = [{"start": st, "YB_eq": p["outcomes"]["all"]["YB"], "YB_A3B2": yb_l,
           "verdict": _hb(p["outcomes"]["all"]["YB"] > yb_l)} for st, p in eq3]
    out["E2"] = {"verdict": _verdict([x["verdict"] for x in e2]), "readings": e2}
    e3 = [dict(e3_reading(p["high"]), start=st) for st, p in eq3]
    out["E3"] = {"verdict": _verdict([x["verdict"] for x in e3]), "readings": e3}
    gain_l = a3b2_t3["outcomes"]["lo"]["YA"] - a3b2_t2["outcomes"]["lo"]["YA"]
    e4 = [{"start_T3": s3, "start_T2": s2, "gain_eq": p3["outcomes"]["lo"]["YA"] - p2["outcomes"]["lo"]["YA"],
           "gain_A3B2": gain_l, "verdict": _hb(p3["outcomes"]["lo"]["YA"] - p2["outcomes"]["lo"]["YA"] < gain_l)}
          for s3, p3 in eq3 for s2, p2 in eq2]
    out["E4"] = {"verdict": _verdict([x["verdict"] for x in e4]), "readings": e4}
    cells = {c: _eqs(doc, s) for c, s in CELL.items()}
    e5 = []
    if all(cells.values()):
        keys = ("all/all", "all/none", "none/all", "none/none")
        for a in cells["all/all"]:
            for b in cells["all/none"]:
                for c in cells["none/all"]:
                    for d in cells["none/none"]:
                        combo = (a, b, c, d)
                        mtx = {k: {"YA": p["outcomes"]["all"]["YA"], "YB": p["outcomes"]["all"]["YB"]}
                               for k, (_, p) in zip(keys, combo)}
                        pd = _pd(mtx)
                        e5.append({"starts": {k: st for k, (st, _) in zip(keys, combo)}, "pd": pd,
                                   "verdict": _hb(not pd["holds"])})
    out["E5"] = {"verdict": _verdict([x["verdict"] for x in e5]), "readings": e5}
    out["verdict"] = {k: out[k]["verdict"] for k in ("E1", "E2", "E3", "E4", "E5")}
    return out


def checks(doc: dict) -> dict:
    lad = doc["ladder"].values()
    sol = [r for s in doc["solves"].values() for r in s.values()]
    return {"tree_vs_exact_ok": all(r["checks"]["tree_vs_exact"] < cfr.TOL for r in lad),
            "reproductions_ok": all(v < cfr.TOL for r in lad for v in r["checks"]["reproductions"].values()),
            "best_response_never_worse": all(r["checks"]["min_eps"] >= -cfr.TOL for r in lad) and
            all(min(r["profile"]["eps"]) >= -cfr.TOL for r in sol)}


# -- the note --------------------------------------------------------------------------------------

CONSEQUENCES = {
    ("E1", "fails"): "On 2-2-3 at Q = 2 the ladder is already within 0.01 Y of equilibrium at level 3. SD3-K’s values "
                     "can stand as equilibrium approximations there.",
    ("E1E2", "holds"): "“Reading backfires” is cited only as reading through a stale model. A live counterpart that "
                       "reads silence needs a correct or adaptive model of the focal side, and NEG-33 inherits that "
                       "constraint.",
    ("E2", "fails"): "B does worse at equilibrium than against A3. SD3-K’s off-gate finding is then not a level-k "
                     "artifact, and no write-up presumes that a reading counterpart helps B.",
    ("E3", "holds"): "K3’s consequence (no hiding treatment for the level) is confirmed at equilibrium on 2-2-3.",
    ("E3", "fails"): "The level does not unravel at equilibrium, so K3’s consequence is specific to level 3. The "
                     "hiding-treatment decision goes back to Meadowlark before NEG-33 uses it.",
    ("E4", "fails"): "Hiding is worth more at equilibrium than at level 3: the ladder understates the strategic content "
                     "of selective disclosure. The write-up says so wherever K4 is cited.",
    ("E5", "fails"): "A strict dilemma appears at equilibrium. K5’s consequence (“absent against every counterpart "
                     "tried”) is qualified to the ladder’s counterparts, and NEG-33 inherits the equilibrium framing.",
    ("unsettled", None): "The prediction is not settled. Nothing is inferred from it, and the write-up reports the "
                         "profiles found.",
}
READINGS = {"E1": "the ladder has not converged: NashConv(A3, B2) in T3 > 0.01 Y",
            "E2": "B2’s loss to A3 is a level-k artifact: Y_B(σ*_T3) > Y_B(A3, B2)",
            "E3": "the level unravels at equilibrium (hi answers HIGH ≥ 0.95, P(lo | decline) ≥ 0.9)",
            "E4": "hiding is worth even less at equilibrium than at level 3",
            "E5": "no all-or-nothing dilemma at equilibrium"}


def _f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def _e(x):
    return "—" if x is None else f"{x:.1e}"


def applied_consequences(v: dict) -> List[tuple]:
    """The registered consequences that the verdicts trigger, as (heading, quoted text)."""
    out = []
    if v["E1"] == "fails":
        out.append(("E1 fails", CONSEQUENCES[("E1", "fails")]))
    if v["E1"] == "holds" and v["E2"] == "holds":
        out.append(("E1 and E2 hold", CONSEQUENCES[("E1E2", "holds")]))
    for k, verdict in ((k, v[k]) for k in ("E2", "E3", "E4", "E5")):
        if (k, verdict) in CONSEQUENCES:
            out.append((f"{k} {verdict}", CONSEQUENCES[(k, verdict)]))
        elif verdict in ("not certified", "equilibrium-dependent"):
            out.append((f"{k} {verdict}", CONSEQUENCES[("unsettled", None)]))
    return out


def render(doc: dict, path: str) -> str:
    g, v = doc["gate"], doc["gate"]["verdict"]
    L = [f"# Split Deal v0.3 — gate SD3-E, the CFR check on {'-'.join(map(str, doc['config']['instance']))}", "",
         f"_Rendered by `sweep_e.py render` from `{path}` (sweep `{doc['sweep']}`, commit `{doc.get('commit', '?')}`, "
         f"{doc.get('seconds', '?')} s). Floating point; every number is formatted from the JSON. Predictions as "
         f"registered in `DESIGN_SPLIT_CFR.md` §5; their operational reading is fixed in the docstring of "
         f"`sweep_e.py`._", "",
         f"Game G: instance {'-'.join(map(str, doc['config']['instance']))}, Q = {doc['config']['q']}, "
         f"{doc['config']['bank']} bank, π = {doc['config']['pi']}. Equilibria by CFR+ from a uniform start and from "
         f"(A3, B2), certified when NashConv ≤ {doc['config']['eps_bar']} Y.", "",
         "## Gate SD3-E", "", "| prediction | reading | verdict |", "|---|---|---|"]
    for k in ("E1", "E2", "E3", "E4", "E5"):
        L.append(f"| {k} | {READINGS[k]} | **{v[k]}** |")
    L += ["", f"- **E1** **{v['E1']}**: NashConv(A3, B2) in T3 = {_f(g['E1']['nashconv'])}."]
    for x in g["E2"]["readings"]:
        L.append(f"- **E2** ({x['start']} start): Y_B(σ*_T3) {_f(x['YB_eq'])} against Y_B(A3, B2) "
                 f"{_f(x['YB_A3B2'])}: {x['verdict']}.")
    for x in g["E3"]["readings"]:
        if x["verdict"] == "not testable":
            asked = x.get("p_asked")
            more = f": P(B asks HIGH) lo {_pct(asked['lo'])}, hi {_pct(asked['hi'])}" if asked else ""
            L.append(f"- **E3** ({x['start']} start): not testable ({x['why']}{more}).")
        else:
            post = "no declines on path (read as met)" if x["no_declines"] else _pct(x["p_lo_given_decline"])
            L.append(f"- **E3** ({x['start']} start): hi answers HIGH {_pct(x['hi_answer_rate'])}; P(lo | decline) {post}; "
                     f"P(B asks HIGH) lo {_pct(x['p_asked']['lo'])}, hi {_pct(x['p_asked']['hi'])}: {x['verdict']}.")
    for x in g["E4"]["readings"]:
        L.append(f"- **E4** (T3 {x['start_T3']}, T2 {x['start_T2']}): lo’s gain from T3 over T2 at equilibrium {x['gain_eq']:+.4f} against {x['gain_A3B2']:+.4f} "
                 f"in (A3, B2): {x['verdict']}.")
    if g["E5"]["readings"]:
        x = g["E5"]["readings"][0]
        pd = x["pd"]
        L.append(f"- **E5** (every cell from the {STARTS[0]} start where certified): strict PD for A {pd['A_holds']}, "
                 f"for B {pd['B_holds']}; withholding dominant for A {pd['withholding_dominant_A']}, for B "
                 f"{pd['withholding_dominant_B']}: {x['verdict']}.")
        tally = {vd: sum(r["verdict"] == vd for r in g["E5"]["readings"]) for vd in ("holds", "fails")}
        L.append(f"- **E5** over all {len(g['E5']['readings'])} combinations of the cells’ certified equilibria: "
                 f"holds in {tally['holds']}, fails in {tally['fails']}.")
    for k in ("E2", "E3", "E4", "E5"):
        if not g[k]["readings"]:
            L.append(f"- **{k}**: {v[k]} (no certified equilibrium in a structure it needs).")
    c = doc["checks"]
    L += ["", f"Checks: ladder profiles on the tree equal the exact rollouts: {c['tree_vs_exact_ok']}; the best responses "
              f"to B0, A1 and B2 have the values of A1, B2 and A3: {c['reproductions_ok']}; a best response is never "
              f"worse than its profile: {c['best_response_never_worse']}.", "", "## Registered consequences", ""]
    for head, text in applied_consequences(v):
        L += [f"**{head}.** Registered: “{text}”", ""]
    L += ["## Solves", "", "| structure | start | certified | stage | variant | iterations | NashConv | ε_A | ε_B | "
          "seconds | CPU s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in STRUCTS:
        for st in STARTS:
            r = doc["solves"][s][st]
            p = r["profile"]
            L.append(f"| {s} | {st} | {r['certified']} | {r['stage']} | {r['variant']} | {r['iterations']} | "
                     f"{_e(p['nashconv'])} | {_e(p['eps'][0])} | {_e(p['eps'][1])} | {r['seconds']} | {r['cpu']} |")
    L += ["", "## Exploitability (reported)", "",
          "| structure | profile | NashConv | Y_A | Y_B | A’s best response | B’s best response | B at A’s best response | "
          "A at B’s best response |", "|---|---|---|---|---|---|---|---|---|"]
    for s in STRUCTS:
        rows = [(pr, doc["ladder"][s]["profiles"][pr]) for pr in LADDER]
        rows += [(f"σ* ({st})", doc["solves"][s][st]["profile"]) for st in STARTS]
        for name, p in rows:
            L.append(f"| {s} | {name} | {_e(p['nashconv'])} | {_f(p['value'][0])} | {_f(p['value'][1])} | "
                     f"{_f(p['br_value'][0])} | {_f(p['br_value'][1])} | {_f(p['b_at_a_br'])} | {_f(p['a_at_b_br'])} |")
    L += ["", "## Behaviour (reported)", "", "| structure | profile | A level | Y_A | Y_B | Y_A common | Y_B common | "
          "agree, A proposes | agree, B proposes | E |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in STRUCTS:
        rows = [(pr, doc["ladder"][s]["profiles"][pr]) for pr in LADDER]
        rows += [(f"σ* ({st})" + ("" if doc["solves"][s][st]["certified"] else ", not certified"),
                  doc["solves"][s][st]["profile"]) for st in STARTS]
        for name, p in rows:
            for sc in SCOPES:
                o = p["outcomes"][sc]
                L.append(f"| {s} | {name} | {sc} | {_f(o['YA'])} | {_f(o['YB'])} | {_f(o['YA_c'])} | {_f(o['YB_c'])} | "
                         f"{_pct(o['agree_a'])} | {_pct(o['agree_b'])} | {_f(o['E'])} |")
    L += ["", "**A’s answering at B’s HIGH in T3** (masses from the prior; answer rate = answered / asked):", "",
          "| profile | round | lo asked | lo answers | hi asked | hi answers |", "|---|---|---|---|---|---|"]
    rows = [(pr, doc["ladder"]["T3"]["profiles"][pr]) for pr in LADDER]
    rows += [(f"σ* ({st})", doc["solves"]["T3"][st]["profile"]) for st in STARTS]
    for name, p in rows:
        h = p["high"]
        if h is None:
            continue
        for rd, r in h["rounds"].items():
            rate = {lvl: (r[lvl]["answered"] / r[lvl]["asked"]) if r[lvl]["asked"] else None for lvl in ("lo", "hi")}
            L.append(f"| {name} | {rd} | {_f(r['lo']['asked'])} | {_pct(rate['lo'])} | {_f(r['hi']['asked'])} | "
                     f"{_pct(rate['hi'])} |")
        L.append(f"| {name} | pooled P(lo given decline) | {_pct(h['p_lo_given_decline'])} | | | |")
    L += ["", "## Bridges (reported)", "", "| structure | pairing | Y_A on G | Y_B on G | Y_A, full bank, Q = 2 | "
          "Y_B, full bank, Q = 2 | Y_A, SD3-K (full bank, Q = 3) | Y_B, SD3-K |", "|---|---|---|---|---|---|---|---|"]
    for s in STRUCTS:
        lad = doc["ladder"][s]
        for pr in LADDER:
            k = (doc["sd3k_q3_full"] or {}).get(s, {}).get(pr, {}).get("all")
            L.append(f"| {s} | {pr} | {_f(lad['exact'][pr][0])} | {_f(lad['exact'][pr][1])} | "
                     f"{_f(lad['full_bank'][pr][0])} | {_f(lad['full_bank'][pr][1])} | "
                     f"{_f(k['YA']) if k else '—'} | {_f(k['YB']) if k else '—'} |")
    L += ["", "Not run here: the extension of these verdicts to the nine instances at Q = 3 (§7 of `DESIGN_SPLIT_CFR.md`)."]
    return "\n".join(L) + "\n"


# -- commands ----------------------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run(args) -> int:
    t0 = time.time()
    doc = _jsonable(compute(workers=args.workers))
    doc["commit"] = _commit()
    doc["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out} ({doc['seconds']} s); " + ", ".join(f"{k} {x}" for k, x in doc["gate"]["verdict"].items()))
    return 0


def render_cmd(args) -> int:
    with open(args.json) as f:
        doc = json.load(f)
    with open(args.note, "w") as f:
        f.write(render(doc, args.json))
    print(f"wrote {args.note}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=WORKERS)
    n = sub.add_parser("render")
    n.add_argument("json")
    n.add_argument("--note", required=True)
    args = ap.parse_args(argv)
    return {"run": run, "render": render_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
