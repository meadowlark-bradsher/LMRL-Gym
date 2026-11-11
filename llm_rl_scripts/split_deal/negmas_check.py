"""Cross-check of the L0 frontier against NegMAS (NEG-29).

NegMAS is AGPL-3.0-or-later. This is the only module in ``split_deal`` that imports it, and nothing
in the package imports this module (``test_negmas_check.py`` enforces both), so the game, the oracles
and the sweeps never depend on it. It is a test-only dependency, installed in its own environment:

    python3 -m venv <env> && <env>/bin/pip install negmas==0.17.0
    PYTHONPATH=. <env>/bin/python -m llm_rl_scripts.split_deal.negmas_check \
        --out results/split_negmas_check_20261111/check.json

For every ordered pair of types, NegMAS builds both sides' utility functions from the type's per-unit
values (linear additive over one integer issue per item kind), enumerates the outcomes itself, and
computes the Pareto frontier (``pareto_frontier``) and the Nash point(s) (``nash_points``). These are
compared with ours:

    v0.2, both regimes: the acceptable set, S* (``frontier.pair_frontier`` and ``sweep_l1._s_star``),
                        the Pareto set and the Nash point(s) of ``frontier.pair_frontier``;
    v0.3, own d each:   S* of ``private_d.PrivateTables.s_star``.

Two conventions are aligned before comparing, both forced by NegMAS's API rather than chosen:

- NegMAS's frontier is over all outcomes and ``nash_points`` multiplies (u - reserved) without checking
  rationality (two negative factors give a positive product), so its frontier is restricted to the
  outcomes both sides accept before the Nash search. A Pareto-dominated acceptable outcome is dominated
  by an acceptable one, so this restriction equals our Pareto set among the acceptable outcomes.
- Ours keeps every acceptable maximiser of the Nash product, NegMAS searches its frontier only. They
  differ only where the maximal product is 0; those pairs ("degenerate") compare the Nash points that
  lie on the frontier and are counted.

Sets are compared as sets of utility pairs. NegMAS returns one outcome per frontier point while ours keeps
every allocation giving a frontier pair, so for the Pareto set NegMAS's outcomes must be a subset of ours;
pairs where ours holds more (ties in utility) are counted as ``pareto_ties``. Values are compared within
``TOL``; all of them are integers or halves, so any drift at all is a disagreement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Tuple

from negmas import __version__ as NEGMAS_VERSION
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction, nash_points, pareto_frontier
from negmas.preferences.value_fun import AffineFun, LinearFun

from .belief import Tables
from .frontier import pair_frontier, utilities
from .game import ITEMS, REGIMES, Instance, canonical_instances, outside_option
from .private_d import PrivateTables
from .sweep_l1 import _commit, _s_star

CHECK_VERSION = "split-deal-negmas-check-v0.1"
TOL = 1e-9


class NegmasPair:
    """NegMAS's view of one instance: its outcome space, its enumeration and per-type utility functions."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self.os = make_os([make_issue((0, c), ITEMS[k]) for k, c in enumerate(inst.counts)])
        self.outcomes = list(self.os.enumerate())

    def ufun_a(self, v, d: float):
        return LinearAdditiveUtilityFunction(values=[LinearFun(x) for x in v], weights=[1.0] * len(v),
                                             outcome_space=self.os, reserved_value=float(d))

    def ufun_b(self, v, d: float):
        return LinearAdditiveUtilityFunction(values=[AffineFun(-x, x * c) for x, c in zip(v, self.inst.counts)],
                                             weights=[1.0] * len(v), outcome_space=self.os,
                                             reserved_value=float(d))

    def solve(self, va, vb, da: float, db: float) -> dict:
        ua, ub = self.ufun_a(va, da), self.ufun_b(vb, db)
        utils = {o: (float(ua(o)), float(ub(o))) for o in self.outcomes}
        acceptable = {o for o, (x, y) in utils.items() if x >= da - TOL and y >= db - TOL}
        frontier, idx = pareto_frontier([ua, ub], outcomes=self.outcomes)
        rational = [(tuple(map(float, p)), self.outcomes[i]) for p, i in zip(frontier, idx)
                    if p[0] >= da - TOL and p[1] >= db - TOL]
        if not rational:
            return {"acceptable": acceptable, "s_star": None, "pareto": set(), "pareto_outcomes": set(),
                    "nash": set()}
        pts = [p for p, _ in rational]
        nash = nash_points([ua, ub], pts, outcomes=self.outcomes)
        return {"acceptable": acceptable,
                "s_star": max(x + y for x, y in pts) - da - db,
                "pareto": set(pts), "pareto_outcomes": {o for _, o in rational},
                "nash": {tuple(map(float, p)) for p, _ in nash}}


def _pairs_set(inst, va, vb, allocs) -> set:
    return {tuple(float(u) for u in utilities(inst, va, vb, a)) for a in allocs}


def _close(x, y) -> bool:
    if x is None or y is None:
        return x is None and y is None
    return abs(float(x) - float(y)) <= TOL


def _sets_close(s, t) -> bool:
    return len(s) == len(t) and all(any(all(abs(a - b) <= TOL for a, b in zip(p, q)) for q in t) for p in s)


def check_v02(inst: Instance, regime: str) -> dict:
    np_ = NegmasPair(inst)
    tab = Tables(inst, regime)
    d = outside_option(regime)
    out = {"instance": inst.key, "regime": regime, "pairs": 0, "pairs_with_deal": 0, "degenerate": 0,
           "pareto_ties": 0,
           "mismatch": {k: 0 for k in ("acceptable", "s_star", "s_star_sweep_l1", "pareto", "pareto_outcomes",
                                       "nash")},
           "max_abs_s_star_diff": 0.0}
    types = inst.types()
    for va in types:
        for vb in types:
            out["pairs"] += 1
            ours = pair_frontier(inst, va, vb, d)
            theirs = np_.solve(va, vb, float(d), float(d))
            m = out["mismatch"]
            m["acceptable"] += set(ours.acceptable) != theirs["acceptable"]
            m["s_star"] += not _close(ours.s_star, theirs["s_star"])
            m["s_star_sweep_l1"] += not _close(_s_star(tab, va, vb), theirs["s_star"])
            if ours.s_star is not None and theirs["s_star"] is not None:
                out["max_abs_s_star_diff"] = max(out["max_abs_s_star_diff"],
                                                 abs(float(ours.s_star) - theirs["s_star"]))
            if ours.s_star is None:
                continue
            out["pairs_with_deal"] += 1
            m["pareto"] += not _sets_close(_pairs_set(inst, va, vb, ours.pareto), theirs["pareto"])
            m["pareto_outcomes"] += not theirs["pareto_outcomes"] <= set(ours.pareto)
            out["pareto_ties"] += len(ours.pareto) > len(theirs["pareto_outcomes"])
            best = max((float(x) - float(d)) * (float(y) - float(d))
                       for x, y in _pairs_set(inst, va, vb, ours.nash))
            nash_ours = _pairs_set(inst, va, vb, ours.nash)
            if best <= TOL:
                out["degenerate"] += 1
                nash_ours = nash_ours & _pairs_set(inst, va, vb, ours.pareto)
            m["nash"] += not _sets_close(nash_ours, theirs["nash"])
    return out


def check_v03(inst: Instance) -> dict:
    np_ = NegmasPair(inst)
    tab = PrivateTables(inst)
    out = {"instance": inst.key, "pairs": 0, "pairs_with_deal": 0, "mismatch": {"s_star": 0},
           "max_abs_s_star_diff": 0.0}
    for ta in tab.types:
        for tb in tab.types:
            out["pairs"] += 1
            ours = tab.s_star(ta, tb)
            theirs = np_.solve(ta[0], tb[0], float(tab.d(ta)), float(tab.d(tb)))["s_star"]
            out["mismatch"]["s_star"] += not _close(ours, theirs)
            if ours is not None:
                out["pairs_with_deal"] += 1
                if theirs is not None:
                    out["max_abs_s_star_diff"] = max(out["max_abs_s_star_diff"], abs(float(ours) - theirs))
    return out


def compute(instances: List[Instance] = None) -> dict:
    instances = instances or canonical_instances()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v02 = [check_v02(i, r) for i in instances for r in REGIMES]
        v03 = [check_v03(i) for i in instances]
    total = sum(sum(g["mismatch"].values()) for g in v02 + v03)
    return {"check": CHECK_VERSION, "negmas": NEGMAS_VERSION, "tol": TOL,
            "instances": [i.key for i in instances], "agree": total == 0, "mismatches": total,
            "v02": v02, "v03": v03}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    doc = compute()
    doc["commit"] = _commit()
    doc["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    pairs = sum(g["pairs"] for g in doc["v02"]) + sum(g["pairs"] for g in doc["v03"])
    print(f"wrote {args.out} ({doc['seconds']} s): NegMAS {doc['negmas']}, {pairs} type pairs, "
          f"{doc['mismatches']} mismatches, degenerate Nash pairs {sum(g['degenerate'] for g in doc['v02'])}")
    return 0 if doc["agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
