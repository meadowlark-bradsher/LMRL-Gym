"""One-sided Car Dealer: what the opponent's words are worth when the rules decide its moves.

Design evidence for a reshaped Stage 2 (no API, exact-on-model). The focal side is the only
proposer; the opponent never proposes, asks or holds. Its moves are a rule, not a model:

* The focal spends one of H opportunities per move: OFFER d, ASK q (one of the nine bank
  questions, answered truthfully), or WALK (free, ends with no deal).
* An OFFER made with m opportunities left (m - 1 after it) is accepted iff the opponent's
  utility under its planted ranking is >= aspiration(m - 1) = r + (alpha0 - r)(m - 1)/H,
  the T1 acceptance rule of ``car_dealer_live.opponent`` at accept_noise 0 (so the last offer
  is judged at r, as at the live deadline). A rejection is exact evidence: every ranking that
  would have accepted is eliminated.
* ``hazard`` h: after any move that does not end the game, the opponent walks with
  probability h, the same for every ranking (time is costly, and the walk carries no evidence).
* Payoff: net gain U_focal(d) - r on acceptance, 0 otherwise, minus ask_cost per question.

Because every observation is deterministic given the ranking, the posterior is always uniform
over a surviving subset of the six rankings, so each value below is an exact Fraction.

Values per cell (focal role x focal ranking), at the uniform prior over the opponent's ranking:

    V_one     one offer, no information, best timing           (nothing learned at all)
    V_offers  offers only, learning from rejections            (the words are ignored)
    V_words   offers + questions with exactly read answers     (the oracle-read words)
    V_reveal  the opponent's ranking known                      (everything words could carry)

    D_offers = V_offers - V_one, D_words = V_words - V_offers, D_rest = V_reveal - V_words.

D_words is the ceiling a likelihood seat can recover from the answers; D_words + D_rest bounds
anything the opponent's words could carry. Ignoring answers is the same as not asking: a
question whose answer is ignored is weakly dominated by offering the focal's own ideal deal,
which every ranking rejects (opponent utility 0 < r), so V_offers forbids ASK.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.sweep_onesided run \
        --out results/onesided_sweep_20261109/sweep.json
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.sweep_onesided render \
        results/onesided_sweep_20261109/sweep.json \
        --anchor results/noul_stage1_20261108/analysis.json \
        --note llm_rl_scripts/car_dealer_noul/ONESIDED_SWEEP.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS, utility
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS
from llm_rl_scripts.car_dealer_live.protocol_live import ROLES, other

SWEEP_VERSION = "onesided-car-dealer-v0.1"
CHANNELS = ("offers", "words", "reveal")
N_TYPES = len(ALL_RANKINGS)
ALL_TYPES = frozenset(range(N_TYPES))
# question index -> the rankings (indices) answering yes
YES_SETS: Tuple[FrozenSet[int], ...] = tuple(
    frozenset(i for i, y in enumerate(q.truthful_yes) if y) for q in BANK)


@dataclass(frozen=True)
class OneSidedConfig:
    regime: str = "easy"
    horizon: int = 4
    alpha0: Optional[Fraction] = Fraction(4, 5)   # None = flat: the opponent accepts iff U >= r throughout
    hazard: Fraction = Fraction(0)
    ask_cost: Fraction = Fraction(0)

    @property
    def reservation(self) -> Fraction:
        return RESERVATIONS[self.regime]

    def aspiration(self, remaining: int) -> Fraction:
        r = self.reservation
        if self.alpha0 is None:
            return r
        return r + (self.alpha0 - r) * Fraction(remaining, self.horizon)

    def key(self) -> str:
        a = "flat" if self.alpha0 is None else str(self.alpha0)
        return f"{self.regime}|H{self.horizon}|a{a}|h{self.hazard}|c{self.ask_cost}"

    def to_json(self) -> dict:
        return {"regime": self.regime, "reservation": str(self.reservation), "horizon": self.horizon,
                "alpha0": "flat" if self.alpha0 is None else str(self.alpha0),
                "hazard": str(self.hazard), "ask_cost": str(self.ask_cost)}


class Cell:
    """Exact solver for one focal (role, ranking) under one config."""

    def __init__(self, cfg: OneSidedConfig, role: str, ranking):
        self.cfg = cfg
        r = cfg.reservation
        self.gain = [utility(role, ranking, d) - r for d in ALL_DEALS]
        self.u_opp = [[utility(other(role), t, d) for t in ALL_RANKINGS] for d in ALL_DEALS]
        self.keep = 1 - cfg.hazard
        self._memo: Dict[tuple, tuple] = {}

    def _accepters(self, d: int, m: int, S: FrozenSet[int]) -> FrozenSet[int]:
        a = self.cfg.aspiration(m - 1)
        return frozenset(t for t in S if self.u_opp[d][t] >= a)

    def best(self, m: int, S: FrozenSet[int], asked: FrozenSet[int], ask: bool) -> tuple:
        """(value, best move, value of the best non-ASK move); move = ('walk',) / ('offer', d) / ('ask', q).
        Ties go to the non-ASK move, so a question is counted only where it is strictly optimal."""
        if m == 0:
            return (Fraction(0), ("walk",), Fraction(0))
        key = (m, S, asked if ask else None, ask)
        if key in self._memo:
            return self._memo[key]
        n = len(S)
        best_v, best_mv = Fraction(0), ("walk",)
        for d in range(len(ALL_DEALS)):
            acc = self._accepters(d, m, S)
            v = Fraction(len(acc), n) * self.gain[d]
            rej = S - acc
            if rej and m > 1:
                v += Fraction(len(rej), n) * self.keep * self.best(m - 1, rej, asked, ask)[0]
            if v > best_v:
                best_v, best_mv = v, ("offer", d)
        noask_v = best_v
        if ask and n > 1:
            for q in range(len(BANK)):
                if q in asked:
                    continue
                v = -self.cfg.ask_cost
                for part in (S & YES_SETS[q], S - YES_SETS[q]):
                    if part:
                        v += Fraction(len(part), n) * self.keep * self.best(m - 1, part, asked | {q}, ask)[0]
                if v > best_v:
                    best_v, best_mv = v, ("ask", q)
        out = (best_v, best_mv, noask_v)
        self._memo[key] = out
        return out

    def policy_stats(self, m: int, S: FrozenSet[int], asked: FrozenSet[int], ask: bool) -> Dict[str, Fraction]:
        """Expected questions, offers and P(deal) under the optimal policy from (m, S), uniform over S."""
        zero = {"asks": Fraction(0), "offers": Fraction(0), "p_deal": Fraction(0)}
        if m == 0:
            return zero
        _, mv, _ = self.best(m, S, asked, ask)
        n = len(S)
        if mv[0] == "walk":
            return zero
        out = dict(zero)
        if mv[0] == "offer":
            acc = self._accepters(mv[1], m, S)
            out["offers"] = Fraction(1)
            out["p_deal"] = Fraction(len(acc), n)
            branches = [(S - acc, asked)]
        else:
            q = mv[1]
            out["asks"] = Fraction(1)
            branches = [(S & YES_SETS[q], asked | {q}), (S - YES_SETS[q], asked | {q})]
        for part, nxt in branches:
            if part and m > 1:
                sub = self.policy_stats(m - 1, part, nxt, ask)
                w = Fraction(len(part), n) * self.keep
                for k in out:
                    out[k] += w * sub[k]
        return out

    def v_one(self) -> Fraction:
        """One offer, no information, at the best opportunity (earlier ones spent on sure rejections)."""
        H = self.cfg.horizon
        best = Fraction(0)
        for m in range(1, H + 1):
            surv = self.keep ** (H - m)
            for d in range(len(ALL_DEALS)):
                acc = self._accepters(d, m, ALL_TYPES)
                best = max(best, surv * Fraction(len(acc), N_TYPES) * self.gain[d])
        return best

    def solve(self) -> dict:
        H = self.cfg.horizon
        none = frozenset()
        v_off = self.best(H, ALL_TYPES, none, ask=False)[0]
        v_words, root, root_noask = self.best(H, ALL_TYPES, none, ask=True)
        v_rev = sum((self.best(H, frozenset([t]), none, ask=False)[0] for t in range(N_TYPES)), Fraction(0)) / N_TYPES
        stats_w = self.policy_stats(H, ALL_TYPES, none, ask=True)
        stats_o = self.policy_stats(H, ALL_TYPES, none, ask=False)
        return {"V_one": self.v_one(), "V_offers": v_off, "V_words": v_words, "V_reveal": v_rev,
                "root_words": root[0], "words_asks": stats_w["asks"], "words_offers": stats_w["offers"],
                "words_p_deal": stats_w["p_deal"], "offers_offers": stats_o["offers"],
                "offers_p_deal": stats_o["p_deal"]}


VALUE_KEYS = ("V_one", "V_offers", "V_words", "V_reveal")
STAT_KEYS = ("words_asks", "words_offers", "words_p_deal", "offers_offers", "offers_p_deal")


def solve_config(cfg: OneSidedConfig) -> dict:
    cells = []
    for role in ROLES:
        for ti, ranking in enumerate(ALL_RANKINGS):
            res = Cell(cfg, role, ranking).solve()
            cells.append({"role": role, "focal_type": ti, **res})
    y = 1 - cfg.reservation
    n = len(cells)
    mean = {k: sum((c[k] for c in cells), Fraction(0)) / n for k in VALUE_KEYS + STAT_KEYS}
    summary = {k: float(mean[k] / y) for k in VALUE_KEYS}
    summary.update({"D_offers": float((mean["V_offers"] - mean["V_one"]) / y),
                    "D_words": float((mean["V_words"] - mean["V_offers"]) / y),
                    "D_rest": float((mean["V_reveal"] - mean["V_words"]) / y),
                    "D_info": float((mean["V_reveal"] - mean["V_one"]) / y)})
    summary.update({k: float(mean[k]) for k in STAT_KEYS})
    summary["cells_D_words_positive"] = sum(1 for c in cells if c["V_words"] > c["V_offers"])
    summary["cells_root_ask"] = sum(1 for c in cells if c["root_words"] == "ask")
    summary["exact_mean_Y"] = {k: str(mean[k] / y) for k in VALUE_KEYS}
    return {"config": cfg.to_json(), "key": cfg.key(), "y_scale": float(y), "n_cells": n, "summary": summary,
            "cells": [{k: (str(v) if isinstance(v, Fraction) else v) for k, v in c.items()} for c in cells]}


def grid(horizons, regimes, alphas, hazards, costs) -> List[OneSidedConfig]:
    return [OneSidedConfig(regime=g, horizon=H, alpha0=a, hazard=h, ask_cost=c)
            for g, a, h, c, H in itertools.product(regimes, alphas, hazards, costs, horizons)]


def _parse_alpha(s: str) -> Optional[Fraction]:
    return None if s == "flat" else Fraction(s)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(args) -> int:
    cfgs = grid([int(x) for x in args.horizons.split(",")], args.regimes.split(","),
                [_parse_alpha(x) for x in args.alpha0.split(",")],
                [Fraction(x) for x in args.hazards.split(",")], [Fraction(x) for x in args.ask_costs.split(",")])
    t0 = time.time()
    out = []
    for i, cfg in enumerate(cfgs):
        out.append(solve_config(cfg))
        s = out[-1]["summary"]
        print(f"[{i + 1}/{len(cfgs)}] {cfg.key():40s} D_offers {s['D_offers']:.4f} D_words {s['D_words']:.4f} "
              f"D_rest {s['D_rest']:.4f} asks {s['words_asks']:.2f}", flush=True)
    doc = {"sweep": SWEEP_VERSION, "commit": _commit(), "seconds": round(time.time() - t0, 1),
           "argv": {k: v for k, v in vars(args).items() if k not in ("func", "out")},
           "configs": out}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out} ({len(out)} configs, {doc['seconds']} s)")
    return 0


# -- note -----------------------------------------------------------------------

def _find(doc, **want):
    for c in doc["configs"]:
        if all(c["config"][k] == v for k, v in want.items()):
            yield c


def _table(rows, cols) -> List[str]:
    lines = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f(r) for _, f in cols) + " |")
    return lines


def render(doc: dict, anchor: Optional[dict], sweep_path: str, anchor_path: Optional[str]) -> str:
    f4 = lambda k: (lambda c: f"{c['summary'][k]:.4f}")
    cols = [("H", lambda c: str(c["config"]["horizon"])), ("V_one", f4("V_one")), ("V_offers", f4("V_offers")),
            ("V_words", f4("V_words")), ("V_reveal", f4("V_reveal")), ("D_offers", f4("D_offers")),
            ("**D_words**", f4("D_words")), ("D_rest", f4("D_rest")),
            ("words share of D_info", lambda c: (f"{c['summary']['D_words'] / c['summary']['D_info']:.2f}"
                                                  if c["summary"]["D_info"] > 0 else "—")),
            ("questions / ep", lambda c: f"{c['summary']['words_asks']:.2f}"),
            ("cells D_words > 0", lambda c: f"{c['summary']['cells_D_words_positive']} / {c['n_cells']}")]
    L = [f"# One-sided Car Dealer — the value of the opponent's words, by horizon",
         "",
         f"_Rendered by `sweep_onesided.py render` from `{sweep_path}` (sweep `{doc['sweep']}`, commit "
         f"`{doc['commit']}`, {len(doc['configs'])} configs, {doc['seconds']} s). Exact on the model: no sampling, no API; "
         "every number is formatted from the JSON. Design evidence, not registered._",
         "",
         "The rules decide the opponent's moves (accept iff its utility under the planted ranking ≥ the T1 "
         "aspiration; it never proposes); its only words are truthful answers to the nine bank questions. "
         "Values are means over 12 cells (2 focal roles × 6 focal rankings) at the uniform prior over the "
         "opponent's six rankings, in Y = net gain / (1 − r). `V_one` one offer, nothing learned; `V_offers` "
         "offers only, learning from rejections (words ignored); `V_words` + exactly read answers; `V_reveal` "
         "ranking known. `D_words = V_words − V_offers` is what a likelihood seat could recover from the answers; "
         "`D_words + D_rest` bounds anything the words could carry. Questions / ep is under the `V_words` policy "
         "(ties go to not asking); it can be positive where `D_words` is 0, when the words policy opens with a "
         "different offer and a question is strictly optimal further down. The aspiration schedule is stretched "
         "over H, so a longer game is not a superset of a shorter one: under a hazard, values can fall with H.",
         ""]
    if anchor:
        fo = anchor["ceilings"]["focal_opens"]
        L += [f"Anchor (two-sided live game, `{anchor_path}`, H = 4, easy, T1 kernel, focal opens): value of asking "
              f"Δ*_menu = {fo['mean_delta_menu_Y']:.4f} Y (ask {fo['mean_menu_ask'] / fo['y_scale']:.4f}, "
              f"no-ask {fo['mean_menu_noask'] / fo['y_scale']:.4f}).", ""]
    variants = [("Registered T1 aspiration (α₀ = 4/5), no hazard", "4/5", "0"),
                ("Registered T1 aspiration (α₀ = 4/5), hazard 1/10 per move", "4/5", "1/10"),
                ("Flat aspiration (accept iff U ≥ r throughout), no hazard", "flat", "0"),
                ("Flat aspiration, hazard 1/10 per move", "flat", "1/10")]
    for regime in ("easy", "hard"):
        for title, a, h in variants:
            rows = sorted(_find(doc, regime=regime, alpha0=a, hazard=h, ask_cost="0"),
                          key=lambda c: c["config"]["horizon"])
            if not rows:
                continue
            r = rows[0]["config"]["reservation"]
            L += [f"## {regime} (r = {r}) — {title}, ask cost 0", ""] + _table(rows, cols) + [""]
    costs = sorted({c["config"]["ask_cost"] for c in doc["configs"]}, key=Fraction)
    if len(costs) > 1:
        L += ["## Question cost (easy, α₀ = 4/5, no hazard): D_words by horizon", ""]
        hs = sorted({c["config"]["horizon"] for c in doc["configs"]})
        hdr = "| ask cost | " + " | ".join(f"H = {H}" for H in hs) + " |"
        L += [hdr, "|" + "---|" * (len(hs) + 1)]
        for cost in costs:
            vals = []
            for H in hs:
                m = list(_find(doc, regime="easy", alpha0="4/5", hazard="0", ask_cost=cost, horizon=H))
                vals.append(f"{m[0]['summary']['D_words']:.4f} ({m[0]['summary']['words_asks']:.2f} q)" if m else "—")
            L.append(f"| {cost} | " + " | ".join(vals) + " |")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def render_cmd(args) -> int:
    with open(args.sweep) as f:
        doc = json.load(f)
    anchor = None
    if args.anchor:
        with open(args.anchor) as f:
            anchor = json.load(f)
    text = render(doc, anchor, args.sweep, args.anchor)
    with open(args.note, "w") as f:
        f.write(text)
    print(f"wrote {args.note}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--horizons", default="1,2,3,4,5,6,7,8")
    r.add_argument("--regimes", default="easy,hard")
    r.add_argument("--alpha0", default="4/5,flat")
    r.add_argument("--hazards", default="0,1/10")
    r.add_argument("--ask-costs", default="0,1/100,3/100")
    r.set_defaults(func=run)
    n = sub.add_parser("render")
    n.add_argument("sweep")
    n.add_argument("--anchor")
    n.add_argument("--note", required=True)
    n.set_defaults(func=render_cmd)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
