"""Structural numbers cited by the CFR check design (`DESIGN_SPLIT_CFR.md` §3).

Counts only: the size of the public game tree of the smallest instance, 2-2-3, on the v0.3 game, per
information structure, talk length Q and question bank. No strategy is played and no value is computed,
so nothing registered in the note is previewed.

The tree is the one the ladder already lives in (`twosided.py`, `readers.py`): A asks only when B will
answer and only a question that splits A's answer-consistent belief about B; B asks only a question it has
not had declined and that splits its answer-consistent belief about A; either may pass. Counted per tree:

* talk paths: public talk histories that reach the final stage;
* decision points: (own type, public history) pairs at which A or B moves in the talk (asking or passing,
  and for A in T3 answering or declining), i.e. information sets, each own type counted only where it is
  consistent with its own answers so far;
* offer points: the same at the final stage, for both seats (the proposer is drawn after the talk);
* leaf pairs: type pairs reaching a final stage, summed over talk paths. One CFR pass evaluates each of
  them for every offer (allocations plus walking) in each seat.

The merged bank keeps one question per distinct partition of the type set (a question and its complement
being the same partition), the one with the lowest bank index.

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.spec_numbers_cfr run \
        --out results/split_cfr_spec_20261111/spec_numbers.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.spec_numbers_cfr render \
        results/split_cfr_spec_20261111/spec_numbers.json --note llm_rl_scripts/split_deal/DESIGN_SPLIT_CFR.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from typing import Dict, List, Tuple

from .game import Instance
from .private_d import PrivateTables
from .sweep_l1 import _commit
from .twosided import GAME, STRUCTURES, Structure

SPEC_VERSION = "split-deal-cfr-spec-v0.1"
INSTANCE = (2, 2, 3)
QS = (1, 2, 3)
ARMS = {"T1": STRUCTURES["T1"], "T2": STRUCTURES["T2"], "T3": STRUCTURES["T3"],
        **{f"game:{a}/{b}": s for (a, b), s in GAME.items()}}
BEGIN, END = "<!-- spec-numbers-cfr:begin -->", "<!-- spec-numbers-cfr:end -->"


def merged_questions(tab: PrivateTables) -> List[int]:
    """Bank indices kept by the merged bank: the lowest index of each distinct non-trivial partition."""
    seen, keep = set(), []
    for j, y in enumerate(tab.yes):
        key = min(y, tab.full & ~y)
        if key and key not in seen:
            seen.add(key)
            keep.append(j)
    return keep


def tree_counts(tab: PrivateTables, q: int, st: Structure, questions: List[int]) -> Dict[str, int]:
    yes = tuple(tab.yes[j] for j in questions)
    n_offers = len(tab.allocs) + 1

    @lru_cache(maxsize=None)
    def f(r: int, phase: str, SA: int, SB: int, declined: frozenset, last_b: str) -> Tuple[int, int, int, int, int]:
        na, nb = SA.bit_count(), SB.bit_count()
        if r == q:
            return 1, 0, 0, na + nb, na * nb
        out = [0, 0, 0, 0, 0]

        def add(x):
            for i, v in enumerate(x):
                out[i] += v

        if phase == "A":
            if st.a_asks:
                out[1] += na
            add(f(r, "B", SA, SB, declined, last_b))                                   # pass
            b_will = st.b_answers == "all" or (st.b_answers == "reciprocal" and last_b != "declined")
            if st.a_asks and b_will:
                for y in yes:
                    for part in (SB & y, SB & ~y):
                        if part and part != SB:
                            add(f(r, "B", SA, part, declined, last_b))
        else:
            if st.b_asks:
                out[2] += nb
            add(f(r + 1, "A", SA, SB, declined, last_b))                               # pass
            if st.b_asks:
                for j, y in enumerate(yes):
                    if j in declined or not SA & y or not SA & ~y:
                        continue
                    if st.a_answers == "choose":
                        out[1] += na                                                     # A: answer or decline
                    if st.a_answers != "none":
                        for part in (SA & y, SA & ~y):
                            add(f(r + 1, "A", part, SB, declined, "answered"))
                    if st.a_answers != "all":
                        add(f(r + 1, "A", SA, SB, declined | {j}, "declined"))
        return tuple(out)

    paths, a_pts, b_pts, offer_pts, pairs = f(0, "A", tab.full, tab.full, frozenset(), "none")
    return {"talk_paths": paths, "a_points": a_pts, "b_points": b_pts, "offer_points": offer_pts,
            "leaf_pairs": pairs, "leaf_evaluations": 2 * n_offers * pairs}


def compute() -> dict:
    tab = PrivateTables(Instance(INSTANCE))
    banks = {"full": list(range(len(tab.questions))), "merged": merged_questions(tab)}
    rows = []
    for bank, qs in banks.items():
        for q in QS:
            for arm, st in ARMS.items():
                rows.append({"bank": bank, "q": q, "arm": arm, **tree_counts(tab, q, st, qs)})
    return {"spec": SPEC_VERSION, "instance": "-".join(map(str, INSTANCE)), "types": len(tab.types),
            "allocations": len(tab.allocs), "offers_per_seat": len(tab.allocs) + 1,
            "banks": {k: [tab.questions[j].id for j in v] for k, v in banks.items()}, "rows": rows}


def run(args) -> int:
    t0 = time.time()
    doc = compute()
    doc["commit"] = _commit()
    doc["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out} ({doc['seconds']} s)")
    return 0


def _n(x: int) -> str:
    return f"{x:,}"


def render_block(doc: dict, path: str) -> str:
    L = [BEGIN,
         f"_Rendered by `spec_numbers_cfr.py render` from `{path}` (`{doc['spec']}`). Counts by enumeration of "
         f"the public tree; no strategy is played and no value is computed._",
         "",
         f"Instance {doc['instance']}: {doc['types']} types per side, {doc['allocations']} allocations, so "
         f"{doc['offers_per_seat']} offers per seat with walking. Merged bank ({len(doc['banks']['merged'])} of "
         f"{len(doc['banks']['full'])} questions): " + ", ".join(f"`{q}`" for q in doc["banks"]["merged"]) + ".",
         "",
         "| bank | Q | structure | talk paths | A decision points | B decision points | offer points | "
         "leaf pairs | leaf evaluations per CFR pass |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        if not (r["arm"] in ("T2", "T3") or (r["q"] == 2 and r["arm"].startswith("game:"))):
            continue                                                  # T1 and other cells: in the JSON only
        L.append(f"| {r['bank']} | {r['q']} | {r['arm']} | {_n(r['talk_paths'])} | {_n(r['a_points'])} | "
                 f"{_n(r['b_points'])} | {_n(r['offer_points'])} | {_n(r['leaf_pairs'])} | "
                 f"{_n(r['leaf_evaluations'])} |")
    L += ["", "T2 and T3 at every Q, and the four 2 × 2 cells at Q = 2; T1 and the other cells are in the JSON.", END]
    return "\n".join(L)


def replace_block(text: str, block: str) -> str:
    i, j = text.index(BEGIN), text.index(END) + len(END)
    return text[:i] + block + text[j:]


def render(args) -> int:
    with open(args.json) as f:
        doc = json.load(f)
    with open(args.note) as f:
        text = f.read()
    with open(args.note, "w") as f:
        f.write(replace_block(text, render_block(doc, args.json)))
    print(f"rendered into {args.note}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    n = sub.add_parser("render")
    n.add_argument("json")
    n.add_argument("--note", required=True)
    args = p.parse_args(argv)
    return {"run": run, "render": render}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
