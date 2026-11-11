"""Structural numbers cited by the Split Deal v0.3 spec (`DESIGN_SPLIT_V03.md` §3).

Counts only, as for v0.2's ``spec_numbers.py``: types, answer patterns under the 13-question bank,
how often a pair of types has an allocation both accept, split by the pair's outside-option levels,
and where HIGH falls on B0's question path. No decision value is computed, so the bars of §5 are not
previewed.

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.spec_numbers_v03 run \
        --out results/split_v03_spec_20261110/spec_numbers.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.spec_numbers_v03 render \
        results/split_v03_spec_20261110/spec_numbers.json --note llm_rl_scripts/split_deal/DESIGN_SPLIT_V03.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Optional

from .game import Instance, canonical_instances
from .private_d import HIGH, LEVELS, PrivateTables, answer_private
from .sweep_l1 import _commit
from .twosided import Shared

SPEC_VERSION = "split-deal-spec-v0.3"
Q = 3                                                                # the gated question budget (decision 4)
BEGIN, END = "<!-- spec-numbers-v03:begin -->", "<!-- spec-numbers-v03:end -->"


def instance_numbers(inst: Instance) -> dict:
    tab = PrivateTables(inst)
    n = len(tab.types)
    patterns = collections.Counter(tuple((y >> i) & 1 for y in tab.yes) for i in range(n))
    by_levels = collections.Counter()
    deals = collections.Counter()
    for ta in tab.types:
        for tb in tab.types:
            key = "/".join(sorted((ta[1], tb[1])))
            by_levels[key] += 1
            deals[key] += tab.s_star(ta, tb) is not None
    first = Shared(tab).b0_question(tab.full, frozenset())          # B0's opening question: a rule, not a value
    return {"instance": inst.key, "n_types": n, "n_questions": len(tab.questions), "bank_classes": len(patterns),
            "largest_class": max(patterns.values()), "pairs": dict(by_levels), "pairs_with_deal": dict(deals),
            "b0_first_question": tab.questions[first].id, "b0_high_round": high_rounds(tab)}


def b0_high_path(tab: PrivateTables, t, q: int = Q) -> Optional[int]:
    """The round in which B0 asks HIGH when A (type ``t``) answers everything, as in T2; None if not within ``q``.

    A rule property: it uses B0's question rule and A's truthful answers, no policy or offer value."""
    sh, jh = Shared(tab), tab.questions.index(HIGH)
    SA, asked = tab.full, frozenset()
    for r in range(1, q + 1):
        j = sh.b0_question(SA, asked)
        if j is None:
            return None
        if j == jh:
            return r
        SA &= tab.yes[j] if answer_private(tab.inst, t, tab.questions[j]) else ~tab.yes[j]
        asked |= {j}
    return None


def high_rounds(tab: PrivateTables) -> dict:
    """How many of A's types meet HIGH in each round of B0's T2 path ("none" if not within Q)."""
    c = collections.Counter(b0_high_path(tab, t) for t in tab.types)
    return {("none" if r is None else str(r)): c[r] for r in sorted(c, key=lambda r: (r is None, r or 0))}


def compute() -> dict:
    return {"spec": SPEC_VERSION, "levels": {k: str(v) for k, v in LEVELS.items()},
            "instances": [instance_numbers(i) for i in canonical_instances()]}


def run(args) -> int:
    doc = compute()
    doc["commit"] = _commit()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out}")
    return 0


def render_block(doc: dict, path: str) -> str:
    keys = ("lo/lo", "hi/lo", "hi/hi")
    out = [BEGIN,
           f"_Rendered by `spec_numbers_v03.py render` from `{path}` (`{doc['spec']}`). Counts by enumeration; no "
           "decision value is computed._", "",
           "| counts | types | answer patterns (13 questions) | largest unresolved class | pairs with a deal, both lo | "
           "one lo, one hi | both hi | B0's first question | round B0 asks HIGH (T2 path) |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for r in doc["instances"]:
        cells = [r["pairs_with_deal"][k] / r["pairs"][k] for k in keys]
        rounds = r["b0_high_round"]
        when = (next(iter(rounds)) if len(rounds) == 1
                else ", ".join(f"{k}: {v}" for k, v in rounds.items()))   # one round for every A type, or the counts
        out.append(f"| {r['instance']} | {r['n_types']} | {r['bank_classes']} | {r['largest_class']} | "
                   + " | ".join(f"{c:.2f}" for c in cells) + f" | {r['b0_first_question']} | {when} |")
        for k in keys:
            tot[k + ":n"] += r["pairs"][k]
            tot[k + ":d"] += r["pairs_with_deal"][k]
    n = sum(tot[k + ":n"] for k in keys)
    d = sum(tot[k + ":d"] for k in keys)
    split = ", ".join(f"{k} {tot[k + ':d'] / tot[k + ':n']:.2f}" for k in keys)
    out += ["", f"Pooled over instances and level pairs, a mutually acceptable allocation exists for {d / n:.2f} of "
            f"type pairs ({split}; pair-weighted).", END]
    return "\n".join(out)


def replace_v03_block(text: str, block: str) -> str:
    i, j = text.index(BEGIN), text.index(END) + len(END)
    return text[:i] + block + text[j:]


def render(args) -> int:
    with open(args.json) as f:
        doc = json.load(f)
    with open(args.note) as f:
        text = f.read()
    with open(args.note, "w") as f:
        f.write(replace_v03_block(text, render_block(doc, args.json)))
    print(f"updated the v0.3 spec-numbers block in {args.note}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    n = sub.add_parser("render")
    n.add_argument("json")
    n.add_argument("--note", required=True)
    n.set_defaults(func=render)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
