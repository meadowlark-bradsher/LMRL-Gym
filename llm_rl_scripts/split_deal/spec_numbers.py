"""Structural numbers cited by the Split Deal design note (`DESIGN_SPLIT.md` §2-§3).

Counts only: how many types, allocations and distinguishable answer patterns each instance has,
and how often a pair of types has integrative potential or any mutually acceptable allocation.
Deliberately no decision value is computed here: the words-value bar of the note (§7) is fixed
before the L1 sweep runs, and this module must not preview it.

    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.spec_numbers run \
        --out results/split_spec_20261110/spec_numbers.json
    PYTHONPATH=. python3 -m llm_rl_scripts.split_deal.spec_numbers render \
        results/split_spec_20261110/spec_numbers.json --note llm_rl_scripts/split_deal/DESIGN_SPLIT.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

from .game import REGIMES, TOTAL, Instance, bank, canonical_instances, outside_option, signature

SPEC_VERSION = "split-deal-spec-v0.2"
BEGIN, END = "<!-- spec-numbers:begin -->", "<!-- spec-numbers:end -->"


def pair_constraints(va, vb) -> bool:
    """The value-pair constraints of the sensitivity prior (`DESIGN_SPLIT.md` §2): no kind is worthless
    to both sides, and some kind is valued by both."""
    return (not any(x == 0 and y == 0 for x, y in zip(va, vb))) and any(x > 0 and y > 0 for x, y in zip(va, vb))


def instance_numbers(inst: Instance) -> dict:
    types, allocs, qs = inst.types(), inst.allocations(), bank(len(inst.counts))
    classes = collections.Counter(signature(inst, v, qs) for v in types)
    u_own = {v: [inst.utility(v, a) for a in allocs] for v in types}                 # as the focal side
    u_rest = {v: [inst.utility(v, inst.rest(a)) for a in allocs] for v in types}     # as the counterpart
    integrative, joint_sum, constrained, conflict_free = 0, 0, 0, 0
    with_deal = {g: 0 for g in REGIMES}
    for va in types:
        for vb in types:
            joint = [x + y for x, y in zip(u_own[va], u_rest[vb])]
            joint_sum += max(joint)
            integrative += max(joint) > TOTAL
            conflict_free += max(joint) == 2 * TOTAL
            constrained += pair_constraints(va, vb)
            for g in REGIMES:
                d = outside_option(g)
                with_deal[g] += any(x >= d and y >= d for x, y in zip(u_own[va], u_rest[vb]))
    return {"instance": inst.key, "counts": list(inst.counts), "n_types": len(types), "n_allocations": len(allocs),
            "n_questions": len(qs), "bank_classes": len(classes), "largest_class": max(classes.values()),
            "n_pairs": len(types) ** 2, "pairs_integrative": integrative, "max_joint_sum": joint_sum,
            "pairs_with_deal": with_deal, "pairs_constrained": constrained, "pairs_conflict_free": conflict_free}


def compute() -> dict:
    rows = [instance_numbers(inst) for inst in canonical_instances()]
    return {"spec": SPEC_VERSION, "total": TOTAL, "regimes": {g: str(r) for g, r in REGIMES.items()},
            "instances": rows}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(args) -> int:
    doc = compute()
    doc["commit"] = _commit()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out} ({len(doc['instances'])} instances)")
    return 0


# -- note block ---------------------------------------------------------------------

def render_block(doc: dict, path: str) -> str:
    rows = doc["instances"]
    out = [BEGIN,
           f"_Rendered by `spec_numbers.py render` from `{path}` (`{doc['spec']}`). Counts by enumeration; "
           "no decision value is computed._", "",
           "| counts | types | allocations | answer patterns (12 questions) | largest unresolved class | "
           "type pairs | integrative pairs | mean best joint value | pairs with a deal, easy | pairs with a deal, hard | "
           "pairs meeting the value-pair constraints | conflict-free pairs |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for r in rows:
        n = r["n_pairs"]
        out.append(f"| {r['instance']} | {r['n_types']} | {r['n_allocations']} | {r['bank_classes']} | "
                   f"{r['largest_class']} | {n} | {r['pairs_integrative'] / n:.2f} | {r['max_joint_sum'] / n:.2f} | "
                   f"{r['pairs_with_deal']['easy'] / n:.2f} | {r['pairs_with_deal']['hard'] / n:.2f} | "
                   f"{r['pairs_constrained'] / n:.2f} | {r['pairs_conflict_free'] / n:.2f} |")
        tot.update({"n": n, "int": r["pairs_integrative"], "joint": r["max_joint_sum"],
                    "easy": r["pairs_with_deal"]["easy"], "hard": r["pairs_with_deal"]["hard"],
                    "con": r["pairs_constrained"], "free": r["pairs_conflict_free"]})
    n = tot["n"]
    types = [r["n_types"] for r in rows]
    allocs = [r["n_allocations"] for r in rows]
    out += ["",
            f"Across the {len(rows)} instances: {min(types)}–{max(types)} types and {min(allocs)}–{max(allocs)} "
            f"allocations per instance; {n} type pairs in all, of which {tot['int'] / n:.2f} are integrative "
            f"(best joint value above {doc['total']}), mean best joint value {tot['joint'] / n:.2f}; a mutually "
            f"acceptable allocation exists for {tot['easy'] / n:.2f} of pairs in easy and {tot['hard'] / n:.2f} "
            f"in hard; {tot['con'] / n:.2f} of pairs meet the value-pair constraints of the sensitivity prior, "
            f"and {tot['free'] / n:.2f} are conflict-free (both sides can have their full {doc['total']}). "
            "All shares pair-weighted.", END]
    return "\n".join(out)


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
    print(f"updated the spec-numbers block in {args.note}")
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
