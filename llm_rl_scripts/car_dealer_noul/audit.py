"""Blind hand-audit of the verbalizer bank (`DESIGN_NOUL.md` §5, gate 1).

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.audit packet --bank results/noul_bank_<date>
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.audit report --bank results/noul_bank_<date>

``packet`` draws ``N_PER_REGIME`` items per regime (seeded, balanced on the truthful answer),
writes ``audit_packet.jsonl`` (item number, role, the question, the reply — nothing else) and
the sealed ``audit_key.json`` (item number -> bank id, regime, truthful answer). The labeller
reads the packet only and writes ``audit_labels.jsonl`` lines ``{"item": n, "label": "yes"|"no"|"unclear"}``.
``report`` joins labels to the key and prints the gate quantities: contradictions (label is
the opposite of the truth) per regime, unclear counts, and the regex decoder's accuracy on
the audited items and on the whole bank.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .decode import regex_answer
from .verbalize import REGIMES, load_bank

AUDIT_SEED = 20261108
N_PER_REGIME = 60
LABELS = ("yes", "no", "unclear")
# Registered thresholds (design §5 gate 1): contradictions allowed per regime, and the regex
# accuracy above which the oblique regime is not oblique enough.
MAX_CONTRADICTIONS = {"plain": 0, "oblique": 2}
OBLIQUE_REGEX_ACCURACY_MAX = 0.9


def build_packet(rows: List[dict], seed: int = AUDIT_SEED, n_per_regime: int = N_PER_REGIME) -> Dict[str, list]:
    rng = random.Random(seed)
    chosen = []
    for regime in REGIMES:
        for truthful in ("yes", "no"):
            pool = sorted((r for r in rows if r["item"]["regime"] == regime and r["item"]["truthful"] == truthful
                           and not r["empty"] and not r["refusal"]), key=lambda r: r["item"]["id"])
            take = min(n_per_regime // 2, len(pool))
            chosen.extend(rng.sample(pool, take))
    rng.shuffle(chosen)
    packet, key = [], []
    for n, r in enumerate(chosen):
        it = r["item"]
        packet.append({"item": n, "role": it["role"], "question": it["question_text"], "reply": r["text"]})
        key.append({"item": n, "id": it["id"], "regime": it["regime"], "truthful": it["truthful"],
                    "question_id": it["question_id"], "ranking": it["ranking"]})
    return {"packet": packet, "key": key}


def write_packet(built: Dict[str, list], out: Path) -> None:
    with (out / "audit_packet.jsonl").open("w") as f:
        for row in built["packet"]:
            f.write(json.dumps(row) + "\n")
    with (out / "audit_key.json").open("w") as f:
        json.dump(built["key"], f, indent=1)
    with (out / "audit_view.txt").open("w") as f:
        for row in built["packet"]:
            f.write(f"### item {row['item']}  ({row['role']} answering)\n")
            f.write(f"Q: {row['question']}\nA: {row['reply']}\n\n")


def load_labels(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["label"] not in LABELS:
                raise ValueError(f"bad label {d['label']!r} on item {d['item']}")
            out[int(d["item"])] = d["label"]
    return out


def report(rows: List[dict], key: List[dict], labels: Dict[int, str]) -> dict:
    by_id = {r["item"]["id"]: r for r in rows}
    out: dict = {"regimes": {}, "gates": {}}
    for regime in REGIMES:
        ks = [k for k in key if k["regime"] == regime]
        lab = Counter(labels[k["item"]] for k in ks)
        contradictions = [k["item"] for k in ks
                          if labels[k["item"]] in ("yes", "no") and labels[k["item"]] != k["truthful"]]
        regex_audit = [regex_answer(by_id[k["id"]]["text"]) == k["truthful"] for k in ks]
        bank = [r for r in rows if r["item"]["regime"] == regime and not r["empty"] and not r["refusal"]]
        regex_bank = [regex_answer(r["text"]) == r["item"]["truthful"] for r in bank]
        unparsed_bank = sum(regex_answer(r["text"]) is None for r in bank)
        out["regimes"][regime] = {
            "n_audited": len(ks), "labels": dict(lab), "contradictions": contradictions,
            "regex_accuracy_audited": sum(regex_audit) / len(regex_audit) if regex_audit else None,
            "regex_accuracy_bank": sum(regex_bank) / len(regex_bank) if regex_bank else None,
            "regex_unparsed_bank": unparsed_bank, "n_bank": len(bank),
            "empty_or_refusal_bank": sum(r["empty"] or r["refusal"] for r in rows if r["item"]["regime"] == regime),
        }
        out["gates"][regime] = {
            "contradictions_ok": len(contradictions) <= MAX_CONTRADICTIONS[regime],
        }
    ob = out["regimes"]["oblique"]
    out["gates"]["oblique"]["oblique_enough"] = (ob["regex_accuracy_bank"] is not None
                                                and ob["regex_accuracy_bank"] < OBLIQUE_REGEX_ACCURACY_MAX)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("packet", "report"))
    ap.add_argument("--bank", required=True, help="bank directory (bank.jsonl inside)")
    ap.add_argument("--seed", type=int, default=AUDIT_SEED)
    args = ap.parse_args(argv)
    out = Path(args.bank)
    rows = load_bank(out / "bank.jsonl")
    if args.step == "packet":
        built = build_packet(rows, args.seed)
        write_packet(built, out)
        print(f"packet: {len(built['packet'])} items -> {out / 'audit_packet.jsonl'} (view: audit_view.txt); "
              f"key sealed in {out / 'audit_key.json'}")
        return 0
    with (out / "audit_key.json").open() as f:
        key = json.load(f)
    labels = load_labels(out / "audit_labels.jsonl")
    missing = [k["item"] for k in key if k["item"] not in labels]
    if missing:
        ap.error(f"{len(missing)} audited items have no label: {missing[:10]}...")
    rep = report(rows, key, labels)
    with (out / "audit_report.json").open("w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
