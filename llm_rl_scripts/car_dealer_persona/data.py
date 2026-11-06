"""Loading, de-duplication and frozen splits for the released Car Dealer dataset.

The files are not committed (37 MB). ``fetch`` downloads them and ``load`` refuses to
proceed unless their sha256 matches the pins below, so every number downstream is tied
to the exact bytes named in ``PREREGISTRATION_BASELINE2023.md``.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.data --fetch

A record's id is the sha1 of its ``original_conversation``. Exact duplicates collapse to
one record; a train record whose id also occurs in eval is dropped from train. The DEV /
GOLD / RESERVE splits are a seeded draw from the distinct train records; everything
else, plus all of eval, is CONFIRMATORY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request
from typing import Dict, List

BASE_URL = "https://rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset/car-dealer/"
SHA256 = {
    "train.json": "20c0ab1376d7867343bc330023d295a8a55aa41c9ef7107b6a82ded8e3d7e743",
    "eval.json": "bf244da9144529ff02ac77c9ee4b8abec60f20f0ad2ae748890c1d146d014eab",
}
DEFAULT_DIR = os.path.join("data", "car_dealer_persona")

SPLIT_SEED = 20261106
N_DEV, N_GOLD, N_RESERVE = 200, 80, 80

BUYER = {
    "The buyer will never buy from the seller unless given a discount.": "B-discount",
    "The buyer will only buy the car if it meets all of their specifications.": "B-spec",
    "The buyer is extremely impatient and strongly dislikes unnecessary information "
    "about discounts and features.": "B-impatient",
}
SELLER = {
    "The agent never talks about the features of the car and loves to give discounts.": "S-discount",
    "The agent never gives discounts and loves to talk a lot about features of the car.": "S-features",
    "The agent never gives discounts, never talks about the features of the car, and is "
    "very concise.": "S-concise",
}
COMPATIBLE = {"B-discount": "S-discount", "B-spec": "S-features", "B-impatient": "S-concise"}


def record_id(rec: dict) -> str:
    return hashlib.sha1(rec["original_conversation"].encode("utf-8")).hexdigest()


def budget_dollars(rec: dict) -> int:
    """'30k' -> 30000."""
    return int(str(rec["buyer_info"]["budget"]).lower().rstrip("k")) * 1000


def persona(rec: dict) -> Dict[str, str]:
    info = rec["buyer_info"]
    return {"buyer": BUYER[info["background_buyer"]], "seller": SELLER[info["background_seller"]]}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(data_dir: str = DEFAULT_DIR) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for name, want in SHA256.items():
        path = os.path.join(data_dir, name)
        if not (os.path.exists(path) and _sha256(path) == want):
            urllib.request.urlretrieve(BASE_URL + name, path)
        got = _sha256(path)
        if got != want:
            raise RuntimeError(f"{name}: sha256 {got} does not match the registered {want}")
        print(f"{name}: ok")


def _load_file(data_dir: str, name: str) -> List[dict]:
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run this module with --fetch")
    got = _sha256(path)
    if got != SHA256[name]:
        raise RuntimeError(f"{name}: sha256 {got} does not match the registered {SHA256[name]}")
    with open(path) as f:
        return json.load(f)


def load(data_dir: str = DEFAULT_DIR) -> Dict[str, List[dict]]:
    """Distinct records per source file, each tagged with ``id`` and ``source``."""
    out: Dict[str, List[dict]] = {}
    eval_ids = set()
    for name in ("eval.json", "train.json"):          # eval first: it wins ties
        seen, recs = set(), []
        for rec in _load_file(data_dir, name):
            rid = record_id(rec)
            if rid in seen or (name == "train.json" and rid in eval_ids):
                continue
            seen.add(rid)
            recs.append({**rec, "id": rid, "source": name[:-5]})
        if name == "eval.json":
            eval_ids = seen
        out[name[:-5]] = recs
    return out


def splits(data_dir: str = DEFAULT_DIR) -> Dict[str, List[dict]]:
    """DEV / GOLD / RESERVE / CONFIRMATORY, deterministic in ``SPLIT_SEED``."""
    d = load(data_dir)
    train = sorted(d["train"], key=lambda r: r["id"])
    random.Random(SPLIT_SEED).shuffle(train)
    a, b, c = N_DEV, N_DEV + N_GOLD, N_DEV + N_GOLD + N_RESERVE
    return {"dev": train[:a], "gold": train[a:b], "reserve": train[b:c],
            "confirmatory": train[c:] + d["eval"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--data-dir", default=DEFAULT_DIR)
    args = p.parse_args(argv)
    if args.fetch:
        fetch(args.data_dir)
    s = splits(args.data_dir)
    print({k: len(v) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
