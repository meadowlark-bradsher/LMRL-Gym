"""FLYW-3: remediation-class maps for the 20q secret pool (Flywheel experiment 1).

    # build every map + the index (deterministic, offline, a few seconds):
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.class_maps \
        --out llm_rl_scripts/twenty_questions/jev/results/flyw1_class_maps_20261118

    # mechanics self-test:
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.class_maps --self-test

A *class map* assigns each of the 158 real objects to one class. A class stands in
for a remediation: two secrets in the same class need the same action, so a policy
aiming at the action need not tell them apart (``FLYW1_DECISION_PREREG.md``).

The family varies three things:

* **number of classes** ``K`` in {2, 4, 8, 16};
* **size profile**: ``bal`` (equal sizes) or ``skew`` (geometric sizes, ratio 1/2,
  so the largest class holds about half the pool);
* **alignment**: how well class boundaries follow the questions that are naturally
  informative about *which object*.

Plus one natural map, ``cat17``: the benchmark's 17 categories, which the bank's
category questions test directly.

**Construction.** All construction and measurement use the noise-free binarized
answer table: the 44 attribute cells of ``matrix_jev.json`` thresholded at 0.5 (the
matrix oracle's and the hard kill rule's threshold), plus the 17 category questions.

* The *identity tree* splits the pool greedily on the most balanced question (largest
  split entropy; ties by question id) until no question splits a node. Its leaf order
  (yes branch first) keeps every subtree contiguous.
* An **aligned** map (``align``) cuts that leaf order into contiguous blocks of the
  profile's sizes, so each class is close to a union of subtrees.
* A **mixed** map (``mix05``, ``mix15``) takes the aligned map and swaps the classes
  of random pairs of objects in different classes until the stated share of objects
  has moved. Sizes are unchanged.
* A **random** map (``rand``) deals a random permutation into the same sizes: class
  boundaries cut across the natural splits.

Mixed and random maps come in two seeds each, so 7 maps per (K, profile) and
4 x 2 x 7 + 1 = **57 maps**.

**Measures** (per map, noise-free, uniform over objects, in questions; the final
guess or action is not counted, since every arm spends exactly one):

* ``id_depth``: mean leaf depth of the identity tree (the same for every map).
* ``pure_depth``: mean depth at which an object's node on the identity tree first
  holds a single class. This is what arm B (identity questions, class stopping)
  needs, noise-free.
* ``alignment`` = 1 - pure_depth / id_depth: the share of identity questions that
  stopping at the class saves. **This is the alignment index.** It is 0 when classes
  are only resolved at the leaves and approaches 1 when the first questions already
  separate the classes.
* ``class_depth``: mean depth of a greedy tree that maximises information about the
  class label (arm C, noise-free).
* ``ec2_depth``: mean depth of the greedy EC^2 tree (Golovin, Krause & Ray 2010),
  uniform prior, edges between objects in different classes (arm D, noise-free).
* ``best_q_nmi``: the best single question's mutual information with the class
  label, over H(class).
* ``unresolvable``: objects whose identity-tree leaf (a group with identical answers
  to every question) mixes classes. No question can separate them; only a by-name
  guess or an action can.

The trees are design-time calibration on the binarized table, not the arms: the
arms run P0a with the graded likelihood (``FLYW1_DECISION_PREREG.md``).

Each map is written as ``maps/<map_id>.json`` with its assignment, sizes, measures
and provenance (matrix hash, generator version, seed). ``index.json`` lists them all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .bed import ATTRIBUTE_BANK, load_hypotheses

GENERATOR_VERSION = 1
MATRIX_FILE = os.path.join(os.path.dirname(__file__), "matrix_jev.json")
KS = (2, 4, 8, 16)
PROFILES = ("bal", "skew")
MIXES = (("mix05", 0.05), ("mix15", 0.15))
SEEDS = (1, 2)

Table = Dict[str, Dict[str, bool]]   # object -> question id -> noise-free yes


# --------------------------------------------------------------------------- #
# Answer table                                                                 #
# --------------------------------------------------------------------------- #

def load_table(matrix_path: str = MATRIX_FILE) -> Tuple[List[str], List[str], Table, dict]:
    """Objects (benchmark order), question ids, the binarized table and provenance."""
    with open(matrix_path) as f:
        data = json.load(f)
    m = data["matrix"]
    hyps = load_hypotheses()
    objs = [h.name for h in hyps]
    cats = sorted({h.category for h in hyps})
    qids = ["attribute:" + aid for aid, _ in ATTRIBUTE_BANK] + ["category:" + c for c in cats]
    table: Table = {}
    for h in hyps:
        row = {"attribute:" + aid: m[aid][h.name] >= 0.5 for aid, _ in ATTRIBUTE_BANK}
        row.update({"category:" + c: h.category == c for c in cats})
        table[h.name] = row
    with open(matrix_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:12]
    prov = {"matrix_file": os.path.basename(matrix_path), "matrix_sha256_12": sha,
            "attribute_hash": data.get("attribute_hash"), "model": data.get("model"),
            "n_objects": len(objs), "n_questions": len(qids), "threshold": 0.5}
    return objs, qids, table, prov


def _h(counts: Sequence[int]) -> float:
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


# --------------------------------------------------------------------------- #
# Trees                                                                        #
# --------------------------------------------------------------------------- #

def identity_tree(objs: Sequence[str], qids: Sequence[str], table: Table
                  ) -> Tuple[List[str], Dict[str, List[Tuple[str, ...]]]]:
    """Greedy most-balanced splits. Returns the leaf order and, per object, the
    node path: the sorted member tuple of every node from root to its leaf."""
    order: List[str] = []
    paths: Dict[str, List[Tuple[str, ...]]] = {o: [] for o in objs}

    def grow(node: List[str]) -> None:
        for o in node:
            paths[o].append(tuple(node))
        best, best_h = None, 0.0
        for q in qids:
            y = sum(table[o][q] for o in node)
            hh = _h((y, len(node) - y))
            if hh > best_h + 1e-12:
                best, best_h = q, hh
        if best is None:            # no question splits: an identical-answer leaf
            order.extend(node)
            return
        grow([o for o in node if table[o][best]])
        grow([o for o in node if not table[o][best]])

    grow(list(objs))
    return order, paths


def class_greedy_depth(objs: Sequence[str], qids: Sequence[str], table: Table,
                       cls: Dict[str, int], score: str) -> Tuple[float, int]:
    """Mean depth of a greedy tree that stops at class-pure nodes. ``score`` is
    ``"ig"`` (information about the class label, arm C) or ``"ec2"`` (weight of
    cross-class edges cut, arm D). Returns (mean depth, objects left in impure
    nodes no question splits)."""
    depth: Dict[str, int] = {}
    stuck = 0

    def edges(node: Sequence[str]) -> float:
        c = Counter(cls[o] for o in node)
        n = len(node)
        return (n * n - sum(v * v for v in c.values())) / 2.0

    def grow(node: List[str], d: int) -> None:
        nonlocal stuck
        if len({cls[o] for o in node}) == 1:
            for o in node:
                depth[o] = d
            return
        n = len(node)
        hc = _h(list(Counter(cls[o] for o in node).values()))
        e0 = edges(node)
        best, best_s = None, 1e-12
        for q in qids:
            yes = [o for o in node if table[o][q]]
            if not yes or len(yes) == n:
                continue
            no = [o for o in node if not table[o][q]]
            if score == "ig":
                s = hc - (len(yes) / n) * _h(list(Counter(cls[o] for o in yes).values())) \
                       - (len(no) / n) * _h(list(Counter(cls[o] for o in no).values()))
            else:   # EC^2: expected cut weight = e0 - sum_a P(a) * edges kept under a
                s = e0 - (len(yes) / n) * edges(yes) - (len(no) / n) * edges(no)
            if s > best_s + 1e-12:
                best, best_s = q, s
        if best is None:
            # No question gains on the class objective. If some question still splits
            # the node, take the most balanced one so the tree can continue (a greedy
            # class score can be zero on every split: the myopia EC^2 was built for).
            for q in qids:
                y = sum(table[o][q] for o in node)
                if 0 < y < n and (best is None or _h((y, n - y)) > best_s):
                    best, best_s = q, _h((y, n - y))
        if best is None:
            for o in node:
                depth[o] = d
            stuck += len(node)
            return
        grow([o for o in node if table[o][best]], d + 1)
        grow([o for o in node if not table[o][best]], d + 1)

    grow(list(objs), 0)
    return sum(depth.values()) / len(depth), stuck


# --------------------------------------------------------------------------- #
# Maps                                                                         #
# --------------------------------------------------------------------------- #

def profile_sizes(n: int, k: int, profile: str) -> List[int]:
    """Class sizes summing to ``n``: equal, or geometric with ratio 1/2 (largest
    first; every class keeps at least 2 objects)."""
    if profile == "bal":
        w = [1.0] * k
    elif profile == "skew":
        w = [0.5 ** i for i in range(k)]
    else:
        raise ValueError(profile)
    raw = [max(2, int(round(n * x / sum(w)))) for x in w]
    i = 0
    while sum(raw) != n:            # repair rounding on the largest classes
        j = i % k
        if sum(raw) > n and raw[j] > 2:
            raw[j] -= 1
        elif sum(raw) < n:
            raw[j] += 1
        i += 1
    return raw


def blocks(order: Sequence[str], sizes: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    i = 0
    for c, s in enumerate(sizes):
        for o in order[i:i + s]:
            out[o] = c
        i += s
    return out


def mix(aligned: Dict[str, int], objs: Sequence[str], share: float, seed: int) -> Dict[str, int]:
    """Swap classes of random cross-class pairs until ``share`` of objects moved."""
    rng = random.Random(seed)
    cls = dict(aligned)
    target = int(round(share * len(objs)))
    moved: set = set()
    pool = list(objs)
    while len(moved) < target:
        a, b = rng.sample(pool, 2)
        if a in moved or b in moved or cls[a] == cls[b]:
            continue
        cls[a], cls[b] = cls[b], cls[a]
        moved.update((a, b))
    return cls


def shuffled(objs: Sequence[str], sizes: Sequence[int], seed: int) -> Dict[str, int]:
    perm = list(objs)
    random.Random(seed).shuffle(perm)
    return blocks(perm, sizes)


def measures(objs: Sequence[str], qids: Sequence[str], table: Table,
             paths: Dict[str, List[Tuple[str, ...]]], cls: Dict[str, int]) -> dict:
    id_depth = sum(len(paths[o]) - 1 for o in objs) / len(objs)
    pure = []
    unresolvable = 0
    for o in objs:
        d = next((i for i, node in enumerate(paths[o])
                  if len({cls[x] for x in node}) == 1), None)
        if d is None:               # the leaf itself mixes classes
            unresolvable += 1
            d = len(paths[o]) - 1
        pure.append(d)
    pure_depth = sum(pure) / len(pure)
    class_depth, class_stuck = class_greedy_depth(objs, qids, table, cls, "ig")
    ec2_depth, ec2_stuck = class_greedy_depth(objs, qids, table, cls, "ec2")
    labels = [cls[o] for o in objs]
    h_cls = _h(list(Counter(labels).values()))
    best_nmi = 0.0
    for q in qids:
        yes = [cls[o] for o in objs if table[o][q]]
        no = [cls[o] for o in objs if not table[o][q]]
        n = len(objs)
        mi = h_cls - len(yes) / n * _h(list(Counter(yes).values())) \
                   - len(no) / n * _h(list(Counter(no).values()))
        best_nmi = max(best_nmi, mi / h_cls if h_cls else 0.0)
    return {"id_depth": round(id_depth, 4), "pure_depth": round(pure_depth, 4),
            "alignment": round(1.0 - pure_depth / id_depth, 4),
            "class_depth": round(class_depth, 4), "ec2_depth": round(ec2_depth, 4),
            "best_q_nmi": round(best_nmi, 4), "unresolvable": unresolvable,
            "class_tree_stuck": class_stuck, "ec2_tree_stuck": ec2_stuck,
            "h_class_bits": round(h_cls, 4)}


def build_all(matrix_path: str = MATRIX_FILE) -> Tuple[List[dict], dict]:
    objs, qids, table, prov = load_table(matrix_path)
    order, paths = identity_tree(objs, qids, table)
    hyps = load_hypotheses()
    maps: List[dict] = []

    def add(map_id: str, family: str, k: int, profile: str, align: str,
            seed: Optional[int], cls: Dict[str, int], class_names: Optional[List[str]] = None):
        sizes = [sum(1 for o in objs if cls[o] == c) for c in range(k)]
        assert sum(sizes) == len(objs) and min(sizes) >= 1, (map_id, sizes)
        body = {"map_id": map_id, "version": GENERATOR_VERSION, "family": family, "k": k,
                "profile": profile, "alignment_construction": align, "seed": seed,
                "sizes": sizes, "classes": {o: cls[o] for o in objs}}
        if class_names:
            body["class_names"] = class_names
        body["measures"] = measures(objs, qids, table, paths, cls)
        digest = hashlib.sha256(json.dumps(body["classes"], sort_keys=True).encode()).hexdigest()
        body["assignment_sha256_12"] = digest[:12]
        body["provenance"] = dict(prov, generator="class_maps.py",
                                  generator_version=GENERATOR_VERSION)
        maps.append(body)

    cats = sorted({h.category for h in hyps})
    cat_cls = {h.name: cats.index(h.category) for h in hyps}
    add("cat17", "natural", len(cats), "natural", "category", None, cat_cls, cats)
    for k in KS:
        for prof in PROFILES:
            sizes = profile_sizes(len(objs), k, prof)
            aligned = blocks(order, sizes)
            add("k{}_{}_align".format(k, prof), "synthetic", k, prof, "align", None, aligned)
            for tag, share in MIXES:
                for s in SEEDS:
                    add("k{}_{}_{}_s{}".format(k, prof, tag, s), "synthetic", k, prof, tag,
                        s, mix(aligned, objs, share, seed=1000 * k + 10 * s + int(100 * share)))
            for s in SEEDS:
                add("k{}_{}_rand_s{}".format(k, prof, s), "synthetic", k, prof, "rand", s,
                    shuffled(objs, sizes, seed=1000 * k + s))
    return maps, prov


def write_all(out: str, maps: Sequence[dict], prov: dict) -> None:
    os.makedirs(os.path.join(out, "maps"), exist_ok=True)
    rows = []
    for mp in maps:
        with open(os.path.join(out, "maps", mp["map_id"] + ".json"), "w") as f:
            json.dump(mp, f, indent=1, sort_keys=False)
            f.write("\n")
        rows.append({"map_id": mp["map_id"], "k": mp["k"], "profile": mp["profile"],
                     "alignment_construction": mp["alignment_construction"],
                     "seed": mp["seed"], "sizes": mp["sizes"],
                     "assignment_sha256_12": mp["assignment_sha256_12"], **mp["measures"]})
    index = {"generator": "class_maps.py", "generator_version": GENERATOR_VERSION,
             "provenance": prov, "n_maps": len(rows), "maps": rows}
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")


def load_map(path: str) -> Dict[str, int]:
    """What the arms load: object name -> class id."""
    with open(path) as f:
        return json.load(f)["classes"]


def self_test() -> None:
    objs, qids, table, _ = load_table()
    assert len(objs) == 158 and len(qids) == 44 + 17
    order, paths = identity_tree(objs, qids, table)
    assert sorted(order) == sorted(objs)
    for k in KS:
        for p in PROFILES:
            s = profile_sizes(158, k, p)
            assert sum(s) == 158 and len(s) == k and min(s) >= 2, (k, p, s)
    sizes = profile_sizes(158, 4, "bal")
    al = blocks(order, sizes)
    mx = mix(al, objs, 0.5, seed=3)
    assert Counter(mx.values()) == Counter(al.values())          # sizes preserved
    assert sum(al[o] != mx[o] for o in objs) in (79, 80)          # half moved (pairs)
    m_al = measures(objs, qids, table, paths, al)
    m_rd = measures(objs, qids, table, paths, shuffled(objs, sizes, seed=1))
    assert m_al["alignment"] > m_rd["alignment"], (m_al, m_rd)
    # A one-class map is pure at the root.
    one = {o: 0 for o in objs}
    m_one = measures(objs, qids, table, paths, one)
    assert m_one["pure_depth"] == 0 and m_one["alignment"] == 1.0
    # Identity map: purity only at the leaves, so alignment is 0 up to identical leaves.
    ident = {o: i for i, o in enumerate(objs)}
    assert measures(objs, qids, table, paths, ident)["alignment"] == 0.0
    # Determinism.
    a, _ = build_all()
    b, _ = build_all()
    assert [x["assignment_sha256_12"] for x in a] == [x["assignment_sha256_12"] for x in b]
    assert len(a) == 57
    print("class_maps self-test PASS (57 maps; aligned {:.3f} > random {:.3f} at K=4 bal)"
          .format(m_al["alignment"], m_rd["alignment"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.out:
        ap.error("--out is required")
    maps, prov = build_all()
    write_all(a.out, maps, prov)
    print("wrote {} maps to {}".format(len(maps), a.out))
    for mp in maps:
        m = mp["measures"]
        print("{:24s} sizes={:<40s} align={:.3f} pure={:.2f} class={:.2f} ec2={:.2f} "
              "nmi={:.3f} unres={}".format(mp["map_id"], str(mp["sizes"])[:40], m["alignment"],
                                           m["pure_depth"], m["class_depth"], m["ec2_depth"],
                                           m["best_q_nmi"], m["unresolvable"]))


if __name__ == "__main__":
    main()
