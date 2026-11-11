"""Checks for the CFR check's tree counts (``spec_numbers_cfr.py``): the merged bank, the counts against a naive
enumeration of explicit histories, and regeneration of the stored counts, the note's table and the figures its
prose cites.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_spec_numbers_cfr.py
"""

import json
import os

import pytest

from . import spec_numbers_cfr as sc
from .game import Instance
from .private_d import HIGH, PrivateTables, answer_private

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = "results/split_cfr_spec_20261111/spec_numbers.json"
NOTE = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "DESIGN_SPLIT_CFR.md")


@pytest.fixture(scope="module")
def tab():
    return PrivateTables(Instance(sc.INSTANCE))


def test_merged_bank(tab):
    keep = sc.merged_questions(tab)
    assert keep == sorted(keep) and tab.questions.index(HIGH) in keep
    parts = [min(tab.yes[j], tab.full & ~tab.yes[j]) for j in keep]
    assert len(set(parts)) == len(parts)                                  # one question per partition
    for j, y in enumerate(tab.yes):                                       # every question is represented
        assert min(y, tab.full & ~y) in parts


def _naive(tab, q, st, questions):
    """Explicit histories over explicit type lists: no masks, no memo."""
    types = tab.types
    inst = tab.inst
    qs = [tab.questions[j] for j in questions]
    paths = a_pts = b_pts = offer_pts = pairs = 0

    def walk(r, phase, SA, SB, declined, last_b):
        nonlocal paths, a_pts, b_pts, offer_pts, pairs
        if r == q:
            paths += 1
            offer_pts += len(SA) + len(SB)
            pairs += len(SA) * len(SB)
            return
        if phase == "A":
            if st.a_asks:
                a_pts += len(SA)
            walk(r, "B", SA, SB, declined, last_b)
            b_will = st.b_answers == "all" or (st.b_answers == "reciprocal" and last_b != "declined")
            if st.a_asks and b_will:
                for x in qs:
                    yes = [t for t in SB if answer_private(inst, t, x)]
                    no = [t for t in SB if not answer_private(inst, t, x)]
                    if yes and no:
                        walk(r, "B", SA, yes, declined, last_b)
                        walk(r, "B", SA, no, declined, last_b)
        else:
            if st.b_asks:
                b_pts += len(SB)
            walk(r + 1, "A", SA, SB, declined, last_b)
            if st.b_asks:
                for x in qs:
                    yes = [t for t in SA if answer_private(inst, t, x)]
                    no = [t for t in SA if not answer_private(inst, t, x)]
                    if x.id in declined or not yes or not no:
                        continue
                    if st.a_answers == "choose":
                        a_pts += len(SA)
                    if st.a_answers != "none":
                        walk(r + 1, "A", yes, SB, declined, "answered")
                        walk(r + 1, "A", no, SB, declined, "answered")
                    if st.a_answers != "all":
                        walk(r + 1, "A", SA, SB, declined | {x.id}, "declined")

    walk(0, "A", list(types), list(types), frozenset(), "none")
    return {"talk_paths": paths, "a_points": a_pts, "b_points": b_pts, "offer_points": offer_pts,
            "leaf_pairs": pairs, "leaf_evaluations": 2 * (len(tab.allocs) + 1) * pairs}


@pytest.mark.parametrize("q,bank,arm", [(1, "full", "T3"), (1, "full", "game:none/all"), (2, "merged", "T3"),
                                        (2, "merged", "T2"), (2, "merged", "game:all/none")])
def test_counts_match_naive_enumeration(tab, q, bank, arm):
    questions = list(range(len(tab.questions))) if bank == "full" else sc.merged_questions(tab)
    assert sc.tree_counts(tab, q, sc.ARMS[arm], questions) == _naive(tab, q, sc.ARMS[arm], questions)


def _stored():
    path = os.path.join(ROOT, *REL.split("/"))
    if not os.path.exists(path):
        pytest.skip("counts missing")
    with open(path) as f:
        return json.load(f)


def test_regenerate_counts():
    stored = _stored()
    fresh = json.loads(json.dumps(sc.compute()))
    assert {k: v for k, v in stored.items() if k not in ("commit", "seconds")} == fresh


def test_note_table_and_prose():
    stored = _stored()
    with open(NOTE) as f:
        text = f.read()
    assert sc.render_block(stored, REL) in text
    row = {(r["bank"], r["q"], r["arm"]): r["leaf_evaluations"] for r in stored["rows"]}
    cited = {"7.3 × 10¹⁰": row["full", 3, "T3"], "2.3 × 10¹⁰": row["merged", 3, "T3"],
             "1.1 × 10⁹": row["full", 2, "T3"], "4.6 × 10⁸": row["merged", 2, "T3"]}
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    for s, v in cited.items():
        mant, exp = s.split(" × 10")
        assert s in text and f"{v / 10 ** int(exp.translate(sup)):.1f}" == mant, s
    assert "by a factor of about 2.4 at Q = 2" in text
    assert f"{row['full', 2, 'T3'] / row['merged', 2, 'T3']:.1f}" == "2.4"
    assert f"13 questions induce only {len(stored['banks']['merged'])} distinct partitions of the " \
           f"{stored['types']} types" in text
