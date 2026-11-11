"""Checks for Split Deal v0.3 tables (``private_d.py``) with the unchanged L1 and L2 programs.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_private_d.py

The naive references here are literal simulations over frozensets with each type's own outside
option, written separately from the bitmask programs.
"""

import math
from fractions import Fraction

import pytest

from .belief import Cell, Tables
from .game import Instance
from .private_d import HIGH, LEVELS, PrivateTables, answer_private
from .twosided import GAME, STRUCTURES, Program, Shared

SMALL = Instance((2, 2, 3))
ALL = dict(STRUCTURES, **{f"game:{a}/{b}": s for (a, b), s in GAME.items()})


def test_one_level_without_high_reproduces_v02():
    for key in ((2, 2, 3), (1, 2, 3), (1, 1, 3)):
        inst = Instance(key)
        old, new = Tables(inst, "easy"), PrivateTables(inst, {"x": Fraction(4)}, high_question=False)
        assert [t[0] for t in new.types] == old.types
        assert new.acc == old.acc and new.yes == old.yes and new.full == old.full
        for v, t in zip(old.types, new.types):
            assert new.gains(t) == old.gains(v)
        sh_old, sh_new = Shared(old), Shared(new)
        for ia in range(0, len(old.types), 3):
            assert Cell(new, new.types[ia], new.full).words(new.full, 3) == Cell(old, old.types[ia], old.full).words(old.full, 3)
            for name in ("T2", "T3"):
                st = STRUCTURES[name]
                assert Program(sh_new, ia, st, 3, Fraction(1, 2)).root() == Program(sh_old, ia, st, 3, Fraction(1, 2)).root()


def test_types_levels_and_high_question():
    tab = PrivateTables(SMALL)
    assert len(tab.types) == 2 * len(SMALL.types())
    j = tab.questions.index(HIGH)
    assert tab.yes[j].bit_count() * 2 == len(tab.types)                  # HIGH splits the prior exactly in half
    for i, t in enumerate(tab.types):
        assert ((tab.yes[j] >> i) & 1) == (t[1] == "hi")
        assert tab.d(t) == LEVELS[t[1]]
        assert max(tab.gains(t)) == 1                                     # everything to this side
    # a high-level responder accepts fewer allocations than the same values at the low level
    for a in range(len(tab.allocs)):
        for i in range(0, len(tab.types), 2):
            lo, hi = (tab.acc[a] >> i) & 1, (tab.acc[a] >> (i + 1)) & 1
            assert hi <= lo


# -- naive references with per-type outside options -------------------------------------

class World:
    def __init__(self, inst):
        self.tab = PrivateTables(inst)
        self.inst, self.types, self.allocs, self.qs = inst, self.tab.types, self.tab.allocs, self.tab.questions

    def g(self, t, share):
        return (Fraction(self.inst.utility(t[0], share)) - self.tab.d(t)) / self.tab.scale(t)

    def accepts(self, t, share):                                         # responder t receives ``share``
        return self.inst.utility(t[0], share) >= self.tab.d(t)

    def final(self, t, K):
        best = Fraction(0)
        for s in self.allocs:
            k = sum(1 for u in K if self.accepts(u, self.inst.rest(s)))
            best = max(best, Fraction(k, len(K)) * self.g(t, s))
        return best

    def best_offer(self, t, K):
        best, best_i, best_k = Fraction(0), None, 0
        for i, s in enumerate(self.allocs):
            k = sum(1 for u in K if self.accepts(u, self.inst.rest(s)))
            num = k * self.g(t, s)
            if num > best or (num == best and num > 0 and k > best_k):
                best, best_i, best_k = num, i, k
        return best_i

    def part(self, K, q, ans):
        return frozenset(u for u in K if answer_private(self.inst, u, q) == ans)


def _naive_words(w, t, K, asked, q):
    best = w.final(t, K)
    if q:
        for i, qu in enumerate(w.qs):
            if i not in asked:
                v = sum((Fraction(len(p), len(K)) * _naive_words(w, t, p, asked | {i}, q - 1)
                         for p in (w.part(K, qu, True), w.part(K, qu, False)) if p), Fraction(0))
                best = max(best, v)
    return best


def test_l1_matches_naive():
    w = World(SMALL)
    K = frozenset(w.types)
    for ia in range(0, len(w.types), 3):
        cell = Cell(w.tab, w.types[ia], w.tab.full)
        for q in (0, 1, 2):
            assert cell.words(w.tab.full, q)[0] == _naive_words(w, w.types[ia], K, frozenset(), q)


def test_spec_numbers_v03_regenerate_and_match_v02_corners():
    import json
    import os

    from .spec_numbers import compute as compute_v02
    from .spec_numbers_v03 import BEGIN, END, compute, render_block
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rel = "results/split_v03_spec_20261110/spec_numbers.json"
    fresh = compute()
    v02 = {r["instance"]: r for r in compute_v02()["instances"]}
    for r in fresh["instances"]:
        old = v02[r["instance"]]
        assert r["n_types"] == 2 * old["n_types"]
        assert r["pairs_with_deal"]["lo/lo"] * old["n_pairs"] == old["pairs_with_deal"]["easy"] * r["pairs"]["lo/lo"]
        assert r["pairs_with_deal"]["hi/hi"] * old["n_pairs"] == old["pairs_with_deal"]["hard"] * r["pairs"]["hi/hi"]
    path = os.path.join(root, *rel.split("/"))
    if os.path.exists(path):
        with open(path) as f:
            stored = json.load(f)
        fresh["commit"] = stored["commit"]
        assert json.loads(json.dumps(fresh)) == stored
        note = os.path.join(root, "llm_rl_scripts", "split_deal", "DESIGN_SPLIT_V03.md")
        with open(note) as f:
            text = f.read()
        assert text[text.index(BEGIN):text.index(END) + len(END)] == render_block(stored, rel)


class NaiveL2:
    def __init__(self, w, ia, st, q, pi):
        self.w, self.me, self.st, self.q, self.pi, self.memo = w, w.types[ia], st, q, pi, {}

    def terminal(self, KB, KA):
        w, me = self.w, self.me
        resp = Fraction(0)
        for b in KB:
            i = w.best_offer(b, KA)
            if i is not None:
                mine = w.inst.rest(w.allocs[i])
                if w.accepts(me, mine):
                    resp += w.g(me, mine)
        return self.pi * w.final(me, KB) + (1 - self.pi) * resp / len(KB)

    def b_question(self, KA, b_asked):
        best, choice = None, None
        for j, qu in enumerate(self.w.qs):
            if j in b_asked:
                continue
            k1 = len(self.w.part(KA, qu, True))
            k0 = len(KA) - k1
            if k1 and k0:
                e = k1 * math.log(k1) + k0 * math.log(k0)
                if best is None or e < best:
                    best, choice = e, j
        return choice

    def v(self, *s):
        if s not in self.memo:
            self.memo[s] = self._v(*s)
        return self.memo[s]

    def _v(self, r, phase, KB, KA, a_asked, b_asked, owed):
        if r == 0:
            return self.terminal(KB, KA)
        w = self.w
        if phase == "A":
            best = self.v(r, "B", KB, KA, a_asked, b_asked, owed)
            if self.st.a_asks:
                answers = self.st.b_answers == "all" or (self.st.b_answers == "reciprocal" and owed)
                for j, qu in enumerate(w.qs):
                    if j in a_asked:
                        continue
                    if answers:
                        val = sum((Fraction(len(p), len(KB)) * self.v(r, "B", p, KA, a_asked | {j}, b_asked, owed)
                                   for p in (w.part(KB, qu, True), w.part(KB, qu, False)) if p), Fraction(0))
                    else:
                        val = self.v(r, "B", KB, KA, a_asked | {j}, b_asked, owed)
                    best = max(best, val)
            return best
        j = self.b_question(KA, b_asked) if self.st.b_asks else None
        if j is None:
            return self.v(r - 1, "A", KB, KA, a_asked, b_asked, owed)
        told = w.part(KA, w.qs[j], answer_private(w.inst, self.me, w.qs[j]))
        v_ans = self.v(r - 1, "A", KB, told, a_asked, b_asked | {j}, True)
        v_dec = self.v(r - 1, "A", KB, KA, a_asked, b_asked | {j}, False)
        return {"all": v_ans, "none": v_dec, "choose": max(v_ans, v_dec)}[self.st.a_answers]

    def root(self):
        K = frozenset(self.w.types)
        return self.v(self.q, "A", K, K, frozenset(), frozenset(), True)


def test_b0_high_round_matches_naive():
    from .spec_numbers_v03 import Q, high_rounds
    for key in ((2, 2, 3), (1, 2, 4), (1, 1, 3)):
        w = World(Instance(key))
        naive = NaiveL2(w, 0, STRUCTURES["T2"], Q, Fraction(1, 2))
        jh = w.qs.index(HIGH)
        expect = {}
        for t in w.types:
            KA, asked, when = frozenset(w.types), frozenset(), "none"
            for r in range(1, Q + 1):
                j = naive.b_question(KA, asked)
                if j is None:
                    break
                if j == jh:
                    when = str(r)
                    break
                KA, asked = w.part(KA, w.qs[j], answer_private(w.inst, t, w.qs[j])), asked | {j}
            expect[when] = expect.get(when, 0) + 1
        assert high_rounds(w.tab) == expect, key


@pytest.mark.parametrize("name", list(ALL))
def test_l2_matches_naive(name):
    w = World(SMALL)
    sh = Shared(w.tab)
    st = ALL[name]
    for ia in range(0, len(w.types), 4):
        for q in (1, 2):
            assert Program(sh, ia, st, q, Fraction(1, 2)).root() == NaiveL2(w, ia, st, q, Fraction(1, 2)).root()


def test_l2_rollouts_realize_values():
    tab = PrivateTables(SMALL)
    sh = Shared(tab)
    n, pi = len(tab.types), Fraction(1, 2)
    for ia in range(0, n, 3):
        me = sh.seat[ia]
        for name, st in ALL.items():
            prog = Program(sh, ia, st, 3, pi)
            total = Fraction(0)
            for b in range(n):
                o = prog.rollout(b)
                total += pi * (me.gain[o["a_deal"]] if o["a_deal"] is not None else 0) + \
                    (1 - pi) * (me.gain[o["b_deal"]] if o["b_deal"] is not None else 0)
            assert total / n == prog.root(), name
