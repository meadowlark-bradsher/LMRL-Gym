"""Checks for the level-k readers (``readers.py``): B2, B2-d, A1' and A3 on the v0.3 game.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_readers.py

The naive references are literal recursions over frozensets: A1's moves come from ``NaiveL2`` (the
frozenset reference of ``test_private_d.py``), and B2's reading, its off-path rule and A3 are written
again here without the bitmask programs. They check values, not only rollouts. The frozenset recursion is slow, so the deeper naive checks
(Q = 2) run on TINY, a small non-canonical instance (counts 2-3-4: 5 value vectors, 10 v0.3 types,
60 allocations); 2-2-3 gets the naive check at Q = 1 and the bitmask-only checks at Q = 3.
"""

from fractions import Fraction

import pytest

from .game import Instance
from .readers import (A1Model, A3, B2, rollout_a1_b0, rollout_a1_b2, rollout_a1_b2d, rollout_a3_b2)
from .test_private_d import NaiveL2, World
from .twosided import GAME, STRUCTURES, Shared

SMALL = Instance((2, 2, 3))
TINY = Instance((2, 3, 4))
PI = Fraction(1, 2)
ARMS = {"T1": STRUCTURES["T1"], "T2": STRUCTURES["T2"], "T3": STRUCTURES["T3"],
        "game:all/none": GAME[("all", "none")], "game:none/all": GAME[("none", "all")]}


# -- naive references ---------------------------------------------------------------------

class CachedWorld(World):
    """``World`` with its pure functions remembered: the same literal definitions, computed once."""

    def __init__(self, inst):
        super().__init__(inst)
        self._c = {}

    def _memo(self, name, fn, *args):
        key = (name,) + args
        if key not in self._c:
            self._c[key] = fn(*args)
        return self._c[key]

    def final(self, t, K):
        return self._memo("final", super().final, t, K)

    def best_offer(self, t, K):
        return self._memo("best", super().best_offer, t, K)

    def accepts(self, t, share):
        return self._memo("acc", super().accepts, t, share)

    def g(self, t, share):
        return self._memo("g", super().g, t, share)

    def part(self, K, q, ans):
        return self._memo("part", super().part, K, q, ans)


class NaiveA1:
    """A1's moves for every A type, from the frozenset reference."""

    def __init__(self, w, st, q, ties=False):
        self.w, self.st, self.ties = w, st, ties
        self.nv = {t: NaiveL2(w, i, st, q, PI) for i, t in enumerate(w.types)}

    def b_answers(self, owed):
        return self.st.b_answers == "all" or (self.st.b_answers == "reciprocal" and owed)

    def ask(self, t, r, KB, KA, a_asked, b_asked, owed):
        if not (self.st.a_asks and self.b_answers(owed)):
            return None
        nv, w = self.nv[t], self.w
        best, choice = nv.v(r, "B", KB, KA, a_asked, b_asked, owed), None
        for j, qu in enumerate(w.qs):
            if j in a_asked:
                continue
            val = sum((Fraction(len(p), len(KB)) * nv.v(r, "B", p, KA, a_asked | {j}, b_asked, owed)
                       for p in (w.part(KB, qu, True), w.part(KB, qu, False)) if p), Fraction(0))
            if val > best:
                best, choice = val, j
        return choice

    def answers(self, t, r, KB, KA, a_asked, b_asked, owed, j):
        if self.st.a_answers != "choose":
            return self.st.a_answers == "all"
        nv, w = self.nv[t], self.w
        told = w.part(KA, w.qs[j], self.truth(t, j))
        v_ans = nv.v(r - 1, "A", KB, told, a_asked, b_asked | {j}, True)
        v_dec = nv.v(r - 1, "A", KB, KA, a_asked, b_asked | {j}, False)
        return v_ans > v_dec or (self.ties and v_ans == v_dec)

    def truth(self, t, j):
        from .private_d import answer_private
        return answer_private(self.w.inst, t, self.w.qs[j])


class NaiveWorld:
    """B2 for every B type, B2's reading, and A3, all over frozensets."""

    def __init__(self, w, st, q, ties=False):
        self.w, self.st, self.q = w, st, q
        self.a1 = NaiveA1(w, st, q, ties)
        self.memo_b, self.memo_a = {}, {}

    # B2's reading (decision 4)
    def read_ask(self, K, r, KB, KA, aa, ba, owed, j):
        Kc = frozenset(t for t in K if self.a1.ask(t, r, KB, KA, aa, ba, owed) == j)
        return Kc or K

    def read_answer(self, K, r, KB, KA, aa, ba, owed, j, yes, KA_after):
        fit = frozenset(t for t in K if self.a1.truth(t, j) == yes)
        act = frozenset(t for t in fit if self.a1.answers(t, r, KB, KA, aa, ba, owed, j))
        return act or fit or KA_after

    def read_decline(self, K, r, KB, KA, aa, ba, owed, j):
        Kd = frozenset(t for t in K if not self.a1.answers(t, r, KB, KA, aa, ba, owed, j))
        return Kd or K

    # B2 for B type b
    def b2(self, b, r, phase, KB, KA, aa, ba, owed, K):
        key = (b, r, phase, KB, KA, aa, ba, owed, K)
        if key not in self.memo_b:
            self.memo_b[key] = self._b2(b, r, phase, KB, KA, aa, ba, owed, K)
        return self.memo_b[key]

    def _b2(self, b, r, phase, KB, KA, aa, ba, owed, K):
        w, a1, st = self.w, self.a1, self.st
        if r == 0:
            tot = Fraction(0)
            for t in K:
                i = w.best_offer(t, KB)
                if i is not None and w.accepts(b, w.inst.rest(w.allocs[i])):
                    tot += w.g(b, w.inst.rest(w.allocs[i]))
            return PI * tot / len(K) + (1 - PI) * w.final(b, K), None
        if phase == "A":
            if st.a_asks and a1.b_answers(owed):
                groups = {}
                for t in K:
                    groups.setdefault(a1.ask(t, r, KB, KA, aa, ba, owed), set()).add(t)
                v = Fraction(0)
                for j, Kj in groups.items():
                    Kj = frozenset(Kj)
                    if j is None:
                        v += len(Kj) * self.b2(b, r, "B", KB, KA, aa, ba, owed, Kj)[0]
                    else:
                        nb = w.part(KB, w.qs[j], a1.truth(b, j))
                        v += len(Kj) * self.b2(b, r, "B", nb, KA, aa | {j}, ba, owed, Kj)[0]
                return v / len(K), None
            return self.b2(b, r, "B", KB, KA, aa, ba, owed, K)[0], None
        if not st.b_asks:
            return self.b2(b, r - 1, "A", KB, KA, aa, ba, owed, K)[0], None
        best, choice = self.b2(b, r - 1, "A", KB, KA, aa, ba, owed, K)[0], None
        for j, qu in enumerate(w.qs):
            if j in ba or not w.part(KA, qu, True) or not w.part(KA, qu, False):
                continue
            groups = {}
            for t in K:
                ans = a1.answers(t, r, KB, KA, aa, ba, owed, j)
                groups.setdefault((ans, a1.truth(t, j) if ans else None), set()).add(t)
            v = Fraction(0)
            for (ans, truth), Kp in groups.items():
                Kp = frozenset(Kp)
                if ans:
                    v += len(Kp) * self.b2(b, r - 1, "A", KB, w.part(KA, qu, truth), aa, ba | {j}, True, Kp)[0]
                else:
                    v += len(Kp) * self.b2(b, r - 1, "A", KB, KA, aa, ba | {j}, False, Kp)[0]
            v /= len(K)
            if v > best:
                best, choice = v, j
        return best, choice

    def b2_root(self, b):
        K = frozenset(self.w.types)
        return self.b2(b, self.q, "A", K, K, frozenset(), frozenset(), True, K)[0]

    # A3 for A type ia
    def a3(self, ia, r, phase, KB, KA, aa, ba, owed, K, S3):
        key = (ia, r, phase, KB, KA, aa, ba, owed, K, S3)
        if key not in self.memo_a:
            self.memo_a[key] = self._a3(ia, r, phase, KB, KA, aa, ba, owed, K, S3)
        return self.memo_a[key]

    def _a3(self, ia, r, phase, KB, KA, aa, ba, owed, K, S3):
        w, a1, st = self.w, self.a1, self.st
        if r == 0:
            resp = Fraction(0)
            for b in S3:
                i = w.best_offer(b, K)
                if i is not None:
                    mine = w.inst.rest(w.allocs[i])
                    if w.accepts(ia, mine):
                        resp += w.g(ia, mine)
            return PI * w.final(ia, S3) + (1 - PI) * resp / len(S3)
        if phase == "A":
            best = self.a3(ia, r, "B", KB, KA, aa, ba, owed, self.read_ask(K, r, KB, KA, aa, ba, owed, None), S3)
            if st.a_asks and a1.b_answers(owed):
                for j, qu in enumerate(w.qs):
                    if j in aa or not w.part(KB, qu, True) or not w.part(KB, qu, False):
                        continue
                    Kj = self.read_ask(K, r, KB, KA, aa, ba, owed, j)
                    v = Fraction(0)
                    for yes in (True, False):
                        part = frozenset(b for b in S3 if a1.truth(b, j) == yes)
                        if part:
                            v += len(part) * self.a3(ia, r, "B", w.part(KB, qu, yes), KA, aa | {j}, ba, owed, Kj, part)
                    best = max(best, v / len(S3))
            return best
        if not st.b_asks:
            return self.a3(ia, r - 1, "A", KB, KA, aa, ba, owed, K, S3)
        groups = {}
        for b in S3:
            groups.setdefault(self.b2(b, r, "B", KB, KA, aa, ba, owed, K)[1], set()).add(b)
        v = Fraction(0)
        for j, part in groups.items():
            part = frozenset(part)
            if j is None:
                v += len(part) * self.a3(ia, r - 1, "A", KB, KA, aa, ba, owed, K, part)
                continue
            truth = a1.truth(ia, j)
            KA_after = w.part(KA, w.qs[j], truth)
            vals = []
            if st.a_answers != "none":
                Ka = self.read_answer(K, r, KB, KA, aa, ba, owed, j, truth, KA_after)
                vals.append(self.a3(ia, r - 1, "A", KB, KA_after, aa, ba | {j}, True, Ka, part))
            if st.a_answers != "all":
                Kd = self.read_decline(K, r, KB, KA, aa, ba, owed, j)
                vals.append(self.a3(ia, r - 1, "A", KB, KA, aa, ba | {j}, False, Kd, part))
            v += len(part) * max(vals)
        return v / len(S3)

    def a3_root(self, ia):
        K = frozenset(self.w.types)
        return self.a3(ia, self.q, "A", K, K, frozenset(), frozenset(), True, K, K)


# -- fixtures -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    w = CachedWorld(SMALL)
    return w, Shared(w.tab)


@pytest.fixture(scope="module")
def tiny():
    w = CachedWorld(TINY)
    return w, Shared(w.tab)


def _gain_b(sh, b, k):
    return sh.seat[b].gain[sh.mirror[k]] if k is not None else Fraction(0)


def _gain_a(sh, ia, k):
    return sh.seat[ia].gain[k] if k is not None else Fraction(0)


# -- B2 ---------------------------------------------------------------------------------

def _check_b2(w, sh, st, q, bs, ties=False):
    m = A1Model(sh, st, q, PI, answer_ties=ties)
    nw = NaiveWorld(w, st, q, ties=ties)
    for b in bs:
        assert B2(sh, m, b).root() == nw.b2_root(w.types[b]), (q, b)


@pytest.mark.parametrize("arm", list(ARMS))
def test_b2_matches_naive_small_q1(world, arm):
    w, sh = world
    _check_b2(w, sh, ARMS[arm], 1, range(0, len(w.types), 4))


@pytest.mark.parametrize("arm", list(ARMS))
def test_b2_matches_naive_tiny_q2(tiny, arm):
    w, sh = tiny
    _check_b2(w, sh, ARMS[arm], 2, range(len(w.types)))


@pytest.mark.parametrize("ties", [False, True])
def test_b2_on_a1_prime_matches_naive(tiny, ties):
    w, sh = tiny
    _check_b2(w, sh, STRUCTURES["T3"], 2, range(len(w.types)), ties=ties)


@pytest.mark.parametrize("arm", ["T2", "T3", "game:none/all"])
def test_b2_rollouts_realize_and_beat_b0(world, arm):
    """B2's value is what its rollouts against A1 realize, never misreads A1, and is at least B0's."""
    w, sh = world
    st, n = ARMS[arm], len(w.types)
    m = A1Model(sh, st, 3, PI)
    for b in range(n):
        b2 = B2(sh, m, b)
        tot_b2 = tot_b0 = Fraction(0)
        for ia in range(n):
            o = rollout_a1_b2(sh, m, ia, b2)
            assert not o["misread"]
            tot_b2 += PI * _gain_b(sh, b, o["a_deal"]) + (1 - PI) * _gain_b(sh, b, o["b_deal"])
            z = rollout_a1_b0(sh, m, ia, b)
            tot_b0 += PI * _gain_b(sh, b, z["a_deal"]) + (1 - PI) * _gain_b(sh, b, z["b_deal"])
        assert tot_b2 / n == b2.root()
        assert tot_b2 >= tot_b0


@pytest.mark.parametrize("arm", ["T1", "T2", "T3"])
def test_b0_episode_reproduces_program(world, arm):
    w, sh = world
    st, n = ARMS[arm], len(w.types)
    m = A1Model(sh, st, 3, PI)
    for ia in range(0, n, 2):
        prog = m.prog(ia)
        for b in range(n):
            o, z = prog.rollout(b), rollout_a1_b0(sh, m, ia, b)
            assert (o["a_deal"], o["b_deal"]) == (z["a_deal"], z["b_deal"])
            if st.a_answers == "choose":                 # Program.rollout records A's choices only in T3
                assert [(j, a) for j, a, _ in o["decisions"]] == [(j, a) for j, a, _ in z["decisions"]]


def test_b2d_reads_without_misreading(world):
    w, sh = world
    st, n = STRUCTURES["T3"], len(w.types)
    m = A1Model(sh, st, 3, PI)
    for ia in range(n):
        for b in range(0, n, 3):
            o = rollout_a1_b2d(sh, m, ia, b)
            assert not o["misread"] and (o["K"] >> ia) & 1


def test_a1_prime_answers_whenever_a1_does(world):
    w, sh = world
    st = STRUCTURES["T3"]
    m, mp = A1Model(sh, st, 3, PI), A1Model(sh, st, 3, PI, answer_ties=True)
    full = w.tab.full
    for ia in range(len(w.types)):
        for j in range(len(w.tab.questions)):
            for r in (1, 2, 3):
                a, ap = m.answers(ia, r, full, full, frozenset(), True, j), mp.answers(ia, r, full, full, frozenset(), True, j)
                assert ap or not a
    assert m.prog(0).root() == mp.prog(0).root()


# -- A3 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["T2", "T3", "game:all/none"])
@pytest.mark.parametrize("ties", [False, True])
def test_a3_matches_naive(tiny, arm, ties):
    w, sh = tiny
    st, n = ARMS[arm], len(w.types)
    m = A1Model(sh, st, 2, PI, answer_ties=ties)
    b2s = {b: B2(sh, m, b) for b in range(n)}
    nw = NaiveWorld(w, st, 2, ties=ties)
    for ia in range(n):
        assert A3(sh, m, b2s, ia).root() == nw.a3_root(w.types[ia]), (arm, ia)


@pytest.mark.parametrize("arm", ["T2", "T3"])
def test_a3_rollouts_realize_and_beat_a1(world, arm):
    """A3's value is what its rollouts realize, and at least A1's against the same B2."""
    w, sh = world
    st, n = ARMS[arm], len(w.types)
    m = A1Model(sh, st, 3, PI)
    b2s = {b: B2(sh, m, b) for b in range(n)}
    for ia in range(0, n, 2):
        a3 = A3(sh, m, b2s, ia)
        tot3 = tot1 = Fraction(0)
        for b in range(n):
            o = rollout_a3_b2(sh, a3, b)
            tot3 += PI * _gain_a(sh, ia, o["a_deal"]) + (1 - PI) * _gain_a(sh, ia, o["b_deal"])
            z = rollout_a1_b2(sh, m, ia, b2s[b])
            tot1 += PI * _gain_a(sh, ia, z["a_deal"]) + (1 - PI) * _gain_a(sh, ia, z["b_deal"])
        assert tot3 / n == a3.root()
        assert tot3 >= tot1
