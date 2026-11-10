"""Exactness checks for the L2 two-sided program (``twosided.py``) and the SD-D sweep (``sweep_l2.py``).

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_twosided.py

The main check is a naive literal simulation of protocol 4.2 (frozensets, explicit asked sets for both
sides, A allowed every move including asks that will be declined), written separately from the bitmask
program and compared with it on small instances.
"""

import json
import math
import os
from fractions import Fraction

import pytest

from .belief import Tables
from .game import Instance, answer, bank
from .sweep_l2 import compute, render
from .twosided import GAME, STRUCTURES, Program, Shared

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = "results/split_l2_20261110/sweep.json"
OUT = os.path.join(ROOT, *REL.split("/"))
NOTE = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "SWEEP_SPLIT_L2.md")
ALL = dict(STRUCTURES, **{f"game:{a}/{b}": s for (a, b), s in GAME.items()})


# -- naive reference ----------------------------------------------------------------------

class Naive:
    def __init__(self, inst, regime, ia, st, q, pi):
        self.tab = Tables(inst, regime)
        self.inst, self.types, self.allocs, self.d = inst, self.tab.types, self.tab.allocs, self.tab.d
        self.va, self.st, self.q, self.pi = self.types[ia], st, q, pi
        self.qs = bank()
        self.memo = {}                                  # plain memoisation of v; does not change what it computes

    def g(self, v, share):
        return (Fraction(self.inst.utility(v, share)) - self.d) / self.tab.scale

    def best_offer(self, v, K):
        """Index of the proposer's share for type v against belief K (frozenset of types), or None."""
        best, best_i, best_k = Fraction(0), None, 0
        for i, s in enumerate(self.allocs):
            k = sum(1 for t in K if self.inst.utility(t, self.inst.rest(s)) >= self.d)
            num = k * self.g(v, s)
            if num > best or (num == best and num > 0 and k > best_k):
                best, best_i, best_k = num, i, k
        return best_i

    def terminal(self, KB, KA):
        i = self.best_offer(self.va, KB)
        prop = Fraction(0)
        if i is not None:
            s = self.allocs[i]
            prop = Fraction(sum(1 for t in KB if self.inst.utility(t, self.inst.rest(s)) >= self.d), len(KB)) * self.g(self.va, s)
        resp = Fraction(0)
        for vb in KB:
            j = self.best_offer(vb, KA)
            if j is not None:
                mine = self.inst.rest(self.allocs[j])
                if self.inst.utility(self.va, mine) >= self.d:
                    resp += self.g(self.va, mine)
        return self.pi * prop + (1 - self.pi) * resp / len(KB)

    def b_question(self, KA, b_asked):
        best, choice = None, None
        for j, qu in enumerate(self.qs):
            if j in b_asked:
                continue
            k1 = sum(1 for t in KA if answer(self.inst, t, qu))
            k0 = len(KA) - k1
            if k1 and k0:
                e = k1 * math.log(k1) + k0 * math.log(k0)
                if best is None or e < best:
                    best, choice = e, j
        return choice

    def b_answers(self, owed):
        return self.st.b_answers == "all" or (self.st.b_answers == "reciprocal" and owed)

    def v(self, *state):
        if state not in self.memo:
            self.memo[state] = self._v(*state)
        return self.memo[state]

    def _v(self, r, phase, KB, KA, a_asked, b_asked, owed):
        if r == 0:
            return self.terminal(KB, KA)
        if phase == "A":
            best = self.v(r, "B", KB, KA, a_asked, b_asked, owed)
            if self.st.a_asks:
                for j, qu in enumerate(self.qs):
                    if j in a_asked:
                        continue
                    if self.b_answers(owed):
                        val = Fraction(0)
                        for ans in (True, False):
                            part = frozenset(t for t in KB if answer(self.inst, t, qu) == ans)
                            if part:
                                val += Fraction(len(part), len(KB)) * self.v(r, "B", part, KA, a_asked | {j}, b_asked, owed)
                    else:
                        val = self.v(r, "B", KB, KA, a_asked | {j}, b_asked, owed)
                    best = max(best, val)
            return best
        j = self.b_question(KA, b_asked) if self.st.b_asks else None
        if j is None:
            return self.v(r - 1, "A", KB, KA, a_asked, b_asked, owed)
        mine = answer(self.inst, self.va, self.qs[j])
        told = frozenset(t for t in KA if answer(self.inst, t, self.qs[j]) == mine)
        v_ans = self.v(r - 1, "A", KB, told, a_asked, b_asked | {j}, True)
        v_dec = self.v(r - 1, "A", KB, KA, a_asked, b_asked | {j}, False)
        return {"all": v_ans, "none": v_dec, "choose": max(v_ans, v_dec)}[self.st.a_answers]

    def root(self):
        K = frozenset(self.types)
        return self.v(self.q, "A", K, K, frozenset(), frozenset(), True)


SMALL = [Instance((2, 2, 3)), Instance((1, 2, 3))]


@pytest.mark.parametrize("inst", SMALL, ids=lambda i: i.key)
@pytest.mark.parametrize("regime", ["easy", "hard"])
def test_program_matches_naive(inst, regime):
    sh = Shared(Tables(inst, regime))
    for name, st in ALL.items():
        for q, pis in ((1, (Fraction(1, 2), Fraction(1, 4))), (2, (Fraction(1, 2),))):
            for pi in pis:
                for ia in range(0, len(sh.t.types), 2):
                    want = Naive(inst, regime, ia, st, q, pi).root()
                    assert Program(sh, ia, st, q, pi).root() == want, (name, q, pi, ia)


def test_t3_matches_naive_at_three_rounds():
    inst = Instance((2, 2, 3))
    sh = Shared(Tables(inst, "easy"))
    for ia in range(len(sh.t.types)):
        assert Program(sh, ia, STRUCTURES["T3"], 3, Fraction(1, 2)).root() == \
            Naive(inst, "easy", ia, STRUCTURES["T3"], 3, Fraction(1, 2)).root()


@pytest.mark.parametrize("inst", SMALL + [Instance((1, 1, 3))], ids=lambda i: i.key)
def test_decomposition_rollouts_and_game_cells(inst):
    for regime in ("easy", "hard"):
        sh = Shared(Tables(inst, regime))
        n, pi, q = len(sh.t.types), Fraction(1, 2), 3
        for ia in range(n):
            me = sh.seat[ia]
            roots = {}
            for name, st in ALL.items():
                prog = Program(sh, ia, st, q, pi)
                roots[name] = prog.root()
                realized = Fraction(0)
                for b in range(n):
                    o = prog.rollout(b)
                    realized += pi * (me.gain[o["a_deal"]] if o["a_deal"] is not None else 0) + \
                        (1 - pi) * (me.gain[o["b_deal"]] if o["b_deal"] is not None else 0)
                assert realized / n == roots[name], name
            # T1 and T2 decompose: A's asking is the L1 program; B's side is B0's offer
            resp0 = Program(sh, ia, STRUCTURES["T0"], q, pi).responder(sh.t.full, sh.t.full)
            assert roots["T1"] == pi * me.words(sh.t.full, q)[0] + (1 - pi) * resp0
            assert roots["T0"] == pi * me.final(sh.t.full)[0] + (1 - pi) * resp0
            # under T2, B0's proposer branch is B's EIG-greedy L1 policy seen from B's seat
            t2 = Program(sh, ia, STRUCTURES["T2"], q, pi)
            for b in range(n):
                k_seat = sh.seat[b].rollout(ia, q, "eig")[1]
                want = None if k_seat is None else sh.mirror[k_seat]
                assert t2.rollout(b)["b_deal"] == want
            # the 2 x 2 cells coincide with the structures they reduce to
            assert roots["game:none/none"] == roots["T0"]
            assert roots["game:none/all"] == roots["T1"]
            assert roots["game:all/all"] == roots["T2"]
            # in T3 A can decline everything and still have its first question answered, so T3 >= T0
            assert roots["T3"] >= roots["T0"]


@pytest.mark.skipif(not os.path.exists(OUT), reason="sweep results missing")
def test_regeneration():
    with open(OUT) as f:
        stored = json.load(f)
    fresh = compute(workers=max(1, (os.cpu_count() or 2) - 2))
    fresh["commit"], fresh["seconds"] = stored["commit"], stored["seconds"]
    assert json.loads(json.dumps(fresh)) == stored
    if os.path.exists(NOTE):
        with open(NOTE) as f:
            assert f.read() == render(stored, REL)
