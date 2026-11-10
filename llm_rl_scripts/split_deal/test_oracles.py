"""Exactness checks for the L0 frontier oracle and the L1 belief program of Split Deal.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_oracles.py

The L1 checks compare the bitmask program with a naive one written independently (frozensets of
types, explicit record of the questions asked, every question allowed, every allocation tried).
"""

import itertools
from fractions import Fraction

import pytest

from .belief import Cell, Tables
from .frontier import efficiency, pair_frontier, share_a, surplus
from .game import TOTAL, Instance, answer, bank, outside_option
from .spec_numbers import pair_constraints

SMALL = [Instance((2, 2, 3)), Instance((1, 2, 3))]
COSTS = [Fraction(0), Fraction(1, 100)]


# -- L0 -------------------------------------------------------------------------------

def test_frontier_by_hand():
    inst = Instance((1, 1, 1))
    va, vb = (5, 5, 0), (0, 5, 5)                  # A wants book + hat, B wants hat + ball
    f = pair_frontier(inst, va, vb, Fraction(4))
    # acceptable: A needs book or hat (>= 5 >= 4), B needs hat or ball
    assert (1, 0, 0) in f.acceptable and (0, 0, 0) not in f.acceptable
    assert f.s_star == 15 - 8                      # book + hat to A (10) and ball to B (5), or the mirror
    # (book) to A: (5 - 4)(10 - 4) = 6; (book, hat) to A: (10 - 4)(5 - 4) = 6; a tie, both Nash points
    assert set(f.nash) == {(1, 0, 0), (1, 1, 0)}
    assert set(f.pareto) == {(1, 0, 0), (1, 1, 0)}


def test_frontier_matches_brute_force():
    for inst in SMALL:
        for regime in ("easy", "hard"):
            d = outside_option(regime)
            types = inst.types()
            for va, vb in itertools.islice(itertools.product(types, types), 0, None, 7):
                f = pair_frontier(inst, va, vb, d)
                u = {a: (inst.utility(va, a), inst.utility(vb, inst.rest(a))) for a in inst.allocations()}
                acc = [a for a in u if u[a][0] >= d and u[a][1] >= d]
                assert list(f.acceptable) == acc
                if not acc:
                    assert f.s_star is None and f.pareto == () and f.nash == ()
                    continue
                assert f.s_star == max(sum(u[a]) - 2 * d for a in acc)
                for a in acc:
                    dominated = any(u[b][0] >= u[a][0] and u[b][1] >= u[a][1] and u[b] != u[a] for b in acc)
                    assert (a in f.pareto) == (not dominated)
                prod = {a: (u[a][0] - d) * (u[a][1] - d) for a in acc}
                assert set(f.nash) == {a for a in acc if prod[a] == max(prod.values())}
                assert set(f.nash) <= set(f.pareto) or max(prod.values()) == 0
                best = max(acc, key=lambda a: sum(u[a]))
                if f.s_star > 0:
                    assert efficiency(inst, va, vb, d, best, f.s_star) == 1
                    assert efficiency(inst, va, vb, d, None, f.s_star) == 0
                assert share_a(inst, va, vb, d, None) is None


def test_metrics_definitions():
    inst = Instance((2, 2, 2))
    va, vb, d = (5, 0, 0), (0, 0, 5), Fraction(4)
    a = (2, 1, 0)                                  # A: books (10); B: one hat + balls (10)
    assert surplus(inst, va, vb, d, a) == 12
    assert share_a(inst, va, vb, d, a) == Fraction(1, 2)
    assert pair_frontier(inst, va, vb, d).s_star == 12


# -- L1, naive reference ----------------------------------------------------------------

def _naive_final(inst, va, S, d, scale):
    best = Fraction(0)
    for a in inst.allocations():
        g = (Fraction(inst.utility(va, a)) - d) / scale
        k = sum(1 for t in S if inst.utility(t, inst.rest(a)) >= d)
        best = max(best, k * g / len(S))
    return best


def _naive_words(inst, va, S, asked, q, d, scale, cost):
    best = _naive_final(inst, va, S, d, scale)
    if q == 0:
        return best
    for i, qu in enumerate(bank()):
        if i in asked:
            continue
        v = -cost
        for ans in (True, False):
            part = frozenset(t for t in S if answer(inst, t, qu) == ans)
            if part:
                v += Fraction(len(part), len(S)) * _naive_words(inst, va, part, asked | {i}, q - 1, d, scale, cost)
        best = max(best, v)
    return best


def _naive_reoffer(inst, va, S, m, d, scale, h, asks):
    if m == 0:
        return Fraction(0)
    best = Fraction(0)
    for a in inst.allocations():
        g = (Fraction(inst.utility(va, a)) - d) / scale
        acc = frozenset(t for t in S if inst.utility(t, inst.rest(a)) >= d)
        rej = S - acc
        v = Fraction(len(acc), len(S)) * g
        if rej:
            v += Fraction(len(rej), len(S)) * (1 - h) * _naive_reoffer(inst, va, rej, m - 1, d, scale, h, asks)
        best = max(best, v)
    if asks:
        for qu in bank():
            v = Fraction(0)
            for ans in (True, False):
                part = frozenset(t for t in S if answer(inst, t, qu) == ans)
                if part:
                    v += Fraction(len(part), len(S)) * _naive_reoffer(inst, va, part, m - 1, d, scale, h, asks)
            best = max(best, v)
    return best


def _cells(inst, regime, cost=Fraction(0), constrained=False):
    tab = Tables(inst, regime)
    for va in tab.types:
        prior = tab.full
        if constrained:
            prior = sum(1 << i for i, vb in enumerate(tab.types) if pair_constraints(va, vb))
        if prior:
            yield tab, Cell(tab, va, prior, cost)


def _support(tab, prior):
    return frozenset(t for i, t in enumerate(tab.types) if (prior >> i) & 1)


@pytest.mark.parametrize("inst", SMALL, ids=lambda i: i.key)
@pytest.mark.parametrize("regime", ["easy", "hard"])
def test_words_matches_naive_up_to_two_questions(inst, regime):
    for cost in COSTS:
        for constrained in (False, True):
            for tab, cell in _cells(inst, regime, cost, constrained):
                S = _support(tab, cell.prior)
                for q in (0, 1, 2):
                    want = _naive_words(inst, cell.va, S, frozenset(), q, tab.d, tab.scale, cost)
                    assert cell.words(cell.prior, q)[0] == want


@pytest.mark.parametrize("inst", SMALL, ids=lambda i: i.key)
def test_reoffer_matches_naive_up_to_two_moves(inst):
    for regime in ("easy", "hard"):
        for h in (Fraction(0), Fraction(1, 4)):
            for tab, cell in _cells(inst, regime):
                S = _support(tab, cell.prior)
                for asks in (False, True):
                    for m in (1, 2):
                        want = _naive_reoffer(inst, cell.va, S, m, tab.d, tab.scale, h, asks)
                        assert cell.reoffer(cell.prior, m, h, asks) == want


def _realized(cell, q, policy):
    total = Fraction(0)
    bs = [b for b in range(len(cell.t.types)) if (cell.prior >> b) & 1]
    for b in bs:
        asks, agreed, _ = cell.rollout(b, q, policy)
        total += (cell.gain[agreed] if agreed is not None else 0) - asks * cell.cost
    return total / len(bs)


@pytest.mark.parametrize("inst", SMALL + [Instance((1, 1, 3))], ids=lambda i: i.key)
def test_rollouts_realize_the_values_and_orderings_hold(inst):
    for regime in ("easy", "hard"):
        for cost in (Fraction(0), Fraction(3, 100)):
            for tab, cell in _cells(inst, regime, cost):
                prev = None
                for q in range(0, 5):
                    v = cell.words(cell.prior, q)[0]
                    assert _realized(cell, q, "exact") == v
                    if cost == 0:
                        for rule in ("eig", "value"):
                            g = cell.greedy(cell.prior, q, rule)
                            assert _realized(cell, q, rule) == g
                            assert g <= v
                    if prev is not None:
                        assert v >= prev
                    prev = v
                assert cell.words(cell.prior, 0)[0] == cell.final(cell.prior)[0]
                if cost == 0:
                    assert prev <= cell.reveal()
                    assert cell.reoffer(cell.prior, 4, Fraction(0), True) >= cell.words(cell.prior, 3)[0]
                    assert cell.reoffer(cell.prior, 4, Fraction(0), True) >= cell.reoffer(cell.prior, 4, Fraction(0), False)


def test_all_questions_reach_the_bank_classes():
    inst = Instance((1, 3, 3))                      # every type has its own answer pattern (largest class 1)
    for tab, cell in _cells(inst, "easy"):
        assert cell.words(cell.prior, 12)[0] == cell.reveal()


def test_units_and_walks():
    inst = Instance((2, 2, 2))
    tab = Tables(inst, "hard")
    va = (5, 0, 0)
    cell = Cell(tab, va, tab.full)
    # everything to A: gain (10 - 5.5) / 4.5 = 1
    assert max(cell.gain) == 1 and tab.scale == TOTAL - Fraction(11, 2)
    # a B type that values only books leaves A nothing acceptable to both above d
    b = tab.types.index((5, 0, 0))
    assert cell.final(1 << b) == (0, None)
