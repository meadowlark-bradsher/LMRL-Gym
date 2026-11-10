"""L1, the one-sided belief program of Split Deal (`DESIGN_SPLIT.md` §5, protocol §4.1).

A (the focal side) knows its own type. B's type is drawn from a known prior over the instance's types;
B answers bank questions truthfully and accepts a final offer iff its utility is >= d. Answers are
deterministic given the type, so A's posterior is always uniform over a subset S of the prior's
support, held here as a bitmask over the instance's types, and every value is an exact Fraction in Y
units (net gain over d as a share of 10 - d).

Protocol 4.1, Q talk moves then one binding offer or a walk:

    final(S)    = max(0, max_a |acc(a) & S| / |S| * gain(a))         the best single offer, or walk
    words(S, q) = max(final(S), max_j -c + sum_parts |part| / |S| * words(part, q - 1))

A question that does not split S is never worth asking (c >= 0), so the state needs no record of which
questions were asked: (S, q) suffices. Ties go to not asking, then to the lower bank index; among
offers, to the higher acceptance count, then the lower allocation index; walking wins a tie at 0.

Also here, for the reported comparisons of §7:

* greedy policies at c = 0, asking while a question splits S: EIG-greedy (T4, most expected
  information about B's type, i.e. least sum k log k over the answer parts) and value-greedy (the
  largest one-step expected offer value, ties by EIG);
* the re-offer variant: H moves, each an ask, a binding offer or a walk; a rejected offer continues
  with probability 1 - h, and a rejection eliminates the types that would have accepted.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .game import TOTAL, Instance, Values, answer, bank, outside_option

ZERO = Fraction(0)


class Tables:
    """What every A type of one (instance, regime) shares: B's acceptance sets and answer sets."""

    def __init__(self, inst: Instance, regime: str):
        self.inst, self.regime = inst, regime
        self.types: List[Values] = inst.types()
        self.allocs = inst.allocations()
        self.questions = bank(len(inst.counts))
        self.d = outside_option(regime)
        self.scale = TOTAL - self.d
        self.full = (1 << len(self.types)) - 1
        self.acc = [self._mask(lambda t, a=a: inst.utility(t, inst.rest(a)) >= self.d) for a in self.allocs]
        self.yes = [self._mask(lambda t, q=q: answer(inst, t, q)) for q in self.questions]

    def _mask(self, pred) -> int:
        return sum(1 << i for i, t in enumerate(self.types) if pred(t))

    def gains(self, va: Values) -> List[Fraction]:
        return [(Fraction(self.inst.utility(va, a)) - self.d) / self.scale for a in self.allocs]


def _eig_key(k1: int, k0: int) -> float:
    """Smaller is more informative: sum of k log k over the answer parts (uniform posterior)."""
    return k1 * math.log(k1) + k0 * math.log(k0)


class Cell:
    """One A type against one prior over B's types, with ask cost ``cost`` in Y."""

    def __init__(self, tables: Tables, va: Values, prior: int, cost: Fraction = ZERO):
        self.t, self.va, self.prior, self.cost = tables, va, prior, cost
        self.gain = tables.gains(va)
        self._final: Dict[int, Tuple[Fraction, Optional[int]]] = {}
        self._words: Dict[Tuple[int, int], Tuple[Fraction, Optional[int]]] = {}
        self._greedy: Dict[Tuple[int, int, str], Fraction] = {}
        self._reoffer: Dict[Tuple[int, int, Fraction, bool], Fraction] = {}

    # -- the final offer ------------------------------------------------------------

    def final(self, S: int) -> Tuple[Fraction, Optional[int]]:
        """(value, allocation index), or (0, None) for a walk."""
        hit = self._final.get(S)
        if hit is not None:
            return hit
        best, best_a, best_k = ZERO, None, 0
        for i, g in enumerate(self.gain):
            if g <= 0:
                continue
            k = (self.t.acc[i] & S).bit_count()
            if k == 0:
                continue
            num = k * g
            if num > best or (num == best and k > best_k):
                best, best_a, best_k = num, i, k
        out = (best / S.bit_count(), best_a)
        self._final[S] = out
        return out

    def splits(self, S: int):
        for j, y in enumerate(self.t.yes):
            s1, s0 = S & y, S & ~y
            if s1 and s0:
                yield j, s1, s0

    # -- the exact program (T5) -----------------------------------------------------

    def words(self, S: int, q: int) -> Tuple[Fraction, Optional[int]]:
        """(value, question index to ask, or None to stop and make the final offer)."""
        key = (S, q)
        hit = self._words.get(key)
        if hit is not None:
            return hit
        best, choice = self.final(S)[0], None
        if q > 0:
            n = S.bit_count()
            for j, s1, s0 in self.splits(S):
                v = (s1.bit_count() * self.words(s1, q - 1)[0] + s0.bit_count() * self.words(s0, q - 1)[0]) / n
                v -= self.cost
                if v > best:
                    best, choice = v, j
        self._words[key] = (best, choice)
        return best, choice

    # -- greedy policies at c = 0 ---------------------------------------------------

    def greedy_choice(self, S: int, rule: str) -> Optional[int]:
        best_key, choice = None, None
        for j, s1, s0 in self.splits(S):
            k1, k0 = s1.bit_count(), s0.bit_count()
            eig = _eig_key(k1, k0)
            if rule == "eig":
                key = (eig,)
            elif rule == "value":
                key = (-(k1 * self.final(s1)[0] + k0 * self.final(s0)[0]), eig)
            else:
                raise ValueError(rule)
            if best_key is None or key < best_key:
                best_key, choice = key, j
        return choice

    def greedy(self, S: int, q: int, rule: str) -> Fraction:
        key = (S, q, rule)
        hit = self._greedy.get(key)
        if hit is not None:
            return hit
        j = self.greedy_choice(S, rule) if q > 0 else None
        if j is None:
            v = self.final(S)[0]
        else:
            y = self.t.yes[j]
            s1, s0 = S & y, S & ~y
            v = (s1.bit_count() * self.greedy(s1, q - 1, rule) + s0.bit_count() * self.greedy(s0, q - 1, rule))
            v /= S.bit_count()
        self._greedy[key] = v
        return v

    # -- playing a policy against one B type ----------------------------------------

    def rollout(self, b: int, q: int, policy: str = "exact") -> Tuple[int, Optional[int], bool]:
        """(questions asked, allocation agreed or None, walked) when B's type is index ``b``."""
        S, asks = self.prior, 0
        while q > 0:
            j = self.words(S, q)[1] if policy == "exact" else self.greedy_choice(S, policy)
            if j is None:
                break
            S = S & self.t.yes[j] if (self.t.yes[j] >> b) & 1 else S & ~self.t.yes[j]
            q, asks = q - 1, asks + 1
        a = self.final(S)[1]
        if a is None:
            return asks, None, True
        return asks, (a if (self.t.acc[a] >> b) & 1 else None), False

    # -- the re-offer variant -------------------------------------------------------

    def reoffer(self, S: int, m: int, h: Fraction, asks: bool) -> Fraction:
        """H = m moves left, each an ask (if allowed), a binding offer, or a walk; hazard h after a rejection."""
        if m == 0:
            return ZERO
        key = (S, m, h, asks)
        hit = self._reoffer.get(key)
        if hit is not None:
            return hit
        n, best = S.bit_count(), ZERO
        for i, g in enumerate(self.gain):
            acc = self.t.acc[i] & S
            if not acc:
                continue                       # a sure rejection is never better than waiting a move
            rej = S & ~self.t.acc[i]
            v = acc.bit_count() * g
            if rej:
                v += (1 - h) * rej.bit_count() * self.reoffer(rej, m - 1, h, asks)
            best = max(best, v / n)
        if asks:
            for _, s1, s0 in self.splits(S):
                v = (s1.bit_count() * self.reoffer(s1, m - 1, h, asks)
                     + s0.bit_count() * self.reoffer(s0, m - 1, h, asks)) / n - self.cost
                best = max(best, v)
        self._reoffer[key] = best
        return best

    # -- B's type known ---------------------------------------------------------------

    def reveal(self) -> Fraction:
        bs = [b for b in range(len(self.t.types)) if (self.prior >> b) & 1]
        return sum((self.final(1 << b)[0] for b in bs), ZERO) / len(bs)
