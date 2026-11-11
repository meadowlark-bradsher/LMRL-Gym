"""Level-k readers for Split Deal v0.3 (`DESIGN_SPLIT_B2.md`): A1 as a model, B2, B2-d and A3. Offline, exact.

Protocol 4.2 and the structures of ``twosided.py`` are unchanged; only the counterpart changes.

A1 as a model (``A1Model``): A1 is ``twosided.Program`` (the exact best response to B0) for every A type.
B2 needs A1's move at any state, including states B0 would never reach:
* asks: ``Program.step`` at the A phase (unchanged; ties to not asking, then the lower bank index);
* answering question j at a B phase: answer iff answering is strictly better than declining with B0
  playing the rest (``Program.value`` at the two continuations). A1' is the same with ties broken to
  answering. A1 and A1' have the same value against B0 (decision 3);
* the final offer: ``Cell.final`` on its belief S_B.
A1's state is public: S_B (from B's answers), S_A (B0's belief: the types consistent with A's answers),
the questions A declined, and whether B owes an answer (T3).

B2 (``B2``), for one B type b, the exact best response to the A1 model:
* belief K: the A types whose A1 policy is consistent with everything A has done (its asks, answers and
  declines), uniform over K, kept beside the public state;
* questions: at its turn B2 passes or asks any question B0 could ask (not declined, splitting S_A),
  choosing the one with the largest expected Y for b; ties to passing, then the lower bank index. This
  keeps B0's policy feasible for B2, so B2 does at least as well as B0 against A1;
* offer: ``Cell.final`` for b on K (B0's offer rule on B2's belief); accepts iff U_B >= its own d;
* answering: fixed by the structure, as for B0 (decision 5).

B2-d (``b2d_rollout``): B0's question rule and offer rule, with belief K read from A's answers and
declines only (A's asks are not read). Played against A1 only, as the decomposition of decision 1.

A3 (``A3``), for one A type, the exact best response to B2 (B2 built for every B type on the same A1
model): A3 reads B2's question choices, which depend on B's type, as well as B's answers. Its asks are
drawn from A1's domain (questions splitting S_B), ties to not asking, then the lower index; at B2's
questions it answers or declines as the structure allows, breaking ties as the A1 model does.

B2's reading when A does something no type in K would do under the A1 model (decision 4; only A3 can):
* an ask or a decline: ignored, K unchanged;
* an answer: K keeps the types in K whose truthful answer matches, read as if they had answered; if
  none matches, K resets to S_A after the answer (the types consistent with all of A's answers).
Inside B2's own program every action is one the A1 model predicts, so the rule only acts on A3's
deviations; ``update_*`` below is the one implementation both use.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, Optional, Tuple

from .twosided import Program, Shared, Structure

ZERO = Fraction(0)


def bits(mask: int):
    i = 0
    while mask:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1


def partition(mask: int, key: Callable[[int], object]) -> Dict[object, int]:
    out: Dict[object, int] = {}
    for t in bits(mask):
        k = key(t)
        out[k] = out.get(k, 0) | (1 << t)
    return out


class A1Model:
    """A1 (or A1' with ``answer_ties``) for every A type of one (instance, structure, Q, pi)."""

    def __init__(self, shared: Shared, structure: Structure, q: int, pi: Fraction, answer_ties: bool = False):
        self.s, self.st, self.q, self.pi, self.answer_ties = shared, structure, q, pi, answer_ties
        self.t = shared.t
        self._prog: Dict[int, Program] = {}

    def prog(self, ia: int) -> Program:
        p = self._prog.get(ia)
        if p is None:
            p = self._prog[ia] = Program(self.s, ia, self.st, self.q, self.pi)
        return p

    def ask(self, ia: int, r: int, SB: int, SA: int, declined: frozenset, owed: bool) -> Optional[int]:
        return self.prog(ia).step(r, "A", SB, SA, declined, owed)[1]

    def answers(self, ia: int, r: int, SB: int, SA: int, declined: frozenset, owed: bool, j: int) -> bool:
        if self.st.a_answers == "all":
            return True
        if self.st.a_answers == "none":
            return False
        p = self.prog(ia)
        v_ans = p.value(r - 1, "A", SB, p._told(SA, j), declined, True)
        v_dec = p.value(r - 1, "A", SB, SA, declined | {j}, False)
        return v_ans > v_dec or (self.answer_ties and v_ans == v_dec)

    def offer(self, ia: int, SB: int) -> Optional[int]:
        return self.prog(ia).me.final(SB)[1]


# -- B2's reading of A's actions (decision 4) ---------------------------------------------

def update_ask(m: A1Model, K: int, r, SB, SA, declined, owed, choice) -> int:
    Kc = sum(1 << t for t in bits(K) if m.ask(t, r, SB, SA, declined, owed) == choice)
    return Kc or K


def update_answer(m: A1Model, K: int, r, SB, SA, declined, owed, j: int, yes: bool, SA_after: int) -> int:
    y = m.t.yes[j]
    fit = K & y if yes else K & ~y
    act = sum(1 << t for t in bits(fit) if m.answers(t, r, SB, SA, declined, owed, j))
    return act or fit or SA_after


def update_decline(m: A1Model, K: int, r, SB, SA, declined, owed, j: int) -> int:
    Kd = sum(1 << t for t in bits(K) if not m.answers(t, r, SB, SA, declined, owed, j))
    return Kd or K


def b_will_answer(st: Structure, owed: bool) -> bool:
    return st.b_answers == "all" or (st.b_answers == "reciprocal" and owed)


class B2:
    """B2 for B type ``b``: the exact best response to the A1 model ``m``. Values in b's Y."""

    def __init__(self, shared: Shared, m: A1Model, b: int):
        self.s, self.m, self.b = shared, m, b
        self.t, self.st, self.pi = shared.t, m.st, m.pi
        self.me = shared.seat[b]
        self._v: Dict[tuple, Tuple[Fraction, Optional[int]]] = {}
        self._term: Dict[Tuple[int, int], Fraction] = {}

    def terminal(self, SB: int, K: int) -> Fraction:
        key = (SB, K)
        hit = self._term.get(key)
        if hit is not None:
            return hit
        tot = ZERO
        for t in bits(K):
            k = self.m.offer(t, SB)
            if k is not None:
                g = self.me.gain[self.s.mirror[k]]
                if g >= 0:
                    tot += g
        v = self.pi * tot / K.bit_count() + (1 - self.pi) * self.me.final(K)[0]
        self._term[key] = v
        return v

    def options(self, SA: int, declined: frozenset):
        for j, y in enumerate(self.t.yes):
            if j not in declined and SA & y and SA & ~y:
                yield j

    def value(self, *s) -> Fraction:
        return self.step(*s)[0]

    def step(self, r, phase, SB, SA, declined, owed, K):
        """(value for b, choice): at a B phase the question to ask or None; at an A phase None."""
        if r == 0:
            return self.terminal(SB, K), None
        key = (r, phase, SB, SA, declined, owed, K)
        hit = self._v.get(key)
        if hit is not None:
            return hit
        m, n = self.m, K.bit_count()
        if phase == "A":
            if self.st.a_asks and b_will_answer(self.st, owed):
                v = ZERO
                for j, Kj in partition(K, lambda t: m.ask(t, r, SB, SA, declined, owed)).items():
                    if j is None:
                        nb = SB
                    else:
                        y = self.t.yes[j]
                        nb = SB & y if (y >> self.b) & 1 else SB & ~y
                    v += Kj.bit_count() * self.value(r, "B", nb, SA, declined, owed, Kj)
                out = (v / n, None)
            else:
                out = (self.value(r, "B", SB, SA, declined, owed, K), None)
        elif self.st.b_asks:
            best, choice = self.value(r - 1, "A", SB, SA, declined, owed, K), None
            for j in self.options(SA, declined):
                v = self._ask_value(r, SB, SA, declined, owed, K, j)
                if v > best:
                    best, choice = v, j
            out = (best, choice)
        else:
            out = (self.value(r - 1, "A", SB, SA, declined, owed, K), None)
        self._v[key] = out
        return out

    def _ask_value(self, r, SB, SA, declined, owed, K, j) -> Fraction:
        m, y = self.m, self.t.yes[j]
        v = ZERO
        def reading(t):                    # a decline hides the answer: decliners are not split by it
            ans = m.answers(t, r, SB, SA, declined, owed, j)
            return ans, (bool((y >> t) & 1) if ans else None)

        for (ans, truth), Kp in partition(K, reading).items():
            if ans:
                v += Kp.bit_count() * self.value(r - 1, "A", SB, SA & y if truth else SA & ~y, declined, True, Kp)
            else:
                v += Kp.bit_count() * self.value(r - 1, "A", SB, SA, declined | {j}, False, Kp)
        return v / K.bit_count()

    def root(self) -> Fraction:
        full = self.t.full
        return self.value(self.m.q, "A", full, full, frozenset(), True, full)

    def offer(self, K: int) -> Optional[int]:
        """B2's offer as proposer, as the index of the share it gives A; None to walk."""
        k = self.me.final(K)[1]
        return None if k is None else self.s.mirror[k]


# -- playing one episode ----------------------------------------------------------------

def _episode(shared: Shared, m: A1Model, ia: int, b: int, a_move, b_move, read: str, narrow=None) -> dict:
    """One episode between A type ``ia`` and B type ``b``.

    ``a_move(state, K, S3)`` gives A's ask (state "A"), whether A answers B's question (state "B", ending with
    the question) and A's final offer (state "final"). ``b_move(state, K)`` gives B's question at a B phase.
    ``read`` is how B's belief K is updated: "full" (B2), "declines" (B2-d) or "none" (B0, whose belief is
    S_A). ``narrow(state, K, S3, j)``, if given, is A's reading of B's question choice (A3's belief S3 keeps
    the B types whose B2 would have asked j). ``misread`` is True if A's true type ever leaves K."""
    t, st, q = shared.t, m.st, m.q
    SB = SA = K = S3 = t.full
    declined, owed = frozenset(), True
    decisions = []                                     # (question, answered, round)
    misread = False
    for r in range(q, 0, -1):
        # A's turn
        if st.a_asks and b_will_answer(st, owed):
            j = a_move(("A", r, SB, SA, declined, owed), K, S3)
            if read == "full":
                K = update_ask(m, K, r, SB, SA, declined, owed, j)
            if j is not None:
                y = t.yes[j]
                yb = bool((y >> b) & 1)
                SB = SB & y if yb else SB & ~y
                S3 = S3 & y if yb else S3 & ~y
        # B's turn
        if st.b_asks:
            j = b_move(("B", r, SB, SA, declined, owed), K)
            if narrow is not None:
                S3 = narrow(("B", r, SB, SA, declined, owed), K, S3, j)
            if j is not None:
                ans = a_move(("B", r, SB, SA, declined, owed, j), K, S3)
                decisions.append((j, ans, q - r + 1))
                y = t.yes[j]
                truth = bool((y >> ia) & 1)
                if ans:
                    SA_after = SA & y if truth else SA & ~y
                    if read != "none":
                        K = update_answer(m, K, r, SB, SA, declined, owed, j, truth, SA_after)
                    SA, owed = SA_after, True
                else:
                    if read != "none":
                        K = update_decline(m, K, r, SB, SA, declined, owed, j)
                    declined, owed = declined | {j}, False
                if read == "none":
                    K = SA
        if not (K >> ia) & 1:
            misread = True
    if read == "none":
        K = SA
    k_a = a_move(("final", SB, S3), K, S3)
    a_deal = k_a if (k_a is not None and (t.acc[k_a] >> b) & 1) else None
    me = m.prog(ia).me
    k = shared.seat[b].final(K)[1]
    k_b = None if k is None else shared.mirror[k]
    b_deal = k_b if (k_b is not None and me.gain[k_b] >= 0) else None
    return {"a_deal": a_deal, "b_deal": b_deal, "decisions": decisions, "misread": misread, "K": K}


def a1_mover(m: A1Model, ia: int):
    def move(state, K, S3):
        if state[0] == "A":
            _, r, SB, SA, declined, owed = state
            return m.ask(ia, r, SB, SA, declined, owed)
        if state[0] == "B":
            _, r, SB, SA, declined, owed, j = state
            return m.answers(ia, r, SB, SA, declined, owed, j)
        return m.offer(ia, state[1])
    return move


def b2_mover(b2: B2):
    def move(state, K):
        _, r, SB, SA, declined, owed = state
        return b2.step(r, "B", SB, SA, declined, owed, K)[1]
    return move


def b0_mover(shared: Shared):
    def move(state, K):
        _, r, SB, SA, declined, owed = state
        return shared.b0_question(SA, declined)
    return move


def rollout_a1_b2(shared, m, ia, b2: B2) -> dict:
    return _episode(shared, m, ia, b2.b, a1_mover(m, ia), b2_mover(b2), "full")


def rollout_a1_b2d(shared, m, ia, b) -> dict:
    return _episode(shared, m, ia, b, a1_mover(m, ia), b0_mover(shared) if m.st.b_asks else (lambda s, K: None),
                    "declines")


def rollout_a1_b0(shared, m, ia, b) -> dict:
    return _episode(shared, m, ia, b, a1_mover(m, ia), b0_mover(shared) if m.st.b_asks else (lambda s, K: None),
                    "none")


# -- A3 -------------------------------------------------------------------------------------

class A3:
    """A3 for A type ``ia``: the exact best response to B2 (``b2s[b]`` for every B type)."""

    def __init__(self, shared: Shared, m: A1Model, b2s: Dict[int, B2], ia: int):
        self.s, self.m, self.b2s, self.ia = shared, m, b2s, ia
        self.t, self.st, self.pi = shared.t, m.st, m.pi
        self.me = shared.seat[ia]
        self._v: Dict[tuple, Tuple[Fraction, object]] = {}
        self._term: Dict[Tuple[int, int, int], Fraction] = {}

    def terminal(self, SB, K, S3) -> Fraction:
        key = (SB, K, S3)
        hit = self._term.get(key)
        if hit is not None:
            return hit
        resp = ZERO
        for b in bits(S3):
            k = self.b2s[b].offer(K)
            if k is not None and self.me.gain[k] >= 0:
                resp += self.me.gain[k]
        v = self.pi * self.me.final(S3)[0] + (1 - self.pi) * resp / S3.bit_count()
        self._term[key] = v
        return v

    def value(self, *s) -> Fraction:
        return self.step(*s)[0]

    def step(self, r, phase, SB, SA, declined, owed, K, S3):
        """(value, choice): at an A phase the question to ask or None; at a B phase, for each question
        B2 may ask, whether A3 answers it (a dict)."""
        if r == 0:
            return self.terminal(SB, K, S3), None
        key = (r, phase, SB, SA, declined, owed, K, S3)
        hit = self._v.get(key)
        if hit is not None:
            return hit
        m, t = self.m, self.t
        if phase == "A":
            Kn = update_ask(m, K, r, SB, SA, declined, owed, None)
            best, choice = self.value(r, "B", SB, SA, declined, owed, Kn, S3), None
            if self.st.a_asks and b_will_answer(self.st, owed):
                n = S3.bit_count()
                for j, _, _ in self.me.splits(SB):
                    Kj = update_ask(m, K, r, SB, SA, declined, owed, j)
                    y = t.yes[j]
                    v = ZERO
                    for part, nb in ((S3 & y, SB & y), (S3 & ~y, SB & ~y)):
                        if part:
                            v += part.bit_count() * self.value(r, "B", nb, SA, declined, owed, Kj, part)
                    v /= n
                    if v > best:
                        best, choice = v, j
            out = (best, choice)
        elif self.st.b_asks:
            n, v, choice = S3.bit_count(), ZERO, {}
            for j, part in partition(S3, lambda b: self.b2s[b].step(r, "B", SB, SA, declined, owed, K)[1]).items():
                if j is None:
                    v += part.bit_count() * self.value(r - 1, "A", SB, SA, declined, owed, K, part)
                    continue
                v_ans, v_dec = self._answer_values(r, SB, SA, declined, owed, K, part, j)
                if self.st.a_answers == "all":
                    ans = True
                elif self.st.a_answers == "none":
                    ans = False
                else:
                    ans = v_ans > v_dec or (m.answer_ties and v_ans == v_dec)
                choice[j] = ans
                v += part.bit_count() * (v_ans if ans else v_dec)
            out = (v / n, choice)
        else:
            out = (self.value(r - 1, "A", SB, SA, declined, owed, K, S3), None)
        self._v[key] = out
        return out

    def _answer_values(self, r, SB, SA, declined, owed, K, S3, j):
        m, y = self.m, self.t.yes[j]
        truth = bool((y >> self.ia) & 1)
        SA_after = SA & y if truth else SA & ~y
        v_ans = v_dec = None
        if self.st.a_answers != "none":
            Ka = update_answer(m, K, r, SB, SA, declined, owed, j, truth, SA_after)
            v_ans = self.value(r - 1, "A", SB, SA_after, declined, True, Ka, S3)
        if self.st.a_answers != "all":
            Kd = update_decline(m, K, r, SB, SA, declined, owed, j)
            v_dec = self.value(r - 1, "A", SB, SA, declined | {j}, False, Kd, S3)
        return v_ans, v_dec

    def root(self) -> Fraction:
        full = self.t.full
        return self.value(self.m.q, "A", full, full, frozenset(), True, full, full)

    def mover(self):
        def move(state, K, S3):
            if state[0] == "A":
                _, r, SB, SA, declined, owed = state
                return self.step(r, "A", SB, SA, declined, owed, K, S3)[1]
            if state[0] == "B":
                _, r, SB, SA, declined, owed, j = state
                return self.step(r, "B", SB, SA, declined, owed, K, S3)[1][j]
            return self.me.final(state[2])[1]
        return move


def rollout_a3_b2(shared, a3: A3, b: int) -> dict:
    def narrow(state, K, S3, j):
        _, r, SB, SA, declined, owed = state
        return sum(1 << c for c in bits(S3) if a3.b2s[c].step(r, "B", SB, SA, declined, owed, K)[1] == j)
    return _episode(shared, a3.m, a3.ia, b, a3.mover(), b2_mover(a3.b2s[b]), "full", narrow)
