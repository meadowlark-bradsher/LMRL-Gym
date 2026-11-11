"""ST-0b answer channels: structured answerer errors for the stress tests.

Every factory returns an ``OracleModel`` -- ``oracle(q, true_h) -> bool`` -- that
the open-world loop calls exactly like ``matrix_oracle``. The answerer reads its
own ``answer_matrix``, which may differ from the matrix the planner believes
(ST-4 A1: a second model's rows). Oracles that need to know where a game starts
expose ``begin_game(game_index)``; ``run_accumulation`` and the snapshot harness
call it when present, so oracles without it behave exactly as before.

The i.i.d. arm (ST-4 A0) stays ``openworld.noisy_matrix_oracle``, untouched, so
its random stream -- and every committed noisy run -- is unchanged.

All randomness is seeded from integers or strings, never from ``hash()``, which
varies across processes (the CD-MIS hash-seed bug).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .bed import ATTRIBUTE, ATTRIBUTE_BANK, GUESS, Hypothesis, OracleModel, matrix_oracle

Matrix = Dict[str, Dict[str, float]]


def _threshold_answer(matrix: Matrix, aid: str, name: str, threshold: float) -> bool:
    return matrix[aid][name] >= threshold


def sampled_oracle(answer_matrix: Matrix, seed: int) -> OracleModel:
    """ST-4 A3: each attribute answer is drawn yes ~ Bernoulli(p) from the Noul
    row instead of thresholding it. Category and guess answers stay exact."""
    base = matrix_oracle(answer_matrix)
    rng = random.Random(seed)

    def oracle(q, true_h: Hypothesis) -> bool:
        if q.kind == ATTRIBUTE:
            return rng.random() < answer_matrix[q.key][true_h.name]
        return base(q, true_h)

    return oracle


def sticky_oracle(answer_matrix: Matrix, m: int, seed: int,
                  threshold: float = 0.5) -> OracleModel:
    """ST-4 A2: for each object, ``m`` attribute cells (chosen once, seeded by
    ``seed`` and the object's name) are always answered wrong. Re-asking never
    helps, and the same cells are wrong every time the object recurs."""
    base = matrix_oracle(answer_matrix, threshold=threshold)
    aids = [aid for aid, _ in ATTRIBUTE_BANK]
    wrong: Dict[str, set] = {}

    def cells_for(name: str) -> set:
        if name not in wrong:
            wrong[name] = set(random.Random("{}:{}".format(seed, name)).sample(aids, m))
        return wrong[name]

    def oracle(q, true_h: Hypothesis) -> bool:
        ans = base(q, true_h)
        if q.kind == ATTRIBUTE and q.key in cells_for(true_h.name):
            return not ans
        return ans

    oracle.wrong_cells = cells_for  # for diagnostics and tests
    return oracle


def bursty_oracle(answer_matrix: Matrix, burst_p: float, inner_flip: float,
                  seed: int, threshold: float = 0.5) -> OracleModel:
    """ST-4 A4: each game is "confused" with probability ``burst_p``; inside a
    confused game every attribute answer flips with probability ``inner_flip``,
    otherwise answers are clean. The marginal flip rate is burst_p * inner_flip.
    The game state is drawn in ``begin_game``; a game that never called it is
    drawn lazily on its first question."""
    base = matrix_oracle(answer_matrix, threshold=threshold)
    rng = random.Random(seed)
    state = {"confused": None}

    def begin_game(game_index: int) -> None:
        state["confused"] = rng.random() < burst_p

    def oracle(q, true_h: Hypothesis) -> bool:
        if state["confused"] is None:
            begin_game(-1)
        ans = base(q, true_h)
        if q.kind == ATTRIBUTE and state["confused"] and rng.random() < inner_flip:
            return not ans
        return ans

    oracle.begin_game = begin_game
    oracle.state = state
    return oracle


def hamming(matrix: Matrix, a: str, b: str, threshold: float = 0.5) -> int:
    return sum((matrix[aid][a] >= threshold) != (matrix[aid][b] >= threshold)
               for aid, _ in ATTRIBUTE_BANK)


def guess_noise(base: OracleModel, answer_matrix: Matrix, false_yes: float,
                false_no: float, seed: int, near: int = 2,
                threshold: float = 0.5) -> OracleModel:
    """ST-4 G / ST-2 G: wrap ``base`` so by-name guesses are no longer exact.
    A guess of the true name is answered "no" with probability ``false_no``; a
    guess of another object whose binarized row is within ``near`` attributes of
    the secret's is answered "yes" with probability ``false_yes``. All other
    questions go to ``base`` unchanged. Pair with ST-0c scoring, which counts a
    "yes" to the wrong name as ``false_confirm`` rather than a win."""
    rng = random.Random(seed)

    def oracle(q, true_h: Hypothesis) -> bool:
        if q.kind != GUESS:
            return base(q, true_h)
        if q.key == true_h.name:
            return not (false_no > 0.0 and rng.random() < false_no)
        if false_yes > 0.0 and hamming(answer_matrix, q.key, true_h.name, threshold) <= near:
            return rng.random() < false_yes
        return False

    if hasattr(base, "begin_game"):
        oracle.begin_game = base.begin_game
    return oracle


def lookalike_confusion(base: OracleModel, parent_of: Dict[str, str],
                        false_yes: float, seed: int) -> OracleModel:
    """ST-2 G arm: when the secret is a look-alike and the planner guesses its
    *parent* by name, answer "yes" with probability ``false_yes`` (the answerer
    takes the subspecies for the species). ``parent_of`` maps look-alike name to
    parent name; only the look-alikes the study marks confusable belong in it."""
    rng = random.Random(seed)

    def oracle(q, true_h: Hypothesis) -> bool:
        if (q.kind == GUESS and parent_of.get(true_h.name) == q.key
                and false_yes > 0.0):
            return rng.random() < false_yes
        return base(q, true_h)

    if hasattr(base, "begin_game"):
        oracle.begin_game = base.begin_game
    return oracle


class FlipCounter:
    """Wrap any oracle and count how often its attribute answers disagree with a
    0.5 threshold of the *belief* matrix: the realised flip rate on the questions
    actually asked, which ST-4 matches its i.i.d. control against."""

    def __init__(self, oracle: OracleModel, belief_matrix: Matrix,
                 threshold: float = 0.5):
        self.oracle = oracle
        self.belief = belief_matrix
        self.threshold = threshold
        self.attr_asked = 0
        self.attr_flipped = 0
        self.guess_asked = 0
        self.guess_wrong = 0
        if hasattr(oracle, "begin_game"):
            self.begin_game = oracle.begin_game

    def __call__(self, q, true_h: Hypothesis) -> bool:
        ans = self.oracle(q, true_h)
        if q.kind == ATTRIBUTE:
            self.attr_asked += 1
            truth = _threshold_answer(self.belief, q.key, true_h.name, self.threshold)
            self.attr_flipped += int(ans != truth)
        elif q.kind == GUESS:
            self.guess_asked += 1
            self.guess_wrong += int(ans != (q.key == true_h.name))
        return ans

    def rates(self) -> Dict[str, Optional[float]]:
        return {
            "attr_asked": self.attr_asked,
            "attr_flip_rate": (self.attr_flipped / self.attr_asked
                               if self.attr_asked else None),
            "guess_asked": self.guess_asked,
            "guess_error_rate": (self.guess_wrong / self.guess_asked
                                 if self.guess_asked else None),
        }
