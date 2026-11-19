"""FLYW-7 game engine: more than one cause (``FLYW2_COMPOUND_PREREG.md``).

Shared by the calibration (``flyw2_calibration.py``) and, after registration, the
runner and analyzer.

* ``KeyedSetOracle`` -- the matrix oracle over a *set* of secrets. An answer is the
  logical OR of the members' exact answers (threshold 0.5 for attributes, closed form
  for category and by-name questions). Attribute answers are then flipped with
  probability ``flip_p`` by a keyed hash of (seed, game key, question), FLYW-1's paired
  noise, under a fresh ``flyw2`` namespace so no FLYW-1 stream is reused.
* ``play_s`` -- arm S, today's single-cause P0a loop (``run_openworld_episode`` with
  ``modeswitch_guard``, unchanged) against that oracle, with a soft-posterior shadow fed
  the same answers. The record keeps every step, so any decision scoring can be
  re-derived.
* ``holdout_novels`` -- the fixed novel set: 10 % of each category, rounded half up,
  never a category's last member.
* ``pair_overlap`` -- how the two secrets' answers combine: ``d_min`` is the number of
  questions on which the better-fitting member is contradicted by the OR answers when
  every question is asked (0 = one member's yes-set contains the other's: *masked*).
* ``ThreeWay`` -- the gap-10 classifier, run on the answers seen when P0a escalates:
  posterior over {single known + noise ("error"), known pair ("compound"), catch-all
  ("novel")} from mean likelihoods.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bed import (ATTRIBUTE, ATTRIBUTE_BANK, CATEGORY, GUESS, Hypothesis, Question,
                  matrix_oracle)
from .openworld import (UNKNOWN, UNKNOWN_HYP, background_marginals, category_marginals,
                        make_openworld_likelihood, run_openworld_episode)
from .openworld_backends import SoftPosterior

Matrix = Dict[str, Dict[str, float]]

# The prereg's fixed P0a settings (FLYW-1's, plus an open-world catch-all prior).
P0A = dict(max_probes=20, confidence=0.9, eps=0.05, tolerance=3)
P_UNKNOWN = 0.10          # catch-all prior, for P0a and the soft shadow, every game
NAMESPACE = "flyw2"


# --------------------------------------------------------------------------- #
# Keyed (paired) noise over a set of secrets                                   #
# --------------------------------------------------------------------------- #

def keyed_uniform(seed: int, game_key: str, qid: str) -> float:
    h = hashlib.sha256("{}|{}|{}|{}".format(NAMESPACE, seed, game_key, qid).encode()
                       ).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def game_key(names: Sequence[str], rep: int = 0) -> str:
    """Order-free key: a pair {a, b} is the same game whichever member is listed first."""
    return "{}#{}".format("|".join(sorted(names)), rep)


class KeyedSetOracle:
    """OR over the secrets' exact answers; attribute flips keyed on (seed, game, q).
    Category and by-name answers stay exact. ``begin(names)`` sets the game."""

    def __init__(self, matrix: Matrix, flip_p: float, seed: int):
        self.base = matrix_oracle(matrix)
        self.flip_p = flip_p
        self.seed = seed
        self.truths: List[Hypothesis] = []
        self.key = ""
        self.n_flips = 0

    def begin(self, truths: Sequence[Hypothesis], rep: int = 0) -> None:
        self.truths = list(truths)
        self.key = game_key([t.name for t in truths], rep)
        self.n_flips = 0

    def exact(self, q: Question) -> bool:
        return any(self.base(q, t) for t in self.truths)

    def __call__(self, q: Question, _true_h: Hypothesis = None) -> bool:
        ans = self.exact(q)
        if (q.kind == ATTRIBUTE and self.flip_p > 0.0
                and keyed_uniform(self.seed, self.key, q.qid) < self.flip_p):
            self.n_flips += 1
            return not ans
        return ans


# --------------------------------------------------------------------------- #
# Pools                                                                        #
# --------------------------------------------------------------------------- #

def holdout_novels(hyps: Sequence[Hypothesis], seed: str = "flyw2-novels") -> List[str]:
    """10 % of each category (half up), never its last member, by a fixed seed."""
    rng = random.Random(seed)
    names = [h.name for h in hyps]
    out: List[str] = []
    for cat in sorted({h.category for h in hyps}):
        mem = [h.name for h in hyps if h.category == cat]
        take = min(int(math.floor(0.1 * len(mem) + 0.5)), len(mem) - 1)
        if take > 0:
            out.extend(rng.sample(mem, take))
    return sorted(out, key=names.index)


def all_questions(known: Sequence[Hypothesis]) -> List[Question]:
    from .openworld import build_openworld_bank
    bank = build_openworld_bank(known)
    return list(bank[CATEGORY]) + list(bank[ATTRIBUTE])


def yes_set(h: Hypothesis, qs: Sequence[Question], base) -> frozenset:
    return frozenset(q.qid for q in qs if base(q, h))


def pair_overlap(a: Hypothesis, b: Hypothesis, qs: Sequence[Question], base) -> dict:
    ya, yb = yes_set(a, qs, base), yes_set(b, qs, base)
    extra_a, extra_b = len(yb - ya), len(ya - yb)   # contradictions a / b would suffer
    return {"same_category": a.category == b.category,
            "d_a": extra_a, "d_b": extra_b, "d_min": min(extra_a, extra_b),
            "d_max": max(extra_a, extra_b),
            "jaccard": len(ya & yb) / max(1, len(ya | yb))}


# --------------------------------------------------------------------------- #
# Arm S: today's single-cause P0a                                              #
# --------------------------------------------------------------------------- #

def play_s(truths: Sequence[Hypothesis], known: Sequence[Hypothesis], matrix: Matrix,
           oracle: KeyedSetOracle, rep: int = 0, p_unknown: float = P_UNKNOWN) -> dict:
    """One game of today's loop. ``truths`` has one member (single / novel game) or
    two (compound game). The episode is ``run_openworld_episode`` unchanged; the
    oracle ignores the ``true_h`` it is handed and answers for the whole set."""
    oracle.begin(truths, rep)
    all_h = list(known) + [UNKNOWN_HYP]
    by_name = {h.name: h for h in all_h}
    like = make_openworld_likelihood(matrix, known, eps=P0A["eps"])
    soft = SoftPosterior(known, by_name, like, UNKNOWN, p_unknown, P0A["eps"])
    tr = {"exhaust_turns": [], "live": [], "soft_pu": [], "revives": 0}

    def obs(turn, q, ans, bk, surv_before):
        # A live set that empties on this answer is an exhaustion event, whether or
        # not a reinstate then revives a candidate.
        revived = bk.n_revives > tr["revives"]
        tr["revives"] = bk.n_revives
        if surv_before and (not bk.survivors or revived):
            tr["exhaust_turns"].append(turn)
        tr["live"].append(len(bk.survivors))
        soft.observe(q, ans)
        tr["soft_pu"].append(round(soft.distribution().get(UNKNOWN, 0.0), 6))

    ep = run_openworld_episode(truths[0], known, matrix, p_unknown, oracle,
                               max_questions=P0A["max_probes"],
                               confidence=P0A["confidence"], eps=P0A["eps"],
                               backend="modeswitch_guard", tolerance=P0A["tolerance"],
                               no_repeat=True, end_rule="always", observer=obs)
    names = {t.name for t in truths}
    steps = [[s.kind[0], s.qid.split(":", 1)[1], bool(s.answer_yes)] for s in ep.steps]
    named = [k for kind, k, a in steps if kind == "g" and a]
    wrong = sum(1 for kind, k, a in steps if kind == "g" and not a)
    if named:
        end = "named"
    elif ep.end_budget:
        end = "budget"
    else:
        end = "exhausted"
    dist = soft.distribution()
    soft_top = max(dist, key=lambda n: dist[n])
    pu = tr["soft_pu"]
    fire = next((i + 1 for i, v in enumerate(pu) if v >= P0A["confidence"]), None)
    assert all(n in names for n in named), "an exact oracle confirmed a non-secret"
    return {"truths": sorted(names), "kind": None, "end": end, "named": named,
            "n_probes": len(steps), "wrong": wrong, "reinstates": ep.n_retractions,
            "flips": oracle.n_flips, "exhaust_turns": tr["exhaust_turns"],
            "soft_fire_turn": fire, "soft_pu_max": max(pu) if pu else p_unknown,
            "soft_pu_final": pu[-1] if pu else p_unknown,
            "soft_top_final": soft_top, "steps": steps, "live": tr["live"]}


# --------------------------------------------------------------------------- #
# Decision scoring                                                             #
# --------------------------------------------------------------------------- #

def score(rec: dict, c_expert: float, c_wrong: float, verified: bool = True) -> dict:
    """Total cost of one game. Every by-name guess is an action (FLYW-1's convention);
    a "no" is a wrong action. A game is resolved when every secret it holds has been
    named. An unresolved game ends with one expert call: after an exhaustion or the
    budget end (today's escalation), or -- in a compound game that named one member --
    when the verification check shows the incident has not cleared (``verified``). With
    ``verified=False`` (sensitivity) a partial answer is a silent miss charged
    ``c_wrong`` instead of an expert call."""
    resolved = len(rec["named"]) == len(rec["truths"])
    partial = 0 < len(rec["named"]) < len(rec["truths"])
    expert = 0 if resolved else 1
    miss = 0
    if partial and not verified:
        expert, miss = 0, 1
    cost = rec["n_probes"] + c_wrong * (rec["wrong"] + miss) + c_expert * expert
    return {"cost": cost, "expert": expert, "resolved": resolved, "partial": partial}


def score_class(rec: dict, cls: Dict[str, object], c_expert: float,
                c_wrong: float) -> dict:
    """Reported reading 1: decisions on a class map (``cat17``: the category is the
    remediation), FLYW-1's 'Scoring A as a decision'. The game is cut at the first
    by-name guess in a truth's class (a right action); earlier guesses in other
    classes are wrong actions. If the truths span two classes, verification then
    shows the incident has not cleared and the expert is called."""
    tc = {cls[t] for t in rec["truths"]}
    wrong = 0
    for i, (kind, key, _a) in enumerate(rec["steps"], start=1):
        if kind == "g":
            if cls[key] in tc:
                unresolved = len(tc) > 1
                return {"cost": i + c_wrong * wrong + c_expert * unresolved,
                        "wrong": wrong, "right": True}
            wrong += 1
    return {"cost": rec["n_probes"] + c_wrong * wrong + c_expert, "wrong": wrong,
            "right": False}


def score_diagnostic(rec: dict, c_expert: float) -> float:
    """Reported reading 2: a by-name guess is an exact *diagnostic* (one probe), not a
    remediation, so there are no wrong actions; cost = probes + c_expert if any truth
    is left unnamed."""
    return rec["n_probes"] + c_expert * (len(rec["named"]) < len(rec["truths"]))


# --------------------------------------------------------------------------- #
# Gap 10: novel vs error vs compound                                           #
# --------------------------------------------------------------------------- #

class ThreeWay:
    """Posterior over the three explanations of an escalation, from the answers seen.

    * single ("error"): mean over known objects of the product of P0a's ε-clamped
      likelihoods (a flip-struck truth is a known object with a contradiction);
    * compound: mean over known pairs, with the pair's likelihood of "yes" the larger
      of its members' (the OR at the oracle's threshold), ε-clamped;
    * novel: the catch-all's background likelihood (P0a's ``UNKNOWN``).

    Prior: single (1 − w)(1 − ν), compound w, novel (1 − w) ν. The means make each
    term the class's share of an object-level posterior with a uniform prior inside
    the class (FLYW-1's T-loss argument)."""

    def __init__(self, known: Sequence[Hypothesis], matrix: Matrix,
                 eps: float = P0A["eps"]):
        self.names = [h.name for h in known]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.cat = np.array([h.category for h in known])
        self.matrix = matrix
        self.eps = eps
        self.attr_bg = background_marginals(matrix, self.names)
        self.cat_bg = category_marginals(known)
        n = len(self.names)
        self.pi, self.pj = np.triu_indices(n, k=1)

    def _p_yes(self, kind: str, key: str) -> Tuple[np.ndarray, float]:
        eps = self.eps
        if kind == "a":
            p = np.array([self.matrix[key][n] for n in self.names])
            bg = self.attr_bg[key]
        elif kind == "c":
            p = (self.cat == key).astype(float)
            bg = self.cat_bg.get(key, 0.0)
        else:  # by-name guess
            p = np.array([1.0 if n == key else 0.0 for n in self.names])
            bg = 0.0
        return p, min(max(bg, eps), 1 - eps)

    def log_evidence(self, steps: Sequence[Sequence]) -> Dict[str, float]:
        eps = self.eps
        ls = np.zeros(len(self.names))
        lp = np.zeros(len(self.pi))
        lb = 0.0
        for kind, key, ans in steps:
            p, bg = self._p_yes(kind, key)
            pp = np.maximum(p[self.pi], p[self.pj])
            ps = np.clip(p, eps, 1 - eps)
            pp = np.clip(pp, eps, 1 - eps)
            if ans:
                ls += np.log(ps)
                lp += np.log(pp)
                lb += math.log(bg)
            else:
                ls += np.log1p(-ps)
                lp += np.log1p(-pp)
                lb += math.log(1 - bg)

        def lmean(v):
            m = v.max()
            return float(m + math.log(np.exp(v - m).mean()))

        return {"single": lmean(ls), "compound": lmean(lp), "novel": lb}

    @staticmethod
    def posterior(ev: Dict[str, float], w: float, nu: float) -> Dict[str, float]:
        prior = {"single": (1 - w) * (1 - nu), "compound": w, "novel": (1 - w) * nu}
        lg = {k: ev[k] + math.log(prior[k]) for k in prior if prior[k] > 0}
        m = max(lg.values())
        z = sum(math.exp(v - m) for v in lg.values())
        return {k: (math.exp(lg[k] - m) / z if k in lg else 0.0) for k in prior}


def pairs_of(names: Sequence[str]) -> List[Tuple[str, str]]:
    return list(itertools.combinations(names, 2))
