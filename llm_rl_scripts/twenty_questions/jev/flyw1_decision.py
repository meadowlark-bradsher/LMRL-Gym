"""FLYW-1 game engine: aim at a decision, not an identity (``FLYW1_DECISION_PREREG.md``).

Shared by the runner (``flyw1_run.py``) and the analyzer (``analyze_flyw1.py``):

* ``keyed_oracle`` -- the matrix oracle with *paired* attribute flips: whether the
  answer to question q in game g under noise seed k is flipped is a hash of
  (k, g, q), so every arm that asks q in game g gets the same answer.
* ``play_a`` -- arm A, today's P0a game (``run_openworld_episode``, unchanged), as a
  compact record that the analyzer scores against any class map.
* ``play_class_arm`` -- arms B, C and D: the P0a belief (``modeswitch_guard``) with
  the class stopping rule, exact class actions and the arm's question score.
* ``class_belief`` / ``tloss_continuation`` -- Part 2's expected-loss trigger,
  evaluated at the points where today's rule calls the expert. The game is identical
  up to that point under every trigger, so one trajectory serves T-mass, T-loss at
  every cost ratio, the max-likelihood shadow and the p_U sensitivity values.
* ``holdouts`` -- Part 2's rotated hold-outs.

**Clarification of one prereg rule (applied to B, C and D alike).** "No question
scores above zero" is read on P0a's hard belief: a question *splits* the live set when
some answer would strike at least one live object and leave at least one (P0a's kill
rule strikes an object whose likelihood of the answer is below 0.5). Only splitting
questions are candidates; a non-splitting question
cannot change the live set, and the graded EIG can still be slightly positive on one,
which would spend the budget on unresolvable objects. When no candidate splits, the
arm acts on the class with the most live members (ties: lowest id).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .bed import (ATTRIBUTE, CATEGORY, GUESS, Hypothesis, OracleModel, Question,
                  expected_info_gain, matrix_oracle)
from .openworld import (UNKNOWN, UNKNOWN_HYP, build_openworld_bank,
                        make_openworld_likelihood, run_openworld_episode)
from .openworld_backends import ModeSwitchGuarded, _in_order

Matrix = Dict[str, Dict[str, float]]
ClassMap = Dict[str, int]

# The prereg's fixed P0a settings.
P0A = dict(max_probes=20, confidence=0.9, eps=0.05, tolerance=3)
TLOSS_MAX_ACTIONS = 2


# --------------------------------------------------------------------------- #
# Keyed (paired) noise                                                         #
# --------------------------------------------------------------------------- #

def keyed_uniform(seed: int, game_key: str, qid: str) -> float:
    h = hashlib.sha256("flyw1|{}|{}|{}".format(seed, game_key, qid).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


class KeyedOracle:
    """Matrix oracle (threshold 0.5) with attribute flips keyed on (seed, game, q).
    Category and guess answers stay exact. ``begin(game_key)`` names the game;
    a game key is the secret's name plus a repeat index."""

    def __init__(self, matrix: Matrix, flip_p: float, seed: int):
        self.base = matrix_oracle(matrix)
        self.flip_p = flip_p
        self.seed = seed
        self.game_key = ""
        self.n_flips = 0

    def begin(self, game_key: str) -> None:
        self.game_key = game_key
        self.n_flips = 0

    def __call__(self, q: Question, true_h: Hypothesis) -> bool:
        ans = self.base(q, true_h)
        if (q.kind == ATTRIBUTE and self.flip_p > 0.0
                and keyed_uniform(self.seed, self.game_key, q.qid) < self.flip_p):
            self.n_flips += 1
            return not ans
        return ans


# --------------------------------------------------------------------------- #
# Arm A                                                                        #
# --------------------------------------------------------------------------- #

def play_a(true_h: Hypothesis, known: Sequence[Hypothesis], matrix: Matrix,
           oracle: KeyedOracle, p_unknown_prior: float) -> dict:
    """Today's P0a game. Steps are kept as (kind, key) so any class map can score
    it: the first guess in the secret's class is the right action."""
    live_sizes: List[int] = []

    def obs(turn, q, ans, bk, surv_before):
        live_sizes.append(len(bk.survivors))

    ep = run_openworld_episode(true_h, known, matrix, p_unknown_prior, oracle,
                               max_questions=P0A["max_probes"],
                               confidence=P0A["confidence"], eps=P0A["eps"],
                               backend="modeswitch_guard", tolerance=P0A["tolerance"],
                               no_repeat=True, end_rule="always", observer=obs)
    return {"arm": "A", "secret": true_h.name, "outcome": ep.outcome,
            "n_probes": len(ep.steps), "reinstates": ep.n_retractions,
            "flips": oracle.n_flips,
            "steps": [[s.kind[0], s.qid.split(":", 1)[1]] for s in ep.steps],
            "live": live_sizes}


def score_a_record(rec: dict, cls: ClassMap) -> dict:
    """Decision scoring of an A record on one map (prereg: 'Scoring A as a
    decision'). A guess of an object in the secret's class ends the game right."""
    target = cls.get(rec["secret"])
    wrong = 0
    for i, (kind, key) in enumerate(rec["steps"], start=1):
        if kind == "g":
            if cls.get(key) == target:
                return {"probes": i, "actions": wrong + 1, "wrong": wrong,
                        "questions": i - wrong - 1, "outcome": "right"}
            wrong += 1
    n = rec["n_probes"]
    return {"probes": n, "actions": wrong, "wrong": wrong, "questions": n - wrong,
            "outcome": "expert"}


# --------------------------------------------------------------------------- #
# Arms B, C, D                                                                 #
# --------------------------------------------------------------------------- #

def _entropy(masses: Sequence[float]) -> float:
    t = sum(masses)
    if t <= 0:
        return 0.0
    return -sum(m / t * math.log2(m / t) for m in masses if m > 0)


def class_ig(q: Question, dist: Dict[str, float], by_name, like, cls: ClassMap) -> float:
    """Arm C: information about the class of the secret (UNKNOWN is its own class)."""
    def key(n):
        return "U" if n == UNKNOWN else cls[n]
    prior: Dict[object, float] = Counter()
    yes: Dict[object, float] = Counter()
    for n, w in dist.items():
        if w <= 0:
            continue
        ly = like(q, by_name[n])
        prior[key(n)] += w
        yes[key(n)] += w * ly
    p_yes = sum(yes.values())
    p_no = sum(prior.values()) - p_yes
    if p_yes <= 0 or p_no <= 0:
        return 0.0
    no = {k: prior[k] - yes[k] for k in prior}
    return _entropy(list(prior.values())) - (p_yes * _entropy(list(yes.values()))
                                             + p_no * _entropy(list(no.values())))


def _edges(counts: Counter) -> float:
    n = sum(counts.values())
    return (n * n - sum(v * v for v in counts.values())) / 2.0


def ec2_score(pred_yes: Dict[str, bool], live: Sequence[str], cls: ClassMap) -> float:
    """Arm D: EC^2 on the live set, uniform weights, deterministic predictions at the
    kill threshold. Expected weight of cross-class edges cut."""
    n = len(live)
    ys = Counter(cls[o] for o in live if pred_yes[o])
    ns = Counter(cls[o] for o in live if not pred_yes[o])
    ny = sum(ys.values())
    allc = ys + ns
    return _edges(allc) - (ny / n) * _edges(ys) - ((n - ny) / n) * _edges(ns)


class ClassGame:
    """One B/C/D game on the P0a belief. ``run()`` plays until the first trigger
    point (today's expert call: the live set is empty after the reinstate budget,
    or the probes run out) or a right action."""

    def __init__(self, arm: str, true_h: Hypothesis, known: Sequence[Hypothesis],
                 matrix: Matrix, cls: ClassMap, oracle: OracleModel,
                 p_unknown_prior: float):
        assert arm in ("B", "C", "D", "Cbin")
        self.arm = arm
        self.true_h = true_h
        self.cls = cls
        self.oracle = oracle
        self.p_u = p_unknown_prior
        self.known = list(known)
        self.like = make_openworld_likelihood(matrix, self.known, eps=P0A["eps"])
        eps = P0A["eps"]
        # C-bin (reported only): C's class objective on binarized answers, which
        # separates "class objective" from "graded likelihood" in the C-D comparison.
        self.like_bin = lambda q, h: (self.like(q, h) if h.name == UNKNOWN else
                                      (1.0 - eps if self.like(q, h) >= 0.5 else eps))
        bank = build_openworld_bank(self.known)
        self.cands: List[Question] = list(bank[CATEGORY]) + list(bank[ATTRIBUTE])
        self.by_name = {h.name: h for h in self.known + [UNKNOWN_HYP]}
        self.bk = ModeSwitchGuarded(self.known, self.by_name, self.like, UNKNOWN,
                                    p_unknown_prior, P0A["eps"],
                                    tolerance=P0A["tolerance"])
        self.members: Dict[int, List[str]] = {}
        for h in self.known:
            self.members.setdefault(cls[h.name], []).append(h.name)
        self.struck: set = set()
        self.asked: set = set()
        self.answers: List[Tuple[Question, bool]] = []
        self.probes = 0
        self.wrong = 0
        self.actions = 0
        self.fallbacks = 0
        self.live_trace: List[Tuple[int, int]] = []
        self.outcome: Optional[str] = None      # "right" | "trigger"
        self.trigger: Optional[str] = None      # "empty" | "budget"

    # -- actions ----------------------------------------------------------------
    def act(self, x: int) -> bool:
        self.probes += 1
        self.actions += 1
        if self.cls.get(self.true_h.name) == x:
            return True
        self.wrong += 1
        self.strike(x)
        return False

    def strike(self, x: int) -> None:
        """A failed action on class x: its members are out, exactly; reinstate never
        revives them. If the live set empties, P0a's bounded reinstate runs as it
        would after an answer (guarded on the probes left)."""
        bk = self.bk
        self.struck.add(x)
        for n in self.members.get(x, []):
            bk.live.discard(n)
            bk.tombstoned.discard(n)
            bk.reinstated_once.add(n)
        turns_left = P0A["max_probes"] - self.probes
        if bk.live or bk.n_revives >= bk.K or turns_left < 1:
            return
        cands = sorted((len(bk.killers[n]), n) for n in bk.tombstoned
                       if n not in bk.reinstated_once and len(bk.killers[n]) <= 1)
        if cands:
            _, best = cands[0]
            bk.live.add(best)
            bk.tombstoned.discard(best)
            bk.reinstated_once.add(best)
            bk.n_revives += 1

    # -- questions --------------------------------------------------------------
    def pick(self, live: List[str]) -> Optional[Question]:
        dist = self.bk.distribution()
        best, best_s = None, 0.0
        scored = []
        for q in self.cands:
            if q.qid in self.asked:
                continue
            ls = [self.like(q, self.by_name[n]) for n in live]
            n_lo = sum(1 for x in ls if x < 0.5)     # struck by a "yes"
            n_hi = sum(1 for x in ls if x > 0.5)     # struck by a "no"
            if not (0 < n_lo < len(ls) or 0 < n_hi < len(ls)):
                continue                      # no answer can change the live set
            pred = {n: x >= 0.5 for n, x in zip(live, ls)}
            if self.arm == "B":
                s = expected_info_gain(q, dist, self.by_name, self.like)
            elif self.arm == "C":
                s = class_ig(q, dist, self.by_name, self.like, self.cls)
            elif self.arm == "Cbin":
                s = class_ig(q, dist, self.by_name, self.like_bin, self.cls)
            else:
                s = ec2_score(pred, live, self.cls)
            scored.append((-s, q.qid, q))
        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], t[1]))
        if -scored[0][0] <= 1e-12:
            return None
        return scored[0][2]

    def run(self) -> "ClassGame":
        bk = self.bk
        while self.probes < P0A["max_probes"]:
            if bk.decision(P0A["confidence"])[0] == "expert":
                self.outcome, self.trigger = "trigger", "empty"
                return self
            live = _in_order(bk.names, bk.live)
            classes = {self.cls[n] for n in live}
            self.live_trace.append((len(live), len(classes)))
            if len(classes) == 1:
                if self.act(next(iter(classes))):
                    self.outcome = "right"
                    return self
                continue
            q = self.pick(live)
            if q is None:
                cnt = Counter(self.cls[n] for n in live)
                x = sorted(cnt, key=lambda c: (-cnt[c], c))[0]
                self.fallbacks += 1
                if self.act(x):
                    self.outcome = "right"
                    return self
                continue
            ans = self.oracle(q, self.true_h)
            self.asked.add(q.qid)
            self.answers.append((q, ans))
            self.probes += 1
            bk.turns_left = P0A["max_probes"] - self.probes
            bk.observe(q, ans)
        self.outcome, self.trigger = "trigger", "budget"
        return self

    def record(self) -> dict:
        return {"arm": self.arm, "secret": self.true_h.name, "outcome": self.outcome,
                "trigger": self.trigger, "probes": self.probes,
                "questions": self.probes - self.actions, "actions": self.actions,
                "wrong": self.wrong, "reinstates": self.bk.n_revives,
                "fallbacks": self.fallbacks, "live": self.live_trace,
                "flips": getattr(self.oracle, "n_flips", 0)}


def kill_rule_twins(matrix: Matrix, hyps: Sequence[Hypothesis]) -> Dict[str, List[str]]:
    """For each object as the secret, the other objects no noise-free answer strikes
    under P0a's kill rule (likelihood of the answer < 0.5), over every category and
    attribute question. This differs from the class maps' binarized identity tree only
    on the 17 cells that are exactly 0.5: a 0.5 object survives either answer."""
    like = make_openworld_likelihood(matrix, list(hyps), eps=P0A["eps"])
    bank = build_openworld_bank(list(hyps))
    qs = list(bank[CATEGORY]) + list(bank[ATTRIBUTE])
    base = matrix_oracle(matrix)
    out: Dict[str, List[str]] = {}
    for h in hyps:
        ans = [(q, base(q, h)) for q in qs]
        twins = []
        for o in hyps:
            if o.name == h.name:
                continue
            alive = True
            for q, a in ans:
                p = like(q, o)
                if (p if a else 1.0 - p) < 0.5:
                    alive = False
                    break
            if alive:
                twins.append(o.name)
        out[h.name] = twins
    return out


# --------------------------------------------------------------------------- #
# Part 2: class belief and the T-loss continuation                             #
# --------------------------------------------------------------------------- #

def _logsumexp(v: Sequence[float]) -> float:
    m = max(v)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in v))


def class_belief(answers: Sequence[Tuple[Question, bool]], known: Sequence[Hypothesis],
                 like, cls: ClassMap, p_u: float, struck: set, eps: float,
                 mode: str = "mean") -> Dict[object, float]:
    """P(X) over known classes plus "new" (prereg Part 2). ``mode`` "mean" is the
    registered belief; "max" is the logged shadow."""
    def loglik(h: Hypothesis) -> float:
        s = 0.0
        for q, a in answers:
            p = min(max(like(q, h), eps), 1.0 - eps)
            s += math.log(p if a else 1.0 - p)
        return s
    by_cls: Dict[int, List[float]] = {}
    for h in known:
        by_cls.setdefault(cls[h.name], []).append(loglik(h))
    n_known = len(known)
    logp: Dict[object, float] = {}
    for x, lls in by_cls.items():
        if x in struck:
            continue
        if mode == "mean":
            ll = _logsumexp(lls) - math.log(len(lls))
        else:
            ll = max(lls)
        prior = (1.0 - p_u) * len(lls) / n_known
        logp[x] = (math.log(prior) if prior > 0 else -math.inf) + ll
    logp["new"] = (math.log(p_u) if p_u > 0 else -math.inf) + loglik(UNKNOWN_HYP)
    z = _logsumexp(list(logp.values()))
    return {x: (math.exp(v - z) if z > -math.inf else 0.0) for x, v in logp.items()}


def tloss_continuation(game: ClassGame, ratio: float, p_u_belief: float,
                       mode: str = "mean") -> dict:
    """T-loss at the trigger point of ``game`` (which must have ended "trigger"):
    act on the best known class while 1 - P(X*) <= ratio, at most twice; then the
    expert. Does not mutate the game."""
    struck = set(game.struck)
    right_class = game.cls.get(game.true_h.name)
    extra_wrong = 0
    actions = 0
    p_star_seq = []
    while actions < TLOSS_MAX_ACTIONS:
        belief = class_belief(game.answers, game.known, game.like, game.cls,
                              p_u_belief, struck, P0A["eps"], mode=mode)
        knowns = [x for x in belief if x != "new"]
        if not knowns:
            break
        x_star = max(knowns, key=lambda x: (belief[x], -x))
        p_star = belief[x_star]
        p_star_seq.append(round(p_star, 4))
        if 1.0 - p_star > ratio:
            break
        actions += 1
        if x_star == right_class:
            return {"outcome": "right", "extra_actions": actions,
                    "extra_wrong": extra_wrong, "p_star": p_star_seq}
        extra_wrong += 1
        struck.add(x_star)
    return {"outcome": "expert", "extra_actions": actions, "extra_wrong": extra_wrong,
            "p_star": p_star_seq}


# --------------------------------------------------------------------------- #
# Part 2: hold-outs                                                            #
# --------------------------------------------------------------------------- #

def rotation_classes(sizes: Sequence[int], k: int) -> List[int]:
    """K = 4: every class. Otherwise the classes at the 25th/50th/75th size positions
    (sorted by size, ties by id; position f (n - 1) rounded half up)."""
    n = len(sizes)
    if k == 4:
        return list(range(n))
    order = sorted(range(n), key=lambda c: (sizes[c], c))
    return [order[int(math.floor(f * (n - 1) + 0.5))] for f in (0.25, 0.5, 0.75)]


def holdouts(map_id: str, cls: ClassMap, objs: Sequence[str], k: int) -> List[dict]:
    sizes = [sum(1 for o in objs if cls[o] == c) for c in range(max(cls.values()) + 1)]
    out = []
    for held in rotation_classes(sizes, k):
        rng = random.Random("flyw1-holdout|{}|{}".format(map_id, held))
        same: List[str] = []
        for c in range(len(sizes)):
            if c == held:
                continue
            mem = [o for o in objs if cls[o] == c]
            take = min(int(math.floor(0.1 * len(mem) + 0.5)), len(mem) - 1)
            if take > 0:
                same.extend(sorted(rng.sample(mem, take), key=objs.index))
        new = [o for o in objs if cls[o] == held]
        known = [o for o in objs if o not in set(new) | set(same)]
        out.append({"rotation": "{}_r{}".format(map_id, held), "map_id": map_id,
                    "held_class": held, "new_class_novels": new,
                    "same_class_novels": same, "known": known})
    return out
