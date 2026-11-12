"""Swappable belief backends for the open-world Twenty Questions loop.

The §3.11 loop hard-coded one belief representation: a soft, ε-floored Bayesian
posterior. This module factors belief behind a small interface so the EIG planner,
the oracle, the catch-all, and the accumulation loop are all shared across arms, and
only the belief *update* differs. That is what makes the retraction study
(`OPENWORLD_RETRACTION_PREREG.md`) a clean ablation.

A backend owns exactly three things:

* ``distribution()`` — belief over ``H_known ∪ {UNKNOWN}``, which the planner reads
  for EIG and the loop reads for ``p_unknown`` and logging;
* ``decision(confidence)`` — ``ask`` / ``guess(name)`` / ``expert``, so each backend
  carries its own commit rule (a distribution gate for soft; a survivor-count gate
  for hard elimination, which can rule *out* but not *in* against the catch-all);
* ``observe(question, answer)`` — the belief update.

Backends import only from ``bed`` (the likelihood callable and the UNKNOWN token are
injected), so there is no import cycle with ``openworld``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

from .bed import (GUESS, Hypothesis, LikelihoodModel, Posterior, Question,
                  posterior_update)

# A decision is ('ask', None) | ('guess', object_name) | ('expert', None).
Decision = Tuple[str, Optional[str]]


class BeliefBackend(ABC):
    """Belief over the known objects plus one catch-all ``UNKNOWN`` seat."""

    name = "base"

    @abstractmethod
    def distribution(self) -> Posterior:
        """Normalized belief over known names + the UNKNOWN token."""

    @abstractmethod
    def decision(self, confidence: float) -> Decision:
        """Whether to ask another question, guess a named object, or call the
        expert — each backend's own commit rule."""

    @abstractmethod
    def observe(self, q: Question, answer_yes: bool) -> None:
        """Fold one answer into the belief."""


def _in_order(names: Sequence[str], members: set) -> List[str]:
    """``members`` in the fixed ``known_hyps`` order. The survivor-set backends keep
    membership in a ``set`` of names, whose iteration order follows Python's string
    hash (``PYTHONHASHSEED``); every iteration that can feed a tie-break (the
    distribution's key order, read by ``max`` and by ``candidate_questions``' stable
    sort) goes through this instead, so a run does not depend on the hash seed."""
    return [n for n in names if n in members]


def _uniform_known_prior(known_hyps: Sequence[Hypothesis], unknown: str,
                         p_unknown_prior: float) -> Posterior:
    post: Posterior = {}
    k = len(known_hyps)
    rest = 1.0 - p_unknown_prior
    for h in known_hyps:
        post[h.name] = rest / k if k else 0.0
    post[unknown] = p_unknown_prior if k else 1.0
    return post


class SoftPosterior(BeliefBackend):
    """The §3.11 belief: an ε-floored Bayesian filter that never hard-eliminates.

    Behaviour-preserving — a loop driven by this backend reproduces the committed
    open-world run exactly (it is the refactor's regression gate)."""

    name = "soft"

    def __init__(self, known_hyps: Sequence[Hypothesis],
                 by_name: Dict[str, Hypothesis], like: LikelihoodModel,
                 unknown: str, p_unknown_prior: float, eps: float = 0.05):
        self.by_name = by_name
        self.like = like
        self.U = unknown
        self.eps = eps
        self.post = _uniform_known_prior(known_hyps, unknown, p_unknown_prior)

    def distribution(self) -> Posterior:
        return self.post

    def decision(self, confidence: float) -> Decision:
        support = sum(1 for p in self.post.values() if p > 1e-9)
        top = max(self.post, key=lambda n: self.post[n])
        if self.post[top] >= confidence or support == 1:
            return ("expert", None) if top == self.U else ("guess", top)
        return ("ask", None)

    def observe(self, q: Question, answer_yes: bool) -> None:
        self.post = posterior_update(self.post, q, answer_yes, self.by_name,
                                     self.like, eps=self.eps)


class HardMonotone(BeliefBackend):
    """Monotone hard elimination — the CMBS baseline stance. A known object is
    struck when an answer contradicts it (observed-answer likelihood below
    ``kill``), and never returns.

    Hard elimination can rule a hypothesis *out* but not *in*: the catch-all
    ``UNKNOWN`` has an un-contradictable background likelihood, so it never dies.
    Therefore a known object is committed only when it is the sole surviving known,
    and the expert fires when the surviving knowns empty or the Pitman–Yor
    ``UNKNOWN`` mass dominates. The distribution is uniform over survivors for the
    known mass; ``UNKNOWN`` carries the PY mass (1.0 once the knowns are gone)."""

    name = "hard"

    def __init__(self, known_hyps: Sequence[Hypothesis],
                 by_name: Dict[str, Hypothesis], like: LikelihoodModel,
                 unknown: str, p_unknown_prior: float, eps: float = 0.05,
                 kill: float = 0.5):
        self.by_name = by_name
        self.like = like
        self.U = unknown
        self.u_prior = p_unknown_prior
        self.kill = kill
        self.names = [h.name for h in known_hyps]
        self.survivors = set(self.names)

    def _u(self) -> float:
        return 1.0 if not self.survivors else self.u_prior

    def distribution(self) -> Posterior:
        u = self._u()
        dist: Posterior = {}
        if self.survivors:
            share = (1.0 - u) / len(self.survivors)
            for n in _in_order(self.names, self.survivors):
                dist[n] = share
        dist[self.U] = u
        return dist

    def decision(self, confidence: float) -> Decision:
        if not self.survivors or self._u() >= confidence:
            return ("expert", None)
        if len(self.survivors) == 1:
            return ("guess", next(iter(self.survivors)))
        dist = self.distribution()
        top = max(dist, key=lambda n: dist[n])
        if top != self.U and dist[top] >= confidence:
            return ("guess", top)
        return ("ask", None)

    def observe(self, q: Question, answer_yes: bool) -> None:
        killed = set()
        for n in _in_order(self.names, self.survivors):
            p_yes = self.like(q, self.by_name[n])
            p_obs = p_yes if answer_yes else (1.0 - p_yes)
            if p_obs < self.kill:
                killed.add(n)
        self.survivors -= killed


class RetractingElimination(BeliefBackend):
    """Hard elimination made reversible — the reference ``retract`` backend, and the
    measurable core of the CMBS un-eliminate op.

    The **strike** rule is exactly ``hard``'s: a surviving known is struck the moment
    an answer contradicts it (observed-answer likelihood below ``kill``). This is what
    gives the novelty detector its teeth — a hard, absolute rule empties the survivor
    set for an out-of-pool secret, which a margin-relative rule (which always keeps
    the least-bad known) cannot. So on a trusted oracle ``retract`` behaves like
    ``hard`` exactly: no answer ever contradicts the truth, nothing is revived.

    What ``retract`` adds is **revival**. A running ε-floored log-likelihood
    ``logL[h]`` is kept for every known and for the catch-all, struck or not, so no
    information is lost. A struck known is un-eliminated — a logged *revive* — once
    its accumulated evidence recovers to within ``margin`` bits of the best live
    hypothesis (a surviving known or the catch-all). Under a noisy answerer a single
    flip strikes the true object, but the flip is bounded (the ε floor caps it near
    ``log2 ε`` bits) and the truth is otherwise the best-supported hypothesis, so it
    is revived; a genuinely wrong known, contradicted on its merits, stays low and is
    not. A hypothesis struck *this* turn is not revived until the next, so a kill and
    its revival never thrash in place. The whole state replays from the (question,
    answer) sequence.

    ``margin`` governs revival only: larger revives more readily (more recovery, more
    ops), smaller is stricter. Clean behaviour is ``hard``'s for any margin."""

    name = "retract"

    def __init__(self, known_hyps: Sequence[Hypothesis],
                 by_name: Dict[str, Hypothesis], like: LikelihoodModel,
                 unknown: str, p_unknown_prior: float, eps: float = 0.05,
                 margin: float = 2.0, kill: float = 0.5):
        self.by_name = by_name
        self.like = like
        self.U = unknown
        self.u_prior = p_unknown_prior
        self.eps = eps
        self.margin = margin
        self.kill = kill
        self.names = [h.name for h in known_hyps]
        self.logL: Dict[str, float] = {n: 0.0 for n in self.names}
        self.logL[unknown] = 0.0            # the catch-all runs in the same race
        self.survivors = set(self.names)
        self.struck: set = set()
        self.ops: List[Tuple[str, str]] = []   # append-only ('kill'|'revive', name)
        self.n_kills = 0
        self.n_revives = 0

    def _p_obs(self, name: str, answer_yes: bool) -> float:
        p_yes = self.like(self._q, self.by_name[name])
        return p_yes if answer_yes else (1.0 - p_yes)

    def _best_alive(self) -> float:
        vals = [self.logL[n] for n in self.survivors] + [self.logL[self.U]]
        return max(vals)

    def observe(self, q: Question, answer_yes: bool) -> None:
        self._q = q
        # 1. accumulate ε-floored log-likelihood for every known and the catch-all.
        for n in list(self.names) + [self.U]:
            p = min(max(self._p_obs(n, answer_yes), self.eps), 1.0 - self.eps)
            self.logL[n] += math.log2(p)
        # 2. strike (absolute, hard rule): a surviving known contradicted this answer.
        new_strikes = {n for n in self.survivors
                       if self._p_obs(n, answer_yes) < self.kill}
        for n in _in_order(self.names, new_strikes):
            self.ops.append(("kill", n))
            self.n_kills += 1
        self.survivors -= new_strikes
        self.struck |= new_strikes
        # 3. revive previously-struck knowns whose evidence has recovered (not those
        #    struck this turn -- a kill and its revival never thrash in place).
        best = self._best_alive()
        revived = {n for n in (self.struck - new_strikes)
                   if self.logL[n] >= best - self.margin}
        for n in _in_order(self.names, revived):
            self.ops.append(("revive", n))
            self.n_revives += 1
        self.struck -= revived
        self.survivors |= revived

    def _u(self) -> float:
        # Hard's detector exactly: the catch-all carries the PY mass while a known
        # survives, and all of it once the knowns are struck. So novelty detection
        # and commit are identical to `hard`, and revival is the *only* difference.
        return 1.0 if not self.survivors else self.u_prior

    def distribution(self) -> Posterior:
        u = self._u()
        dist: Posterior = {}
        if self.survivors:
            share = (1.0 - u) / len(self.survivors)
            for n in _in_order(self.names, self.survivors):
                dist[n] = share
        dist[self.U] = u
        return dist

    def decision(self, confidence: float) -> Decision:
        if not self.survivors or self._u() >= confidence:
            return ("expert", None)
        if len(self.survivors) == 1:
            return ("guess", next(iter(self.survivors)))
        dist = self.distribution()
        top = max(dist, key=lambda n: dist[n])
        if top != self.U and dist[top] >= confidence:
            return ("guess", top)
        return ("ask", None)


class TombstoneRetraction(BeliefBackend):
    """CMBS v0.2.0-aligned structured retraction: per-killer tombstones.

    Where ``retract`` scored each hypothesis with a scalar cumulative log-likelihood
    and revived on a margin, this backend mirrors CMBS's mask-meet elimination and
    its reinstate op. Each answer is a mask that eliminates the survivors inconsistent
    with it. The structure CMBS's op-log keeps — *which* answers eliminated each
    hypothesis — is tracked here as a per-hypothesis set of **distinct killers** (the
    question ids that contradicted it). A hypothesis is asserted dead (tombstoned)
    only when its distinct-killer count exceeds ``tolerance``; below that, each
    eliminating answer is voided by a logged reinstate — tolerance is a reinstate
    policy keyed to the killer record, which the default monotone reducer lacks.

    This is the per-killer structure the scalar margin could not express: killer
    *count* separates a truth struck by one outlier answer (1 killer, kept at T≥1)
    from a wrong contradicted on its merits (many distinct killers, tombstoned) — the
    distinction a cumulative score blurs. Detection is ``hard``'s: a novel secret
    contradicts every known on many questions, so all cross tolerance, the survivors
    empty, and the catch-all fires the expert.

    ``tolerance`` is the one knob: 0 reproduces ``hard``; higher forgives more
    outliers."""

    name = "tombstone"

    def __init__(self, known_hyps: Sequence[Hypothesis],
                 by_name: Dict[str, Hypothesis], like: LikelihoodModel,
                 unknown: str, p_unknown_prior: float, eps: float = 0.05,
                 tolerance: int = 1, kill: float = 0.5):
        self.by_name = by_name
        self.like = like
        self.U = unknown
        self.u_prior = p_unknown_prior
        self.kill = kill
        self.R = tolerance        # reinstate iff a tombstoned known has <= R killers
        self.names = [h.name for h in known_hyps]
        self.killers: Dict[str, set] = {n: set() for n in self.names}
        self.live = set(self.names)
        self.tombstoned: set = set()
        self.reinstated_once: set = set()
        self.n_kills = 0
        self.n_revives = 0        # reinstate (un-eliminate) ops

    @property
    def survivors(self) -> set:
        return self.live

    def _u(self) -> float:
        return 1.0 if not self.live else self.u_prior

    def distribution(self) -> Posterior:
        u = 1.0 if not self.live else self.u_prior
        dist: Posterior = {}
        if self.live:
            share = (1.0 - u) / len(self.live)
            for n in _in_order(self.names, self.live):
                dist[n] = share
        dist[self.U] = u
        return dist

    def decision(self, confidence: float) -> Decision:
        # Hard's detector and commit: a novel secret strikes every known (and the
        # flip victims cannot be reinstated, below), so the live set empties and the
        # catch-all fires the expert; an in-pool game collapses to the sole survivor.
        if not self.live or self.u_prior >= confidence:
            return ("expert", None)
        if len(self.live) == 1:
            return ("guess", next(iter(self.live)))
        dist = self.distribution()
        top = max(dist, key=lambda n: dist[n])
        if top != self.U and dist[top] >= confidence:
            return ("guess", top)
        return ("ask", None)

    def observe(self, q: Question, answer_yes: bool) -> None:
        qid = q.qid
        # Strike like hard (first contradiction removes), but keep accruing killers
        # for tombstoned knowns too, so a hypothesis contradicted again while dead
        # (an ordinary wrong, or a novel-secret mismatch) climbs past the reinstate
        # bar while a flip-struck truth (consistent after its one killer) stays at it.
        for n in self.names:
            p_yes = self.like(q, self.by_name[n])
            p_obs = p_yes if answer_yes else (1.0 - p_yes)
            if p_obs < self.kill:
                # record the distinct killer (the set dedupes) for reinstate
                # eligibility, but strike on *any* contradiction so a reinstated known
                # stays killable by the very question that first eliminated it.
                self.killers[n].add(qid)
                if n in self.live:
                    self.live.discard(n)
                    self.tombstoned.add(n)
                    self.n_kills += 1
        # Reinstate, keyed to the killer record: when the live set would empty,
        # revive the fewest-killer tombstoned known that still has <= R killers and
        # has not been reinstated before. A flip victim (1 killer, otherwise
        # consistent) comes back; a genuinely-wrong known (many killers) does not.
        if not self.live:
            cands = sorted((len(self.killers[n]), n) for n in self.tombstoned
                           if n not in self.reinstated_once
                           and len(self.killers[n]) <= self.R)
            if cands:
                _, best = cands[0]
                self.live.add(best)
                self.tombstoned.discard(best)
                self.reinstated_once.add(best)
                self.n_revives += 1


class ModeSwitchedRetraction(BeliefBackend):
    """The mode switch: `hard`'s elimination and `hard`'s escalation *view*, with a
    bounded reinstate-and-**test** for resolution only.

    `tombstone` failed by reinstating at the escalation boundary without bound and
    letting the revived hypothesis stall as an unkillable zombie, so novel secrets'
    wrongs were guessed and the detector, which reads the empty set, never fired. This
    backend keeps the two jobs apart:

    * **Escalation / detection stays hard.** The live set is hard-monotone; when it is
      empty and the reinstate budget is spent, the catch-all fires the expert exactly
      as `hard`, so the novelty signal is `hard`'s — delayed by at most the budget.
    * **Reinstate is a bounded hypothesis *test*.** When the live set would empty, up
      to `tolerance` (=K) times per game, revive the fewest-killer single-contradiction
      candidate (≤ 1 killer, each known once). A revived candidate is the sole
      survivor, so the loop tests it with a by-name guess, which is exact and not
      subject to attribute noise: a flip-struck truth answers *yes* and is recovered
      (a won game, not a false escalation); a revived wrong answers *no*, is re-struck,
      and the next candidate (or the expert) follows. After K failed tests the set
      stays empty and the expert fires. The budget is what bounds the detector cost —
      a novel secret escalates after K probes instead of being chased forever."""

    name = "modeswitch"

    def __init__(self, known_hyps: Sequence[Hypothesis],
                 by_name: Dict[str, Hypothesis], like: LikelihoodModel,
                 unknown: str, p_unknown_prior: float, eps: float = 0.05,
                 tolerance: int = 3, kill: float = 0.5, budget_guard: bool = False):
        self.by_name = by_name
        self.like = like
        self.U = unknown
        self.u_prior = p_unknown_prior
        self.kill = kill
        self.K = tolerance        # max reinstate-and-test attempts per game (budget)
        # Budget guard (opt-in; ``modeswitch_guard``): reinstate only when a turn is
        # left to test the candidate. The episode loop sets ``turns_left`` before each
        # ``observe``; None (a caller that never sets it) means unguarded.
        self.budget_guard = budget_guard
        self.turns_left: Optional[int] = None
        self.names = [h.name for h in known_hyps]
        self.killers: Dict[str, set] = {n: set() for n in self.names}
        self.live = set(self.names)
        self.tombstoned: set = set()
        self.reinstated_once: set = set()
        self.n_kills = 0
        self.n_revives = 0

    @property
    def survivors(self) -> set:
        return self.live

    def _u(self) -> float:
        return 1.0 if not self.live else self.u_prior

    def distribution(self) -> Posterior:
        u = 1.0 if not self.live else self.u_prior
        dist: Posterior = {}
        if self.live:
            share = (1.0 - u) / len(self.live)
            for n in _in_order(self.names, self.live):
                dist[n] = share
        dist[self.U] = u
        return dist

    def decision(self, confidence: float) -> Decision:
        # Hard's decision: a sole live known is guessed directly. A reinstated
        # candidate is a sole survivor, so it is tested by a by-name guess next turn.
        if not self.live or self.u_prior >= confidence:
            return ("expert", None)
        if len(self.live) == 1:
            return ("guess", next(iter(self.live)))
        dist = self.distribution()
        top = max(dist, key=lambda n: dist[n])
        if top != self.U and dist[top] >= confidence:
            return ("guess", top)
        return ("ask", None)

    def observe(self, q: Question, answer_yes: bool) -> None:
        qid = q.qid

        def contradicts(n: str) -> bool:
            p_yes = self.like(q, self.by_name[n])
            p_obs = p_yes if answer_yes else (1.0 - p_yes)
            return p_obs < self.kill

        # Hard strike (on any contradiction, so a reinstated known stays killable).
        for n in self.names:
            if contradicts(n):
                self.killers[n].add(qid)
                if n in self.live:
                    self.live.discard(n)
                    self.tombstoned.add(n)
                    self.n_kills += 1
        # Reinstate only at the escalation boundary, bounded to K attempts per game:
        # revive the fewest-killer single-contradiction candidate. It is a sole
        # survivor, so the loop immediately *tests* it with a by-name guess (exact,
        # un-noisy) -- a flip-struck truth answers yes and wins; a revived wrong
        # answers no and is re-struck. After K failed tests the set stays empty and
        # the catch-all escalates, so the detector is hard's (just K turns later).
        # With the budget guard, a reinstate on the final answer is skipped: there is
        # no turn left to test the candidate, so the set stays empty and the game
        # escalates, as ``hard`` would, instead of ending on an untested known.
        if (self.budget_guard and self.turns_left is not None
                and self.turns_left < 1):
            return
        if not self.live and self.n_revives < self.K:
            cands = sorted((len(self.killers[n]), n) for n in self.tombstoned
                           if n not in self.reinstated_once
                           and len(self.killers[n]) <= 1)
            if cands:
                _, best = cands[0]
                self.live.add(best)
                self.tombstoned.discard(best)
                self.reinstated_once.add(best)
                self.n_revives += 1


class ModeSwitchGuarded(ModeSwitchedRetraction):
    """``modeswitch`` with the budget guard on: a reinstate that no turn remains to
    test is skipped, so the game escalates instead of ending on an untested,
    reinstated known (the *reinstate overrun* of ``STRESS_ST1_RESULTS.md`` §4). The
    guard acts only on the last answer of a game, so it never changes which
    questions are asked; it can only turn a budget-end ``wrong`` into ``expert``.

    This is the P0a production backend (``P0A_BACKEND``; adopted 2026-11-12 after the
    gates of ``STRESS_REINSTATE_GUARD.md``). ``modeswitch`` stays unguarded as the
    reference that every committed ``modeswitch`` result regenerates with."""

    name = "modeswitch_guard"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("budget_guard", True)
        super().__init__(*args, **kwargs)


# Registry the loop looks a backend up by name.
BACKENDS: Dict[str, type] = {
    "soft": SoftPosterior,
    "hard": HardMonotone,
    "retract": RetractingElimination,
    "tombstone": TombstoneRetraction,
    "modeswitch": ModeSwitchedRetraction,
    "modeswitch_guard": ModeSwitchGuarded,
}

# The production rule P0a (PRODUCTION_RECOMMENDATIONS.md): hard elimination with a
# bounded, budget-aware reinstate-and-test. New work that means "P0a" selects this;
# preregistered studies keep the backend names they registered.
P0A_BACKEND = "modeswitch_guard"
