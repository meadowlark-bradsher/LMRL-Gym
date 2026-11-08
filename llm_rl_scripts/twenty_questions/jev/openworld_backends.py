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

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

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
        self.survivors = {h.name for h in known_hyps}

    def _u(self) -> float:
        return 1.0 if not self.survivors else self.u_prior

    def distribution(self) -> Posterior:
        u = self._u()
        dist: Posterior = {}
        if self.survivors:
            share = (1.0 - u) / len(self.survivors)
            for n in self.survivors:
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
        for n in self.survivors:
            p_yes = self.like(q, self.by_name[n])
            p_obs = p_yes if answer_yes else (1.0 - p_yes)
            if p_obs < self.kill:
                killed.add(n)
        self.survivors -= killed


# Registry the loop looks a backend up by name. ``retract`` (the CMBS-aligned
# reversible backend) is added in the next step; its extension point is here.
BACKENDS: Dict[str, type] = {
    "soft": SoftPosterior,
    "hard": HardMonotone,
}
