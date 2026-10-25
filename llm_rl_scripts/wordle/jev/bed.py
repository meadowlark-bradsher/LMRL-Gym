"""Exact Bayesian experimental design for Wordle.

Wordle is the degenerate (closed-form) case of BED-LLM's loop: the latent is the
hidden word, the design is the guess, the observation is the green/yellow/black
pattern, and -- unlike 20 Questions -- the likelihood p(feedback | guess, word)
is a *deterministic* function. So the posterior is just "the words still
consistent with every pattern seen", and the expected information gain of a
guess has a closed form over the feedback-pattern partition of the posterior.
No model is needed to maintain the posterior or to score a guess.

This module is pure code (no Jev, no LLM). It reuses the repo's own game
semantics -- ``WordleState.transition_state`` / ``Vocabulary`` define the
feedback and the consistency filter -- so the posterior here and the exact-EIG
optimum here are the *same* quantities the repo's ``OptimalPolicy`` uses
(scripted_policies.py), which is what makes the Jev-vs-optimal benchmark a fair,
state-matched comparison.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from llm_rl_scripts.wordle.env.game import (
    N_CHARS,
    N_TRIES,
    Vocabulary,
    WordleState,
)


def feedback_str(guess: str, target: str) -> str:
    """The ``<g>``/``<y>``/``<b>`` pattern for ``guess`` against ``target``.

    This matches the per-character coloring in ``WordleState.transition_state``
    exactly (green if the letter is right here, yellow if the letter is in the
    word but not here, black otherwise), so patterns computed here agree with
    the repo's consistency filter. Note this is the repo's game definition,
    which -- like the env and OptimalPolicy -- does not apply NYT-style
    duplicate-letter accounting; staying with it keeps the benchmark honest
    against the baseline.
    """
    out = []
    for i, c in enumerate(guess):
        if c == target[i]:
            out.append("<g>")
        elif c in target:
            out.append("<y>")
        else:
            out.append("<b>")
    return "".join(out)


def partition_sizes(candidate: str, posterior: List[str]) -> Dict[str, int]:
    """How the current posterior splits into feedback buckets if we guess
    ``candidate`` -- the empirical p(feedback | guess) under a uniform posterior."""
    buckets: Dict[str, int] = defaultdict(int)
    for w in posterior:
        buckets[feedback_str(candidate, w)] += 1
    return dict(buckets)


def expected_info_gain(candidate: str, posterior: List[str]) -> float:
    """Exact EIG (bits) of guessing ``candidate`` given the uniform posterior
    ``posterior`` (the words still possible). This is BED's acquisition in
    closed form: H(prior) - E_feedback[H(posterior | feedback)], where every
    bucket is itself uniform, so H(bucket) = log2(|bucket|)."""
    n = len(posterior)
    if n <= 1:
        return 0.0
    buckets = partition_sizes(candidate, posterior)
    prior_h = math.log2(n)
    exp_post_h = sum(cnt * math.log2(cnt) for cnt in buckets.values()) / n
    return prior_h - exp_post_h


def rank_by_eig(posterior: List[str], candidates: List[str]) -> List[Tuple[str, float]]:
    """``(word, eig)`` for each candidate, best first. Ties keep input order,
    which is broken deterministically by the caller when needed."""
    scored = [(w, expected_info_gain(w, posterior)) for w in candidates]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def optimal_guesses(posterior: List[str]) -> Tuple[List[str], float]:
    """Every exact-EIG-optimal guess drawn from the posterior, and that EIG.

    This mirrors ``OptimalPolicy.act``: candidates are restricted to the words
    still possible, and all argmax words are returned (the policy then breaks
    ties at random). The restriction matters -- it is the same greedy-entropy
    optimum the repo reports, not an unrestricted-probe optimum."""
    best_words: List[str] = []
    best = float("-inf")
    for w in posterior:
        g = expected_info_gain(w, posterior)
        if g > best:
            best_words, best = [w], g
        elif g == best:
            best_words.append(w)
    return best_words, best


@dataclass
class FixedTargetWordle:
    """A single Wordle game against a *fixed* hidden word.

    Fixing the target (rather than the env's lazy consistent sampling) is what
    lets two policies be compared on the identical game: same target, same
    deterministic feedback, so per-turn picks and whole-game outcomes are
    paired. The posterior is ``self.vocab.filtered_vocab`` and shrinks exactly
    as the repo's filter dictates."""

    target: str
    vocab: Vocabulary
    state: WordleState
    guesses: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    max_guesses: int = N_TRIES  # the guess budget; lower it to make selection bite

    @classmethod
    def new(cls, target: str, full_vocab: Vocabulary,
            max_guesses: int = N_TRIES) -> "FixedTargetWordle":
        assert len(target) == N_CHARS, "target must be {} letters".format(N_CHARS)
        init = WordleState.initial_state()
        return cls(target=target, vocab=full_vocab.update_vocab(init), state=init,
                   max_guesses=max_guesses)

    @property
    def posterior(self) -> List[str]:
        """Words still consistent with all feedback so far (the BED posterior)."""
        return list(self.vocab.filtered_vocab)

    @property
    def solved(self) -> bool:
        return len(self.guesses) > 0 and self.guesses[-1] == self.target

    @property
    def done(self) -> bool:
        return self.solved or len(self.guesses) >= self.max_guesses

    def step(self, guess: str) -> Tuple[str, bool, bool]:
        """Play ``guess``; return (feedback pattern, solved, done)."""
        pattern = feedback_str(guess, self.target)
        self.guesses.append(guess)
        self.patterns.append(pattern)
        self.state = self.state.transition_state(guess, self.target)
        self.vocab = self.vocab.update_vocab(self.state)
        return pattern, self.solved, self.done
