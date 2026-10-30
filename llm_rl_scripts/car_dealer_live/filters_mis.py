"""CD-MIS-1 belief filters (design document section 7.2). All of them use the agent's ASSUMED
kernel for the likelihood; they differ in how they treat it.

* A_prior:      no update (the prior is the belief at every decision).
* B_naive:      Bayes under the assumed kernel, with a floor eta = 1e-3 of background mass so an
                observation the assumed model gives zero probability (e.g. an early WALK, an
                acceptance the assumed aspiration forbids) does not break the filter.
* B_tempered:   b' ∝ b · p_eff^tau, tau in {0.5, 0.75} (tau = 1 is B_naive).
* B_mixture:    p_eff = (1 - eta) p_model + eta p_background, eta in {0.05, 0.2}.
                p_background is uniform over the opponent's LEGAL actions at that node (the
                observation category) and uniform over {yes, no} for answers.
* B_ans{a}:     CD-MIS-2 hedged ANSWER model: Bayes under the assumed behavioural kernel, but every
                answer likelihood assumes a flip rate a (p_model = (1 - a) p_truthful + a (1 - p_truthful));
                a = 0 is B_naive. The ASK-allowed arms C_ans{a} plan with the same hedged kernel, so the
                planner's expected information gain and the filter's update use the same answer model.
The exact filter under the TRUE kernel (what O_true holds) is kept beside them for diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

from . import belief_live as bl
from .opponent import T1Kernel
from .protocol_live import ASK, Action, PublicState, legal_actions

N = len(ALL_RANKINGS)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    update: bool = True          # False = A_prior
    eta: float = 1e-3            # background mixture weight
    tau: float = 1.0             # likelihood tempering
    answer_noise: float = 0.0    # assumed answer flip rate (hedged answer model; 0 = truthful)

    def to_json(self) -> dict:
        return {"name": self.name, "update": self.update, "eta": self.eta, "tau": self.tau, "answer_noise": self.answer_noise}


FILTERS = {
    "A_prior": FilterSpec("A_prior", update=False),
    "B_naive": FilterSpec("B_naive"),
    "B_temp0.5": FilterSpec("B_temp0.5", tau=0.5),
    "B_temp0.75": FilterSpec("B_temp0.75", tau=0.75),
    "B_mix0.05": FilterSpec("B_mix0.05", eta=0.05),
    "B_mix0.2": FilterSpec("B_mix0.2", eta=0.2),
    "B_ans0.1": FilterSpec("B_ans0.1", answer_noise=0.1),
    "B_ans0.2": FilterSpec("B_ans0.2", answer_noise=0.2),
    "B_ans0.3": FilterSpec("B_ans0.3", answer_noise=0.3),
}


def effective_likelihood(p_model: Tuple[float, ...], p_bg: float, eta: float) -> Tuple[float, ...]:
    return tuple((1.0 - eta) * p + eta * p_bg for p in p_model)


def opponent_action_likelihood(assumed: T1Kernel, state: PublicState, act: Action, spec: FilterSpec) -> Tuple[float, ...]:
    """Per-type effective likelihood of an observed opponent action under the assumed kernel."""
    liks = bl.action_likelihoods(assumed, state)
    p_model = liks.get(act, tuple(0.0 for _ in range(N)))
    p_bg = 1.0 / len(legal_actions(state, assumed.role))
    return effective_likelihood(p_model, p_bg, spec.eta)


def answer_likelihood(assumed: T1Kernel, question_id: str, answer: str, spec: FilterSpec) -> Tuple[float, ...]:
    """Per-type effective likelihood of an answer. ``assumed`` is the agent's (truthful) kernel; a hedged
    spec mixes in its assumed flip rate before the background floor is applied."""
    p_model = bl.answer_likelihoods(assumed, question_id)[answer]
    a = spec.answer_noise
    if a > 0:
        p_model = tuple((1.0 - a) * p + a * (1.0 - p) for p in p_model)
    return effective_likelihood(p_model, 0.5, spec.eta)


def update(belief: bl.Belief, lik: Tuple[float, ...], spec: FilterSpec) -> bl.Belief:
    if not spec.update:
        return belief
    w = [b * (l ** spec.tau) for b, l in zip(belief, lik)]
    z = sum(w)
    if z <= 0:            # cannot happen with eta > 0; keep the belief rather than divide by zero
        return belief
    return tuple(x / z for x in w)


def predictive_logloss_bits(belief: bl.Belief, lik: Tuple[float, ...]) -> float:
    """-log2 of the agent's predictive probability of the observation it just saw."""
    m = bl.marginal(belief, lik)
    return -math.log2(max(m, 1e-300))


def belief_diagnostics(belief: bl.Belief, true_index: int) -> dict:
    p_true = belief[true_index]
    wrong = max(p for i, p in enumerate(belief) if i != true_index)
    return {"p_true": p_true, "logloss_true_bits": -math.log2(max(p_true, 1e-300)), "entropy_bits": bl.entropy_bits(belief),
            "max_wrong": wrong, "argmax_is_true": max(range(N), key=lambda i: belief[i]) == true_index}
