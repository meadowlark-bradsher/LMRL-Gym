"""External acceptance-probability aid: the "Jev + explicit belief filter" condition.

The J/J salience probe showed the binding failure is on the OFFER side: in 130
offers neither Jev seat ever proposed a deal worth >= r to the recipient. Jev has
no forward model of what the other party will accept. This module supplies that
forward model — and ONLY that — as an external, inspectable component.

Division of labour (do not blur it, or you measure the aid instead of Jev):

  * The aid maintains a finite Bayesian posterior over the opponent's 6 possible
    rankings and returns, for every candidate offer, P(opponent accepts).
  * Jev receives that number as an annotation on each OFFER option and still
    makes the Choice. The aid never ranks options, computes expected value,
    recommends, or prunes the legal action set. The posterior itself is logged
    for the evaluator (it unlocks the section-4.3 epistemic metrics) and is NOT
    shown to Jev.

The aid sees only the policy's own :class:`Observation` — never the opponent's
true ranking — so it is leakage-safe by construction.

Response model (public, deliberately simple, documented as optimistic):
    P(accept d | theta) = 1 - eps  if U_opp(d; theta) >= r   (individually rational)
                        = eps      otherwise
The noise floor ``eps`` keeps a *strategic* rejection of an IR deal (spec 3.2:
rejection does not prove sub-reservation value) from hard-eliminating the true
type and leaving the posterior confidently wrong.

Offer likelihood (what an opponent of type theta tends to propose):
    P(offer d | theta) = softmax_beta( U_opp(d; theta) ) over the 27 deals.
Note the ideal deal has utility 1.0 under EVERY ranking, so an ideal offer carries
no information and leaves the posterior uniform. Type only leaks through
concessions — *which* attribute an agent gives up reveals its priorities.

Modes (mechanism controls; the policy/runner code path is identical across them,
only the emitted numbers differ, so the instruction sentence is the same string):
  * ``bayes``      — the real filter above (default).
  * ``prior_only`` — never updates: always the uniform-prior P(accept). Isolates
                     the prior-averaged forward model from opponent inference.
  * ``placebo``    — every offer annotated with a constant ``placebo_value``.
                     Isolates 'an acceptance field exists' from its information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import (
    ALL_DEALS,
    ALL_RANKINGS,
    DEALS_BY_ID,
    Deal,
    Ranking,
    utility,
)
from llm_rl_scripts.car_dealer_discrete.protocol import (
    ACCEPT,
    OFFER,
    REJECT,
    Observation,
)


MODES = ("bayes", "prior_only", "placebo")


@dataclass(frozen=True)
class AidConfig:
    beta: float = 8.0           # softmax sharpness of the opponent-offer likelihood
    noise: float = 0.05         # acceptance noise floor eps, in (0, 0.5)
    mode: str = "bayes"         # one of MODES
    placebo_value: float = 0.5  # constant annotation used in placebo mode

    def __post_init__(self) -> None:
        if not (0.0 < self.noise < 0.5):
            raise ValueError("noise must be in (0, 0.5)")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not (0.0 <= self.placebo_value <= 1.0):
            raise ValueError("placebo_value must be in [0, 1]")


def _other(role: str) -> str:
    return "seller" if role == "buyer" else "buyer"


def ranking_key(theta: Ranking) -> str:
    """Stable JSON key for a ranking, e.g. 'price>warranty>delivery'."""
    return ">".join(theta)


def entropy_bits(q: Dict[Ranking, float]) -> float:
    return -sum(p * math.log2(p) for p in q.values() if p > 0.0)


class AcceptanceAid:
    """Stateless: the posterior is recomputed from ``obs.history`` every turn."""

    def __init__(self, config: Optional[AidConfig] = None) -> None:
        self.cfg = config or AidConfig()

    # -- response model ------------------------------------------------------

    def p_accept_given_type(
        self, deal: Deal, opp_role: str, theta: Ranking, r: Fraction
    ) -> float:
        ir = utility(opp_role, theta, deal) >= r
        return (1.0 - self.cfg.noise) if ir else self.cfg.noise

    # -- offer likelihood ----------------------------------------------------

    def p_offer_given_type(self, deal: Deal, opp_role: str, theta: Ranking) -> float:
        b = self.cfg.beta
        logits = [b * float(utility(opp_role, theta, d)) for d in ALL_DEALS]
        m = max(logits)
        z = sum(math.exp(x - m) for x in logits)
        return math.exp(b * float(utility(opp_role, theta, deal)) - m) / z

    # -- belief filter -------------------------------------------------------

    def posterior(self, obs: Observation) -> Dict[Ranking, float]:
        """Posterior over the opponent's ranking given the public history.

        Walks the history once, tracking whose offer is outstanding, and
        applies a likelihood for each OPPONENT action:
          * REJECT or counter-OFFER while OUR offer is outstanding
                -> non-acceptance of that offer:  1 - P(accept | d_us, theta)
          * OFFER d                               -> P(offer d | theta)
        Our own actions carry no information about the opponent.
        """
        if self.cfg.mode != "bayes":
            # Control modes carry no belief: report the uniform prior so the
            # evaluator's epistemic metrics show exactly zero learning.
            return {theta: 1.0 / len(ALL_RANKINGS) for theta in ALL_RANKINGS}

        me, opp = obs.role, _other(obs.role)
        r = obs.reservation
        logw: Dict[Ranking, float] = {theta: 0.0 for theta in ALL_RANKINGS}
        outstanding: Optional[Tuple[str, Deal]] = None  # (who offered, deal)

        for turn in obs.history:
            a = turn.action
            if turn.actor == opp:
                if (
                    outstanding is not None
                    and outstanding[0] == me
                    and a.kind in (REJECT, OFFER)
                ):
                    d_us = outstanding[1]
                    for theta in ALL_RANKINGS:
                        logw[theta] += math.log(
                            1.0 - self.p_accept_given_type(d_us, opp, theta, r)
                        )
                if a.kind == OFFER:
                    d = DEALS_BY_ID[a.deal_id]
                    for theta in ALL_RANKINGS:
                        logw[theta] += math.log(self.p_offer_given_type(d, opp, theta))
            # track the outstanding offer regardless of who acted
            if a.kind == OFFER:
                outstanding = (turn.actor, DEALS_BY_ID[a.deal_id])
            elif a.kind in (REJECT, ACCEPT):
                outstanding = None

        m = max(logw.values())
        w = {theta: math.exp(v - m) for theta, v in logw.items()}
        z = sum(w.values())
        return {theta: v / z for theta, v in w.items()}

    # -- what the policy consumes -------------------------------------------

    def acceptance_probabilities(
        self, obs: Observation
    ) -> Tuple[Dict[Ranking, float], Dict[str, float]]:
        """(posterior, {deal_id: P(opponent accepts)}) for every legal deal."""
        q = self.posterior(obs)
        if self.cfg.mode == "placebo":
            return q, {d.id: self.cfg.placebo_value for d in ALL_DEALS}
        opp = _other(obs.role)
        r = obs.reservation
        pacc = {
            d.id: sum(
                q[theta] * self.p_accept_given_type(d, opp, theta, r)
                for theta in ALL_RANKINGS
            )
            for d in ALL_DEALS
        }
        return q, pacc
