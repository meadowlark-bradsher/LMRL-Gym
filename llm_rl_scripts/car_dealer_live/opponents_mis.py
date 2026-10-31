"""CD-MIS-1: the TRUE opponent processes (design document section 7.1), each a perturbation of
the registered T1 kernel along one prespecified axis. The agent's ASSUMED kernel stays the
registered T1 (alpha0 = 0.8, beta = 8, p_counter = 1, p_hold = 1, shortlist 3, truthful answers);
only the process that generates the opponent's behaviour changes.

Every variant keeps the T1Kernel interface (``distribution(state, ranking)`` sums to 1 over legal
actions, ``answer_distribution``), so the true-model oracle O_true and the matched filter under
the truth can be built from it unchanged. The labels are experimental mechanisms, not claims
about real negotiators (design section 7.1). Parameters are frozen in PREREGISTRATION_MIS.md.

| id | true process | perturbation |
|---|---|---|
| matched | T1 reference | none (sanity: the posterior should be calibrated) |
| threshold | deterministic threshold responder | accepts anything with U >= r at any time; always proposes its best remaining deal |
| stochastic | noisier acceptance and proposals | rejects an acceptable offer w.p. 0.15; softmax beta = 2 over a shortlist of 5 |
| anchoring | high anchor, late concession | alpha0 = 0.95 and aspiration r + (alpha0 - r) * (remaining / H)^3 |
| withholding | strategic stalling | never accepts while more than one opportunity remains (rejects or counters instead); accepts IR at the end |
| history | history-contingent countering | aspiration rises by 0.1 for every question the focal has asked and every offer the focal has rejected (capped at 1) |
| evasive | noisy answers | each answer is flipped w.p. 0.3 |

CD-MIS-2 (severity sweep): a process id ``family@value`` sets the one scalar severity parameter of
that family -- ``history@0.4`` (aspiration step), ``stochastic@0.5`` (acceptance noise; beta 2 and
shortlist 5 kept), ``anchoring@6`` (concession power), ``evasive@0.1`` (answer flip rate). A bare
family name is the CD-MIS-1 level (``MIS1_LEVEL``). ``matched``, ``threshold`` and ``withholding``
have no severity parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Dict, List

from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, DEALS_BY_ID, Deal, Ranking, utility

from .opponent import KernelParams, T1Kernel
from .protocol_live import ACCEPT, HOLD, PROPOSE, REJECT, WALK, Action, LiveConfig, PublicState, other

REFERENCE = KernelParams()   # the registered T1 (alpha0 4/5, beta 8, p_counter 1, p_hold 1, shortlist 3)


class ThresholdKernel(T1Kernel):
    """Accepts strictly at reservation at any time; rigid proposal rule (its best remaining deal)."""

    def aspiration(self, remaining: int) -> Fraction:
        return self.cfg.reservation

    def proposal_distribution(self, state: PublicState, ranking: Ranking) -> Dict[Action, float]:
        rejected = state.rejected_by(other(self.role))
        own_standing = state.standing.deal_id if (state.standing and state.standing.proposer == self.role) else None
        cands = [d for d in ALL_DEALS if d.id not in rejected and d.id != own_standing] or list(ALL_DEALS)
        best = max(cands, key=lambda d: (utility(self.role, ranking, d), -ALL_DEALS.index(d)))
        return {Action(PROPOSE, deal_id=best.id): 1.0}


class AnchoringKernel(T1Kernel):
    """High anchor that concedes late: aspiration r + (alpha0 - r) * (remaining / H)^power."""

    power = 3

    def __init__(self, role, cfg, params=REFERENCE, power: int = 3):
        super().__init__(role, cfg, params)
        self.power = power

    def aspiration(self, remaining: int) -> Fraction:
        r = self.cfg.reservation
        return r + (self.p.alpha0 - r) * Fraction(remaining, self.cfg.horizon) ** self.power


class WithholdingKernel(T1Kernel):
    """Suppresses otherwise acceptable early offers: while more than one opportunity remains it
    never accepts (it counters w.p. p_counter, else rejects); from the last opportunity on it is T1."""

    def distribution(self, state: PublicState, ranking: Ranking) -> Dict[Action, float]:
        key = (state.key(), ranking)
        if key in self._cache:
            return self._cache[key]
        st = state.standing
        if st is not None and st.proposer != self.role and state.remaining > 1:
            out: Dict[Action, float] = {}
            pc = self.p.p_counter
            for act, pr in self.proposal_distribution(state, ranking).items():
                out[act] = out.get(act, 0.0) + pc * pr
            if 1.0 - pc > 0:
                out[Action(REJECT)] = out.get(Action(REJECT), 0.0) + (1.0 - pc)
            assert abs(sum(out.values()) - 1.0) < 1e-9
            self._cache[key] = out
            return out
        return super().distribution(state, ranking)


class HistoryKernel(T1Kernel):
    """Counter selection depends on the focal's previous acts: every question the focal has asked
    and every offer it has rejected raises the opponent's aspiration by ``step`` (capped at 1)."""

    step = Fraction(1, 10)

    def __init__(self, role, cfg, params=REFERENCE, step: Fraction = Fraction(1, 10)):
        super().__init__(role, cfg, params)
        self.step = step

    def _penalty(self, state: PublicState) -> int:
        f = other(self.role)
        return len(state.asked_by(f)) + len(state.rejected_by(f))

    def distribution(self, state: PublicState, ranking: Ranking) -> Dict[Action, float]:
        # the aspiration depends on the state, so bind it per call (the cache key already holds the counts)
        self._pen = self._penalty(state)
        return super().distribution(state, ranking)

    def aspiration(self, remaining: int) -> Fraction:
        base = super().aspiration(remaining)
        return min(Fraction(1), base + self.step * getattr(self, "_pen", 0))


MIS1_LEVEL = {"history": Fraction(1, 10), "stochastic": Fraction(3, 20), "anchoring": 3, "evasive": Fraction(3, 10)}
SEVERITY_AXIS = {"history": "aspiration step per focal question or rejection", "stochastic": "P(reject an acceptable offer)",
                 "anchoring": "concession power (remaining/H)^p", "evasive": "P(answer flipped)"}


def parse_process(process: str):
    """'history@0.4' -> ('history', Fraction(2, 5)); 'history' -> ('history', None)."""
    if "@" in process:
        fam, sev = process.split("@", 1)
        if fam not in MIS1_LEVEL:
            raise ValueError(f"{fam} has no severity parameter")
        return fam, Fraction(sev)
    return process, None


def family(process: str) -> str:
    return parse_process(process)[0]


def severity(process: str):
    """The family's severity parameter (the CD-MIS-1 level for a bare family name; None otherwise)."""
    fam, sev = parse_process(process)
    if sev is None:
        return MIS1_LEVEL.get(fam)
    return sev


def true_kernel(process: str, role: str, cfg: LiveConfig) -> T1Kernel:
    """The TRUE opponent for a CD-MIS condition. ``cfg`` carries the registered horizon / cost;
    answer noise is applied only by the 'evasive' family."""
    fam, sev = parse_process(process)
    assumed = LiveConfig(regime=cfg.regime, horizon=cfg.horizon, ask_cost=cfg.ask_cost, answer_noise=Fraction(0))
    if fam == "matched":
        return T1Kernel(role, assumed, REFERENCE)
    if fam == "threshold":
        return ThresholdKernel(role, assumed, replace(REFERENCE, shortlist=1))
    if fam == "stochastic":
        noise = MIS1_LEVEL[fam] if sev is None else sev
        return T1Kernel(role, assumed, replace(REFERENCE, beta=2.0, accept_noise=float(noise), shortlist=5))
    if fam == "anchoring":
        power = MIS1_LEVEL[fam] if sev is None else int(sev)
        return AnchoringKernel(role, assumed, replace(REFERENCE, alpha0=Fraction(19, 20)), power=power)
    if fam == "withholding":
        return WithholdingKernel(role, assumed, REFERENCE)
    if fam == "history":
        return HistoryKernel(role, assumed, REFERENCE, step=MIS1_LEVEL[fam] if sev is None else sev)
    if fam == "evasive":
        rate = MIS1_LEVEL[fam] if sev is None else sev
        noisy = LiveConfig(regime=cfg.regime, horizon=cfg.horizon, ask_cost=cfg.ask_cost, answer_noise=rate)
        return T1Kernel(role, noisy, REFERENCE)
    raise ValueError(process)


PROCESSES = ("matched", "threshold", "stochastic", "anchoring", "withholding", "history", "evasive")


def assumed_kernel(role: str, cfg: LiveConfig, answer_noise: Fraction = Fraction(0)) -> T1Kernel:
    """The agent's ASSUMED (registered) kernel: reference parameters; truthful answers unless the
    agent hedges its answer model (CD-MIS-2 ``C_ans`` arms set ``answer_noise`` to the assumed flip rate)."""
    return T1Kernel(role, LiveConfig(regime=cfg.regime, horizon=cfg.horizon, ask_cost=cfg.ask_cost, answer_noise=answer_noise), REFERENCE)
