"""Featurization for the CD-DRL focal policy.

Two feature groups, both derived only from what a focal policy legitimately
observes (the matched belief + the public state + its own payoffs):

* **state features** (``STATE_DIM``) — a fixed-length encoding of the focal
  decision node. It is a superset of the oracle's own memoisation key
  (``PublicState.key()`` = turn, remaining, standing, asked/rejected masks) plus
  the belief and the focal's full 27-deal own-utility profile, so two nodes with
  different optimal actions never collide (``test_drl`` asserts this). ``turn``
  is omitted (always the focal at a focal node), as is ``terminal`` (never set),
  and ``opener`` (the oracle value is a function of ``key() + belief`` only).
* **row features** (``ROW_DIM``) — one vector per menu ``Row``: its kind, the
  exact EIG, P(accept) and own utility the menu already computed, and its tag.
  These are the identical numbers the Jev arm is shown.

Everything is plain NumPy; the Torch net consumes the arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_binned.questions import BANK
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, ALL_RANKINGS, DEALS_BY_ID, Ranking, utility
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.menu_live import AMBITIOUS, BALANCED, PLAN, PLAN_ASK, PLAN_Q, SAFE, Row
from llm_rl_scripts.car_dealer_live.protocol_live import (
    ACCEPT, ASK, HOLD, LiveConfig, PROPOSE, PublicState, REJECT, WALK, other,
)

# -- stable index maps (the bit-mask column order) --------------------------------
DEAL_IDX: Dict[str, int] = {d.id: i for i, d in enumerate(ALL_DEALS)}
QUESTION_IDX: Dict[str, int] = {q.id: i for i, q in enumerate(BANK)}
N_DEALS = len(ALL_DEALS)        # 27
N_QUESTIONS = len(BANK)         # 9
N_RANKINGS = len(ALL_RANKINGS)  # 6

KINDS = (ACCEPT, REJECT, HOLD, PROPOSE, ASK, WALK)
KIND_IDX = {k: i for i, k in enumerate(KINDS)}
TAGS = (None, SAFE, AMBITIOUS, PLAN, BALANCED, PLAN_ASK, PLAN_Q)
TAG_IDX = {t: i for i, t in enumerate(TAGS)}

_LOG2_N = float(np.log2(N_RANKINGS))
_H_NORM = 8.0            # horizon scale (H is small; keep the feature O(1))
_ATTR_MAX = float(max(max(d.price, d.warranty, d.delivery) for d in ALL_DEALS)) or 1.0  # level scale (deals are 3^3)


@dataclass(frozen=True)
class FocalContext:
    """The per-cell constants a focal policy carries: who it is, its payoff
    profile, and the game knobs. Built once per (role, focal ranking)."""

    role: str
    ranking: Ranking
    cfg: LiveConfig

    @property
    def own_util(self) -> np.ndarray:
        """Own utility of every deal in ``ALL_DEALS`` order (already in [0, 1])."""
        return np.array([float(utility(self.role, self.ranking, d)) for d in ALL_DEALS], dtype=np.float32)


def _mask(ids, idx: Dict[str, int], n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    for x in ids:
        v[idx[x]] = 1.0
    return v


def state_features(ctx: FocalContext, state: PublicState, belief: bl.Belief) -> np.ndarray:
    """Fixed-length encoding of a focal decision node. See module docstring."""
    opp = other(ctx.role)
    st = state.standing
    standing_none = 1.0 if st is None else 0.0
    standing_mine = 1.0 if (st is not None and st.proposer == ctx.role) else 0.0
    standing_theirs = 1.0 if (st is not None and st.proposer == opp) else 0.0
    standing_u = float(ctx.own_util[DEAL_IDX[st.deal_id]]) if st is not None else 0.0
    parts = [
        np.asarray(belief, dtype=np.float32),                              # 6
        np.array([bl.entropy_bits(belief) / _LOG2_N], dtype=np.float32),   # 1
        np.array([state.remaining / max(ctx.cfg.horizon, 1)], dtype=np.float32),  # 1
        np.array([1.0 if state.at_deadline else 0.0], dtype=np.float32),   # 1
        np.array([1.0 if ctx.role == "buyer" else 0.0], dtype=np.float32), # 1
        np.array([standing_none, standing_mine, standing_theirs, standing_u], dtype=np.float32),  # 4
        _mask(state.asked_by(ctx.role), QUESTION_IDX, N_QUESTIONS),        # 9
        _mask(state.asked_by(opp), QUESTION_IDX, N_QUESTIONS),             # 9
        _mask(state.rejected_by(ctx.role), DEAL_IDX, N_DEALS),             # 27
        _mask(state.rejected_by(opp), DEAL_IDX, N_DEALS),                  # 27
        ctx.own_util,                                                      # 27
        np.array([ctx.cfg.horizon / _H_NORM, float(ctx.cfg.ask_cost)], dtype=np.float32),  # 2
    ]
    return np.concatenate(parts).astype(np.float32)


def row_features(row: Row) -> np.ndarray:
    """One vector per menu row, identifying the action fully so the net can rank
    two options with the same annotations but different downstream value: kind
    one-hot, EIG, P(accept) (+flag), own utility (+flag), tag one-hot, the
    proposed deal's attributes (PROPOSE only), and the question one-hot (ASK only)."""
    kind = np.zeros(len(KINDS), dtype=np.float32)
    kind[KIND_IDX[row.action.kind]] = 1.0
    pacc = 0.0 if row.p_accept is None else float(row.p_accept)
    has_pacc = 0.0 if row.p_accept is None else 1.0
    own = 0.0 if row.own_utility is None else float(row.own_utility)
    has_own = 0.0 if row.own_utility is None else 1.0
    tag = np.zeros(len(TAGS), dtype=np.float32)
    tag[TAG_IDX[row.tag]] = 1.0
    scalars = np.array([float(row.eig_bits), pacc, has_pacc, own, has_own], dtype=np.float32)
    deal_attr = np.zeros(3, dtype=np.float32)
    if row.action.kind == PROPOSE:
        d = DEALS_BY_ID[row.action.deal_id]
        deal_attr[:] = (d.price / _ATTR_MAX, d.warranty / _ATTR_MAX, d.delivery / _ATTR_MAX)
    q_onehot = np.zeros(N_QUESTIONS, dtype=np.float32)
    if row.action.kind == ASK:
        q_onehot[QUESTION_IDX[row.action.question_id]] = 1.0
    return np.concatenate([kind, scalars, tag, deal_attr, q_onehot]).astype(np.float32)


STATE_DIM = 6 + 1 + 1 + 1 + 1 + 4 + N_QUESTIONS + N_QUESTIONS + N_DEALS + N_DEALS + N_DEALS + 2  # 115
ROW_DIM = len(KINDS) + 5 + len(TAGS) + 3 + N_QUESTIONS  # 30


def rows_matrix(rows) -> np.ndarray:
    """``[n_rows, ROW_DIM]`` for a menu."""
    return np.stack([row_features(r) for r in rows]).astype(np.float32)
