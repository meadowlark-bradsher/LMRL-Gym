"""Build the R0 behavior-cloning dataset: focal decision nodes labelled by the
exact ``exact_ask`` oracle.

Coverage comes from rolling out the three scripted arms (``exact_ask``,
``greedy_eig``, ``random``) across every (role, focal type, opener, opponent type)
cell for a few repeats and recording each focal node the focal faces; nodes are
deduplicated by ``(role, focal_type, state.key(), round_belief)``. Each unique
node is then labelled with the teacher: the shortlist menu it sees, the exact Q*
of every row under ``cell.ask`` (the menu ASK oracle = C_plan), V*, and the row
``exact_ask`` would pick (the clone target). Everything is deterministic given
the collection seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live import belief_live as bl
from llm_rl_scripts.car_dealer_live.arms_live import exact_ask as exact_ask_arm, arms_for
from llm_rl_scripts.car_dealer_live.opponent import KernelParams
from llm_rl_scripts.car_dealer_live.oracle import EPS
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, LiveConfig, apply, new_game
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from .features import FocalContext, ROW_DIM, rows_matrix, state_features

# The registered Phase-1 confirmatory regime (DESIGN_LIVE.md section 4): H=4, c_q=0,
# T1 kernel alpha0=0.8, beta=8, p_counter=1, p_hold=1, shortlist 3, menu "shortlist".
REGISTERED_CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))
REGISTERED_PARAMS = KernelParams(alpha0=Fraction(4, 5), beta=8.0, p_counter=1.0, p_hold=1.0, shortlist=3)
ROLES = ("buyer", "seller")
TYPE_IDX = tuple(range(len(ALL_RANKINGS)))


def build_cells(cfg: LiveConfig = REGISTERED_CFG, params: KernelParams = REGISTERED_PARAMS,
                roles=ROLES, types=TYPE_IDX) -> Dict[Tuple[str, int], Cell]:
    """One ``Cell`` (menu + oracles) per (role, focal-type index)."""
    return {(role, i): Cell(cfg, params, role, ALL_RANKINGS[i], "shortlist", 3, 2) for role in roles for i in types}


def context_for(cell: Cell) -> FocalContext:
    return FocalContext(role=cell.focal, ranking=cell.ranking, cfg=cell.cfg)


@dataclass
class LabeledNode:
    role: str
    type_idx: int
    state_feat: np.ndarray            # [STATE_DIM]
    row_feat: np.ndarray             # [n_rows, ROW_DIM]
    best_idx: int                     # the row exact_ask picks (clone target)
    optimal_mask: np.ndarray          # [n_rows] bool: Q* within EPS of the max
    q_star: np.ndarray                # [n_rows] float: exact Q* per row
    v_star: float

    @property
    def n_rows(self) -> int:
        return self.row_feat.shape[0]


def _rollout_collect(cell: Cell, arm, opener: str, opp_ranking, rng: random.Random,
                     scripted: dict, out: Dict[tuple, Tuple[Cell, object, bl.Belief]]):
    """Play one episode under ``arm`` and record every focal node into ``out``
    (keyed for dedup). ``arm`` is a scripted arm callable from ``arms_for``."""
    cfg, focal = cell.cfg, cell.focal
    state = new_game(cfg, opener)
    belief = bl.uniform()
    while state.terminal is None:
        if state.turn != focal:
            act = cell.kernel.sample(state, opp_ranking, rng)
            lik = bl.action_likelihoods(cell.kernel, state)[act]
            state = apply(state, act)
            belief = bl.posterior(belief, lik)
            continue
        rows = cell.menu.rows(state, belief, include_questions=True)
        q_ask = {r.key: cell.ask.q_value(state, belief, r.action) for r in rows}
        q_noask = {r.key: cell.noask.q_value(state, belief, r.action) for r in rows if r.action.kind != ASK}
        key = ((cell.focal, ALL_RANKINGS.index(cell.ranking)), state.key(), bl.round_belief(belief))
        out.setdefault(key, (cell, state, belief))
        act = arm(rows, q_ask, q_noask, rng)   # scripted arms return an Action
        if act.kind == ASK:
            ans = cell.kernel.sample_answer(act.question_id, opp_ranking, rng)
            lik = bl.answer_likelihoods(cell.kernel, act.question_id)[ans]
            state = apply(state, act, ans)
            belief = bl.posterior(belief, lik)
        else:
            state = apply(state, act)


def label_node(cell: Cell, state, belief: bl.Belief) -> LabeledNode:
    """Teacher labels for one focal node: menu rows, exact Q* per row, V*, and the
    row ``exact_ask`` would choose."""
    ctx = context_for(cell)
    rows = cell.menu.rows(state, belief, include_questions=True)
    q_ask = {r.key: cell.ask.q_value(state, belief, r.action) for r in rows}
    q_noask = {r.key: cell.noask.q_value(state, belief, r.action) for r in rows if r.action.kind != ASK}
    q = np.array([q_ask[r.key] for r in rows], dtype=np.float32)
    top = float(q.max())
    optimal = q >= top - EPS
    chosen = exact_ask_arm(rows, q_ask, q_noask, random.Random(0)).key   # deterministic rule
    best_idx = next(i for i, r in enumerate(rows) if r.key == chosen)
    return LabeledNode(
        role=cell.focal, type_idx=ALL_RANKINGS.index(cell.ranking),
        state_feat=state_features(ctx, state, belief), row_feat=rows_matrix(rows),
        best_idx=best_idx, optimal_mask=optimal, q_star=q, v_star=float(cell.ask.value(state, belief)),
    )


def collect_nodes(cells: Dict[Tuple[str, int], Cell], openers=ROLES, opp_types=TYPE_IDX,
                  repeats: int = 3, seed: int = 0, arms=("exact_ask", "greedy_eig", "random")) -> List[LabeledNode]:
    """Collect and label all unique focal nodes reachable under the scripted arms."""
    scripted = arms_for(float(REGISTERED_CFG.reservation))
    seen: Dict[tuple, Tuple[Cell, object, bl.Belief]] = {}
    for (role, i), cell in cells.items():
        for arm_name in arms:
            arm = scripted[arm_name]
            for opener in openers:
                for j in opp_types:
                    for k in range(repeats):
                        rng = random.Random(f"collect|{seed}|{arm_name}|{role}|{opener}|{i}|{j}|{k}")
                        _rollout_collect(cell, arm, opener, ALL_RANKINGS[j], rng, scripted, seen)
    return [label_node(cell, state, belief) for (cell, state, belief) in seen.values()]


@dataclass
class TensorDataset:
    state: "np.ndarray"       # [N, STATE_DIM] float32
    rows: "np.ndarray"        # [N, R, ROW_DIM] float32
    mask: "np.ndarray"        # [N, R] bool
    best_idx: "np.ndarray"    # [N] int64
    optimal: "np.ndarray"     # [N, R] bool
    q_star: "np.ndarray"      # [N, R] float32 (padded 0)
    v_star: "np.ndarray"      # [N] float32
    nodes: List[LabeledNode] = field(default_factory=list)

    @property
    def max_rows(self) -> int:
        return self.rows.shape[1]


def to_tensors(nodes: List[LabeledNode]) -> TensorDataset:
    n = len(nodes)
    R = max(nd.n_rows for nd in nodes)
    S = nodes[0].state_feat.shape[0]
    state = np.zeros((n, S), dtype=np.float32)
    rows = np.zeros((n, R, ROW_DIM), dtype=np.float32)
    mask = np.zeros((n, R), dtype=bool)
    best = np.zeros(n, dtype=np.int64)
    optimal = np.zeros((n, R), dtype=bool)
    q_star = np.zeros((n, R), dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)
    for t, nd in enumerate(nodes):
        r = nd.n_rows
        state[t] = nd.state_feat
        rows[t, :r] = nd.row_feat
        mask[t, :r] = True
        best[t] = nd.best_idx
        optimal[t, :r] = nd.optimal_mask
        q_star[t, :r] = nd.q_star
        v[t] = nd.v_star
    return TensorDataset(state, rows, mask, best, optimal, q_star, v, nodes)
