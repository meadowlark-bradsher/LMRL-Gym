"""The CFR check of Split Deal v0.3 (`DESIGN_SPLIT_CFR.md`): game G, its public tree, behaviour profiles, exact
best responses and CFR+. Floating point throughout; the ladder (``twosided.py``, ``readers.py``) stays in exact
Fractions and is compared with it to within ``TOL``.

Game G (decision 1). ``g_tables`` builds the v0.3 tables of an instance with the full 13-question bank or the
merged one (the lowest bank index of each distinct partition, ``spec_numbers_cfr.merged_questions``).
``split_game`` builds the public tree the ladder lives in and ``spec_numbers_cfr.py`` counts: A asks only when B
will answer and only a question splitting A's answer-consistent belief about B; B asks only a question it has not
had declined and that splits its answer-consistent belief about A; either may pass. Every move is public and a
player's only private information is its type, so an information set is (own type, public node). The final stage
is not a subtree: the proposer is A with probability pi and picks one of the allocations or walks, the responder
accepts iff its gain is at least 0, and both values come from fixed payoff tables.

Engine. A ``Tree`` is breadth-first: the children of a node are contiguous and depth never decreases, so every pass
is one vectorised step per depth. A behaviour strategy is an array over a player's decision edges x own types plus,
where the tree has offer leaves, an array over its offer information sets x offers. Leaves are either offer leaves
(Split Deal; independent prior) or payoff-matrix leaves (any joint prior; used for Kuhn poker). Counterfactual
values are conditional on the player's own type, so a player's value is the prior mean over its types of its
root value.

Operational choices, fixed here before any SD3-E value is computed:

* payoffs are own-scale Y: net gain over own d as a share of 10 - own d; no deal pays 0.
* edge order, which settles best-response ties: pass before any ask, asks in bank order; decline before answer;
  offers by allocation index (A's share), walking last. A best response takes the first edge within ``TIE_TOL``
  of the best.
* CFR+: alternating updates (A, then B), regret matching+, linear averaging (weight t, no delay). DCFR
  (alpha 1.5, beta 0, gamma 2) is the second fallback of decision 4.
* warm start (decision 5): the initial regrets are 1 on the warm profile's actions and 0 elsewhere; the average
  starts empty.
* NashConv of a profile: the sum over both players of (best-response value - value), each a prior mean over own
  types.
* offers: each seat's menu per own type leaves out offers that walking or another offer is at least as good as
  against every responder (``offer_menus``); this changes no best-response value and loses no equilibrium.
* A3 off its own model (reached only when B deviates from B2, as in a best response to A3): a B question that no
  type in S3 would ask under B2 leaves S3 unchanged; a B answer that no type in S3 fits resets S3 to the B types
  consistent with all of B's answers. This is B2's off-path rule (`DESIGN_SPLIT_B2.md`, decision 4) applied to A3.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

# One BLAS thread: the products here are small and many, and a multi-threaded OpenBLAS spins badly on a busy
# machine (about 9x slower in the pilot's environment). The variable works only if numpy is not imported yet;
# threadpoolctl, where installed, also limits a BLAS that is already loaded.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np                                                                  # noqa: E402
from scipy import sparse                                                            # noqa: E402

try:                                                                                # noqa: E402
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:                                                                 # pragma: no cover
    pass

from .game import Instance
from .private_d import PrivateTables
from .readers import (A1Model, A3, B2, b_will_answer, bits, rollout_a1_b0, rollout_a1_b2, rollout_a3_b2,
                      update_answer, update_ask, update_decline)
from .spec_numbers_cfr import merged_questions
from .twosided import GAME, STRUCTURES, Shared, Structure

DEC_A, DEC_B, REV_A, REV_B, PAY, OFFER = range(6)
TOL = 1e-9
TIE_TOL = 1e-12
EPS_BAR = 1e-3                                   # certified iff NashConv <= EPS_BAR (decision 4)
G_INSTANCE, G_Q, G_PI = (2, 2, 3), 2, Fraction(1, 2)
G_ARMS = {"T2": STRUCTURES["T2"], "T3": STRUCTURES["T3"], **{f"game:{a}/{b}": s for (a, b), s in GAME.items()}}


def g_tables(inst: Instance, bank: str = "merged") -> PrivateTables:
    tab = PrivateTables(inst)
    if bank == "merged":
        keep = merged_questions(tab)
        tab.questions = [tab.questions[j] for j in keep]
        tab.yes = [tab.yes[j] for j in keep]
    elif bank != "full":
        raise ValueError(bank)
    return tab


# -- the tree --------------------------------------------------------------------------------------

def _rows(mask: int, n: int) -> np.ndarray:
    return np.array([(mask >> t) & 1 for t in range(n)], dtype=float)


class Tree:
    """A public game tree in breadth-first order, built from ``expand(state)``, which returns one of
    ("dec", player, [(label, child_state), ...]), ("rev", player, [(label, child_state, type_mask), ...]),
    ("offer", SA, SB) or ("pay", matrix_id). Players are 0 (A) and 1 (B)."""

    def __init__(self, root, expand: Callable, n: Tuple[int, int]):
        self.n = n
        kind, parent, first, nch, depth, label, state, mask, leaf = [], [], [], [], [], [], [], [], []

        def add(st, par, lab, msk, dep):
            kind.append(-1); parent.append(par); first.append(-1); nch.append(0)
            depth.append(dep); label.append(lab); state.append(st); mask.append(msk); leaf.append(None)

        add(root, -1, None, None, 0)
        i = 0
        while i < len(state):
            out = expand(state[i])
            if out[0] in ("dec", "rev"):
                pl, children = out[1], out[2]
                kind[i] = (DEC_A, DEC_B)[pl] if out[0] == "dec" else (REV_A, REV_B)[pl]
                first[i], nch[i] = len(state), len(children)
                for ch in children:
                    add(ch[1], i, ch[0], ch[2] if out[0] == "rev" else None, depth[i] + 1)
            elif out[0] == "offer":
                kind[i], leaf[i] = OFFER, (out[1], out[2])
            elif out[0] == "pay":
                kind[i], leaf[i] = PAY, out[1]
            else:
                raise ValueError(out[0])
            i += 1
        self.kind = np.array(kind, dtype=np.int8)
        self.parent = np.array(parent)
        self.first = np.array(first)
        self.nch = np.array(nch)
        self.depth = np.array(depth)
        self.label, self.state, self.leaf = label, state, leaf
        self.N = len(kind)
        D = int(self.depth.max())
        self.D = D
        self.dstart = np.searchsorted(self.depth, np.arange(D + 2))
        self.internal = []
        for d in range(D):
            ids = np.arange(self.dstart[d], self.dstart[d + 1])
            ids = ids[self.nch[ids] > 0]
            starts = self.first[ids] - self.dstart[d + 1]
            width = self.dstart[d + 2] - self.dstart[d + 1]
            cols = np.arange(width)
            rows = np.repeat(np.arange(len(ids)), self.nch[ids])
            agg = sparse.csr_matrix((np.ones(width), (rows, cols)), shape=(len(ids), width))
            self.internal.append((ids, starts, agg))
        pk = np.where(self.parent >= 0, self.kind[np.maximum(self.parent, 0)], -1)
        self.edge_child, self.edge_parent, self.dec_nodes, self.dec_start, self.edge_of = [], [], [], [], []
        self.rev_child, self.rev_mask, self.own_dec = [], [], []
        for p in (0, 1):
            ec = np.nonzero(pk == (DEC_A, DEC_B)[p])[0]
            self.edge_child.append(ec)
            self.edge_parent.append(self.parent[ec])
            dn = np.nonzero(self.kind == (DEC_A, DEC_B)[p])[0]
            self.dec_nodes.append(dn)
            eo = np.full(self.N, -1)
            eo[ec] = np.arange(len(ec))
            self.edge_of.append(eo)
            ds = eo[self.first[dn]]
            self.dec_start.append(ds)
            cnt = np.diff(np.append(ds, len(ec)))
            self.edge_dec = getattr(self, "edge_dec", []) + [np.repeat(np.arange(len(dn)), cnt)]
            self.edge_cnt = getattr(self, "edge_cnt", []) + [cnt]
            self.dec_agg = getattr(self, "dec_agg", []) + [sparse.csr_matrix(
                (np.ones(len(ec)), (self.edge_dec[-1], np.arange(len(ec)))), shape=(len(dn), len(ec)))]
            rc = np.nonzero(pk == (REV_A, REV_B)[p])[0]
            self.rev_child.append(rc)
            self.rev_mask.append(np.array([_rows(mask[c], n[p]) for c in rc]).reshape(len(rc), n[p]))
            self.own_dec.append(self.kind == (DEC_A, DEC_B)[p])
        self.offer_nodes = np.nonzero(self.kind == OFFER)[0]
        self.pay_nodes = np.nonzero(self.kind == PAY)[0]
        # offer information sets: (offer leaf, own type) for the types consistent with the leaf, sorted by leaf
        self.off_leaf, self.off_type = [], []
        for p in (0, 1):
            ls, ts = [], []
            for li, z in enumerate(self.offer_nodes):
                for t in bits(self.leaf[z][p]):
                    ls.append(li)
                    ts.append(t)
            self.off_leaf.append(np.array(ls, dtype=int))
            self.off_type.append(np.array(ts, dtype=int))
        self.off_start = [np.searchsorted(self.off_leaf[p], np.arange(len(self.offer_nodes))) for p in (0, 1)]
        self.off_agg = [sparse.csr_matrix((np.ones(len(self.off_leaf[p])), (self.off_leaf[p],
                                                                           np.arange(len(self.off_leaf[p])))),
                                          shape=(len(self.offer_nodes), len(self.off_leaf[p]))) for p in (0, 1)]


# -- games ------------------------------------------------------------------------------------------

class Game:
    """A tree with its prior, the proposer probability and the leaf payoffs."""

    def __init__(self, tree: Tree, prior: np.ndarray, pi: float = 0.5, pay: Optional[List] = None,
                 offer_tables: Optional[dict] = None):
        self.tree, self.P, self.pi = tree, np.asarray(prior, dtype=float), float(pi)
        self.p = (self.P.sum(axis=1), self.P.sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cond = (np.nan_to_num(self.P / self.p[0][:, None]), np.nan_to_num(self.P.T / self.p[1][:, None]))
        self.pay = pay or []                                  # [(U_A, U_B)], each n_A x n_B
        self.ot = offer_tables
        if len(tree.offer_nodes):
            assert np.allclose(self.P, np.outer(*self.p)), "offer leaves need an independent prior"
            ot = self.ot
            self.m = ot["GA"].shape[1]
            self.M = tuple(ot["KEEP"][p].shape[1] for p in (0, 1))
            w = (self.pi, 1 - self.pi)
            G = (ot["GA"], ot["GB"])
            H = (ot["HA"], ot["HB"])
            self.cols, self.valid, self.qcoef, self.by_type, self.HK = [], [], [], [], []
            for p in (0, 1):
                ty = tree.off_type[p]
                keep = ot["KEEP"][p]
                self.cols.append(keep[ty].astype(np.int32))
                self.valid.append(ot["VALID"][p][ty])
                self.qcoef.append(w[p] * np.take_along_axis(G[p], keep, axis=1)[ty] * self.valid[p])
                self.by_type.append([(t, np.nonzero(ty == t)[0]) for t in range(tree.n[p])])
                # the responder's payoff for each offer on p's menu: H of the other player, rows p's menu
                self.HK.append([H[1 - p][keep[t]] * ot["VALID"][p][t][:, None] for t in range(tree.n[p])])

    # strategies are pairs (talk, offers): talk is E_p x n_p, offers is J_p x M_p over p's offer menu (or None)
    def uniform(self, p: int):
        t = self.tree
        return self.normalise(p, (np.zeros((len(t.edge_child[p]), t.n[p])), self._zeros_off(p)))

    def _zeros_off(self, p):
        return np.zeros((len(self.tree.off_leaf[p]), self.M[p])) if len(self.tree.offer_nodes) else None

    def normalise(self, p: int, s):
        """Regret matching on non-negative weights (uniform where they sum to 0), as a behaviour strategy."""
        t = self.tree
        talk, off = s
        out_t = talk
        if len(talk):
            sums_e = (t.dec_agg[p] @ talk)[t.edge_dec[p]]
            uni = (1.0 / t.edge_cnt[p])[t.edge_dec[p]][:, None]
            pos = sums_e > 0
            out_t = np.where(pos, talk / np.where(pos, sums_e, 1.0), uni)
        out_o = off
        if off is not None and len(off):
            off = off * self.valid[p]
            so = off.sum(axis=1, keepdims=True)
            uni = self.valid[p] / self.valid[p].sum(axis=1, keepdims=True)
            out_o = np.where(so > 0, off / np.where(so > 0, so, 1), uni)
        return out_t, out_o

    def factors(self, p: int, talk: np.ndarray, br: bool = False) -> np.ndarray:
        t = self.tree
        F = np.ones((t.N, t.n[p]))
        if not br:
            F[t.edge_child[p]] = talk
        F[t.rev_child[p]] = t.rev_mask[p]
        return F

    def reach(self, p: int, talk: np.ndarray) -> np.ndarray:
        t = self.tree
        F = self.factors(p, talk)
        R = np.ones((t.N, t.n[p]))
        for d in range(1, t.D + 1):
            ids = slice(t.dstart[d], t.dstart[d + 1])
            R[ids] = R[t.parent[ids]] * F[ids]
        return R

    def leaf_values(self, p: int, own_off, opp_reach: np.ndarray, opp_off):
        """Values at the leaves for player p (conditional on own type), and the offer counterfactual values q."""
        t, o = self.tree, 1 - p
        v = np.zeros((t.N, t.n[p]))
        for z_mid in set(t.leaf[z] for z in t.pay_nodes):
            zs = np.array([z for z in t.pay_nodes if t.leaf[z] == z_mid])
            U = self.pay[z_mid][p] if p == 0 else self.pay[z_mid][p].T
            v[zs] = opp_reach[zs] @ (self.cond[p] * U).T
        q = None
        if len(t.offer_nodes):
            ot, L = self.ot, t.offer_nodes
            rho = opp_reach[L] * self.p[o][None, :]                       # L x n_o, with the opponent's prior
            acc_opp = ot["ACC_B"] if p == 0 else ot["ACC_A"]
            c = rho @ np.ascontiguousarray(acc_opp.T)                     # L x m: acceptance mass of each offer
            q = self.qcoef[p] * c[t.off_leaf[p][:, None], self.cols[p]]   # p proposes, over p's menu
            wts = rho[t.off_leaf[o], t.off_type[o]][:, None] * opp_off   # the opponent proposes
            contrib = np.empty((len(wts), t.n[p]))
            for ty, idx in self.by_type[o]:
                contrib[idx] = wts[idx] @ self.HK[o][ty]
            v[L] = (self.pi if p == 1 else 1 - self.pi) * (t.off_agg[o] @ contrib)    # the opponent proposes
            v[L[t.off_leaf[p]], t.off_type[p]] += (own_off * q).sum(axis=1)       # (leaf, type) pairs are unique
        return v, q

    def evaluate(self, p: int, own, opp, br: bool = False) -> dict:
        """Player p's values under (own, opp), or its best response to opp if ``br``.

        Returns the prior-mean value, the per-node values, and either the counterfactual regrets or the
        best-response policy (one-hot, ties to the first edge within ``TIE_TOL``)."""
        t, o = self.tree, 1 - p
        opp_reach = self.reach(o, opp[0])
        own_off = own[1] if own is not None else None
        if br and len(t.offer_nodes):
            own_off = None
        v, q = self.leaf_values(p, own_off if own_off is not None else self._zeros_off(p), opp_reach, opp[1])
        br_off = None
        if br and q is not None:
            qv = np.where(self.valid[p], q, -np.inf)
            qmax = qv.max(axis=1)
            br_off = np.zeros_like(q)
            br_off[np.arange(len(q)), np.argmax(qv >= qmax[:, None] - TIE_TOL, axis=1)] = 1.0
            v[t.offer_nodes[t.off_leaf[p]], t.off_type[p]] += qmax
        F = self.factors(p, None if br else own[0], br=br)
        for d in range(t.D - 1, -1, -1):
            ids, starts, agg = t.internal[d]
            if not len(ids):
                continue
            C = slice(t.dstart[d + 1], t.dstart[d + 2])
            s = agg @ (F[C] * v[C])
            if br:
                mx = np.maximum.reduceat(v[C], starts, axis=0)
                s = np.where(t.own_dec[p][ids][:, None], mx, s)
            v[ids] = s
        out = {"value": float(self.p[p] @ v[0]), "v": v}
        ec, ep = t.edge_child[p], t.edge_parent[p]
        if br:
            pol = np.zeros((len(ec), t.n[p]))
            if len(ec):
                flag = v[ec] >= v[ep] - TIE_TOL
                pos = np.where(flag, np.arange(len(ec))[:, None], len(ec))
                firstpos = np.minimum.reduceat(pos, t.dec_start[p], axis=0)
                pol[firstpos, np.arange(t.n[p])[None, :]] = 1.0
            out["policy"] = (pol, br_off)
        else:
            out["regret"] = (v[ec] - v[ep],
                             None if q is None else (q - (own[1] * q).sum(axis=1, keepdims=True)) * self.valid[p])
        return out

    def value(self, prof) -> Tuple[float, float]:
        return self.evaluate(0, prof[0], prof[1])["value"], self.evaluate(1, prof[1], prof[0])["value"]

    def nashconv(self, prof) -> dict:
        u = self.value(prof)
        b = (self.evaluate(0, None, prof[1], br=True), self.evaluate(1, None, prof[0], br=True))
        eps = (b[0]["value"] - u[0], b[1]["value"] - u[1])
        return {"value": u, "br_value": (b[0]["value"], b[1]["value"]), "eps": eps, "nashconv": eps[0] + eps[1],
                "br_policy": (b[0]["policy"], b[1]["policy"])}


# -- CFR+ -------------------------------------------------------------------------------------------

def solve(game: Game, iterations: int, variant: str = "cfr+", warm=None, checkpoints: Tuple[int, ...] = (),
          on_checkpoint: Optional[Callable] = None):
    """Run CFR+ (or DCFR) for ``iterations``; returns the average profile. ``warm`` is a profile to start from.
    At each iteration in ``checkpoints``, ``on_checkpoint(it, average)`` is called; if it returns True the run
    stops there and that average is returned."""
    R, S, cur = [], [], []
    for p in (0, 1):
        if warm is None:
            r = (np.zeros((len(game.tree.edge_child[p]), game.tree.n[p])), game._zeros_off(p))
        else:
            r = (warm[p][0].copy(), None if warm[p][1] is None else warm[p][1].copy())
        R.append(list(r))
        S.append([np.zeros_like(r[0]), None if r[1] is None else np.zeros_like(r[1])])
        cur.append(game.normalise(p, r))
    t = game.tree
    for it in range(1, iterations + 1):
        for p in (0, 1):
            o = 1 - p
            own_reach = game.reach(p, cur[p][0])
            if variant == "dcfr":
                g = (it / (it + 1)) ** 2
                S[p][0] *= g
                if S[p][1] is not None:
                    S[p][1] *= g
                w = 1.0
            else:
                w = float(it)
            S[p][0] += w * own_reach[t.edge_parent[p]] * cur[p][0]
            if S[p][1] is not None:
                S[p][1] += w * own_reach[t.offer_nodes[t.off_leaf[p]], t.off_type[p]][:, None] * cur[p][1]
            reg = game.evaluate(p, cur[p], cur[o])["regret"]
            for k in (0, 1):
                if R[p][k] is None:
                    continue
                if variant == "dcfr":
                    R[p][k] += reg[k]
                    a = it ** 1.5 / (it ** 1.5 + 1)
                    R[p][k] = np.where(R[p][k] > 0, R[p][k] * a, R[p][k] * 0.5)
                else:
                    R[p][k] = np.maximum(R[p][k] + reg[k], 0.0)
            cur[p] = game.normalise(p, (np.maximum(R[p][0], 0), None if R[p][1] is None else np.maximum(R[p][1], 0)))
        if it in checkpoints and on_checkpoint is not None:
            avg = average(game, S)
            if on_checkpoint(it, avg):
                return avg
    return average(game, S)


def average(game: Game, S) -> tuple:
    return tuple(game.normalise(p, (S[p][0], S[p][1])) for p in (0, 1))


def as_regrets(prof) -> tuple:
    """A (one-hot) profile as warm-start regrets: 1 on its actions, 0 elsewhere."""
    return tuple((p[0].copy(), None if p[1] is None else p[1].copy()) for p in prof)


# -- Split Deal ------------------------------------------------------------------------------------

def offer_menus(tab: PrivateTables, sh: Shared, prune: bool = True):
    """Each seat's offers per own type, as allocation indices (A's share) in increasing order, walking last.

    With ``prune``, an offer is left out when walking or another offer is at least as good for the proposer's
    type against every responder: its gain is not positive, or another offer gains strictly more and is
    accepted by every responder type that accepts it, or another offer has the same gain and
    the same acceptance set (lower in the proposer's own share order, the order ``Cell.final`` breaks ties in).
    No best-response value changes, every equilibrium of the pruned game is one of the
    unpruned game, and the ladder's offers (``Cell.final``) are never left out (tested)."""
    n, A = len(tab.types), len(tab.allocs)
    walk = A
    menus = []
    for seat in (0, 1):
        rows = []
        for t in range(n):
            if seat == 0:
                gain = [sh.seat[t].gain[k] for k in range(A)]
                acc = [tab.acc[k] for k in range(A)]
                own = list(range(A))                    # the proposer's own share index, Cell.final's tie order
            else:
                gain = [sh.seat[t].gain[sh.mirror[k]] for k in range(A)]
                acc = [sum(1 << a for a in range(n) if sh.seat[a].gain[k] >= 0) for k in range(A)]
                own = [sh.mirror[k] for k in range(A)]
            if prune:
                keep = [k for k in range(A) if gain[k] > 0 and not any(
                    (gain[j] > gain[k] and acc[j] | acc[k] == acc[j]) or (gain[j] == gain[k] and acc[j] == acc[k]
                                                                          and own[j] < own[k]) for j in range(A) if j != k)]
            else:
                keep = list(range(A))
            rows.append(keep + [walk])
        M = max(len(r) for r in rows)
        KEEP = np.full((n, M), walk, dtype=int)
        VALID = np.zeros((n, M), dtype=bool)
        for t, r in enumerate(rows):
            KEEP[t, :len(r)] = r
            VALID[t, :len(r)] = True
        menus.append((KEEP, VALID))
    return menus


def split_game(tab: PrivateTables, st: Structure, q: int, pi: Fraction = G_PI, prune: bool = True) -> Game:
    n = len(tab.types)
    full = tab.full
    yes = tab.yes

    def b_will(owed):
        return st.b_answers == "all" or (st.b_answers == "reciprocal" and owed)

    def nxt(r, SA, SB, dec, owed):
        if r == 0:
            return ("END", SA, SB)
        if st.a_asks:
            return ("A", r, SA, SB, dec, owed)
        return after_a(r, SA, SB, dec, owed)

    def after_a(r, SA, SB, dec, owed):
        if st.b_asks:
            return ("B", r, SA, SB, dec, owed)
        return nxt(r - 1, SA, SB, dec, owed)

    def expand(s):
        k = s[0]
        if k == "END":
            return ("offer", s[1], s[2])
        r, SA, SB, dec, owed = s[1:6]
        if k == "A":
            ch = [("pass", after_a(r, SA, SB, dec, owed))]
            if b_will(owed):
                for j, y in enumerate(yes):
                    if SB & y and SB & ~y:
                        ch.append((("ask", j), ("RB", r, SA, SB, dec, owed, j)))
            return ("dec", 0, ch)
        if k == "RB":
            y = yes[s[6]]
            return ("rev", 1, [("yes", after_a(r, SA, SB & y, dec, owed), y),
                               ("no", after_a(r, SA, SB & ~y, dec, owed), full & ~y)])
        if k == "B":
            ch = [("pass", nxt(r - 1, SA, SB, dec, owed))]
            for j, y in enumerate(yes):
                if j in dec or not SA & y or not SA & ~y:
                    continue
                if st.a_answers == "all":
                    ch.append((("ask", j), ("RA", r, SA, SB, dec, owed, j)))
                elif st.a_answers == "none":
                    ch.append((("ask", j), nxt(r - 1, SA, SB, dec | {j}, False)))
                else:
                    ch.append((("ask", j), ("ANS", r, SA, SB, dec, owed, j)))
            return ("dec", 1, ch)
        if k == "ANS":
            j = s[6]
            return ("dec", 0, [("decline", nxt(r - 1, SA, SB, dec | {j}, False)),
                               ("answer", ("RA", r, SA, SB, dec, owed, j))])
        if k == "RA":
            y = yes[s[6]]
            return ("rev", 0, [("yes", nxt(r - 1, SA & y, SB, dec, True), y),
                               ("no", nxt(r - 1, SA & ~y, SB, dec, True), full & ~y)])
        raise ValueError(k)

    tree = Tree(nxt(q, full, full, frozenset(), True), expand, (n, n))
    sh = Shared(tab)
    m = len(tab.allocs) + 1
    GA, GB = np.zeros((n, m)), np.zeros((n, m))
    ACC_A, ACC_B = np.zeros((m, n)), np.zeros((m, n))
    for t in range(n):
        for k in range(m - 1):
            GA[t, k] = float(sh.seat[t].gain[k])
            GB[t, k] = float(sh.seat[t].gain[sh.mirror[k]])
            ACC_A[k, t] = float(sh.seat[t].gain[k] >= 0)
            ACC_B[k, t] = float((tab.acc[k] >> t) & 1)
    menus = offer_menus(tab, sh, prune)
    ot = {"GA": GA, "GB": GB, "ACC_A": ACC_A, "ACC_B": ACC_B, "HA": GA.T * ACC_A, "HB": GB.T * ACC_B,
          "KEEP": [menus[0][0], menus[1][0]], "VALID": [menus[0][1], menus[1][1]]}
    g = Game(tree, np.full((n, n), 1.0 / (n * n)), float(pi), offer_tables=ot)
    g.tab, g.structure, g.q, g.pi_exact, g.shared = tab, st, q, pi, sh
    return g


def tree_counts(game: Game) -> dict:
    """The counts of ``spec_numbers_cfr.tree_counts``, read off the built tree."""
    t = game.tree
    a_pts = b_pts = 0
    for z in t.dec_nodes[0]:
        a_pts += t.state[z][2].bit_count()
    for z in t.dec_nodes[1]:
        b_pts += t.state[z][3].bit_count()
    off = sum(t.leaf[z][0].bit_count() + t.leaf[z][1].bit_count() for z in t.offer_nodes)
    pairs = sum(t.leaf[z][0].bit_count() * t.leaf[z][1].bit_count() for z in t.offer_nodes)
    return {"talk_paths": len(t.offer_nodes), "a_points": a_pts, "b_points": b_pts, "offer_points": off,
            "leaf_pairs": pairs, "leaf_evaluations": 2 * game.m * pairs}


# -- the ladder on the tree ---------------------------------------------------------------------------

class Ladder:
    """A1, A3, B0 and B2 on one Split Deal game, as one-hot behaviour profiles on its tree, plus their exact
    values from the readers' own rollouts."""

    def __init__(self, game: Game, answer_ties: bool = False):
        self.g = game
        tab, st, q, pi, sh = game.tab, game.structure, game.q, game.pi_exact, game.shared
        self.n = n = len(tab.types)
        self.m = A1Model(sh, st, q, pi, answer_ties=answer_ties)
        self.b2s = {b: B2(sh, self.m, b) for b in range(n)}
        self.a3s = {a: A3(sh, self.m, self.b2s, a) for a in range(n)}
        self._beliefs()

    def _beliefs(self):
        """B2's belief K and A3's belief S3 at every node, by B2's and A3's own update rules."""
        t, m, st, yes = self.g.tree, self.m, self.g.structure, self.g.tab.yes
        K, S3 = [0] * t.N, [0] * t.N
        K[0] = S3[0] = self.g.tab.full
        for c in range(1, t.N):
            p = t.parent[c]
            ps, lab = t.state[p], t.label[c]
            k, s3 = K[p], S3[p]
            kind = ps[0]
            r, SA, SB, dec, owed = ps[1:6]
            if kind == "A":
                if b_will_answer(st, owed):
                    k = update_ask(m, k, r, SB, SA, dec, owed, None if lab == "pass" else lab[1])
            elif kind == "RB":
                y = yes[ps[6]]
                s3 = s3 & (y if lab == "yes" else ~y)
                if not s3:
                    s3 = SB & (y if lab == "yes" else ~y)
            elif kind == "B":
                choice = None if lab == "pass" else lab[1]
                narrowed = sum(1 << b for b in bits(s3) if self.b2s[b].step(r, "B", SB, SA, dec, owed, k)[1] == choice)
                s3 = narrowed or s3
                if choice is not None and st.a_answers == "none":
                    k = update_decline(m, k, r, SB, SA, dec, owed, choice)
            elif kind == "ANS":
                if lab == "decline":
                    k = update_decline(m, k, r, SB, SA, dec, owed, ps[6])
            elif kind == "RA":
                j = ps[6]
                y = yes[j]
                k = update_answer(m, k, r, SB, SA, dec, owed, j, lab == "yes", SA & y if lab == "yes" else SA & ~y)
            K[c], S3[c] = k, s3
        self.K, self.S3 = K, S3

    def _edge(self, node: int, label) -> int:
        t = self.g.tree
        for c in range(t.first[node], t.first[node] + t.nch[node]):
            if t.label[c] == label:
                return c
        raise KeyError((node, label))

    def profile(self, name: str):
        """One-hot behaviour strategy for "A1", "A3", "B0" or "B2"."""
        g, t = self.g, self.g.tree
        p = 0 if name[0] == "A" else 1
        talk = np.zeros((len(t.edge_child[p]), t.n[p]))
        off = g._zeros_off(p)
        st = g.structure
        for z in t.dec_nodes[p]:
            s = t.state[z]
            kind, r, SA, SB, dec, owed = s[:6]
            own = SA if p == 0 else SB
            for ty in range(t.n[p]):
                if not (own >> ty) & 1:
                    lab = t.label[t.first[z]]                    # unreachable for this type: first edge
                else:
                    lab = self._action(name, z, ty)
                talk[t.edge_of[p][self._edge(z, lab)], ty] = 1.0
        slot = [{int(k): i for i, k in enumerate(g.ot["KEEP"][p][ty]) if g.ot["VALID"][p][ty, i]}
                for ty in range(t.n[p])]
        for j, (li, ty) in enumerate(zip(t.off_leaf[p], t.off_type[p])):
            z = t.offer_nodes[li]
            k = self._offer(name, z, ty)
            off[j, slot[ty][g.m - 1 if k is None else k]] = 1.0
        return talk, off

    def _action(self, name, z, ty):
        t, st, m = self.g.tree, self.g.structure, self.m
        s = t.state[z]
        kind, r, SA, SB, dec, owed = s[:6]
        K, S3 = self.K[z], self.S3[z]
        if kind == "A":
            if not b_will_answer(st, owed):
                return "pass"
            if name == "A1":
                j = m.ask(ty, r, SB, SA, dec, owed)
            else:
                j = self.a3s[ty].step(r, "A", SB, SA, dec, owed, K, S3)[1]
            return "pass" if j is None else ("ask", j)
        if kind == "ANS":
            j = s[6]
            if name == "A1":
                ans = m.answers(ty, r, SB, SA, dec, owed, j)
            else:
                v_ans, v_dec = self.a3s[ty]._answer_values(r, SB, SA, dec, owed, K, S3, j)
                ans = v_ans > v_dec or (m.answer_ties and v_ans == v_dec)
            return "answer" if ans else "decline"
        if kind == "B":
            if name == "B0":
                j = self.g.shared.b0_question(SA, dec)
            else:
                j = self.b2s[ty].step(r, "B", SB, SA, dec, owed, K)[1]
            return "pass" if j is None else ("ask", j)
        raise ValueError(kind)

    def _offer(self, name, z, ty):
        SA, SB = self.g.tree.leaf[z]
        if name == "A1":
            return self.m.offer(ty, SB)
        if name == "A3":
            return self.a3s[ty].me.final(self.S3[z])[1]
        if name == "B0":
            return self.g.shared.offer(ty, SA)
        return self.b2s[ty].offer(self.K[z])

    def exact(self, pairing: str) -> Tuple[Fraction, Fraction]:
        """Exact (Y_A, Y_B) of "A1/B0", "A1/B2" or "A3/B2" from the readers' rollouts, means over type pairs."""
        sh, m, n, pi = self.g.shared, self.m, self.n, self.g.pi_exact
        ya = yb = Fraction(0)
        for ia in range(n):
            for b in range(n):
                if pairing == "A1/B0":
                    o = rollout_a1_b0(sh, m, ia, b)
                elif pairing == "A1/B2":
                    o = rollout_a1_b2(sh, m, ia, self.b2s[b])
                else:
                    o = rollout_a3_b2(sh, self.a3s[ia], b)
                ga, gb = sh.seat[ia].gain, sh.seat[b].gain
                ka, kb = o["a_deal"], o["b_deal"]
                ya += pi * (ga[ka] if ka is not None else 0) + (1 - pi) * (ga[kb] if kb is not None else 0)
                yb += pi * (gb[sh.mirror[ka]] if ka is not None else 0) + \
                    (1 - pi) * (gb[sh.mirror[kb]] if kb is not None else 0)
        return ya / (n * n), yb / (n * n)


# -- Kuhn poker (a known answer for the engine) ----------------------------------------------------------

def kuhn_game() -> Game:
    """Kuhn poker: cards 0 < 1 < 2, ante 1, one bet of 1; the first player's equilibrium value is -1/18."""
    def show(stake):
        return np.array([[0 if a == b else stake * (1 if a > b else -1) for b in range(3)] for a in range(3)],
                        dtype=float)
    const = lambda x: np.full((3, 3), float(x))                       # noqa: E731
    pay = [(show(1), -show(1)), (show(2), -show(2)), (const(-1), const(1)), (const(1), const(-1))]

    def expand(s):
        if s == "":
            return ("dec", 0, [("check", "c"), ("bet", "b")])
        if s == "c":
            return ("dec", 1, [("check", "cc"), ("bet", "cb")])
        if s == "cb":
            return ("dec", 0, [("fold", "cbf"), ("call", "cbc")])
        if s == "b":
            return ("dec", 1, [("fold", "bf"), ("call", "bc")])
        return ("pay", {"cc": 0, "cbc": 1, "bc": 1, "cbf": 2, "bf": 3}[s])

    P = (1 - np.eye(3)) / 6
    return Game(Tree("", expand, (3, 3)), P, pay=pay)


# -- the seconds-only pilot (decision 8) --------------------------------------------------------------

def _peak_gib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


def pilot(iterations: int = 20) -> int:
    """Times the parts of one SD3-E job on G's T3. Prints seconds (wall, and CPU of this process, which does not
    depend on how busy the machine is) and peak memory only; every value computed is discarded unread."""
    c0 = time.process_time()
    t0 = time.time()
    tab = g_tables(Instance(G_INSTANCE), "merged")
    g = split_game(tab, STRUCTURES["T3"], G_Q)
    t1 = time.time()
    print(f"tree: {g.tree.N:,} nodes, {len(g.tree.offer_nodes):,} offer leaves, built in {t1 - t0:.1f} s; "
          f"peak {_peak_gib():.2f} GiB", flush=True)
    lad = Ladder(g)
    t2 = time.time()
    profs = {nm: lad.profile(nm) for nm in ("A1", "A3", "B0", "B2")}
    t3 = time.time()
    print(f"ladder beliefs {t2 - t1:.1f} s, ladder profiles {t3 - t2:.1f} s; peak {_peak_gib():.2f} GiB", flush=True)
    t4 = time.time()
    c4 = time.process_time()
    solve(g, iterations)
    t5, c5 = time.time(), time.process_time()
    print(f"CFR+: {iterations} iterations in {t5 - t4:.1f} s wall, {c5 - c4:.1f} s CPU "
          f"({(t5 - t4) / iterations:.2f} s wall, {(c5 - c4) / iterations:.2f} s CPU each); peak {_peak_gib():.2f} GiB",
          flush=True)
    g.nashconv((profs["A3"], profs["B2"]))
    t6, c6 = time.time(), time.process_time()
    print(f"one NashConv evaluation (both best responses) {t6 - t5:.1f} s wall, {c6 - c5:.1f} s CPU; "
          f"peak {_peak_gib():.2f} GiB", flush=True)
    print(f"total {t6 - t0:.1f} s wall, {c6 - c0:.1f} s CPU; load average {os.getloadavg()[0]:.0f} "
          f"on {os.cpu_count()} cores")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot")
    pp.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args(argv)
    return pilot(args.iterations)


if __name__ == "__main__":
    sys.exit(main())
