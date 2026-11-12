"""ST-0f numpy planner: the open-world EIG ranking and soft update, vectorised.

    # parity against the Python path (1e-9) + timing; offline, a few minutes:
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_fastplan --parity
    PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.stress_fastplan --timing

The Python loop scores every candidate question with ``bed.expected_info_gain``,
a per-hypothesis Python loop over the whole posterior, and ``soft`` keeps all
N + 1 hypotheses alive, so at N = 5000 one game costs ~66 x 5000 Python calls per
turn. ``FastPlanner`` is built once per fixed pool (snapshot mode). It holds the
likelihood of every attribute and category question for every known plus
``UNKNOWN`` as two arrays, built with exactly the semantics of
``openworld.make_openworld_likelihood`` (its background marginals are the same
Python sums), and ranks a candidate list with one matrix product. It also builds
the episode's ``like`` and question bank once, instead of once per game.

``run_openworld_episode(planner=...)`` uses it; ``planner=None`` (the default)
is the untouched Python path, so every committed result is unchanged. The numpy
path differs from the Python one in floating-point summation order (pairwise vs
sequential), which the parity check bounds at 1e-9 -- and in one deliberate way:

**Tie-break.** ``bed.rank_questions`` sorts by (-EIG, qid), meaning ties to go to
the qid. But candidates of mathematically equal EIG (guesses of two equal-mass
twins, say) come out of the float sum a few ulp apart, so in practice the
summation order picks the winner, not the qid. Generated pools are full of such
ties, and the summation order of a BLAS matrix product can differ between
machines. ``FastPlanner.rank`` therefore rounds EIG to ``TIE_DECIMALS`` before
the sort, so a tie really goes to the qid, and a run does not depend on the
machine. The parity check compares it against the Python EIG under the same
rule (``SnapPythonPlanner``).
"""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bed import (ATTRIBUTE, ATTRIBUTE_BANK, CATEGORY, CATEGORY_PHRASE, GUESS,
                  Hypothesis, Question, expected_info_gain, posterior_update,
                  rank_questions)
from .openworld import (UNKNOWN, UNKNOWN_HYP, background_marginals,
                        build_openworld_bank, category_marginals,
                        make_openworld_likelihood)
from .openworld_backends import SoftPosterior

Matrix = Dict[str, Dict[str, float]]

TIE_DECIMALS = 9   # EIGs equal to 1e-9 are a tie, broken by qid


def tie_sorted(scored: List[Tuple[Question, float]]) -> List[Tuple[Question, float]]:
    """(-EIG rounded to ``TIE_DECIMALS``, qid): a tie goes to the qid."""
    return sorted(scored, key=lambda t: (-round(t[1], TIE_DECIMALS), t[0].qid))


class FastPlanner:
    """Vectorised EIG ranking for one fixed known set ``known_hyps``."""

    def __init__(self, known_hyps: Sequence[Hypothesis], matrix: Matrix,
                 eps: float = 0.05):
        self.known_hyps = list(known_hyps)
        self.eps = eps
        names = [h.name for h in self.known_hyps]
        self.col: Dict[str, int] = {n: i for i, n in enumerate(names)}
        self.col[UNKNOWN] = len(names)

        def clamp(x: float) -> float:
            return min(max(x, eps), 1.0 - eps)

        attr_bg = background_marginals(matrix, names)
        cat_bg = category_marginals(self.known_hyps)
        n1 = len(names) + 1
        self.attr_row = {aid: i for i, (aid, _) in enumerate(ATTRIBUTE_BANK)}
        self.A = np.empty((len(ATTRIBUTE_BANK), n1))
        for aid, i in self.attr_row.items():
            col = matrix[aid]
            self.A[i, :-1] = [col[n] for n in names]
            self.A[i, -1] = clamp(attr_bg[aid])
        cats = sorted(CATEGORY_PHRASE)
        self.cat_row = {c: i for i, c in enumerate(cats)}
        self.C = np.zeros((len(cats), n1))
        for j, h in enumerate(self.known_hyps):
            self.C[self.cat_row[h.category], j] = 1.0
        for c, i in self.cat_row.items():
            self.C[i, -1] = clamp(cat_bg.get(c, 0.0))
        # Built once per pool rather than once per game (same values either way).
        self.like = make_openworld_likelihood(matrix, self.known_hyps, eps=eps)
        self.bank = build_openworld_bank(self.known_hyps)

    # -- likelihood rows ---------------------------------------------------- #
    def _cols(self, names: Sequence[str]) -> np.ndarray:
        return np.fromiter((self.col[n] for n in names), dtype=np.intp, count=len(names))

    def rows(self, cands: Sequence[Question], names: Sequence[str]) -> np.ndarray:
        """p(yes | h, q) for each candidate (rows) over ``names`` (columns)."""
        cols = self._cols(names)
        out = np.empty((len(cands), len(names)))
        pos = None
        for r, q in enumerate(cands):
            if q.kind == ATTRIBUTE:
                out[r] = self.A[self.attr_row[q.key], cols]
            elif q.kind == CATEGORY:
                out[r] = self.C[self.cat_row[q.key], cols]
            else:  # GUESS: 1 on the named known, eps on UNKNOWN, 0 elsewhere
                if pos is None:
                    pos = {n: i for i, n in enumerate(names)}
                out[r] = 0.0
                if q.key in pos:
                    out[r, pos[q.key]] = 1.0
                if UNKNOWN in pos:
                    out[r, pos[UNKNOWN]] = self.eps
        return out

    # -- EIG ------------------------------------------------------------------ #
    @staticmethod
    def eig_rows(L: np.ndarray, p: np.ndarray) -> np.ndarray:
        """``bed.expected_info_gain`` for every row of ``L`` at once."""
        with np.errstate(divide="ignore", invalid="ignore"):
            nz = p > 0.0
            h_post = -float(np.sum(p[nz] * np.log2(p[nz])))
            p_yes = L @ p
            p_no = 1.0 - p_yes
            valid = (p_yes > 0.0) & (p_yes < 1.0)
            safe_y = np.where(valid, p_yes, 1.0)[:, None]
            safe_n = np.where(valid, p_no, 1.0)[:, None]
            py = p * L / safe_y
            pn = p * (1.0 - L) / safe_n
            hy = -np.sum(np.where(py > 0.0, py * np.log2(np.where(py > 0.0, py, 1.0)), 0.0),
                         axis=1)
            hn = -np.sum(np.where(pn > 0.0, pn * np.log2(np.where(pn > 0.0, pn, 1.0)), 0.0),
                         axis=1)
            e = h_post - (p_yes * hy + p_no * hn)
        return np.where(valid, e, 0.0)

    def eig_guesses(self, keys: Sequence[str], names: Sequence[str],
                    p: np.ndarray) -> np.ndarray:
        """``eig_rows`` for GUESS rows, in closed form: O(N) per call, not O(N) per guess.

        A guess row is 1 on the named known, ``eps`` on UNKNOWN and 0 elsewhere, so
        ``p_yes = p_h + eps p_U`` and the "yes" posterior has two non-zero entries.
        The "no" posterior is ``p_k / p_no`` on every other known, so its entropy
        follows from ``T = sum_known p log2 p`` and ``Q = sum_known p``:
        ``sum_{k != h} (p_k/p_no) log2(p_k/p_no) = ((T - t_h) - (Q - p_h) log2 p_no) / p_no``.
        Same algebra as ``eig_rows`` (p_no = 1 - p_yes, zero-mass terms dropped);
        only the floating-point summation order differs."""
        eps = self.eps
        with np.errstate(divide="ignore", invalid="ignore"):
            nz = p > 0.0
            h_post = -float(np.sum(p[nz] * np.log2(p[nz])))
            plogp = np.where(nz, p * np.log2(np.where(nz, p, 1.0)), 0.0)
            pos = {n: i for i, n in enumerate(names)}
            u = pos.get(UNKNOWN, -1)
            known = np.ones(len(names), dtype=bool)
            if u >= 0:
                known[u] = False
            T = float(np.sum(plogp[known]))
            Q = float(np.sum(p[known]))
            p_u = float(p[u]) if u >= 0 else 0.0
            idx = np.fromiter((pos.get(k, -1) for k in keys), dtype=np.intp, count=len(keys))
            hit = idx >= 0
            safe = np.where(hit, idx, 0)
            ph = np.where(hit, p[safe], 0.0)
            th = np.where(hit, plogp[safe], 0.0)
            p_yes = ph + eps * p_u
            p_no = 1.0 - p_yes
            valid = (p_yes > 0.0) & (p_yes < 1.0)
            sy = np.where(valid, p_yes, 1.0)
            sn = np.where(valid, p_no, 1.0)

            def xlog(x):
                return np.where(x > 0.0, x * np.log2(np.where(x > 0.0, x, 1.0)), 0.0)

            hy = -(xlog(ph / sy) + xlog(eps * p_u / sy))
            rest = ((T - th) - (Q - ph) * np.log2(sn)) / sn
            hn = -(rest + xlog((1.0 - eps) * p_u / sn))
            e = h_post - (p_yes * hy + p_no * hn)
        return np.where(valid, e, 0.0)

    def rank(self, cands: Sequence[Question], dist: Dict[str, float]
             ) -> List[Tuple[Question, float]]:
        """``bed.rank_questions`` with the ``tie_sorted`` tie-break. Attribute and
        category rows go through ``eig_rows``; guesses through ``eig_guesses``."""
        names = list(dist)
        p = np.fromiter(dist.values(), dtype=float, count=len(names))
        gi = [i for i, q in enumerate(cands) if q.kind == GUESS]
        oi = [i for i, q in enumerate(cands) if q.kind != GUESS]
        e = np.empty(len(cands))
        if oi:
            e[oi] = self.eig_rows(self.rows([cands[i] for i in oi], names), p)
        if gi:
            e[gi] = self.eig_guesses([cands[i].key for i in gi], names, p)
        return tie_sorted(list(zip(cands, e.tolist())))

    def rank_dense(self, cands: Sequence[Question], dist: Dict[str, float]
                   ) -> List[Tuple[Question, float]]:
        """The pre-closed-form ``rank`` (every row through ``eig_rows``), kept as
        the A/B reference for ``eig_guesses``."""
        names = list(dist)
        p = np.fromiter(dist.values(), dtype=float, count=len(names))
        e = self.eig_rows(self.rows(cands, names), p)
        return tie_sorted(list(zip(cands, e.tolist())))

    def eig(self, q: Question, dist: Dict[str, float]) -> float:
        names = list(dist)
        p = np.fromiter(dist.values(), dtype=float, count=len(names))
        return float(self.eig_rows(self.rows([q], names), p)[0])

    # -- soft update ---------------------------------------------------------- #
    def update(self, post: Dict[str, float], q: Question, answer_yes: bool
               ) -> Dict[str, float]:
        """``bed.posterior_update`` (attribute likelihoods clamped to [eps, 1-eps];
        category and guess exact)."""
        names = list(post)
        p = np.fromiter(post.values(), dtype=float, count=len(names))
        ly = self.rows([q], names)[0]
        if q.kind == ATTRIBUTE:
            ly = np.clip(ly, self.eps, 1.0 - self.eps)
        new = p * (ly if answer_yes else (1.0 - ly))
        total = float(new.sum())
        if total <= 0.0:
            n = len(names)
            return {name: 1.0 / n for name in names}
        return dict(zip(names, (new / total).tolist()))

    def soft_backend(self, known_hyps, by_name, like, unknown, p_unknown_prior, eps):
        return FastSoftPosterior(self, known_hyps, by_name, like, unknown,
                                 p_unknown_prior, eps)


class FastSoftPosterior(SoftPosterior):
    """``soft`` with its update done by ``FastPlanner.update``."""

    def __init__(self, planner: FastPlanner, *args):
        super().__init__(*args)
        self.planner = planner

    def observe(self, q: Question, answer_yes: bool) -> None:
        self.post = self.planner.update(self.post, q, answer_yes)


# --------------------------------------------------------------------------- #
# Parity                                                                       #
# --------------------------------------------------------------------------- #

class SnapPythonPlanner:
    """The Python EIG and the Python soft update, ranked with ``tie_sorted``:
    the reference the numpy path must reproduce game for game."""

    def __init__(self, fast: FastPlanner):
        self.known_hyps = fast.known_hyps
        self.like = fast.like
        self.bank = fast.bank
        self.by_name = {h.name: h for h in fast.known_hyps + [UNKNOWN_HYP]}

    def rank(self, cands, dist):
        return tie_sorted(rank_questions(cands, dist, self.by_name, self.like))

    def eig(self, q, dist):
        return expected_info_gain(q, dist, self.by_name, self.like)

    def soft_backend(self, known_hyps, by_name, like, unknown, p_unknown_prior, eps):
        return SoftPosterior(known_hyps, by_name, like, unknown, p_unknown_prior, eps)


class ParityPlanner:
    """Drives the ``SnapPythonPlanner`` path (so the trajectory is the Python
    one) while computing the numpy ranking and update on the same inputs and
    recording the gaps. An argmax disagreement can only come from two EIGs
    within 1e-13 of each other straddling a ``TIE_DECIMALS`` rounding boundary;
    its gap is recorded. Plugged in through the same ``planner`` hook."""

    def __init__(self, fast: FastPlanner):
        self.fast = fast
        self.known_hyps = fast.known_hyps
        self.like = fast.like
        self.bank = fast.bank
        self.by_name = {h.name: h for h in fast.known_hyps + [UNKNOWN_HYP]}
        self.max_eig_diff = 0.0
        self.max_post_diff = 0.0
        self.rank_calls = 0
        self.argmax_mismatch = 0
        self.max_mismatch_gap = 0.0

    def rank(self, cands, dist):
        py = tie_sorted(rank_questions(cands, dist, self.by_name, self.like))
        fa = self.fast.rank(cands, dist)
        pyd = {q.qid: e for q, e in py}
        for q, e in fa:
            self.max_eig_diff = max(self.max_eig_diff, abs(e - pyd[q.qid]))
        self.rank_calls += 1
        if py[0][0].qid != fa[0][0].qid:
            self.argmax_mismatch += 1
            self.max_mismatch_gap = max(self.max_mismatch_gap,
                                        abs(py[0][1] - pyd[fa[0][0].qid]))
        return py

    def eig(self, q, dist):
        e_py = expected_info_gain(q, dist, self.by_name, self.like)
        self.max_eig_diff = max(self.max_eig_diff, abs(self.fast.eig(q, dist) - e_py))
        return e_py

    def soft_backend(self, known_hyps, by_name, like, unknown, p_unknown_prior, eps):
        return _ParitySoft(self, known_hyps, by_name, like, unknown, p_unknown_prior, eps)


class _ParitySoft(SoftPosterior):
    def __init__(self, parity: ParityPlanner, *args):
        super().__init__(*args)
        self.parity = parity

    def observe(self, q, answer_yes):
        fast = self.parity.fast.update(self.post, q, answer_yes)
        self.post = posterior_update(self.post, q, answer_yes, self.by_name, self.like,
                                     eps=self.eps)
        d = max(abs(fast[n] - v) for n, v in self.post.items())
        self.parity.max_post_diff = max(self.parity.max_post_diff, d)


def _episode_key(ep) -> tuple:
    return (ep.outcome, ep.n_questions, tuple(s.qid for s in ep.steps),
            tuple(s.answer_yes for s in ep.steps))


def parity(games: int = 200, tol: float = 1e-9, n_generated: int = 342,
           seed: int = 9001) -> dict:
    """Two checks per (pool, backend, flip rate), each over ``games`` secrets.

    * **lockstep** -- the Python trajectory, with the numpy ranking, EIG and
      soft update evaluated on every same input: max |EIG diff|, max |posterior
      diff|, and argmax disagreements (allowed only at near-ties <= ``tol``).
    * **end-to-end** -- each secret played twice, by ``SnapPythonPlanner`` and by
      the numpy path, with identically seeded oracles: identical trajectories
      (questions, answers, outcome) and |p_unknown_max diff| <= ``tol``. A
      trajectory may part only in a game where lockstep saw a boundary straddle.

    Pools: the real 158 objects, and a generated pool of 158 + ``n_generated``
    at rho_real (seed ``seed``, outside any ST-1 grid seed)."""
    from .bed import load_hypotheses
    from .matrix import load_matrix
    from .openworld import noisy_matrix_oracle, run_openworld_episode
    from .bed import matrix_oracle
    from .stress_generator import build_pool, generate_objects, merge

    objs = load_hypotheses()
    matrix = load_matrix("llm_rl_scripts/twenty_questions/jev/matrix_jev.json")["matrix"]
    gen_pool, gen_matrix = build_pool(objs, matrix, len(objs) + n_generated, 0.2345, seed)
    novel, nrows = generate_objects(objs, matrix, 20, 0.2345, seed + 1, prefix="pnov")
    pools = {"real158": (list(objs), merge(matrix, nrows)),
             "gen{}".format(len(gen_pool)): (gen_pool, merge(gen_matrix, nrows))}
    rows = []
    ok = True
    for pname, (pool, mtx) in pools.items():
        fast = FastPlanner(pool, mtx)
        snap = SnapPythonPlanner(fast)
        rng = random.Random(seed)
        secrets = [rng.choice(pool) for _ in range(games - len(novel))] + list(novel)
        for backend in ("soft", "hard", "modeswitch"):
            for flip in (0.0, 0.10):
                def oracle_for(i):
                    if flip == 0.0:
                        return matrix_oracle(mtx)
                    return noisy_matrix_oracle(mtx, flip, seed=10_000 + i)

                par = ParityPlanner(fast)
                same = 0
                pu_diff = 0.0
                for i, t in enumerate(secrets):
                    kw = dict(backend=backend, tolerance=3)
                    run_openworld_episode(t, pool, mtx, 0.091, oracle_for(i),
                                          planner=par, **kw)
                    a = run_openworld_episode(t, pool, mtx, 0.091, oracle_for(i),
                                              planner=snap, **kw)
                    b = run_openworld_episode(t, pool, mtx, 0.091, oracle_for(i),
                                              planner=fast, **kw)
                    if _episode_key(a) == _episode_key(b):
                        same += 1
                        pu_diff = max(pu_diff, abs(a.p_unknown_max - b.p_unknown_max))
                cell_ok = (par.max_eig_diff <= tol and par.max_post_diff <= tol
                           and par.max_mismatch_gap <= tol and pu_diff <= tol
                           and (same == len(secrets) or par.argmax_mismatch > 0))
                ok &= cell_ok
                rows.append({"pool": pname, "backend": backend, "flip": flip,
                             "games": len(secrets),
                             "max_eig_diff": par.max_eig_diff,
                             "max_post_diff": par.max_post_diff,
                             "rank_calls": par.rank_calls,
                             "argmax_mismatch": par.argmax_mismatch,
                             "max_mismatch_gap": par.max_mismatch_gap,
                             "identical_trajectories": same,
                             "max_p_unknown_diff": pu_diff, "ok": cell_ok})
                print("  {:8s} {:10s} flip {:.2f}: max|dEIG| {:.1e}  max|dpost| {:.1e}  "
                      "argmax mismatches {}/{} (gap {:.1e})  identical {}/{}  {}".format(
                          pname, backend, flip, par.max_eig_diff, par.max_post_diff,
                          par.argmax_mismatch, par.rank_calls, par.max_mismatch_gap,
                          same, len(secrets), "OK" if cell_ok else "FAIL"))
    return {"tol": tol, "games": games, "cells": rows, "pass": ok}


def timing(sizes: Sequence[int] = (500, 2000, 5000), games: int = 5,
           seed: int = 9101) -> dict:
    """Seconds per game, Python vs numpy, on pools built from the *synthetic*
    category-coherent matrix (``matrix.synthetic_matrix``), so no ST-1 outcome is
    produced. Used to apply the ST-1 runtime cap before the grid starts."""
    from .bed import load_hypotheses, matrix_oracle
    from .matrix import synthetic_matrix
    from .openworld import run_openworld_episode
    from .stress_generator import build_pool

    objs = load_hypotheses()
    synth = synthetic_matrix(objs)
    out = []
    for n in sizes:
        pool, mtx = build_pool(objs, synth, n, 0.2345, seed)
        oracle = matrix_oracle(mtx)
        rng = random.Random(seed)
        secrets = [rng.choice(pool) for _ in range(games)]
        for backend in ("soft", "hard", "modeswitch"):
            t0 = time.time()
            fast = FastPlanner(pool, mtx)
            build = time.time() - t0
            t0 = time.time()
            for t in secrets:
                run_openworld_episode(t, pool, mtx, 0.091, oracle, backend=backend,
                                      tolerance=3, planner=fast)
            t_fast = (time.time() - t0) / games
            t_py = None
            if n <= 2000 or backend != "soft":
                t0 = time.time()
                for t in secrets[:2]:
                    run_openworld_episode(t, pool, mtx, 0.091, oracle, backend=backend,
                                          tolerance=3)
                t_py = (time.time() - t0) / 2
            row = {"n": n, "backend": backend, "s_per_game_fast": t_fast,
                   "s_per_game_python": t_py, "planner_build_s": build,
                   "est_cell_minutes_fast": t_fast * 1100 / 60}
            out.append(row)
            print("  N={:5d} {:10s} fast {:.3f} s/game  python {}  build {:.2f}s  "
                  "-> ~{:.1f} min per 1100-game run".format(
                      n, backend, t_fast,
                      "{:.3f} s/game".format(t_py) if t_py is not None else "(skipped)",
                      build, row["est_cell_minutes_fast"]))
    return {"games": games, "rows": out}


def main() -> None:
    import json
    import sys
    from .stress_harness import require_hash_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    require_hash_seed()
    res = None
    if args.parity:
        res = parity(games=args.games)
        print("PARITY {}".format("PASS" if res["pass"] else "FAIL"))
    elif args.timing:
        res = timing()
    if res is not None and args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print("wrote {}".format(args.out))
    if args.parity and not res["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
