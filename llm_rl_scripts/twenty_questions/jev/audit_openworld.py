"""ENV-703 — score the backends on auditability (the axis ENV-702 left out).

ENV-702 measured the axes a graded posterior is good at. This module measures the one
the discrete backends exist for: a replayable, append-only, verifiable op-log that can
answer *why* each hypothesis was eliminated or revived.

A ``GameAuditor`` rides the open-world loop through the optional ``observer`` hook
(``openworld.run_openworld_episode``) and reconstructs a canonical op-log **non-
invasively** — purely from the backend's public survivor-set transitions, so no backend
is modified. It then scores, per game:

* **replay cost** — the op-log is sparse (ELIMINATE/REINSTATE records) and replays the
  final survivor set exactly *without* the likelihood model; ``soft`` has no survivor
  set, so it induces no op-log and must store a dense posterior instead;
* **verifiability** — every ELIMINATE's precondition re-checks against the committed
  matrix with an independently rebuilt likelihood, and a hash chain over the records
  makes any edit detectable;
* **answer-the-why** — every ELIMINATE record carries the concrete ``(qid, answer)``
  that struck the hypothesis; a REINSTATE's reason is local (structured killers) or only
  a global scalar race (``retract``'s margin).

    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.audit_openworld --self-test
    PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.audit_openworld --run --out <dir>

See ``OPENWORLD_AUDIT_PREREG.md`` for the metric definitions and pre-registered
predictions, and ``analyze_audit.py`` for the table and figure.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

from .bed import Hypothesis, Question, load_hypotheses
from .matrix import load_matrix
from .openworld import (UNKNOWN, UNKNOWN_HYP, fit_pitman_yor, make_stream,
                        make_openworld_likelihood, run_accumulation)

ELIMINATE = "ELIMINATE"
REINSTATE = "REINSTATE"


def _rec_hash(prev: str, seq: int, turn: int, op: str, hyp: str, cause: str) -> str:
    """One link of the append-only chain. Any altered or dropped record changes every
    downstream hash, so the chain is tamper-evident."""
    payload = "{}|{}|{}|{}|{}|{}".format(prev, seq, turn, op, hyp, cause)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cause_str(cause) -> str:
    return "{}={}".format(cause[0], cause[1])


def verify_chain(records: List[dict]) -> bool:
    """Recompute the hash chain from scratch; True iff every link matches."""
    prev = "GENESIS"
    for r in records:
        h = _rec_hash(prev, r["seq"], r["turn"], r["op"], r["hyp"],
                      _cause_str(r["cause"]))
        if h != r["hash"]:
            return False
        prev = h
    return True


def replay(initial: set, records: List[dict]) -> set:
    """Reconstruct the final survivor set from the op-log alone — no likelihood model,
    no matrix, just apply ELIMINATE/REINSTATE in order to the initial known set."""
    surv = set(initial)
    for r in records:
        if r["op"] == ELIMINATE:
            surv.discard(r["hyp"])
        else:
            surv.add(r["hyp"])
    return surv


class GameAuditor:
    """Observer for one game. Reconstructs the op-log from survivor transitions and an
    independent likelihood recheck; ``finish`` computes the per-game audit metrics and
    appends them to the shared ``collect`` list."""

    def __init__(self, known_hyps: Sequence[Hypothesis],
                 matrix: Dict[str, Dict[str, float]], eps: float, kill: float,
                 collect: List[dict], backend_name: str):
        self.initial = {h.name for h in known_hyps}
        self.by_name = {h.name: h for h in known_hyps}
        self.by_name[UNKNOWN] = UNKNOWN_HYP
        self.like_v = make_openworld_likelihood(matrix, known_hyps, eps=eps)
        self.kill = kill
        self.collect = collect
        self.records: List[dict] = []
        self._seq = 0
        self._prev = "GENESIS"
        self.backend_name = backend_name
        self.discrete = backend_name != "soft"   # soft keeps no survivor set
        self.last_survivors: set = set(self.initial)
        self.n_turns = 0

    def _append(self, turn: int, op: str, hyp: str, cause, verified: bool) -> None:
        h = _rec_hash(self._prev, self._seq, turn, op, hyp, _cause_str(cause))
        self.records.append({"seq": self._seq, "turn": turn, "op": op, "hyp": hyp,
                             "cause": cause, "verified": verified, "hash": h})
        self._prev = h
        self._seq += 1

    def _verify_elim(self, q: Question, ans: bool, hyp: str) -> bool:
        """Independent recheck: does the stated answer genuinely contradict the
        hypothesis under a freshly built likelihood from the committed matrix?"""
        p_yes = self.like_v(q, self.by_name[hyp])
        p_obs = p_yes if ans else (1.0 - p_yes)
        return p_obs < self.kill

    def __call__(self, turn: int, q: Question, ans: bool, bk, surv_before) -> None:
        self.n_turns = turn
        if not hasattr(bk, "survivors"):      # soft: no survivor set, no op-log
            return
        surv_after = set(bk.survivors)
        eliminated = (surv_before or set()) - surv_after
        reinstated = surv_after - (surv_before or set())
        for n in sorted(eliminated):
            self._append(turn, ELIMINATE, n, (q.qid, ans),
                         self._verify_elim(q, ans, n))
        for n in sorted(reinstated):
            killers = getattr(bk, "killers", None)
            if killers is not None and n in killers:
                cause = ("killers", sorted(killers[n]))   # local, checkable
            else:
                cause = ("margin", None)                  # retract: global scalar race
            self._append(turn, REINSTATE, n, cause, True)
        self.last_survivors = surv_after

    def finish(self, ep) -> None:
        discrete = self.discrete
        n_elim = sum(1 for r in self.records if r["op"] == ELIMINATE)
        n_reinstate = sum(1 for r in self.records if r["op"] == REINSTATE)
        n_ops = len(self.records)
        row = {
            "backend": self.backend_name, "in_pool": ep.in_pool,
            "outcome": ep.outcome, "n_questions": ep.n_questions,
            "discrete": discrete, "n_ops": n_ops, "n_elim": n_elim,
            "n_reinstate": n_reinstate,
        }
        if discrete:
            replayed = replay(self.initial, self.records)
            row["replay_exact"] = (replayed == self.last_survivors)
            row["chain_ok"] = verify_chain(self.records)
            # storage to replay the committed state: one record per op (sparse).
            row["log_cells"] = n_ops
            row["verifiable_frac"] = (
                sum(1 for r in self.records
                    if r["op"] == ELIMINATE and r["verified"]) / n_elim
                if n_elim else None)
            # answer-the-why: every ELIMINATE carries a concrete (qid, answer).
            row["why_frac"] = (
                sum(1 for r in self.records
                    if r["op"] == ELIMINATE and r["cause"][1] is not None) / n_elim
                if n_elim else None)
            local = [r for r in self.records
                     if r["op"] == REINSTATE and r["cause"][0] == "killers"]
            row["reinstate_local_frac"] = (len(local) / n_reinstate
                                           if n_reinstate else None)
        else:
            # soft: no op-log. The only replayable trace is the dense posterior — a full
            # weight vector every turn (|H_known| + 1 catch-all cells per turn).
            row["replay_exact"] = False
            row["chain_ok"] = False
            row["log_cells"] = ep.n_questions * (len(self.initial) + 1)
            row["verifiable_frac"] = None     # no discrete decision to verify
            row["why_frac"] = None            # no elimination event to explain
            row["reinstate_local_frac"] = None
        self.collect.append(row)


def make_factory(matrix, eps, kill, collect, backend_name):
    def factory(known_hyps):
        return GameAuditor(known_hyps, matrix, eps, kill, collect, backend_name)
    return factory


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def _mean(vals: List[Optional[float]]):
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: List[dict]) -> dict:
    """Mean each metric over the games of one cell. Fractions average only over games
    where they are defined (i.e. games with the relevant op)."""
    n = len(rows)
    nz = [r for r in rows if r["n_ops"] > 0]        # games that actually acted
    return {
        "games": n,
        "discrete": rows[0]["discrete"] if rows else None,
        "ops_per_game": _mean([r["n_ops"] for r in rows]),
        "elim_per_game": _mean([r["n_elim"] for r in rows]),
        "reinstate_per_game": _mean([r["n_reinstate"] for r in rows]),
        "log_cells_per_game": _mean([r["log_cells"] for r in rows]),
        "replay_exact_frac": _mean([1.0 if r["replay_exact"] else 0.0 for r in nz])
                             if nz else None,
        "chain_ok_frac": _mean([1.0 if r["chain_ok"] else 0.0 for r in nz])
                         if nz else None,
        "verifiable_frac": _mean([r["verifiable_frac"] for r in rows]),
        "why_frac": _mean([r["why_frac"] for r in rows]),
        "reinstate_local_frac": _mean([r["reinstate_local_frac"] for r in rows]),
    }


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def _knobs(bk: str, margins, tolerances):
    if bk == "retract":
        return [("margin", m) for m in margins]
    if bk in ("tombstone", "modeswitch"):
        return [("tolerance", int(t)) for t in tolerances]
    return [(None, None)]


def run_audit(matrix_path: str, games: int, zipf_s: float, seed: int,
              confidence: float, eps: float, kill: float, backends: List[str],
              levels: List[float], n_noise_seeds: int, margins: List[float],
              tolerances: List[int], outdir: str) -> str:
    objs = load_hypotheses()
    matrix = load_matrix(matrix_path)["matrix"]
    stream, _ = make_stream(objs, games, zipf_s, seed)
    seen: set = set()
    new_flags = []
    for idx in stream:
        new_flags.append(objs[idx].name not in seen)
        seen.add(objs[idx].name)
    sigma, theta, _ = fit_pitman_yor(new_flags)

    cells = []
    print("backend       noise  ops/g  logcells/g  replay  verify   why  reinst_local")
    for bk in backends:
        for kname, kval in _knobs(bk, margins, tolerances):
            for lvl in levels:
                rows: List[dict] = []
                nseeds = 1 if lvl == 0.0 else n_noise_seeds
                for ns in range(1, nseeds + 1):
                    collect: List[dict] = []
                    factory = make_factory(matrix, eps, kill, collect, bk)
                    kw = {kname: kval} if kname else {}
                    run_accumulation(objs, matrix, stream, sigma, theta,
                                     confidence=confidence, eps=eps,
                                     oracle_noise=lvl, noise_seed=ns, backend=bk,
                                     observer_factory=factory, **kw)
                    rows.extend(collect)
                a = aggregate(rows)
                label = bk if kname is None else "{}@{:g}".format(bk, kval)
                cells.append({"backend": bk,
                              "margin": kval if kname == "margin" else None,
                              "tolerance": kval if kname == "tolerance" else None,
                              "oracle_noise": lvl, "agg": a})

                def f(x, d="  -  "):
                    return d if x is None else "{:.3f}".format(x)
                print("{:12s} {:4.0f}%  {:5.2f}  {:9.1f}  {:>6s}  {:>6s} {:>5s}  {:>6s}".format(
                    label, lvl * 100, a["ops_per_game"], a["log_cells_per_game"],
                    f(a["replay_exact_frac"]), f(a["verifiable_frac"]),
                    f(a["why_frac"]), f(a["reinstate_local_frac"])))

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(outdir, "openworld_audit_{}".format(stamp))
    os.makedirs(outpath, exist_ok=True)
    payload = {
        "schema": "twenty_questions_openworld_audit_v1",
        "created": stamp,
        "config": {"matrix": os.path.basename(matrix_path), "games": games,
                   "zipf_s": zipf_s, "seed": seed, "confidence": confidence,
                   "eps": eps, "kill": kill, "backends": backends, "levels": levels,
                   "n_noise_seeds": n_noise_seeds, "margins": margins,
                   "tolerances": tolerances},
        "pitman_yor_fit": {"sigma": sigma, "theta": theta},
        "cells": cells,
    }
    with open(os.path.join(outpath, "summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote {}".format(outpath))
    return outpath


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def self_test() -> None:
    """Prove the audit machinery on a real (noisy) stream: replay-exactness, the
    precondition verification, the planted-tamper catch, and the soft asymmetry."""
    objs = load_hypotheses()
    from .matrix import synthetic_matrix
    matrix = synthetic_matrix(objs)
    stream, _ = make_stream(objs, 300, 1.1, seed=7)
    seen: set = set()
    flags = []
    for idx in stream:
        flags.append(objs[idx].name not in seen)
        seen.add(objs[idx].name)
    sigma, theta, _ = fit_pitman_yor(flags)

    print("1. discrete backends: replay exact, every strike verifiable, chain ok")
    for bk in ("hard", "retract", "tombstone", "modeswitch"):
        collect: List[dict] = []
        factory = make_factory(matrix, 0.05, 0.5, collect, bk)
        run_accumulation(objs, matrix, stream, sigma, theta, oracle_noise=0.10,
                         noise_seed=1, backend=bk, observer_factory=factory)
        acted = [r for r in collect if r["n_ops"] > 0]
        assert acted, bk
        assert all(r["replay_exact"] for r in acted), (bk, "replay")
        assert all(r["chain_ok"] for r in acted), (bk, "chain")
        vf = [r["verifiable_frac"] for r in collect if r["verifiable_frac"] is not None]
        assert vf and min(vf) == 1.0, (bk, "verifiable", min(vf) if vf else None)
        wf = [r["why_frac"] for r in collect if r["why_frac"] is not None]
        assert wf and min(wf) == 1.0, (bk, "why")
        tot_ops = sum(r["n_ops"] for r in collect)
        tot_rein = sum(r["n_reinstate"] for r in collect)
        print("   {:11s} games_acted={:3d} total_ops={:4d} reinstates={:3d}  OK".format(
            bk, len(acted), tot_ops, tot_rein))

    print("2. reinstate reason is local for structured, global for scalar")
    for bk, want_local in (("tombstone", True), ("modeswitch", True), ("retract", False)):
        collect = []
        factory = make_factory(matrix, 0.05, 0.5, collect, bk)
        run_accumulation(objs, matrix, stream, sigma, theta, oracle_noise=0.10,
                         noise_seed=1, backend=bk, observer_factory=factory)
        fr = [r["reinstate_local_frac"] for r in collect
              if r["reinstate_local_frac"] is not None]
        if fr:
            got = min(fr) == 1.0
            assert got == want_local, (bk, "reinstate_local", fr[:3])
            print("   {:11s} reinstate_local={} (games with reinstates={})".format(
                bk, want_local, len(fr)))
        else:
            print("   {:11s} no reinstates on this stream (skip)".format(bk))

    print("3. soft induces no op-log (dense trace only)")
    collect = []
    factory = make_factory(matrix, 0.05, 0.5, collect, "soft")
    run_accumulation(objs, matrix, stream, sigma, theta, oracle_noise=0.10,
                     noise_seed=1, backend="soft", observer_factory=factory)
    assert all(r["n_ops"] == 0 for r in collect), "soft produced ops"
    assert all(not r["replay_exact"] for r in collect), "soft replay"
    assert all(r["verifiable_frac"] is None for r in collect), "soft verify"
    dense = [r["log_cells"] for r in collect if r["n_questions"] > 0]
    assert dense and min(dense) > 0, "soft dense trace"
    print("   soft: ops=0 all games, replay_exact=False, dense log_cells>0  OK")

    print("4. planted tamper breaks the chain")
    collect = []
    factory = make_factory(matrix, 0.05, 0.5, collect, "modeswitch")
    run_accumulation(objs, matrix, stream, sigma, theta, oracle_noise=0.10,
                     noise_seed=1, backend="modeswitch", observer_factory=factory)
    # rebuild one game's records and flip a hypothesis name mid-log.
    au = GameAuditor([objs[i] for i in range(3)], matrix, 0.05, 0.5, [], "modeswitch")
    au._append(1, ELIMINATE, objs[0].name, ("q1", True), True)
    au._append(2, ELIMINATE, objs[1].name, ("q2", False), True)
    assert verify_chain(au.records)
    au.records[0]["hyp"] = objs[2].name        # tamper
    assert not verify_chain(au.records), "tamper not caught"
    print("   edited record detected by hash chain  OK")

    print("ALL AUDIT SELF-TESTS PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--matrix", default="llm_rl_scripts/twenty_questions/jev/matrix_jev.json")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--kill", type=float, default=0.5)
    ap.add_argument("--backends", default="soft,hard,retract,tombstone,modeswitch")
    ap.add_argument("--levels", default="0.0,0.05,0.10")
    ap.add_argument("--noise-seeds", type=int, default=2)
    ap.add_argument("--margins", default="4")
    ap.add_argument("--tolerances", default="3")
    ap.add_argument("--out", default="llm_rl_scripts/twenty_questions/jev/results")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.run:
        run_audit(args.matrix, args.games, args.zipf, args.seed, args.confidence,
                  args.eps, args.kill, args.backends.split(","),
                  [float(x) for x in args.levels.split(",")], args.noise_seeds,
                  [float(x) for x in args.margins.split(",")],
                  [int(x) for x in args.tolerances.split(",")], args.out)
        return
    ap.error("pass --self-test or --run")


if __name__ == "__main__":
    main()
