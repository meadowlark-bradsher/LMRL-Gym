"""ST-2 post-hoc reads (NOT registered). Written after scoring to explain two
failures; nothing here changes a verdict in ``analysis.json``.

1. P5 paired check. The stream is identical across jobs, so a parent's in-pool game
   g can be compared between a d-job and the d=12 job of the same look-alike seed
   (same secret, same game index). For each P5 event (as in ``analyze_st2``), the
   paired difference in questions is averaged before and after the look-alike's
   admission. Also splits the registered adjusted value into the parent's raw change
   and the control drift (``raw - did``).
2. P5 games' internals: mean questions and by-name guesses of the parent's
   after-admission games at d vs d=12.
3. ``--replay``: re-plays one job pair (look-alike seed 1, flip 0, ``hard``, d=1 and
   d=12; ~100 s each, one core) and prints the question sequence of a few of the
   parent "Tree"'s games after its look-alike's admission.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.analyze_st2_posthoc --run-dir <run> [--replay]

Writes ``posthoc.json`` into the run directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
from typing import Dict, List

from .analyze_st2 import EXPERT, N_REAL, P5_FROM_GAME, derive, parent_tax_events

REF_D = 12
DS = (1, 2, 6, 8)
BACKENDS = ("soft", "hard", "modeswitch", "modeswitch_guard")
LA_SEEDS = (1, 2, 3)


def _load(run_dir: str, d: int, la: int, bk: str) -> dict:
    path = os.path.join(run_dir, "jobs", "exact_d{}_la{}_f0_s0_g0_{}.json.gz".format(d, la, bk))
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _real_index() -> Dict[str, int]:
    from .bed import load_hypotheses
    return {h.name: i for i, h in enumerate(load_hypotheses())}


def _events(rec: dict, ri: Dict[str, int]):
    """(look-alike index, parent index, admission game) for P5-eligible events,
    using the same eligibility as ``analyze_st2.parent_tax_events``."""
    rec = dict(rec, _real_index=ri)
    games, admitted, pidx = derive(rec)
    first: Dict[int, tuple] = {}
    for la, a in admitted.items():
        if la >= N_REAL:
            p = pidx[la]
            if (a >= P5_FROM_GAME and p in admitted and admitted[p] < a
                    and (p not in first or a < first[p][1])):
                first[p] = (la, a)
    return games, admitted, pidx, [(la, p, a) for p, (la, a) in first.items()]


def paired(run_dir: str, bk: str, d: int, ri) -> dict:
    before, after, nq_d, nq_r, ng_d, ng_r = [], [], [], [], [], []
    n_events = 0
    for la in LA_SEEDS:
        rd, rr = _load(run_dir, d, la, bk), _load(run_dir, REF_D, la, bk)
        gd, gr = rd["games"], rr["games"]
        assert gd["obj"] == gr["obj"], "streams differ"
        _, _, _, evs = _events(rd, ri)
        for _la, p, a in evs:
            pg = [i for i in range(P5_FROM_GAME, len(gd["obj"]))
                  if gd["obj"][i] == p and gd["in_pool"][i] and gr["in_pool"][i]]
            b = [gd["n_questions"][i] - gr["n_questions"][i] for i in pg if i < a]
            af = [gd["n_questions"][i] - gr["n_questions"][i] for i in pg if i > a]
            if not b or not af:
                continue
            n_events += 1
            before.append(statistics.mean(b))
            after.append(statistics.mean(af))
            for i in pg:
                if i > a:
                    nq_d.append(gd["n_questions"][i]); nq_r.append(gr["n_questions"][i])
                    ng_d.append(gd["n_guesses"][i]); ng_r.append(gr["n_guesses"][i])
    diffs = [x - y for x, y in zip(after, before)]
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    return {"backend": bk, "d": d, "ref_d": REF_D, "events": n_events,
            "paired_before": statistics.mean(before), "paired_after": statistics.mean(after),
            "paired_did": statistics.mean(diffs),
            "paired_did_ci": [statistics.mean(diffs) - 1.96 * se,
                              statistics.mean(diffs) + 1.96 * se],
            "after_games": len(nq_d),
            "after_mean_questions": [statistics.mean(nq_d), statistics.mean(nq_r)],
            "after_mean_guesses": [statistics.mean(ng_d), statistics.mean(ng_r)]}


def drift(run_dir: str) -> List[dict]:
    """Registered P5 value split into parent raw change and control drift."""
    a = json.load(open(os.path.join(run_dir, "analysis.json")))
    out = []
    for c in a["cells"]:
        if c["arm"] == "exact" and c["flip"] == 0.0:
            out.append({"backend": c["backend"], "d": c["d"], "did": c["p5_did"],
                        "raw": c["p5_raw"], "control_drift": c["p5_raw"] - c["p5_did"]})
    return sorted(out, key=lambda r: (r["backend"], r["d"]))


def replay(games_wanted=(1410, 1804, 2278, 2364)) -> dict:
    from . import stress_st2 as s2
    from .openworld import run_accumulation
    out: Dict[str, dict] = {}
    for d in (1, REF_D):
        job = dict(arm="exact", d=d, la_seed=1, flip=0.0, noise_seed=0, g=0.0, backend="hard",
                   oracle_seed=2_000_000 + 10_000 + 100 * d,
                   confusion_seed=3_000_000 + 10_000 + 100 * d)
        objects, mtx, parent_of = s2.build(job, "real")
        stream, _, sigma, theta = s2.stream_and_prior(len(objects), s2.GAMES)
        eps = []

        class Obs:
            def __call__(self, *a):
                pass

            def finish(self, ep):
                eps.append(ep)

        recs = run_accumulation(objects, mtx, stream, sigma, theta, max_questions=s2.BUDGET,
                                confidence=s2.CONFIDENCE, eps=s2.EPS, backend="hard",
                                tolerance=s2.K, oracle=s2.make_oracle(job, mtx, parent_of),
                                observer_factory=lambda known: Obs())
        for g in games_wanted:
            out.setdefault(str(g), {})[str(d)] = {
                "secret": recs[g].true_name, "outcome": recs[g].outcome,
                "steps": [s.qid for s in eps[g].steps]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    ri = _real_index()
    res = {"schema": "twenty_questions_stress_st2_posthoc_v1", "registered": False,
           "paired": [paired(args.run_dir, bk, d, ri) for bk in BACKENDS for d in DS],
           "drift": drift(args.run_dir)}
    if args.replay:
        res["replay"] = replay()
    for r in res["paired"]:
        print("{backend:17s} d={d:<2d} vs 12: events {events:3d}  before {paired_before:+.2f}  "
              "after {paired_after:+.2f}  DiD {paired_did:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
              "after-games q {q0:.2f}/{q1:.2f}  guesses {g0:.2f}/{g1:.2f}".format(
                  lo=r["paired_did_ci"][0], hi=r["paired_did_ci"][1],
                  q0=r["after_mean_questions"][0], q1=r["after_mean_questions"][1],
                  g0=r["after_mean_guesses"][0], g1=r["after_mean_guesses"][1], **r))
    for r in res["drift"]:
        print("{backend:17s} d={d:<2d} did {did:+.2f}  raw {raw:+.2f}  control drift "
              "{control_drift:+.2f}".format(**r))
    if args.replay:
        for g, v in res["replay"].items():
            for d, e in v.items():
                print(g, "d=" + d, e["secret"], e["outcome"], " ".join(e["steps"]))
    with open(os.path.join(args.run_dir, "posthoc.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", os.path.join(args.run_dir, "posthoc.json"))


if __name__ == "__main__":
    main()
