"""Evidence registry for the P1 fixed-position report.

Every number cited in ../PRIMARY_RESULTS.md is a deterministic function of the
frozen scores file ``../position_scores.jsonl`` (the live Jev moves were recorded
once there; re-deriving any statistic from it is exact, including the paired
bootstraps, which use a fixed seed). So all claims here are EXACT: a re-run that
drifts means the analysis code or the frozen data changed, not sampling noise.

    python generate.py            # rebuild evidence.json + PROVENANCE.md
    python generate.py --verify   # re-derive and compare to evidence.json

Run from this directory (needs only the stdlib).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "..", "position_scores.jsonl")
EVIDENCE = os.path.join(HERE, "evidence.json")
PROVENANCE = os.path.join(HERE, "PROVENANCE.md")
ABS_TOL = 1e-9  # exact: frozen data + fixed-seed bootstrap


def load():
    with open(SCORES) as f:
        return [json.loads(l) for l in f if l.strip()]


def _rate(rows, pred):
    rows = [r for r in rows if pred(r) is not None]
    return statistics.fmean([1.0 if pred(r) else 0.0 for r in rows]) if rows else None


def _median_nonmate_cpl(rows):
    xs = [r["cp_loss"] for r in rows
          if r.get("is_legal") and r.get("cp_loss") is not None and not r.get("best_is_mate")]
    return statistics.median(xs) if xs else None


def _paired(records, a, b, field, mate_filter=None):
    if mate_filter is not None:
        records = [r for r in records if r.get("best_is_mate") is mate_filter]
    by = defaultdict(dict)
    for r in records:
        by[r["position_id"]][r["arm"]] = r
    pairs = []
    for pid, m in by.items():
        ra, rb = m.get(a), m.get(b)
        if not ra or not rb or ra.get(field) is None or rb.get(field) is None:
            continue
        pairs.append((float(ra[field]) - float(rb[field]), ra.get("source_game", pid)))
    diffs = [d for d, _ in pairs]
    point = statistics.fmean(diffs) if diffs else 0.0
    by_c = defaultdict(list)
    for d, c in pairs:
        by_c[c].append(d)
    clusters = list(by_c)
    rng = random.Random(0)
    means = []
    for _ in range(5000):
        s = []
        for _ in range(len(clusters)):
            s.extend(by_c[rng.choice(clusters)])
        if s:
            means.append(statistics.fmean(s))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": len(pairs)}


def compute():
    recs = load()
    arms = {a: [r for r in recs if r["arm"] == a] for a in {r["arm"] for r in recs}}
    tactical = {a: [r for r in rows if r["phase"] == "tactical"] for a, rows in arms.items()}

    claims = {}

    def c(cid, asserts, cited_in, value):
        claims[cid] = {"asserts": asserts, "cited_in": cited_in, "value": value,
                       "tol": {"kind": "abs", "value": ABS_TOL}, "kind": "exact"}

    for a in ("jev", "heuristic_legal", "uniform_legal"):
        rows = arms[a]
        c("illegal.%s" % a, "illegal-move rate for %s" % a, ["results table"],
          _rate(rows, lambda r: not r["is_legal"]))
        c("match1.%s" % a, "best-move match rate for %s" % a, ["results table"],
          _rate(rows, lambda r: r.get("match_best")))
        c("blunder.%s" % a, "blunder (>=300cp loss) rate for %s" % a, ["results table"],
          _rate(rows, lambda r: r.get("blunder")))
        c("missedmate.%s" % a, "missed-forced-mate rate for %s" % a, ["results table"],
          _rate([r for r in rows if r.get("best_is_mate")], lambda r: r.get("lost_forced_mate")))
        c("medcpl.%s" % a, "median non-mate centipawn loss for %s" % a, ["results table"],
          _median_nonmate_cpl(rows))
        c("tac.match1.%s" % a, "tactical best-move match rate for %s" % a, ["tactical para"],
          _rate(tactical[a], lambda r: r.get("match_best")))
        c("tac.blunder.%s" % a, "tactical blunder rate for %s" % a, ["tactical para"],
          _rate(tactical[a], lambda r: r.get("blunder")))

    # Confidence calibration (Jev only): low vs high confidence tertile.
    jev_conf = sorted([r for r in arms["jev"] if r.get("confidence") is not None],
                      key=lambda r: r["confidence"])
    n = len(jev_conf)
    lo, hi = jev_conf[: n // 3], jev_conf[-(n // 3):]
    for name, grp in (("low", lo), ("high", hi)):
        c("conf.%s.match1" % name, "match@1 in %s-confidence tertile" % name, ["calibration para"],
          _rate(grp, lambda r: r.get("match_best")))
        c("conf.%s.medcpl" % name, "median non-mate cpL in %s-confidence tertile" % name,
          ["calibration para"], _median_nonmate_cpl(grp))

    # Paired comparisons (fixed-seed cluster bootstrap -> exact).
    c("paired.jev_vs_uniform.match1", "jev-uniform paired match@1 diff + 95% CI",
      ["paired table"], _paired(recs, "jev", "uniform_legal", "match_best"))
    c("paired.jev_vs_heuristic.match1", "jev-heuristic paired match@1 diff + 95% CI",
      ["paired table"], _paired(recs, "jev", "heuristic_legal", "match_best"))
    c("paired.jev_vs_heuristic.blunder", "jev-heuristic paired blunder diff + 95% CI",
      ["paired table"], _paired(recs, "jev", "heuristic_legal", "blunder"))
    return claims


def write_provenance(claims):
    lines = ["# PROVENANCE (generated from evidence.json -- do not edit by hand)\n",
             "Source: `position_scores.jsonl` (600 records: 200 positions x 3 arms).\n",
             "All claims are exact functions of that frozen file (abs tol {}).\n".format(ABS_TOL),
             "| claim | asserts | value |", "|-------|---------|-------|"]
    for cid, rec in sorted(claims.items()):
        v = rec["value"]
        if isinstance(v, dict) and "point" in v:
            vs = "{:+.4f} [{:+.4f}, {:+.4f}] (n={})".format(v["point"], v["ci_lo"], v["ci_hi"], v["n"])
        elif isinstance(v, float):
            vs = "{:.4f}".format(v)
        else:
            vs = str(v)
        lines.append("| `{}` | {} | {} |".format(cid, rec["asserts"], vs))
    with open(PROVENANCE, "w") as f:
        f.write("\n".join(lines) + "\n")


def _close(a, b, tol):
    if isinstance(a, dict):
        return all(_close(a[k], b[k], tol) for k in ("point", "ci_lo", "ci_hi") if k in a) and a["n"] == b["n"]
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    fresh = compute()
    if args.verify:
        with open(EVIDENCE) as f:
            stored = json.load(f)
        ok = True
        for cid, rec in sorted(stored.items()):
            got = fresh[cid]["value"]
            want = rec["value"]
            good = _close(got, want, rec["tol"]["value"])
            ok = ok and good
            print("  [{}] {}".format("pass" if good else "FAIL", cid))
            if not good:
                print("      stored={} regenerated={}".format(want, got))
        print("\n{}".format("ALL CLAIMS VERIFY" if ok else "SOME CLAIMS DRIFTED"))
        return 0 if ok else 1
    with open(EVIDENCE, "w") as f:
        json.dump(fresh, f, indent=2, sort_keys=True)
    write_provenance(fresh)
    print("wrote {} claims -> evidence.json + PROVENANCE.md".format(len(fresh)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
