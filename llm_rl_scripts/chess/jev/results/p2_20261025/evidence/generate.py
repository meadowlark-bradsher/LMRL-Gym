"""Evidence registry for the P2 report (../P2_RESULTS.md).

Every cited number is a deterministic function of the frozen
``../combined_scores.jsonl`` (P1 arms + the four P2 LLM arms; the live LLM moves
were recorded once there). Paired bootstraps use a fixed seed, so all claims are
EXACT.

    python generate.py            # rebuild evidence.json + PROVENANCE.md
    python generate.py --verify   # re-derive and compare
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "..", "combined_scores.jsonl")
EVIDENCE = os.path.join(HERE, "evidence.json")
PROVENANCE = os.path.join(HERE, "PROVENANCE.md")
ABS_TOL = 1e-9

ARMS = ["jev", "gpt4o_constrained", "gpt4o_free", "claude_constrained",
        "claude_free", "heuristic_legal", "uniform_legal"]


def load():
    with open(SCORES) as f:
        return [json.loads(l) for l in f if l.strip()]


def _rate(rows, pred):
    rows = [r for r in rows if pred(r) is not None]
    return statistics.fmean([1.0 if pred(r) else 0.0 for r in rows]) if rows else None


def _median_nonmate(rows):
    xs = [r["cp_loss"] for r in rows
          if r.get("is_legal") and r.get("cp_loss") is not None and not r.get("best_is_mate")]
    return statistics.median(xs) if xs else None


def _paired(records, a, b, field):
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
    byc = defaultdict(list)
    for d, c in pairs:
        byc[c].append(d)
    clusters = list(byc)
    rng = random.Random(0)
    means = []
    for _ in range(5000):
        s = []
        for _ in range(len(clusters)):
            s.extend(byc[rng.choice(clusters)])
        if s:
            means.append(statistics.fmean(s))
    means.sort()
    return {"point": point, "ci_lo": means[int(0.025 * len(means))],
            "ci_hi": means[int(0.975 * len(means))], "n": len(pairs)}


def compute():
    recs = load()
    arms = {a: [r for r in recs if r["arm"] == a] for a in ARMS}
    claims = {}

    def c(cid, asserts, value):
        claims[cid] = {"asserts": asserts, "value": value,
                       "tol": {"kind": "abs", "value": ABS_TOL}, "kind": "exact"}

    for a in ARMS:
        rows = arms[a]
        if not rows:
            continue
        c("illegal.%s" % a, "illegal rate for %s" % a, _rate(rows, lambda r: not r["is_legal"]))
        c("match1.%s" % a, "best-move match for %s" % a, _rate(rows, lambda r: r.get("match_best")))
        c("blunder.%s" % a, "blunder rate for %s" % a, _rate(rows, lambda r: r.get("blunder")))
        c("medcpl.%s" % a, "median non-mate cpL for %s" % a, _median_nonmate(rows))

    for base in ["gpt4o_constrained", "claude_constrained"]:
        for field in ["match_best", "blunder"]:
            c("paired.jev_vs_%s.%s" % (base, field),
              "jev-%s paired %s diff + CI" % (base, field),
              _paired(recs, "jev", base, field))
    return claims


def write_provenance(claims):
    lines = ["# PROVENANCE (generated from evidence.json -- do not edit by hand)\n",
             "Source: `combined_scores.jsonl` (P1 arms + 4 P2 LLM arms).\n",
             "Exact functions of the frozen file (abs tol {}).\n".format(ABS_TOL),
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
        return all(_close(a[k], b[k], tol) for k in ("point", "ci_lo", "ci_hi")) and a["n"] == b["n"]
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
            good = _close(fresh[cid]["value"], rec["value"], rec["tol"]["value"])
            ok = ok and good
            print("  [{}] {}".format("pass" if good else "FAIL", cid))
            if not good:
                print("      stored={} got={}".format(rec["value"], fresh[cid]["value"]))
        print("\n{}".format("ALL CLAIMS VERIFY" if ok else "SOME CLAIMS DRIFTED"))
        return 0 if ok else 1
    with open(EVIDENCE, "w") as f:
        json.dump(fresh, f, indent=2, sort_keys=True)
    write_provenance(fresh)
    print("wrote {} claims -> evidence.json + PROVENANCE.md".format(len(fresh)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
