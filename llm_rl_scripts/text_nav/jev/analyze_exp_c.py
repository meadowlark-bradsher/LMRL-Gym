"""Experiment C (free layer) — fixed-state preference stability from logs.

The runner logs Jev's *full-menu* option-probability vector at every state. In the
reactive (H0) conditions the policy input is exactly (context, observation,
admissible set) — no history — so every time the agent re-encounters an identical
such state (across the K rollouts and across episodes) we get an independent draw
of Jev's full-menu Choice distribution at a *frozen* decision context. This script
mines those repeats and measures how stable the distribution and its induced
ranking are.

What this DOES (zero API): the full-menu distribution and its sampling stability —
does an identical fixed state yield a stable choice and a stable ranking?

What this does NOT do: pairwise-in-isolation preferences, transitivity / cycles,
order (presentation) effects, Bradley-Terry over pairwise wins, harmonic residual.
Those require fresh two-option Choice probes on frozen snapshots — see
`probe_exp_c.py` (a bounded live spend). A single full-menu Luce vector is trivially
self-rankable; the meaningful free question is whether that ranking is *stable*
across identical-state resamples, which is what we report.

    python -m llm_rl_scripts.text_nav.jev.analyze_exp_c \
        llm_rl_scripts/text_nav/jev/results/main_A_20261025/steps.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import os
import statistics
import sys
from typing import Dict, List

REACTIVE_ONLY = True   # H0 conditions give a true fixed policy input
MIN_SAMPLES = 8
MIN_OPTIONS = 3


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_steps(path: str) -> List[dict]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fixed_state_key(r: dict):
    # reactive policy input: condition + observation + admissible set (no history)
    return (r["condition"], r["observation"], tuple(r["admissible"]))


def _ranking(prob: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(prob.items(), key=lambda kv: (-kv[1], kv[0]))]


def analyze(path: str) -> dict:
    rows = load_steps(path)
    if REACTIVE_ONLY:
        rows = [r for r in rows if not r["history_present"]]
    groups: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in rows:
        if r.get("probabilities") and len(r["admissible"]) >= MIN_OPTIONS:
            groups[fixed_state_key(r)].append(r)

    per_state = []
    for key, rs in groups.items():
        if len(rs) < MIN_SAMPLES:
            continue
        cond, obs, admissible = key
        opts = list(admissible)
        # argmax stability
        argmaxes = [r["chosen"] for r in rs]
        mode_argmax, mode_ct = collections.Counter(argmaxes).most_common(1)[0]
        argmax_stability = mode_ct / len(rs)
        # ranking stability: fraction whose top-2 (as a set) equals the modal top-2
        rankings = [_ranking(r["probabilities"]) for r in rs if r["probabilities"]]
        top2_sets = [frozenset(rk[:2]) for rk in rankings if len(rk) >= 2]
        if top2_sets:
            mode_top2, t2_ct = collections.Counter(top2_sets).most_common(1)[0]
            top2_stability = t2_ct / len(top2_sets)
        else:
            top2_stability = float("nan")
        # per-option probability spread (mean over options of SD across samples)
        sds = []
        for o in opts:
            vals = [r["probabilities"].get(o, 0.0) for r in rs if r["probabilities"]]
            if len(vals) > 1:
                sds.append(statistics.pstdev(vals))
        prob_sd = sum(sds) / len(sds) if sds else 0.0
        confs = [r["confidence"] for r in rs if r["confidence"] is not None]
        per_state.append({
            "condition": cond, "n_options": len(opts), "n_samples": len(rs),
            "argmax_stability": round(argmax_stability, 4),
            "top2_stability": round(top2_stability, 4),
            "mean_prob_sd": round(prob_sd, 4),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "obs_head": obs.strip().replace("\n", " ")[:70],
        })

    def agg(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return (round(sum(xs) / len(xs), 4), round(min(xs), 4), round(max(xs), 4)) if xs else (None, None, None)

    n = len(per_state)
    unanimous = sum(1 for s in per_state if s["argmax_stability"] == 1.0)
    return {
        "source": os.path.abspath(path),
        "n_fixed_states": n,
        "min_samples": MIN_SAMPLES, "min_options": MIN_OPTIONS,
        "argmax_stability": agg([s["argmax_stability"] for s in per_state]),
        "unanimous_argmax_frac": round(unanimous / n, 4) if n else None,
        "top2_stability": agg([s["top2_stability"] for s in per_state]),
        "mean_prob_sd": agg([s["mean_prob_sd"] for s in per_state]),
        "mean_confidence": agg([s["mean_confidence"] for s in per_state]),
        "per_state": per_state,
    }


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("steps", help="steps.jsonl or steps.jsonl.gz")
    p.add_argument("--out", default=None, help="dir for exp_c_states.csv/json "
                                               "(default: alongside steps)")
    args = p.parse_args(argv)
    a = analyze(args.steps)
    out = args.out or os.path.dirname(os.path.abspath(args.steps))
    with open(os.path.join(out, "exp_c_states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(a["per_state"][0].keys()))
        w.writeheader()
        w.writerows(a["per_state"])
    with open(os.path.join(out, "exp_c_summary.json"), "w") as f:
        json.dump({k: v for k, v in a.items() if k != "per_state"}, f, indent=2,
                  default=str)

    print("Experiment C (free layer) — fixed-state full-menu stability")
    print("  source:", a["source"])
    print("  fixed states (reactive, >= {} samples, >= {} options): {}".format(
        a["min_samples"], a["min_options"], a["n_fixed_states"]))

    def line(name, tup):
        m, lo, hi = tup
        print("    {:<22}: mean={} (range {}..{})".format(name, m, lo, hi))
    line("argmax stability", a["argmax_stability"])
    print("    {:<22}: {}".format("unanimous argmax frac", a["unanimous_argmax_frac"]))
    line("top-2 set stability", a["top2_stability"])
    line("mean per-opt prob SD", a["mean_prob_sd"])
    line("mean confidence", a["mean_confidence"])
    # by condition
    byc = collections.defaultdict(list)
    for s in a["per_state"]:
        byc[s["condition"]].append(s)
    print("  by condition (n states | mean argmax-stability | mean prob SD):")
    for c in sorted(byc):
        ss = byc[c]
        ams = sum(s["argmax_stability"] for s in ss) / len(ss)
        psd = sum(s["mean_prob_sd"] for s in ss) / len(ss)
        print("    {}: {:>4} | {:.3f} | {:.3f}".format(c, len(ss), ams, psd))
    print("\n  wrote exp_c_states.csv, exp_c_summary.json")
    print("  NOTE: pairwise/transitivity/order/harmonic require live probes "
          "(probe_exp_c.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
