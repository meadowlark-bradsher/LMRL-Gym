"""Experiment C (live layer) — fixed-state pairwise / full-menu preference probing.

The free layer (`analyze_exp_c.py`) showed Jev's *full-menu* distribution is stable
at a frozen state. This probe collects the data the free layer cannot: repeated
**pairwise-in-isolation** choices (both presentation orders) and a matched
**full-menu** choice at the SAME frozen snapshot, so we can test transitivity,
order (presentation) effects, menu effects, and fit a scalar (Bradley-Terry) rank.

It freezes the decision context: snapshots are reconstructed from the logged
reactive states (the exact context + observation + admissible set Jev saw, no
history), and the environment is never advanced. Each pairwise comparison and the
full-menu query are independent Jev calls, so this is a **bounded live spend**:
  cost ≈ snapshots × [ full_menu_reps + 3·(pairs)·2·pair_reps ]  Jev calls.

Offline check (mock Jev, no key/network):
    python -m llm_rl_scripts.text_nav.jev.probe_exp_c --self-test

Live (needs TYPESAFE_API_KEY), e.g. 6 snapshots from A1, 12 reps/pair:
    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.text_nav.jev.probe_exp_c \
      --steps llm_rl_scripts/text_nav/jev/results/main_A_20261025/steps.jsonl.gz \
      --condition A1 --snapshots 6 --full-menu-reps 12 --pair-reps 12 --out results_expc
"""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import json
import os
import random
import sys
from typing import Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.text_nav.jev.conditions import ALL_CONDITIONS, UNIFIED_INSTRUCTIONS
from llm_rl_scripts.text_nav.jev.policy import JevTextNavPolicy
from llm_rl_scripts.text_nav.jev.recorder import Recorder, default_run_dir


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def select_snapshots(steps_path: str, condition: str, n: int, min_options: int = 3,
                     max_options: Optional[int] = None) -> List[dict]:
    """Pick the most-resampled reactive fixed states for a condition.

    A snapshot carries everything needed to rebuild the exact frozen policy input:
    the condition (=> context + goal), the observation, and the admissible set.
    ``max_options`` caps the menu size (use 3 for clean A/B/C transitivity triads
    and bounded cost).
    """
    rows = [json.loads(l) for l in _open(steps_path) if l.strip()]
    rows = [r for r in rows if r["condition"] == condition and not r["history_present"]]
    counts = collections.Counter(
        (r["observation"], tuple(r["admissible"])) for r in rows
        if len(r["admissible"]) >= min_options
        and (max_options is None or len(r["admissible"]) <= max_options))
    snaps = []
    for i, ((obs, adm), ct) in enumerate(counts.most_common(n)):
        snaps.append({"snapshot_id": "{}-{:02d}".format(condition, i),
                      "condition": condition, "observation": obs,
                      "admissible": list(adm), "log_samples": ct})
    return snaps


def build_state_for(snap: dict) -> str:
    cond = ALL_CONDITIONS[snap["condition"]]
    policy = JevTextNavPolicy(client=None, instructions=UNIFIED_INSTRUCTIONS,
                              context=cond.context)
    return policy.build_state(transcript="", observation=snap["observation"],
                              objective=cond.goal_text)


def _choice(client: JevClient, state: str, option_names: List[str]) -> dict:
    ans = client.choice(state=state, instructions=UNIFIED_INSTRUCTIONS,
                        options={o: None for o in option_names}, question_id="command")
    return {"choice": ans.choice, "probabilities": ans.probabilities,
            "confidence": ans.confidence}


def probe_snapshot(client: JevClient, snap: dict, full_menu_reps: int,
                   pair_reps: int, rng: random.Random, rec: Optional[Recorder]
                   ) -> dict:
    state = build_state_for(snap)
    opts = list(snap["admissible"])
    if rec:
        rec.snapshot({**snap, "state": state})

    full = [_choice(client, state, opts) for _ in range(full_menu_reps)]

    pair_results = []
    for a, b in itertools.combinations(opts, 2):
        for _ in range(pair_reps):
            order = [a, b] if rng.random() < 0.5 else [b, a]  # balance presentation
            res = _choice(client, state, order)
            row = {"snapshot_id": snap["snapshot_id"], "a": a, "b": b,
                   "first": order[0], "chosen": res["choice"],
                   "probabilities": res["probabilities"]}
            pair_results.append(row)
            if rec:
                rec.pairwise(row)
    return {"snapshot": snap, "state": state, "full_menu": full,
            "pairwise": pair_results}


def analyze_pairwise(pairwise: List[dict]) -> dict:
    """Pairwise win probabilities, order (presentation) effect, and 3-cycles."""
    by_pair = collections.defaultdict(list)
    for r in pairwise:
        by_pair[frozenset((r["a"], r["b"]))].append(r)
    wins = {}
    order_effect = {}
    for pair, rs in by_pair.items():
        a, b = sorted(pair)
        p_a = sum(1 for r in rs if r["chosen"] == a) / len(rs)
        wins[(a, b)] = round(p_a, 4)
        a_first = [r for r in rs if r["first"] == a]
        b_first = [r for r in rs if r["first"] == b]
        if a_first and b_first:
            pa_af = sum(1 for r in a_first if r["chosen"] == a) / len(a_first)
            pa_bf = sum(1 for r in b_first if r["chosen"] == a) / len(b_first)
            order_effect[(a, b)] = round(pa_af - pa_bf, 4)  # >0 => favors first-shown
    # directed graph by majority; detect 3-cycles
    opts = sorted({x for pair in by_pair for x in pair})
    beats = {(a, b): (wins.get((a, b), 1 - wins.get((b, a), 0.5)) > 0.5)
             for a in opts for b in opts if a != b}
    cycles = []
    for a, b, c in itertools.combinations(opts, 3):
        for x, y, z in [(a, b, c), (a, c, b)]:
            if beats.get((x, y)) and beats.get((y, z)) and beats.get((z, x)):
                cycles.append((x, y, z))
    return {"pairwise_win_prob": {f"{a}|{b}": p for (a, b), p in wins.items()},
            "order_effect": {f"{a}|{b}": e for (a, b), e in order_effect.items()},
            "cycles_3": [list(c) for c in cycles]}


# --- self test ---------------------------------------------------------------

def self_test() -> int:
    print("[probe self-test] build_state + probe loop on mock Jev")
    snap = {"snapshot_id": "A1-00", "condition": "A1",
            "observation": "You are in the living room. On the table: milk, fresh food.",
            "admissible": ["take milk from table", "take fresh food from table",
                           "go west"], "log_samples": 99}
    state = build_state_for(snap)
    assert "Context:" in state and "milk" in state, state[:200]
    rng = random.Random(0)
    client = JevClient(backend=MockJevBackend(
        lambda s, q, opts: opts[0]), model="mock")  # deterministic: picks first-shown
    out = probe_snapshot(client, snap, full_menu_reps=3, pair_reps=6, rng=rng, rec=None)
    ana = analyze_pairwise(out["pairwise"])
    assert out["full_menu"] and out["pairwise"], "no probe output"
    # 'picks first-shown' => strong order effect, by construction
    assert any(abs(v) > 0.3 for v in ana["order_effect"].values()), ana["order_effect"]
    print("    ok: {} full-menu, {} pairwise; detected order effect {}".format(
        len(out["full_menu"]), len(out["pairwise"]), ana["order_effect"]))
    print("ALL PROBE SELF-TESTS PASSED")
    return 0


def main(argv: List[str] = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--steps", default=None)
    p.add_argument("--condition", default="A1")
    p.add_argument("--snapshots", type=int, default=6)
    p.add_argument("--max-options", type=int, default=None,
                   help="cap menu size (use 3 for clean transitivity triads)")
    p.add_argument("--full-menu-reps", type=int, default=12)
    p.add_argument("--pair-reps", type=int, default=12)
    p.add_argument("--backend", choices=["http", "mock"], default="http")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.steps:
        print("need --steps (or --self-test)", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    client = JevClient(
        backend=(MockJevBackend(lambda s, q, o: rng.choice(o)) if args.backend == "mock"
                 else HttpJevBackend()),
        model=("mock" if args.backend == "mock" else "jev-latest"))

    snaps = select_snapshots(args.steps, args.condition, args.snapshots,
                             max_options=args.max_options)
    rec = None
    if args.out:
        rec = Recorder(default_run_dir(args.out, "expc_" + args.condition),
                       {"experiment": "C", "condition": args.condition,
                        "snapshots": len(snaps), "full_menu_reps": args.full_menu_reps,
                        "pair_reps": args.pair_reps,
                        "model": ("mock" if args.backend == "mock" else "jev-latest")})
    calls = len(snaps) * (args.full_menu_reps +
                          sum(1 for _ in itertools.combinations(range(3), 2)) * 2
                          * args.pair_reps)  # rough, assumes ~3 options
    print("Exp C probe: {} snapshots from {}  (~{} Jev calls est.)".format(
        len(snaps), args.condition, "~" + str(calls)))
    try:
        results = []
        for snap in snaps:
            out = probe_snapshot(client, snap, args.full_menu_reps, args.pair_reps,
                                 rng, rec)
            ana = analyze_pairwise(out["pairwise"])
            results.append({"snapshot_id": snap["snapshot_id"], **ana})
            print("  {}: win_probs={} order_effect={} cycles={}".format(
                snap["snapshot_id"], ana["pairwise_win_prob"], ana["order_effect"],
                ana["cycles_3"]))
    finally:
        if rec:
            rec.close()
    if rec:
        with open(os.path.join(rec.run_dir, "exp_c_pairwise_analysis.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nlogs + analysis under:", rec.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
