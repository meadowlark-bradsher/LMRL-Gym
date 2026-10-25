"""Validate the real Text-Nav env against the instrumentation -- no API, no Jev.

This is the design's pre-flight check (EXPERIMENT_DESIGN.md sections 9-10): before
spending any Jev credits, confirm on the *real* TextWorld env that

  1. seeding is deterministic (same seed -> same game) and distinguishes seeds;
  2. the facts adapter reduces real world facts without error;
  3. stepping each game's own winning walkthrough drives the MilestoneTracker
     through M1..M5 in order and flips ``won=True`` -- i.e. win-detection and
     milestone scoring agree with the environment.

It builds and steps games only; it never calls Jev. Run it in the LLM_RL conda
env::

    PYTHONPATH="$PWD" conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.text_nav.jev.validate_runner --seeds 0-4
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from llm_rl_scripts.text_nav.jev.milestones import MilestoneTracker, snapshot_from_facts
from llm_rl_scripts.text_nav.jev.run_experiment import parse_seeds


def _opening_feedback(seed: int, food: str) -> str:
    from llm_rl_scripts.text_nav.env.env import TextNavEnv
    return TextNavEnv(food_name=food, seed=seed, request_facts=True).state.feedback


def validate_seed(seed: int, food: str) -> dict:
    from llm_rl_scripts.text_nav.env.env import TextNavEnv
    from llm_rl_scripts.text_nav.env.make_game import build_and_compile_game

    game, _ = build_and_compile_game(hide_location=True, food_name=food, seed=seed)
    walkthrough = list(game.metadata["walkthrough"])

    env = TextNavEnv(food_name=food, seed=seed, request_facts=True)
    tr = MilestoneTracker()
    tr.update(snapshot_from_facts(env.state["facts"], food, won=bool(env.state["won"])))
    for cmd in walkthrough:
        env._step(cmd)
        tr.update(snapshot_from_facts(env.state["facts"], food,
                                      won=bool(env.state["won"])))
    return {
        "seed": seed,
        "walkthrough_len": len(walkthrough),
        "won": bool(env.state["won"]),
        "score": float(env.state["score"]),
        "greatest_ordered": tr.greatest_ordered,
        "ever_M5": tr.ever["M5_complete"],
    }


def main(argv: List[str] = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0-4", help="'0-4', '0,1,2', or a count")
    p.add_argument("--food-name", default="milk")
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    food = args.food_name

    print("Validating real Text-Nav env over seeds {} (food={!r})".format(seeds, food))

    # 1. determinism.
    ok_det = True
    for s in seeds[:3]:  # a few are enough to catch nondeterminism
        if _opening_feedback(s, food) != _opening_feedback(s, food):
            ok_det = False
            print("  seed {}: NOT reproducible".format(s))
    if len(seeds) >= 2:
        if _opening_feedback(seeds[0], food) == _opening_feedback(seeds[1], food):
            print("  note: seeds {} and {} gave identical openings (possible but "
                  "worth a look)".format(seeds[0], seeds[1]))
    print("  determinism: {}".format("ok" if ok_det else "FAILED"))

    # 2+3. facts adapter + walkthrough milestones.
    all_ok = ok_det
    for s in seeds:
        r = validate_seed(s, food)
        good = r["won"] and r["greatest_ordered"] == 5 and r["ever_M5"]
        all_ok = all_ok and good
        print("  seed {:>2}: won={} score={:.0f} ordered=M{} wt_len={}  {}".format(
            r["seed"], r["won"], r["score"], r["greatest_ordered"],
            r["walkthrough_len"], "ok" if good else "FAILED"))

    print("\nVALIDATION", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
