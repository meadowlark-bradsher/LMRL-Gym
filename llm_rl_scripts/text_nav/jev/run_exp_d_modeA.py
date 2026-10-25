"""Experiment D, Mode A — run the belief-space explorer as a baseline policy.

Drives `BeliefSpaceExplorer` (no Jev calls) on the same matched-seed Text-Nav
harness Jev used, so its success rate sits directly beside Jev's. The question:
does principled exploration + belief-tracking + planning — the things Jev
structurally lacks — close the gap on this task?

The agent sees only observation-derived information (room identity + admissible
set) and the stated goal/target; ground-truth facts are read only to score
milestones, never by the agent.

    python -m llm_rl_scripts.text_nav.jev.run_exp_d_modeA --self-test   # offline
    PYTHONPATH="$PWD" conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.text_nav.jev.run_exp_d_modeA --seeds 0-49 --out results_dmodeA
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from llm_rl_scripts.text_nav.jev.explorer_policy import BeliefSpaceExplorer
from llm_rl_scripts.text_nav.jev.milestones import MilestoneTracker, snapshot_from_facts
from llm_rl_scripts.text_nav.jev.recorder import Recorder, default_run_dir
from llm_rl_scripts.text_nav.jev.run_experiment import _is_done, build_env, parse_seeds


def _room_of(env, target: str) -> Optional[str]:
    try:
        facts = env.state["facts"]
    except (KeyError, TypeError):
        return None
    snap = snapshot_from_facts(facts, target_name=target, won=bool(env.state["won"]))
    return snap.extra.get("player_room")


def run_episode(env, agent: BeliefSpaceExplorer, max_steps: int, target: str,
                seed: Optional[int] = None, verbose: bool = False) -> Dict:
    env._reset()
    agent.reset()
    tracker = MilestoneTracker()
    tracker.update(snapshot_from_facts(env.state["facts"], target,
                                       won=bool(env.state["won"])))
    steps = 0
    for _ in range(max_steps):
        if _is_done(env.state) or not list(env.state["admissible_commands"]):
            break
        room = _room_of(env, target) or env.state.feedback  # obs-signature fallback
        admissible = list(env.state["admissible_commands"])
        cmd = agent.choose(room, admissible)
        if verbose:
            print("   room={!r} -> {!r}".format(room, cmd))
        env._step(cmd)
        steps += 1
        tracker.update(snapshot_from_facts(env.state["facts"], target,
                                           won=bool(env.state["won"])))
    return {
        "seed": seed, "won": bool(env.state["won"]), "success": float(env.state["won"]),
        "lost": bool(env.state["lost"]), "steps": steps,
        "reward": float(env.state["score"]),
        "milestones": tracker.to_json(),
    }


def self_test() -> int:
    from llm_rl_scripts.text_nav.jev.fridge_fake_env import FridgeFakeEnv
    print("[Mode A self-test] explorer on fridge-fake")
    env = FridgeFakeEnv(target_name="milk")
    res = run_episode(env, BeliefSpaceExplorer("milk"), max_steps=20, target="milk",
                      verbose=True)
    assert res["won"], res
    assert res["milestones"]["greatest_ordered"] == 5, res["milestones"]
    print("    ok: won in {} steps, reached M5".format(res["steps"]))
    print("ALL MODE-A SELF-TESTS PASSED")
    return 0


def main(argv: List[str] = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seeds", default="0-49")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--food-name", default="milk")
    p.add_argument("--fully-observed", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()

    seeds = parse_seeds(args.seeds)
    target = args.food_name
    rec = None
    if args.out:
        rec = Recorder(default_run_dir(args.out, "DmodeA"),
                       {"experiment": "D_modeA", "policy": "belief_space_explorer",
                        "seeds": seeds, "max_steps": args.max_steps,
                        "food_name": target, "fully_observed": args.fully_observed})
    print("Exp D Mode A (belief-space explorer)  seeds={}  max_steps={}".format(
        len(seeds), args.max_steps))
    results = []
    try:
        for s in seeds:
            env = build_env(False, args.fully_observed, target, s)
            res = run_episode(env, BeliefSpaceExplorer(target), args.max_steps, target,
                              seed=s, verbose=not args.quiet)
            if rec:
                rec.episode(res)
            results.append(res)
            if not args.quiet:
                print("  seed {:>2}: won={} steps={} orderedM={}".format(
                    s, res["won"], res["steps"], res["milestones"]["greatest_ordered"]))
    finally:
        if rec:
            rec.close()

    n = len(results)
    succ = sum(r["success"] for r in results) / n if n else 0.0
    ordm = sum(r["milestones"]["greatest_ordered"] for r in results) / n if n else 0.0
    wins = [r for r in results if r["won"]]
    avg_win_steps = sum(r["steps"] for r in wins) / len(wins) if wins else float("nan")
    print("\n=== Mode A summary ({} seeds) ===".format(n))
    print("  success rate         : {:.3f} ({}/{})".format(succ, len(wins), n))
    print("  mean ordered milestone: {:.2f}".format(ordm))
    print("  mean steps (wins)    : {:.2f}".format(avg_win_steps))
    if rec:
        print("  logs:", rec.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
