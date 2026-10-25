"""Unified runner for the Text-Nav x Jev experiments (A and B).

One code path produces every experimental cell: a (condition, seed) pair becomes
one episode. The condition (see :mod:`conditions`) decides which parts of the
policy state are present -- persistent context, an authored explicit goal, and
episode history -- so the goal/history/narrative/label factors are carried
entirely by state content, never by the instruction wording or the environment's
own objective string.

Instrumentation (milestones, world snapshots, opportunity flags) is read from
ground-truth facts and logged off-prompt (EXPERIMENT_DESIGN.md Appendix A.2/A.3).
Nothing here leaks facts into the policy state.

Offline logic check (no key, no network, no TextWorld)::

    python -m llm_rl_scripts.text_nav.jev.run_experiment --self-test

Live pilot (Experiment A, 10 matched seeds x 4 conditions), PO, needs
TYPESAFE_API_KEY::

    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.text_nav.jev.run_experiment \
        --experiment A --seeds 0-9 --max-steps 20 --out results
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from typing import Callable, Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.text_nav.jev.conditions import (
    Condition,
    UNIFIED_INSTRUCTIONS,
    get_conditions,
)
from llm_rl_scripts.text_nav.jev.eval_jev import _is_done, _strip_task_preamble
from llm_rl_scripts.text_nav.jev.milestones import (
    MilestoneTracker,
    snapshot_from_facts,
)
from llm_rl_scripts.text_nav.jev.policy import JevTextNavPolicy
from llm_rl_scripts.text_nav.jev.recorder import Recorder, default_run_dir


# --- opportunity / diagnostic parsing from admissible commands --------------

def _take_target_cmd(admissible: List[str], target: str) -> Optional[str]:
    for c in admissible:
        if c.startswith("take ") and target in c:
            return c
    return None


def _take_distractor_cmds(admissible: List[str], target: str) -> List[str]:
    out = []
    for c in admissible:
        if c.startswith("take ") and target not in c:
            out.append(c)
    return out


def _classify_command(command: str) -> str:
    c = command.strip()
    if c.startswith("take "):
        return "take"
    if c.startswith("eat "):
        return "eat"
    if c.startswith("drop "):
        return "drop"
    if c.startswith("insert ") and "fridge" in c:
        return "insert"
    if c == "open fridge":
        return "open_fridge"
    if c == "close fridge":
        return "close_fridge"
    if c.startswith("go "):
        return "move"
    return "other"


def _episode_diagnostics_init() -> Dict[str, int]:
    return {
        "steps": 0,
        "take_target": 0,
        "take_distractor": 0,
        "eat": 0,
        "open_fridge": 0,
        "close_fridge": 0,
        "insert": 0,
        "drop": 0,
        "move": 0,
        # opportunity-conditioned target preference (design 4.3):
        "opp_target_offered": 0,          # steps where taking target was legal
        "opp_target_vs_distractor": 0,    # steps where BOTH were legal
        "chose_target_when_both": 0,      # ... and target was chosen
        "opp_eat_offered": 0,
        "chose_eat_when_offered": 0,
    }


# --- one episode ------------------------------------------------------------

def run_episode(
    env,
    policy: JevTextNavPolicy,
    cond: Condition,
    max_steps: int,
    recorder: Optional[Recorder] = None,
    episode_id: str = "",
    seed: Optional[int] = None,
    rollout: int = 0,
    verbose: bool = False,
) -> Dict[str, object]:
    env._reset()
    tracker = MilestoneTracker()
    diag = _episode_diagnostics_init()

    # Record the initial (reset) world state so milestone ordering is aware of
    # the starting configuration.
    tracker.update(_snapshot(env, cond.food_name))

    transcript = ""
    api_error: Optional[str] = None
    for t in range(max_steps):
        if _is_done(env.state) or not list(env.state["admissible_commands"]):
            break
        obs = _strip_task_preamble(env.state.feedback)  # strip in ALL conditions
        admissible = list(env.state["admissible_commands"])

        # Opportunity accounting BEFORE the choice.
        tgt_cmd = _take_target_cmd(admissible, cond.food_name)
        dist_cmds = _take_distractor_cmds(admissible, cond.food_name)
        eat_cmds = [c for c in admissible if c.startswith("eat ")]
        if tgt_cmd:
            diag["opp_target_offered"] += 1
        both = bool(tgt_cmd and dist_cmds)
        if both:
            diag["opp_target_vs_distractor"] += 1
        if eat_cmds:
            diag["opp_eat_offered"] += 1

        # History is carried only in H1 conditions; otherwise the policy sees no
        # transcript (reactive). Goal is the authored G1 string or None (G0).
        hist = transcript if cond.history else ""
        try:
            command = policy.choose(
                hist, obs, admissible, objective=cond.goal_text)
        except Exception as e:  # count API/parse errors apart from task failure
            api_error = "{}: {}".format(type(e).__name__, e)
            break

        ans = policy.last_answer
        kind = _classify_command(command)
        if kind == "take":
            if cond.food_name in command:
                diag["take_target"] += 1
            else:
                diag["take_distractor"] += 1
        elif kind in diag:
            diag[kind] += 1
        if both and command == tgt_cmd:
            diag["chose_target_when_both"] += 1
        if eat_cmds and command in eat_cmds:
            diag["chose_eat_when_offered"] += 1

        if recorder is not None:
            recorder.step({
                "episode_id": episode_id,
                "condition": cond.name,
                "seed": seed,
                "t": t,
                "observation": obs.strip(),
                "admissible": admissible,
                "goal_present": cond.goal_text is not None,
                "history_present": cond.history,
                "chosen": command,
                "command_kind": kind,
                "probabilities": (ans.probabilities if ans else None),
                "confidence": (ans.confidence if ans else None),
                "target_offered": bool(tgt_cmd),
                "target_vs_distractor": both,
                "world": _snapshot(env, cond.food_name).to_json(),
            })

        transcript += "\n{}\n> {}\n".format(obs.strip(), command)
        env._step(command)
        diag["steps"] += 1
        tracker.update(_snapshot(env, cond.food_name))
        if verbose:
            print("    [{}] {!r} -> M{}".format(t, command, tracker.greatest_ordered))

    won = bool(env.state["won"])
    result = {
        "episode_id": episode_id,
        "condition": cond.name,
        "experiment": cond.experiment,
        "factors": cond.factors,
        "seed": seed,
        "rollout": rollout,
        "food_name": cond.food_name,
        "success": float(won),
        "won": won,
        "lost": bool(env.state["lost"]),
        "reward": float(env.state["score"]),
        "steps": diag["steps"],
        "milestones": tracker.to_json(),
        "diagnostics": diag,
        "api_error": api_error,
        "completed": api_error is None,
    }
    if recorder is not None:
        recorder.episode(result)
    return result


def _snapshot(env, food_name: str):
    """Build a WorldSnapshot from env facts when available, else an empty one."""
    try:
        facts = env.state["facts"]
    except (KeyError, TypeError):
        facts = None
    won = bool(env.state["won"])
    if facts is None:
        from llm_rl_scripts.text_nav.jev.milestones import WorldSnapshot
        return WorldSnapshot(won=won)
    return snapshot_from_facts(facts, target_name=food_name, won=won)


# --- env / client builders --------------------------------------------------

def build_env(use_fridge_fake: bool, fully_observed: bool, food_name: str,
              seed: Optional[int]):
    if use_fridge_fake:
        from llm_rl_scripts.text_nav.jev.fridge_fake_env import FridgeFakeEnv
        return FridgeFakeEnv(target_name=food_name, seed=seed)
    from llm_rl_scripts.text_nav.env.env import TextNavEnv
    return TextNavEnv(
        display_location=fully_observed,
        display_inventory=fully_observed,
        food_name=food_name,
        seed=seed,
        request_facts=True,  # off-prompt instrumentation only
    )


def build_policy(cond: Condition, client: JevClient) -> JevTextNavPolicy:
    # history_chars only bounds the transcript in H1; H0 is enforced by the
    # runner passing an empty transcript, so we never hit the history_chars==0
    # slicing quirk.
    return JevTextNavPolicy(
        client,
        instructions=UNIFIED_INSTRUCTIONS,
        context=cond.context,
        history_chars=4000,
    )


def parse_seeds(spec: str) -> List[int]:
    """Accept '0-9', '0,1,2', or a single count 'n' meaning 0..n-1."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    return list(range(int(spec)))


# --- offline self-test ------------------------------------------------------

def _oracle_chooser(target: str) -> Callable:
    """A scripted stand-in for Jev used in the self-test: prefer the winning
    sequence so the full milestone path (M1..M5, won) is exercised, while still
    only ever returning a legal option."""
    priority = [
        "insert {} into fridge".format(target),
        "close fridge",
        "open fridge",
        "take {} from table".format(target),
        "go west",
    ]

    def choose(state, qid, options):
        for p in priority:
            if p in options:
                return p
        # otherwise first non-eat, non-drop option; fall back to anything
        for o in options:
            if not o.startswith(("eat ", "drop ")):
                return o
        return options[0]

    return choose


def self_test() -> int:
    print("[self-test] condition matrix")
    a = get_conditions(experiment="A")
    b = get_conditions(experiment="B")
    assert [c.name for c in a] == ["A1", "A2", "A3", "A4"], a
    assert [c.name for c in b] == ["B1", "B2", "B3", "B4"], b
    # G factor is exactly the presence of goal_text; H factor is history flag.
    assert [bool(c.goal_text) for c in a] == [False, True, False, True]
    assert [c.history for c in a] == [False, False, True, True]
    print("    ok: A={} B={}".format([c.name for c in a], [c.name for c in b]))

    print("[self-test] milestone tracker (synthetic snapshots)")
    _milestone_unit_test()
    print("    ok: ordered progression + out-of-order guard")

    print("[self-test] end-to-end on fridge fake (oracle 'Jev', with logging)")
    import tempfile
    from llm_rl_scripts.text_nav.jev.fridge_fake_env import FridgeFakeEnv  # noqa: F401
    target = "milk"
    client = JevClient(backend=MockJevBackend(_oracle_chooser(target)), model="mock")
    run_dir = tempfile.mkdtemp(prefix="jev_selftest_")
    with Recorder(run_dir, {"mode": "self-test", "model": "mock"}) as rec:
        results = []
        for cond in get_conditions(experiment="A"):
            cond = Condition(cond.name, cond.context, cond.goal_text, cond.history,
                             target, cond.experiment, cond.factors)
            env = build_env(True, False, target, seed=0)
            res = run_episode(env, build_policy(cond, client), cond, max_steps=20,
                              recorder=rec, episode_id="st-" + cond.name, seed=0)
            results.append(res)
            assert res["success"] == 1.0, (cond.name, res)
            assert res["milestones"]["greatest_ordered"] == 5, res["milestones"]
            assert res["milestones"]["ever"]["M5_complete"], res
            assert res["completed"]
    print("    ok: all 4 A-conditions reach M5 + won; logs -> {}".format(run_dir))

    print("[self-test] eat path fails cleanly")
    eat_client = JevClient(
        backend=MockJevBackend(lambda s, q, opts: next(
            (o for o in opts if o.startswith("eat ")),
            next((o for o in opts if o.startswith("take ")), opts[0]))),
        model="mock")
    cond = get_conditions(names=["A1"])[0]
    cond = Condition(cond.name, cond.context, cond.goal_text, cond.history,
                     target, cond.experiment, cond.factors)
    env = build_env(True, False, target, seed=0)
    res = run_episode(env, build_policy(cond, eat_client), cond, max_steps=20)
    assert res["won"] is False and res["lost"] is True, res
    assert res["milestones"]["ever"]["M1_acquire"], res
    assert not res["milestones"]["ever"]["M5_complete"], res
    print("    ok: eating target -> lost, M1 only")

    print("\nALL SELF-TESTS PASSED")
    return 0


def _milestone_unit_test() -> None:
    from llm_rl_scripts.text_nav.jev.milestones import WorldSnapshot, MilestoneTracker

    # In-order run: empty -> carry -> carry+kitchen -> carry+open -> in fridge
    # -> in fridge+closed+won.
    tr = MilestoneTracker()
    tr.update(WorldSnapshot(carrying_target=False, player_with_fridge=False,
                            fridge_open=False, target_in_fridge=False))
    assert tr.greatest_ordered == 0
    tr.update(WorldSnapshot(carrying_target=True, player_with_fridge=False,
                            fridge_open=False, target_in_fridge=False))
    assert tr.greatest_ordered == 1, tr.greatest_ordered
    tr.update(WorldSnapshot(carrying_target=True, player_with_fridge=True,
                            fridge_open=False, target_in_fridge=False))
    assert tr.greatest_ordered == 2, tr.greatest_ordered
    tr.update(WorldSnapshot(carrying_target=True, player_with_fridge=True,
                            fridge_open=True, target_in_fridge=False))
    assert tr.greatest_ordered == 3, tr.greatest_ordered
    tr.update(WorldSnapshot(carrying_target=False, player_with_fridge=True,
                            fridge_open=True, target_in_fridge=True))
    assert tr.greatest_ordered == 4, tr.greatest_ordered
    tr.update(WorldSnapshot(carrying_target=False, player_with_fridge=True,
                            fridge_open=False, target_in_fridge=True, won=True))
    assert tr.greatest_ordered == 5, tr.greatest_ordered

    # Out-of-order guard: target appears in fridge (M4 condition) without the
    # agent ever carrying it -> ordered progress must stay 0 (M1 never met).
    tr2 = MilestoneTracker()
    tr2.update(WorldSnapshot(carrying_target=False, target_in_fridge=True))
    assert tr2.ever["M4_store"] is True
    assert tr2.greatest_ordered == 0, tr2.greatest_ordered


# --- main -------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["A", "B"], default=None)
    p.add_argument("--conditions", default=None,
                   help="comma-separated condition names (e.g. A1,A4); "
                        "overrides --experiment")
    p.add_argument("--seeds", default="0-9",
                   help="'0-9', '0,1,2', or a count 'n' => 0..n-1")
    p.add_argument("--repeats", type=int, default=1,
                   help="independent Jev rollouts per (seed, condition). The game "
                        "is identical across rollouts (same seed); only Jev's "
                        "stochastic choices differ, so >1 estimates a success "
                        "rate rather than a single-pass coin flip.")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--fully-observed", action="store_true")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--backend", choices=["http", "mock"], default="http")
    p.add_argument("--env", choices=["real", "fridge-fake"], default="real")
    p.add_argument("--out", default=None, help="results root dir (timestamped "
                                               "subdir is created)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    names = [s.strip() for s in args.conditions.split(",")] if args.conditions else None
    conditions = get_conditions(names=names, experiment=args.experiment) \
        if (names or args.experiment) else get_conditions(experiment="A")
    seeds = parse_seeds(args.seeds)
    use_mock = args.backend == "mock"
    use_fake = args.env == "fridge-fake"
    verbose = not args.quiet

    rng = random.Random(0)
    mock_chooser = (lambda s, q, opts: rng.choice(opts)) if use_mock else None
    backend = MockJevBackend(mock_chooser) if use_mock else HttpJevBackend()
    client = JevClient(backend=backend, model=("mock" if use_mock else args.model))

    tag = (args.experiment or "mix")
    recorder = None
    if args.out:
        recorder = Recorder(default_run_dir(args.out, tag), {
            "experiment": args.experiment,
            "conditions": [c.name for c in conditions],
            "seeds": seeds,
            "repeats": args.repeats,
            "max_steps": args.max_steps,
            "fully_observed": args.fully_observed,
            "model": ("mock" if use_mock else args.model),
            "env": args.env,
        })

    print("Text-Nav x Jev experiment  [{} conditions x {} seeds]  model={}".format(
        len(conditions), len(seeds), "mock" if use_mock else args.model))

    try:
        agg: Dict[str, List[Dict]] = {c.name: [] for c in conditions}
        for cond in conditions:
            for seed in seeds:
                for r in range(args.repeats):
                    env = build_env(use_fake, args.fully_observed, cond.food_name,
                                    seed)
                    eid = "{}-s{}-r{}".format(cond.name, seed, r)
                    if verbose:
                        print("  {}  [{}]".format(eid, cond.describe()))
                    res = run_episode(env, build_policy(cond, client), cond,
                                      max_steps=args.max_steps, recorder=recorder,
                                      episode_id=eid, seed=seed, rollout=r,
                                      verbose=verbose)
                    agg[cond.name].append(res)
    finally:
        if recorder is not None:
            recorder.close()

    print("\n=== per-condition summary ===")
    for name, rows in agg.items():
        n = len(rows)
        succ = sum(r["success"] for r in rows) / n if n else 0.0
        ordm = sum(r["milestones"]["greatest_ordered"] for r in rows) / n if n else 0.0
        errs = sum(0 if r["completed"] else 1 for r in rows)
        print("  {}: success={:.3f}  mean_ordered_milestone={:.2f}  n={}  errors={}"
              .format(name, succ, ordm, n, errs))
    if recorder is not None:
        print("\nlogs written under: {}".format(recorder.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
