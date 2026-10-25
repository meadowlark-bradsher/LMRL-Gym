"""Evaluate Jev as a zero-shot policy on the Text-Nav task.

Jev picks each move from the environment's admissible commands via the Choice
primitive; there is no RL training. This mirrors the role the paper's GPT-4
baseline plays (a prompted model, no fine-tuning), so the natural comparison is
the GPT-4 column of Table 2 -- except that Jev can only ever return a legal
command, so it cannot lose reward to invalid actions the way a free-form
generator does.

Run the real benchmark environment in the LLM_RL conda env. Use
``--no-capture-output`` and ``python -u`` so progress streams -- plain
``conda run`` buffers all output until the process exits, which looks like a
hang. Start small to confirm the live path before scaling up:

    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.text_nav.jev.eval_jev --episodes 3 --max-steps 20
    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.text_nav.jev.eval_jev --episodes 50 --max-steps 20
    # add --fully-observed for the FO variant

That needs TYPESAFE_API_KEY set and makes one Jev API call per move.

Validate the loop offline with no TextWorld, no key and no network:

    python -m llm_rl_scripts.text_nav.jev.eval_jev --self-test
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from typing import Callable, Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.text_nav.jev.policy import (
    JevTextNavPolicy,
    NARRATIVE_CONTEXT,
    NARRATIVE_INSTRUCTIONS,
)


def _is_done(state) -> bool:
    """TextWorld's GameState has no ``done`` key; an episode ends on won/lost."""
    return bool(state["won"]) or bool(state["lost"])


# Removes TextWorld's auto-generated objective preamble (the "profound round ...
# Your task for today is to ... ." sentences that appear in the first
# observation). Narrative mode strips it so the leaked instruction does not
# contaminate the test of whether context alone evokes the behavior.
_PREAMBLE_RE = re.compile(
    r"You are now playing a profound round of TextWorld![^.]*\."
    r"(\s*Your task for today is[^.]*\.)?",
    re.IGNORECASE,
)


def _strip_task_preamble(text: str) -> str:
    return _PREAMBLE_RE.sub("", text).lstrip()


def run_episode(
    env,
    policy: JevTextNavPolicy,
    max_steps: int,
    verbose: bool = False,
    use_objective: bool = True,
) -> Dict[str, float]:
    """Drive one episode against the env's _reset/_step/state surface.

    We read the state directly (feedback, admissible_commands, score, done)
    rather than the TextNavEnv.step wrapper, which is both what the admissible
    commands live behind and a way around two bugs in that wrapper
    (``tuple(feedback,)`` and an undefined ``true``).
    """
    env._reset()
    # The objective is stable across the episode; read it once. The fake env
    # has no objective key, so fall back to None.
    if use_objective:
        try:
            objective = env.state["objective"]
        except (KeyError, TypeError):
            objective = None
    else:
        # Narrative mode: withhold the (misleading) surfaced objective so we
        # test whether context alone evokes the latent behavior.
        objective = None
    transcript = ""
    steps = 0
    for _ in range(max_steps):
        obs = env.state.feedback
        if not use_objective:
            obs = _strip_task_preamble(obs)
        if _is_done(env.state) or not list(env.state["admissible_commands"]):
            break
        admissible = list(env.state["admissible_commands"])
        if verbose:
            print("  obs: {}".format(obs.strip().replace("\n", " ")[:200]))
        command = policy.choose(transcript, obs, admissible, objective=objective)
        if verbose:
            conf = policy.last_answer.confidence if policy.last_answer else None
            print("  -> {!r} (confidence={})".format(command, conf))
        transcript += "\n{}\n> {}\n".format(obs.strip(), command)
        env._step(command)
        steps += 1

    reward = float(env.state["score"])
    won = bool(env.state["won"])
    return {
        "reward": reward,
        "done": float(_is_done(env.state)),
        "success": float(won),
        "steps": float(steps),
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    n = len(results)
    keys = ["reward", "success", "done", "steps"]
    return {k: sum(r[k] for r in results) / n for k in keys} if n else {}


def build_env(use_fake: bool, fully_observed: bool, food_name: str = "stale food"):
    if use_fake:
        from llm_rl_scripts.text_nav.jev.fake_env import FakeTextNavEnv

        return FakeTextNavEnv()
    # Imported lazily so the offline self-test needs no TextWorld.
    from llm_rl_scripts.text_nav.env.env import TextNavEnv

    return TextNavEnv(
        display_location=fully_observed,
        display_inventory=fully_observed,
        food_name=food_name,
    )


def build_client(use_mock: bool, model: str, mock_chooser: Optional[Callable]) -> JevClient:
    backend = MockJevBackend(mock_chooser) if use_mock else HttpJevBackend()
    return JevClient(backend=backend, model=model)


def main():
    # Stream output line by line even when stdout is a pipe (e.g. under
    # conda run), so a long live run shows progress instead of looking hung.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--fully-observed", action="store_true")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--backend", choices=["http", "mock"], default="http")
    p.add_argument("--env", choices=["real", "fake"], default="real")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="offline smoke test: fake env + mock Jev, no key or network",
    )
    p.add_argument(
        "--narrative",
        action="store_true",
        help="narrative mode: pin a common-sense persona, strip the surfaced "
        "objective, and (by default) rename the perishable to 'milk' -- tests "
        "whether context alone evokes the latent 'refrigerate it' behavior",
    )
    p.add_argument(
        "--food-name",
        default=None,
        help="name the perishable item (default: 'stale food'; 'milk' in "
        "narrative mode unless overridden)",
    )
    args = p.parse_args()

    use_mock = args.backend == "mock" or args.self_test
    use_fake = args.env == "fake" or args.self_test
    verbose = not args.quiet

    # The mock chooser stands in for Jev offline: pick a random admissible
    # command so some episodes reach the goal and the success accounting shows.
    rng = random.Random(args.seed)
    mock_chooser = (lambda state, qid, options: rng.choice(options)) if use_mock else None

    client = build_client(use_mock, args.model, mock_chooser)
    if args.narrative:
        policy = JevTextNavPolicy(
            client,
            instructions=NARRATIVE_INSTRUCTIONS,
            context=NARRATIVE_CONTEXT,
        )
    else:
        policy = JevTextNavPolicy(client)

    food_name = args.food_name or ("milk" if args.narrative else "stale food")

    mode = []
    mode.append("fake-env" if use_fake else "real-env")
    mode.append("mock-jev" if use_mock else "jev:{}".format(args.model))
    mode.append("FO" if args.fully_observed else "PO")
    if args.narrative:
        mode.append("narrative")
    if not use_fake:
        mode.append("food={}".format(food_name))
    print("Text-Nav x Jev  [{}]  episodes={} max_steps={}".format(
        " ".join(mode), args.episodes, args.max_steps))

    results = []
    for i in range(args.episodes):
        env = build_env(use_fake, args.fully_observed, food_name)
        if verbose:
            print("episode {}".format(i + 1))
        res = run_episode(
            env, policy, args.max_steps, verbose=verbose,
            use_objective=not args.narrative,
        )
        if verbose:
            print("  result: {}".format(res))
        results.append(res)

    agg = summarize(results)
    print("\n=== summary over {} episodes ===".format(len(results)))
    print("success rate : {:.3f}".format(agg.get("success", 0.0)))
    print("avg reward   : {:.3f}".format(agg.get("reward", 0.0)))
    print("done rate    : {:.3f}".format(agg.get("done", 0.0)))
    print("avg steps    : {:.2f}".format(agg.get("steps", 0.0)))


if __name__ == "__main__":
    main()
