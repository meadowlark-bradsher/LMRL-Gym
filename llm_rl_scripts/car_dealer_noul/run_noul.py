"""CD-NOUL-1 Stage 1 runner.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.run_noul --phase pilot \
        --bank results/noul_bank_20261108 --out results/noul_pilot_<date> \
        [--arms plan_ask-noul_answer-0.2,plan_noask] [--regimes plain,oblique] \
        [--jev-backend mock|http] [--sonnet-backend fake|anthropic]

Cells: (focal role, opener, focal type, opponent type) = 2 x 2 x 6 x 6 = 144, each repeated
``repeats`` times; every text arm runs in every regime, ``plan_noask`` once (it sees no text).
Episodes append to ``<out>/runs.jsonl`` as they finish and the run resumes by episode key.
Seeds: the opponent's stream is ``(seed, role, opener, i, j, k)`` exactly as in Phase 1, so
``plan_ask-oracle-0`` on Phase 1's seed replays CD-VOI-1's ``exact_ask``; the bank variants come
from ``arms_noul.variant_plan`` (arm-independent). The confirmatory phase refuses to run unless
HEAD carries the registration tag.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.protocol_live import ENV_VERSION, LiveConfig
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_persona.run import head_has_tag

from . import arms_noul as an
from .agents_noul import SonnetAgent, TextJevPolicy
from .verbalize import MODEL as SONNET_MODEL, REGIMES

DESIGN = "CD-NOUL-1"
# (repeats, seed); the confirmatory values are fixed by the registration.
PHASES = {"pilot": (1, 20261111), "confirmatory": (5, 20261112)}
DEFAULT_ARMS = an.planner_arms() + list(an.AGENT_ARMS)
ROLES = ("buyer", "seller")


def _tag(r):
    return "/".join(a[0] for a in r)


def episode_key(arm: str, regime: str, role: str, opener: str, i: int, j: int, k: int) -> str:
    return f"{arm}|{regime}|{role}|{opener}|{i}|{j}|{k}"


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["episode_key"] for line in f if line.strip()}


class FakeSonnet:
    """Offline stand-in for the agent arm: always answers 'A'."""

    def complete(self, system, messages, max_tokens):
        from llm_rl_scripts.car_dealer_persona.run import Completion
        return Completion(text="A", model="fake", stop_reason="end_turn", input_tokens=1, output_tokens=1)


def regimes_for(arm: an.Arm, regimes: List[str]) -> List[str]:
    return list(regimes) if arm.uses_text else [an.NO_TEXT_REGIME]


def run(cfg: LiveConfig, arms: List[str], regimes: List[str], bank: an.TextBank, out: Path, repeats: int, seed: int,
        jev_client=None, sonnet_client=None, roles=ROLES, openers=ROLES, types=None, opp_types=None,
        shuffle: bool = True, log=sys.stderr) -> dict:
    types = list(range(len(ALL_RANKINGS))) if types is None else types
    opp_types = list(range(len(ALL_RANKINGS))) if opp_types is None else opp_types
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    done = done_keys(path)
    t0 = time.time()
    cells: Dict[tuple, an.NoulCell] = {}
    ceilings: Dict[str, dict] = {}
    for role in roles:
        for i in types:
            c = an.NoulCell(cfg, role, ALL_RANKINGS[i])
            cells[(role, i)] = c
            for opener in openers:
                ceilings[f"{role}|{opener}|{_tag(ALL_RANKINGS[i])}"] = c.ceilings(opener)
    with (out / "ceilings.json").open("w") as f:
        json.dump({"config": cfg.to_json(), "kernel": REFERENCE.to_json(), "cells": ceilings}, f, indent=1)
    print(f"[{time.time()-t0:6.1f}s] ceilings solved for {len(ceilings)} (role, opener, type) cells", file=log, flush=True)
    n = errors = skipped = 0
    shuffle_master = random.Random(seed)
    with path.open("a") as f:
        for arm_name in arms:
            arm = an.parse_arm(arm_name)
            for regime in regimes_for(arm, regimes):
                for role in roles:
                    for i in types:
                        cell = cells[(role, i)]
                        agent = None
                        if arm.kind == an.JEV_AGENT:
                            agent = TextJevPolicy(jev_client, cfg, role, ALL_RANKINGS[i],
                                                  shuffle_rng=random.Random(shuffle_master.random()) if shuffle else None)
                        elif arm.kind == an.SONNET_AGENT:
                            agent = SonnetAgent(sonnet_client, cfg, role, ALL_RANKINGS[i],
                                                shuffle_rng=random.Random(shuffle_master.random()) if shuffle else None)
                        for opener in openers:
                            for j in opp_types:
                                for k in range(repeats):
                                    key = episode_key(arm.name, regime, role, opener, i, j, k)
                                    if key in done:
                                        skipped += 1
                                        continue
                                    rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|{k}")
                                    variants = an.variant_plan(seed, role, opener, i, j, k, bank.k)
                                    try:
                                        rec = an.play_episode(cell, arm, regime, opener, ALL_RANKINGS[j], rng, variants, bank, k, agent)
                                    except Exception as e:   # recorded as a stub, never silently dropped
                                        rec = {"arm": arm.name, "regime": regime, "focal": role, "opener": opener,
                                               "focal_ranking": list(ALL_RANKINGS[i]), "repeat": k,
                                               "private_truth_evaluator_only": {"opp_ranking": list(ALL_RANKINGS[j])},
                                               "error": f"{type(e).__name__}: {e}"}
                                        errors += 1
                                        print(f"  episode failed: {rec['error']}", file=log, flush=True)
                                    rec["episode_key"] = key
                                    f.write(json.dumps(rec) + "\n")
                                    f.flush()
                                    n += 1
                print(f"[{time.time()-t0:6.1f}s] arm={arm.name} regime={regime} done ({n} new episodes, {skipped} resumed)", file=log, flush=True)
    summary = {"env": ENV_VERSION, "design": DESIGN, "config": cfg.to_json(), "kernel": REFERENCE.to_json(),
               "bank": str(bank.dir), "bank_variants": bank.k, "arms": arms, "regimes": regimes,
               "agent_menu_decoder": an.AGENT_MENU_DECODER, "registered_hedge": str(an.REGISTERED_HEDGE), "filter_eta": an.ETA,
               "roles": list(roles), "openers": list(openers), "focal_types": [_tag(ALL_RANKINGS[i]) for i in types],
               "opp_types": [_tag(ALL_RANKINGS[j]) for j in opp_types], "repeats": repeats, "seed": seed,
               "sonnet_model": SONNET_MODEL, "episodes_written": n, "episodes_resumed": skipped, "error_stubs": errors,
               "elapsed_s": round(time.time() - t0, 2)}
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", help="registration tag; required and checked for --phase confirmatory")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--repeats", type=int, help="override the phase default (pilot only)")
    ap.add_argument("--seed", type=int, help="override the phase default (pilot only)")
    ap.add_argument("--types", default="all"); ap.add_argument("--opp-types", default="all")
    ap.add_argument("--roles", default=",".join(ROLES)); ap.add_argument("--openers", default=",".join(ROLES))
    ap.add_argument("--jev-backend", default="mock", choices=("mock", "http"))
    ap.add_argument("--sonnet-backend", default="fake", choices=("fake", "anthropic"))
    ap.add_argument("--no-shuffle", action="store_true")
    args = ap.parse_args(argv)

    repeats, seed = PHASES[args.phase]
    if args.phase == "confirmatory":
        if not args.tag:
            ap.error("--phase confirmatory needs --tag <registration tag>")
        if not head_has_tag(args.tag):
            ap.error(f"HEAD does not carry tag {args.tag}; the confirmatory run is made from the tagged registration commit")
        if args.repeats or args.seed or args.jev_backend != "http" or args.sonnet_backend != "anthropic":
            ap.error("the confirmatory phase takes the registered repeats and seed and the live backends")
    else:
        repeats = args.repeats or repeats
        seed = args.seed or seed
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        an.parse_arm(a)
    regimes = [r for r in args.regimes.split(",") if r]
    for r in regimes:
        if r not in REGIMES:
            ap.error(f"unknown regime {r!r}")
    types = None if args.types == "all" else [int(x) for x in args.types.split(",")]
    opp_types = None if args.opp_types == "all" else [int(x) for x in args.opp_types.split(",")]
    cfg = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))
    bank = an.TextBank(args.bank)
    jev_client = sonnet_client = None
    if any(an.parse_arm(a).kind == an.JEV_AGENT for a in arms):
        from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
        jev_client = JevClient(MockJevBackend() if args.jev_backend == "mock" else HttpJevBackend())
    if any(an.parse_arm(a).kind == an.SONNET_AGENT for a in arms):
        if args.sonnet_backend == "fake":
            sonnet_client = FakeSonnet()
        else:
            from llm_rl_scripts.car_dealer_persona.run import AnthropicClient
            sonnet_client = AnthropicClient(SONNET_MODEL)
    summary = run(cfg, arms, regimes, bank, Path(args.out), repeats, seed, jev_client, sonnet_client,
                  roles=args.roles.split(","), openers=args.openers.split(","), types=types, opp_types=opp_types,
                  shuffle=not args.no_shuffle)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
