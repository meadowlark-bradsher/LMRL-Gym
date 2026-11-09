"""CD-NOUL-2 Stage 2 runner (NEG-8).

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.run_noul2 --phase pilot --out results/noul2_pilot_<date> \
        --opponent kernel|scripted|llm --realizer template|llm --jev-backend mock|http --sonnet-backend fake|anthropic \
        [--arms ...] [--briefs candid,guarded,adversarial] [--kernel-fit <kernel_fit.json>] [--map platt --clamp 0.95] [--workers 8]

Cells: (focal role, opener, focal type, opponent type) = 144 per (arm, brief), repeated ``repeats`` times; episode
key = (arm, brief, role, opener, i, j, k); episodes append to ``<out>/runs.jsonl`` as they finish and the run
resumes by key. Opponent moves are logged to ``<out>/opponent_moves.jsonl`` for ``kernel_fit`` (gate 4). Work is
split over the 12 (role, focal type) planner cells, each owned by one worker (the oracles' caches are not shared
across threads); the opponent, realizer and judge are network calls in the live modes. The confirmatory phase
refuses to run unless HEAD carries the registration tag and every backend is live.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.protocol_live import ENV_VERSION, LiveConfig
from llm_rl_scripts.car_dealer_persona.run import head_has_tag

from . import arms_noul2 as a2
from . import opponent_llm as op
from .agents_noul import SonnetAgent, TextJevPolicy
from .realizer import EchoClient, Realizer, TemplateRealizer
from .verbalize import MODEL as SONNET_MODEL

DESIGN = "CD-NOUL-2"
PHASES = {"pilot": (1, 20261120), "confirmatory": (5, 20261121)}     # placeholders; the registration fixes them
ROLES = ("buyer", "seller")
DEFAULT_CALIBRATION = "results/noul_bank_20261108/calibration_noul2.json"


def _tag(r):
    return "/".join(a[0] for a in r)


def episode_key(arm: str, brief: str, role: str, opener: str, i: int, j: int, k: int) -> str:
    return f"{arm}|{brief}|{role}|{opener}|{i}|{j}|{k}"


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["episode_key"] for line in f if line.strip()}


class FakeSonnet:
    """Offline stand-in for Sonnet in every seat: the agent answers 'A'; the judge returns p_yes 0.5."""

    def complete(self, system, messages, max_tokens):
        from llm_rl_scripts.car_dealer_persona.run import Completion
        text = '{"p_yes": 0.5}' if system and system.startswith("You grade one reply") else "A"
        return Completion(text=text, model="fake", stop_reason="end_turn", input_tokens=1, output_tokens=1)


def make_opponent(mode: str, role: str, ranking, cfg: LiveConfig, brief: str, rng: random.Random, sonnet_client,
                  true_params=REFERENCE):
    kernel = T1Kernel(role, cfg, true_params)
    if mode == "scripted":
        return a2.ScriptedT1Opponent(kernel, ranking, rng)
    if mode == "kernel":
        client = op.KernelClient(kernel, ranking, rng, cfg)
        return op.KernelOpponent(client, brief, role, ranking, cfg, fallback=kernel, rng=rng)
    if mode == "llm":
        return op.LLMOpponent(sonnet_client, brief, role, ranking, cfg, fallback=kernel, rng=rng)
    raise ValueError(mode)


def make_realizer(mode: str, arm: a2.Arm, role: str, cfg: LiveConfig, sonnet_client, rng: random.Random):
    """Templated text when the run or the arm says so; otherwise the model realizer (a live call)."""
    if mode == "template" or arm.realizer == "template":
        return TemplateRealizer()
    return Realizer(sonnet_client, role, cfg, rng=rng)


def load_kernel_fits(path: Optional[str]) -> Dict[str, dict]:
    """{brief: KernelParams json} from one or more kernel_fit outputs (a JSON list or a single report)."""
    if not path:
        return {}
    data = json.load(open(path))
    reports = data if isinstance(data, list) else [data]
    out = {}
    for rep in reports:
        if rep.get("adopt_fit"):
            out[rep.get("brief") or "*"] = rep["fitted_all"]
    return out


def _cell_job(args) -> List[Tuple[str, dict, list]]:
    (cfg, role, i, jobs, seed, opponent_mode, realizer_mode, calibration, decoders, jev_client, sonnet_client,
     shuffle, fitted_by_brief) = args
    out = []
    for arm_name, brief, opener, j, k, key in jobs:
        arm = a2.parse_arm(arm_name)
        fitted = a2.params_from_json(fitted_by_brief[brief]) if brief in fitted_by_brief else (
            a2.params_from_json(fitted_by_brief["*"]) if "*" in fitted_by_brief else REFERENCE)
        cell_key = (role, i, str(fitted.to_json()))
        cell = _CELLS.get(cell_key)
        if cell is None:
            cell = _CELLS[cell_key] = a2.Stage2Cell(cfg, role, ALL_RANKINGS[i], fitted=fitted, t1=REFERENCE)
        rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|{k}")
        agent = None
        if arm.kind == a2.JEV_AGENT:
            agent = TextJevPolicy(jev_client, cfg, role, ALL_RANKINGS[i], shuffle_rng=random.Random(f"{seed}|shuffle|{key}") if shuffle else None)
        elif arm.kind == a2.SONNET_AGENT:
            agent = SonnetAgent(sonnet_client, cfg, role, ALL_RANKINGS[i], shuffle_rng=random.Random(f"{seed}|shuffle|{key}") if shuffle else None)
        opp_rng = random.Random(f"{seed}|opp|{key}") if opponent_mode != "scripted" else rng
        opponent = make_opponent(opponent_mode, a2.other(role), ALL_RANKINGS[j], cfg, brief, opp_rng, sonnet_client)
        realizer = make_realizer(realizer_mode, arm, role, cfg, sonnet_client, random.Random(f"{seed}|real|{key}"))
        moves: list = []
        try:
            rec = a2.play_episode(cell, arm, brief, opener, ALL_RANKINGS[j], rng, opponent, realizer, decoders, calibration, k,
                                  agent, move_log=moves, episode_id=key)
        except Exception as e:   # recorded as a stub, never silently dropped
            rec = {"arm": arm.name, "brief": brief, "focal": role, "opener": opener, "focal_ranking": list(ALL_RANKINGS[i]),
                   "repeat": k, "private_truth_evaluator_only": {"opp_ranking": list(ALL_RANKINGS[j])},
                   "error": f"{type(e).__name__}: {e}"}
        rec["episode_key"] = key
        out.append((key, rec, moves if opponent_mode != "scripted" else []))
    return out


_CELLS: Dict[tuple, a2.Stage2Cell] = {}   # per-process cache; each (role, i) is owned by one worker at a time


def run(cfg: LiveConfig, arms: List[str], briefs: List[str], out: Path, repeats: int, seed: int, *, opponent_mode: str,
        realizer_mode: str, calibration: a2.Calibration, jev_client=None, sonnet_client=None, kernel_fits: Optional[Dict[str, dict]] = None,
        roles=ROLES, openers=ROLES, types=None, opp_types=None, shuffle: bool = True, workers: int = 1, log=sys.stderr) -> dict:
    types = list(range(len(ALL_RANKINGS))) if types is None else types
    opp_types = list(range(len(ALL_RANKINGS))) if opp_types is None else opp_types
    kernel_fits = kernel_fits or {}
    out.mkdir(parents=True, exist_ok=True)
    path, moves_path = out / "runs.jsonl", out / "opponent_moves.jsonl"
    done = done_keys(path)
    t0 = time.time()
    decoders = a2.Decoders(calibration, jev_client, sonnet_client)
    # ceilings under the reference (fitted or T1) kernel
    ceilings = {}
    for role in roles:
        for i in types:
            fitted = a2.params_from_json(kernel_fits["*"]) if "*" in kernel_fits else REFERENCE
            c = a2.Stage2Cell(cfg, role, ALL_RANKINGS[i], fitted=fitted)
            for opener in openers:
                ceilings[f"{role}|{opener}|{_tag(ALL_RANKINGS[i])}"] = c.ceilings(opener)
    with (out / "ceilings.json").open("w") as f:
        json.dump({"config": cfg.to_json(), "kernel": (kernel_fits.get("*") or REFERENCE.to_json()), "cells": ceilings}, f, indent=1)
    print(f"[{time.time()-t0:6.1f}s] ceilings solved for {len(ceilings)} cells", file=log, flush=True)
    # jobs grouped by (role, i): one worker owns a planner cell at a time
    groups: Dict[tuple, list] = {}
    skipped = 0
    for arm_name in arms:
        for brief in a2.briefs_for(arm_name, tuple(briefs)):
            for role in roles:
                for i in types:
                    for opener in openers:
                        for j in opp_types:
                            for k in range(repeats):
                                key = episode_key(arm_name, brief, role, opener, i, j, k)
                                if key in done:
                                    skipped += 1
                                    continue
                                groups.setdefault((role, i), []).append((arm_name, brief, opener, j, k, key))
    n = errors = moves_n = 0
    with path.open("a") as f, moves_path.open("a") as mf:
        args = [(cfg, role, i, jobs, seed, opponent_mode, realizer_mode, calibration, decoders, jev_client, sonnet_client, shuffle, kernel_fits)
                for (role, i), jobs in sorted(groups.items())]
        if workers <= 1:
            results = (_cell_job(a) for a in args)
            for res in results:
                for key, rec, moves in res:
                    errors += int("error" in rec); n += 1
                    f.write(json.dumps(rec) + "\n")
                    for m in moves:
                        mf.write(json.dumps(m) + "\n"); moves_n += 1
                f.flush(); mf.flush()
                print(f"[{time.time()-t0:6.1f}s] {n} episodes written ({errors} stubs)", file=log, flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_cell_job, a) for a in args]
                for fut in as_completed(futs):
                    for key, rec, moves in fut.result():
                        errors += int("error" in rec); n += 1
                        f.write(json.dumps(rec) + "\n")
                        for m in moves:
                            mf.write(json.dumps(m) + "\n"); moves_n += 1
                    f.flush(); mf.flush()
                    print(f"[{time.time()-t0:6.1f}s] {n} episodes written ({errors} stubs)", file=log, flush=True)
    summary = {"env": ENV_VERSION, "design": DESIGN, "config": cfg.to_json(), "reference_kernel": REFERENCE.to_json(),
               "kernel_fits": kernel_fits, "arms": arms, "briefs": briefs, "opponent": opponent_mode, "realizer": realizer_mode,
               "calibration": calibration.to_json(), "agent_menu": [a2.AGENT_MENU_DECODER, a2.AGENT_MENU_HEDGE], "filter_eta": a2.ETA,
               "brief_hash": op.brief_hash(), "offer_prompt_hash": a2.dc.offer_prompt_hash(), "roles": list(roles), "openers": list(openers),
               "focal_types": [_tag(ALL_RANKINGS[i]) for i in types], "opp_types": [_tag(ALL_RANKINGS[j]) for j in opp_types],
               "repeats": repeats, "seed": seed, "sonnet_model": SONNET_MODEL, "episodes_written": n, "episodes_resumed": skipped,
               "error_stubs": errors, "opponent_moves_logged": moves_n, "workers": workers, "elapsed_s": round(time.time() - t0, 2)}
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", help="registration tag; required and checked for --phase confirmatory")
    ap.add_argument("--arms", default=",".join(a2.design_arms()))
    ap.add_argument("--briefs", default=",".join(op.BRIEFS))
    ap.add_argument("--opponent", default="kernel", choices=("scripted", "kernel", "llm"))
    ap.add_argument("--realizer", default="template", choices=("template", "llm"))
    ap.add_argument("--jev-backend", default="mock", choices=("mock", "http"))
    ap.add_argument("--sonnet-backend", default="fake", choices=("fake", "anthropic"))
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--map", default=a2.DEFAULT_MAP); ap.add_argument("--clamp", type=float, default=a2.DEFAULT_CLAMP)
    ap.add_argument("--kernel-fit", default=None, help="kernel_fit.py report(s) JSON; adopted fits are used per brief")
    ap.add_argument("--repeats", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--types", default="all"); ap.add_argument("--opp-types", default="all")
    ap.add_argument("--roles", default=",".join(ROLES)); ap.add_argument("--openers", default=",".join(ROLES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-shuffle", action="store_true")
    args = ap.parse_args(argv)

    repeats, seed = PHASES[args.phase]
    if args.phase == "confirmatory":
        if not args.tag:
            ap.error("--phase confirmatory needs --tag <registration tag>")
        if not head_has_tag(args.tag):
            ap.error(f"HEAD does not carry tag {args.tag}; the confirmatory run is made from the tagged registration commit")
        if (args.repeats or args.seed or args.opponent != "llm" or args.realizer != "llm" or args.jev_backend != "http"
                or args.sonnet_backend != "anthropic"):
            ap.error("the confirmatory phase takes the registered repeats and seed, the live opponent and realizer, and the live backends")
    else:
        repeats = args.repeats or repeats
        seed = args.seed or seed
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        a2.parse_arm(a)
    briefs = [b for b in args.briefs.split(",") if b]
    for b in briefs:
        if b not in op.BRIEFS:
            ap.error(f"unknown brief {b!r}")
    types = None if args.types == "all" else [int(x) for x in args.types.split(",")]
    opp_types = None if args.opp_types == "all" else [int(x) for x in args.opp_types.split(",")]
    cfg = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))
    calibration = a2.Calibration(args.calibration, a2.MapSpec(args.map, args.clamp))
    from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
    jev_client = JevClient(MockJevBackend() if args.jev_backend == "mock" else HttpJevBackend())
    if args.sonnet_backend == "fake":
        sonnet_client = FakeSonnet()
    else:
        from llm_rl_scripts.car_dealer_persona.run import AnthropicClient
        sonnet_client = AnthropicClient(SONNET_MODEL)
    if args.opponent == "llm" and args.sonnet_backend != "anthropic":
        ap.error("--opponent llm needs --sonnet-backend anthropic (a live call); use --opponent kernel offline")
    summary = run(cfg, arms, briefs, Path(args.out), repeats, seed, opponent_mode=args.opponent, realizer_mode=args.realizer,
                  calibration=calibration, jev_client=jev_client, sonnet_client=sonnet_client, kernel_fits=load_kernel_fits(args.kernel_fit),
                  roles=args.roles.split(","), openers=args.openers.split(","), types=types, opp_types=opp_types,
                  shuffle=not args.no_shuffle, workers=args.workers)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
