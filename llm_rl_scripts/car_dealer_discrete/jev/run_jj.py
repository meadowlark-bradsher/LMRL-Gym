"""Run the Jev x Jev (J/J) cell of the discretized Car Dealer benchmark.

Jev sits in *both* seats (buyer and seller), each with its own private ranking,
and negotiates through the shared state machine. We score only the outcome and
efficiency families (sections 4.1/4.2/4.4) — pure Jev-only has no belief report.

Because jev-latest is stochastic at near-tie decisions (the lesson from the
Text-Nav program), ``--repeats K`` runs K independent rollouts per
(type pair, regime, starting side); the game is fixed by those, only Jev's
choices vary. Estimate success *rates*, not single trajectories.

Examples:
    # Offline logic check — no API key, no network, deterministic mock agents:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.run_jj --self-test

    # Tiny LIVE smoke (needs TYPESAFE_API_KEY); 2 sampled pairs, easy regime:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
        --pairs 2 --regime easy --start buyer --repeats 1

    # Fuller live run: all 36 pairs, both regimes, both starts, K=5
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.run_jj \
        --pairs all --regime both --start both --repeats 5 --out results/jj_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from fractions import Fraction
from typing import Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete import GAME_VERSION
from llm_rl_scripts.car_dealer_discrete.game import Deal, Ranking, all_type_pairs, utility
from llm_rl_scripts.car_dealer_discrete.jev.policy import JevNegotiationPolicy
from llm_rl_scripts.car_dealer_discrete.protocol import (
    ACCEPT,
    NegotiationGame,
    Observation,
    Policy,
)
from llm_rl_scripts.car_dealer_discrete.scoring import (
    RESERVATIONS,
    analyze_type_pair,
    score_episode,
)

COMPROMISE_ID = "P1-W1-D1"  # the unique deal worth exactly 0.5 to every agent


# --- offline mock agent (self-test only) -----------------------------------

def make_compromise_chooser(role: str, ranking: Ranking, r: Fraction):
    """A trivial, deterministic negotiator used only to exercise the loop offline.

    Accept any offer worth at least the reservation; otherwise propose the
    universal 0.5 compromise; otherwise reject / walk. Not a real agent — it just
    drives every code path (valid agreement in the easy regime, no-deal in hard).
    """

    def chooser(state: str, qid: str, options: List[str]) -> str:
        for opt in options:  # accept an acceptable standing offer
            if opt.startswith("ACCEPT "):
                deal_id = opt.split(" ", 1)[1]
                if utility(role, ranking, Deal.from_id(deal_id)) >= r:
                    return opt
        compromise = f"OFFER {COMPROMISE_ID}"
        if compromise in options:
            return compromise
        if "REJECT" in options:
            return "REJECT"
        if "WALK" in options:
            return "WALK"
        return options[0]

    return chooser


# --- a wrapper that records Jev's per-turn diagnostics ----------------------

class RecordingPolicy:
    """Wraps a policy, logging the choice/confidence it returns each turn."""

    def __init__(self, inner: Policy):
        self.inner = inner
        self.steps: List[dict] = []

    def __call__(self, obs: Observation):
        action = self.inner(obs)
        ans = getattr(self.inner, "last_answer", None)
        probs = ans.probabilities if ans else {}
        top = sorted(probs.values(), reverse=True)[:2] if probs else []
        margin = (top[0] - top[1]) if len(top) == 2 else None
        self.steps.append(
            {
                "actor": obs.role,
                "chosen": ans.choice if ans else None,
                "confidence": ans.confidence if ans else None,
                "margin": margin,
            }
        )
        return action


def make_seat_policy(
    role: str,
    buyer_rank: Ranking,
    seller_rank: Ranking,
    r: Fraction,
    *,
    live_client: Optional[JevClient],
    self_test: bool,
    reveal: bool,
    salience: bool,
    shuffle_rng: Optional[random.Random],
) -> RecordingPolicy:
    ranking = buyer_rank if role == "buyer" else seller_rank
    if self_test:
        backend = MockJevBackend(chooser=make_compromise_chooser(role, ranking, r))
        client = JevClient(backend)
    else:
        assert live_client is not None
        client = live_client
    inner = JevNegotiationPolicy(
        client, reveal_own_utility=reveal, salience=salience, shuffle_rng=shuffle_rng
    )
    return RecordingPolicy(inner)


# --- running ----------------------------------------------------------------

def run_episode(
    buyer_rank: Ranking,
    seller_rank: Ranking,
    regime: str,
    starting_side: str,
    seed: int,
    *,
    live_client: Optional[JevClient],
    self_test: bool,
    reveal: bool,
    salience: bool,
    shuffle: bool,
    max_proposals: int,
) -> dict:
    r = RESERVATIONS[regime]
    srng = random.Random(seed) if shuffle else None
    buyer_policy = make_seat_policy(
        "buyer", buyer_rank, seller_rank, r,
        live_client=live_client, self_test=self_test, reveal=reveal,
        salience=salience, shuffle_rng=srng,
    )
    seller_policy = make_seat_policy(
        "seller", buyer_rank, seller_rank, r,
        live_client=live_client, self_test=self_test, reveal=reveal,
        salience=salience, shuffle_rng=srng,
    )
    game = NegotiationGame(
        buyer_rank, seller_rank, r, buyer_policy, seller_policy,
        max_proposals=max_proposals, starting_side=starting_side,
    )
    result = game.play()
    analysis = analyze_type_pair(buyer_rank, seller_rank, r)
    score = score_episode(result, analysis)
    condition = {"salience": salience, "reveal_utility": reveal, "shuffle_options": shuffle}
    return _episode_record(
        result, score, regime, seed, starting_side, buyer_policy, seller_policy, condition
    )


def _frac(x) -> Optional[float]:
    return float(x) if x is not None else None


def _episode_record(result, score, regime, seed, starting_side, bp, sp, condition) -> dict:
    return {
        "game_version": GAME_VERSION,
        "seed": seed,
        "reservation_regime": regime,
        "condition": condition,
        "private_truth_evaluator_only": {
            "buyer_rank": list(result.buyer_rank),
            "seller_rank": list(result.seller_rank),
        },
        "policies": {"buyer": "jev", "seller": "jev"},
        "starting_side": starting_side,
        "turns": [
            {"actor": t.actor, "action": t.action.kind, "deal_id": t.action.deal_id}
            for t in result.turns
        ],
        "jev_steps": {"buyer": bp.steps, "seller": sp.steps},
        "result": {
            "outcome_type": result.outcome_type,
            "agreement": result.agreement,
            "accepted_deal_id": result.accepted_deal_id,
            "buyer_utility": _frac(result.buyer_utility),
            "seller_utility": _frac(result.seller_utility),
            "individually_rational": result.individually_rational,
            "proposals_used": result.proposals_used,
            "decisions": result.decisions,
        },
        "score": {
            "feasible": score.feasible,
            "valid_agreement": score.valid_agreement,
            "correct_no_deal": score.correct_no_deal,
            "buyer_gain": _frac(score.buyer_gain),
            "seller_gain": _frac(score.seller_gain),
            "min_gain": _frac(score.min_gain),
            "on_pareto_frontier": score.on_pareto_frontier,
            "nash_ratio": _frac(score.nash_ratio),
        },
    }


def summarize(records: List[dict]) -> Dict[str, dict]:
    by_regime: Dict[str, List[dict]] = {}
    for rec in records:
        by_regime.setdefault(rec["reservation_regime"], []).append(rec)

    out: Dict[str, dict] = {}
    for regime, recs in sorted(by_regime.items()):
        feas = [r for r in recs if r["score"]["feasible"]]
        infeas = [r for r in recs if not r["score"]["feasible"]]
        valid = [r for r in feas if r["score"]["valid_agreement"]]

        def mean(vals):
            vals = [v for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else None

        out[regime] = {
            "episodes": len(recs),
            "feasible_episodes": len(feas),
            "infeasible_episodes": len(infeas),
            "agreement_rate": mean([1.0 if r["result"]["agreement"] else 0.0 for r in recs]),
            "invalid_acceptances": sum(
                1 for r in recs if r["result"]["outcome_type"] == "invalid_acceptance"
            ),
            "valid_agreement_rate_feasible": mean(
                [1.0 if r["score"]["valid_agreement"] else 0.0 for r in feas]
            ),
            "correct_no_deal_rate_infeasible": mean(
                [1.0 if r["score"]["correct_no_deal"] else 0.0 for r in infeas]
            ),
            "mean_buyer_gain_feasible": mean([r["score"]["buyer_gain"] for r in feas]),
            "mean_seller_gain_feasible": mean([r["score"]["seller_gain"] for r in feas]),
            "mean_min_gain_feasible": mean([r["score"]["min_gain"] for r in feas]),
            "pareto_frontier_rate_valid": mean(
                [1.0 if r["score"]["on_pareto_frontier"] else 0.0 for r in valid]
            ),
            "mean_nash_ratio_feasible": mean([r["score"]["nash_ratio"] for r in feas]),
            "mean_proposals_to_agreement": mean(
                [r["result"]["proposals_used"] for r in valid]
            ),
        }
    return out


def select_pairs(spec: str, seed: int):
    pairs = all_type_pairs()
    if spec == "all":
        return pairs
    n = int(spec)
    rng = random.Random(seed)
    return rng.sample(pairs, min(n, len(pairs)))


def build_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jev x Jev discretized Car Dealer runner.")
    p.add_argument("--self-test", action="store_true", help="Offline mock agents; no API, no key.")
    p.add_argument("--pairs", default="all", help="'all' (36) or an integer sample size.")
    p.add_argument("--regime", choices=["easy", "hard", "both"], default="both")
    p.add_argument("--start", choices=["buyer", "seller", "both"], default="both")
    p.add_argument("--repeats", type=int, default=1, help="Independent rollouts per cell.")
    p.add_argument("--max-proposals", type=int, default=6)
    p.add_argument("--no-reveal-utility", action="store_true",
                   help="Ablation: hide the agent's own-utility numbers from option text.")
    p.add_argument("--salience", action="store_true",
                   help="Salience condition: state the deadline and standing/best-seen "
                        "offer in value terms relative to the no-deal payoff.")
    p.add_argument("--shuffle-options", action="store_true",
                   help="Randomize candidate order (section 5.1.6).")
    p.add_argument("--seed", type=int, default=0, help="Base seed for pair sampling / shuffling.")
    p.add_argument("--out", default=None, help="Directory for runs.jsonl + summary.json.")
    p.add_argument("--model", default=None, help="Override Jev model id (live only).")
    p.add_argument("--max-episodes", type=int, default=None, help="Hard cap on total episodes.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)
    regimes = ["easy", "hard"] if args.regime == "both" else [args.regime]
    starts = ["buyer", "seller"] if args.start == "both" else [args.start]
    pairs = select_pairs(args.pairs, args.seed)

    live_client: Optional[JevClient] = None
    if not args.self_test:
        backend = HttpJevBackend()  # raises if TYPESAFE_API_KEY is unset
        live_client = JevClient(backend, model=args.model) if args.model else JevClient(backend)

    records: List[dict] = []
    episode_index = 0
    stop = False
    for regime in regimes:
        for buyer_rank, seller_rank in pairs:
            for starting_side in starts:
                for k in range(args.repeats):
                    if args.max_episodes is not None and episode_index >= args.max_episodes:
                        stop = True
                        break
                    seed = args.seed * 1_000_003 + episode_index
                    rec = run_episode(
                        buyer_rank, seller_rank, regime, starting_side, seed,
                        live_client=live_client, self_test=args.self_test,
                        reveal=not args.no_reveal_utility, salience=args.salience,
                        shuffle=args.shuffle_options, max_proposals=args.max_proposals,
                    )
                    records.append(rec)
                    episode_index += 1
                    if args.verbose:
                        res = rec["result"]
                        print(
                            f"[{regime}] B={''.join(a[0] for a in buyer_rank)} "
                            f"S={''.join(a[0] for a in seller_rank)} start={starting_side} "
                            f"k={k}: {res['outcome_type']} "
                            f"deal={res['accepted_deal_id']} "
                            f"U=({res['buyer_utility']},{res['seller_utility']})"
                        )
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    summary = summarize(records)
    print("\n=== J/J summary ({} episodes) ===".format(len(records)))
    print(json.dumps(summary, indent=2, default=str))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "runs.jsonl"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote {len(records)} episodes to {args.out}/runs.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
