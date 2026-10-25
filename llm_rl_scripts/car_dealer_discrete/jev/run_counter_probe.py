"""Single-decision probe: the counter an UNINFORMED responder makes when it MUST counter.

Hypothesis under test (PREREGISTRATION_COUNTER.md): under identical, mirror-symmetric
information (prior-only aid), a seller-as-responder's counter is less acceptable to
the starter than a buyer-as-responder's.

Why a single-decision probe rather than full episodes: the quantity is one decision,
and in natural episodes responders rarely counter (10-16%), plausibly only in the
tough situations — a selection problem. Here the responder is shown the starter's
opening and a menu of OFFERS ONLY (no ACCEPT / REJECT / WALK), so it must counter,
and every context yields a decision. One API call per decision.

Contexts are the feasible openings actually logged in a prior-only run (the opening
is the starter's first offer, made under the uniform prior with identical
information in both roles). Fixing the context and drawing fresh responder
decisions isolates the counter.

Example:
    # offline logic check
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.run_counter_probe \
        results/jj_hard_20261025/prior_only --self-test --repeats 1 --out /tmp/x
    # live (pre-registered command in PREREGISTRATION_COUNTER.md)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete import GAME_VERSION
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS, Deal, utility
from llm_rl_scripts.car_dealer_discrete.jev.aid import AcceptanceAid, AidConfig
from llm_rl_scripts.car_dealer_discrete.jev.policy import JevNegotiationPolicy
from llm_rl_scripts.car_dealer_discrete.protocol import OFFER, Action, Observation, TurnRecord
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

OTHER = {"buyer": "seller", "seller": "buyer"}


def load_contexts(path: str) -> List[dict]:
    runs = path if path.endswith(".jsonl") else os.path.join(path, "runs.jsonl")
    out = []
    with open(runs) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "result" not in r or not r["score"]["feasible"]:
                continue
            t0 = r["turns"][0]
            assert t0["actor"] == r["starting_side"] and t0["action"] == OFFER, "first turn must be the starter's offer"
            tr = r["private_truth_evaluator_only"]
            out.append({
                "buyer_rank": tuple(tr["buyer_rank"]), "seller_rank": tuple(tr["seller_rank"]),
                "start": r["starting_side"], "opening": t0["deal_id"], "regime": r["reservation_regime"],
                "source_seed": r.get("seed"),
            })
    return out


def build_obs(ctx: dict):
    start, resp = ctx["start"], OTHER[ctx["start"]]
    rank = {"buyer": ctx["buyer_rank"], "seller": ctx["seller_rank"]}
    r = RESERVATIONS[ctx["regime"]]
    legal = tuple(Action(OFFER, d.id) for d in ALL_DEALS)  # forced counter: offers only
    history = (TurnRecord(start, Action(OFFER, ctx["opening"])),)
    obs = Observation(role=resp, own_ranking=rank[resp], reservation=r, max_proposals=6,
                      proposals_used=1, outstanding_offer=ctx["opening"], history=history,
                      legal_actions=legal)
    return obs, rank, r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("contexts", help="runs.jsonl (or its dir) whose feasible openings become contexts.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--aid-mode", choices=["prior_only", "bayes", "placebo"], default="prior_only")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="Provenance only; Jev is stochastic server-side.")
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-backoff", type=float, default=5.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    ctxs = load_contexts(args.contexts)
    client = JevClient(MockJevBackend()) if args.self_test else JevClient(HttpJevBackend())
    policy = JevNegotiationPolicy(client, aid=AcceptanceAid(AidConfig(mode=args.aid_mode)))

    os.makedirs(args.out, exist_ok=True)
    out_f = open(os.path.join(args.out, "decisions.jsonl"), "w")
    n = errors = 0
    cells = {"seller_responds": [0, 0], "buyer_responds": [0, 0]}  # [acceptable, total]
    stop = False
    for ci, ctx in enumerate(ctxs):
        for k in range(args.repeats):
            if args.max_decisions is not None and n >= args.max_decisions:
                stop = True; break
            obs, rank, r = build_obs(ctx)
            start, resp = ctx["start"], OTHER[ctx["start"]]
            rec: Optional[dict] = None; last_err = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    action = policy(obs)
                    if action.kind != OFFER:
                        raise RuntimeError(f"forced-counter menu violated: {action}")
                    ans = policy.last_answer
                    d = Deal.from_id(action.deal_id); od = Deal.from_id(ctx["opening"])
                    rec = {
                        "game_version": GAME_VERSION, "probe": "forced_counter", "aid_mode": args.aid_mode,
                        "context_index": ci, "repeat": k, "regime": ctx["regime"], "start": start, "responder": resp,
                        "cell": "seller_responds" if resp == "seller" else "buyer_responds",
                        "private_truth_evaluator_only": {"buyer_rank": list(ctx["buyer_rank"]), "seller_rank": list(ctx["seller_rank"])},
                        "opening": ctx["opening"],
                        "opening_u_to_responder": float(utility(resp, rank[resp], od)),
                        "opening_acceptable_to_responder": bool(utility(resp, rank[resp], od) >= r),
                        "counter": action.deal_id,
                        "counter_own_u": float(utility(resp, rank[resp], d)),
                        "counter_u_to_starter": float(utility(start, rank[start], d)),
                        "counter_acceptable_to_starter": bool(utility(start, rank[start], d) >= r),
                        "shown_pacc_of_counter": policy.last_acceptance[action.deal_id] if policy.last_acceptance else None,
                        "confidence": ans.confidence if ans else None,
                    }
                    break
                except Exception as e:
                    last_err = e
                    print(f"[retry {attempt}/{args.max_retries}] decision {n} failed: {e}", file=sys.stderr, flush=True)
                    time.sleep(args.retry_backoff * attempt)
            if rec is None:
                errors += 1
                rec = {"error": repr(last_err), "context_index": ci, "repeat": k, "start": start, "cell": "seller_responds" if resp == "seller" else "buyer_responds"}
            else:
                c = cells[rec["cell"]]; c[0] += int(rec["counter_acceptable_to_starter"]); c[1] += 1
            out_f.write(json.dumps(rec) + "\n"); out_f.flush(); n += 1
            if args.verbose and "counter" in rec:
                print(f"[{rec['cell']}] opening={rec['opening']} -> counter={rec['counter']} acceptable={rec['counter_acceptable_to_starter']} pacc={rec['shown_pacc_of_counter']:.2f}")
        if stop: break
    out_f.close()
    summary = {"contexts": len(ctxs), "decisions_attempted": n, "decisions_failed": errors, "aid_mode": args.aid_mode,
               "acceptable_rate_by_cell": {c: (v[0] / v[1] if v[1] else None) for c, v in cells.items()},
               "n_by_cell": {c: v[1] for c, v in cells.items()}}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2)); print(f"\nWrote {n} decisions ({errors} failed) to {args.out}/decisions.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
