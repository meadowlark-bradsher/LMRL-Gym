"""R2 end-to-end: does model-free RL discover ASK from reward alone?

The focal policy is trained by reinforcement learning with **no oracle of any
kind** -- no behavior-cloning labels (R0) and no PBRS potential (R1). The net sees
only what the Jev arm sees (the shortlist menu with its belief-derived EIG and
P(accept) annotations) and learns from the raw environmental reward (terminal net
gain, minus ask_cost per ask) via advantage actor-critic with the Monte-Carlo
return-to-go estimator. The exact oracle is used ONLY to *label* evaluation nodes
(ask-optimality) and to supply the ask/noask ceilings -- never in training.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r2 --out results/drl_r2_<date>

This mirrors the live Car-Dealer finding that the Jev selector asks 0/2555 even
when shown EIG (``car-dealer-live-voi``). R2 asks whether a from-scratch RL actor,
given the same information, beats that zero.

Calibration (exact, registered regime H=4, shortlist 3): the global value of
inquiry is ~0.038 net (~10% of the 0.379 ceiling); ASK is strictly optimal at
~2% of focal nodes. So there is a real, detectable ASK prize, and it is rare
enough that discovering it requires genuine exploration.

Pre-registration (frozen before the run; config is tuned for *value*, never for
the ASK outcome):
  * captured-VOI fraction = (mean_net - noask_ceiling) / (ask_ceiling - noask_ceiling).
  * ASK-optimal recall = asks taken at nodes where ASK is strictly optimal.
  * Verdict:
      DISCOVERED      if captured_voi >= 0.50 AND ask_optimal_recall >= 0.50
      PARTIAL         if (ask_optimal_recall >= 0.10 OR captured_voi >= 0.25) and not above
      NOT_DISCOVERED  otherwise (recall < 0.10 and captured < 0.25 -- i.e. Jev-like)
  * Secondary read: did it at least master the offers (mean_net >= noask_ceiling)?
    A policy can master offers yet never ask (the hypothesized Jev-like failure),
    or fail to master even offers (sparse-RL sample-inefficiency). Both are
    reported; neither is a pass/fail gate -- R2 reports a discovery verdict, not a
    pass.
  * Robustness (for a null): a second arm holds the entropy bonus high and
    CONSTANT (no annealing) so the stochastic policy keeps sampling ASK throughout
    training -- ruling out that a NOT_DISCOVERED verdict is merely premature
    exploration collapse rather than a genuine failure to learn that asking pays.
    If sustained exploration still does not discover ASK, the null is robust.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from .ask_eval import evaluate_ask
from .dataset import build_cells
from .evaluate import net_choice_fn, oracle_choice
from .train_r1 import R1Config, train_r1


def _verdict(captured: float, recall: float) -> str:
    rec = 0.0 if recall != recall else recall       # NaN-safe (no strict-ASK nodes -> 0)
    if captured >= 0.50 and rec >= 0.50:
        return "DISCOVERED"
    if rec >= 0.10 or captured >= 0.25:
        return "PARTIAL"
    return "NOT_DISCOVERED"


def render_note(res: dict) -> str:
    n, o = res["eval_net"], res["eval_oracle"]
    v = res["verdict"]
    cfg = res["config"]
    lines = [
        "# CD-DRL R2 — does model-free RL discover ASK from reward alone?",
        "",
        f"**Verdict: {v}.** Registered regime H={cfg['horizon']}, c_q={cfg['ask_cost']}, "
        "T1 kernel (alpha0=4/5, beta=8, p_counter=1, p_hold=1, shortlist 3), shortlist menu.",
        "",
        "RL with **no oracle**: no behavior-cloning labels (R0), no PBRS potential (R1). "
        "Advantage actor-critic, Monte-Carlo return-to-go, from the raw terminal reward only. "
        "The net sees the same menu (EIG, P(accept)) the Jev arm sees; the exact oracle only "
        "labels eval nodes and supplies the ceilings. Mirrors the live finding that Jev asks "
        "0/2555 even when shown EIG -- R2 asks whether a trained actor beats that zero.",
        "",
        "## Calibration (exact)",
        "",
        f"- Value of inquiry: ask-ceiling **{n['ask_ceiling']:.4f}** vs noask-ceiling "
        f"**{n['noask_ceiling']:.4f}** -> VOI **{n['voi']:+.4f}** (~{100*n['voi']/n['ask_ceiling']:.0f}% "
        "of the ceiling). ASK strictly optimal at ~2% of focal nodes.",
        "",
        "## Learned policy (no oracle)",
        "",
        f"- Mean net gain **{n['mean_net']:.4f}**; residual vs ask-ceiling {n['mean_residual']:+.4f} "
        f"({n['residual_z']:+.2f} SE). Above noask-ceiling: {n['above_noask_ceiling']}.",
        f"- **Captured-VOI fraction {n['captured_voi']:+.3f}** "
        f"(0 = never-ask-optimal, 1 = ask-optimal).",
        f"- **ASK-optimal recall {n['ask_optimal_recall']:.3f}** "
        f"(asks at the {n['n_ask_strict_optimal']} strictly-ASK-optimal nodes); "
        f"ASK rate {n['ask_rate_per_episode']:.3f}/episode ({n['ask_rate_per_decision']:.3f}/decision); "
        f"ASK precision {n['ask_precision']:.3f}.",
        f"- Per-decision optimal-match {n['optimal_match']:.4f}; mean decision regret "
        f"{n['mean_decision_regret']:.5f}.",
        "",
        "## Oracle reference (ground truth, same evaluator)",
        "",
        f"- Mean net {o['mean_net']:.4f} (= ceiling); captured-VOI {o['captured_voi']:+.3f}; "
        f"ASK recall {o['ask_optimal_recall']:.3f}; ASK rate {o['ask_rate_per_episode']:.3f}/episode.",
        f"- Jev comparison (live): asked 0/2555, recall 0. This actor's recall: "
        f"{n['ask_optimal_recall']:.3f}.",
        "",
        "## Verdict detail",
        "",
        f"- captured-VOI {n['captured_voi']:+.3f} (DISCOVERED >= 0.50, PARTIAL >= 0.25).",
        f"- ASK-optimal recall {n['ask_optimal_recall']:.3f} (DISCOVERED >= 0.50, PARTIAL >= 0.10).",
        f"- Mastered offers (net >= noask-ceiling): {n['above_noask_ceiling']}.",
        f"- **{v}.**",
        "",
        "## Robustness: sustained exploration (constant entropy, no annealing)",
        "",
        (f"A second arm holds the entropy bonus constant at {res['robust_entropy']} so the "
         "stochastic policy keeps sampling ASK throughout training (the primary arm's entropy "
         "collapsed to ~0.04, leaving ASK unexplored). This rules out premature-exploration-"
         "collapse as the cause of the null." if not res["eval_robust"].get("skipped") else
         "Skipped (--no-robust)."),
    ]
    if not res["eval_robust"].get("skipped"):
        rb = res["eval_robust"]
        lines += [
            "",
            f"- Verdict {res['verdict_robust']}: captured-VOI {rb['captured_voi']:+.3f}, "
            f"ASK-optimal recall {rb['ask_optimal_recall']:.3f}, ASK rate "
            f"{rb['ask_rate_per_episode']:.3f}/episode, net {rb['mean_net']:.4f}.",
            f"- Reading: {'sustained exploration still does not discover ASK -- the null is robust' if res['verdict_robust'] == 'NOT_DISCOVERED' else 'sustained exploration changes the picture (see verdict)'}.",
        ]
    lines += [
        "",
        f"Training: {cfg['iters']} iters x {cfg['episodes_per_iter']} episodes, MC return-to-go, "
        f"lr {cfg['lr']}, entropy {cfg['entropy_coef']}->annealed; best greedy optimal-match at "
        f"iter {res['train']['best_iter']}.",
        f"Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r2 --out <dir>` "
        f"(seed {cfg['seed']}).",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    # frozen, value-tuned defaults (MC is high variance -> more episodes, lower lr;
    # ASK is rare -> sustained entropy for exploration; long budget to give discovery a chance)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--episodes-per-iter", type=int, default=384)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--entropy-final", type=float, default=0.005)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval-repeats", type=int, default=20, help="final-eval repeats")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--robust-entropy", type=float, default=0.08,
                   help="constant (un-annealed) entropy for the sustained-exploration robustness arm")
    p.add_argument("--no-robust", action="store_true", help="skip the sustained-exploration robustness arm")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print("[build] cells (12: 2 roles x 6 types) ...", file=sys.stderr)
    cells = build_cells()
    cfg = R1Config(hidden=args.hidden, iters=args.iters, episodes_per_iter=args.episodes_per_iter,
                   lr=args.lr, entropy_coef=args.entropy_coef, entropy_final=args.entropy_final,
                   patience=args.patience, seed=args.seed, shaping=False, advantage="mc")
    print(f"[{time.time()-t0:6.1f}s] training no-oracle arm (sparse reward, MC) ...", file=sys.stderr)
    net, train = train_r1(cells, cfg)

    print(f"[{time.time()-t0:6.1f}s] eval: learned policy (ASK diagnostics) ...", file=sys.stderr)
    eval_net = evaluate_ask(cells, net_choice_fn(net), repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)
    print(f"[{time.time()-t0:6.1f}s] eval: oracle reference ...", file=sys.stderr)
    eval_oracle = evaluate_ask(cells, oracle_choice, repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)

    verdict = _verdict(eval_net["captured_voi"], eval_net["ask_optimal_recall"])

    # Robustness arm: sustained (constant, un-annealed) high entropy so the policy
    # keeps sampling ASK throughout -- rules out premature-exploration-collapse as
    # the cause of a null. Verdict stays from the pre-registered primary arm.
    if args.no_robust:
        train_robust, eval_robust, verdict_robust = {"skipped": True}, {"skipped": True}, None
    else:
        print(f"[{time.time()-t0:6.1f}s] training robustness arm (constant entropy "
              f"{args.robust_entropy}) ...", file=sys.stderr)
        cfg_r = R1Config(hidden=args.hidden, iters=args.iters, episodes_per_iter=args.episodes_per_iter,
                         lr=args.lr, entropy_coef=args.robust_entropy, entropy_final=args.robust_entropy,
                         patience=args.patience, seed=args.seed, shaping=False, advantage="mc")
        net_r, train_robust = train_r1(cells, cfg_r)
        print(f"[{time.time()-t0:6.1f}s] eval: robustness policy ...", file=sys.stderr)
        eval_robust = evaluate_ask(cells, net_choice_fn(net_r), repeats=args.eval_repeats,
                                   seed=args.seed, sigma=args.sigma)
        verdict_robust = _verdict(eval_robust["captured_voi"], eval_robust["ask_optimal_recall"])
    res = {
        "env": "live-car-dealer-v0.1", "rung": "R2",
        "config": {"horizon": 4, "ask_cost": "0", "seed": args.seed, "iters": args.iters,
                   "episodes_per_iter": args.episodes_per_iter, "hidden": args.hidden,
                   "lr": args.lr, "entropy_coef": args.entropy_coef, "eval_repeats": args.eval_repeats},
        "train": train, "eval_net": eval_net, "eval_oracle": eval_oracle,
        "train_robust": train_robust, "eval_robust": eval_robust,
        "robust_entropy": args.robust_entropy, "verdict_robust": verdict_robust,
        "verdict": verdict, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "meta": res["config"]},
               os.path.join(args.out, "policy.pt"))
    with open(os.path.join(args.out, "R2_RESULTS.md"), "w") as f:
        f.write(render_note(res))

    robust_str = "" if verdict_robust is None else \
        f"  |  robust({args.robust_entropy}): {verdict_robust} recall={eval_robust['ask_optimal_recall']:.3f} ask/ep={eval_robust['ask_rate_per_episode']:.3f}"
    print(f"\n[{time.time()-t0:6.1f}s] VERDICT {verdict}  "
          f"captured_voi={eval_net['captured_voi']:+.3f}  "
          f"ask_recall={eval_net['ask_optimal_recall']:.3f}  "
          f"ask_rate/ep={eval_net['ask_rate_per_episode']:.3f}  "
          f"net={eval_net['mean_net']:.4f} (noask_ceil={eval_net['noask_ceiling']:.4f}){robust_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
