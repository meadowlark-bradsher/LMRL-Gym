"""R1 end-to-end: train the focal policy by reinforcement learning on the
PBRS-shaped belief-MDP (Phi = V*) and gate it against ``exact_ask``, with a
pre-registered shaping-off ablation.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r1 --out results/drl_r1_<date>

Writes ``results.json`` (frozen), ``policy.pt`` and ``R1_RESULTS.md``.

Pre-registration (frozen before the run; do not move after seeing numbers):
  * H1 (primary): the shaped-RL greedy policy recovers V* -- episodic mean net
    gain within ``--sigma`` (=4) SE of the exact ceiling, pooled over all
    (role, opener, focal type) x opponent type x repeat episodes. This is the
    same tolerance R0 and ``car_dealer_live`` use.
  * H2 (secondary): the shaped-RL policy is clearly on-policy-optimal --
    per-decision optimal-match >= ``--match-gate`` (=0.90, the level R0's clone
    reached in play), well above chance.
  * H3 (contrast): shaping is what buys it at this budget. The shaping-off arm
    (Phi = 0, sparse terminal reward, identical architecture, budget and seed)
    is reported alongside. Each arm uses the advantage estimator that is correct
    for it (best-effort both ways): the shaped arm the one-step shaped reward
    (low variance *because* Phi = V* absorbs the continuation), the sparse arm
    the Monte-Carlo return-to-go (the standard estimator for sparse terminal
    reward -- the one-step reward would be identically zero on every non-terminal
    step without a potential, starving intermediate decisions by construction).
    Expectation: the sparse arm recovers less of the ceiling within the same
    budget (residual further from 0 and/or lower optimal-match). If it *also*
    passes H1/H2, the honest conclusion is that the registered horizon (H=4,
    <=~3 focal decisions) is too short for shaping to matter here -- a finding
    that motivates the longer-horizon rungs, not a failure of R1.

The gate verdict is H1 AND H2 on the shaped arm. H3 is recorded as evidence, not
a pass/fail gate (its direction is pre-registered but the short horizon may make
shaping unnecessary).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from .dataset import build_cells
from .evaluate import evaluate, net_choice_fn, oracle_choice
from .train_r1 import R1Config, train_r1


def render_note(res: dict) -> str:
    g = res["gate"]
    sh, sp, orc = res["eval_shaped"], res["eval_sparse"], res["eval_oracle_shim"]
    verdict = "PASS" if g["pass"] else "FAIL"
    cfg = res["config"]
    lines = [
        "# CD-DRL R1 — RL recovers V* with dense help (PBRS, Phi = V*)",
        "",
        f"**Gate: {verdict}.** Registered regime H={cfg['horizon']}, c_q={cfg['ask_cost']}, "
        "T1 kernel (alpha0=4/5, beta=8, p_counter=1, p_hold=1, shortlist 3), shortlist menu.",
        "",
        "Advantage actor-critic on the belief-MDP. The net learns its policy from reward only; "
        "the exact oracle enters **solely** as the potential `Phi = V*` in the Ng-Harada-Russell "
        "shaped reward `r' = R + Phi(s') - Phi(s)` (terminal clamp `Phi = 0`), never as an action "
        "label. Phi = V* makes the expected one-step shaped reward equal the optimal advantage "
        "`A*(s,a)`, so credit assignment is dense while the optimal policy is unchanged.",
        "",
        "## Result (shaped arm)",
        "",
        f"- Training: {cfg['iters']} iters x {cfg['episodes_per_iter']} episodes; best greedy "
        f"optimal-match at iter {res['train_shaped']['best_iter']}.",
        f"- Episodic ({sh['n_episodes']} episodes): mean net gain **{sh['mean_net']:.4f}** vs exact "
        f"ceiling **{sh['mean_ceiling']:.4f}**; residual {sh['mean_residual']:+.4f} "
        f"({sh['residual_z']:+.2f} SE, within {sh['sigma']:.0f} SE: {sh['within_sigma_se']}).",
        f"- Per-decision optimal-match **{sh['optimal_match']:.4f}**; mean decision regret "
        f"{sh['mean_decision_regret']:.5f}.",
        "",
        "## Contrast (H3): shaping off (Phi = 0, sparse terminal reward, same budget/seed)",
        "",
        "Best-effort both ways: the shaped arm uses the one-step shaped-reward advantage "
        "(low variance because Phi=V* absorbs the continuation); the sparse arm uses the "
        "Monte-Carlo return-to-go (without a potential the one-step reward is 0 on every "
        "non-terminal step, so MC is the correct sparse-reward estimator).",
        "",
        f"- Mean net gain {sp['mean_net']:.4f}; residual {sp['mean_residual']:+.4f} "
        f"({sp['residual_z']:+.2f} SE, within {sp['sigma']:.0f} SE: {sp['within_sigma_se']}).",
        f"- Per-decision optimal-match {sp['optimal_match']:.4f}; mean decision regret "
        f"{sp['mean_decision_regret']:.5f}.",
        f"- Shaping delta: optimal-match {sh['optimal_match'] - sp['optimal_match']:+.4f}, "
        f"regret {sp['mean_decision_regret'] - sh['mean_decision_regret']:+.5f} "
        f"(positive = shaping helps). Read: {res['shaping_read']}",
        "",
        f"Harness sanity (oracle-shim policy): residual {orc['mean_residual']:+.4f} "
        f"({orc['residual_z']:+.2f} SE), optimal-match {orc['optimal_match']:.4f}.",
        "",
        "## Gate detail",
        "",
        f"- H1 primary (shaped within {g['sigma']:.0f} SE of ceiling): z = {sh['residual_z']:+.2f} "
        f"-> {g['primary_pass']}.",
        f"- H2 secondary (shaped optimal-match >= {g['match_gate']}): {sh['optimal_match']:.4f} "
        f"-> {g['secondary_pass']}.",
        f"- H3 contrast (direction pre-registered, not a gate): {res['shaping_read']}",
        "",
        f"Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r1 --out <dir>` "
        f"(seed {cfg['seed']}).",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--episodes-per-iter", type=int, default=192)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--eval-repeats", type=int, default=20, help="final-gate eval repeats")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=4.0, help="SE tolerance of the primary gate")
    p.add_argument("--match-gate", type=float, default=0.90, help="min shaped optimal-match (H2)")
    p.add_argument("--no-ablation", action="store_true", help="skip the shaping-off contrast arm")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print("[build] cells (12: 2 roles x 6 types) ...", file=sys.stderr)
    cells = build_cells()

    def cfg_for(shaping: bool) -> R1Config:
        # Best-effort estimator per arm: shaped -> one-step shaped reward (low variance
        # because Phi=V* absorbs the continuation); sparse -> MC return-to-go (the
        # one-step reward is 0 on non-terminal steps without a potential).
        return R1Config(hidden=args.hidden, iters=args.iters, episodes_per_iter=args.episodes_per_iter,
                        lr=args.lr, entropy_coef=args.entropy_coef, seed=args.seed, shaping=shaping,
                        advantage="nstep1" if shaping else "mc")

    print(f"[{time.time()-t0:6.1f}s] training shaped arm (Phi=V*) ...", file=sys.stderr)
    net, train_shaped = train_r1(cells, cfg_for(True))
    print(f"[{time.time()-t0:6.1f}s] eval: shaped greedy ...", file=sys.stderr)
    eval_shaped = evaluate(cells, net_choice_fn(net), repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)

    if args.no_ablation:
        train_sparse, eval_sparse = {"skipped": True}, {"skipped": True, "mean_net": float("nan"),
            "mean_residual": float("nan"), "residual_z": float("nan"), "within_sigma_se": None,
            "optimal_match": float("nan"), "mean_decision_regret": float("nan"), "sigma": args.sigma}
    else:
        print(f"[{time.time()-t0:6.1f}s] training sparse arm (Phi=0) ...", file=sys.stderr)
        net_sp, train_sparse = train_r1(cells, cfg_for(False))
        print(f"[{time.time()-t0:6.1f}s] eval: sparse greedy ...", file=sys.stderr)
        eval_sparse = evaluate(cells, net_choice_fn(net_sp), repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)

    print(f"[{time.time()-t0:6.1f}s] eval: oracle shim ...", file=sys.stderr)
    eval_oracle = evaluate(cells, oracle_choice, repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)

    primary = bool(eval_shaped["within_sigma_se"])
    secondary = bool(eval_shaped["optimal_match"] >= args.match_gate)
    helps = (not args.no_ablation) and (
        eval_shaped["optimal_match"] > eval_sparse["optimal_match"] + 1e-6
        or eval_sparse["mean_decision_regret"] > eval_shaped["mean_decision_regret"] + 1e-6)
    if args.no_ablation:
        shaping_read = "ablation skipped"
    elif helps:
        shaping_read = "shaping helps (H3 direction confirmed)"
    elif eval_sparse["within_sigma_se"] and eval_sparse["optimal_match"] >= args.match_gate:
        shaping_read = "horizon too short for shaping to matter (sparse also solves it)"
    else:
        shaping_read = "shaping no better on these metrics"

    gate = {"pass": bool(primary and secondary), "primary_pass": primary, "secondary_pass": secondary,
            "sigma": args.sigma, "match_gate": args.match_gate}
    res = {
        "env": "live-car-dealer-v0.1", "rung": "R1",
        "config": {"horizon": 4, "ask_cost": "0", "seed": args.seed, "iters": args.iters,
                   "episodes_per_iter": args.episodes_per_iter, "hidden": args.hidden,
                   "lr": args.lr, "entropy_coef": args.entropy_coef, "eval_repeats": args.eval_repeats},
        "train_shaped": train_shaped, "train_sparse": train_sparse,
        "eval_shaped": eval_shaped, "eval_sparse": eval_sparse, "eval_oracle_shim": eval_oracle,
        "shaping_read": shaping_read, "gate": gate, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "meta": res["config"]},
               os.path.join(args.out, "policy.pt"))
    with open(os.path.join(args.out, "R1_RESULTS.md"), "w") as f:
        f.write(render_note(res))

    print(f"\n[{time.time()-t0:6.1f}s] GATE {'PASS' if gate['pass'] else 'FAIL'}  "
          f"shaped: residual={eval_shaped['mean_residual']:+.4f} ({eval_shaped['residual_z']:+.2f} SE) "
          f"optimal_match={eval_shaped['optimal_match']:.4f}  |  {shaping_read}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
