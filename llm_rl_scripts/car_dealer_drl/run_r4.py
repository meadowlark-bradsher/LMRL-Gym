"""R4 end-to-end: how much of R1's V* recovery survives as the shaping potential
degrades from the exact oracle toward oracle-free?

A degradation ladder in the registered H=4 matched regime (opponent = assumed =
registered T1, so the exact ceiling 0.3792 is known), holding the learner fixed
(advantage actor-critic, one-step shaped-reward estimator -- the R1 setup) and
varying ONLY the potential Phi:

    exact     Phi = V*_assumed              (the R1 reference; unbiased bootstrap)
    mismatch  Phi = V* under a WRONG kernel (biased parameters, still value-shaped)
    noask     Phi = V*_noask                (weaker oracle: ignores the value of inquiry)
    entropy   Phi = 1 - H(belief)/log2(N)   (NO oracle; the deepink entropy potential)
    none      Phi = 0                       (no shaping; nstep1 floor, ~ R2)

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r4 --out results/drl_r4_<date>

Pre-registration (frozen before the run):
  * Each arm's net gain is scored vs the exact ceiling; recovery fraction =
    (net - none_net) / (ceiling - none_net): 1 = full V* recovery, 0 = the floor.
  * The entropy potential rewards information gain, and in Car-Dealer ASK is
    strictly optimal at only ~2% of nodes (unlike 20Q, where asking is the whole
    game). Under the bootstrapped nstep1 estimator that can bias the actor toward
    OVER-ASKING, so entropy may recover less and raise the ASK rate above exact; the
    weaker value oracle V*_noask biases the opposite way (UNDER-asking). ASK rate
    and optimal-match are reported per arm to see this directly.
  * An arm "recovers V*" if it sits within 4 SE of the ceiling (the R1 gate).
  The wrong kernel and the entropy scale are fixed below before the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import torch

from .ask_eval import evaluate_ask
from .dataset import build_cells
from .evaluate import net_choice_fn
from .net import PolicyValueNet
from .potentials import make_phi_mismatch, phi_entropy, phi_exact, phi_noask
from .train_r1 import R1Config, train_r1

from llm_rl_scripts.car_dealer_live.opponent import KernelParams

# Frozen before the run: a plausibly-wrong opponent model, and the entropy scales.
WRONG_KERNEL = KernelParams(alpha0=Fraction(3, 5), beta=4.0, p_counter=1.0, p_hold=1.0, shortlist=3)
# Scale sweep for the oracle-free entropy potential: the primary (pre-registered) scale is
# 0.38 (~ the ceiling); the gentler scales probe whether any failure is scale-sensitivity
# (a weaker potential helps) or categorical (rewarding information, not value, misleads here).
ENTROPY_SCALES = (0.05, 0.15, 0.38)


def render_note(res: dict) -> str:
    arms = res["arms"]
    cfg = res["config"]
    lines = [
        "# CD-DRL R4 — weaker / absent oracles: how much V* recovery survives a degraded potential",
        "",
        f"Registered regime H={cfg['horizon']}, c_q={cfg['ask_cost']}, matched (opponent = assumed = "
        "registered T1), shortlist menu. One learner (nstep1 actor-critic, R1 setup); only the shaping "
        "potential Phi changes. Exact ceiling (= R1): "
        f"**{arms['exact']['ceiling']:.4f}**.",
        "",
        "Under the one-step estimator Phi enters the bootstrap, so potential quality is the whole story: "
        "Phi = V* is unbiased (R1 recovers V*), weaker potentials trade bias for being cheap or oracle-free.",
        "",
        "## Ladder (mean net vs the exact ceiling)",
        "",
        "| potential | oracle | mean net | residual (SE) | recovery | optimal-match | ASK/ep | ASK recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ent_arms = [k for k in arms if k.startswith("entropy@")]
    order = ["exact", "mismatch", "noask"] + ent_arms + ["none"]
    oracle_of = {"exact": "exact V*", "mismatch": "wrong-kernel V*", "noask": "V*_noask (no inquiry)",
                 "none": "none", **{k: "none (belief entropy)" for k in ent_arms}}
    for name in order:
        a = arms[name]
        lines.append(f"| {name} | {oracle_of[name]} | {a['mean_net']:.4f} | "
                     f"{a['mean_residual']:+.4f} ({a['residual_z']:+.1f}) | {a['recovery']:+.2f} | "
                     f"{a['optimal_match']:.3f} | {a['ask_rate_per_episode']:.2f} | {a['ask_optimal_recall']:.3f} |")
    lines += [
        "",
        f"Recovery fraction = (net - none_net) / (ceiling - none_net): 1 = full V* recovery, 0 = the "
        "no-shaping floor. 'recovers V*' (within 4 SE of the ceiling): "
        + ", ".join(f"{n}={arms[n]['recovers']}" for n in order) + ".",
        "",
        "## Reading",
        "",
        res["reading"],
        "",
        f"Wrong kernel: alpha0={WRONG_KERNEL.alpha0}, beta={WRONG_KERNEL.beta} (registered: 4/5, 8). "
        f"Entropy scales {ENTROPY_SCALES} (primary 0.38). Training: {cfg['iters']} iters x {cfg['episodes_per_iter']} "
        "episodes, nstep1, per arm.",
        f"Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r4 --out <dir>` "
        f"(seed {cfg['seed']}).",
    ]
    return "\n".join(lines) + "\n"


def _arm_config(args, shaping: bool) -> R1Config:
    return R1Config(hidden=args.hidden, iters=args.iters, episodes_per_iter=args.episodes_per_iter,
                    lr=args.lr, entropy_coef=args.entropy_coef, entropy_final=args.entropy_final,
                    advantage="nstep1", shaping=shaping, seed=args.seed)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--episodes-per-iter", type=int, default=192)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--entropy-final", type=float, default=0.001)
    p.add_argument("--eval-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=4.0)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print("[build] cells (12) ...", file=sys.stderr)
    cells = build_cells()
    potentials = {
        "exact": (phi_exact, True),
        "mismatch": (make_phi_mismatch(cells, WRONG_KERNEL), True),
        "noask": (phi_noask, True),
    }
    for s in ENTROPY_SCALES:
        potentials[f"entropy@{s}"] = (phi_entropy(s), True)
    potentials["none"] = (None, False)
    arms = {}
    nets = {}
    for name, (phi, shaping) in potentials.items():
        print(f"[{time.time()-t0:6.1f}s] training arm '{name}' ...", file=sys.stderr)
        net, _ = train_r1(cells, _arm_config(args, shaping), potential=phi)
        nets[name] = net
        ev = evaluate_ask(cells, net_choice_fn(net), repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)
        arms[name] = {"mean_net": ev["mean_net"], "ceiling": ev["ask_ceiling"],
                      "mean_residual": ev["mean_residual"], "residual_z": ev["residual_z"],
                      "recovers": bool(ev["within_sigma_se"]), "optimal_match": ev["optimal_match"],
                      "ask_rate_per_episode": ev["ask_rate_per_episode"],
                      "ask_optimal_recall": ev["ask_optimal_recall"], "mean_decision_regret": ev["mean_decision_regret"]}

    ceiling = arms["exact"]["ceiling"]
    floor = arms["none"]["mean_net"]
    span = ceiling - floor
    for a in arms.values():
        a["recovery"] = (a["mean_net"] - floor) / span if abs(span) > 1e-9 else float("nan")

    # pre-registered reading
    exa, mis, noa = arms["exact"], arms["mismatch"], arms["noask"]
    ent_keys = sorted([k for k in arms if k.startswith("entropy@")], key=lambda k: float(k.split("@")[1]))
    parts = [f"Recovery: exact {exa['recovery']:+.2f}, mismatch {mis['recovery']:+.2f}, "
             f"noask {noa['recovery']:+.2f}, "
             + ", ".join(f"{k} {arms[k]['recovery']:+.2f}" for k in ent_keys) + ", none 0."]
    if abs(mis["recovery"] - exa["recovery"]) < 0.1 and abs(noa["recovery"] - exa["recovery"]) < 0.1:
        parts.append("All three VALUE potentials (exact, wrong-kernel, inquiry-blind V*_noask) recover V* "
                     "fully: V* is nearly invariant to the opponent model here, and the small value of inquiry "
                     "is captured from the true one-step reward even when the potential ignores it.")
    best_ent = max(ent_keys, key=lambda k: arms[k]["recovery"])
    if all(arms[k]["recovery"] < 0.5 for k in ent_keys):
        parts.append(f"The oracle-free entropy potential fails at every scale tested (best {best_ent}: "
                     f"{arms[best_ent]['recovery']:+.2f}): rewarding information gain rather than value misleads "
                     "the bootstrapped learner in Car-Dealer, where ASK is optimal at only ~2% of nodes -- the "
                     "opposite of 20Q, where information gain IS the objective. This is categorical, not mere "
                     "scale-sensitivity.")
    else:
        parts.append(f"The entropy potential is scale-sensitive: it fails at large scale but {best_ent} recovers "
                     f"{arms[best_ent]['recovery']:+.2f}. A gentle oracle-free information bonus can help; a strong "
                     "one corrupts the bootstrapped policy.")
    reading = " ".join(parts)

    res = {
        "env": "live-car-dealer-v0.1", "rung": "R4",
        "config": {"horizon": 4, "ask_cost": "0", "seed": args.seed, "iters": args.iters,
                   "episodes_per_iter": args.episodes_per_iter, "hidden": args.hidden, "lr": args.lr,
                   "entropy_coef": args.entropy_coef, "eval_repeats": args.eval_repeats,
                   "wrong_kernel": {"alpha0": str(WRONG_KERNEL.alpha0), "beta": WRONG_KERNEL.beta},
                   "entropy_scales": list(ENTROPY_SCALES)},
        "arms": arms, "reading": reading, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save({name: n.state_dict() for name, n in nets.items()}, os.path.join(args.out, "policies.pt"))
    with open(os.path.join(args.out, "R4_RESULTS.md"), "w") as f:
        f.write(render_note(res))

    _order = ["exact", "mismatch", "noask"] + ent_keys + ["none"]
    print(f"\n[{time.time()-t0:6.1f}s] R4 recovery: "
          + "  ".join(f"{n}={arms[n]['recovery']:+.2f}(ask {arms[n]['ask_rate_per_episode']:.2f})"
                      for n in _order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
