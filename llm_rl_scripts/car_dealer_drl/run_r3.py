"""R3 end-to-end: best-response to the CD-MIS opponent population, with transfer to
held-out opponents.

One policy is trained by RL against a MIXTURE of true opponents (the training
processes), planning and observing only with the assumed (registered T1) kernel --
shortlist menu, robustified belief filter (B_naive), Phi = V*_assumed shaping. It
never sees which process it faces. We then score it, O_assumed (the registered
shortlist oracle deployed under misspecification) and O_true (the clairvoyant
per-draw ceiling) on every process, split into the training mixture and the
held-out processes.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r3 --out results/drl_r3_<date>

Pre-registration (frozen before the run):
  * Transfer split: TRAIN = {matched, stochastic, anchoring, history};
    HELDOUT = {threshold, withholding, evasive} (evasive = answer noise, absent
    from training; the largest misspecification gap). Frozen in evaluate_mis.
  * RL-competence check: on 'matched' (no misspecification) the learned actor must
    reach O_assumed within max(2 SE, 0.01) -- an absolute 0.01 floor (~2.5% of the
    ceiling), since 2 SE over the full grid is a ~0.004 band that even an
    economically-optimal near-tie policy (cf. R0's in-play 0.922) would miss. If it
    does not, a shortfall elsewhere cannot be attributed to misspecification rather
    than to RL under-training, and the robustness read is withheld.
  * Per group, advantage_over_assumed = learned_net - O_assumed_net, with z = adv/SE:
      BEATS_PLANNER    z > +2
      MATCHES_PLANNER  |z| <= 2
      BELOW_PLANNER    z < -2
  * Transfer verdict: does the held-out advantage match the training advantage
    (robust transfer) or fall below it (overfit to the training mixture)?
  * Context: misspecification-gap-recovered = adv / (O_true - O_assumed); regret vs
    O_true (the unrestricted clairvoyant ceiling) is reported but is not the test --
    no menu-restricted, model-blind actor can reach O_true.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from .env_mis import net_greedy
from .evaluate_mis import HELDOUT_PROCESSES, TRAIN_PROCESSES, build_r3, evaluate_mis
from .train_r3 import R3Config, train_r3


def _group_verdict(g: dict) -> str:
    z = g["advantage_z"]
    if z > 2.0:
        return "BEATS_PLANNER"
    if z < -2.0:
        return "BELOW_PLANNER"
    return "MATCHES_PLANNER"


def render_note(res: dict) -> str:
    ev = res["eval"]
    tr, ho, al = ev["train"], ev["heldout"], ev["all"]
    pp = ev["per_process"]
    cfg = res["config"]
    comp = res["rl_competence"]
    lines = [
        "# CD-DRL R3 — best-response to the CD-MIS opponent population, with transfer",
        "",
        f"**Robustness (train mix): {res['verdict_train']}. Transfer (held-out): {res['verdict_heldout']}.** "
        f"Registered regime H={cfg['horizon']}, c_q={cfg['ask_cost']}; assumed kernel = registered T1, "
        "shortlist menu, B_naive belief filter.",
        "",
        "One RL policy trained against a mixture of TRUE opponents (planning/observing only with the "
        "assumed kernel; Phi = V*_assumed shaping; one-step shaped-reward advantage). It never sees the "
        "process. Scored against O_assumed (the registered oracle deployed under misspecification -- same "
        "menu and belief) and O_true (the clairvoyant per-draw ceiling, full action set + true belief).",
        "",
        f"Train processes: {', '.join(TRAIN_PROCESSES)}. Held-out: {', '.join(HELDOUT_PROCESSES)}.",
        "",
        "## RL-competence check (on 'matched', no misspecification)",
        "",
        f"- learned {comp['policy_net']:.4f} vs O_assumed {comp['oassumed_net']:.4f} "
        f"(adv {comp['advantage']:+.4f}); competent: {comp['competent']}. "
        + ("" if comp["competent"] else "**Robustness read withheld: shortfall may be RL under-training.**"),
        "",
        "## Results by group (mean true net)",
        "",
        "| group | learned | O_assumed | O_true | adv over assumed | z | gap recovered | regret vs O_true |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, g in (("train", tr), ("held-out", ho), ("all", al)):
        lines.append(f"| {name} | {g['policy_net']:.4f} | {g['oassumed_net']:.4f} | {g['otrue_net']:.4f} | "
                     f"{g['advantage_over_assumed']:+.4f} | {g['advantage_z']:+.2f} | "
                     f"{g['misspecification_gap_recovered']:+.2f} | {g['regret_vs_otrue']:.4f} |")
    lines += ["", "## Per process (learned vs O_assumed vs O_true)", "",
              "| process | split | learned | O_assumed | O_true | adv | ask/ep |",
              "|---|---|---|---|---|---|---|"]
    for p in TRAIN_PROCESSES + HELDOUT_PROCESSES:
        d = pp[p]
        split = "train" if p in TRAIN_PROCESSES else "held-out"
        lines.append(f"| {p} | {split} | {d['policy_net']:.4f} | {d['oassumed_net']:.4f} | "
                     f"{d['otrue_net']:.4f} | {d['advantage_over_assumed']:+.4f} | {d['ask_rate_per_episode']:.2f} |")
    lines += [
        "",
        "## Verdict detail",
        "",
        f"- Robustness (train): adv {tr['advantage_over_assumed']:+.4f} ({tr['advantage_z']:+.2f} SE) -> "
        f"**{res['verdict_train']}**.",
        f"- Transfer (held-out): adv {ho['advantage_over_assumed']:+.4f} ({ho['advantage_z']:+.2f} SE) -> "
        f"**{res['verdict_heldout']}**. {res['transfer_read']}",
        "",
        f"Training: {cfg['iters']} iters x {cfg['episodes_per_iter']} episodes on the training mixture; "
        f"best train-mean-net at iter {res['train_info']['best_iter']}.",
        f"Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r3 --out <dir>` "
        f"(seed {cfg['seed']}).",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--episodes-per-iter", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--eval-repeats", type=int, default=16, help="final-eval repeats per process")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print("[build] R3 bundle (assumed cells + per-process true kernels + O_true) ...", file=sys.stderr)
    b = build_r3()
    cfg = R3Config(hidden=args.hidden, iters=args.iters, episodes_per_iter=args.episodes_per_iter,
                   lr=args.lr, entropy_coef=args.entropy_coef, seed=args.seed)
    print(f"[{time.time()-t0:6.1f}s] training best-response to the training mixture ...", file=sys.stderr)
    net, train_info = train_r3(b, TRAIN_PROCESSES, cfg)
    print(f"[{time.time()-t0:6.1f}s] eval: learned vs O_assumed vs O_true, all processes ...", file=sys.stderr)
    ev = evaluate_mis(net_greedy(net), b, repeats=args.eval_repeats, seed=args.seed)

    mp = ev["per_process"]["matched"]
    # competence tolerance: max(2 SE, 0.01). The 0.01 floor (~2.5% of ceiling) keeps
    # an economically-optimal near-tie policy from failing a ~0.004 (2 SE) band.
    comp_se = mp["policy_se"]
    comp_tol = max(2.0 * comp_se, 0.01)
    competent = (mp["policy_net"] - mp["oassumed_net"]) >= -comp_tol
    rl_competence = {"policy_net": mp["policy_net"], "oassumed_net": mp["oassumed_net"],
                     "advantage": mp["policy_net"] - mp["oassumed_net"], "se": comp_se,
                     "tolerance": comp_tol, "competent": bool(competent)}

    verdict_train = _group_verdict(ev["train"]) if competent else "WITHHELD (RL under-trained)"
    verdict_heldout = _group_verdict(ev["heldout"]) if competent else "WITHHELD (RL under-trained)"
    d_adv = ev["heldout"]["advantage_over_assumed"] - ev["train"]["advantage_over_assumed"]
    transfer_read = ("held-out advantage tracks training (robust transfer)" if abs(d_adv) <= 2 * ev["heldout"]["advantage_se"]
                     else ("held-out advantage exceeds training" if d_adv > 0 else
                           "held-out advantage falls below training (partial overfit to the mixture)"))

    res = {
        "env": "live-car-dealer-v0.1", "rung": "R3",
        "config": {"horizon": 4, "ask_cost": "0", "seed": args.seed, "iters": args.iters,
                   "episodes_per_iter": args.episodes_per_iter, "hidden": args.hidden,
                   "lr": args.lr, "entropy_coef": args.entropy_coef, "eval_repeats": args.eval_repeats,
                   "train_processes": list(TRAIN_PROCESSES), "heldout_processes": list(HELDOUT_PROCESSES)},
        "train_info": train_info, "eval": ev, "rl_competence": rl_competence,
        "verdict_train": verdict_train, "verdict_heldout": verdict_heldout, "transfer_read": transfer_read,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "meta": res["config"]},
               os.path.join(args.out, "policy.pt"))
    with open(os.path.join(args.out, "R3_RESULTS.md"), "w") as f:
        f.write(render_note(res))

    print(f"\n[{time.time()-t0:6.1f}s] R3  competent={competent}  "
          f"train: {verdict_train} adv={ev['train']['advantage_over_assumed']:+.4f}  |  "
          f"held-out: {verdict_heldout} adv={ev['heldout']['advantage_over_assumed']:+.4f}  "
          f"({transfer_read})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
