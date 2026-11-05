"""R0 end-to-end: collect the BC dataset from the oracle, train the per-row scorer,
and gate it against ``exact_ask``.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r0 --out results/drl_r0_<date>

Writes ``results.json`` (frozen), ``policy.pt`` (the trained net + meta) and a
rendered ``R0_RESULTS.md`` note. The gate passes when the trained policy sits
within ``--sigma`` SE of the exact ceiling (primary) and matches the oracle's
optimal action on at least ``--match-gate`` of validation nodes (secondary). The
oracle-shim evaluation is run first as a harness sanity check (it must sit within
SE of the ceiling by construction).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from .dataset import build_cells, collect_nodes, to_tensors
from .evaluate import evaluate, net_choice_fn, oracle_choice
from .net import PolicyValueNet
from .train import TrainConfig, train


def render_note(res: dict) -> str:
    g, s, o = res["gate"], res["eval_net"], res["eval_oracle_shim"]
    tr = res["train"]
    verdict = "PASS" if g["pass"] else "FAIL"
    lines = [
        "# CD-DRL R0 — behavior cloning of the exact menu policy",
        "",
        f"**Gate: {verdict}.** Registered regime H={res['config']['horizon']}, c_q={res['config']['ask_cost']}, "
        "T1 kernel (alpha0=4/5, beta=8, p_counter=1, p_hold=1, shortlist 3), shortlist menu.",
        "",
        "A standalone PyTorch per-row scorer + value head, trained purely by behavior cloning "
        "against `car_dealer_live.Cell.ask` (the `exact_ask` / C_plan oracle) over the same shortlist "
        "menu the Jev arm is shown. No RL, no reward shaping — R0 only establishes that the "
        "featurization and a tiny net can reproduce the exact menu policy.",
        "",
        "## Result",
        "",
        f"- Dataset: {tr['n_nodes']} unique focal nodes ({tr['n_train']} train / {tr['n_val']} val), "
        f"max menu rows {tr['max_rows']}.",
        f"- Validation optimal-match (argmax row in the oracle's within-EPS optimal set): "
        f"**{tr['val']['optimal_match']:.4f}** (exact same-row match {tr['val']['exact_match']:.4f}).",
        f"- Episodic ({s['n_episodes']} episodes, {s['repeats']} repeats x {len(s['opp_types'])} opponent types "
        f"x 24 (role,opener,type) cells): mean net gain **{s['mean_net']:.4f}** vs exact ceiling "
        f"**{s['mean_ceiling']:.4f}**; residual {s['mean_residual']:+.4f} "
        f"({s['residual_z']:+.2f} SE, within {s['sigma']:.0f} SE: {s['within_sigma_se']}).",
        f"- Per-decision optimal-match in play: {s['optimal_match']:.4f}; mean decision regret {s['mean_decision_regret']:.5f}.",
        "",
        f"Harness sanity (oracle-shim policy): residual {o['mean_residual']:+.4f} ({o['residual_z']:+.2f} SE), "
        f"optimal-match {o['optimal_match']:.4f} — confirms the eval loop reproduces `exact_ask`.",
        "",
        "## Gate detail",
        "",
        f"- Primary (within {g['sigma']:.0f} SE of ceiling): residual z = {s['residual_z']:+.2f} -> {g['primary_pass']}.",
        f"- Secondary (val optimal-match >= {g['match_gate']}): {tr['val']['optimal_match']:.4f} -> {g['secondary_pass']}.",
        "",
        "Regenerate: `PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_drl.run_r0 --out <dir>` "
        f"(seed {res['config']['seed']}).",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--collect-repeats", type=int, default=6)
    p.add_argument("--eval-repeats", type=int, default=20)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.01, help="Q* distillation temperature")
    p.add_argument("--ce-coef", type=float, default=0.25, help="auxiliary hard-CE weight")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=4.0, help="SE tolerance of the primary gate")
    p.add_argument("--match-gate", type=float, default=0.99, help="min validation optimal-match")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print("[build] cells (12: 2 roles x 6 types) ...", file=sys.stderr)
    cells = build_cells()
    print(f"[{time.time()-t0:6.1f}s] collecting BC nodes (repeats={args.collect_repeats}) ...", file=sys.stderr)
    nodes = collect_nodes(cells, repeats=args.collect_repeats, seed=args.seed)
    ds = to_tensors(nodes)
    print(f"[{time.time()-t0:6.1f}s] {len(nodes)} unique nodes; training (epochs={args.epochs}) ...", file=sys.stderr)
    net, tr = train(ds, TrainConfig(hidden=args.hidden, epochs=args.epochs, seed=args.seed,
                                    temperature=args.temperature, ce_coef=args.ce_coef))

    print(f"[{time.time()-t0:6.1f}s] eval: oracle shim ...", file=sys.stderr)
    eval_oracle = evaluate(cells, oracle_choice, repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)
    print(f"[{time.time()-t0:6.1f}s] eval: trained net ...", file=sys.stderr)
    eval_net = evaluate(cells, net_choice_fn(net), repeats=args.eval_repeats, seed=args.seed, sigma=args.sigma)

    primary = eval_net["within_sigma_se"]
    secondary = tr["val"]["optimal_match"] >= args.match_gate
    gate = {"pass": bool(primary and secondary), "primary_pass": bool(primary), "secondary_pass": bool(secondary),
            "sigma": args.sigma, "match_gate": args.match_gate}
    res = {
        "env": "live-car-dealer-v0.1", "rung": "R0",
        "config": {"horizon": 4, "ask_cost": "0", "seed": args.seed,
                   "collect_repeats": args.collect_repeats, "eval_repeats": args.eval_repeats,
                   "epochs": args.epochs, "hidden": args.hidden},
        "train": tr, "eval_net": eval_net, "eval_oracle_shim": eval_oracle, "gate": gate,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "meta": res["config"]},
               os.path.join(args.out, "policy.pt"))
    with open(os.path.join(args.out, "R0_RESULTS.md"), "w") as f:
        f.write(render_note(res))

    print(f"\n[{time.time()-t0:6.1f}s] GATE {'PASS' if gate['pass'] else 'FAIL'}  "
          f"val_optimal_match={tr['val']['optimal_match']:.4f}  "
          f"residual={eval_net['mean_residual']:+.4f} ({eval_net['residual_z']:+.2f} SE)  "
          f"optimal_match_in_play={eval_net['optimal_match']:.4f}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
