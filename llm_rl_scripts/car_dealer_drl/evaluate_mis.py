"""R3 evaluation under opponent misspecification: score a focal policy against the
CD-MIS population, per process, between two references.

* **O_true** (clairvoyant, per-draw ceiling): the exact true-kernel oracle over the
  FULL action set at the exact true posterior -- the best response to the *known*
  opponent. Upper reference.
* **O_assumed** (misspecified planner): the registered shortlist oracle on the
  assumed belief, run against the true opponent -- i.e. R0/R1's exact planner
  deployed without knowing the opponent changed. The apples-to-apples baseline for
  the menu-restricted learned actor (same menu, same belief filter).

The learned actor sits somewhere between. The R3 question is whether a policy
trained to best-respond to the *population* beats O_assumed (robustness) and
whether that advantage holds on held-out processes (transfer). All three are scored
on the same (role, focal type) x opponent type x repeat grid with the ``evaluate``
seed formula.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import PROCESSES, true_kernel
from llm_rl_scripts.car_dealer_live.oracle import Oracle
from llm_rl_scripts.car_dealer_live.protocol_live import other
from llm_rl_scripts.car_dealer_live.run_phase1 import Cell

from .dataset import REGISTERED_CFG, REGISTERED_PARAMS, ROLES, TYPE_IDX, build_cells
from .env import ChooseFn
from .env_mis import DEFAULT_FILTER, assumed_oracle_choose, otrue_net_gain, rollout_mis

# Pre-registered transfer split: train on the mixture below, hold the rest out.
TRAIN_PROCESSES = ("matched", "stochastic", "anchoring", "history")
HELDOUT_PROCESSES = ("threshold", "withholding", "evasive")


@dataclass
class R3Bundle:
    """Assumed-side cells (shared across processes) plus per-process true kernels and
    clairvoyant oracles, for every (role, focal type)."""
    assumed_cells: Dict[Tuple[str, int], Cell]
    true_kernels: Dict[Tuple[str, int, str], T1Kernel] = field(default_factory=dict)
    o_true: Dict[Tuple[str, int, str], Oracle] = field(default_factory=dict)


def build_r3(processes: Sequence[str] = PROCESSES, roles=ROLES, types=TYPE_IDX,
             cfg=REGISTERED_CFG, params=REGISTERED_PARAMS) -> R3Bundle:
    assumed = build_cells(cfg, params, roles, types)   # registered T1, shortlist menu
    b = R3Bundle(assumed_cells=assumed)
    for role in roles:
        for i in types:
            ranking = ALL_RANKINGS[i]
            for p in processes:
                tk = true_kernel(p, other(role), cfg)
                b.true_kernels[(role, i, p)] = tk
                b.o_true[(role, i, p)] = Oracle(cfg, role, ranking, tk, ask_allowed=True)
    return b


def _policy_by_process(choose: ChooseFn, b: R3Bundle, processes: Sequence[str],
                       roles, types, opp_types, repeats: int, seed: int,
                       shaping: bool = False, spec=DEFAULT_FILTER) -> Dict[str, Dict]:
    """Mean true net gain (and ASK rate) of ``choose`` per process, via rollout_mis."""
    out: Dict[str, Dict] = {}
    for p in processes:
        nets, asks = [], []
        for role in roles:
            for i in types:
                cell = b.assumed_cells[(role, i)]
                tk = b.true_kernels[(role, i, p)]
                for opener in roles:
                    for j in opp_types:
                        for k in range(repeats):
                            rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|{k}")
                            ep = rollout_mis(cell, tk, opener, ALL_RANKINGS[j], rng, choose,
                                             shaping=shaping, spec=spec)
                            nets.append(ep.net_gain); asks.append(ep.asks)
        out[p] = {"mean_net": float(np.mean(nets)), "se": float(np.std(nets, ddof=1) / np.sqrt(len(nets))),
                  "ask_rate_per_episode": float(np.mean(asks)), "n": len(nets)}
    return out


def _otrue_by_process(b: R3Bundle, processes: Sequence[str], roles, types, opp_types,
                      repeats: int, seed: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in processes:
        nets = []
        for role in roles:
            for i in types:
                cell = b.assumed_cells[(role, i)]
                tk = b.true_kernels[(role, i, p)]
                ot = b.o_true[(role, i, p)]
                for opener in roles:
                    for j in opp_types:
                        for k in range(repeats):
                            rng = random.Random(f"{seed}|{role}|{opener}|{i}|{j}|{k}")
                            nets.append(otrue_net_gain(cell.cfg, role, ALL_RANKINGS[i], tk, ot,
                                                       opener, ALL_RANKINGS[j], rng))
        out[p] = {"mean_net": float(np.mean(nets)), "n": len(nets)}
    return out


def mean_net(choose: ChooseFn, b: R3Bundle, processes: Sequence[str], roles=ROLES, types=TYPE_IDX,
             opp_types=TYPE_IDX, repeats: int = 4, seed: int = 0, shaping: bool = False) -> float:
    """Scalar mean net over the given processes (for model selection during training)."""
    per = _policy_by_process(choose, b, processes, roles, types, opp_types, repeats, seed, shaping)
    return float(np.mean([per[p]["mean_net"] for p in processes]))


def _group(per_policy, per_oassumed, per_otrue, processes) -> Dict:
    pol = np.mean([per_policy[p]["mean_net"] for p in processes])
    oa = np.mean([per_oassumed[p]["mean_net"] for p in processes])
    ot = np.mean([per_otrue[p]["mean_net"] for p in processes])
    adv = pol - oa
    # SE of the mean advantage over processes (processes weighted equally, treated independent)
    adv_se = float(np.sqrt(sum(per_policy[p].get("se", 0.0) ** 2 + per_oassumed[p].get("se", 0.0) ** 2
                               for p in processes)) / len(processes))
    z = adv / adv_se if adv_se > 0 else 0.0
    denom = ot - oa
    recovered = adv / denom if abs(denom) > 1e-9 else float("nan")
    return {"policy_net": float(pol), "oassumed_net": float(oa), "otrue_net": float(ot),
            "advantage_over_assumed": float(adv), "advantage_se": adv_se, "advantage_z": float(z),
            "regret_vs_otrue": float(ot - pol), "misspecification_gap_recovered": recovered,
            "processes": list(processes)}


def evaluate_mis(choose: ChooseFn, b: R3Bundle, roles=ROLES, types=TYPE_IDX, opp_types=TYPE_IDX,
                 repeats: int = 12, seed: int = 0,
                 train_processes=TRAIN_PROCESSES, heldout_processes=HELDOUT_PROCESSES) -> Dict:
    """Full R3 report: the learned policy, O_assumed and O_true per process, grouped
    into the pre-registered train / held-out splits."""
    all_p = tuple(train_processes) + tuple(heldout_processes)
    per_policy = _policy_by_process(choose, b, all_p, roles, types, opp_types, repeats, seed)
    per_oassumed = _policy_by_process(assumed_oracle_choose, b, all_p, roles, types, opp_types, repeats, seed)
    per_otrue = _otrue_by_process(b, all_p, roles, types, opp_types, repeats, seed)
    per_process = {p: {"policy_net": per_policy[p]["mean_net"], "policy_se": per_policy[p]["se"],
                       "oassumed_net": per_oassumed[p]["mean_net"], "otrue_net": per_otrue[p]["mean_net"],
                       "advantage_over_assumed": per_policy[p]["mean_net"] - per_oassumed[p]["mean_net"],
                       "ask_rate_per_episode": per_policy[p]["ask_rate_per_episode"]} for p in all_p}
    return {
        "repeats": repeats, "seed": seed,
        "train": _group(per_policy, per_oassumed, per_otrue, train_processes),
        "heldout": _group(per_policy, per_oassumed, per_otrue, heldout_processes),
        "all": _group(per_policy, per_oassumed, per_otrue, all_p),
        "per_process": per_process,
    }
