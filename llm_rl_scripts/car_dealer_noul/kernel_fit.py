"""CD-NOUL-2 Phase 0: fit the T1 kernel's parameters to an opponent's logged moves (NEG-7).

The planner needs p(opponent move | type). Stage 2 keeps the T1 kernel *form* (`opponent.T1Kernel`:
aspiration falling to the reservation, softmax(β·U) proposals over a shortlist, accept iff U ≥ aspiration,
hold / concede) and asks only which **parameters** fit the live opponent: α₀, β, p_counter, p_hold,
accept_noise (the shortlist stays at the registered value — it caps the planner's branching). The fit is
maximum likelihood of the observed moves under `T1Kernel.distribution`, by coordinate descent with a
golden-section search per parameter (no scipy), from the registered T1 and from a mid-point start.

Records are one JSON object per opponent move (``record_move`` writes them; ``replay`` rebuilds the public
state), so the pilot's runner logs them and this module never needs the model. The held-out report splits
by episode, fits on the training episodes and scores the test episodes under the fitted and the T1
parameters; gate 4 of ``DESIGN_NOUL2.md`` adopts the fit for a brief only if the gain is ≥ 0.05 nats per move.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.kernel_fit --records <moves.jsonl> --out <fit.json>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from llm_rl_scripts.car_dealer_live.opponent import KernelParams, T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, PROPOSE, Action, Event, LiveConfig, PublicState, apply, new_game

FLOOR = 1e-6                      # probability floor for a move the kernel gives zero mass (counted)
GATE_GAIN_NATS = 0.05             # DESIGN_NOUL2 §6 gate 4
BOUNDS = {"alpha0": None, "beta": (0.0, 40.0), "p_counter": (0.0, 1.0), "p_hold": (0.0, 1.0), "accept_noise": (0.0, 0.5)}
FIT_PARAMS = ("alpha0", "beta", "p_counter", "p_hold", "accept_noise")
ALPHA_DENOM = 1000


# ---- records -----------------------------------------------------------------------------------------

def action_from_key(key: str) -> Action:
    kind, _, rest = key.partition(" ")
    if kind == PROPOSE:
        return Action(PROPOSE, deal_id=rest)
    if kind == ASK:
        return Action(ASK, question_id=rest)
    return Action(kind)


def events_to_json(events: Sequence[Event]) -> List[list]:
    return [[e.actor, e.action.key, e.answer] for e in events]


def replay(cfg: LiveConfig, opener: str, events: Iterable[list]) -> PublicState:
    state = new_game(cfg, opener)
    for actor, key, answer in events:
        assert state.turn == actor, f"replay: expected {state.turn} to move, record says {actor}"
        state = apply(state, action_from_key(key), answer)
    return state


def record_move(episode: str, state: PublicState, role: str, ranking, action: Action, brief: str = "") -> dict:
    """One logged opponent move: enough to rebuild the state the move was taken in."""
    assert state.turn == role
    return {"episode": episode, "brief": brief, "opener": state.opener, "role": role, "ranking": list(ranking),
            "events": events_to_json(state.events), "action": action.key}


def load_records(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---- likelihood --------------------------------------------------------------------------------------

@dataclass
class Prepared:
    role: str
    ranking: tuple
    state: PublicState
    action: Action
    episode: str


def prepare(records: Iterable[dict], cfg: LiveConfig) -> List[Prepared]:
    out = []
    for r in records:
        st = replay(cfg, r["opener"], r["events"])
        out.append(Prepared(r["role"], tuple(r["ranking"]), st, action_from_key(r["action"]), r["episode"]))
    return out


def log_likelihood(prepared: Sequence[Prepared], params: KernelParams, cfg: LiveConfig) -> Tuple[float, int]:
    """(total log-likelihood in nats, number of moves the kernel gives zero probability)."""
    kernels: Dict[str, T1Kernel] = {}
    total, zeros = 0.0, 0
    for p in prepared:
        k = kernels.get(p.role)
        if k is None:
            k = kernels[p.role] = T1Kernel(p.role, cfg, params)
        pr = k.distribution(p.state, p.ranking).get(p.action, 0.0)
        if pr <= 0.0:
            zeros += 1
            pr = FLOOR
        total += math.log(pr)
    return total, zeros


def mean_ll(prepared: Sequence[Prepared], params: KernelParams, cfg: LiveConfig) -> float:
    return log_likelihood(prepared, params, cfg)[0] / max(1, len(prepared))


# ---- fitting -----------------------------------------------------------------------------------------

def _set(params: KernelParams, name: str, x: float) -> KernelParams:
    if name == "alpha0":
        return replace(params, alpha0=Fraction(x).limit_denominator(ALPHA_DENOM))
    return replace(params, **{name: float(x)})


def _golden(f, a: float, b: float, iters: int) -> Tuple[float, float]:
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:                 # maximise
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = (a + b) / 2.0
    return x, f(x)


def fit(prepared: Sequence[Prepared], cfg: LiveConfig, start: KernelParams = REFERENCE, *, rounds: int = 3,
        iters: int = 14, params_to_fit: Sequence[str] = FIT_PARAMS) -> Tuple[KernelParams, float]:
    """Coordinate-ascent maximum likelihood from ``start``; returns (params, mean log-likelihood)."""
    bounds = dict(BOUNDS)
    bounds["alpha0"] = (float(cfg.reservation), 1.0)
    cur = start
    best = mean_ll(prepared, cur, cfg)
    for _ in range(rounds):
        improved = False
        for name in params_to_fit:
            lo, hi = bounds[name]
            x, val = _golden(lambda v: mean_ll(prepared, _set(cur, name, v), cfg), lo, hi, iters)
            if val > best + 1e-12:
                cur, best, improved = _set(cur, name, x), val, True
        if not improved:
            break
    return cur, best


def mid_start(cfg: LiveConfig, reference: KernelParams = REFERENCE) -> KernelParams:
    r = float(cfg.reservation)
    return replace(reference, alpha0=Fraction((r + 1.0) / 2.0).limit_denominator(ALPHA_DENOM), beta=4.0,
                   p_counter=0.5, p_hold=0.5, accept_noise=0.1)


def fit_best(prepared: Sequence[Prepared], cfg: LiveConfig, reference: KernelParams = REFERENCE, **kw) -> Tuple[KernelParams, float]:
    cands = [fit(prepared, cfg, reference, **kw), fit(prepared, cfg, mid_start(cfg, reference), **kw)]
    return max(cands, key=lambda c: c[1])


# ---- held-out score -----------------------------------------------------------------------------------

def split_by_episode(prepared: Sequence[Prepared], test_frac: float, seed: int) -> Tuple[List[Prepared], List[Prepared]]:
    eps = sorted({p.episode for p in prepared})
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_test = max(1, int(round(test_frac * len(eps))))
    test = set(eps[:n_test])
    return [p for p in prepared if p.episode not in test], [p for p in prepared if p.episode in test]


def params_json(p: KernelParams) -> dict:
    return p.to_json()


def heldout_report(records: Sequence[dict], cfg: LiveConfig, reference: KernelParams = REFERENCE, *, test_frac: float = 0.3,
                   seed: int = 0, folds: int = 1, **fit_kw) -> dict:
    """Fit on training episodes, score held-out episodes under the fit and under ``reference``; the gain is
    mean nats per held-out move. With ``folds`` > 1 the split is repeated with seeds seed..seed+folds-1 and
    the gains averaged. Also reports the fit on all records (the parameters a registration would pin)."""
    prepared = prepare(records, cfg)
    gains, per_fold = [], []
    for k in range(folds):
        train, test = split_by_episode(prepared, test_frac, seed + k)
        fitted, train_ll = fit_best(train, cfg, reference, **fit_kw)
        ll_fit, z_fit = log_likelihood(test, fitted, cfg)
        ll_ref, z_ref = log_likelihood(test, reference, cfg)
        gain = (ll_fit - ll_ref) / max(1, len(test))
        gains.append(gain)
        per_fold.append({"seed": seed + k, "n_train": len(train), "n_test": len(test), "fitted": params_json(fitted),
                         "train_mean_ll": train_ll, "test_mean_ll_fitted": ll_fit / max(1, len(test)),
                         "test_mean_ll_reference": ll_ref / max(1, len(test)), "zero_prob_fitted": z_fit,
                         "zero_prob_reference": z_ref, "gain_nats_per_move": gain})
    full, full_ll = fit_best(prepared, cfg, reference, **fit_kw)
    ref_ll, ref_zero = log_likelihood(prepared, reference, cfg)
    mean_gain = sum(gains) / len(gains)
    return {"n_moves": len(prepared), "n_episodes": len({p.episode for p in prepared}), "reference": params_json(reference),
            "fitted_all": params_json(full), "mean_ll_fitted_all": full_ll, "mean_ll_reference_all": ref_ll / max(1, len(prepared)),
            "zero_prob_reference_all": ref_zero, "folds": per_fold, "heldout_gain_nats_per_move": mean_gain,
            "gate_threshold": GATE_GAIN_NATS, "adopt_fit": mean_gain >= GATE_GAIN_NATS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--brief", default=None, help="keep only records of this brief")
    ap.add_argument("--regime", default="easy")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = LiveConfig(regime=args.regime, horizon=args.horizon, ask_cost=Fraction(0))
    recs = load_records(args.records)
    if args.brief:
        recs = [r for r in recs if r.get("brief") == args.brief]
    rep = heldout_report(recs, cfg, folds=args.folds, seed=args.seed)
    rep["brief"] = args.brief
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"{len(recs)} moves, {rep['n_episodes']} episodes; fitted {rep['fitted_all']}; held-out gain "
          f"{rep['heldout_gain_nats_per_move']:+.4f} nats/move (gate {GATE_GAIN_NATS}): adopt_fit={rep['adopt_fit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
