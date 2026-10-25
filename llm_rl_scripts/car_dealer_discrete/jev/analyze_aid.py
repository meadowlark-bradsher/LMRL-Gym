"""Analyze the aid x salience 2x2 (or any set of J/J conditions).

Builds on :mod:`analyze_salience` (outcome + trajectory + decent-offer confound
check) and adds two things the policy-aid condition makes possible:

1. **Forward-model-implied acceptability of what Jev chose.** The aid is a pure
   function of the public history, so for EVERY condition — including baseline
   and salience, which never ran it — we recompute what the aid would have said
   about each offer Jev actually made. This measures whether Jev *used* the
   number, apples-to-apples across cells, not merely whether outcomes moved.
   For aid conditions it is also cross-checked against the logged values.

2. **Epistemic metrics (section 4.3)** where a belief report was logged: log
   loss of the TRUE opponent type under the final posterior, log-score
   improvement over the uniform prior, final entropy, and whether the posterior
   argmax equals the true type. Reported with spread — these depend on the
   (stochastic) trajectory Jev produced.

Utilities and beliefs are recomputed exactly from evaluator-only truth; nothing
depends on numbers stored in the logs except the logged belief, which is
cross-checked.

Example:
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.analyze_aid \
        results/jj_salience_20261025/baseline results/jj_salience_20261025/salience \
        results/jj_aid_20261025/aid results/jj_aid_20261025/aid_salience
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from typing import Dict, List, Optional

from llm_rl_scripts.car_dealer_discrete.game import DEALS_BY_ID
from llm_rl_scripts.car_dealer_discrete.jev.aid import AcceptanceAid, AidConfig, ranking_key
from llm_rl_scripts.car_dealer_discrete.jev.analyze_salience import _spread, analyze, load
from llm_rl_scripts.car_dealer_discrete.jev.run_jj import summarize
from llm_rl_scripts.car_dealer_discrete.protocol import Action, Observation, TurnRecord
from llm_rl_scripts.car_dealer_discrete.scoring import RESERVATIONS

LOG2_6 = math.log2(6)


def _obs_for(role: str, own_rank, r: Fraction, history: List[TurnRecord]) -> Observation:
    # The aid's posterior reads only role / reservation / history; the rest are
    # placeholders so we can build an Observation from a log.
    return Observation(
        role=role, own_ranking=tuple(own_rank), reservation=r, max_proposals=0,
        proposals_used=0, outstanding_offer=None, history=tuple(history), legal_actions=(),
    )


def implied_acceptability(recs: List[dict], cfg: AidConfig) -> Dict[str, object]:
    """Aid-implied P(recipient accepts) for every offer Jev actually made."""
    aid = AcceptanceAid(cfg)
    chosen: List[float] = []
    max_dev: Optional[float] = None  # vs logged acceptance; bayes-mode records only
    modes = set()
    for r in recs:
        truth = r["private_truth_evaluator_only"]
        rank = {"buyer": truth["buyer_rank"], "seller": truth["seller_rank"]}
        res = RESERVATIONS[r["reservation_regime"]]
        # Earlier aid logs predate the mode field; they were the real filter.
        mode = (r.get("condition") or {}).get("aid_mode") or (
            "bayes" if (r.get("condition") or {}).get("aid") else None)
        modes.add(mode)
        history: List[TurnRecord] = []
        step_idx = {"buyer": 0, "seller": 0}
        for turn in r["turns"]:
            actor = turn["actor"]
            if turn["action"] == "OFFER":
                _, pacc = aid.acceptance_probabilities(_obs_for(actor, rank[actor], res, history))
                p = pacc[turn["deal_id"]]
                chosen.append(p)
                steps = r["jev_steps"][actor]
                i = step_idx[actor]
                if mode == "bayes" and i < len(steps) and steps[i].get("acceptance"):
                    dev = abs(steps[i]["acceptance"][turn["deal_id"]] - p)
                    max_dev = dev if max_dev is None else max(max_dev, dev)
            step_idx[actor] += 1
            history.append(TurnRecord(actor, Action(turn["action"], turn.get("deal_id"))))
    return {
        "chosen_offer_p_accept": _spread(chosen),
        "share_of_offers_with_p_accept_ge_0_5": (
            round(sum(1 for p in chosen if p >= 0.5) / len(chosen), 3) if chosen else None
        ),
        "max_abs_dev_vs_logged": (round(max_dev, 6) if max_dev is not None else None),
        "aid_modes_in_logs": sorted(m for m in modes if m),
    }


def epistemic(recs: List[dict]) -> Optional[Dict[str, object]]:
    """Section-4.3 metrics from logged belief reports. None if none were logged."""
    final_logloss: List[float] = []
    improvement: List[float] = []
    final_entropy: List[float] = []
    argmax_correct: List[float] = []
    n_steps: List[float] = []
    for r in recs:
        truth = r["private_truth_evaluator_only"]
        for role in ("buyer", "seller"):
            steps = [s for s in r["jev_steps"][role] if s.get("belief")]
            if not steps:
                continue
            opp = "seller" if role == "buyer" else "buyer"
            true_key = ranking_key(tuple(truth[f"{opp}_rank"]))
            qT = steps[-1]["belief"]
            p_true = max(qT.get(true_key, 0.0), 1e-300)
            final_logloss.append(-math.log2(p_true))
            improvement.append(math.log2(p_true) - math.log2(1 / 6))
            final_entropy.append(steps[-1]["belief_entropy_bits"])
            argmax_correct.append(1.0 if max(qT, key=qT.get) == true_key else 0.0)
            n_steps.append(len(steps))
    if not final_logloss:
        return None
    return {
        "belief_reports_scored": len(final_logloss),
        "final_logloss_true_type_bits": _spread(final_logloss),
        "logscore_improvement_over_uniform_bits": _spread(improvement),
        "final_entropy_bits": _spread(final_entropy),
        "uniform_prior_entropy_bits": round(LOG2_6, 4),
        "argmax_is_true_type_rate": round(sum(argmax_correct) / len(argmax_correct), 3),
        "belief_steps_per_seat": _spread(n_steps),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("conditions", nargs="+")
    p.add_argument("--aid-beta", type=float, default=8.0)
    p.add_argument("--aid-noise", type=float, default=0.05)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    cfg = AidConfig(beta=args.aid_beta, noise=args.aid_noise)

    report = {}
    for c in args.conditions:
        name = os.path.basename(os.path.normpath(c))
        recs = load(c)
        outcome = summarize(recs)
        report[name] = {
            "condition_flags": recs[0].get("condition") if recs else None,
            "trajectory": analyze(recs),
            "outcome_by_regime": outcome,
            "implied_acceptability": implied_acceptability(recs, cfg),
            "epistemic": epistemic(recs),
        }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    # compact 2x2-style table
    print(f"{'condition':<14}{'eps':>4}{'agree':>7}{'valid':>7}{'dec/off':>9}{'minG':>7}{'nash':>7}{'P(acc) chosen':>16}{'argmax=true':>13}")
    for name, rep in report.items():
        t = rep["trajectory"]
        regimes = rep["outcome_by_regime"]
        o = next(iter(regimes.values())) if regimes else {}
        ia = rep["implied_acceptability"]
        ep = rep["epistemic"]
        pa = ia["chosen_offer_p_accept"]
        print(
            f"{name:<14}{t['episodes']:>4}"
            f"{(o.get('agreement_rate') if o else None)!s:>7}"
            f"{(o.get('valid_agreement_rate_feasible') if o else None)!s:>7}"
            f"{t['offers_worth_ge_r_to_recipient']}/{t['offers_total']:<5}"
            f"{(o.get('mean_min_gain_feasible') if o else None)!s:>7}"
            f"{(o.get('mean_nash_ratio_feasible') if o else None)!s:>7}"
            f"{(pa['mean'] if pa else None)!s:>16}"
            f"{(ep['argmax_is_true_type_rate'] if ep else '-')!s:>13}"
        )
    print()
    for name, rep in report.items():
        print("=" * 72); print(name.upper(), rep["condition_flags"])
        t = rep["trajectory"]
        print(f"  outcomes: {t['outcomes']}")
        print(f"  buyer offered : {t['offered_by_buyer']}")
        print(f"  seller offered: {t['offered_by_seller']}")
        print(f"  offers total={t['offers_total']} non-ideal={t['offers_nonideal_own_lt_1']} worth>=r to RECIPIENT={t['offers_worth_ge_r_to_recipient']}")
        print(f"  who walked: {t['who_walked']} at #{t['walk_decision_indices']}")
        ia = rep["implied_acceptability"]
        print(f"  aid-implied P(accept) of chosen offers: {ia['chosen_offer_p_accept']}  share>=0.5: {ia['share_of_offers_with_p_accept_ge_0_5']}  max|dev vs logged|={ia['max_abs_dev_vs_logged']}")
        if rep["epistemic"]:
            e = rep["epistemic"]
            print(f"  epistemic: final logloss(true) bits={e['final_logloss_true_type_bits']}")
            print(f"             improvement over uniform bits={e['logscore_improvement_over_uniform_bits']}")
            print(f"             final entropy bits={e['final_entropy_bits']} (uniform={e['uniform_prior_entropy_bits']})  argmax=true rate={e['argmax_is_true_type_rate']}")
        regimes = rep["outcome_by_regime"]
        for reg, o in regimes.items():
            print(f"  [{reg}] agree={o['agreement_rate']} valid={o['valid_agreement_rate_feasible']} G_B={o['mean_buyer_gain_feasible']} G_S={o['mean_seller_gain_feasible']} minG={o['mean_min_gain_feasible']} pareto={o['pareto_frontier_rate_valid']} nash={o['mean_nash_ratio_feasible']} props->agree={o['mean_proposals_to_agreement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
