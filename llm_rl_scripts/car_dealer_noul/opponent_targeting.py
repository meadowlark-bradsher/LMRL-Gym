"""Does the live opponent target the focal side? An offline re-analysis of the CD-NOUL-2 pilot (no API).

The T1 kernel's proposals depend on the opponent's own ranking and the clock only; it never updates on
what the focal side reveals. If the live `claude-sonnet-5-5` opponent instead proposes deals fitted to the
*focal side's* ranking, it is playing a two-way game the planner's kernel cannot express (one candidate
reason for gate 4's support misfit, DESIGN_NOUL2 §3.3a).

Targeting statistic, for a deal d offered to a focal side of role f with true ranking θ:

    spec(d) = U_f(θ, d) − mean over the five other rankings θ' of U_f(θ', d)

How much better d is for the focal side's actual ranking than for the rankings it might have had.
Σ_θ spec(d; θ) = 0 for every d, so a proposal rule that does not depend on θ has mean spec 0 over a design
balanced in θ, which the pilot is (6 focal × 6 opponent rankings per role and opener). Every opponent
PROPOSE in the pilot is scored, together with the expected spec of the registered T1 kernel's proposal
distribution in the same public state with the same opponent ranking (the counterfactual "what the
planner's model would have proposed here"). Information about the focal side reaches the opponent only
through the focal side's moves (the realizer phrases the action and the public history only) and its
templated truthful answers to the opponent's questions, so the statistic is split by how many focal moves
the opponent had seen. The opponent's opening proposal (no focal move yet) is the placebo: no rule can
target there, so its mean spec must be about 0.

A heuristic text count of the opponent's stated reasoning (the reply before `DECISION:`) is reported
beside it: the share of moves whose reasoning names the counterpart, and the share that attributes a
preference to it. It is descriptive only.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.opponent_targeting run \
        --pilot results/noul2_pilot_20261108 --out results/noul2_targeting_20261109/targeting.json
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.opponent_targeting render \
        results/noul2_targeting_20261109/targeting.json --note llm_rl_scripts/car_dealer_noul/TARGETING_NOUL2.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, DEALS_BY_ID, utility
from llm_rl_scripts.car_dealer_live.analyze_phase1 import paired_bootstrap
from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.protocol_live import PROPOSE, LiveConfig, other

from .kernel_fit import load_records, replay

BRIEFS = ("candid", "guarded", "adversarial")
BINS = ("0", "1", "2", "3+")          # focal moves the opponent had seen before proposing
B, SEED = 5000, 0
N_EXAMPLES = 3


def spec(role: str, ranking, deal) -> Fraction:
    """How much better ``deal`` is for ``ranking`` than for the other five rankings of the same role."""
    u = utility(role, tuple(ranking), deal)
    others = [utility(role, t, deal) for t in ALL_RANKINGS if t != tuple(ranking)]
    return u - sum(others, Fraction(0)) / len(others)


def parse_episode(key: str) -> dict:
    arm, brief, focal, opener, fi, oi, rep = key.split("|")
    return {"arm": arm, "brief": brief, "focal": focal, "opener": opener,
            "focal_ranking": ALL_RANKINGS[int(fi)], "opp_ranking": ALL_RANKINGS[int(oi)], "repeat": int(rep)}


def cluster(focal: str, focal_ranking, opp_ranking) -> Tuple[str, str]:
    b, s = (focal_ranking, opp_ranking) if focal == "buyer" else (opp_ranking, focal_ranking)
    return ("/".join(b), "/".join(s))


def bin_of(n: int) -> str:
    return "3+" if n >= 3 else str(n)


def score_record(rec: dict, cfg: LiveConfig, kernels: Dict[str, T1Kernel]) -> Optional[dict]:
    """One opponent PROPOSE → its spec, focal / own utility, and the T1 counterfactual in the same state."""
    if not rec["action"].startswith(PROPOSE):
        return None
    ep = parse_episode(rec["episode"])
    role, f = rec["role"], ep["focal"]
    assert role == other(f) and tuple(rec["ranking"]) == ep["opp_ranking"], rec["episode"]
    fr, orr = ep["focal_ranking"], ep["opp_ranking"]
    deal = DEALS_BY_ID[rec["action"].split(" ", 1)[1]]
    state = replay(cfg, rec["opener"], rec["events"])
    k = kernels.setdefault(role, T1Kernel(role, cfg, REFERENCE))
    dist = k.proposal_distribution(state, orr)
    t1 = {"spec": 0.0, "u_focal": 0.0, "u_opp": 0.0}
    for act, p in dist.items():
        d = DEALS_BY_ID[act.deal_id]
        t1["spec"] += p * float(spec(f, fr, d))
        t1["u_focal"] += p * float(utility(f, fr, d))
        t1["u_opp"] += p * float(utility(role, orr, d))
    focal_moves = sum(1 for e in state.events if e.actor == f)
    got_answer = any(e.actor == role and e.action.kind == "ASK" for e in state.events)
    out = {"brief": rec["brief"], "episode": rec["episode"], "cluster": cluster(f, fr, orr),
           "bin": bin_of(focal_moves), "got_answer": got_answer,
           "spec": float(spec(f, fr, deal)), "u_focal": float(utility(f, fr, deal)),
           "u_opp": float(utility(role, orr, deal)),
           "t1_spec": t1["spec"], "t1_u_focal": t1["u_focal"], "t1_u_opp": t1["u_opp"]}
    last = [e.action.deal_id for e in state.events if e.actor == f and e.action.kind == PROPOSE]
    if last:
        focal_last = DEALS_BY_ID[last[-1]]
        mid = midpoint(focal_last, ideal(role, orr))
        out["controls"] = {"anchor_spec": float(spec(f, fr, mid)),
                           "anchor_match": float(deal == mid),
                           "anchor_l1": float(sum(abs(deal.level(a) - mid.level(a)) for a in ("price", "warranty", "delivery"))),
                           "echo_spec": float(spec(f, fr, focal_last)),
                           "best_same_concession_spec": float(spec(f, fr, best_at_concession(f, fr, role, orr,
                                                                                             utility(role, orr, deal))))}
    return out


def ideal(role: str, ranking):
    return max(DEALS_BY_ID.values(), key=lambda d: (utility(role, ranking, d), d.id))


def midpoint(a, b):
    """Attribute-wise midpoint of deals a (the focal side's last offer) and b (the opponent's ideal), a half
    level rounded toward b: 'meet in the middle', a reactive rule with no model of the focal side."""
    lv = {}
    for attr in ("price", "warranty", "delivery"):
        x, y = a.level(attr), b.level(attr)
        lv[attr] = (x + y) // 2 if y <= x else (x + y + 1) // 2
    return next(d for d in DEALS_BY_ID.values() if all(d.level(k) == v for k, v in lv.items()))


def best_at_concession(f: str, fr, role: str, orr, floor: Fraction):
    """The deal best for the focal side's true ranking among those worth >= ``floor`` to the opponent:
    perfect targeting at the live proposal's concession."""
    pool = [d for d in DEALS_BY_ID.values() if utility(role, orr, d) >= floor]
    return max(pool, key=lambda d: (utility(f, fr, d), spec(f, fr, d), d.id))


def _cluster_means(rows: List[dict], key: str) -> Dict[tuple, float]:
    acc = defaultdict(list)
    for r in rows:
        acc[r["cluster"]].append(r[key])
    return {c: sum(v) / len(v) for c, v in acc.items()}


def _mean_ci(rows: List[dict], key: str) -> dict:
    a = _cluster_means(rows, key)
    out = paired_bootstrap(a, {c: 0.0 for c in a}, B, SEED)
    out["n_moves"] = len(rows)
    return out


def _contrast(rows: List[dict], key_a: str, key_b: str) -> dict:
    out = paired_bootstrap(_cluster_means(rows, key_a), _cluster_means(rows, key_b), B, SEED)
    out["n_moves"] = len(rows)
    return out


def summarize(rows: List[dict]) -> dict:
    out = {"spec_llm": _mean_ci(rows, "spec"), "spec_t1": _mean_ci(rows, "t1_spec"),
           "spec_llm_minus_t1": _contrast(rows, "spec", "t1_spec"),
           "u_focal_llm_minus_t1": _contrast(rows, "u_focal", "t1_u_focal"),
           "u_opp_llm_minus_t1": _contrast(rows, "u_opp", "t1_u_opp"),
           "mean_u_focal_llm": sum(r["u_focal"] for r in rows) / len(rows),
           "mean_u_focal_t1": sum(r["t1_u_focal"] for r in rows) / len(rows),
           "mean_u_opp_llm": sum(r["u_opp"] for r in rows) / len(rows),
           "mean_u_opp_t1": sum(r["t1_u_opp"] for r in rows) / len(rows),
           "by_bin": {}, "by_answer": {}}
    for b in BINS:
        sub = [r for r in rows if r["bin"] == b]
        if sub:
            out["by_bin"][b] = {"spec_llm": _mean_ci(sub, "spec"), "spec_t1": _mean_ci(sub, "t1_spec"),
                                "spec_llm_minus_t1": _contrast(sub, "spec", "t1_spec")}
    ctl = [dict(r, **r["controls"]) for r in rows if "controls" in r]
    if ctl:
        out["controls"] = {"n_moves": len(ctl), "spec_llm": _mean_ci(ctl, "spec"), "spec_t1": _mean_ci(ctl, "t1_spec"),
                           "anchor_spec": _mean_ci(ctl, "anchor_spec"), "echo_spec": _mean_ci(ctl, "echo_spec"),
                           "best_same_concession_spec": _mean_ci(ctl, "best_same_concession_spec"),
                           "spec_llm_minus_anchor": _contrast(ctl, "spec", "anchor_spec"),
                           "anchor_match_share": sum(r["anchor_match"] for r in ctl) / len(ctl),
                           "anchor_within1_share": sum(1 for r in ctl if r["anchor_l1"] <= 1) / len(ctl),
                           "anchor_mean_l1": sum(r["anchor_l1"] for r in ctl) / len(ctl)}
    for flag in (False, True):
        sub = [r for r in rows if r["got_answer"] == flag and r["bin"] != "0"]
        if sub:
            out["by_answer"][str(flag).lower()] = {"n_moves": len(sub), "n_episodes": len({r["episode"] for r in sub}),
                                                   "mean_spec_llm": sum(r["spec"] for r in sub) / len(sub),
                                                   "mean_spec_t1": sum(r["t1_spec"] for r in sub) / len(sub)}
    return out


# -- stated reasoning (heuristic, descriptive) --------------------------------------------------------

PREF = re.compile(r"\b(want|wants|care|cares|prefer|prefers|priorit\w*|value|values|matters?|sticking point|"
                  r"shown|willing|asks?|asked for|likes?)\b", re.I)


def reasoning_of(reply: str) -> str:
    return reply.split("DECISION:", 1)[0]


def text_scan(runs: List[dict], seed: int = SEED) -> dict:
    per = defaultdict(lambda: {"moves": 0, "names_counterpart": 0, "attributes_preference": 0, "examples": []})
    pools = defaultdict(list)
    for r in runs:
        counterpart = r["focal"]                      # the opponent's counterpart is the focal side
        word = re.compile(rf"\b(the {counterpart}|{counterpart}'s|they|their|them)\b", re.I)
        for o, raw in zip(r["opponent"], r["opponent_raw"]):
            if o.get("kind") != "move":
                continue
            text = reasoning_of(raw.get("reply", ""))
            p = per[r["brief"]]
            p["moves"] += 1
            sents = re.split(r"(?<=[.!?])\s+", text)
            named = [s for s in sents if word.search(s)]
            pref = [s for s in named if PREF.search(s)]
            p["names_counterpart"] += bool(named)
            p["attributes_preference"] += bool(pref)
            pools[r["brief"]] += [s.strip() for s in pref]
    rng = random.Random(seed)
    for b, p in per.items():
        pool = sorted(set(pools[b]))
        p["examples"] = rng.sample(pool, min(N_EXAMPLES, len(pool)))
        p["share_names_counterpart"] = p["names_counterpart"] / p["moves"]
        p["share_attributes_preference"] = p["attributes_preference"] / p["moves"]
    return dict(per)


# -- run / render ---------------------------------------------------------------------------------------

def load_cfg(pilot: str) -> LiveConfig:
    with open(os.path.join(pilot, "summary.json")) as f:
        c = json.load(f)["config"]
    return LiveConfig(regime=c["regime"], horizon=int(c["horizon"]), ask_cost=Fraction(c["ask_cost"]),
                      answer_noise=Fraction(c["answer_noise"]))


def analyze(pilot: str) -> dict:
    cfg = load_cfg(pilot)
    kernels: Dict[str, T1Kernel] = {}
    rows = [s for s in (score_record(r, cfg, kernels) for r in load_records(os.path.join(pilot, "opponent_moves.jsonl")))
            if s is not None]
    with open(os.path.join(pilot, "runs.jsonl")) as f:
        runs = [json.loads(l) for l in f if l.strip()]
    for r in runs:                                     # the key's rankings agree with the evaluator's record
        ep = parse_episode(r["episode_key"])
        assert tuple(r["focal_ranking"]) == ep["focal_ranking"]
        assert tuple(r["private_truth_evaluator_only"]["opp_ranking"]) == ep["opp_ranking"]
    out = {"pilot": pilot, "config": cfg.to_json(), "bootstrap": {"B": B, "seed": SEED,
           "clusters": "(buyer ranking, seller ranking) type pair"},
           "n_proposals": len(rows), "per_brief": {}, "pooled": summarize(rows), "reasoning": text_scan(runs)}
    for b in BRIEFS:
        sub = [r for r in rows if r["brief"] == b]
        if sub:
            out["per_brief"][b] = summarize(sub)
    return out


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_cmd(args) -> int:
    doc = analyze(args.pilot)
    doc["commit"] = _commit()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out} ({doc['n_proposals']} proposals)")
    return 0


def _ci(x: dict) -> str:
    return f"{x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def render(doc: dict, path: str) -> str:
    L = ["# Does the live opponent target the focal side? (CD-NOUL-2 pilot, offline)", "",
         f"_Rendered by `opponent_targeting.py render` from `{path}` (pilot `{doc['pilot']}`, commit `{doc['commit']}`, "
         f"{doc['n_proposals']} opponent proposals). No API calls; every number is formatted from the JSON. "
         f"Cluster bootstrap B = {doc['bootstrap']['B']} (seed {doc['bootstrap']['seed']}) over the "
         f"{doc['bootstrap']['clusters']}s. Exploratory: not one of the §6 gates._", "",
         "`spec(d)` = the focal side's utility of the proposed deal under its true ranking, minus its mean "
         "utility under the five other rankings. It averages 0 for any proposal rule that ignores the focal "
         "side's ranking. `T1` is the expected value under the registered T1 kernel's proposal distribution in "
         "the same public state with the same opponent ranking. Utilities are in [0, 1] (not Y).", "",
         "## Targeting, per brief", "",
         "| brief | proposals | spec, live opponent | spec, T1 in same state | live − T1 | focal utility, live − T1 | own utility, live − T1 |",
         "|---|---|---|---|---|---|---|"]
    blocks = [(b, doc["per_brief"][b]) for b in BRIEFS if b in doc["per_brief"]] + [("pooled", doc["pooled"])]
    for name, s in blocks:
        L.append(f"| {name} | {s['spec_llm']['n_moves']} | {_ci(s['spec_llm'])} | {_ci(s['spec_t1'])} | "
                 f"{_ci(s['spec_llm_minus_t1'])} | {_ci(s['u_focal_llm_minus_t1'])} | {_ci(s['u_opp_llm_minus_t1'])} |")
    L += ["", "## By how many focal moves the opponent had seen (pooled briefs)", "",
          "Bin 0 is the opponent's opening proposal, before any focal move: the placebo.", "",
          "| focal moves seen | proposals | spec, live opponent | spec, T1 | live − T1 |", "|---|---|---|---|---|"]
    for b, s in doc["pooled"]["by_bin"].items():
        L.append(f"| {b} | {s['spec_llm']['n_moves']} | {_ci(s['spec_llm'])} | {_ci(s['spec_t1'])} | {_ci(s['spec_llm_minus_t1'])} |")
    c = doc["pooled"].get("controls")
    if c:
        L += ["", "## Controls: is it inference or reaction? (pooled; proposals after at least one focal offer)", "",
              "`meet in the middle` = the attribute-wise midpoint of the focal side's last offer and the opponent's "
              "ideal deal (half levels toward the opponent), a reactive rule with no model of the focal side. "
              "`echo` = the focal side's own last offer. `best at same concession` = the deal best for the focal "
              "side's true ranking among those worth at least as much to the opponent as the live proposal "
              "(perfect targeting at the same cost to the opponent).", "",
              "| rule | spec |", "|---|---|",
              f"| live opponent | {_ci(c['spec_llm'])} |", f"| T1 in same state | {_ci(c['spec_t1'])} |",
              f"| meet in the middle | {_ci(c['anchor_spec'])} |", f"| echo the focal offer | {_ci(c['echo_spec'])} |",
              f"| best at same concession | {_ci(c['best_same_concession_spec'])} |",
              f"| live − meet in the middle | {_ci(c['spec_llm_minus_anchor'])} |",
              "", f"{c['n_moves']} proposals. The live proposal equals the meet-in-the-middle deal in "
              f"{c['anchor_match_share']:.3f} of them and is within one attribute level of it in "
              f"{c['anchor_within1_share']:.3f} (mean distance {c['anchor_mean_l1']:.2f} levels; 27 deals, distance 0–6)."]
    L += ["", "## After the opponent heard an answer (pooled, bins ≥ 1; descriptive means, no CI)", "",
          "| opponent had asked and been answered | proposals | episodes | mean spec, live | mean spec, T1 |", "|---|---|---|---|---|"]
    for flag, s in doc["pooled"]["by_answer"].items():
        L.append(f"| {flag} | {s['n_moves']} | {s['n_episodes']} | {s['mean_spec_llm']:+.4f} | {s['mean_spec_t1']:+.4f} |")
    L += ["", "## Stated reasoning (heuristic text count, descriptive)", "",
          "Share of the opponent's moves whose reasoning (the reply before `DECISION:`) names the counterpart, and "
          "the share with a sentence that names it together with a preference word (want, care, prefer, priority, "
          "value, matter, sticking point, shown, willing, ask, like).", "",
          "| brief | moves | names counterpart | attributes a preference | example sentences |", "|---|---|---|---|---|"]
    for b in BRIEFS:
        s = doc["reasoning"].get(b)
        if s:
            ex = " · ".join(f"“{e}”" for e in s["examples"]).replace("|", "/")
            L.append(f"| {b} | {s['moves']} | {s['share_names_counterpart']:.3f} | {s['share_attributes_preference']:.3f} | {ex} |")
    return "\n".join(L).rstrip() + "\n"


def render_cmd(args) -> int:
    with open(args.json) as f:
        doc = json.load(f)
    with open(args.note, "w") as f:
        f.write(render(doc, args.json))
    print(f"wrote {args.note}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run_cmd)
    n = sub.add_parser("render")
    n.add_argument("json")
    n.add_argument("--note", required=True)
    n.set_defaults(func=render_cmd)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
