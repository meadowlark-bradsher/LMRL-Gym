"""CD-NOUL-2 Phase 0 gates computed from the pilot (``DESIGN_NOUL2.md`` §6): gate 1 (opponent protocol),
gate 4 (kernel fit per brief), gate 5 (realizer fidelity, judged blind) and gate 6 (clairvoyant gap on
``guarded``, focal opens). Gates 2, 3 and 9 read opponent *answers*, which the pilot arms never elicit
(``plan_noask`` / ``plan_reveal`` never ask); they need the answer probe and are reported as pending here.
Nothing below reads an outcome contrast other than the one gate 6 names.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 protocol        --dir results/<pilot>
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 kernel          --dir results/<pilot> [--folds 3 --seed 0]
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 realizer-packet --dir results/<pilot> [--n 200]
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 realizer-judge  --dir results/<pilot> --backend fake|anthropic
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 realizer-report --dir results/<pilot>
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 clairvoyant     --dir results/<pilot> [--B 5000 --seed 0]
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.gates_noul2 render          --dir results/<pilot> --out llm_rl_scripts/car_dealer_noul/GATES_NOUL2.md

Every step writes under ``<dir>/gates/``; ``render`` reads only those files (no number in the note is typed).

The realizer audit mirrors Stage 1's ``audit.py``: ``realizer-packet`` replays every pilot episode from the
focal's logged decisions and the opponent's logged moves (checked against the opponent-move records),
draws ``N_PER_ROLE`` realized moves per role, and writes a packet holding the realized words and the lettered
legal moves of that state — never the chosen move — plus a sealed key. ``realizer-judge`` labels the packet
(a Sonnet judge, or ``fake``) and ``realizer-report`` joins labels to the key: recovery = the judge's label
is the chosen move; an added commitment = the judge quotes words beyond the move.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from llm_rl_scripts.car_dealer_live.analyze_phase1 import cluster_of, in_set, paired_bootstrap, y
from llm_rl_scripts.car_dealer_live.protocol_live import ASK, LiveConfig, apply, legal_actions, new_game, other, truthful_answer

from . import kernel_fit as kf
from .opponent_llm import MODEL, lettered, moves_text, parse_reply

# ---- registered thresholds (DESIGN_NOUL2 §5–6) --------------------------------------------------------
WELL_FORMED_MIN = 0.99          # gate 1
DEFAULTED_MAX = 0.01            # gate 1
RECOVERY_MIN = 0.99             # gate 5
ADDED_MAX = 0                   # gate 5
N_PER_ROLE = 200                # gate 5
DELTA = 0.02                    # gate 6 (δ, §5)
PACKET_SEED = 20261108
BRIEFS = ("candid", "guarded", "adversarial")
JUDGE_MAX_TOKENS = 300

JUDGE_SYSTEM = """You read one line said by the {role} in a negotiation over a used car with a {other}. The deal will set the price, the warranty and the delivery time. You are shown the moves the {role} could make at that point, each with a label. Decide which ONE move the line makes.

Then say whether the line adds a commitment beyond that move: it names or offers a second deal, or it offers or promises a concession on price, warranty or delivery. A plain question to the other side ("how does that sound?"), politeness, or a reason for the move is not a commitment.

Reply with exactly two lines
DECISION: <label>
ADDED: none
and if the line does add a commitment, replace "none" with the words that add it."""

_ADDED_RE = re.compile(r"^\s*ADDED\s*:\s*(.*)$", re.M)


# ---- loading ---------------------------------------------------------------------------------------------

def load_rows(d: Path) -> List[dict]:
    rows, stubs = [], 0
    with (d / "runs.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                stubs += 1
                continue
            rows.append(r)
    return rows


def config_of(d: Path) -> LiveConfig:
    c = json.load((d / "summary.json").open())["config"]
    return LiveConfig(regime=c["regime"], horizon=int(c["horizon"]), ask_cost=Fraction(c["ask_cost"]),
                      answer_noise=Fraction(c["answer_noise"]))


def gates_dir(d: Path) -> Path:
    g = d / "gates"
    g.mkdir(exist_ok=True)
    return g


def _dump(obj, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


# ---- gate 1: opponent protocol -------------------------------------------------------------------------

def protocol(rows: List[dict]) -> dict:
    out = {"threshold": {"well_formed_min": WELL_FORMED_MIN, "defaulted_max": DEFAULTED_MAX}, "briefs": {}, "episodes": len(rows)}
    for b in BRIEFS:
        sub = [r for r in rows if r["brief"] == b]
        c: Counter = Counter({k: 0 for k in ("episodes", "opponent_asked_episodes", "calls", "kind_move", "kind_answer", "well_formed",
                                           "retried", "defaulted", "stop_end_turn", "realizer_calls", "realizer_ok",
                                           "realizer_fallback", "realizer_retried")})
        tok_in = tok_out = 0
        for r in sub:
            c["episodes"] += 1
            c["opponent_asked_episodes"] += bool(r.get("opponent_asked"))
            for o in r["opponent"]:
                c["calls"] += 1
                c["kind_" + o["kind"]] += 1
                c["well_formed"] += bool(o["well_formed"])
                c["retried"] += bool(o["retried"])
                c["defaulted"] += bool(o["defaulted"])
            for raw in r.get("opponent_raw", []):
                c["stop_" + str(raw.get("stop_reason"))] += 1
                tok_in += raw.get("input_tokens", 0)
                tok_out += raw.get("output_tokens", 0)
            for rc in r["realizer_calls"]:
                c["realizer_calls"] += 1
                c["realizer_ok"] += bool(rc["ok"])
                c["realizer_fallback"] += bool(rc["fallback"])
                c["realizer_retried"] += bool(rc["retried"])
        n = max(1, c["calls"])
        wf, df = c["well_formed"] / n, c["defaulted"] / n
        out["briefs"][b] = {**{k: v for k, v in c.items()}, "well_formed_share": wf, "defaulted_share": df,
                            "retried_share": c["retried"] / n, "input_tokens": tok_in, "output_tokens": tok_out,
                            "passes": wf >= WELL_FORMED_MIN and df <= DEFAULTED_MAX}
    out["passes"] = all(v["passes"] for v in out["briefs"].values())
    return out


# ---- gate 4: kernel fit per brief ------------------------------------------------------------------------

def support_breakdown(prepared, params, cfg: LiveConfig) -> Dict[str, dict]:
    """Per kind of the logged opponent move: how many the kernel ``params`` gives probability 0 (outside its support).
    Not a fit quantity — it says how much of the live opponent's behaviour the T1 family can express at all."""
    from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
    kernels: Dict[str, T1Kernel] = {}
    out: Dict[str, Counter] = defaultdict(Counter)
    for p in prepared:
        k = kernels.get(p.role)
        if k is None:
            k = kernels[p.role] = T1Kernel(p.role, cfg, params)
        pr = k.distribution(p.state, p.ranking).get(p.action, 0.0)
        out[p.action.kind]["n"] += 1
        out[p.action.kind]["zero"] += pr <= 0.0
    return {kind: dict(v) for kind, v in sorted(out.items())}


def kernel(d: Path, cfg: LiveConfig, folds: int, seed: int) -> dict:
    records = kf.load_records(str(d / "opponent_moves.jsonl"))
    out = {"threshold_gain_nats_per_move": kf.GATE_GAIN_NATS, "briefs": {}}
    for b in BRIEFS:
        sub = [r for r in records if r["brief"] == b]
        if not sub:
            out["briefs"][b] = {"n_moves": 0}
            continue
        rep = kf.heldout_report(sub, cfg, folds=folds, seed=seed)
        rep["brief"] = b
        prepared = kf.prepare(sub, cfg)
        rep["support_reference"] = support_breakdown(prepared, kf.REFERENCE, cfg)
        rep["support_fitted"] = support_breakdown(prepared, kf.KernelParams.from_json(rep["fitted_all"]) if hasattr(kf.KernelParams, "from_json") else _params_from_json(rep["fitted_all"]), cfg)
        _dump(rep, gates_dir(d) / f"kernel_fit_{b}.json")
        out["briefs"][b] = {"n_moves": rep["n_moves"], "n_episodes": rep["n_episodes"],
                            "heldout_gain_nats_per_move": rep["heldout_gain_nats_per_move"], "adopt_fit": rep["adopt_fit"],
                            "fitted_all": rep["fitted_all"], "reference": rep["reference"],
                            "mean_ll_fitted_all": rep["mean_ll_fitted_all"], "mean_ll_reference_all": rep["mean_ll_reference_all"],
                            "zero_prob_reference_all": rep["zero_prob_reference_all"], "folds": folds, "seed": seed,
                            "support_reference": rep["support_reference"], "support_fitted": rep["support_fitted"]}
    return out


def split_breakdown(prepared, params, cfg: LiveConfig) -> dict:
    """Of the logged moves ``params`` gives zero probability at the opponent's TRUE ranking: how many are zero
    under every ranking (no type evidence; the eta = 0 filter keeps its belief) and how many are positive under
    some other ranking (a *split*: the eta = 0 filter sets the true ranking's belief to exactly 0 for the rest of
    the episode). Counted per action kind, with the number of episodes holding at least one split."""
    from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
    from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
    kernels: Dict[str, T1Kernel] = {}
    c: Counter = Counter()
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    hit, eps = set(), set()
    for p in prepared:
        eps.add(p.episode)
        k = kernels.get(p.role) or kernels.setdefault(p.role, T1Kernel(p.role, cfg, params))
        vec = [k.distribution(p.state, th).get(p.action, 0.0) for th in ALL_RANKINGS]
        c["n_moves"] += 1
        if vec[ALL_RANKINGS.index(p.ranking)] > 0:
            continue
        c["zero_at_true"] += 1
        kind = "split" if any(v > 0 for v in vec) else "zero_all"
        c[kind] += 1
        by_kind[p.action.kind][kind] += 1
        if kind == "split":
            hit.add(p.episode)
    for key in ("zero_at_true", "zero_all", "split"):
        c[key] += 0
    return {**dict(c), "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
            "episodes_with_split": len(hit), "n_episodes": len(eps)}


def ll_on_t1_support(prepared, params, cfg: LiveConfig) -> dict:
    """Mean log-likelihood of ``params`` over only the moves the registered T1 kernel gives positive probability at
    the true ranking — a floor-free comparison (the held-out gain against T1 is dominated by T1's FLOOR on the
    moves outside its support)."""
    from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
    import math
    ref: Dict[str, T1Kernel] = {}
    ks: Dict[str, T1Kernel] = {}
    tot, n = 0.0, 0
    for p in prepared:
        r = ref.get(p.role) or ref.setdefault(p.role, T1Kernel(p.role, cfg, kf.REFERENCE))
        if r.distribution(p.state, p.ranking).get(p.action, 0.0) <= 0.0:
            continue
        k = ks.get(p.role) or ks.setdefault(p.role, T1Kernel(p.role, cfg, params))
        tot += math.log(max(k.distribution(p.state, p.ranking).get(p.action, 0.0), kf.FLOOR))
        n += 1
    return {"n_moves": n, "mean_ll": tot / max(1, n)}


def kernel_wide(d: Path, cfg: LiveConfig, folds: int, seed: int) -> dict:
    """Gate 4 under the widened family (DESIGN_NOUL2 §3.3a): the same held-out rule against T1 (≥ 0.05 nats
    per move), plus the split count under T1, the T1-family fit (``kernel.json``) and the widened fit, and the
    widened fit's mean log-likelihood against the T1-family fit's (descriptive: both are fits to all moves)."""
    records = kf.load_records(str(d / "opponent_moves.jsonl"))
    narrow = json.loads((gates_dir(d) / "kernel.json").read_text())["briefs"]
    out = {"family": "wide", "fit_params": list(kf.FIT_PARAMS_WIDE), "threshold_gain_nats_per_move": kf.GATE_GAIN_NATS,
           "floor": kf.FLOOR, "briefs": {}}
    for b in BRIEFS:
        sub = [r for r in records if r["brief"] == b]
        if not sub:
            out["briefs"][b] = {"n_moves": 0}
            continue
        rep = kf.heldout_report(sub, cfg, folds=folds, seed=seed, params_to_fit=kf.FIT_PARAMS_WIDE)
        rep["brief"], rep["family"] = b, "wide"
        prepared = kf.prepare(sub, cfg)
        wide = _params_from_json(rep["fitted_all"])
        t1fit = _params_from_json(narrow[b]["fitted_all"])
        rep["support_fitted"] = support_breakdown(prepared, wide, cfg)
        rep["split"] = {"T1": split_breakdown(prepared, kf.REFERENCE, cfg), "t1_family_fit": split_breakdown(prepared, t1fit, cfg),
                        "wide_fit": split_breakdown(prepared, wide, cfg)}
        rep["mean_ll_t1_family_fit_all"] = kf.mean_ll(prepared, t1fit, cfg)
        rep["on_t1_support"] = {"T1": ll_on_t1_support(prepared, kf.REFERENCE, cfg),
                                "t1_family_fit": ll_on_t1_support(prepared, t1fit, cfg),
                                "wide_fit": ll_on_t1_support(prepared, wide, cfg)}
        _dump(rep, gates_dir(d) / f"kernel_fit_wide_{b}.json")
        out["briefs"][b] = {k: rep[k] for k in ("n_moves", "n_episodes", "heldout_gain_nats_per_move", "adopt_fit", "fitted_all",
                                                "reference", "mean_ll_fitted_all", "mean_ll_reference_all",
                                                "mean_ll_t1_family_fit_all", "support_fitted", "split", "on_t1_support")}
        out["briefs"][b].update({"folds": folds, "seed": seed, "t1_family_fit": narrow[b]["fitted_all"],
                                 "t1_family_adopt_fit": narrow[b]["adopt_fit"],
                                 "t1_family_heldout_gain_nats_per_move": narrow[b]["heldout_gain_nats_per_move"]})
    return out


def planner_cost(d: Path, cfg: LiveConfig, brief: str, role: str = "buyer", ranking_index: int = 0) -> dict:
    """Wall time, memo states and peak memory of one Stage 2 planner (the evaluator reference: 4 oracles, both
    openers) for one cell, under T1 and under the widened fit of ``brief``. Each kernel is timed in a fresh
    subprocess so peak RSS is its own. A single-machine measurement: the times are for sizing the run, not results."""
    import platform
    W = json.loads((gates_dir(d) / "kernel_wide.json").read_text())
    code = (
        "import json, resource, sys, time\n"
        "from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS\n"
        "from llm_rl_scripts.car_dealer_noul import arms_noul2 as a2, gates_noul2 as g\n"
        "from pathlib import Path\n"
        "cfg = g.config_of(Path(sys.argv[1])); p = a2.params_from_json(json.loads(sys.argv[2]))\n"
        "t = time.time(); c = a2.Stage2Cell(cfg, sys.argv[3], ALL_RANKINGS[int(sys.argv[4])], fitted=p)\n"
        "ce = {o: c.ceilings(o) for o in ('buyer', 'seller')}\n"
        "el = time.time() - t; h = c.reference()\n"
        "rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "rss = rss / 2**30 if sys.platform == 'darwin' else rss / 2**20\n"
        "print(json.dumps({'seconds': el, 'peak_rss_gib': rss, 'memo_states': sum(len(h[k]._v) for k in ('ask', 'noask', 'full_ask', 'full_noask'))}))\n")
    out = {"brief": brief, "role": role, "ranking_index": ranking_index, "machine": platform.platform(), "python": platform.python_version(),
           "commit": _git_head(), "kernels": {}}
    for name, params in (("T1", kf.REFERENCE.to_json()), ("wide_fit", W["briefs"][brief]["fitted_all"])):
        r = subprocess.run([sys.executable, "-c", code, str(d), json.dumps(params), role, str(ranking_index)],
                           capture_output=True, text=True, check=True, env={**__import__("os").environ, "PYTHONPATH": str(Path.cwd())})
        out["kernels"][name] = json.loads(r.stdout.strip().splitlines()[-1])
    out["slowdown"] = out["kernels"]["wide_fit"]["seconds"] / out["kernels"]["T1"]["seconds"]
    return out


def _params_from_json(j: dict):
    from dataclasses import replace
    return replace(kf.REFERENCE, alpha0=Fraction(j["alpha0"]), beta=float(j["beta"]), p_counter=float(j["p_counter"]),
                   p_hold=float(j["p_hold"]), accept_noise=float(j["accept_noise"]), shortlist=int(j["shortlist"]),
                   eps_propose=float(j.get("eps_propose", 0.0)), p_ask=float(j.get("p_ask", 0.0)))


# ---- gate 5: realizer fidelity ---------------------------------------------------------------------------

def replay_focal_moves(rows: List[dict], moves: List[dict], cfg: LiveConfig) -> Tuple[List[dict], Counter]:
    """Every realized focal move of the pilot with the state it was made in: the focal's decisions and the
    opponent's logged moves are interleaved from the opener and checked against the opponent-move records'
    event prefixes. Returns (moves, skipped-episode counter)."""
    by_ep: Dict[str, List[dict]] = defaultdict(list)
    for m in moves:
        by_ep[m["episode"]].append(m)
    out, skipped = [], Counter()
    for r in rows:
        key, focal = r["episode_key"], r["focal"]
        decisions, opp = r["decisions"], sorted(by_ep.get(key, []), key=lambda m: len(m["events"]))
        if len(decisions) != len(r["realizer_calls"]):
            skipped["decisions_vs_realizer_calls"] += 1
            continue
        state, k, j, ok = new_game(cfg, r["opener"]), 0, 0, True
        while state.terminal is None:
            if state.turn == focal:
                if k >= len(decisions):
                    ok = False
                    break
                act = kf.action_from_key(decisions[k]["chosen"])
                if act.kind == ASK:                       # the pilot arms never ask; a typed answer is not logged here
                    ok = False
                    break
                legal = legal_actions(state, focal)
                out.append({"episode": key, "k": k, "role": focal, "brief": r["brief"], "arm": r["arm"], "kind": act.kind,
                            "action": act.key, "legal": [a.key for a in legal], "text": decisions[k]["realized"],
                            "auto_ok": bool(r["realizer_calls"][k]["ok"]), "fallback": bool(r["realizer_calls"][k]["fallback"])})
                state = apply(state, act, None)
                k += 1
            else:
                if j >= len(opp):
                    ok = False
                    break
                m = opp[j]
                if kf.events_to_json(state.events) != m["events"]:
                    ok = False
                    break
                act = kf.action_from_key(m["action"])
                ans = truthful_answer(act.question_id, tuple(r["focal_ranking"])) if act.kind == ASK else None
                state = apply(state, act, ans)
                j += 1
        if not ok or k != len(decisions) or j != len(opp):
            skipped["replay_mismatch"] += 1
            out = [o for o in out if o["episode"] != key]
    return out, skipped


def build_packet(moves: List[dict], n_per_role: int = N_PER_ROLE, seed: int = PACKET_SEED) -> Dict[str, list]:
    rng = random.Random(seed)
    chosen = []
    for role in ("buyer", "seller"):
        pool = sorted((m for m in moves if m["role"] == role), key=lambda m: (m["episode"], m["k"]))
        chosen.extend(rng.sample(pool, min(n_per_role, len(pool))))
    rng.shuffle(chosen)
    packet, key = [], []
    for n, m in enumerate(chosen):
        letters = lettered([kf.action_from_key(k) for k in m["legal"]], random.Random(f"{seed}|{n}"))
        true_letter = next(L for L, a in letters.items() if a.key == m["action"])
        packet.append({"item": n, "role": m["role"], "text": m["text"], "moves": moves_text(letters), "labels": list(letters)})
        key.append({"item": n, "episode": m["episode"], "k": m["k"], "role": m["role"], "brief": m["brief"], "arm": m["arm"],
                    "kind": m["kind"], "action": m["action"], "letter": true_letter, "auto_ok": m["auto_ok"], "fallback": m["fallback"]})
    return {"packet": packet, "key": key}


def write_packet(built: Dict[str, list], g: Path) -> None:
    with (g / "realizer_packet.jsonl").open("w") as f:
        for row in built["packet"]:
            f.write(json.dumps(row) + "\n")
    _dump(built["key"], g / "realizer_key.json")
    with (g / "realizer_view.txt").open("w") as f:
        for row in built["packet"]:
            f.write(f"### item {row['item']}  ({row['role']} speaking)\n{row['text']}\n\nLegal moves:\n{row['moves']}\n\n")


class FakeJudge:
    """Offline judge: always the first label, nothing added (so a packet's recovery is low — the fail path)."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, max_tokens):
        from .opponent_llm import Completion
        self.calls += 1
        first = re.search(r"^\s+([A-Z]{1,2})\. ", messages[-1]["content"], re.M).group(1)
        return Completion(f"DECISION: {first}\nADDED: none")


def judge_prompt(row: dict) -> str:
    return (f"The {row['role']} says:\n\"{row['text']}\"\n\nThe moves the {row['role']} could make:\n{row['moves']}\n\n"
            "Which one move does the line make, and does it add a commitment beyond it?")


def parse_judgement(text: str, labels: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    letter, _ = parse_reply(text, labels)
    m = _ADDED_RE.search(text)
    added = m.group(1).strip() if m else None
    if added is not None and added.strip(" .`\"'").lower() in ("none", ""):
        added = ""
    return letter, added


def judge(packet: List[dict], client, log_path: Optional[Path] = None) -> List[dict]:
    labels = []
    with (log_path.open("w") if log_path else _Null()) as log:
        for row in packet:
            system = JUDGE_SYSTEM.format(role=row["role"], other=other(row["role"]))
            reply = client.complete(system, [{"role": "user", "content": judge_prompt(row)}], JUDGE_MAX_TOKENS)
            letter, added = parse_judgement(reply.text, row["labels"])
            labels.append({"item": row["item"], "letter": letter, "added": added, "model": reply.model,
                           "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens})
            if log_path:
                log.write(json.dumps({"item": row["item"], "reply": reply.text}) + "\n")
    return labels


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, s):
        pass


def realizer_report(key: List[dict], labels: List[dict], all_moves_auto: Optional[Counter] = None) -> dict:
    lab = {l["item"]: l for l in labels}
    out = {"threshold": {"recovery_min": RECOVERY_MIN, "added_max": ADDED_MAX, "n_per_role": N_PER_ROLE}, "roles": {}, "by_kind": {}}
    for role in ("buyer", "seller"):
        ks = [k for k in key if k["role"] == role]
        rec = [lab[k["item"]]["letter"] == k["letter"] for k in ks]
        added = [k["item"] for k in ks if lab[k["item"]]["added"]]
        unparsed = [k["item"] for k in ks if lab[k["item"]]["letter"] is None]
        missed = [k["item"] for k in ks if lab[k["item"]]["letter"] != k["letter"]]
        n = len(ks)
        out["roles"][role] = {"n_judged": n, "recovered": sum(rec), "recovery": sum(rec) / n if n else None,
                              "added_commitments": len(added), "added_items": added, "missed_items": missed,
                              "unparsed_judgements": len(unparsed), "auto_ok": sum(k["auto_ok"] for k in ks),
                              "fallbacks": sum(k["fallback"] for k in ks),
                              "passes": n > 0 and sum(rec) / n >= RECOVERY_MIN and len(added) <= ADDED_MAX}
    for kind in sorted({k["kind"] for k in key}):
        ks = [k for k in key if k["kind"] == kind]
        out["by_kind"][kind] = {"n": len(ks), "recovered": sum(lab[k["item"]]["letter"] == k["letter"] for k in ks),
                                "added": sum(bool(lab[k["item"]]["added"]) for k in ks)}
    if all_moves_auto is not None:
        out["all_realized_moves"] = dict(all_moves_auto)
    out["passes"] = all(v["passes"] for v in out["roles"].values())
    return out


# ---- gate 6: clairvoyant gap ------------------------------------------------------------------------------

def cluster_means(recs: Iterable[dict]) -> Dict[tuple, float]:
    by = defaultdict(list)
    for r in recs:
        by[cluster_of(r)].append(y(r))
    return {k: statistics.mean(v) for k, v in by.items()}


def clairvoyant(rows: List[dict], ceilings: dict, B: int, seed: int, brief: str = "guarded", cell_set: str = "focal_opens") -> dict:
    sub = [r for r in rows if r["brief"] == brief and in_set(r, cell_set)]
    a = cluster_means(r for r in sub if r["arm"] == "plan_reveal")
    b = cluster_means(r for r in sub if r["arm"] == "plan_noask")
    pb = paired_bootstrap(a, b, B, seed)
    umax = sorted({round(r["u_max"]["float"], 12) for r in sub})
    assert len(umax) == 1, f"u_max differs across records: {umax}"
    r_res = float(Fraction(ceilings["config"]["reservation"]))
    y_scale = umax[0] - r_res
    cells = {k: v for k, v in ceilings["cells"].items() if in_set(tuple(k.split("|")), cell_set)}
    exact = statistics.mean(v["delta_full"] for v in cells.values()) / y_scale if cells else None
    return {"brief": brief, "cell_set": cell_set, "delta": DELTA, "B": B, "seed": seed, "n_episodes": len(sub),
            "n_clusters": pb["n_clusters"], "point": pb["point"], "ci95": pb["ci95"],
            "exact_delta_full_Y_under_assumed_kernel": exact, "y_scale": y_scale,
            "passes": pb["point"] is not None and pb["point"] >= DELTA}


# ---- render ----------------------------------------------------------------------------------------------

def _f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def render(d: Path, probe: Optional[Path] = None) -> str:
    g = gates_dir(d)
    A = json.load((probe / "gates_answers.json").open()) if probe and (probe / "gates_answers.json").exists() else None
    P = json.load((g / "protocol.json").open())
    K = json.load((g / "kernel.json").open()) if (g / "kernel.json").exists() else None
    W = json.load((g / "kernel_wide.json").open()) if (g / "kernel_wide.json").exists() else None
    R = json.load((g / "realizer_report.json").open()) if (g / "realizer_report.json").exists() else None
    C = json.load((g / "clairvoyant.json").open()) if (g / "clairvoyant.json").exists() else None
    H = json.load((d / "frozen_hashes.json").open()) if (d / "frozen_hashes.json").exists() else {"brief_hash": "n/a", "realizer_hash": "n/a"}
    S = json.load((d / "summary.json").open())
    L = []
    L.append("# CD-NOUL-2 Phase 0 gates from the pilot\n")
    L.append(f"Rendered by `gates_noul2.py render` from `{d}/gates/` (pilot `{d}`, {S['episodes_written']} episodes, "
             f"{S['error_stubs']} error stubs; brief hash `{H['brief_hash']}`, realizer hash `{H['realizer_hash']}`; "
             f"rendered at commit `{_git_head()}`). Thresholds are `DESIGN_NOUL2.md` §6. Pilot outcome contrasts were not read; "
             "gate 6 is the one contrast §6 names.\n")
    L.append("## Verdicts\n")
    L.append("| gate | quantity | threshold | value | verdict |\n|---|---|---|---|---|")
    for b in BRIEFS:
        p = P["briefs"][b]
        L.append(f"| 1 opponent protocol, `{b}` | well-formed share / defaulted share over {p['calls']} moves | ≥ {WELL_FORMED_MIN} / ≤ {DEFAULTED_MAX} | "
                 f"{_f(p['well_formed_share'])} / {_f(p['defaulted_share'])} | {'PASS' if p['passes'] else 'FAIL'} |")
    if A:
        c, gd, ad = A["briefs"]["candid"], A["briefs"]["guarded"], A["briefs"]["adversarial"]
        G = A["gates"]
        L.append(f"| 2 never-lies audit | contradictions in blind-labelled probe answers: candid {len(c['contradictions'])} of {c['n_audited']}, "
                 f"guarded {len(gd['contradictions'])} of {gd['n_audited']}; adversarial misleads {_f(ad['contradiction_share'], 2)} of {ad['n_audited']} | "
                 f"0 candid, ≤ 2 guarded; adversarial ≥ 0.2 | — | candid {'PASS' if G['2_candid_never_lies'] else 'FAIL'}, guarded {'PASS' if G['2_guarded_never_lies'] else 'FAIL'}, "
                 f"adversarial {'MISLEADS' if G['2_adversarial_misleads'] else 'INERT'} |")
        L.append(f"| 3 guarded is guarded | `regex` accuracy on the {gd['n_bank']} guarded probe answers | < 0.9 | {_f(gd['regex_accuracy_bank'])} | "
                 f"{'PASS' if G['3_guarded_is_guarded'] else 'FAIL'} |")
    else:
        L.append("| 2 never-lies audit | contradictions in 60 blind-labelled answers per brief | 0 candid, ≤ 2 guarded; adversarial misleads ≥ 20 % | — | PENDING: the pilot arms never ask, so no opponent answers exist; needs the answer probe |")
        L.append("| 3 guarded is guarded | `regex` accuracy on guarded answers | < 0.9 | — | PENDING (answer probe) |")
    if W:
        for b in BRIEFS:
            k = W["briefs"][b]
            if k.get("n_moves", 0) == 0:
                continue
            L.append(f"| 4 kernel fit (widened family, §3.3a), `{b}` | held-out gain over T1, nats per move ({k['n_moves']} moves, {k['folds']} folds); "
                     f"T1-family fit {_f(k['t1_family_heldout_gain_nats_per_move'])} (superseded) | ≥ {W['threshold_gain_nats_per_move']} to adopt | "
                     f"{_f(k['heldout_gain_nats_per_move'])} | {'ADOPT FIT' if k['adopt_fit'] else 'KEEP T1'} |")
    elif K:
        for b in BRIEFS:
            k = K["briefs"][b]
            if k.get("n_moves", 0) == 0:
                continue
            L.append(f"| 4 kernel fit, `{b}` | held-out gain over T1, nats per move ({k['n_moves']} moves, {k['folds']} folds) | ≥ {K['threshold_gain_nats_per_move']} to adopt | "
                     f"{_f(k['heldout_gain_nats_per_move'])} | {'ADOPT FIT' if k['adopt_fit'] else 'KEEP T1'} |")
    if R:
        for role in ("buyer", "seller"):
            r = R["roles"][role]
            L.append(f"| 5 realizer fidelity, {role} | blind recovery / added commitments in {r['n_judged']} realized moves | ≥ {RECOVERY_MIN} / {ADDED_MAX} | "
                     f"{_f(r['recovery'])} / {r['added_commitments']} | {'PASS' if r['passes'] else 'FAIL'} |")
    if C:
        L.append(f"| 6 clairvoyant gap | Δ_clair = plan_reveal − plan_noask, Y, `{C['brief']}`, {C['cell_set'].replace('_', ' ')} ({C['n_clusters']} clusters) | ≥ δ = {C['delta']} | "
                 f"{_f(C['point'])} [{_f(C['ci95'][0])}, {_f(C['ci95'][1])}] | {'PASS' if C['passes'] else 'FAIL'} |")
    L.append("| 7 parity | Stage 2 runner reproduces Stage 1 `plan_ask-oracle-0` with scripted T1 + templates + `oracle` | exact | — | PASS (`test_arms_noul2.py::test_parity_with_stage1`, offline) |")
    L.append("| 8 calibration constants | Platt map and a_q vectors from the frozen Stage 1 matrices | committed | — | PASS (`results/noul_bank_20261108/calibration_noul2.json`, `calibrate_noul.py`) |")
    if A and "noul" in A["briefs"]["candid"]:
        cn, gn_ = A["briefs"]["candid"]["noul"], A["briefs"]["guarded"].get("noul", {})
        L.append(f"| 9 no confident-wrong reads on candid | confident-wrong `noulcal` reads ({cn['map']}, clamp {cn['clamp']}) on the audited candid answers; guarded reported | 0 | "
                 f"candid {len(cn['confident_wrong_audited'])} of {A['briefs']['candid']['n_audited']} (bank {len(cn['confident_wrong_bank'])} of {cn['n_read']}); "
                 f"guarded bank {len(gn_.get('confident_wrong_bank', []))} of {gn_.get('n_read', 0)} | {'PASS' if A['gates'].get('9_candid_no_confident_wrong') else 'FAIL'} |")
    else:
        L.append("| 9 no confident-wrong reads on candid | confident-wrong reads under Platt + clamp 0.95 on blind-audited candid answers | 0 | — | PENDING (answer probe) |")
    L.append("")
    L.append("## Gate 1 — opponent protocol, per brief\n")
    L.append("| brief | episodes | opponent moves | well-formed | retried | defaulted | stop end_turn | realizer calls ok / fallback | opponent asked (episodes) | tokens in / out |\n|---|---|---|---|---|---|---|---|---|---|")
    for b in BRIEFS:
        p = P["briefs"][b]
        L.append(f"| {b} | {p['episodes']} | {p['calls']} | {p['well_formed']} | {p['retried']} | {p['defaulted']} | {p.get('stop_end_turn', 0)} | "
                 f"{p['realizer_ok']} / {p['realizer_fallback']} of {p['realizer_calls']} | {p['opponent_asked_episodes']} | {p['input_tokens']} / {p['output_tokens']} |")
    L.append("")
    if K:
        if W:
            L.append("## Gate 4 — widened kernel family (`DESIGN_NOUL2.md` §3.3a; `gates_noul2 kernel-wide`, `gates/kernel_wide.json`)\n")
            L.append("The family adds `eps_propose` (weight on a uniform draw over all legal proposals) and `p_ask` (a type-independent "
                     "question rate) to the T1 form; both are fitted with the T1 parameters, the rule is unchanged (held-out gain over the "
                     f"registered T1 ≥ {W['threshold_gain_nats_per_move']} nats per move).\n")
            L.append("| brief | held-out gain over T1 | adopt | eps_propose | p_ask | α₀ | β | p_counter | p_hold | accept_noise | mean LL widened / T1-family fit / T1 (all moves) |\n|---|---|---|---|---|---|---|---|---|---|---|")
            for b in BRIEFS:
                k = W["briefs"][b]
                if k.get("n_moves", 0) == 0:
                    continue
                f = k["fitted_all"]
                L.append(f"| {b} | {_f(k['heldout_gain_nats_per_move'])} | {'yes' if k['adopt_fit'] else 'no'} | {_f(f.get('eps_propose', 0.0), 3)} | {_f(f.get('p_ask', 0.0), 3)} | "
                         f"{f['alpha0']} | {_f(f['beta'], 3)} | {_f(f['p_counter'], 3)} | {_f(f['p_hold'], 3)} | {_f(f['accept_noise'], 3)} | "
                         f"{_f(k['mean_ll_fitted_all'])} / {_f(k['mean_ll_t1_family_fit_all'])} / {_f(k['mean_ll_reference_all'])} |")
            L.append(f"\nThe held-out gain is dominated by T1's floor ({kf.FLOOR:g}) on the moves outside its support. Floor-free check — mean "
                     "log-likelihood over only the moves T1 itself supports (all-moves fits, descriptive):\n")
            L.append("| brief | moves T1 supports | T1 | T1-family fit | widened fit |\n|---|---|---|---|---|")
            for b in BRIEFS:
                k = W["briefs"][b]
                if k.get("n_moves", 0) == 0 or "on_t1_support" not in k:
                    continue
                o = k["on_t1_support"]
                L.append(f"| {b} | {o['T1']['n_moves']} | {_f(o['T1']['mean_ll'])} | {_f(o['t1_family_fit']['mean_ll'])} | {_f(o['wide_fit']['mean_ll'])} |")
            L.append("\nSplit count (`split_breakdown`): of the moves a kernel gives zero probability at the opponent's true ranking, those zero under "
                     "every ranking (no type evidence; the η = 0 filter keeps its belief) and those positive under some other ranking (a split: "
                     "the filter sets the true ranking's belief to 0 for the rest of the episode):\n")
            L.append("| brief | kernel | zero at true ranking | zero under every ranking | split | split by kind | episodes with a split |\n|---|---|---|---|---|---|---|")
            names = {"T1": "T1", "t1_family_fit": "T1-family fit", "wide_fit": "widened fit"}
            for b in BRIEFS:
                k = W["briefs"][b]
                if k.get("n_moves", 0) == 0:
                    continue
                for key, label in names.items():
                    sp = k["split"][key]
                    kinds = ", ".join(f"{kk} {v['split']}" for kk, v in sp["by_kind"].items() if v.get("split")) or "—"
                    L.append(f"| {b} | {label} | {sp['zero_at_true']} | {sp['zero_all']} | {sp['split']} | {kinds} | {sp['episodes_with_split']} of {sp['n_episodes']} |")
            L.append("")
        L.append("## Gate 4 — kernel fit per brief, T1 family" + (" (superseded by §3.3a; kept as the record)" if W else "") +
                 " (`kernel_fit.heldout_report`, episode-level held-out split)\n")
        L.append("| brief | moves | episodes | mean LL fitted (all) | mean LL T1 (all) | T1 zero-probability moves | held-out gain | adopt | fitted α₀ | β | p_counter | p_hold | accept_noise |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in BRIEFS:
            k = K["briefs"][b]
            if k.get("n_moves", 0) == 0:
                continue
            f = k["fitted_all"]
            L.append(f"| {b} | {k['n_moves']} | {k['n_episodes']} | {_f(k['mean_ll_fitted_all'])} | {_f(k['mean_ll_reference_all'])} | {k['zero_prob_reference_all']} | "
                     f"{_f(k['heldout_gain_nats_per_move'])} | {'yes' if k['adopt_fit'] else 'no'} | {f['alpha0']} | {_f(f['beta'], 3)} | {_f(f['p_counter'], 3)} | {_f(f['p_hold'], 3)} | {_f(f['accept_noise'], 3)} |")
        L.append("\nA T1 zero-probability move is a logged opponent move the reference kernel gives probability 0 (it is floored at "
                 f"{kf.FLOOR:g} in the likelihood). Per-fold detail: `gates/kernel_fit_<brief>.json`.\n")
        L.append("Support of the T1 family over the live opponent's moves (moves outside the kernel's support, by kind of the logged move; "
                 "T1 never asks, so every opponent question is outside it):\n")
        L.append("| brief | kind | moves | outside T1 support | outside fitted support |\n|---|---|---|---|---|")
        for b in BRIEFS:
            k = K["briefs"][b]
            if k.get("n_moves", 0) == 0:
                continue
            for kind, v in k["support_reference"].items():
                fz = k["support_fitted"].get(kind, {}).get("zero", 0)
                L.append(f"| {b} | {kind} | {v['n']} | {v['zero']} | {fz} |")
        L.append("")
        for b in BRIEFS:
            k = K["briefs"][b]
            if k.get("n_moves", 0) == 0:
                continue
            pr = k["support_reference"].get("PROPOSE", {"n": 0, "zero": 0})
            L.append(f"Reading, `{b}`: {pr['zero']} of the opponent's {pr['n']} proposals ({_f(pr['zero'] / max(1, pr['n']), 2)}) are deals the T1 family "
                     f"cannot propose from any parameter setting (off its aspiration shortlist), so the fit cannot fix them; the fitted β of "
                     f"{_f(k['fitted_all']['beta'], 3)} makes the within-shortlist choice near-uniform, and whatever held-out gain there is comes from "
                     f"accept_noise ({_f(k['fitted_all']['accept_noise'], 3)}) and p_hold ({_f(k['fitted_all']['p_hold'], 3)}). "
                     "The T1 family is misspecified in support for this opponent" + ("; DESIGN_NOUL2 §3.3a widens it (above).\n" if W else
                     "; the verdict column applies the registered rule as written.\n"))
    if R:
        L.append("## Gate 5 — realizer fidelity (blind judge over the packet; `gates/realizer_packet.jsonl`, key `realizer_key.json`, labels `realizer_labels.jsonl`)\n")
        L.append("| role | judged | recovered | recovery | added commitments | judge unparsed | auto round-trip ok | fallbacks |\n|---|---|---|---|---|---|---|---|")
        for role in ("buyer", "seller"):
            r = R["roles"][role]
            L.append(f"| {role} | {r['n_judged']} | {r['recovered']} | {_f(r['recovery'])} | {r['added_commitments']} | {r['unparsed_judgements']} | {r['auto_ok']} | {r['fallbacks']} |")
        L.append("\n| move kind | judged | recovered | added |\n|---|---|---|---|")
        for kind, v in R["by_kind"].items():
            L.append(f"| {kind} | {v['n']} | {v['recovered']} | {v['added']} |")
        if R.get("all_realized_moves"):
            a = R["all_realized_moves"]
            L.append(f"\nAll realized moves in the pilot: {a.get('moves', 0)} (auto round-trip ok {a.get('auto_ok', 0)}, fallbacks {a.get('fallback', 0)}); "
                     f"replay skipped episodes: {a.get('skipped_episodes', 0)}.\n")
        if R.get("judge"):
            L.append(f"Judge: `{R['judge']}`.\n")
    if C:
        L.append("## Gate 6 — clairvoyant gap\n")
        L.append(f"Δ_clair = `plan_reveal − plan_noask` in Y = net gain / (U_max − r), `{C['brief']}`, {C['cell_set'].replace('_', ' ')}: paired over "
                 f"{C['n_clusters']} (buyer ranking, seller ranking) clusters from {C['n_episodes']} episodes, percentile cluster bootstrap B = {C['B']} seed {C['seed']}.\n")
        L.append("| quantity | value |\n|---|---|")
        L.append(f"| Δ_clair point | {_f(C['point'])} |")
        L.append(f"| 95 % CI | [{_f(C['ci95'][0])}, {_f(C['ci95'][1])}] |")
        L.append(f"| δ | {C['delta']} |")
        L.append(f"| exact Δ_full under the planner's assumed kernel (Y, same cells; `ceilings.json`) | {_f(C['exact_delta_full_Y_under_assumed_kernel'])} |")
        L.append("")
    if A:
        L.append("## Gates 2, 3, 9 — the answer probe (`answer_probe.py`; `" + str(probe) + "`)\n")
        L.append("The pilot arms never ask, so the opponent's answers come from a probe: every bank question asked as the focal's opening move "
                 "to the live opponent under each brief, role and planted ranking (3 × 2 × 6 × 9 answers), every raw reply stored. "
                 f"Blind packet of {sum(v['n_audited'] for v in A['briefs'].values())} answers balanced on the truthful answer, sealed key, labeller: "
                 f"{A.get('labeller') or 'unrecorded'}. Limitation: no prior exchange, so this probes the briefs' answering, not answers under negotiating pressure.\n")
        L.append("| brief | answers | audited | labels yes / no / unclear | contradictions | agree with truth | `regex` acc. (bank) | `regex` unparsed | retried / defaulted |\n|---|---|---|---|---|---|---|---|---|")
        for b in BRIEFS:
            v = A["briefs"][b]
            lab = v["labels"]
            L.append(f"| {b} | {v['n_bank']} | {v['n_audited']} | {lab.get('yes', 0)} / {lab.get('no', 0)} / {lab.get('unclear', 0)} | {len(v['contradictions'])} | "
                     f"{_f(v['agree_share'], 2)} | {_f(v['regex_accuracy_bank'])} | {v['regex_unparsed_bank']} | {v['protocol']['retried']} / {v['protocol']['defaulted']} |")
        if "noul" in A["briefs"]["candid"]:
            L.append("\n| brief | Noul reads | map (T, b, clamp) | confident-wrong, audited | confident-wrong, bank | confident-wrong, raw bank | mean read error |\n|---|---|---|---|---|---|---|")
            for b in BRIEFS:
                v = A["briefs"][b].get("noul")
                if not v:
                    continue
                L.append(f"| {b} | {v['n_read']} | {v['map']} ({_f(v['T'], 3)}, {_f(v['b'], 3)}, {v['clamp']}) | {len(v['confident_wrong_audited'])} | "
                         f"{len(v['confident_wrong_bank'])} | {len(v['confident_wrong_raw_bank'])} | {_f(v['mean_read_error'])} |")
        L.append("")
    else:
        L.append("## Pending\n")
        L.append("Gates 2, 3 and 9 need opponent answers to the bank's questions under each brief (the answer probe: "
                 "`LLMOpponent.answer` under planted rankings, about 60 answers per brief, blind-labelled with a sealed key as Stage 1's `audit.py`). "
                 "The pilot arms `plan_noask` and `plan_reveal` never ask, so the pilot holds none.\n")
    return "\n".join(L)


# ---- CLI ---------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("protocol", "kernel", "kernel-wide", "planner-cost", "realizer-packet", "realizer-judge", "realizer-report", "clairvoyant", "render"))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=N_PER_ROLE)
    ap.add_argument("--packet-seed", type=int, default=PACKET_SEED)
    ap.add_argument("--backend", default="fake", choices=("fake", "anthropic"))
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--out", default=None, help="render: path of the Markdown note")
    ap.add_argument("--probe-dir", default=None, help="render: answer-probe dir whose gates_answers.json fills gates 2, 3, 9")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    g = gates_dir(d)
    cfg = config_of(d)
    if args.step == "protocol":
        rep = protocol(load_rows(d))
        _dump(rep, g / "protocol.json")
        print(json.dumps({b: (v["well_formed_share"], v["defaulted_share"], v["passes"]) for b, v in rep["briefs"].items()}))
        return 0
    if args.step == "kernel":
        rep = kernel(d, cfg, args.folds, args.seed)
        _dump(rep, g / "kernel.json")
        print(json.dumps({b: (v.get("heldout_gain_nats_per_move"), v.get("adopt_fit")) for b, v in rep["briefs"].items()}))
        return 0
    if args.step == "planner-cost":
        rep = planner_cost(d, cfg, "guarded")
        _dump(rep, g / "planner_cost.json")
        print(json.dumps(rep))
        return 0
    if args.step == "kernel-wide":
        rep = kernel_wide(d, cfg, args.folds, args.seed)
        _dump(rep, g / "kernel_wide.json")
        print(json.dumps({b: (v.get("heldout_gain_nats_per_move"), v.get("adopt_fit"), v.get("split", {}).get("wide_fit", {}).get("split"))
                          for b, v in rep["briefs"].items()}))
        return 0
    if args.step == "realizer-packet":
        rows = load_rows(d)
        moves, skipped = replay_focal_moves(rows, kf.load_records(str(d / "opponent_moves.jsonl")), cfg)
        built = build_packet(moves, args.n, args.packet_seed)
        write_packet(built, g)
        auto = Counter(moves=len(moves), auto_ok=sum(m["auto_ok"] for m in moves), fallback=sum(m["fallback"] for m in moves),
                       skipped_episodes=sum(skipped.values()))
        _dump({**auto, "skipped": dict(skipped)}, g / "realizer_all_moves.json")
        print(f"packet: {len(built['packet'])} items from {len(moves)} realized moves (skipped {dict(skipped)}) -> {g / 'realizer_packet.jsonl'}; key sealed")
        return 0
    if args.step == "realizer-judge":
        packet = [json.loads(l) for l in (g / "realizer_packet.jsonl").open() if l.strip()]
        if args.backend == "anthropic":
            from llm_rl_scripts.car_dealer_persona.run import AnthropicClient
            client = AnthropicClient(MODEL)
        else:
            client = FakeJudge()
        labels = judge(packet, client, g / "realizer_judge_raw.jsonl")
        with (g / "realizer_labels.jsonl").open("w") as f:
            for l in labels:
                f.write(json.dumps(l) + "\n")
        print(f"judged {len(labels)} items with {args.backend} ({MODEL if args.backend == 'anthropic' else 'fake'})")
        return 0
    if args.step == "realizer-report":
        key = json.load((g / "realizer_key.json").open())
        labels = [json.loads(l) for l in (g / "realizer_labels.jsonl").open() if l.strip()]
        missing = {k["item"] for k in key} - {l["item"] for l in labels}
        if missing:
            ap.error(f"{len(missing)} packet items have no label")
        auto = Counter(json.load((g / "realizer_all_moves.json").open())) if (g / "realizer_all_moves.json").exists() else None
        if auto is not None:
            auto.pop("skipped", None)
        rep = realizer_report(key, labels, auto)
        models = Counter(l.get("model") for l in labels)
        rep["judge"] = ", ".join(f"{m} ×{n}" for m, n in models.items())
        _dump(rep, g / "realizer_report.json")
        print(json.dumps({r: (v["recovery"], v["added_commitments"], v["passes"]) for r, v in rep["roles"].items()}))
        return 0
    if args.step == "clairvoyant":
        rep = clairvoyant(load_rows(d), json.load((d / "ceilings.json").open()), args.B, args.seed)
        _dump(rep, g / "clairvoyant.json")
        print(json.dumps({k: rep[k] for k in ("point", "ci95", "n_clusters", "passes")}))
        return 0
    text = render(d, Path(args.probe_dir) if args.probe_dir else None)
    out = Path(args.out) if args.out else d / "GATES_NOUL2.md"
    out.write_text(text)
    print(f"rendered {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
