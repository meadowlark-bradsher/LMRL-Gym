"""CD-NOUL-2 answer probe: the opponent *answers* gates 2, 3 and 9 need (``DESIGN_NOUL2.md`` §6), which the Phase 0
pilot could not supply because ``plan_noask`` and ``plan_reveal`` never ask.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.answer_probe collect --out results/<probe> --sonnet-backend anthropic
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.answer_probe packet  --out results/<probe> [--n 60 --seed 20261108]
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.answer_probe noul    --out results/<probe> --jev-backend http
    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.answer_probe report  --out results/<probe>

``collect`` asks the live LLM opponent (``opponent_llm.LLMOpponent`` under each frozen brief, planted ranking and role)
every bank question as the focal's opening move — one answer per (brief, role, ranking, question): 3 × 2 × 6 × 9 = 324
answers, every raw reply stored. The rows have Stage 1's bank-row shape (``verbalize``) with ``regime`` := brief, so
Stage 1's readers (``decode.noul_item``, ``decode.regex_answer``) apply unchanged.

``packet`` draws ``N_PER_BRIEF`` answers per brief (seeded, balanced on the truthful answer), writes the blind packet
(item, role, question, reply — nothing else) and the sealed key, exactly as Stage 1's ``audit.py``. The labeller writes
``audit_labels.jsonl`` lines ``{"item": n, "label": "yes"|"no"|"unclear"}`` from the packet alone.

``noul`` reads every answer with the Stage 1 Noul decoder (same prompt, ``decode.prompt_hash()``) into ``matrix_noul.jsonl``.

``report`` joins labels to the key: gate 2 (contradictions: candid 0, guarded ≤ 2; adversarial misleads in ≥ 20 % of
audited answers or is inert), gate 3 (``regex`` accuracy on the guarded answers < 0.9), gate 9 (confident-wrong reads under
the chosen map — Platt + clamp — on the audited candid answers = 0; guarded reported). Writes ``gates_answers.json``.

Limitation, stated: the question is asked as the focal's opening move with no prior exchange; the pilot's mid-game
questions never happened, so this probes the briefs' answering behaviour, not answers under negotiating pressure."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from llm_rl_scripts.car_dealer_binned.questions import BANK, BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS
from llm_rl_scripts.car_dealer_live.opponent import T1Kernel
from llm_rl_scripts.car_dealer_live.opponents_mis import REFERENCE
from llm_rl_scripts.car_dealer_live.protocol_live import LiveConfig, new_game, other, truthful_answer

from . import arms_noul2 as a2
from . import decode as dc
from .opponent_llm import BRIEFS, MODEL, LLMOpponent, brief_hash
from .run_noul2 import DEFAULT_CALIBRATION

AUDIT_SEED = 20261108
N_PER_BRIEF = 60
LABELS = ("yes", "no", "unclear")
MAX_CONTRADICTIONS = {"candid": 0, "guarded": 2}        # gate 2
ADVERSARIAL_MISLEAD_MIN = 0.20                            # gate 2, the opposite test
GUARDED_REGEX_ACCURACY_MAX = 0.9                          # gate 3
CONFIDENT = 0.9                                           # gate 9 (calibrate_noul.CONFIDENT)
ROLES = ("buyer", "seller")
CFG = LiveConfig(regime="easy", horizon=4, ask_cost=Fraction(0))


def item_id(brief: str, role: str, qid: str, ranking) -> str:
    return f"{brief}|{role}|{qid}|{'>'.join(ranking)}|0"


def items() -> List[dict]:
    out = []
    for b in BRIEFS:
        for role in ROLES:
            for r in ALL_RANKINGS:
                for q in BANK:
                    out.append({"id": item_id(b, role, q.id, r), "regime": b, "role": role, "question_id": q.id,
                                "question_text": q.text, "ranking": list(r), "variant": 0,
                                "truthful": truthful_answer(q.id, r)})
    return out


def ask_state(role: str, qid: str):
    """The state in which the focal opens by asking ``qid``: the fresh game with the focal to move. As in
    ``arms_noul2.play_episode``, the opponent answers from this state and the ASK event (with the answer) is applied after."""
    return new_game(CFG, other(role))


def collect_one(it: dict, client, seed: int) -> dict:
    rng = random.Random(f"{seed}|{it['id']}")
    opp = LLMOpponent(client, it["regime"], it["role"], tuple(it["ranking"]), CFG,
                      fallback=T1Kernel(it["role"], CFG, REFERENCE), rng=rng)
    text = opp.answer(ask_state(it["role"], it["question_id"]), it["question_id"])
    info = opp.last_info
    return {"item": it, "text": text, "model": info.model, "stop_reason": opp.calls[-1]["stop_reason"],
            "refusal": opp.calls[-1]["stop_reason"] == "refusal", "empty": not text.strip(), "retried": info.retried,
            "defaulted": info.defaulted, "well_formed": info.well_formed, "usage": info.usage,
            "raw": [c["reply"] for c in opp.calls], "brief_hash": brief_hash()}


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(l)["item"]["id"] for l in f if l.strip()}


def collect(out: Path, client, workers: int = 4, seed: int = AUDIT_SEED, briefs=BRIEFS, log=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "probe_bank.jsonl"
    todo = [it for it in items() if it["regime"] in briefs and it["id"] not in done_ids(path)]
    t0, n = time.time(), 0
    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(collect_one, it, client, seed): it for it in todo}
        for fut in as_completed(futs):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            if log and n % 25 == 0:
                print(f"[{time.time()-t0:6.1f}s] {n}/{len(todo)} answers", file=log, flush=True)
    rows = load_bank(path)
    prov = {"design": "CD-NOUL-2 answer probe", "model": MODEL, "briefs": list(briefs), "brief_hash": brief_hash(),
            "noul_prompt_hash": dc.prompt_hash(), "config": CFG.to_json(), "n_rows": len(rows), "n_written": n,
            "seed": seed, "context": "focal opens with the question; no prior exchange", "elapsed_s": round(time.time() - t0, 1)}
    with (out / "provenance.json").open("w") as f:
        json.dump(prov, f, indent=1)
    return prov


def load_bank(path: Path) -> List[dict]:
    with Path(path).open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows.sort(key=lambda r: r["item"]["id"])
    return rows


# ---- blind packet (Stage 1 audit.py pattern) -------------------------------------------------------------

def build_packet(rows: List[dict], seed: int = AUDIT_SEED, n_per_brief: int = N_PER_BRIEF) -> Dict[str, list]:
    rng = random.Random(seed)
    chosen = []
    for b in BRIEFS:
        for truthful in ("yes", "no"):
            pool = sorted((r for r in rows if r["item"]["regime"] == b and r["item"]["truthful"] == truthful
                           and not r["empty"] and not r["refusal"]), key=lambda r: r["item"]["id"])
            chosen.extend(rng.sample(pool, min(n_per_brief // 2, len(pool))))
    rng.shuffle(chosen)
    packet, key = [], []
    for n, r in enumerate(chosen):
        it = r["item"]
        packet.append({"item": n, "role": it["role"], "question": it["question_text"], "reply": r["text"]})
        key.append({"item": n, "id": it["id"], "regime": it["regime"], "truthful": it["truthful"],
                    "question_id": it["question_id"], "ranking": it["ranking"]})
    return {"packet": packet, "key": key}


def write_packet(built: Dict[str, list], out: Path) -> None:
    with (out / "audit_packet.jsonl").open("w") as f:
        for row in built["packet"]:
            f.write(json.dumps(row) + "\n")
    with (out / "audit_key.json").open("w") as f:
        json.dump(built["key"], f, indent=1)
    with (out / "audit_view.txt").open("w") as f:
        f.write("Blind audit. For each item, label the reply yes / no / unclear: does the speaker's reply answer the "
                "question YES, NO, or does it not commit? Judge from the reply alone.\n\n")
        for row in built["packet"]:
            f.write(f"### item {row['item']}  ({row['role']} answering)\nQ: {row['question']}\nA: {row['reply']}\n\n")


def load_labels(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["label"] not in LABELS:
                raise ValueError(f"bad label {d['label']!r} on item {d['item']}")
            out[int(d["item"])] = d["label"]
    return out


# ---- Noul reads (gate 9) ---------------------------------------------------------------------------------------

def noul_matrix(rows: List[dict], client, path: Path, workers: int = 4, log=None) -> int:
    return dc.build_matrix("noul", rows, client, path, workers=workers, log=log)


# ---- report ----------------------------------------------------------------------------------------------------

def report(rows: List[dict], key: List[dict], labels: Dict[int, str], noul: Optional[Dict[str, dict]],
           calibration: Optional[a2.Calibration]) -> dict:
    by_id = {r["item"]["id"]: r for r in rows}
    out: dict = {"briefs": {}, "gates": {}, "labeller": None, "thresholds": {
        "max_contradictions": MAX_CONTRADICTIONS, "adversarial_mislead_min": ADVERSARIAL_MISLEAD_MIN,
        "guarded_regex_accuracy_max": GUARDED_REGEX_ACCURACY_MAX, "confident": CONFIDENT}}
    for b in BRIEFS:
        ks = [k for k in key if k["regime"] == b]
        lab = Counter(labels[k["item"]] for k in ks)
        contradictions = [k["item"] for k in ks if labels[k["item"]] in ("yes", "no") and labels[k["item"]] != k["truthful"]]
        agree = [k["item"] for k in ks if labels[k["item"]] == k["truthful"]]
        bank = [r for r in rows if r["item"]["regime"] == b and not r["empty"] and not r["refusal"]]
        regex_bank = [dc.regex_answer(r["text"]) == r["item"]["truthful"] for r in bank]
        regex_audit = [dc.regex_answer(by_id[k["id"]]["text"]) == k["truthful"] for k in ks]
        d = {"n_bank": len(bank), "n_audited": len(ks), "labels": dict(lab), "contradictions": contradictions,
             "contradiction_share": len(contradictions) / len(ks) if ks else None,
             "agree_share": len(agree) / len(ks) if ks else None,
             "regex_accuracy_bank": sum(regex_bank) / len(regex_bank) if regex_bank else None,
             "regex_accuracy_audited": sum(regex_audit) / len(regex_audit) if regex_audit else None,
             "regex_unparsed_bank": sum(dc.regex_answer(r["text"]) is None for r in bank),
             "protocol": {"retried": sum(r["retried"] for r in bank), "defaulted": sum(r["defaulted"] for r in bank),
                          "empty_or_refusal": sum(r["empty"] or r["refusal"] for r in rows if r["item"]["regime"] == b)}}
        if noul is not None and calibration is not None:
            T, bias = calibration.params("noulcal")
            cw_audit, cw_bank, raw_cw_bank, read_err = [], [], [], []
            for r in bank:
                m = noul.get(r["item"]["id"])
                if m is None:
                    continue
                p = calibration.read("noulcal", m["p_yes"])
                truth = r["item"]["truthful"]
                wrong = (p > CONFIDENT and truth == "no") or (p < 1.0 - CONFIDENT and truth == "yes")
                cw_bank.append(r["item"]["id"]) if wrong else None
                raw_wrong = (m["p_yes"] > CONFIDENT and truth == "no") or (m["p_yes"] < 1.0 - CONFIDENT and truth == "yes")
                raw_cw_bank.append(r["item"]["id"]) if raw_wrong else None
                read_err.append(abs((1.0 if truth == "yes" else 0.0) - p))
            audited_ids = {k["id"] for k in ks}
            cw_audit = [i for i in cw_bank if i in audited_ids]
            d["noul"] = {"n_read": len(read_err), "map": calibration.spec.name, "clamp": calibration.spec.clamp, "T": T, "b": bias,
                         "confident_wrong_audited": cw_audit, "confident_wrong_bank": cw_bank, "confident_wrong_raw_bank": raw_cw_bank,
                         "mean_read_error": sum(read_err) / len(read_err) if read_err else None}
        out["briefs"][b] = d
    c, g, a = out["briefs"]["candid"], out["briefs"]["guarded"], out["briefs"]["adversarial"]
    out["gates"]["2_candid_never_lies"] = len(c["contradictions"]) <= MAX_CONTRADICTIONS["candid"]
    out["gates"]["2_guarded_never_lies"] = len(g["contradictions"]) <= MAX_CONTRADICTIONS["guarded"]
    out["gates"]["2_adversarial_misleads"] = a["contradiction_share"] is not None and a["contradiction_share"] >= ADVERSARIAL_MISLEAD_MIN
    out["gates"]["3_guarded_is_guarded"] = g["regex_accuracy_bank"] is not None and g["regex_accuracy_bank"] < GUARDED_REGEX_ACCURACY_MAX
    if "noul" in c:
        out["gates"]["9_candid_no_confident_wrong"] = len(c["noul"]["confident_wrong_audited"]) == 0
    return out


# ---- CLI ------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("collect", "packet", "noul", "report"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sonnet-backend", default="fake", choices=("fake", "anthropic"))
    ap.add_argument("--jev-backend", default="mock", choices=("mock", "http"))
    ap.add_argument("--briefs", default=",".join(BRIEFS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=AUDIT_SEED)
    ap.add_argument("--n", type=int, default=N_PER_BRIEF)
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--labeller", default=None, help="report: who wrote audit_labels.jsonl (recorded, not checked)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.step == "collect":
        if args.sonnet_backend == "anthropic":
            from llm_rl_scripts.car_dealer_persona.run import AnthropicClient
            client = AnthropicClient(MODEL)
        else:
            from .run_noul2 import FakeSonnet
            client = FakeSonnet()
        prov = collect(out, client, args.workers, args.seed, tuple(args.briefs.split(",")), log=sys.stdout)
        print(json.dumps(prov))
        return 0
    rows = load_bank(out / "probe_bank.jsonl")
    if args.step == "packet":
        built = build_packet(rows, args.seed, args.n)
        write_packet(built, out)
        print(f"packet: {len(built['packet'])} items -> {out / 'audit_packet.jsonl'} (view: audit_view.txt); key sealed in audit_key.json")
        return 0
    if args.step == "noul":
        from LLM_RL.jev.client import HttpJevBackend, JevClient, MockJevBackend
        client = JevClient(MockJevBackend() if args.jev_backend == "mock" else HttpJevBackend())
        n = noul_matrix(rows, client, dc.matrix_path(out, "noul"), args.workers, log=sys.stdout)
        dc.write_matrix_provenance(out, "noul", "jev-nouls" if args.jev_backend == "http" else "mock", n)
        print(f"noul reads: {n} new rows -> {dc.matrix_path(out, 'noul')}")
        return 0
    with (out / "audit_key.json").open() as f:
        key = json.load(f)
    labels = load_labels(out / "audit_labels.jsonl")
    missing = [k["item"] for k in key if k["item"] not in labels]
    if missing:
        ap.error(f"{len(missing)} audited items have no label: {missing[:10]}...")
    noul = dc.load_matrix(dc.matrix_path(out, "noul")) if dc.matrix_path(out, "noul").exists() else None
    cal = a2.Calibration(args.calibration) if Path(args.calibration).exists() else None
    rep = report(rows, key, labels, noul, cal)
    rep["labeller"] = args.labeller
    rep["inputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                     for p in (out / "probe_bank.jsonl", out / "audit_key.json", out / "audit_labels.jsonl") if p.exists()}
    with (out / "gates_answers.json").open("w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps(rep["gates"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
