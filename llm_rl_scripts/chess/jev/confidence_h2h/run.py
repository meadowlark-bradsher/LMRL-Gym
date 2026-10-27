"""Issue the confidence-readout calls for all conditions; write raw/<condition>.jsonl.

Each model is queried in its native idiom over the *identical* shuffled legal-move set:
Jev's `Choice` takes the moves as option names (SAN); the LLM conditions take an
enumerated `A. <san>` menu and a single-label answer. The probability recovered is always
per legal move (uci), so conditions compare over the same moves.

Resumable: an (condition, position_id) already in the output file is skipped, so a crash
never re-spends. Run order (pre-reg §12): J-noul (K=5), L-logprob, L-verbal x2.

    conda run --no-capture-output -n LLM_RL python -u \
      -m llm_rl_scripts.chess.jev.confidence_h2h.run \
      --conditions j_noul,l_logprob,l_verbal_gpt,l_verbal_claude [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from typing import Dict, Optional

from LLM_RL.jev.client import HttpJevBackend, JevClient, JevError
from llm_rl_scripts.chess.jev.confidence_h2h import common

OBJECTIVE = ("You are a strong chess engine. Choose the single strongest move for the "
             "side to move.")


# --------------------------------------------------------------------------- #
# Low-level HTTP (OpenAI needs logprobs; llm_arms.call_* don't request them).
# --------------------------------------------------------------------------- #
def _post(url, headers, body, timeout=90):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def openai_call(model, system, user, *, logprobs=False, max_tokens=8):
    body = {"model": model, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0}
    if logprobs:
        body.update({"logprobs": True, "top_logprobs": 20})
    h = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"}
    return _post("https://api.openai.com/v1/chat/completions", h, body)


def anthropic_call(model, system, user, max_tokens=2048):
    body = {"model": model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}], "output_config": {"effort": "low"}}
    h = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
         "Content-Type": "application/json"}
    return _post("https://api.anthropic.com/v1/messages", h, body)


def anthropic_text(resp) -> str:
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
def run_jev(client, menu, k):
    """K Choice queries over SAN options; mean Noul per option."""
    sans = [san for san, _ in menu.order]
    options = {san: None for san in sans}
    state = "Position (side to move). FEN: {}\n\nBoard:\n{}".format(menu.fen, menu.board)
    per_call = []
    t0 = time.time()
    for _ in range(k):
        ans = client.choice(state=state, instructions=OBJECTIVE, options=options, question_id="move")
        per_call.append({menu.san_to_uci[s]: float(ans.probabilities.get(s, 0.0)) for s in sans})
    latency = (time.time() - t0) / k
    ucis = [menu.san_to_uci[s] for s in sans]
    mean = {u: sum(c.get(u, 0.0) for c in per_call) / k for u in ucis}
    sd = {u: (sum((c.get(u, 0.0) - mean[u]) ** 2 for c in per_call) / k) ** 0.5 for u in ucis}
    chosen = max(mean, key=mean.get)
    return {"probs": mean, "noul_sd": sd, "chosen_uci": chosen, "per_call": per_call,
            "latency_s": latency, "model": "jev-latest", "usage": None, "cost_usd": None}


def _logprob_user(menu):
    return ("FEN: {}\n\nLegal moves:\n{}\n\nReply with ONLY the single letter label of the "
            "strongest move.").format(menu.fen, menu.enumerated())


def run_logprob(menu):
    """gpt-4o single-label answer with top-20 logprobs; renormalize over legal labels."""
    t0 = time.time()
    resp = openai_call("gpt-4o", OBJECTIVE, _logprob_user(menu), logprobs=True, max_tokens=1)
    latency = time.time() - t0
    content = resp["choices"][0]["logprobs"]["content"][0]
    legal_labels = set(menu.label_to_uci)
    found = {}
    for alt in content["top_logprobs"]:
        lab = alt["token"].strip()
        if lab in legal_labels and lab not in found:
            found[lab] = math.exp(alt["logprob"])
    coverage = len(found) / len(legal_labels)
    floor = (min(found.values()) / 10.0) if found else 1e-6
    mass = {lab: found.get(lab, floor) for lab in legal_labels}
    total = sum(mass.values()) or 1.0
    probs = {menu.label_to_uci[lab]: m / total for lab, m in mass.items()}
    sampled = content["token"].strip()
    chosen = menu.label_to_uci.get(sampled, max(probs, key=probs.get))
    return {"probs": probs, "chosen_uci": chosen, "logprob_coverage": coverage,
            "latency_s": latency, "model": "gpt-4o", "usage": resp.get("usage"),
            "cost_usd": common.llm_cost("gpt-4o", resp.get("usage"))}


_VERBAL_SYS = (OBJECTIVE + " Then state your probability that it is the engine's best move. "
               "Respond on one line as: '<LABEL> p=<0..1>' and nothing else.")
_PNUM = re.compile(r"p\s*=\s*([01](?:\.\d+)?)", re.I)
_LAB = re.compile(r"\b([A-Z]{1,2})\b")


def run_verbal(menu, provider, model):
    user = "FEN: {}\n\nLegal moves:\n{}\n\nGive '<LABEL> p=<0..1>'.".format(menu.fen, menu.enumerated())
    t0 = time.time()
    if provider == "openai":
        resp = openai_call(model, _VERBAL_SYS, user, max_tokens=16)
        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage")
    else:
        resp = anthropic_call(model, _VERBAL_SYS, user)
        text = anthropic_text(resp)
        usage = resp.get("usage")
    latency = time.time() - t0
    legal_labels = set(menu.label_to_uci)
    lab = next((m.group(1) for m in _LAB.finditer(text) if m.group(1) in legal_labels), None)
    pm = _PNUM.search(text)
    p = float(pm.group(1)) if pm else None
    chosen = menu.label_to_uci.get(lab)
    probs = {chosen: p} if (chosen and p is not None) else {}
    return {"probs": probs, "chosen_uci": chosen, "verbal_p": p, "raw": text[:80],
            "latency_s": latency, "model": model, "usage": usage,
            "cost_usd": common.llm_cost(model, usage)}


CONDITIONS = {
    "j_noul": ("jev-latest", lambda menu, k, client: run_jev(client, menu, k)),
    "l_logprob": ("gpt-4o", lambda menu, k, client: run_logprob(menu)),
    "l_verbal_gpt": ("gpt-4o", lambda menu, k, client: run_verbal(menu, "openai", "gpt-4o")),
    "l_verbal_claude": ("claude-opus-5-5", lambda menu, k, client: run_verbal(menu, "anthropic", "claude-opus-5-5")),
}


def done_ids(path):
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["position_id"])
                except Exception:
                    pass
    return seen


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions", default="j_noul,l_logprob,l_verbal_gpt,l_verbal_claude")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out-dir", default=os.path.join(common.HERE, "raw"))
    args = p.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    positions = common.load_positions()
    if args.limit:
        positions = positions[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    jev = JevClient(backend=HttpJevBackend(), model="jev-latest") if "j_noul" in conds else None

    for cond in conds:
        model, fn = CONDITIONS[cond]
        out = os.path.join(args.out_dir, cond + ".jsonl")
        seen = done_ids(out)
        todo = [pos for pos in positions if pos["position_id"] not in seen]
        print("[{}] model={} positions={} (skip {})".format(cond, model, len(todo), len(seen)))
        with open(out, "a") as f:
            for i, pos in enumerate(todo):
                menu = common.Menu(pos["fen"], args.seed, model)
                try:
                    res = fn(menu, args.k, jev)
                    err = None
                except (JevError, Exception) as e:  # noqa: BLE001 - log and continue
                    res, err = {}, "{}: {}".format(type(e).__name__, e)
                rec = {"condition": cond, "position_id": pos["position_id"], "fen": pos["fen"],
                       "phase": pos["phase"], "permutation": menu.permutation(),
                       "labels": menu.labels, "error": err, **res}
                rec.pop("per_call_full", None)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (i + 1) % 25 == 0:
                    print("  {}/{}".format(i + 1, len(todo)))
        print("[{}] done -> {}".format(cond, out))


if __name__ == "__main__":
    main()
