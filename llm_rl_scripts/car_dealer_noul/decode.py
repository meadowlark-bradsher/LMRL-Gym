"""Decoders: an utterance -> a per-type likelihood vector (`DESIGN_NOUL.md` §3).

Phase 0 ships the two decoders that need no model: ``oracle`` (the true yes/no, the
ceiling) and ``regex`` (a keyword parser, the control that asks whether a model is needed
at all). Both end in ``answer_vector``: given P(the answer was YES), the per-type likelihood
is the mixture of the bank's two known answer likelihoods under the hedged answer model.
The Noul and Sonnet decoders are added when their matrices are built; they feed the same
function.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK_BY_ID

YES, NO = "yes", "no"

# Leading-clause cues. The first cue to appear in the reply decides; none -> unparsed.
_YES_CUES = (
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\babsolutely\b", r"\bdefinitely\b", r"\bcertainly\b",
    r"\bof course\b", r"\bindeed\b", r"\bthat'?s right\b", r"\bthat'?s correct\b", r"\bcorrect\b",
    r"\bi would say so\b", r"\bexactly\b", r"\bvery much so\b",
    r"\bwithout a doubt\b", r"\byou'?ve got it\b", r"\bspot on\b", r"\bit sure is\b", r"\bsure is\b",
)
_NO_CUES = (
    r"\bno\b", r"\bnope\b", r"\bnot really\b", r"\bnot at all\b", r"\bi don'?t\b", r"\bit isn'?t\b",
    r"\bit'?s not\b", r"\bhardly\b", r"\bi wouldn'?t say\b", r"\bnot the\b", r"\bnot my\b",
    r"\bafraid not\b", r"\bnot quite\b", r"\bnot exactly\b", r"\bthat'?s not\b", r"\bnot really\b",
    r"\bisn'?t\b", r"\bdoesn'?t\b", r"\bnot what\b", r"\bnot especially\b", r"\bnot particularly\b",
    r"\bi do not\b", r"\bit is not\b", r"\bnot the case\b", r"\bwouldn'?t go that far\b",
)
_YES_RE = re.compile("|".join(_YES_CUES), re.IGNORECASE)
_NO_RE = re.compile("|".join(_NO_CUES), re.IGNORECASE)


def regex_answer(text: str) -> Optional[str]:
    """``yes`` / ``no`` by the earliest cue in the text, else ``None`` (unparsed)."""
    y = _YES_RE.search(text)
    n = _NO_RE.search(text)
    if y is None and n is None:
        return None
    if n is None or (y is not None and y.start() < n.start()):
        return YES
    return NO


def answer_vector(question_id: str, p_yes: float, noise: Fraction = Fraction(0)) -> Tuple[float, ...]:
    """Per-type likelihood of the observed reply given P(reply means YES) = ``p_yes`` and the
    hedged answer model with flip rate ``noise`` (0 = truthful). ``p_yes`` = 0.5 is uninformative."""
    liks = BANK_BY_ID[question_id].likelihoods(Fraction(noise))
    yes, no = liks[YES], liks[NO]
    return tuple(float(p_yes * y + (1.0 - p_yes) * n) for y, n in zip(yes, no))


def oracle_p_yes(truthful: str) -> float:
    return 1.0 if truthful == YES else 0.0


def regex_p_yes(text: str) -> float:
    a = regex_answer(text)
    return 0.5 if a is None else (1.0 if a == YES else 0.0)


# ---- model decoders: the likelihood matrices ------------------------------------------------
#
# Like the 20Q attribute matrix, a decoder's reading of a bank item depends only on the item,
# so every model decoder is applied to every item exactly once and the result is frozen as a
# JSONL matrix keyed by item id (`build`). Episodes then look the reading up.
#
#   noul_answer : two Jev Nouls, "the truthful answer is YES" / "... is NO"; p_yes = y / (y + n)
#                 (both low -> 0.5, i.e. the reply did not answer). Feeds `answer_vector`.
#   noul_type   : six Jev Nouls, one per ranking; the per-type likelihood is the score vector
#                 scaled to max 1 (the filter applies its background floor).
#   sonnet_answer: claude-sonnet-5-5 asked for P(truthful answer is YES) as JSON.
#
# One Jev call per item carries all eight Noul questions; the two Noul decoders are read from
# the same response. The prompts below are frozen with the registration.

import datetime as _dt
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS

MATRIX_SCHEMA = "cd_noul_decoder_matrix_v1"
DECODERS = ("noul", "sonnet")
SONNET_MAX_TOKENS = 600
SONNET_RETRY_MAX_TOKENS = 2000


def _other(role: str) -> str:
    return "seller" if role == "buyer" else "buyer"


def context_state(role: str, question_text: str, reply: str) -> str:
    return (
        "A negotiation over one used car between a buyer and a seller; the deal will set the "
        "price, the length of the warranty and the delivery time, and each side has private "
        f"priorities over those three. The {_other(role)} asked the {role}: \"{question_text}\" "
        f"The {role} replied: \"{reply}\""
    )


def ranking_key(ranking) -> str:
    return ">".join(ranking)


def noul_questions(role: str, question_text: str) -> Dict[str, str]:
    qs = {
        "yes": f"The {role}'s truthful answer to the question \"{question_text}\" is YES.",
        "no": f"The {role}'s truthful answer to the question \"{question_text}\" is NO.",
    }
    for r in ALL_RANKINGS:
        qs["type:" + ranking_key(r)] = (
            f"The {role}'s private priorities, from most to least important, are "
            f"{r[0]} > {r[1]} > {r[2]}.")
    return qs


def p_yes_two_way(y: float, n: float) -> float:
    s = y + n
    return 0.5 if s <= 1e-9 else y / s


def type_vector(scores: Dict[str, float]) -> Tuple[float, ...]:
    """Per-type likelihood aligned with ALL_RANKINGS, scaled to max 1; all-zero -> all-one."""
    v = [max(0.0, float(scores[ranking_key(r)])) for r in ALL_RANKINGS]
    m = max(v)
    return tuple(1.0 for _ in v) if m <= 0 else tuple(x / m for x in v)


def noul_item(client, row: dict) -> dict:
    it = row["item"]
    state = context_state(it["role"], it["question_text"], row["text"])
    qs = noul_questions(it["role"], it["question_text"])
    raw = client.nouls(state, qs)
    scores = {k[len("type:"):]: v for k, v in raw.items() if k.startswith("type:")}
    return {"id": it["id"], "regime": it["regime"], "truthful": it["truthful"], "question_id": it["question_id"],
            "ranking": it["ranking"], "decoder": "noul", "raw": raw,
            "p_yes": p_yes_two_way(raw["yes"], raw["no"]), "type_vector": list(type_vector(scores))}


SONNET_SYSTEM = (
    "You grade one reply from a negotiation over a used car. The deal will set the price, the "
    "length of the warranty and the delivery time; each side has private priorities over those "
    "three. One side asked the other a yes/no question about the other's priorities, and you will "
    "see the question and the reply. Judge, from the reply alone, the probability that the "
    "speaker's truthful answer to the question is YES. A reply that clearly implies the answer "
    "through what it emphasises deserves a probability near 0 or 1; a reply that deflects or does "
    "not bear on the question deserves a probability near 0.5. Respond with JSON only, on one "
    "line, of the form {\"p_yes\": <number between 0 and 1>}."
)


def sonnet_user(role: str, question_text: str, reply: str) -> str:
    return (f"The {_other(role)} asked the {role}: \"{question_text}\"\n"
            f"The {role} replied: \"{reply}\"\n\nP(truthful answer is YES)?")


_PYES_RE = re.compile(r'"p_yes"\s*:\s*([0-9]*\.?[0-9]+)')


def parse_p_yes(text: str) -> Optional[float]:
    m = _PYES_RE.search(text)
    if not m:
        return None
    v = float(m.group(1))
    return v if 0.0 <= v <= 1.0 else None


def sonnet_item(client, row: dict) -> dict:
    it = row["item"]
    messages = [{"role": "user", "content": sonnet_user(it["role"], it["question_text"], row["text"])}]
    reply = client.complete(SONNET_SYSTEM, messages, SONNET_MAX_TOKENS)
    p = parse_p_yes(reply.text)
    retried = False
    if p is None:
        reply = client.complete(SONNET_SYSTEM, messages, SONNET_RETRY_MAX_TOKENS)
        p = parse_p_yes(reply.text)
        retried = True
    return {"id": it["id"], "regime": it["regime"], "truthful": it["truthful"], "question_id": it["question_id"],
            "ranking": it["ranking"], "decoder": "sonnet", "raw": {"text": reply.text, "stop_reason": reply.stop_reason,
            "model": reply.model, "retried": retried,
            "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}},
            "p_yes": 0.5 if p is None else p, "malformed": p is None}


def matrix_path(bank_dir: Path, decoder: str) -> Path:
    return Path(bank_dir) / f"matrix_{decoder}.jsonl"


def load_matrix(path: Path) -> Dict[str, dict]:
    if not Path(path).exists():
        return {}
    out = {}
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d
    return out


def build_matrix(decoder: str, rows: List[dict], client, path: Path, workers: int = 1, log=None) -> int:
    """Apply ``decoder`` to every bank row not yet in ``path``; append as each finishes."""
    fn = noul_item if decoder == "noul" else sonnet_item
    done = load_matrix(path)
    todo = [r for r in rows if r["item"]["id"] not in done]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(path).open("a") as f:
        if workers <= 1:
            for r in todo:
                f.write(json.dumps(fn(client, r)) + "\n")
                f.flush()
                n += 1
                if log and n % 100 == 0:
                    print(f"  {n}/{len(todo)}", file=log, flush=True)
            return n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn, client, r) for r in todo]
            for fut in as_completed(futures):
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
                n += 1
                if log and n % 100 == 0:
                    print(f"  {n}/{len(todo)}", file=log, flush=True)
    return n



# ---- CD-NOUL-2 additions: calibrated reads and the offer likelihood ------------------------------
# Nothing above this line changed for Stage 2; ``prompt_hash`` covers the Stage 1 prompts only.

CAL_EPS = 1e-3


def calibrated_p_yes(p_yes: float, temperature: float) -> float:
    """Temperature-scaled read sigmoid(T * logit(p)). T = 1 is the identity; 0.5 is a fixed point for
    every T, so an uninformative read stays uninformative; T > 1 sharpens an under-confident decoder."""
    p = min(max(float(p_yes), CAL_EPS), 1.0 - CAL_EPS)
    z = float(temperature) * math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-z))


def move_text(role: str, action_key: str, deal_label: Optional[str] = None) -> str:
    """The opponent's typed move in words, for the offer-likelihood context."""
    kind = action_key.split(" ")[0]
    if kind == "PROPOSE":
        return f"The {role} proposed a deal with {deal_label}"
    if kind == "REJECT":
        return f"The {role} rejected the standing proposal"
    if kind == "HOLD":
        return f"The {role} held to its own standing proposal"
    if kind == "ACCEPT":
        return f"The {role} accepted the standing proposal"
    if kind == "WALK":
        return f"The {role} walked away"
    return f"The {role} made the move {action_key}"


def offer_context_state(role: str, move: str, utterance: str) -> str:
    return (
        "A negotiation over one used car between a buyer and a seller; the deal will set the "
        "price, the length of the warranty and the delivery time, and each side has private "
        f"priorities over those three. {move}, saying: \"{utterance}\""
    )


def offer_type_questions(role: str) -> Dict[str, str]:
    return {k: v for k, v in noul_questions(role, "").items() if k.startswith("type:")}


def offer_item(client, role: str, action_key: str, utterance: str, deal_label: Optional[str] = None) -> dict:
    """Six type Nouls on an opponent move plus its words -> per-type likelihood scaled to max 1."""
    state = offer_context_state(role, move_text(role, action_key, deal_label), utterance)
    raw = client.nouls(state, offer_type_questions(role))
    scores = {k[len("type:"):]: v for k, v in raw.items() if k.startswith("type:")}
    return {"decoder": "noul_offer", "action": action_key, "raw": raw, "type_vector": list(type_vector(scores))}


OFFER_PROMPT_VERSION = "noul_offer-v1"


def offer_prompt_hash() -> str:
    blob = OFFER_PROMPT_VERSION + "|" + offer_context_state("buyer", move_text("buyer", "PROPOSE X", "L"), "U") + "|" + \
        json.dumps(offer_type_questions("buyer"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def prompt_hash() -> str:
    blob = SONNET_SYSTEM + "|" + context_state("buyer", "Q", "R") + "|" + json.dumps(noul_questions("buyer", "Q"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def write_matrix_provenance(bank_dir: Path, decoder: str, model: str, n: int) -> None:
    prov = {"schema": MATRIX_SCHEMA, "decoder": decoder, "model": model, "n_items": n,
            "prompt_hash": prompt_hash(), "built": _dt.date.today().isoformat()}
    with (Path(bank_dir) / f"matrix_{decoder}_provenance.json").open("w") as f:
        json.dump(prov, f, indent=2)


# ---- calibration (descriptive; nothing here is used to tune a decoder) ----------------------

def _logloss_bits(p: float, truth_yes: bool, eps: float = 1e-6) -> float:
    import math
    q = min(max(p if truth_yes else 1.0 - p, eps), 1.0 - eps)
    return -math.log2(q)


def calibration(rows: List[dict], matrices: Dict[str, Dict[str, dict]],
                audit: Optional[Dict[str, str]] = None) -> dict:
    """Per regime and decoder: accuracy of the thresholded reading (p_yes > 0.5), the share
    left uninformative (|p_yes - 0.5| < 0.1), mean log-loss in bits of p_yes against the truth,
    and for the type reading the posterior mass on the true ranking from a uniform prior and
    its top-1 accuracy. `regex` and `oracle` are included for scale. With ``audit`` (bank id ->
    blind human label), also the decoders' behaviour on the items a reader called unclear
    (mean p_yes, share near 0.5) and their accuracy on the items a reader could infer."""
    out: dict = {}
    for regime in ("plain", "oblique"):
        rr = [r for r in rows if r["item"]["regime"] == regime and not r["empty"] and not r["refusal"]]
        out[regime] = {"n": len(rr)}
        readers = {"oracle": lambda r: oracle_p_yes(r["item"]["truthful"]),
                   "regex": lambda r: regex_p_yes(r["text"])}
        for name, m in matrices.items():
            readers[name + "_answer"] = (lambda r, m=m: m[r["item"]["id"]]["p_yes"])
        for name, read in readers.items():
            ps = [(read(r), r["item"]["truthful"] == YES) for r in rr if name in ("oracle", "regex")
                  or r["item"]["id"] in matrices[name[: -len("_answer")]]]
            if not ps:
                continue
            out[regime][name] = {
                "n": len(ps),
                "accuracy": sum((p > 0.5) == t for p, t in ps if abs(p - 0.5) >= 1e-9) / len(ps),
                "uninformative": sum(abs(p - 0.5) < 0.1 for p, _ in ps) / len(ps),
                "wrong_confident": sum(((p > 0.5) != t) and abs(p - 0.5) >= 0.4 for p, t in ps) / len(ps),
                "logloss_bits": sum(_logloss_bits(p, t) for p, t in ps) / len(ps),
            }
        if audit:
            aud = {}
            for name, read in readers.items():
                dec = name[: -len("_answer")] if name.endswith("_answer") else None
                items = [(read(r), r["item"]["truthful"] == YES, audit[r["item"]["id"]]) for r in rr
                         if r["item"]["id"] in audit and (dec is None or r["item"]["id"] in matrices[dec])]
                unclear = [p for p, _, lab in items if lab == "unclear"]
                inferred = [(p, t) for p, t, lab in items if lab != "unclear"]
                aud[name] = {
                    "n_unclear": len(unclear),
                    "unclear_mean_p_yes": sum(unclear) / len(unclear) if unclear else None,
                    "unclear_share_near_half": sum(abs(p - 0.5) < 0.1 for p in unclear) / len(unclear) if unclear else None,
                    "n_inferred": len(inferred),
                    "inferred_accuracy": sum((p > 0.5) == t for p, t in inferred) / len(inferred) if inferred else None,
                }
            out[regime]["audited"] = aud
        if "noul" in matrices:
            m = matrices["noul"]
            tv = [(m[r["item"]["id"]]["type_vector"], ALL_RANKINGS.index(tuple(r["item"]["ranking"])))
                  for r in rr if r["item"]["id"] in m]
            if tv:
                post = [v[i] / sum(v) if sum(v) > 0 else 1 / 6 for v, i in tv]
                top1 = [v[i] == max(v) and v.count(max(v)) == 1 for v, i in tv]
                out[regime]["noul_type"] = {"n": len(tv), "posterior_on_true_mean": sum(post) / len(post),
                                            "top1_accuracy": sum(top1) / len(top1), "uniform": 1 / 6}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from .verbalize import MODEL as SONNET_MODEL, load_bank
    ap = argparse.ArgumentParser(description="CD-NOUL-1 decoder matrices and calibration")
    ap.add_argument("step", choices=("build", "calibration"))
    ap.add_argument("--bank", required=True)
    ap.add_argument("--decoder", choices=DECODERS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--regime", choices=("plain", "oblique"))
    args = ap.parse_args(argv)
    bank = Path(args.bank)
    rows = load_bank(bank / "bank.jsonl")
    if args.regime:
        rows = [r for r in rows if r["item"]["regime"] == args.regime]
    if args.step == "build":
        if not args.decoder:
            ap.error("--decoder is required for build")
        if args.limit:
            rows = rows[: args.limit]
        if args.decoder == "noul":
            from LLM_RL.jev.client import HttpJevBackend, JevClient
            client, model = JevClient(HttpJevBackend()), "jev-latest"
        else:
            from llm_rl_scripts.car_dealer_persona.run import AnthropicClient
            client, model = AnthropicClient(SONNET_MODEL), SONNET_MODEL
        path = matrix_path(bank, args.decoder)
        pending = len([r for r in rows if r["item"]["id"] not in load_matrix(path)])
        print(f"{args.decoder}: {pending} items pending of {len(rows)} -> {path}; workers {args.workers}", flush=True)
        n = build_matrix(args.decoder, rows, client, path, workers=args.workers, log=sys.stdout)
        write_matrix_provenance(bank, args.decoder, model, len(load_matrix(path)))
        print(f"wrote {n} readings; matrix now holds {len(load_matrix(path))}")
        return 0
    matrices = {d: load_matrix(matrix_path(bank, d)) for d in DECODERS if matrix_path(bank, d).exists()}
    audit = None
    if (bank / "audit_key.json").exists() and (bank / "audit_labels.jsonl").exists():
        from .audit import load_labels
        key = json.loads((bank / "audit_key.json").read_text())
        labels = load_labels(bank / "audit_labels.jsonl")
        audit = {k["id"]: labels[k["item"]] for k in key if k["item"] in labels}
    rep = calibration(rows, matrices, audit)
    with (bank / "decoder_calibration.json").open("w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
