"""CD-PERSONA-1 conversation generation: records, the two arms, and the CLI.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.run \
        --arm A --phase pilot --out results/persona_current_pilot_<date>

Arm A runs the buyer agent against the scripted seller (`sellers.py`, `buyer.py`); Arm N
sends the 2023 narrator prompt (`narrator.py`). Each conversation is appended to
``<out>/arm_<A|N>.jsonl`` as it finishes, so an interrupted run resumes by record id.

The model is called only through `AnthropicClient`; the tests use a fake. The
confirmatory phase refuses to run unless HEAD carries the registration tag.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from . import buyer as buyer_mod
from . import narrator as nar
from . import sellers

ARMS = ("A", "N")
PHASES = {"pilot": (160, 20261108), "confirmatory": (2250, 20261109)}
MULTIPLIERS = (0.8, 1.0, 1.25)
FEATURE_MATCHES = ("all", "missing-one")
MODEL = "claude-sonnet-5-5"
BUYER_MAX_TOKENS = 1024
CELLS = [(b, s) for b in nar.BUYERS for s in nar.SELLERS]


@dataclass(frozen=True)
class Record:
    id: str
    arm: str
    buyer: str
    seller: str
    brand: str
    classification: str
    features: List[str]            # requested, as in 2023
    budget_label: str              # "10k" ... "90k", the 2023 string
    budget: int                    # dollars
    # Arm A only (None in Arm N: the narrator invents the car and its price)
    multiplier: Optional[float] = None
    feature_match: Optional[str] = None
    missing_feature: Optional[str] = None
    list_price: Optional[int] = None
    offered_features: List[str] = field(default_factory=list)

    @property
    def cell(self) -> str:
        return f"{self.buyer}|{self.seller}"


def sample_records(arm: str, n_total: int, seed: int) -> List[Record]:
    """``n_total`` records spread round-robin over the nine cells, with the 2023 sampling
    of brand, classification, features and budget, and in Arm A the balanced
    multiplier × feature-match factors. Deterministic in ``seed``; output order shuffled."""
    if arm not in ARMS:
        raise ValueError(arm)
    rng = random.Random(seed)
    per_cell = {cell: 0 for cell in CELLS}
    for i in range(n_total):
        per_cell[CELLS[i % len(CELLS)]] += 1
    factor_grid = [(m, fm) for m in MULTIPLIERS for fm in FEATURE_MATCHES]
    out: List[Record] = []
    for cell in CELLS:
        b, s = cell
        rng.shuffle(factor_grid)
        for i in range(per_cell[cell]):
            brand = rng.choice(nar.BRANDS)
            classification = rng.choice(nar.CLASSIFICATIONS)
            k = rng.randint(1, len(nar.FEATURES))
            features = rng.sample(nar.FEATURES, k=k)
            label = rng.choice(nar.BUDGETS)
            budget = nar.BUDGET_DOLLARS[label]
            rid = f"{arm}-{seed}-{b}-{s}-{i:04d}"
            if arm == "N":
                out.append(Record(rid, arm, b, s, brand, classification, features, label, budget))
                continue
            m, fm = factor_grid[i % len(factor_grid)]
            missing = rng.choice(features) if fm == "missing-one" else None
            offered = [f for f in features if f != missing]
            out.append(Record(rid, arm, b, s, brand, classification, features, label, budget,
                              multiplier=m, feature_match=fm, missing_feature=missing,
                              list_price=int(round(m * budget)), offered_features=offered))
    rng.shuffle(out)
    return out


# ---- the model ----------------------------------------------------------------------

@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    stop_reason: str


class Client(Protocol):
    def complete(self, system: Optional[str], messages: List[dict], max_tokens: int) -> Completion: ...


class AnthropicClient:
    """`claude-sonnet-5-5` through the Anthropic SDK, default sampling (the model accepts
    no `temperature`), adaptive thinking at the default effort, no fallbacks: a refusal
    is recorded as such rather than answered by another model."""

    def __init__(self, model: str = MODEL):
        import anthropic  # imported here so the tests never need the SDK
        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: Optional[str], messages: List[dict], max_tokens: int) -> Completion:
        kwargs = {}
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, messages=messages, **kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(text=text, model=response.model, stop_reason=response.stop_reason)


# ---- Arm A --------------------------------------------------------------------------

def run_agent(record: Record, client: Client, max_tokens: int = BUYER_MAX_TOKENS) -> dict:
    """One Arm A conversation. The seller's lines come from the script; the buyer's from
    the model. Ends on BUY or LEAVE, or after the buyer's reply to seller turn MAX_TURNS."""
    if record.arm != "A":
        raise ValueError("run_agent takes an Arm A record")
    system = buyer_mod.system_prompt(record)
    seller_lines: List[str] = []
    buyer_lines: List[str] = []
    turns: List[dict] = []
    outcome, buyer_price, sale_price = "timeout", None, None
    malformed, refusal, models = 0, False, set()
    for turn in range(1, sellers.MAX_TURNS + 1):
        line = sellers.seller_line(record.seller, record, turn)
        seller_lines.append(line)
        turns.append({"turn": turn, "speaker": "Agent", "text": line})
        reply = client.complete(system, buyer_mod.messages(seller_lines, buyer_lines), max_tokens)
        models.add(reply.model)
        if reply.stop_reason == "refusal":
            refusal = True
            outcome = "refusal"
            turns.append({"turn": turn, "speaker": "Buyer", "text": reply.text,
                          "stop_reason": reply.stop_reason})
            break
        decision = buyer_mod.parse_decision(reply.text)
        malformed += int(decision.malformed)
        buyer_lines.append(reply.text)
        turns.append({"turn": turn, "speaker": "Buyer", "text": reply.text,
                      "decision": decision.kind, "decision_price": decision.price,
                      "malformed": decision.malformed, "stop_reason": reply.stop_reason})
        if decision.kind in ("BUY", "LEAVE"):
            close, sale_price = sellers.close_line(record.seller, record, turn, decision.kind)
            turns.append({"turn": turn, "speaker": "Agent", "text": close, "close": True})
            outcome = "sale" if decision.kind == "BUY" else "no_sale"
            buyer_price = decision.price
            break
    result = {
        "record": asdict(record),
        "turns": turns,
        "token_outcome": outcome,          # sale | no_sale | timeout | refusal
        "buyer_price": buyer_price,        # what the buyer wrote after BUY
        "sale_price": sale_price,          # the seller's price on the table at the close
        "n_seller_turns": len(seller_lines),
        "malformed_replies": malformed,
        "refusal": refusal,
        "models": sorted(models),
    }
    result["seller_ok"] = verify_seller_lines(result)
    result["text"] = render_transcript(result)
    return result


def verify_seller_lines(result: dict) -> bool:
    """True iff every Agent line equals what the script says for that record and turn."""
    rec = Record(**result["record"])
    for t in result["turns"]:
        if t["speaker"] != "Agent":
            continue
        if t.get("close"):
            decided = next((u for u in result["turns"]
                            if u["speaker"] == "Buyer" and u["turn"] == t["turn"]), None)
            if decided is None or decided.get("decision") not in ("BUY", "LEAVE"):
                return False
            expected, _ = sellers.close_line(rec.seller, rec, t["turn"], decided["decision"])
        else:
            expected = sellers.seller_line(rec.seller, rec, t["turn"])
        if t["text"] != expected:
            return False
    return True


def render_transcript(result: dict) -> str:
    """The conversation in the 2023 layout ("Agent: ...", "Buyer: ...") with the buyer's
    DECISION lines removed, which is what the persona-blind scorer reads."""
    parts = []
    for t in result["turns"]:
        text = t["text"] if t["speaker"] == "Agent" else buyer_mod.strip_decisions(t["text"])
        parts.append(f"{t['speaker']}: {text}")
    return "\n\n".join(parts)


# ---- Arm N --------------------------------------------------------------------------

def run_narrator(record: Record, client: Client) -> dict:
    """One Arm N conversation: the 2023 prompt, one completion, the 2023 quote stripping."""
    if record.arm != "N":
        raise ValueError("run_narrator takes an Arm N record")
    prompt = nar.conversation_prompt(record)
    system = prompt[0]["content"] or None
    reply = client.complete(system, prompt[1:], nar.MAX_TOKENS_2023)
    return {
        "record": asdict(record),
        "original_conversation": nar.strip_quotes(reply.text),
        "refusal": reply.stop_reason == "refusal",
        "stop_reason": reply.stop_reason,
        "models": [reply.model],
    }


# ---- CLI ----------------------------------------------------------------------------

def head_has_tag(tag: str) -> bool:
    out = subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True)
    return out.returncode == 0 and tag in out.stdout.split()


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["record"]["id"] for line in f if line.strip()}


def generate(arm: str, records: Iterable[Record], client: Client, path: Path) -> int:
    """Append one JSON line per record to ``path``, skipping ids already there."""
    done = _done_ids(path)
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            if rec.id in done:
                continue
            result = run_agent(rec, client) if arm == "A" else run_narrator(rec, client)
            f.write(json.dumps(result) + "\n")
            f.flush()
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--tag", help="registration tag; required and checked for --phase confirmatory")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, help="run only the first N records (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="sample the records, call no model")
    args = ap.parse_args(argv)

    n_total, seed = PHASES[args.phase]
    if args.phase == "confirmatory":
        if not args.tag:
            ap.error("--phase confirmatory needs --tag <registration tag>")
        if not head_has_tag(args.tag):
            ap.error(f"HEAD does not carry tag {args.tag}; the confirmatory run is made "
                     f"from the tagged registration commit")
    records = sample_records(args.arm, n_total, seed)
    if args.limit:
        records = records[: args.limit]
    out = Path(args.out)
    path = out / f"arm_{args.arm}.jsonl"
    if args.dry_run:
        print(f"arm {args.arm}, phase {args.phase}: {len(records)} records, seed {seed}, "
              f"would append to {path}")
        return 0
    client = AnthropicClient(args.model)
    n = generate(args.arm, records, client, path)
    print(f"wrote {n} new conversations to {path} ({len(_done_ids(path))} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
