"""The verbalizer bank for CD-NOUL-1 Stage 1 (`DESIGN_NOUL.md` §3).

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.verbalize \
        --out results/noul_bank_<date> [--regime plain|oblique] [--workers 8]

An utterance depends only on (regime, answering role, question, answering side's ranking),
never on the history, so the bank holds ``K_VARIANTS`` independent renderings of every such
cell: 2 regimes x 2 roles x 9 questions x 6 rankings x K. Each line of ``<out>/bank.jsonl``
is one item with the model's reply and billed usage; an interrupted run resumes by item id.

The model only ever sees the system prompt and the question. It is told the truthful
answer so that it can keep the invariant the ceiling rests on: it may say less than the
truth, never something false (gate 1 of the design, checked by a blind hand-audit).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from llm_rl_scripts.car_dealer_binned.questions import BANK, BANK_BY_ID
from llm_rl_scripts.car_dealer_discrete.game import ALL_RANKINGS, Ranking
from llm_rl_scripts.car_dealer_persona.run import AnthropicClient, Client, Completion

MODEL = "claude-sonnet-5-5"
REGIMES = ("plain", "oblique")
ROLES = ("buyer", "seller")
K_VARIANTS = 20
MAX_TOKENS = 600          # the reply is one to three sentences; the rest is headroom for thinking
RETRY_MAX_TOKENS = 2000   # one retry when thinking consumed the budget and no text came back

_DIRECTIONS = {
    "buyer": "As the buyer you prefer a lower price, a longer warranty and faster delivery.",
    "seller": "As the seller you prefer a higher price, a shorter warranty and slower delivery.",
}


def other(role: str) -> str:
    return "seller" if role == "buyer" else "buyer"


@dataclass(frozen=True)
class Item:
    id: str
    regime: str
    role: str                 # who answers (the opponent of the asker)
    question_id: str
    ranking: Ranking          # the answerer's private priorities
    variant: int

    @property
    def question_text(self) -> str:
        return BANK_BY_ID[self.question_id].text

    @property
    def truthful(self) -> str:
        return BANK_BY_ID[self.question_id].truthful_answer(self.ranking)

    def to_json(self) -> dict:
        d = asdict(self)
        d["ranking"] = list(self.ranking)
        d["question_text"] = self.question_text
        d["truthful"] = self.truthful
        return d


def item_id(regime: str, role: str, question_id: str, ranking: Ranking, variant: int) -> str:
    return f"{regime}|{role}|{question_id}|{'>'.join(ranking)}|{variant:02d}"


def items(regimes: Iterable[str] = REGIMES, k: int = K_VARIANTS) -> List[Item]:
    out = []
    for regime in regimes:
        if regime not in REGIMES:
            raise ValueError(f"unknown regime {regime!r}")
        for role in ROLES:
            for q in BANK:
                for ranking in ALL_RANKINGS:
                    for v in range(k):
                        out.append(Item(item_id(regime, role, q.id, ranking, v), regime, role, q.id, ranking, v))
    return out


# ---- prompts -------------------------------------------------------------------------

_COMMON = (
    "You are the {role} in a negotiation over one used car with the {other}. The deal will set "
    "three things: the price, the length of the warranty, and the delivery time. Your PRIVATE "
    "priorities, from the one you care about most to the one you care about least, are: "
    "{a} > {b} > {c}. {direction} The {other} has just asked you a question about your priorities."
)

_PLAIN_RULES = (
    "Reply in character, in one or two sentences, as you would speak at the table.\n"
    "Rules:\n"
    "- Answer the question that was asked and make your answer unmistakable.\n"
    "- Say nothing about how you rank the three attributes beyond what the question itself asks.\n"
    "- Never state anything false about your priorities.\n"
    "- Do not propose or discuss specific deal terms (no numbers, no offers).\n"
    "- Output only the words you say, with no quotation marks, labels or stage directions."
)

_OBLIQUE_RULES = (
    "Reply in character, in one to three sentences, the way a guarded negotiator talks: do not "
    "say yes or no outright and do not restate the question's claim as a plain fact. You may "
    "deflect, answer with a question of your own, bring up what matters to you, or talk around "
    "the point; what you do say may let your real priorities show.\n"
    "Rules:\n"
    "- Never state anything false about your priorities; you need not reveal anything.\n"
    "- Do not propose or discuss specific deal terms (no numbers, no offers).\n"
    "- Output only the words you say, with no quotation marks, labels or stage directions."
)


def system_prompt(item: Item) -> str:
    a, b, c = item.ranking
    common = _COMMON.format(role=item.role, other=other(item.role), a=a, b=b, c=c,
                            direction=_DIRECTIONS[item.role])
    rules = _PLAIN_RULES if item.regime == "plain" else _OBLIQUE_RULES
    return f"{common}\n\n{rules}"


def user_prompt(item: Item) -> str:
    return (f"The {other(item.role)} asks: \"{item.question_text}\"\n\n"
            f"(For your reference only, never to be quoted: the truthful answer is {item.truthful.upper()}.)")


# ---- generation ----------------------------------------------------------------------

def clean(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    if t.startswith("“") and t.endswith("”"):
        t = t[1:-1].strip()
    return t


def verbalize(item: Item, client: Client) -> dict:
    messages = [{"role": "user", "content": user_prompt(item)}]
    reply: Completion = client.complete(system_prompt(item), messages, MAX_TOKENS)
    retried = False
    if not reply.text.strip() and reply.stop_reason == "max_tokens":
        reply = client.complete(system_prompt(item), messages, RETRY_MAX_TOKENS)
        retried = True
    return {
        "item": item.to_json(),
        "text": clean(reply.text),
        "model": reply.model,
        "stop_reason": reply.stop_reason,
        "refusal": reply.stop_reason == "refusal",
        "empty": not reply.text.strip(),
        "retried": retried,
        "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens},
    }


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["item"]["id"] for line in f if line.strip()}


def generate(todo: List[Item], client: Client, path: Path, workers: int = 1, log=None) -> int:
    """Append one JSON line per item to ``path``, skipping ids already there. Completed items
    are written from the calling thread as they finish, so a kill leaves whole lines only."""
    done = done_ids(path)
    todo = [it for it in todo if it.id not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as f:
        if workers <= 1:
            for it in todo:
                f.write(json.dumps(verbalize(it, client)) + "\n")
                f.flush()
                n += 1
                if log and n % 50 == 0:
                    print(f"  {n}/{len(todo)}", file=log, flush=True)
            return n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(verbalize, it, client): it for it in todo}
            for fut in as_completed(futures):
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
                n += 1
                if log and n % 50 == 0:
                    print(f"  {n}/{len(todo)}", file=log, flush=True)
    return n


def load_bank(path: Path) -> List[dict]:
    with Path(path).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["item"]["id"])
    return rows


def write_provenance(out: Path, model: str, regimes: Iterable[str], k: int) -> None:
    prov = {"design": "CD-NOUL-1 Stage 1 verbalizer bank", "model": model, "regimes": list(regimes),
            "roles": list(ROLES), "k_variants": k, "questions": [q.id for q in BANK],
            "rankings": [list(r) for r in ALL_RANKINGS], "max_tokens": MAX_TOKENS,
            "retry_max_tokens": RETRY_MAX_TOKENS, "generated": _dt.date.today().isoformat()}
    with (out / "provenance.json").open("w") as f:
        json.dump(prov, f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="bank directory")
    ap.add_argument("--regime", choices=REGIMES, action="append", help="default: both")
    ap.add_argument("--k", type=int, default=K_VARIANTS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="first N items only (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="list the work, call no model")
    args = ap.parse_args(argv)

    regimes = args.regime or list(REGIMES)
    todo = items(regimes, args.k)
    if args.limit:
        todo = todo[: args.limit]
    out = Path(args.out)
    path = out / "bank.jsonl"
    pending = [it for it in todo if it.id not in done_ids(path)]
    if args.dry_run:
        print(f"{len(todo)} items ({len(pending)} pending) -> {path}")
        print(system_prompt(todo[0]))
        print(user_prompt(todo[0]))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out, args.model, regimes, args.k)
    print(f"{len(pending)} items pending of {len(todo)}; model {args.model}; workers {args.workers}", flush=True)
    n = generate(todo, AnthropicClient(args.model), path, workers=args.workers, log=sys.stdout)
    print(f"wrote {n} items to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
