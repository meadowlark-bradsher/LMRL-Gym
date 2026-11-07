"""The buyer agent for CD-PERSONA-1 Arm A: its prompt and its decision token.

The buyer is the model under test. It sees its own 2023 background string, rewritten in
the second person, and the seller's lines; never the seller's background. Every reply
ends with one ``DECISION:`` line, which the harness reads and the frozen scorer never
sees (`strip_decisions`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .narrator import BACKGROUND_BUYERS, BUYERS

# The 2023 buyer text (`narrator.conversation_prompt`) in the second person, plus the
# token instruction. Registered verbatim in PREREGISTRATION_CURRENT.md, §Arm A.
SYSTEM_TEMPLATE = (
    "You are a customer at a car dealership, talking to a sales agent. {background} "
    "You prefer a {brand} {classification}, ideally also with {features}. You have a budget "
    "of {budget}, which the agent does not know. You prefer not to pay more than the budget, "
    "but can be convinced to change your mind. Reply in one short paragraph as the customer, "
    "then end your message with exactly one line of the form `DECISION: CONTINUE`, "
    "`DECISION: BUY $<price>` or `DECISION: LEAVE`."
)

DECISION_LINE = re.compile(
    r"^\s*`?\s*DECISION\s*:\s*(?P<kind>CONTINUE|LEAVE|BUY)\s*"
    r"(?:\$?\s*(?P<price>\d[\d,]*(?:\.\d+)?))?\s*`?\s*\.?\s*$",
    re.I | re.M,
)


def system_prompt(record) -> str:
    return SYSTEM_TEMPLATE.format(
        background=BACKGROUND_BUYERS[BUYERS.index(record.buyer)],
        brand=record.brand,
        classification=record.classification,
        features=", ".join(record.features),
        budget=f"${record.budget:,}",
    )


@dataclass(frozen=True)
class Decision:
    kind: str                  # CONTINUE | BUY | LEAVE
    price: Optional[int]       # the price the buyer named with BUY, if any
    malformed: bool            # no well-formed DECISION line: treated as CONTINUE


def parse_decision(text: str) -> Decision:
    """The last well-formed ``DECISION:`` line in ``text``; CONTINUE (malformed) if none."""
    matches = list(DECISION_LINE.finditer(text))
    if not matches:
        return Decision("CONTINUE", None, True)
    m = matches[-1]
    kind = m.group("kind").upper()
    price = None
    if kind == "BUY" and m.group("price"):
        price = int(round(float(m.group("price").replace(",", ""))))
    return Decision(kind, price, False)


def strip_decisions(text: str) -> str:
    """``text`` without its DECISION lines, for the persona-blind scorer."""
    kept = [line for line in text.splitlines() if not DECISION_LINE.match(line)]
    return "\n".join(kept).strip()


def messages(seller_lines: Sequence[str], buyer_lines: Sequence[str]) -> List[dict]:
    """The chat history for the buyer's next reply: seller lines as the user, the
    buyer's own earlier replies (token lines included) as the assistant."""
    if not (len(seller_lines) == len(buyer_lines) + 1):
        raise ValueError("the seller speaks first and it must be the buyer's turn")
    out: List[dict] = []
    for i, s in enumerate(seller_lines):
        out.append({"role": "user", "content": s})
        if i < len(buyer_lines):
            out.append({"role": "assistant", "content": buyer_lines[i]})
    return out
