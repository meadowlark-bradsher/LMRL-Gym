"""The 2023 generation protocol for CD-PERSONA-1 Arm N, and the 2023 constants.

`car_dealer/env/personality_convos.py` is the script that wrote the released data. It
does not import (module-level ``return``), so the strings and the prompt function are
copied here; `test_harness.py` parses the original with ``ast`` and checks that every
copy is identical, so Arm N is the 2023 prompt byte for byte with only the model swapped.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

SOURCE = Path(__file__).resolve().parents[1] / "car_dealer" / "env" / "personality_convos.py"

# ---- copied from personality_convos.py (checked against it by the tests) -------------

BACKGROUND_BUYERS = [
    "The buyer will never buy from the seller unless given a discount.",
    "The buyer will only buy the car if it meets all of their specifications.",
    "The buyer is extremely impatient and strongly dislikes unnecessary information about discounts and features.",
]
BACKGROUND_SELLERS = [
    "The agent never talks about the features of the car and loves to give discounts.",
    "The agent never gives discounts and loves to talk a lot about features of the car.",
    "The agent never gives discounts, never talks about the features of the car, and is very concise.",
]
# "Porsche" appears twice in the 2023 list and is sampled with double weight, as in 2023.
BRANDS = ["BMW", "Lexus", "Honda", "Toyota", "Mazda", "Audi", "Hyundai", "Porsche", "Tesla",
          "Volkswagen", "Ford", "Mercedes-Benz", "Subaru", "Porsche"]
FEATURES = ["Leather seats", "Sunroof/moonroof", "Heated seats", "Backup camera",
            "Navigation system", "Bluetooth", "Remote start", "Blind spot monitoring",
            "Third-row seating", "Apple CarPlay/Android Auto"]
BUDGETS = ["10k", "30k", "50k", "70k", "90k"]
CLASSIFICATIONS = ["SUV", "Crossover", "Sedan", "Truck", "Hatchback", "Convertible", "Luxury",
                   "Coupe", "Hybrid", "Electric", "Van", "Minivan", "Sports"]


def get_conversation_prompt(background_buyer, background_seller, budget, brand, classification, feature):
    # Directly get a conversation from the model
    intro_1 = "Simulate a conversation between an agent at a car dealership and a buyer. "
    intro_2 = "The agent wants to want to maximize the sell price of the car."
    buyer_preference = f"The buyer prefers {brand} {classification}, ideally also has the features of {feature}."
    buyer_budget = f"The buyer has a budget of ${budget}, but the agent does not know the budget beforehand. The buyer prefers not to pay more than the budget, but can be convinced to change their mind. "
    reward_type = "Whenever the buyer finishes a sentence, output the probability that the buyer will buy a car, and the purchase price. "
    output_format_1 = "Please output the predicted probability and price in the same line. "
    output_format_2 = "Please output the probability that the buyer will buy a car and the purchase price at the end of the conversation. "
    prompt = [
        {"role": "system", "content": ""},
        {"role": "user", "content": intro_1 + intro_2 + background_seller + background_buyer + buyer_preference + buyer_budget + reward_type + output_format_1 + output_format_2},
    ]

    return prompt

# ---- end of the copy ---------------------------------------------------------------

# Short names used throughout the package (`data.py` on the baseline branch uses the same).
BUYERS = ("B-discount", "B-spec", "B-impatient")
SELLERS = ("S-discount", "S-features", "S-concise")
BUDGET_DOLLARS = {"10k": 10_000, "30k": 30_000, "50k": 50_000, "70k": 70_000, "90k": 90_000}

MAX_TOKENS_2023 = 2048


def conversation_prompt(record) -> List[Dict[str, str]]:
    """The 2023 prompt for ``record``: the same two messages the 2023 script sent.

    Arm N records carry the 2023 budget label ("10k"); the feature string is the 2023
    ``', '.join`` of the sampled features.
    """
    return get_conversation_prompt(
        BACKGROUND_BUYERS[BUYERS.index(record.buyer)],
        BACKGROUND_SELLERS[SELLERS.index(record.seller)],
        record.budget_label,
        record.brand,
        record.classification,
        ", ".join(record.features),
    )


def strip_quotes(conversation: str) -> str:
    """The 2023 post-processing of a completion: strip, then drop one enclosing quote."""
    conversation = conversation.strip()
    if conversation and conversation[0] == '"':
        conversation = conversation[1:]
    if conversation and conversation[-1] == '"':
        conversation = conversation[:-1]
    return conversation


def source_constants() -> dict:
    """The lists and the prompt function as defined in `personality_convos.py`, read
    with ``ast`` so the module's side effects and its syntax error never run."""
    tree = ast.parse(SOURCE.read_text())
    wanted = {"background_buyers", "background_sellers", "brands", "features", "budgets",
              "classifications"}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_conversation_prompt":
            keep.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {}
    exec(compile(module, str(SOURCE), "exec"), ns)  # noqa: S102 - our own checked-in file
    return ns
