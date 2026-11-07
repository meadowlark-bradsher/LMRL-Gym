"""Scripted sellers for CD-PERSONA-1 Arm A (`PREREGISTRATION_CURRENT.md`, §Design).

Each seller type is a fixed sequence of at most ``MAX_TURNS`` lines. A line is a pure
function of the record (brand, classification, list price, offered features) and the
turn number; nothing the buyer says reaches it. So the seller cannot break its persona,
and `run.verify_seller_lines` can recompute every line after the fact.

The three types mirror the 2023 backgrounds:

- ``S-discount``: never talks about features, offers 5%, 10% and 15% off in turn;
- ``S-features``: never discounts, talks about the features at length;
- ``S-concise``: never discounts, never talks about features, one short line per turn.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

SELLERS = ("S-discount", "S-features", "S-concise")
MAX_TURNS = 5

# Fraction off the list price that S-discount has offered by the end of each turn.
DISCOUNT_SCHEDULE = {1: 0.0, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.15}

# One sentence per 2023 feature, for S-features. None of these may use a word that reads
# as a price concession ("discount", "off", "%"); `test_harness.py` checks every line.
FEATURE_BLURBS = {
    "Leather seats": "The leather seats are full-grain and ventilated, so they stay comfortable on long drives.",
    "Sunroof/moonroof": "The sunroof is panoramic and opens the whole cabin up to the sky.",
    "Heated seats": "The heated seats warm up in under a minute, front and rear.",
    "Backup camera": "The backup camera has a wide-angle lens and dynamic guidelines, which makes parking effortless.",
    "Navigation system": "The navigation system has live traffic and updates its maps over the air.",
    "Bluetooth": "Bluetooth pairs two phones at once, for calls and for music.",
    "Remote start": "Remote start lets you warm the car up from inside the house.",
    "Blind spot monitoring": "Blind spot monitoring watches both sides and warns you before a lane change.",
    "Third-row seating": "The third-row seating folds flat, so you can carry seven people or a full load of cargo.",
    "Apple CarPlay/Android Auto": "Apple CarPlay and Android Auto are wireless, so your phone is on the screen the moment you sit down.",
}


def fmt_price(dollars: float) -> str:
    return f"${int(round(dollars)):,}"


def current_price(seller: str, record, turn: int) -> int:
    """The price on the table after the seller's line at ``turn``."""
    if seller not in SELLERS:
        raise ValueError(seller)
    if not 1 <= turn <= MAX_TURNS:
        raise ValueError(turn)
    if seller == "S-discount":
        return int(round(record.list_price * (1.0 - DISCOUNT_SCHEDULE[turn])))
    return int(record.list_price)


def _car(record) -> str:
    return f"{record.brand} {record.classification}"


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _discount_line(record, turn: int) -> str:
    p = fmt_price
    if turn == 1:
        return (f"Welcome in! You're looking at a {_car(record)}. It's listed at "
                f"{p(record.list_price)}. I'd love to get you into it today, and I have some "
                f"room on the price.")
    if turn == 2:
        return f"Tell you what: I can take 5% off, so {p(current_price('S-discount', record, 2))}."
    if turn == 3:
        return f"I can go to 10% off. That's {p(current_price('S-discount', record, 3))}."
    if turn == 4:
        return (f"Final offer: 15% off, {p(current_price('S-discount', record, 4))}. "
                f"That's the best I can do.")
    return f"Shall we write it up at {p(current_price('S-discount', record, 5))}?"


def _features_line(record, turn: int) -> str:
    p = fmt_price(record.list_price)
    offered = list(record.offered_features)
    first, rest = offered[:2], offered[2:]
    if turn == 1:
        equipped = (f"It comes with {_join(first)}." if first
                    else "It comes very well equipped.")
        return f"Welcome! This {_car(record)} is listed at {p}. {equipped}"
    if turn == 2:
        if rest:
            body = " ".join(FEATURE_BLURBS[f] for f in rest)
            return f"Let me walk you through the rest of what it has. {body} The price is {p}."
        return ("Everything on it is standard equipment, and the build quality is "
                f"excellent throughout. The price is {p}.")
    if turn == 3:
        if offered:
            return (f"You'd really appreciate the {offered[0].lower()} on a test drive. "
                    f"{FEATURE_BLURBS[offered[0]]} I can set one up today. The price is {p}.")
        return ("You'd really appreciate how it drives; I can set up a test drive today. "
                f"The price is {p}.")
    if turn == 4:
        recap = (f"So to recap, it has {_join(offered)}." if offered
                 else "So to recap, it is a very well equipped car.")
        return f"{recap} The price is firm at {p}."
    return f"Shall we write it up at {p}?"


def _concise_line(record, turn: int) -> str:
    p = fmt_price(record.list_price)
    return {
        1: f"{_car(record)}. {p}.",
        2: f"The price is {p}.",
        3: "No discounts.",
        4: "The price is firm.",
        5: "Yes or no?",
    }[turn]


def seller_line(seller: str, record, turn: int) -> str:
    """The seller's scripted line for ``turn`` (1-based)."""
    if not 1 <= turn <= MAX_TURNS:
        raise ValueError(turn)
    if seller == "S-discount":
        return _discount_line(record, turn)
    if seller == "S-features":
        return _features_line(record, turn)
    if seller == "S-concise":
        return _concise_line(record, turn)
    raise ValueError(seller)


def close_line(seller: str, record, turn: int, decision: str) -> Tuple[str, Optional[int]]:
    """The seller's final line after the buyer's BUY or LEAVE at ``turn``.

    Returns the line and, for a sale, the price it closes at: the seller's price on the
    table at that turn (not whatever price the buyer named).
    """
    if decision == "BUY":
        price = current_price(seller, record, turn)
        return f"Great, let's finalize it at {fmt_price(price)}.", price
    if decision == "LEAVE":
        return "Understood. Thanks for coming in.", None
    raise ValueError(decision)
