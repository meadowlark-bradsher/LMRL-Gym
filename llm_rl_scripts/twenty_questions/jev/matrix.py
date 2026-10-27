"""Build and cache the object x attribute likelihood matrix from Jev's Noul.

The likelihood ``p(answer=yes | object, attribute)`` depends only on the object
and the attribute, never on the game history, so it is computed once for every
(object, attribute) pair and reused across every episode and every arm. One Jev
``nouls`` call per object returns the yes-probability for the whole attribute
bank at once (see the live probe: the Noul is graded -- 0.95, 0.04, ... -- where
a yes/no Choice would saturate to 0/1 and throw the likelihood away).

The matrix is saved as JSON with provenance (model, attribute ids, object list)
so cited numbers regenerate.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

from LLM_RL.jev.client import JevClient
from .bed import ATTRIBUTE_BANK, Hypothesis, load_hypotheses


def _noul_instruction(attr_text: str) -> str:
    return (
        'The truthful yes/no answer to the question "{}" about this object is '
        "YES.".format(attr_text)
    )


def _object_state(h: Hypothesis) -> str:
    return (
        "A game of Twenty Questions. The hidden object is: {} "
        "(a kind of {}). Judge each statement about this object.".format(
            h.name, h.category
        )
    )


def _attr_hash(attribute_bank: Sequence) -> str:
    blob = "|".join("{}={}".format(aid, text) for aid, text in attribute_bank)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def build_attribute_matrix(client: JevClient,
                           hypotheses: Optional[Sequence[Hypothesis]] = None,
                           attribute_bank: Sequence = ATTRIBUTE_BANK,
                           batch_size: int = 30,
                           log: bool = True) -> Dict[str, Dict[str, float]]:
    """Return ``matrix[attr_id][object_name] = p(yes)``.

    One or more ``nouls`` calls per object (attributes chunked by ``batch_size``).
    """
    if hypotheses is None:
        hypotheses = load_hypotheses()
    matrix: Dict[str, Dict[str, float]] = {aid: {} for aid, _ in attribute_bank}
    n = len(hypotheses)
    for i, h in enumerate(hypotheses):
        state = _object_state(h)
        for start in range(0, len(attribute_bank), batch_size):
            chunk = attribute_bank[start:start + batch_size]
            questions = {aid: _noul_instruction(text) for aid, text in chunk}
            answers = client.nouls(state, questions)
            for aid, _ in chunk:
                matrix[aid][h.name] = float(answers[aid])
        if log and (i + 1) % 10 == 0:
            print("  built {}/{} objects".format(i + 1, n), flush=True)
    return matrix


def matrix_payload(matrix: Dict[str, Dict[str, float]],
                   hypotheses: Sequence[Hypothesis],
                   model: str,
                   attribute_bank: Sequence = ATTRIBUTE_BANK) -> dict:
    return {
        "schema": "twenty_questions_jev_attribute_matrix_v1",
        "created": _dt.datetime.utcnow().isoformat() + "Z",
        "model": model,
        "n_objects": len(hypotheses),
        "n_attributes": len(attribute_bank),
        "attribute_hash": _attr_hash(attribute_bank),
        "attributes": [{"id": aid, "text": text} for aid, text in attribute_bank],
        "objects": [h.name for h in hypotheses],
        "matrix": matrix,
    }


def save_matrix(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_matrix(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def synthetic_matrix(hypotheses: Sequence[Hypothesis],
                     attribute_bank: Sequence = ATTRIBUTE_BANK
                     ) -> Dict[str, Dict[str, float]]:
    """A deterministic, category-coherent matrix for offline self-tests (no API).

    A handful of attributes are tied to the true category so that BED can tell
    objects apart; the rest are a fixed pseudo-random function of (attr, object).
    Values are near 0/1 so the loop behaves like a real discriminative bank.
    """
    # attr_id -> set of categories for which the attribute is "yes".
    cat_truth = {
        "alive": {"Animals"},
        "living_part": {"Animals", "Fruits", "Vegetables", "Nature"},
        "animal": {"Animals"},
        "plant": {"Fruits", "Vegetables", "Nature"},
        "food": {"Fruits", "Vegetables"},
        "edible": {"Fruits", "Vegetables"},
        "instrument": {"Musical Instruments"},
        "furniture": {"Furniture"},
        "electronic": {"Electronics"},
        "transport": {"Vehicles"},
        "has_motor": {"Vehicles"},
        "wearable": {"Clothes", "Jewelry"},
        "fabric": {"Clothes"},
        "decoration": {"Jewelry"},
        "sports_games": {"Sports", "Toys"},
        "tool": {"Tools", "Kitchen Tools", "Garden Supplies"},
        "garden": {"Garden Supplies"},
        "write_draw": {"Office Supplies", "Art"},
        "office_class": {"Office Supplies"},
        "outdoors_nature": {"Nature"},
        "toy_play": {"Toys"},
    }
    matrix: Dict[str, Dict[str, float]] = {}
    for aid, text in attribute_bank:
        row: Dict[str, float] = {}
        truth = cat_truth.get(aid)
        for h in hypotheses:
            if truth is not None:
                row[h.name] = 0.95 if h.category in truth else 0.05
            else:
                # Deterministic pseudo-random split, object-specific but stable.
                digest = hashlib.sha256("{}:{}".format(aid, h.name).encode()).digest()
                row[h.name] = 0.9 if (digest[0] & 1) else 0.1
        matrix[aid] = row
    return matrix
