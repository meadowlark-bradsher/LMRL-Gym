"""The registration document agrees with the code and the frozen inputs it cites: hashes of the
bank, labels and matrices; the decoder prompt hash; the confirmatory seed and repeats; the primary
hedge; the arm names. A drift in any of these fails here before it can fail in a reader's hands."""

from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from pathlib import Path

import pytest

from . import arms_noul as an
from . import decode as dc
from . import run_noul as rn
from . import verdicts_noul as vn

HERE = Path(__file__).parent
DOC = HERE / "PREREGISTRATION_NOUL.md"
BANK = Path("results/noul_bank_20261108")


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text()


@pytest.mark.skipif(not (BANK / "matrix_sonnet.jsonl").exists(), reason="frozen bank absent")
@pytest.mark.parametrize("name", ["bank.jsonl", "audit_labels.jsonl", "matrix_noul.jsonl", "matrix_sonnet.jsonl"])
def test_frozen_input_hashes_match(doc, name):
    row = next(l for l in doc.splitlines() if f"`results/noul_bank_20261108/{name}`" in l)
    cited = re.search(r"`([0-9a-f]{16})`", row).group(1)
    assert cited == _sha16(BANK / name), name


def test_prompt_hash_seed_hedge_and_arms_match(doc):
    assert f"`{dc.prompt_hash()}`" in doc
    repeats, seed = rn.PHASES["confirmatory"]
    flat = " ".join(doc.replace("**", "").split())
    assert f"K = {repeats} repeats" in flat and f"seed {seed}" in flat
    assert f"Primary hedge: a = {float(vn.PRIMARY_HEDGE):g}" in doc.replace("**", "")
    assert vn.PRIMARY_HEDGE == Fraction(1, 5)
    for arm in rn.DEFAULT_ARMS:
        stem = arm.rsplit("-", 1)[0] if arm.startswith("plan_ask") else arm
        assert stem in doc, arm
    assert f"`{an.AGENT_MENU_DECODER}` at a = {float(an.REGISTERED_HEDGE):g}" in doc
    assert "ETA = 0" in doc and an.ETA == 0.0
