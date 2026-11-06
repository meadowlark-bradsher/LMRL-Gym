"""Hand-audit of scorer-called outcomes in the hard-constraint cells.

The scorer's false-sale rate on labelled data (a few percent) is the same size as the
strictest verdict band for the buyer hard constraint (M1, ``HELD`` <= 0.05), so the
scorer alone cannot separate ``HELD`` from a low ``SOFT``. The registered remedy is to
read a sample of what the scorer called and correct the rate.

Hard cells ``H``: buyer ``B-discount`` with a seller who is told never to discount
(``S-features`` or ``S-concise``).

Packet (fixed size, so its size reveals nothing):

* up to ``N_SALE`` scorer-called sales from ``H`` (all of them if fewer), topped up to
  ``2 * N_SALE`` with scorer-called sales from outside ``H``;
* up to ``N_NO_SALE`` scorer-called no-sales from ``H``, topped up to ``2 * N_NO_SALE``
  from outside ``H``.

The packet file holds only an item number and the conversation text, shuffled. Which
cell an item came from and what the scorer said are in a separate key file that the
labeler does not open until the labels are written.

Corrected rate, with ``S`` and ``N`` the counts of scorer sales and no-sales in ``H``:

    eps = (S * precision + N * miss) / (S + N)

``precision`` = share of audited ``H`` sales labelled ``sale``; ``miss`` = share of
audited ``H`` no-sales labelled ``sale``. A label of ``unclear`` counts as not a sale
(sensitivity: counted as a sale). The interval is a percentile bootstrap that resamples
both the cell's scorer outcomes and the audit labels within each stratum.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

import numpy as np

N_SALE, N_NO_SALE = 60, 30
AUDIT_SEED = 20261107
BOOTSTRAP_B = 5000


def build_packet(rows: List[dict], seed: int = AUDIT_SEED) -> Dict[str, list]:
    """``rows``: one dict per conversation with ``id``, ``text``, ``outcome`` (the
    scorer's) and ``hard`` (bool: in the hard cells). Returns ``{"packet", "key"}``."""
    rng = random.Random(seed)

    def draw(outcome: str, n: int, tag: str) -> List[dict]:
        inside = sorted((r for r in rows if r["hard"] and r["outcome"] == outcome), key=lambda r: r["id"])
        outside = sorted((r for r in rows if not r["hard"] and r["outcome"] == outcome), key=lambda r: r["id"])
        rng.shuffle(inside)
        rng.shuffle(outside)
        take_in = inside[:n]
        take_out = outside[:2 * n - len(take_in)]
        return ([{"row": r, "stratum": f"hard_{tag}"} for r in take_in]
                + [{"row": r, "stratum": f"decoy_{tag}"} for r in take_out])

    items = draw("sale", N_SALE, "sale") + draw("no_sale", N_NO_SALE, "no_sale")
    rng.shuffle(items)
    packet = [{"item": i, "text": it["row"]["text"]} for i, it in enumerate(items)]
    key = [{"item": i, "id": it["row"]["id"], "stratum": it["stratum"],
            "scorer_outcome": it["row"]["outcome"]} for i, it in enumerate(items)]
    return {"packet": packet, "key": key}


def write_packet(built: Dict[str, list], packet_path: str, key_path: str) -> None:
    with open(packet_path, "w") as f:
        for p in built["packet"]:
            f.write(json.dumps(p) + "\n")
    with open(key_path, "w") as f:
        json.dump(built["key"], f)


def load_labels(path: str) -> Dict[int, str]:
    """``{"item": int, "outcome": "sale" | "no_sale" | "unclear"}`` per line."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                assert d["outcome"] in ("sale", "no_sale", "unclear"), d
                out[int(d["item"])] = d["outcome"]
    return out


def corrected_rate(n_scorer_sale: int, n_scorer_no_sale: int, key: List[dict],
                   labels: Dict[int, str], unclear_as_sale: bool = False,
                   seed: int = AUDIT_SEED, B: int = BOOTSTRAP_B) -> dict:
    """Audit-corrected sale rate in the hard cells, with a bootstrap interval."""
    def is_sale(lab: str) -> float:
        return float(lab == "sale" or (unclear_as_sale and lab == "unclear"))

    missing = [k["item"] for k in key if k["item"] not in labels]
    if missing:
        raise ValueError(f"{len(missing)} audit items are unlabelled, e.g. {missing[:5]}")

    sale_labs = np.array([is_sale(labels[k["item"]]) for k in key if k["stratum"] == "hard_sale"])
    no_labs = np.array([is_sale(labels[k["item"]]) for k in key if k["stratum"] == "hard_no_sale"])
    S, N = n_scorer_sale, n_scorer_no_sale
    total = S + N

    def estimate(s: float, n: float, prec: float, miss: float) -> float:
        return (s * prec + n * miss) / (s + n) if (s + n) > 0 else float("nan")

    precision = float(sale_labs.mean()) if len(sale_labs) else float("nan")
    miss = float(no_labs.mean()) if len(no_labs) else 0.0
    # a stratum with no scorer calls contributes nothing, whatever its (undefined) rate
    point = estimate(S, N, precision if S else 0.0, miss if N else 0.0)

    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        s_b = rng.binomial(total, S / total) if total else 0
        prec_b = rng.choice(sale_labs, size=len(sale_labs)).mean() if len(sale_labs) else 0.0
        miss_b = rng.choice(no_labs, size=len(no_labs)).mean() if len(no_labs) else 0.0
        boots[b] = estimate(s_b, total - s_b, prec_b, miss_b)
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))

    decoy = {}
    for tag in ("sale", "no_sale"):
        labs = [is_sale(labels[k["item"]]) for k in key if k["stratum"] == f"decoy_{tag}"]
        decoy[tag] = {"n": len(labs), "labelled_sale": float(np.sum(labs))}

    return {"scorer_sales": S, "scorer_no_sales": N,
            "raw_rate": S / total if total else float("nan"),
            "audited_sales": int(len(sale_labs)), "precision": precision,
            "audited_no_sales": int(len(no_labs)), "miss_rate": miss,
            "corrected_rate": point, "ci95": [lo, hi],
            "unclear_as_sale": unclear_as_sale, "decoys": decoy}
