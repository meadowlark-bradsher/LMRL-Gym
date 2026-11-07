"""Validity gate for the persona-blind scorer, against hand-labelled conversations.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.validate [--out DIR]

Gate (``PREREGISTRATION_BASELINE2023.md``):

* sale outcome: accuracy >= 0.90 on items where gold and scorer are both decided, and
  the scorer returns ``unclear`` on <= 15% of gold-decided items;
* discount detector: agreement >= 0.85 (if this alone fails, the measures that use it
  are exploratory).

The gate set is RESERVE (``gold/reserve_labels.jsonl``). The GOLD split
(``gold/gold_labels.jsonl``) was the gate set for the first scorer version; its two
errors were used to fix the deferral rules, so it is development data now and is
reported only for reference.

Also reported, not gated: the direction of the outcome errors (false sales vs missed
sales), exact agreement on the agreed price among labelled-priced sales, and agreement
between the dialogue outcome and an unambiguous (0 or 1) self-report on DEV + GOLD +
RESERVE. Nothing here reads the CONFIRMATORY split or any persona field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import data as d
from . import scorer as s

_GOLD_DIR = os.path.join(os.path.dirname(__file__), "gold")
GATE_PATH = os.path.join(_GOLD_DIR, "reserve_labels.jsonl")      # the registered gate set
DEV_LABELS_PATH = os.path.join(_GOLD_DIR, "gold_labels.jsonl")   # used to revise the rules
MIN_ACCURACY, MAX_UNCLEAR, MIN_DISCOUNT = 0.90, 0.15, 0.85


def load_gold(path: str = GATE_PATH) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(gold: list) -> dict:
    scored = [(g, s.score(g["original_conversation"])) for g in gold]

    confusion = {}
    for g, x in scored:
        key = f"{g['outcome']}->{x['outcome']}"
        confusion[key] = confusion.get(key, 0) + 1

    gold_decided = [(g, x) for g, x in scored if g["outcome"] != "unclear"]
    both = [(g, x) for g, x in gold_decided if x["outcome"] != "unclear"]
    accuracy = sum(g["outcome"] == x["outcome"] for g, x in both) / len(both)
    unclear_rate = 1.0 - len(both) / len(gold_decided)
    discount = sum(g["discount_offered"] == x["discount_offered"] for g, x in scored) / len(scored)

    priced = [(g, x) for g, x in scored if g["outcome"] == "sale" and g["price"] is not None]
    price_exact = sum(x["price"] is not None and abs(x["price"] - g["price"]) < 1 for g, x in priced)

    errors = [{"index": g["index"], "id": g["id"][:10], "gold": g["outcome"], "scorer": x["outcome"]}
              for g, x in scored if g["outcome"] != x["outcome"]]
    discount_errors = [{"index": g["index"], "id": g["id"][:10], "gold": g["discount_offered"],
                        "scorer": x["discount_offered"]}
                       for g, x in scored if g["discount_offered"] != x["discount_offered"]]
    price_errors = [{"index": g["index"], "id": g["id"][:10], "gold": g["price"], "scorer": x["price"]}
                    for g, x in priced if x["price"] is None or abs(x["price"] - g["price"]) >= 1]

    n_no_sale = sum(g["outcome"] == "no_sale" for g, _ in scored)
    n_sale = sum(g["outcome"] == "sale" for g, _ in scored)
    false_sales = confusion.get("no_sale->sale", 0)
    missed_sales = confusion.get("sale->no_sale", 0)

    sale_gate = accuracy >= MIN_ACCURACY and unclear_rate <= MAX_UNCLEAR
    return {"n": len(scored), "confusion": confusion,
            "false_sales": false_sales, "n_labelled_no_sale": n_no_sale,
            "missed_sales": missed_sales, "n_labelled_sale": n_sale,
            "n_gold_decided": len(gold_decided), "n_both_decided": len(both),
            "accuracy": accuracy, "scorer_unclear_rate": unclear_rate,
            "discount_agreement": discount,
            "price_exact": price_exact, "n_priced": len(priced),
            "sale_gate": sale_gate, "discount_gate": discount >= MIN_DISCOUNT,
            "outcome_errors": errors, "discount_errors": discount_errors,
            "price_errors": price_errors}


def silver(data_dir: str = d.DEFAULT_DIR) -> dict:
    """Dialogue outcome vs an unambiguous self-report, on DEV + GOLD only."""
    sp = d.splits(data_dir)
    n = agree = 0
    for rec in sp["dev"] + sp["gold"] + sp["reserve"]:
        x = s.score(rec["original_conversation"])
        if x["self_report"] in (0.0, 1.0) and x["outcome"] != "unclear":
            n += 1
            agree += (x["outcome"] == "sale") == (x["self_report"] == 1.0)
    return {"n": n, "agree": agree}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="optional dir to freeze validation.json into")
    p.add_argument("--data-dir", default=d.DEFAULT_DIR)
    args = p.parse_args(argv)

    r = validate(load_gold(GATE_PATH))
    r["development_set"] = validate(load_gold(DEV_LABELS_PATH))
    r["silver_dev_gold_reserve"] = silver(args.data_dir)

    def show(name, x, gated):
        print(f"{name}: n={x['n']}  confusion (label->scorer): {x['confusion']}")
        verdict = f"  -> {'PASS' if x['sale_gate'] else 'FAIL'}" if gated else ""
        print(f"  sale outcome: accuracy {x['accuracy']:.3f} on {x['n_both_decided']} both-decided "
              f"(gate >= {MIN_ACCURACY}); scorer unclear on {x['scorer_unclear_rate']:.3f} of "
              f"{x['n_gold_decided']} decided (gate <= {MAX_UNCLEAR}){verdict}")
        print(f"  error direction: {x['false_sales']} false sales of {x['n_labelled_no_sale']} labelled "
              f"no_sale; {x['missed_sales']} missed sales of {x['n_labelled_sale']} labelled sale")
        verdict = f"  -> {'PASS' if x['discount_gate'] else 'FAIL'}" if gated else ""
        print(f"  discount detector: agreement {x['discount_agreement']:.3f} "
              f"(gate >= {MIN_DISCOUNT}){verdict}")
        print(f"  agreed price: exact on {x['price_exact']} of {x['n_priced']} labelled-priced sales")
        for key in ("outcome_errors", "discount_errors", "price_errors"):
            if x[key]:
                print(f"  {key}: {x[key]}")

    show("RESERVE (gate set)", r, gated=True)
    show("GOLD (development data since the rule revision)", r["development_set"], gated=False)
    sv = r["silver_dev_gold_reserve"]
    print(f"self-report of exactly 0 or 1 vs dialogue outcome (DEV+GOLD+RESERVE): "
          f"{sv['agree']} of {sv['n']} agree")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "validation.json"), "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
